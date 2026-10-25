const $ = (id) => document.getElementById(id);
const labels = {
  loading: ['Загружаем модели', 'Первый запуск может занять несколько минут.'],
  ready: ['Готов к разговору', 'Включите микрофон и начните с любой фразы.'],
  listening: ['Слушаю вас', 'Говорите свободно. После паузы я отвечу.'],
  transcribing: ['Распознаю речь', 'Перевожу вашу реплику в текст.'],
  thinking: ['Формирую ответ', 'Скоро начну говорить.'],
  speaking: ['Говорю', 'Можете перебить меня в любой момент.'],
  error: ['Нужна ваша помощь', 'Подробности указаны в сообщении выше.'],
};
let socket, audioContext, mic, active = false, starting = false, ready = false, pending = false;
let micStream = null, startEpoch = 0;
let turnId = 0, audioHeader = null, nextAudioAt = 0, generationDone = false;
let lastVoiceAt = 0, speechEndAt = 0, voiceFrameAt = 0;
let chunks = [], records = new Map(), currentState = 'loading', lastStatus;
let report = { started_at: new Date().toISOString(), turns: [] };
function exportRecords() {
  return [...records.values()].map(({id,user,assistant,metrics,interrupted,speech_started_at}) => ({id,user,assistant,metrics,interrupted,speech_started_at}));
}

function state(name) {
  currentState = name;
  const [title, hint] = labels[name] || labels.ready;
  $('state').textContent = title;
  $('stateHint').textContent = hint;
  $('orb').className = `orb ${name}`;
}
function error(message) { $('error').textContent = message; $('error').hidden = false; }
function clearError() { $('error').hidden = true; }
function send(event) { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(event)); }
function updateControls() {
  $('start').disabled = !ready || active || starting || pending;
  $('stop').disabled = !active && !starting && !pending && !chunks.some(c => !c.ended);
  $('preview').disabled = !ready || active || starting || pending || chunks.some(c => !c.ended);
  $('voice').disabled = active || starting || pending || chunks.some(c => !c.ended);
  $('instruction').disabled = active || starting;
}
function row(role, text, id) {
  $('empty')?.remove();
  const box = document.createElement('div'); box.className = `message ${role}`;
  const title = document.createElement('div'); title.className = 'role'; title.textContent = role === 'user' ? 'ВЫ' : 'СОБЕСЕДНИК';
  const body = document.createElement('div'); body.className = 'body'; body.textContent = text;
  box.append(title, body); $('messages').append(box);
  const record = records.get(id) || {id, user: '', assistant: '', phrases: new Map()};
  record[role === 'user' ? 'userEl' : 'assistantEl'] = box;
  record[role] = text; records.set(id, record);
  $('messages').scrollTop = $('messages').scrollHeight;
  return box;
}
function getRecord(id) {
  if (!records.has(id)) records.set(id, {id, user: '', assistant: '', phrases: new Map()});
  return records.get(id);
}
function played(c) {
  if (!audioContext || c.start > audioContext.currentTime) return 0;
  return Math.min(c.samples.length, Math.max(0, Math.floor((audioContext.currentTime - c.start) * c.rate)));
}
function receipt() {
  return {turn_id: chunks[0]?.turn_id ?? turnId, chunks: chunks.map(c => [c.id, c.stoppedAt ?? played(c)])};
}
function stopAudio() {
  const ack = receipt();
  for (const c of chunks) {
    c.stoppedAt = played(c);
    c.source.onended = null;
    try { c.source.stop(); } catch {}
  }
  const rec = records.get(chunks[0]?.turn_id);
  if (rec && chunks.some(c => played(c) < c.samples.length)) {
    rec.interrupted = true;
    rec.assistantEl?.classList.add('interrupted');
    const full = [];
    for (const [id, phrase] of rec.phrases) {
      const parts = chunks.filter(c => c.phrase_id === id);
      if (phrase.ended && parts.length && parts.every(c => c.stoppedAt === c.samples.length)) full.push(phrase.text);
      else break;
    }
    if (rec.assistantEl) {
      rec.assistantEl.querySelector('.body').textContent = full.join(' ') || 'Ответ прерван';
      rec.assistant = full.join(' ');
      const tail = document.createElement('div'); tail.className = 'tail';
      tail.textContent = 'Прервано · озвученный фрагмент уточняется перед следующим ответом';
      rec.assistantEl.append(tail);
    }
  }
  chunks = []; nextAudioAt = 0; generationDone = false; pending = false;
  updateControls();
  return ack;
}
function interrupt(type) {
  const playback = stopAudio();
  turnId += 1;
  send({type, playback});
}
async function ensureAudio() {
  audioContext ||= new AudioContext({latencyHint: 'interactive'});
  await audioContext.resume();
}
async function connect() {
  if (socket?.readyState === WebSocket.OPEN) return;
  const opened = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/session`);
  socket = opened;
  await new Promise((resolve, reject) => {
    opened.binaryType = 'arraybuffer';
    const timer = setTimeout(() => { opened.close(); reject(new Error('Не удалось подключиться к локальному серверу.')); }, 10000);
    opened.onmessage = (event) => {
      if (socket !== opened) { clearTimeout(timer); opened.close(); reject(new Error('Соединение заменено.')); return; }
      if (event.data instanceof ArrayBuffer) { receiveAudio(event.data); return; }
      const data = JSON.parse(event.data);
      if (data.type === 'connected') {
        clearTimeout(timer);
        if (records.size) {
          report.turns.push(...exportRecords()); records.clear();
          const divider = document.createElement('p'); divider.className = 'small center';
          divider.textContent = 'Новое подключение · начат новый разговор'; $('messages').append(divider);
        }
        turnId = data.turn_id; resolve();
      }
      handle(data);
    };
    opened.onerror = () => { clearTimeout(timer); reject(new Error('Соединение с сервером недоступно.')); };
    opened.onclose = async () => {
      clearTimeout(timer); reject(new Error('Соединение закрыто.'));
      if (socket !== opened) return;
      startEpoch += 1;
      const wasActive = active || starting;
      active = false; starting = false;
      stopAudio(); await stopMic();
      if (socket !== opened) return;
      $('connection').textContent = 'Отключено';
      if (wasActive) error('Связь с локальным сервером потеряна. Проверьте терминал и начните разговор заново.');
      updateControls(); state(ready ? 'ready' : 'error');
    };
  });
  $('connection').textContent = 'Подключено';
}
function handle(data) {
  if (data.type === 'error') { pending = false; error(data.message); state('error'); updateControls(); return; }
  if (data.type === 'cancelled') { turnId = Math.max(turnId, data.turn_id); return; }
  if (data.type === 'audio') { audioHeader = data; return; }
  if (data.turn_id != null && data.turn_id !== turnId) return;
  switch (data.type) {
    case 'state': pending = ['transcribing', 'thinking'].includes(data.state); if (!chunks.some(c => !c.ended)) state(data.state); updateControls(); break;
    case 'transcript': row('user', data.text, turnId); break;
    case 'text_delta': {
      const rec = getRecord(turnId); rec.assistant += data.text;
      if (!rec.assistantEl) row('assistant', rec.assistant, turnId);
      else rec.assistantEl.querySelector('.body').textContent = rec.assistant;
      $('messages').scrollTop = $('messages').scrollHeight; break;
    }
    case 'phrase': getRecord(turnId).phrases.set(data.phrase_id, {text: data.text, ended: false}); break;
    case 'phrase_end': {
      const phrase = getRecord(turnId).phrases.get(data.phrase_id);
      if (phrase) phrase.ended = true;
      break;
    }
    case 'history_update': {
      const rec = records.get(data.round_id);
      if (rec) {
        rec.assistant = data.text;
        if (rec.assistantEl) {
          rec.assistantEl.querySelector('.body').textContent = data.text || 'Ответ прерван до начала речи';
          const tail = rec.assistantEl.querySelector('.tail');
          if (tail) tail.textContent = 'Прервано · сохранена только озвученная часть';
        }
      } break;
    }
    case 'metrics': {
      const rec = getRecord(turnId); rec.metrics = {...rec.metrics, ...data};
      $('metricAsr').textContent = data.asr_ms == null ? '—' : `${data.asr_ms} мс`;
      $('metricLlm').textContent = `${data.llm_first_token_ms} мс`;
      $('metricRtf').textContent = data.tts_rtf == null ? '—' : `${data.tts_rtf.toFixed(2)}×`;
      break;
    }
    case 'done': generationDone = true; maybeFinished(); break;
    case 'empty': pending = false; state(active ? 'listening' : 'ready'); updateControls(); break;
  }
}
function receiveAudio(bytes) {
  const meta = audioHeader; audioHeader = null;
  if (!meta || meta.turn_id !== turnId || !audioContext) return;
  const samples = new Float32Array(bytes);
  if (!samples.length) return;
  const buffer = audioContext.createBuffer(1, samples.length, meta.sample_rate);
  buffer.copyToChannel(samples, 0);
  const source = audioContext.createBufferSource(); source.buffer = buffer; source.connect(audioContext.destination);
  const start = Math.max(audioContext.currentTime + 0.025, nextAudioAt);
  const c = {id: meta.chunk_id, turn_id: turnId, phrase_id: meta.phrase_id, samples, rate: meta.sample_rate, start, source, ended: false};
  chunks.push(c); nextAudioAt = start + buffer.duration;
  if (chunks.length === 1) {
    const rec = getRecord(turnId);
    rec.metrics = {...rec.metrics, end_to_audio_ms: speechEndAt ? Math.round(performance.now() - speechEndAt + (start - audioContext.currentTime) * 1000) : null};
    $('metricFirst').textContent = rec.metrics.end_to_audio_ms == null ? '—' : `${(rec.metrics.end_to_audio_ms / 1000).toFixed(2)} с`;
  }
  source.onended = () => { c.ended = true; send({type: 'playback', ...receipt()}); maybeFinished(); };
  source.start(start); state('speaking'); updateControls();
}
function maybeFinished() {
  if (generationDone && chunks.every(c => c.ended)) {
    send({type: 'playback', ...receipt()});
    // Retain receipts until the next interruption/reset, but release source nodes.
    pending = false; state(active ? 'listening' : 'ready'); updateControls();
  }
}
async function disposeMic(instance, stream) {
  if (instance) { try { await instance.destroy(); } catch (err) { console.warn('VAD cleanup:', err.message); } }
  stream?.getTracks().forEach(track => track.stop());
}
async function stopMic() {
  const old = mic, stream = micStream; mic = null; micStream = null;
  await disposeMic(old, stream);
  if (!active) $('level').style.width = '0%';
}
$('start').onclick = async () => {
  if (starting || active) return;
  const epoch = ++startEpoch;
  let ownedStream, createdMic;
  starting = true; clearError(); updateControls();
  try {
    await ensureAudio(); await connect();
    if (epoch !== startEpoch) return;
    send({type: 'configure', voice: $('voice').value, instruction: $('instruction').value});
    if (!globalThis.vad) throw new Error('Не найдены локальные файлы VAD. Выполните ./scripts/setup.sh');
    const stream = ownedStream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1}});
    if (epoch !== startEpoch) { stream.getTracks().forEach(track => track.stop()); return; }
    micStream = stream;
    createdMic = await vad.MicVAD.new({
      model: 'v5', startOnLoad: false, processorType: 'AudioWorklet', audioContext,
      baseAssetPath: '/static/assets/vad/', onnxWASMBasePath: '/static/assets/ort/',
      ortConfig: (ort) => { ort.env.wasm.numThreads = 1; },
      positiveSpeechThreshold: 0.6, negativeSpeechThreshold: 0.35,
      minSpeechMs: 160, preSpeechPadMs: 320, redemptionMs: Number($('pause').value),
      getStream: async () => stream,
      resumeStream: async () => {
        ownedStream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1}});
        if (epoch !== startEpoch) { ownedStream.getTracks().forEach(track => track.stop()); throw new DOMException('Сессия остановлена', 'AbortError'); }
        micStream = ownedStream;
        return micStream;
      },
      onFrameProcessed: (probability, frame) => {
        if (!active || epoch !== startEpoch) return;
        let sum = 0; for (const x of frame) sum += x * x;
        const level = Math.min(100, Math.sqrt(sum / frame.length) * 600);
        $('level').style.width = `${level}%`; $('level').parentElement.setAttribute('aria-valuenow', Math.round(level));
        voiceFrameAt = performance.now();
        if (probability.isSpeech >= 0.35) lastVoiceAt = voiceFrameAt;
        if (socket?.readyState === WebSocket.OPEN) {
          if (socket.bufferedAmount > 1_000_000) { error('Аудио не успевает передаваться. Начните сессию заново.'); void stop(); return; }
          socket.send(frame.buffer.slice(frame.byteOffset, frame.byteOffset + frame.byteLength));
        }
      },
      onSpeechStart: () => {
        if (!active || epoch !== startEpoch) return;
        interrupt('speech_start'); state('listening');
        const rec = getRecord(turnId); rec.speech_started_at = new Date().toISOString();
      },
      onSpeechEnd: () => {
        if (!active || epoch !== startEpoch) return;
        speechEndAt = lastVoiceAt || performance.now(); send({type: 'speech_end'}); state('transcribing');
      },
      onVADMisfire: () => { if (active && epoch === startEpoch) { interrupt('speech_misfire'); state('listening'); } },
    });
    if (epoch !== startEpoch) { await disposeMic(createdMic, ownedStream); return; }
    mic = createdMic; active = true; await createdMic.start();
    if (epoch !== startEpoch) { await disposeMic(createdMic, ownedStream); return; }
    starting = false; state('listening'); updateControls();
  } catch (err) {
    if (epoch !== startEpoch) { await disposeMic(createdMic, ownedStream); return; }
    if (!mic && createdMic) await disposeMic(createdMic, ownedStream);
    active = false; starting = false; await stopMic(); updateControls(); state('error');
    error(err.name === 'NotAllowedError' ? 'Доступ к микрофону запрещён. Разрешите его в настройках браузера для localhost и нажмите «Начать разговор».' : err.name === 'NotFoundError' ? 'Микрофон не найден. Подключите микрофон и повторите.' : err.message);
  }
};
async function stop() {
  const epoch = ++startEpoch;
  active = false; starting = false;
  interrupt('stop'); await stopMic();
  if (epoch === startEpoch) { state(ready ? 'ready' : 'error'); updateControls(); }
}
$('stop').onclick = stop;
$('preview').onclick = async () => {
  const epoch = ++startEpoch;
  clearError(); starting = true; updateControls();
  try {
    await ensureAudio(); await connect();
    if (epoch !== startEpoch) return;
    send({type: 'configure', voice: $('voice').value, instruction: $('instruction').value});
    speechEndAt = 0; interrupt('preview'); pending = true; state('thinking');
  } catch (err) { if (epoch === startEpoch) error(err.message); } finally { if (epoch === startEpoch) { starting = false; updateControls(); } }
};
$('pause').oninput = () => { $('pauseValue').value = `${$('pause').value} мс`; mic?.setOptions({redemptionMs: Number($('pause').value)}); };
$('reset').onclick = async () => {
  interrupt('reset'); records.clear(); report = {started_at: new Date().toISOString(), turns: []};
  $('messages').replaceChildren();
  for (const id of ['metricFirst', 'metricAsr', 'metricLlm', 'metricRtf']) $(id).textContent = '—';
  if (mic) {
    const current = mic;
    try { await current.pause(); if (mic === current && active) await current.start(); }
    catch (err) { if (mic !== current || !active) return; if (err.name !== 'AbortError') error(err.message); await stop(); }
  }
  state(active ? 'listening' : ready ? 'ready' : 'loading');
};
$('export').onclick = () => {
  send({type: 'playback', ...receipt()});
  const turns = [...report.turns, ...exportRecords()];
  const blob = new Blob([JSON.stringify({...report, model_ids: lastStatus?.model_ids, turns}, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = `voice-lab-${new Date().toISOString().replace(/[:.]/g, '-')}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function status() {
  try {
    const response = await fetch('/api/status'); if (!response.ok) throw new Error('Сервер недоступен');
    const data = await response.json(); lastStatus = data;
    $('modelBadge').textContent = `${data.llm_profile === 'small' ? 'Qwen3 1.7B' : 'Qwen3 4B'} · MLX`;
    $('voiceBackend').textContent = data.tts_backend === 'piper' ? '· Piper' : '· Qwen3';
    ready = data.state === 'ready' && data.assets_ready;
    $('models').textContent = Object.entries(data.models).map(([name, value]) => `${name.toUpperCase()} ${value === 'ready' ? '✓' : value === 'loading' ? 'загружается…' : 'ожидание'}`).join('   ·   ');
    $('resources').textContent = `${data.mlx_active_mb == null ? 'Процесс ' + (data.memory_mb / 1024).toFixed(1) : 'MLX ' + (data.mlx_active_mb / 1024).toFixed(1)} ГБ · доступно ОЗУ ${(data.available_memory_mb / 1024).toFixed(1)} ГБ · диск ${data.disk_free_gb} ГБ`;
    if (data.voices && [...$('voice').options].map(o => o.value).join('|') !== data.voices.join('|')) {
      const selected = $('voice').value;
      $('voice').replaceChildren(...data.voices.map(v => new Option(v, v)));
      if (data.voices.includes(selected)) $('voice').value = selected;
    }
    if (data.error) { error(data.error); state('error'); }
    else if (!data.assets_ready) { error('Не найдены локальные файлы VAD. Выполните ./scripts/setup.sh'); state('error'); }
    else if (ready && !active && !starting && ['loading', 'error'].includes(currentState)) state('ready');
    updateControls();
  } catch { ready = false; updateControls(); if (!active) { state('error'); error('Локальный сервер недоступен. Запустите ./scripts/run.sh'); } }
}
setInterval(() => { if (chunks.length) send({type: 'playback', ...receipt()}); }, 150);
setInterval(status, 2500); void status();
window.addEventListener('pagehide', () => { stopAudio(); socket?.close(); void stopMic(); });
