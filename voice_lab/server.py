import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import os
import shutil

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import psutil

from .config import MODELS, ROOT, VOICES
from .engine import MLXEngine
from .session import Session


def create_app(engine=None):
    engine = engine or MLXEngine()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.get_running_loop().run_in_executor(executor, engine.load)
        yield
        await task
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(lifespan=lifespan)
    app.state.owner = None
    app.state.engine = engine

    @app.middleware("http")
    async def local_only(request, call_next):
        from fastapi.responses import JSONResponse
        host = request.headers.get("host", "").split(":")[0]
        if host not in ("localhost", "127.0.0.1", "testserver"):
            return JSONResponse({"detail": "Local connections only"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/status")
    async def status():
        return {
            **engine.status, "model_ids": getattr(engine, "model_ids", MODELS), "voices": getattr(engine, "voices", VOICES),
            "tts_backend": getattr(engine, "tts_backend", "qwen"),
            "llm_profile": getattr(engine, "llm_profile", "4b"),
            "busy": app.state.owner is not None,
            "memory_mb": round(psutil.Process(os.getpid()).memory_info().rss / 1024**2),
            "available_memory_mb": round(psutil.virtual_memory().available / 1024**2),
            "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1024**3, 1),
            "assets_ready": all((ROOT / "web/assets" / file).exists() for file in (
                "vad/bundle.min.js", "vad/vad.worklet.bundle.min.js", "vad/silero_vad_v5.onnx",
                "ort/ort.wasm.min.js", "ort/ort-wasm-simd-threaded.mjs", "ort/ort-wasm-simd-threaded.wasm",
            )),
        }

    @app.websocket("/api/session")
    async def session(ws: WebSocket):
        origin = ws.headers.get("origin")
        from urllib.parse import urlparse
        if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1", "testserver"):
            await ws.close(code=1008)
            return
        await ws.accept()
        if app.state.owner is not None:
            await ws.send_json({"type": "error", "message": "Разговор уже открыт в другой вкладке."})
            await ws.close(code=1013)
            return
        current = Session(ws, engine, executor)
        app.state.owner = current
        try:
            await current.run()
        finally:
            if app.state.owner is current:
                app.state.owner = None

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "web/index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app


app = create_app()
