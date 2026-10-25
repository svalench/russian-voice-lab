from voice_lab.engine import MLXEngine, phrase_boundary


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.template_options = kwargs
        return "|".join(m["content"] for m in messages)

    def encode(self, text):
        return list(text)


def test_context_reclaims_space_once_and_keeps_prefix_for_following_turns():
    engine = MLXEngine()
    engine.tokenizer = Tokenizer()
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i) + "x" * 180} for i in range(24)]
    prompt = engine.messages(history, "", "Вопрос")
    assert len(prompt) <= 2048
    floor = engine.history_drop
    assert floor > 0
    history.extend([{"role": "user", "content": "Вопрос"}, {"role": "assistant", "content": "Ответ"}])
    prompt = engine.messages(history, "", "Следующий")
    assert len(prompt) <= 3840
    assert engine.history_drop == floor
    engine.messages([], "", "Новый разговор")
    assert engine.history_drop == 0
    assert engine.tokenizer.template_options["enable_thinking"] is False


def test_phrase_split_keeps_decimal_numbers_and_shortens_first_clause():
    assert phrase_boundary("Число 3.") == 0
    assert phrase_boundary("Число 3.14. ") == len("Число 3.14. ")
    text = "Минск — столица Беларуси, современный город с историей"
    assert phrase_boundary(text, first=True) == len("Минск — столица Беларуси, ")
