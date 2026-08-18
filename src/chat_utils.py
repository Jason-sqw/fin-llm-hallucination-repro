"""Chat template helpers for models without a built-in template."""

DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}<|system|>
{{ message['content'] }}
{% elif message['role'] == 'user' %}<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}{{ eos_token }}
{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"""


def ensure_chat_template(tokenizer):
    """Attach a Zephyr-style template when the tokenizer ships without one."""
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
