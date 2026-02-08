from __future__ import annotations

import re

OFF_TOPIC_PATTERNS = [
    r"\bchiste\b",
    r"\bcuéntame un chiste\b",
    r"\bcuentame un chiste\b",
    r"\bpoema\b",
]


def is_off_topic_query(query: str) -> bool:
    text = (query or "").strip().lower()
    return any(re.search(pattern, text) for pattern in OFF_TOPIC_PATTERNS)


def off_topic_response() -> str:
    return (
        "Soy un asistente de e-commerce. Puedo ayudarte a buscar productos, "
        "ver stock, gestionar el carrito y finalizar compras."
    )
