import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(self._strip_sensitive(record.extra))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

    def _strip_sensitive(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: self._strip_sensitive(v)
                for k, v in value.items()
                if k not in ("user_id", "trace_id", "timestamp", "ts")
            }
        if isinstance(value, list):
            return [self._strip_sensitive(v) for v in value]
        return value


def setup_structured_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("AGENTE_BACKEND")
    logger.setLevel(level)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
