"""Structured logging configuration with correlation IDs and JSON formatting."""

import contextvars
import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def get_request_id() -> str:
    """Get current request ID from context."""
    return _request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set request ID for current context."""
    _request_id_var.set(request_id)


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = get_request_id()
        if request_id:
            log_entry["request_id"] = request_id

        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_entry["stack_info"] = record.stack_info

        # Add extra fields, sanitizing values
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ):
                log_entry[key] = sanitize_data(value)

        return json.dumps(log_entry, default=str)


def sanitize_data(data: Any) -> Any:
    """
    Recursively sanitize data for logging by redacting secrets.
    Does not truncate. Use sanitize_for_logging() to also truncate strings.
    Redacts:
    - OpenAI-style API keys: sk-...
    - Bearer tokens
    - Basic auth credentials
    - Database connection strings with passwords
    - Long base64-like strings
    """
    if isinstance(data, str):
        return _sanitize_string(data)
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_data(item) for item in data)
    else:
        return data


def _sanitize_string(s: str) -> str:
    """Redact patterns in a string."""
    # OpenAI API key: sk- followed by at least 20 alphanumeric/underscore/hyphen
    s = re.sub(r"sk-[A-Za-z0-9_\-]{20,}", "sk-REDACTED", s)

    # Bearer token: Bearer <token>
    s = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer REDACTED", s, flags=re.IGNORECASE)

    # Basic auth: Basic <base64>
    s = re.sub(r"Basic\s+[A-Za-z0-9=]+", "Basic REDACTED", s, flags=re.IGNORECASE)

    # Database URLs with passwords: postgresql://user:pass@host/db
    s = re.sub(r":\/\/([^:]+):[^@]+@", "://\\1:REDACTED@", s)

    # Generic base64 strings longer than 50 chars (potential secrets)
    if len(s) >= 50 and re.fullmatch(r"[A-Za-z0-9+/=]+", s):
        return s[:20] + "...[REDACTED base64]"

    return s


def truncate_strings(data: Any, max_length: int) -> Any:
    """Recursively truncate strings to max_length."""
    if isinstance(data, str):
        if len(data) > max_length:
            return data[: max_length - 3] + "..."
        return data
    elif isinstance(data, dict):
        return {k: truncate_strings(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_strings(item, max_length) for item in data]
    elif isinstance(data, tuple):
        return tuple(truncate_strings(item, max_length) for item in data)
    else:
        return data


def sanitize_for_logging(data: Any, max_length: int = 200) -> Any:
    """
    Sanitize and truncate data for logging.
    This is the public API used by application code when logging user-provided data.
    """
    return truncate_strings(sanitize_data(data), max_length)


def setup_logging(level: int = logging.INFO, log_format: str = "text") -> None:
    """
    Configure root logger with structured logging.

    Args:
        level: Logging level (e.g., logging.INFO)
        log_format: "text" or "json"
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        formatter = JSONFormatter()
    else:
        # Traditional text format with request ID if available
        class RequestIDFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                request_id = get_request_id()
                if request_id:
                    record.msg = f"[{request_id}] {record.msg}"
                return super().format(record)

        formatter = RequestIDFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with structured logging support."""
    return logging.getLogger(name)
