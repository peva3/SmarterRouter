"""Admin audit logging for SmarterRouter.

Records all admin actions with IP, user agent, parameters, result, and timing.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from router.database import get_session
from router.models import AdminAuditLog

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """In-flight audit entry builder."""

    action: str
    endpoint: str
    method: str
    ip_address: str | None = None
    user_agent: str | None = None
    parameters: dict[str, Any] | None = None
    result_summary: str | None = None
    status_code: int = 200
    duration_ms: float | None = None
    _start_time: float = field(default_factory=time.monotonic, repr=False)

    def set_result(self, summary: str, status_code: int = 200) -> None:
        """Set the result summary and status code."""
        self.result_summary = summary[:500] if summary else None  # Truncate for safety
        self.status_code = status_code

    def finish(self) -> None:
        """Calculate duration from start time."""
        self.duration_ms = (time.monotonic() - self._start_time) * 1000


def record_audit_log(entry: AuditEntry) -> None:
    """Persist an audit entry to the database.

    Runs synchronously in a short-lived session. Failures are logged but
    do not propagate — audit logging must never break admin operations.
    """
    try:
        with get_session() as session:
            log = AdminAuditLog(
                action=entry.action,
                endpoint=entry.endpoint,
                method=entry.method,
                ip_address=entry.ip_address,
                user_agent=_sanitize_user_agent(entry.user_agent),
                parameters=entry.parameters,
                result_summary=entry.result_summary,
                status_code=entry.status_code,
                duration_ms=entry.duration_ms,
            )
            session.add(log)
            session.commit()
    except Exception:
        logger.warning("Failed to write admin audit log", exc_info=True)


def get_audit_logs(
    *,
    action: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[AdminAuditLog], int]:
    """Query audit logs with optional filtering.

    Returns (entries, total_count).
    """
    from sqlalchemy import func

    with get_session() as session:
        query = session.query(AdminAuditLog)
        count_query = session.query(func.count(AdminAuditLog.id))

        if action:
            query = query.filter(AdminAuditLog.action == action)
            count_query = count_query.filter(AdminAuditLog.action == action)

        total = count_query.scalar() or 0
        entries = (
            query.order_by(AdminAuditLog.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        # Eagerly load attributes before session closes
        for e in entries:
            _ = (
                e.id, e.timestamp, e.action, e.endpoint, e.method,
                e.ip_address, e.user_agent, e.parameters,
                e.result_summary, e.status_code, e.duration_ms,
            )
        return entries, total


@asynccontextmanager
async def audit_admin_action(
    request: Request,
    action: str,
    parameters: dict[str, Any] | None = None,
):
    """Async context manager that wraps an admin action with audit logging.

    Usage::

        async with audit_admin_action(request, "reprofile", {"force": True}) as audit:
            result = await do_reprofile(force=True)
            audit.set_result(f"Profiled {len(result)} models")

    On exit the entry is persisted. Exceptions set status_code=500 automatically.
    """
    entry = AuditEntry(
        action=action,
        endpoint=request.url.path,
        method=request.method,
        ip_address=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        parameters=parameters,
    )
    try:
        yield entry
    except Exception as exc:
        entry.set_result(f"Error: {type(exc).__name__}: {str(exc)[:200]}", status_code=500)
        raise
    finally:
        entry.finish()
        record_audit_log(entry)


def _get_client_ip(request: Request) -> str | None:
    """Extract client IP, considering X-Forwarded-For header."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _sanitize_user_agent(ua: str | None) -> str | None:
    """Truncate user agent to reasonable length."""
    if ua is None:
        return None
    return ua[:500]
