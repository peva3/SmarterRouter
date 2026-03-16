from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, or_

from router.config import settings
from router.database import get_session
from router.models import BackgroundTaskDLQ

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def enqueue_failed_task(
    task_name: str,
    error_message: str,
    payload: dict[str, Any] | None = None,
    max_retries: int | None = None,
) -> int | None:
    """Store a failed background task in the dead letter queue."""
    if not settings.dlq_enabled:
        return None

    now = _utc_now()
    retries = max_retries if max_retries is not None else settings.dlq_max_retries
    next_retry_at = now + timedelta(seconds=settings.dlq_retry_base_delay_seconds)

    try:
        with get_session() as session:
            entry = BackgroundTaskDLQ(
                task_name=task_name,
                payload=payload,
                error_message=error_message[:2000],
                status="failed",
                attempts=0,
                max_retries=retries,
                next_retry_at=next_retry_at,
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry.id
    except Exception as e:
        logger.error(f"Failed to enqueue DLQ task {task_name}: {e}")
        return None


def list_dlq_entries(status: str | None = None, limit: int = 50, offset: int = 0) -> list[BackgroundTaskDLQ]:
    """List DLQ entries ordered by newest first."""
    with get_session() as session:
        query = session.query(BackgroundTaskDLQ)
        if status:
            query = query.filter(BackgroundTaskDLQ.status == status)
        return query.order_by(BackgroundTaskDLQ.created_at.desc()).offset(offset).limit(limit).all()


def count_dlq_entries(status: str | None = None) -> int:
    """Count DLQ entries optionally filtered by status."""
    with get_session() as session:
        query = session.query(BackgroundTaskDLQ)
        if status:
            query = query.filter(BackgroundTaskDLQ.status == status)
        return query.count()


def get_due_retry_entries(limit: int) -> list[BackgroundTaskDLQ]:
    """Get entries due for retry based on next_retry_at and retry limits."""
    now = _utc_now()
    with get_session() as session:
        return (
            session.query(BackgroundTaskDLQ)
            .filter(
                and_(
                    BackgroundTaskDLQ.attempts < BackgroundTaskDLQ.max_retries,
                    or_(BackgroundTaskDLQ.status == "failed", BackgroundTaskDLQ.status == "retrying"),
                    or_(
                        BackgroundTaskDLQ.next_retry_at.is_(None),
                        BackgroundTaskDLQ.next_retry_at <= now,
                    ),
                )
            )
            .order_by(BackgroundTaskDLQ.created_at.asc())
            .limit(limit)
            .all()
        )


def get_dlq_entry(entry_id: int) -> BackgroundTaskDLQ | None:
    """Get a single DLQ entry by ID."""
    with get_session() as session:
        return session.query(BackgroundTaskDLQ).filter(BackgroundTaskDLQ.id == entry_id).first()


def mark_retry_success(entry_id: int) -> None:
    """Mark DLQ entry as resolved."""
    with get_session() as session:
        entry = session.query(BackgroundTaskDLQ).filter(BackgroundTaskDLQ.id == entry_id).first()
        if not entry:
            return
        entry.status = "resolved"
        entry.resolved_at = _utc_now()
        entry.last_attempt_at = _utc_now()
        session.commit()


def mark_retry_failure(entry_id: int, error_message: str) -> None:
    """Mark DLQ entry retry failure and schedule next attempt or dead state."""
    now = _utc_now()
    with get_session() as session:
        entry = session.query(BackgroundTaskDLQ).filter(BackgroundTaskDLQ.id == entry_id).first()
        if not entry:
            return

        entry.attempts += 1
        entry.last_attempt_at = now
        entry.error_message = error_message[:2000]

        if entry.attempts >= entry.max_retries:
            entry.status = "dead"
            entry.next_retry_at = None
        else:
            entry.status = "retrying"
            delay = settings.dlq_retry_base_delay_seconds * (2 ** (entry.attempts - 1))
            entry.next_retry_at = now + timedelta(seconds=delay)

        session.commit()
