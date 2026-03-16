from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from router.dlq import (
    count_dlq_entries,
    enqueue_failed_task,
    get_due_retry_entries,
    list_dlq_entries,
    mark_retry_failure,
    mark_retry_success,
)
from router.models import BackgroundTaskDLQ, Base


@pytest.fixture
def dlq_db():
    engine = create_engine("sqlite:///:memory:")
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with patch("router.database.engine", engine):
        with patch("router.database.SessionLocal", testing_session_local):
            yield engine


def test_enqueue_and_list_dlq_entries(dlq_db):
    entry_id = enqueue_failed_task(
        task_name="benchmark_sync",
        error_message="sync failed",
        payload={"model_count": 4},
    )
    assert entry_id is not None

    entries = list_dlq_entries()
    assert len(entries) == 1
    assert entries[0].task_name == "benchmark_sync"
    assert entries[0].status == "failed"
    assert count_dlq_entries("failed") == 1


def test_retry_state_transitions(dlq_db):
    entry_id = enqueue_failed_task(task_name="provider_db_download", error_message="network error")
    assert entry_id is not None

    due = get_due_retry_entries(limit=10)
    assert len(due) == 0

    # Force entry to become due now
    from router.database import get_session

    with get_session() as session:
        record = session.query(BackgroundTaskDLQ).filter(BackgroundTaskDLQ.id == entry_id).first()
        assert record is not None
        record.next_retry_at = datetime.now(UTC)
        session.commit()

    due = get_due_retry_entries(limit=10)
    assert len(due) == 1
    assert due[0].id == entry_id

    mark_retry_failure(entry_id, "still failing")
    entries = list_dlq_entries()
    assert entries[0].attempts == 1
    assert entries[0].status in {"retrying", "dead"}

    mark_retry_success(entry_id)
    entries = list_dlq_entries()
    assert entries[0].status == "resolved"
    assert entries[0].resolved_at is not None


def test_mark_retry_failure_sets_dead_after_max_retries(dlq_db):
    with patch("router.config.settings.dlq_retry_base_delay_seconds", 1):
        entry_id = enqueue_failed_task(
            task_name="profile_all_models",
            error_message="initial failure",
            max_retries=1,
        )

    assert entry_id is not None
    mark_retry_failure(entry_id, "final failure")

    entries = list_dlq_entries()
    assert entries[0].status == "dead"
    assert entries[0].attempts == 1
    assert entries[0].next_retry_at is None


def test_list_dlq_by_status(dlq_db):
    enqueue_failed_task("task_a", "err_a")
    enqueue_failed_task("task_b", "err_b")

    failed = list_dlq_entries(status="failed")
    assert len(failed) == 2

    # Insert a resolved record directly for filter validation
    from router.database import get_session

    with get_session() as session:
        record = session.query(BackgroundTaskDLQ).first()
        assert record is not None
        record.status = "resolved"
        record.resolved_at = datetime.now(UTC)
        session.commit()

    resolved = list_dlq_entries(status="resolved")
    assert len(resolved) == 1
