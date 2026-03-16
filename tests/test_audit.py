"""Tests for admin audit logging (Item #24)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from router.audit import (
    AuditEntry,
    audit_admin_action,
    get_audit_logs,
    record_audit_log,
)
from router.models import Base


@pytest.fixture
def audit_db():
    """Create an in-memory SQLite database with audit log table."""
    engine = create_engine("sqlite:///:memory:")
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with patch("router.database.engine", engine):
        with patch("router.database.SessionLocal", testing_session_local):
            # Also patch in audit module since it imports get_session
            yield engine


class TestAuditEntry:
    """Test AuditEntry dataclass."""

    def test_create_entry(self):
        entry = AuditEntry(
            action="reprofile",
            endpoint="/admin/reprofile",
            method="POST",
        )
        assert entry.action == "reprofile"
        assert entry.endpoint == "/admin/reprofile"
        assert entry.method == "POST"
        assert entry.status_code == 200
        assert entry.result_summary is None

    def test_set_result(self):
        entry = AuditEntry(
            action="cache_clear",
            endpoint="/admin/cache/clear",
            method="POST",
        )
        entry.set_result("Cleared 42 entries", status_code=200)
        assert entry.result_summary == "Cleared 42 entries"
        assert entry.status_code == 200

    def test_set_result_truncates(self):
        entry = AuditEntry(
            action="test",
            endpoint="/admin/test",
            method="POST",
        )
        long_summary = "x" * 1000
        entry.set_result(long_summary)
        assert len(entry.result_summary) == 500

    def test_finish_calculates_duration(self):
        entry = AuditEntry(
            action="test",
            endpoint="/admin/test",
            method="POST",
        )
        entry.finish()
        assert entry.duration_ms is not None
        assert entry.duration_ms >= 0


class TestRecordAuditLog:
    """Test persisting audit entries to database."""

    def test_record_audit_log(self, audit_db):
        entry = AuditEntry(
            action="reprofile",
            endpoint="/admin/reprofile",
            method="POST",
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0",
            parameters={"force": True},
        )
        entry.set_result("Profiled 5 models")
        entry.finish()

        record_audit_log(entry)

        # Verify it was saved
        logs, total = get_audit_logs()
        assert total == 1
        assert logs[0].action == "reprofile"
        assert logs[0].endpoint == "/admin/reprofile"
        assert logs[0].method == "POST"
        assert logs[0].ip_address == "192.168.1.100"
        assert logs[0].user_agent == "TestAgent/1.0"
        assert logs[0].parameters == {"force": True}
        assert logs[0].result_summary == "Profiled 5 models"
        assert logs[0].status_code == 200
        assert logs[0].duration_ms is not None

    def test_record_multiple_entries(self, audit_db):
        for i in range(5):
            entry = AuditEntry(
                action=f"action_{i}",
                endpoint=f"/admin/action_{i}",
                method="POST",
            )
            entry.set_result(f"Result {i}")
            entry.finish()
            record_audit_log(entry)

        logs, total = get_audit_logs()
        assert total == 5
        # Most recent first
        assert logs[0].action == "action_4"
        assert logs[4].action == "action_0"

    def test_record_with_none_user_agent(self, audit_db):
        entry = AuditEntry(
            action="test",
            endpoint="/admin/test",
            method="GET",
            user_agent=None,
        )
        entry.finish()
        record_audit_log(entry)

        logs, total = get_audit_logs()
        assert total == 1
        assert logs[0].user_agent is None

    def test_record_survives_db_error(self):
        """Audit logging should never raise — failures are swallowed."""
        entry = AuditEntry(
            action="test",
            endpoint="/admin/test",
            method="POST",
        )
        entry.finish()

        # Simulate a broken database session
        with patch("router.audit.get_session") as mock_session:
            mock_session.side_effect = RuntimeError("DB exploded")
            # Should not raise
            record_audit_log(entry)


class TestGetAuditLogs:
    """Test querying audit logs."""

    def test_filter_by_action(self, audit_db):
        for action in ["reprofile", "cache_clear", "reprofile", "sync_benchmarks"]:
            entry = AuditEntry(action=action, endpoint=f"/admin/{action}", method="POST")
            entry.finish()
            record_audit_log(entry)

        logs, total = get_audit_logs(action="reprofile")
        assert total == 2
        assert len(logs) == 2
        assert all(l.action == "reprofile" for l in logs)

    def test_pagination(self, audit_db):
        for i in range(10):
            entry = AuditEntry(
                action=f"action_{i}",
                endpoint="/admin/test",
                method="POST",
            )
            entry.finish()
            record_audit_log(entry)

        # First page
        logs, total = get_audit_logs(limit=3, offset=0)
        assert total == 10
        assert len(logs) == 3

        # Second page
        logs, total = get_audit_logs(limit=3, offset=3)
        assert total == 10
        assert len(logs) == 3

    def test_empty_log(self, audit_db):
        logs, total = get_audit_logs()
        assert total == 0
        assert logs == []


class TestAuditAdminAction:
    """Test the async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self, audit_db):
        """Successful action records audit entry."""
        mock_request = MagicMock()
        mock_request.url.path = "/admin/reprofile"
        mock_request.method = "POST"
        mock_request.headers = {"user-agent": "TestBot/1.0"}
        mock_request.client = MagicMock()
        mock_request.client.host = "10.0.0.1"

        async with audit_admin_action(
            mock_request, "reprofile", {"force": True}
        ) as audit:
            audit.set_result("Profiled 3 models")

        logs, total = get_audit_logs()
        assert total == 1
        assert logs[0].action == "reprofile"
        assert logs[0].ip_address == "10.0.0.1"
        assert logs[0].parameters == {"force": True}
        assert logs[0].result_summary == "Profiled 3 models"
        assert logs[0].status_code == 200
        assert logs[0].duration_ms is not None
        assert logs[0].duration_ms >= 0

    @pytest.mark.asyncio
    async def test_context_manager_exception(self, audit_db):
        """Exception in action records error with status 500."""
        mock_request = MagicMock()
        mock_request.url.path = "/admin/cache/clear"
        mock_request.method = "POST"
        mock_request.headers = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        with pytest.raises(ValueError, match="boom"):
            async with audit_admin_action(mock_request, "cache_clear") as audit:
                raise ValueError("boom")

        logs, total = get_audit_logs()
        assert total == 1
        assert logs[0].action == "cache_clear"
        assert logs[0].status_code == 500
        assert "ValueError: boom" in logs[0].result_summary

    @pytest.mark.asyncio
    async def test_context_manager_x_forwarded_for(self, audit_db):
        """X-Forwarded-For header is preferred for IP extraction."""
        mock_request = MagicMock()
        mock_request.url.path = "/admin/test"
        mock_request.method = "GET"
        mock_request.headers = {
            "user-agent": "TestBot",
            "x-forwarded-for": "203.0.113.50, 70.41.3.18",
        }
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        async with audit_admin_action(mock_request, "test") as audit:
            audit.set_result("OK")

        logs, total = get_audit_logs()
        assert total == 1
        # Should use first IP from X-Forwarded-For
        assert logs[0].ip_address == "203.0.113.50"

    @pytest.mark.asyncio
    async def test_context_manager_no_client(self, audit_db):
        """Handles request with no client attribute gracefully."""
        mock_request = MagicMock()
        mock_request.url.path = "/admin/test"
        mock_request.method = "GET"
        mock_request.headers = {}
        mock_request.client = None

        async with audit_admin_action(mock_request, "test") as audit:
            audit.set_result("OK")

        logs, total = get_audit_logs()
        assert total == 1
        assert logs[0].ip_address is None
