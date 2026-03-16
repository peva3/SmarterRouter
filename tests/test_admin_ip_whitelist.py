"""Tests for admin endpoint IP whitelist (Item #26)."""

import sys
from unittest.mock import MagicMock

# Mock pandas to avoid binary compatibility issues in test environment
sys.modules.setdefault("pandas", MagicMock())

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

# Test the helper functions directly
from router.state import _get_client_ip, _ip_in_whitelist, verify_admin_token


class TestGetClientIp:
    """Tests for _get_client_ip helper."""

    def test_from_x_forwarded_for(self):
        """Should extract first IP from X-Forwarded-For header."""
        request = MagicMock()
        request.headers = {"x-forwarded-for": "203.0.113.50, 70.41.3.18, 150.172.238.178"}
        request.client = MagicMock(host="127.0.0.1")
        assert _get_client_ip(request) == "203.0.113.50"

    def test_from_x_forwarded_for_single(self):
        """Should handle single IP in X-Forwarded-For."""
        request = MagicMock()
        request.headers = {"x-forwarded-for": "10.0.0.5"}
        request.client = MagicMock(host="127.0.0.1")
        assert _get_client_ip(request) == "10.0.0.5"

    def test_from_client_host(self):
        """Should fall back to request.client.host when no forwarded header."""
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="192.168.1.100")
        assert _get_client_ip(request) == "192.168.1.100"

    def test_no_client(self):
        """Should return 'unknown' when no client info available."""
        request = MagicMock()
        request.headers = {}
        request.client = None
        assert _get_client_ip(request) == "unknown"


class TestIpInWhitelist:
    """Tests for _ip_in_whitelist helper."""

    def test_exact_match(self):
        """Should match exact IP address."""
        assert _ip_in_whitelist("192.168.1.1", ["192.168.1.1"]) is True

    def test_exact_no_match(self):
        """Should not match different IP."""
        assert _ip_in_whitelist("192.168.1.2", ["192.168.1.1"]) is False

    def test_cidr_match(self):
        """Should match IP within CIDR range."""
        assert _ip_in_whitelist("10.0.5.23", ["10.0.0.0/8"]) is True

    def test_cidr_no_match(self):
        """Should not match IP outside CIDR range."""
        assert _ip_in_whitelist("172.16.0.1", ["10.0.0.0/8"]) is False

    def test_cidr_24_match(self):
        """Should match /24 subnet."""
        assert _ip_in_whitelist("192.168.1.254", ["192.168.1.0/24"]) is True

    def test_cidr_24_no_match(self):
        """Should not match IP outside /24 subnet."""
        assert _ip_in_whitelist("192.168.2.1", ["192.168.1.0/24"]) is False

    def test_multiple_entries(self):
        """Should match any entry in the list."""
        whitelist = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
        assert _ip_in_whitelist("172.16.5.10", whitelist) is True

    def test_mixed_exact_and_cidr(self):
        """Should support both exact and CIDR entries."""
        whitelist = ["203.0.113.50", "10.0.0.0/8"]
        assert _ip_in_whitelist("203.0.113.50", whitelist) is True
        assert _ip_in_whitelist("10.5.5.5", whitelist) is True
        assert _ip_in_whitelist("192.168.1.1", whitelist) is False

    def test_invalid_client_ip(self):
        """Should return False for invalid client IP."""
        assert _ip_in_whitelist("not-an-ip", ["10.0.0.0/8"]) is False

    def test_invalid_whitelist_entry_skipped(self):
        """Should skip invalid entries and continue checking."""
        whitelist = ["not-valid", "10.0.0.0/8"]
        assert _ip_in_whitelist("10.5.5.5", whitelist) is True

    def test_unknown_ip(self):
        """Should return False for 'unknown' client IP."""
        assert _ip_in_whitelist("unknown", ["10.0.0.0/8"]) is False

    def test_empty_whitelist(self):
        """Should return False for empty whitelist."""
        assert _ip_in_whitelist("10.0.0.1", []) is False

    def test_localhost_ipv4(self):
        """Should match localhost."""
        assert _ip_in_whitelist("127.0.0.1", ["127.0.0.1"]) is True

    def test_ipv6_exact(self):
        """Should handle IPv6 exact match."""
        assert _ip_in_whitelist("::1", ["::1"]) is True

    def test_ipv6_cidr(self):
        """Should handle IPv6 CIDR match."""
        assert _ip_in_whitelist("fd00::1", ["fd00::/8"]) is True


class TestVerifyAdminTokenWithIpWhitelist:
    """Tests for verify_admin_token IP whitelist integration."""

    @pytest.mark.asyncio
    async def test_no_whitelist_allows_all(self):
        """When admin_allowed_ips is empty, any IP should be allowed (with valid key)."""

        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="1.2.3.4")

        credentials = MagicMock()
        credentials.credentials = "test-key"

        config = MagicMock()
        config.admin_api_key = "test-key"
        config.admin_allowed_ips = []

        result = await verify_admin_token(request, credentials, config)
        assert result is True

    @pytest.mark.asyncio
    async def test_whitelisted_ip_allowed(self):
        """Whitelisted IP with valid key should be allowed."""

        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="10.0.0.5")

        credentials = MagicMock()
        credentials.credentials = "test-key"

        config = MagicMock()
        config.admin_api_key = "test-key"
        config.admin_allowed_ips = ["10.0.0.0/8"]

        result = await verify_admin_token(request, credentials, config)
        assert result is True

    @pytest.mark.asyncio
    async def test_non_whitelisted_ip_rejected(self):
        """Non-whitelisted IP should get 403 even with valid key."""

        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="203.0.113.50")

        credentials = MagicMock()
        credentials.credentials = "test-key"

        config = MagicMock()
        config.admin_api_key = "test-key"
        config.admin_allowed_ips = ["10.0.0.0/8", "192.168.0.0/16"]

        with pytest.raises(HTTPException) as exc_info:
            await verify_admin_token(request, credentials, config)
        assert exc_info.value.status_code == 403
        assert "not in the admin whitelist" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_ip_check_before_credentials(self):
        """IP check should happen before credential check (fail fast)."""
        from main import verify_admin_token

        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="1.2.3.4")

        # Invalid credentials - but IP check should fail first
        credentials = MagicMock()
        credentials.credentials = "wrong-key"

        config = MagicMock()
        config.admin_api_key = "test-key"
        config.admin_allowed_ips = ["10.0.0.0/8"]

        with pytest.raises(HTTPException) as exc_info:
            await verify_admin_token(request, credentials, config)
        # Should be 403 (IP), not 401 (credentials)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_forwarded_ip_checked(self):
        """Should check X-Forwarded-For IP against whitelist."""
        from main import verify_admin_token

        request = MagicMock()
        request.headers = {"x-forwarded-for": "10.0.0.5, 172.16.0.1"}
        request.client = MagicMock(host="127.0.0.1")

        credentials = MagicMock()
        credentials.credentials = "test-key"

        config = MagicMock()
        config.admin_api_key = "test-key"
        config.admin_allowed_ips = ["10.0.0.0/8"]

        result = await verify_admin_token(request, credentials, config)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_api_key_still_401(self):
        """When no API key is configured, should still return 401."""
        from main import verify_admin_token

        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="10.0.0.1")

        credentials = None
        config = MagicMock()
        config.admin_api_key = None
        config.admin_allowed_ips = ["10.0.0.0/8"]

        with pytest.raises(HTTPException) as exc_info:
            await verify_admin_token(request, credentials, config)
        assert exc_info.value.status_code == 401
