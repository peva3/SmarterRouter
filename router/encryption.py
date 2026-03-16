"""Encryption utilities for sensitive data storage.

Item #30: Encrypted storage for API keys using Fernet symmetric encryption.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    Fernet = None  # type: ignore[assignment]
    hashes = None  # type: ignore[assignment]
    PBKDF2HMAC = None  # type: ignore[assignment]
    CRYPTO_AVAILABLE = False

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages Fernet encryption for sensitive data.

    Uses PBKDF2 key derivation from a master key (environment variable or file).
    All encrypted values are stored with a prefix to identify them.
    """

    ENCRYPTED_PREFIX = "enc:"

    def __init__(self, master_key: str | None = None) -> None:
        """Initialize encryption manager.

        Args:
            master_key: Master encryption key. If None, reads from
                       ROUTER_ENCRYPTION_KEY environment variable.
        """
        self._fernet: Fernet | None = None
        self._master_key = master_key or os.environ.get("ROUTER_ENCRYPTION_KEY")

        if self._master_key:
            self._init_fernet()

    def _init_fernet(self) -> None:
        """Initialize Fernet instance from master key."""
        if not CRYPTO_AVAILABLE:
            logger.warning(
                "cryptography is not installed; encrypted API keys are unavailable"
            )
            self._fernet = None
            return

        try:
            assert self._master_key is not None
            # Use PBKDF2 to derive a proper Fernet key from the master key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"smarterrouter_salt_v1",  # Fixed salt for deterministic keys
                iterations=480000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
            self._fernet = Fernet(key)
            logger.info("Encryption manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self._fernet = None

    def is_configured(self) -> bool:
        """Check if encryption is properly configured."""
        return self._fernet is not None

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string value.

        Args:
            plaintext: The value to encrypt.

        Returns:
            Encrypted value with prefix, or plaintext if encryption not configured.

        Raises:
            ValueError: If encryption is not configured and plaintext is not already encrypted.
        """
        if not self._fernet:
            logger.warning("Encryption not configured, storing plaintext")
            return plaintext

        try:
            encrypted = self._fernet.encrypt(plaintext.encode())
            return f"{self.ENCRYPTED_PREFIX}{encrypted.decode()}"
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError(f"Failed to encrypt value: {e}") from e

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt an encrypted value.

        Args:
            ciphertext: The value to decrypt (may include prefix).

        Returns:
            Decrypted plaintext value.

        Raises:
            ValueError: If decryption fails or value is not properly encrypted.
        """
        if not ciphertext.startswith(self.ENCRYPTED_PREFIX):
            # Not encrypted, return as-is
            return ciphertext

        if not self._fernet:
            logger.error("Cannot decrypt: encryption not configured")
            raise ValueError("Encryption not configured but encrypted value found")

        try:
            encrypted_data = ciphertext[len(self.ENCRYPTED_PREFIX) :].encode()
            decrypted = self._fernet.decrypt(encrypted_data)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError(f"Failed to decrypt value: {e}") from e

    def maybe_encrypt(self, value: str | None) -> str | None:
        """Encrypt value if encryption is configured.

        Args:
            value: Value to potentially encrypt.

        Returns:
            Encrypted value with prefix, or original value if None.
        """
        if value is None:
            return None
        if not self._fernet:
            return value
        if value.startswith(self.ENCRYPTED_PREFIX):
            # Already encrypted
            return value
        return self.encrypt(value)

    def maybe_decrypt(self, value: str | None) -> str | None:
        """Decrypt value if it's encrypted.

        Args:
            value: Value to potentially decrypt.

        Returns:
            Decrypted plaintext, or original value if None or not encrypted.
        """
        if value is None:
            return None
        if not value.startswith(self.ENCRYPTED_PREFIX):
            return value
        return self.decrypt(value)


# Global instance
_encryption_manager: EncryptionManager | None = None


def get_encryption_manager() -> EncryptionManager:
    """Get or create the global encryption manager instance."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def encrypt_value(plaintext: str) -> str:
    """Encrypt a value using the global encryption manager."""
    return get_encryption_manager().encrypt(plaintext)


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a value using the global encryption manager."""
    return get_encryption_manager().decrypt(ciphertext)


def is_encrypted(value: str) -> bool:
    """Check if a value is encrypted (has the encrypted prefix)."""
    return value.startswith(EncryptionManager.ENCRYPTED_PREFIX)


def generate_encryption_key() -> str:
    """Generate a secure random encryption key.

    Returns:
        A URL-safe base64-encoded 32-byte key suitable for ROUTER_ENCRYPTION_KEY.
    """
    if CRYPTO_AVAILABLE:
        key = Fernet.generate_key()  # type: ignore[union-attr]
        return key.decode()

    # Fallback when cryptography is unavailable: generate random URL-safe bytes.
    # This is suitable as a high-entropy ROUTER_ENCRYPTION_KEY input.
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


# Convenience functions for API key fields
ENCRYPTED_API_KEY_FIELDS = {
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "cohere_api_key",
    "mistral_api_key",
    "artificial_analysis_api_key",
    "judge_api_key",
}


def encrypt_sensitive_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Encrypt sensitive fields in a data dictionary.

    Args:
        data: Dictionary containing potentially sensitive data.

    Returns:
        New dictionary with sensitive fields encrypted.
    """
    manager = get_encryption_manager()
    result = dict(data)

    for field in ENCRYPTED_API_KEY_FIELDS:
        if field in result and result[field] is not None:
            result[field] = manager.maybe_encrypt(result[field])

    return result


def decrypt_sensitive_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Decrypt sensitive fields in a data dictionary.

    Args:
        data: Dictionary containing potentially encrypted data.

    Returns:
        New dictionary with sensitive fields decrypted.
    """
    manager = get_encryption_manager()
    result = dict(data)

    for field in ENCRYPTED_API_KEY_FIELDS:
        if field in result and result[field] is not None:
            result[field] = manager.maybe_decrypt(result[field])

    return result
