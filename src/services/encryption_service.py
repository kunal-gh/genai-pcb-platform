"""
Encryption service for sensitive user data (Task 16.1).

Provides AES-256 encryption/decryption for sensitive information like design files,
user data, and other confidential information. Uses Fernet (symmetric encryption)
which implements AES-128-CBC with HMAC for authentication.

For production use with AES-256, consider using cryptography.hazmat with AES-GCM.
"""

import logging
import base64
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from ..config import settings

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    Service for encrypting and decrypting sensitive user data.
    
    Uses Fernet symmetric encryption (AES-128-CBC + HMAC) for data at rest.
    The encryption key is derived from the SECRET_KEY in settings.
    """
    
    def __init__(self):
        """Initialize encryption service with key derived from settings."""
        self._fernet = self._create_fernet()
    
    def _create_fernet(self) -> Fernet:
        """
        Create Fernet cipher using key derived from SECRET_KEY.
        
        Uses PBKDF2 to derive a proper Fernet key from the SECRET_KEY.
        """
        # Use a fixed salt for key derivation (in production, consider per-user salts)
        salt = b"genai-pcb-platform-salt-v1"
        
        # Derive a 32-byte key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(
            kdf.derive(settings.SECRET_KEY.encode())
        )
        
        return Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext string to encrypted string.
        
        Args:
            plaintext: The string to encrypt
            
        Returns:
            Base64-encoded encrypted string
            
        Raises:
            ValueError: If plaintext is empty
        """
        if not plaintext:
            raise ValueError("Cannot encrypt empty string")
        
        try:
            encrypted_bytes = self._fernet.encrypt(plaintext.encode('utf-8'))
            return encrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted: str) -> Optional[str]:
        """
        Decrypt encrypted string back to plaintext.
        
        Args:
            encrypted: Base64-encoded encrypted string
            
        Returns:
            Decrypted plaintext string, or None if decryption fails
        """
        if not encrypted:
            return None
        
        try:
            decrypted_bytes = self._fernet.decrypt(encrypted.encode('utf-8'))
            return decrypted_bytes.decode('utf-8')
        except InvalidToken:
            logger.error("Decryption failed: Invalid token or corrupted data")
            return None
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def encrypt_bytes(self, data: bytes) -> bytes:
        """
        Encrypt binary data.
        
        Args:
            data: Binary data to encrypt
            
        Returns:
            Encrypted binary data
            
        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Cannot encrypt empty data")
        
        try:
            return self._fernet.encrypt(data)
        except Exception as e:
            logger.error(f"Binary encryption failed: {e}")
            raise
    
    def decrypt_bytes(self, encrypted_data: bytes) -> Optional[bytes]:
        """
        Decrypt binary data.
        
        Args:
            encrypted_data: Encrypted binary data
            
        Returns:
            Decrypted binary data, or None if decryption fails
        """
        if not encrypted_data:
            return None
        
        try:
            return self._fernet.decrypt(encrypted_data)
        except InvalidToken:
            logger.error("Binary decryption failed: Invalid token or corrupted data")
            return None
        except Exception as e:
            logger.error(f"Binary decryption failed: {e}")
            return None


# Global encryption service instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """
    Get or create the global encryption service instance.
    
    Returns:
        EncryptionService instance
    """
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


# Convenience functions for direct use
def encrypt_string(plaintext: str) -> str:
    """Encrypt a string using the global encryption service."""
    return get_encryption_service().encrypt(plaintext)


def decrypt_string(encrypted: str) -> Optional[str]:
    """Decrypt a string using the global encryption service."""
    return get_encryption_service().decrypt(encrypted)


def encrypt_file_content(content: bytes) -> bytes:
    """Encrypt file content using the global encryption service."""
    return get_encryption_service().encrypt_bytes(content)


def decrypt_file_content(encrypted_content: bytes) -> Optional[bytes]:
    """Decrypt file content using the global encryption service."""
    return get_encryption_service().decrypt_bytes(encrypted_content)
