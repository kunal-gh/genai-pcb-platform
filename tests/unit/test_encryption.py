"""
Unit tests for encryption service (Task 16.1).

Tests encryption/decryption of strings and binary data for sensitive information.
"""

import pytest
from src.services.encryption_service import (
    EncryptionService,
    encrypt_string,
    decrypt_string,
    encrypt_file_content,
    decrypt_file_content,
)


class TestEncryptionService:
    """Test the EncryptionService class."""
    
    def test_encrypt_decrypt_string(self):
        """Test basic string encryption and decryption."""
        service = EncryptionService()
        plaintext = "sensitive user data"
        
        encrypted = service.encrypt(plaintext)
        assert encrypted != plaintext
        assert isinstance(encrypted, str)
        
        decrypted = service.decrypt(encrypted)
        assert decrypted == plaintext
    
    def test_encrypt_empty_string_raises(self):
        """Test that encrypting empty string raises ValueError."""
        service = EncryptionService()
        with pytest.raises(ValueError, match="Cannot encrypt empty string"):
            service.encrypt("")
    
    def test_decrypt_invalid_token_returns_none(self):
        """Test that decrypting invalid data returns None."""
        service = EncryptionService()
        assert service.decrypt("invalid_encrypted_data") is None
        assert service.decrypt("") is None
    
    def test_encrypt_decrypt_unicode(self):
        """Test encryption of unicode characters."""
        service = EncryptionService()
        plaintext = "Hello 世界 🌍 Привет"
        
        encrypted = service.encrypt(plaintext)
        decrypted = service.decrypt(encrypted)
        
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_long_string(self):
        """Test encryption of long strings."""
        service = EncryptionService()
        plaintext = "A" * 10000  # 10KB string
        
        encrypted = service.encrypt(plaintext)
        decrypted = service.decrypt(encrypted)
        
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_bytes(self):
        """Test binary data encryption and decryption."""
        service = EncryptionService()
        data = b"binary file content \x00\x01\x02\xff"
        
        encrypted = service.encrypt_bytes(data)
        assert encrypted != data
        assert isinstance(encrypted, bytes)
        
        decrypted = service.decrypt_bytes(encrypted)
        assert decrypted == data
    
    def test_encrypt_empty_bytes_raises(self):
        """Test that encrypting empty bytes raises ValueError."""
        service = EncryptionService()
        with pytest.raises(ValueError, match="Cannot encrypt empty data"):
            service.encrypt_bytes(b"")
    
    def test_decrypt_invalid_bytes_returns_none(self):
        """Test that decrypting invalid bytes returns None."""
        service = EncryptionService()
        assert service.decrypt_bytes(b"invalid") is None
        assert service.decrypt_bytes(b"") is None
    
    def test_different_instances_same_key(self):
        """Test that different service instances use the same key."""
        service1 = EncryptionService()
        service2 = EncryptionService()
        
        plaintext = "test data"
        encrypted = service1.encrypt(plaintext)
        decrypted = service2.decrypt(encrypted)
        
        assert decrypted == plaintext


class TestConvenienceFunctions:
    """Test the convenience functions for encryption."""
    
    def test_encrypt_decrypt_string_functions(self):
        """Test convenience functions for string encryption."""
        plaintext = "sensitive information"
        
        encrypted = encrypt_string(plaintext)
        assert encrypted != plaintext
        
        decrypted = decrypt_string(encrypted)
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_file_content_functions(self):
        """Test convenience functions for file content encryption."""
        content = b"file content with binary data \x00\xff"
        
        encrypted = encrypt_file_content(content)
        assert encrypted != content
        
        decrypted = decrypt_file_content(encrypted)
        assert decrypted == content
    
    def test_multiple_encryptions_different_output(self):
        """Test that encrypting the same data twice produces different ciphertext."""
        plaintext = "same data"
        
        encrypted1 = encrypt_string(plaintext)
        encrypted2 = encrypt_string(plaintext)
        
        # Fernet includes a timestamp, so ciphertexts will differ
        # But both should decrypt to the same plaintext
        assert decrypt_string(encrypted1) == plaintext
        assert decrypt_string(encrypted2) == plaintext


class TestEncryptionSecurity:
    """Test security properties of encryption."""
    
    def test_encrypted_data_not_readable(self):
        """Test that encrypted data doesn't contain plaintext."""
        service = EncryptionService()
        plaintext = "secret password 12345"
        
        encrypted = service.encrypt(plaintext)
        
        # Encrypted data should not contain the plaintext
        assert "secret" not in encrypted
        assert "password" not in encrypted
        assert "12345" not in encrypted
    
    def test_tampering_detection(self):
        """Test that tampering with encrypted data is detected."""
        service = EncryptionService()
        plaintext = "important data"
        
        encrypted = service.encrypt(plaintext)
        
        # Tamper with the encrypted data
        tampered = encrypted[:-5] + "XXXXX"
        
        # Decryption should fail
        assert service.decrypt(tampered) is None
    
    def test_encryption_deterministic_with_same_key(self):
        """Test that encryption is consistent with the same key."""
        service1 = EncryptionService()
        service2 = EncryptionService()
        
        plaintext = "test"
        encrypted1 = service1.encrypt(plaintext)
        
        # Service2 should be able to decrypt service1's encryption
        decrypted = service2.decrypt(encrypted1)
        assert decrypted == plaintext
