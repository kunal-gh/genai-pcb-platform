"""
Unit tests for secure storage service (Task 16.2).

Tests secure file storage with encryption, integrity verification, and deletion.
"""

import pytest
from uuid import uuid4
from pathlib import Path
import tempfile
import shutil

from src.services.secure_storage_service import SecureStorageService


@pytest.fixture
def storage_service(tmp_path):
    """Create a secure storage service with temporary directory."""
    service = SecureStorageService()
    # Override base directory to use temp directory for testing
    service.base_dir = tmp_path
    return service


@pytest.fixture
def test_user_id():
    """Generate a test user ID."""
    return uuid4()


@pytest.fixture
def test_design_id():
    """Generate a test design ID."""
    return uuid4()


class TestFileStorage:
    """Test file storage operations."""
    
    def test_save_design_file_encrypted(self, storage_service, test_user_id, test_design_id):
        """Test saving a file with encryption."""
        content = b"Test design file content"
        filename = "test_design.kicad_pcb"
        
        file_path, checksum, file_size = storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename=filename,
            content=content,
            encrypt=True,
        )
        
        assert Path(file_path).exists()
        assert checksum is not None
        assert file_size > 0
        
        # Verify file is encrypted (content should be different)
        with open(file_path, 'rb') as f:
            stored_content = f.read()
        assert stored_content != content
    
    def test_save_design_file_unencrypted(self, storage_service, test_user_id, test_design_id):
        """Test saving a file without encryption."""
        content = b"Unencrypted content"
        filename = "test.txt"
        
        file_path, checksum, file_size = storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename=filename,
            content=content,
            encrypt=False,
        )
        
        # Verify file is not encrypted
        with open(file_path, 'rb') as f:
            stored_content = f.read()
        assert stored_content == content
    
    def test_read_design_file_encrypted(self, storage_service, test_user_id, test_design_id):
        """Test reading an encrypted file."""
        original_content = b"Secret design data"
        filename = "encrypted.bin"
        
        # Save encrypted file
        file_path, _, _ = storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename=filename,
            content=original_content,
            encrypt=True,
        )
        
        # Read and decrypt
        decrypted_content = storage_service.read_design_file(file_path, decrypt=True)
        
        assert decrypted_content == original_content
    
    def test_read_nonexistent_file(self, storage_service):
        """Test reading a file that doesn't exist."""
        result = storage_service.read_design_file("/nonexistent/file.txt")
        assert result is None
    
    def test_delete_design_file(self, storage_service, test_user_id, test_design_id):
        """Test deleting a design file."""
        content = b"File to delete"
        filename = "delete_me.txt"
        
        # Save file
        file_path, _, _ = storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename=filename,
            content=content,
        )
        
        # Verify file exists
        assert Path(file_path).exists()
        
        # Delete file
        result = storage_service.delete_design_file(file_path)
        assert result is True
        
        # Verify file is deleted
        assert not Path(file_path).exists()
    
    def test_delete_nonexistent_file(self, storage_service):
        """Test deleting a file that doesn't exist."""
        result = storage_service.delete_design_file("/nonexistent/file.txt")
        assert result is False


class TestDirectoryOperations:
    """Test directory-level operations."""
    
    def test_delete_design_directory(self, storage_service, test_user_id, test_design_id):
        """Test deleting entire design directory."""
        # Create multiple files
        for i in range(3):
            storage_service.save_design_file(
                user_id=test_user_id,
                design_id=test_design_id,
                filename=f"file_{i}.txt",
                content=f"Content {i}".encode(),
            )
        
        # Verify directory exists
        design_dir = storage_service.base_dir / str(test_user_id) / str(test_design_id)
        assert design_dir.exists()
        
        # Delete directory
        result = storage_service.delete_design_directory(test_user_id, test_design_id)
        assert result is True
        
        # Verify directory is deleted
        assert not design_dir.exists()
    
    def test_list_design_files(self, storage_service, test_user_id, test_design_id):
        """Test listing all files for a design."""
        # Create multiple files
        filenames = ["file1.txt", "file2.bin", "file3.kicad_pcb"]
        for filename in filenames:
            storage_service.save_design_file(
                user_id=test_user_id,
                design_id=test_design_id,
                filename=filename,
                content=b"test content",
            )
        
        # List files
        files = storage_service.list_design_files(test_user_id, test_design_id)
        
        assert len(files) == 3
        # Verify all filenames are present
        for filename in filenames:
            assert any(filename in f for f in files)
    
    def test_list_files_empty_directory(self, storage_service, test_user_id, test_design_id):
        """Test listing files for non-existent design."""
        files = storage_service.list_design_files(test_user_id, test_design_id)
        assert files == []


class TestFileIntegrity:
    """Test file integrity verification."""
    
    def test_verify_file_integrity_success(self, storage_service, test_user_id, test_design_id):
        """Test successful integrity verification."""
        content = b"Important data"
        filename = "important.bin"
        
        # Save file
        file_path, checksum, _ = storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename=filename,
            content=content,
            encrypt=True,
        )
        
        # Verify integrity
        is_valid = storage_service.verify_file_integrity(
            file_path=file_path,
            expected_checksum=checksum,
            decrypt=True,
        )
        
        assert is_valid is True
    
    def test_verify_file_integrity_failure(self, storage_service, test_user_id, test_design_id):
        """Test integrity verification with wrong checksum."""
        content = b"Data"
        filename = "data.bin"
        
        # Save file
        file_path, _, _ = storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename=filename,
            content=content,
        )
        
        # Verify with wrong checksum
        is_valid = storage_service.verify_file_integrity(
            file_path=file_path,
            expected_checksum="wrong_checksum_123",
        )
        
        assert is_valid is False
    
    def test_get_file_size(self, storage_service, test_user_id, test_design_id):
        """Test getting file size."""
        content = b"A" * 1000  # 1KB
        filename = "size_test.bin"
        
        file_path, _, _ = storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename=filename,
            content=content,
        )
        
        size = storage_service.get_file_size(file_path)
        assert size > 0


class TestStorageStatistics:
    """Test storage statistics."""
    
    def test_get_storage_stats_for_user(self, storage_service, test_user_id, test_design_id):
        """Test getting storage statistics for a user."""
        # Create multiple files
        for i in range(3):
            storage_service.save_design_file(
                user_id=test_user_id,
                design_id=test_design_id,
                filename=f"file_{i}.txt",
                content=(f"Content {i}" * 100).encode(),
            )
        
        stats = storage_service.get_storage_stats(user_id=test_user_id)
        
        assert stats["user_id"] == str(test_user_id)
        assert stats["total_files"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_mb"] >= 0  # Can be 0.0 for small files
    
    def test_get_global_storage_stats(self, storage_service, test_user_id, test_design_id):
        """Test getting global storage statistics."""
        # Create files for multiple users
        user_id_2 = uuid4()
        design_id_2 = uuid4()
        
        storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename="file1.txt",
            content=b"Content 1",
        )
        
        storage_service.save_design_file(
            user_id=user_id_2,
            design_id=design_id_2,
            filename="file2.txt",
            content=b"Content 2",
        )
        
        stats = storage_service.get_storage_stats()
        
        assert stats["total_files"] == 2
        assert stats["total_size_bytes"] > 0
        assert "total_size_gb" in stats


class TestEncryptionIntegration:
    """Test encryption integration."""
    
    def test_encrypted_file_not_readable_without_decryption(
        self, storage_service, test_user_id, test_design_id
    ):
        """Test that encrypted files cannot be read without decryption."""
        secret_content = b"Super secret design data"
        filename = "secret.bin"
        
        # Save encrypted
        file_path, _, _ = storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename=filename,
            content=secret_content,
            encrypt=True,
        )
        
        # Read without decryption
        encrypted_content = storage_service.read_design_file(file_path, decrypt=False)
        
        # Should not match original
        assert encrypted_content != secret_content
        
        # Read with decryption
        decrypted_content = storage_service.read_design_file(file_path, decrypt=True)
        
        # Should match original
        assert decrypted_content == secret_content
    
    def test_large_file_encryption(self, storage_service, test_user_id, test_design_id):
        """Test encryption of large files."""
        # Create 1MB file
        large_content = b"X" * (1024 * 1024)
        filename = "large_file.bin"
        
        file_path, checksum, file_size = storage_service.save_design_file(
            user_id=test_user_id,
            design_id=test_design_id,
            filename=filename,
            content=large_content,
            encrypt=True,
        )
        
        # Verify can read back
        decrypted = storage_service.read_design_file(file_path, decrypt=True)
        assert decrypted == large_content
        
        # Verify integrity
        is_valid = storage_service.verify_file_integrity(
            file_path=file_path,
            expected_checksum=checksum,
            decrypt=True,
        )
        assert is_valid is True
