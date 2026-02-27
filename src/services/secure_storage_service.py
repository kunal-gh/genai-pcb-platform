"""
Secure storage service (Task 16.2).

Provides secure file storage with encryption for design files and
sensitive user data. All files are encrypted at rest using AES-256.
"""

import logging
import os
import hashlib
from pathlib import Path
from typing import Optional, BinaryIO, Union
from uuid import UUID

from ..services.encryption_service import get_encryption_service
from ..config import settings

logger = logging.getLogger(__name__)


class SecureStorageService:
    """
    Service for secure file storage with encryption.
    
    All files are encrypted at rest using AES-256 encryption.
    File paths are organized by user and design for easy management.
    """
    
    def __init__(self):
        """Initialize secure storage service."""
        self.encryption_service = get_encryption_service()
        self.base_dir = Path(settings.GENERATED_DESIGNS_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_design_file(
        self,
        user_id: UUID,
        design_id: UUID,
        filename: str,
        content: bytes,
        encrypt: bool = True,
    ) -> tuple[str, str, int]:
        """
        Save a design file with optional encryption.
        
        Args:
            user_id: ID of user who owns the design
            design_id: ID of the design
            filename: Name of the file
            content: File content as bytes
            encrypt: Whether to encrypt the file (default: True)
            
        Returns:
            Tuple of (file_path, checksum, file_size)
        """
        # Create directory structure: base_dir/user_id/design_id/
        design_dir = self.base_dir / str(user_id) / str(design_id)
        design_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = design_dir / filename
        
        # Encrypt content if requested
        if encrypt:
            try:
                encrypted_content = self.encryption_service.encrypt_bytes(content)
                content_to_save = encrypted_content
                logger.debug(f"Encrypted file {filename} for design {design_id}")
            except Exception as e:
                logger.error(f"Failed to encrypt file {filename}: {e}")
                raise
        else:
            content_to_save = content
        
        # Calculate checksum of original content (before encryption)
        checksum = hashlib.sha256(content).hexdigest()
        
        # Save file
        try:
            with open(file_path, 'wb') as f:
                f.write(content_to_save)
            
            file_size = len(content_to_save)
            
            logger.info(
                f"Saved {'encrypted ' if encrypt else ''}file {filename} "
                f"for design {design_id} ({file_size} bytes)"
            )
            
            return str(file_path), checksum, file_size
            
        except Exception as e:
            logger.error(f"Failed to save file {filename}: {e}")
            raise
    
    def read_design_file(
        self,
        file_path: str,
        decrypt: bool = True,
    ) -> Optional[bytes]:
        """
        Read a design file with optional decryption.
        
        Args:
            file_path: Path to the file
            decrypt: Whether to decrypt the file (default: True)
            
        Returns:
            File content as bytes, or None if file not found
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            with open(path, 'rb') as f:
                content = f.read()
            
            # Decrypt content if requested
            if decrypt:
                try:
                    decrypted_content = self.encryption_service.decrypt_bytes(content)
                    if decrypted_content is None:
                        logger.error(f"Failed to decrypt file {file_path}")
                        return None
                    return decrypted_content
                except Exception as e:
                    logger.error(f"Failed to decrypt file {file_path}: {e}")
                    return None
            else:
                return content
                
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def delete_design_file(self, file_path: str) -> bool:
        """
        Delete a design file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file was deleted, False otherwise
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return False
        
        try:
            path.unlink()
            logger.info(f"Deleted file {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def delete_design_directory(self, user_id: UUID, design_id: UUID) -> bool:
        """
        Delete entire design directory.
        
        Args:
            user_id: ID of user who owns the design
            design_id: ID of the design
            
        Returns:
            True if directory was deleted, False otherwise
        """
        design_dir = self.base_dir / str(user_id) / str(design_id)
        
        if not design_dir.exists():
            logger.warning(f"Design directory not found: {design_dir}")
            return False
        
        try:
            import shutil
            shutil.rmtree(design_dir)
            logger.info(f"Deleted design directory {design_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete design directory {design_dir}: {e}")
            return False
    
    def verify_file_integrity(
        self,
        file_path: str,
        expected_checksum: str,
        decrypt: bool = True,
    ) -> bool:
        """
        Verify file integrity using checksum.
        
        Args:
            file_path: Path to the file
            expected_checksum: Expected SHA-256 checksum
            decrypt: Whether to decrypt before checking (default: True)
            
        Returns:
            True if checksum matches, False otherwise
        """
        content = self.read_design_file(file_path, decrypt=decrypt)
        if content is None:
            return False
        
        actual_checksum = hashlib.sha256(content).hexdigest()
        
        if actual_checksum != expected_checksum:
            logger.warning(
                f"Checksum mismatch for {file_path}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
            return False
        
        return True
    
    def get_file_size(self, file_path: str) -> Optional[int]:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes, or None if file not found
        """
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        return path.stat().st_size
    
    def list_design_files(self, user_id: UUID, design_id: UUID) -> list[str]:
        """
        List all files for a design.
        
        Args:
            user_id: ID of user who owns the design
            design_id: ID of the design
            
        Returns:
            List of file paths
        """
        design_dir = self.base_dir / str(user_id) / str(design_id)
        
        if not design_dir.exists():
            return []
        
        files = []
        for file_path in design_dir.iterdir():
            if file_path.is_file():
                files.append(str(file_path))
        
        return files
    
    def get_storage_stats(self, user_id: Optional[UUID] = None) -> dict:
        """
        Get storage statistics.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            Dictionary with storage statistics
        """
        if user_id:
            user_dir = self.base_dir / str(user_id)
            if not user_dir.exists():
                return {
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "total_size_mb": 0.0,
                }
            
            total_files = 0
            total_size = 0
            
            for file_path in user_dir.rglob('*'):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
            
            return {
                "user_id": str(user_id),
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }
        else:
            # Global statistics
            total_files = 0
            total_size = 0
            
            for file_path in self.base_dir.rglob('*'):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
            }


# Global service instance
_secure_storage_service: Optional[SecureStorageService] = None


def get_secure_storage_service() -> SecureStorageService:
    """
    Get or create the global secure storage service instance.
    
    Returns:
        SecureStorageService instance
    """
    global _secure_storage_service
    if _secure_storage_service is None:
        _secure_storage_service = SecureStorageService()
    return _secure_storage_service
