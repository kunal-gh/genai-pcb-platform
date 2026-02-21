"""
Data privacy service (Task 16.2).

Provides data privacy features including complete data deletion,
data export, and GDPR compliance capabilities.
"""

import logging
import os
import shutil
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pathlib import Path

from sqlalchemy.orm import Session as DBSession

from ..models.user import User
from ..models.design import DesignProject, DesignFile
from ..models.session import Session
from ..models.audit_log import AuditLog, AuditAction, AuditSeverity
from ..services.audit_service import log_audit_event, log_data_deletion
from ..services.encryption_service import get_encryption_service
from ..config import settings

logger = logging.getLogger(__name__)


class DataPrivacyService:
    """
    Service for handling data privacy operations.
    
    Provides complete data deletion, data export, and compliance features.
    """
    
    def __init__(self):
        """Initialize data privacy service."""
        self.encryption_service = get_encryption_service()
    
    def delete_user_data(
        self,
        db: DBSession,
        user_id: UUID,
        requesting_user_id: UUID,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Completely delete all user data (GDPR right to erasure).
        
        This is a destructive operation that:
        1. Deletes all design projects and files
        2. Deletes all sessions
        3. Anonymizes audit logs (keeps for compliance but removes PII)
        4. Deletes the user account
        
        Args:
            db: Database session
            user_id: ID of user whose data to delete
            requesting_user_id: ID of user requesting deletion (for audit)
            ip_address: IP address of requester
            
        Returns:
            Dictionary with deletion summary
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        username = user.username
        
        logger.info(f"Starting complete data deletion for user {username} ({user_id})")
        
        # Track what we delete
        summary = {
            "user_id": str(user_id),
            "username": username,
            "deleted_at": datetime.utcnow().isoformat(),
            "designs_deleted": 0,
            "files_deleted": 0,
            "sessions_deleted": 0,
            "audit_logs_anonymized": 0,
        }
        
        # 1. Delete all design projects and their files
        designs = db.query(DesignProject).filter(DesignProject.user_id == user_id).all()
        for design in designs:
            # Delete physical files
            files_deleted = self._delete_design_files(design)
            summary["files_deleted"] += files_deleted
            
            # Log the deletion
            log_data_deletion(
                db=db,
                user_id=requesting_user_id,
                username=username,
                resource_type="design",
                resource_id=design.id,
                resource_name=design.name,
                details={"files_deleted": files_deleted},
            )
            
            # Delete database record (cascade will handle related records)
            db.delete(design)
            summary["designs_deleted"] += 1
        
        # 2. Delete all sessions
        sessions = db.query(Session).filter(Session.user_id == user_id).all()
        for session in sessions:
            db.delete(session)
            summary["sessions_deleted"] += 1
        
        # 3. Anonymize audit logs (keep for compliance but remove PII)
        # We keep the logs but remove identifying information
        audit_logs = db.query(AuditLog).filter(AuditLog.user_id == user_id).all()
        for log in audit_logs:
            log.user_id = None
            log.username = f"[DELETED_USER_{user_id}]"
            log.ip_address = None
            log.user_agent = None
            summary["audit_logs_anonymized"] += 1
        
        # 4. Log the user deletion before deleting the user
        log_audit_event(
            db=db,
            action=AuditAction.DATA_DELETED,
            user_id=requesting_user_id,
            username=username,
            severity=AuditSeverity.WARNING,
            ip_address=ip_address,
            resource_type="user",
            resource_id=user_id,
            resource_name=username,
            description=f"Complete data deletion for user {username}",
            details=summary,
        )
        
        # 5. Delete the user account
        db.delete(user)
        
        # Commit all changes
        db.commit()
        
        logger.info(f"Completed data deletion for user {username}: {summary}")
        
        return summary
    
    def delete_design(
        self,
        db: DBSession,
        design_id: UUID,
        user_id: UUID,
        username: str,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Completely delete a design project and all associated files.
        
        Args:
            db: Database session
            design_id: ID of design to delete
            user_id: ID of user requesting deletion
            username: Username of requester
            ip_address: IP address of requester
            
        Returns:
            Dictionary with deletion summary
        """
        design = db.query(DesignProject).filter(DesignProject.id == design_id).first()
        if not design:
            raise ValueError(f"Design {design_id} not found")
        
        # Verify ownership
        if design.user_id != user_id:
            log_audit_event(
                db=db,
                action=AuditAction.UNAUTHORIZED_ACCESS,
                user_id=user_id,
                username=username,
                severity=AuditSeverity.WARNING,
                ip_address=ip_address,
                resource_type="design",
                resource_id=design_id,
                description=f"Unauthorized deletion attempt by {username}",
                success="failure",
            )
            raise PermissionError(f"User {user_id} does not own design {design_id}")
        
        design_name = design.name
        
        # Delete physical files
        files_deleted = self._delete_design_files(design)
        
        summary = {
            "design_id": str(design_id),
            "design_name": design_name,
            "files_deleted": files_deleted,
            "deleted_at": datetime.utcnow().isoformat(),
        }
        
        # Log the deletion
        log_data_deletion(
            db=db,
            user_id=user_id,
            username=username,
            resource_type="design",
            resource_id=design_id,
            resource_name=design_name,
            details=summary,
        )
        
        # Delete database record
        db.delete(design)
        db.commit()
        
        logger.info(f"Deleted design {design_name} ({design_id}): {files_deleted} files")
        
        return summary
    
    def _delete_design_files(self, design: DesignProject) -> int:
        """
        Delete all physical files associated with a design.
        
        Args:
            design: DesignProject instance
            
        Returns:
            Number of files deleted
        """
        files_deleted = 0
        
        for design_file in design.files:
            file_path = Path(design_file.file_path)
            
            # Delete the file if it exists
            if file_path.exists():
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        files_deleted += 1
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        files_deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")
        
        # Also delete the design directory if it exists
        design_dir = Path(settings.GENERATED_DESIGNS_DIR) / str(design.id)
        if design_dir.exists():
            try:
                shutil.rmtree(design_dir)
                logger.info(f"Deleted design directory {design_dir}")
            except Exception as e:
                logger.error(f"Failed to delete design directory {design_dir}: {e}")
        
        return files_deleted
    
    def export_user_data(
        self,
        db: DBSession,
        user_id: UUID,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export all user data (GDPR right to data portability).
        
        Creates a comprehensive export of all user data in JSON format.
        
        Args:
            db: Database session
            user_id: ID of user whose data to export
            ip_address: IP address of requester
            
        Returns:
            Dictionary with all user data
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        logger.info(f"Exporting data for user {user.username} ({user_id})")
        
        # Build comprehensive data export
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "user": {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "updated_at": user.updated_at.isoformat(),
            },
            "designs": [],
            "sessions": [],
            "audit_logs": [],
        }
        
        # Export designs
        designs = db.query(DesignProject).filter(DesignProject.user_id == user_id).all()
        for design in designs:
            design_data = {
                "id": str(design.id),
                "name": design.name,
                "description": design.description,
                "natural_language_prompt": design.natural_language_prompt,
                "status": design.status.value,
                "version": design.version,
                "branch": design.branch,
                "created_at": design.created_at.isoformat(),
                "updated_at": design.updated_at.isoformat(),
                "files": [
                    {
                        "id": str(f.id),
                        "file_type": f.file_type.value,
                        "file_path": f.file_path,
                        "file_size": f.file_size,
                        "created_at": f.created_at.isoformat(),
                    }
                    for f in design.files
                ],
            }
            export_data["designs"].append(design_data)
        
        # Export sessions (active and recent)
        sessions = db.query(Session).filter(Session.user_id == user_id).all()
        for session in sessions:
            session_data = {
                "id": str(session.id),
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "is_active": session.is_active,
                "ip_address": session.ip_address,
                "device_info": session.device_info,
            }
            export_data["sessions"].append(session_data)
        
        # Export audit logs (last 1000 entries)
        audit_logs = (
            db.query(AuditLog)
            .filter(AuditLog.user_id == user_id)
            .order_by(AuditLog.timestamp.desc())
            .limit(1000)
            .all()
        )
        for log in audit_logs:
            log_data = {
                "id": str(log.id),
                "action": log.action.value,
                "severity": log.severity.value,
                "timestamp": log.timestamp.isoformat(),
                "resource_type": log.resource_type,
                "resource_id": str(log.resource_id) if log.resource_id else None,
                "description": log.description,
                "success": log.success,
            }
            export_data["audit_logs"].append(log_data)
        
        # Log the export
        log_audit_event(
            db=db,
            action=AuditAction.DATA_EXPORT_REQUEST,
            user_id=user_id,
            username=user.username,
            ip_address=ip_address,
            description=f"User {user.username} exported their data",
            details={
                "designs_count": len(export_data["designs"]),
                "sessions_count": len(export_data["sessions"]),
                "audit_logs_count": len(export_data["audit_logs"]),
            },
        )
        
        logger.info(f"Exported data for user {user.username}: {len(export_data['designs'])} designs")
        
        return export_data
    
    def anonymize_user_data(
        self,
        db: DBSession,
        user_id: UUID,
        requesting_user_id: UUID,
    ) -> Dict[str, Any]:
        """
        Anonymize user data while preserving designs.
        
        This is useful when a user wants to delete their account but
        preserve their designs for archival purposes.
        
        Args:
            db: Database session
            user_id: ID of user to anonymize
            requesting_user_id: ID of user requesting anonymization
            
        Returns:
            Dictionary with anonymization summary
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        original_username = user.username
        
        # Anonymize user account
        user.username = f"anonymous_{user_id}"
        user.email = f"deleted_{user_id}@example.com"
        user.is_active = False
        
        # Delete all sessions
        sessions_deleted = db.query(Session).filter(Session.user_id == user_id).delete()
        
        # Anonymize audit logs
        audit_logs = db.query(AuditLog).filter(AuditLog.user_id == user_id).all()
        for log in audit_logs:
            log.username = f"[ANONYMIZED_{user_id}]"
            log.ip_address = None
            log.user_agent = None
        
        summary = {
            "user_id": str(user_id),
            "original_username": original_username,
            "anonymized_at": datetime.utcnow().isoformat(),
            "sessions_deleted": sessions_deleted,
            "audit_logs_anonymized": len(audit_logs),
            "designs_preserved": db.query(DesignProject).filter(DesignProject.user_id == user_id).count(),
        }
        
        # Log the anonymization
        log_audit_event(
            db=db,
            action=AuditAction.DATA_DELETED,
            user_id=requesting_user_id,
            username=original_username,
            severity=AuditSeverity.WARNING,
            resource_type="user",
            resource_id=user_id,
            description=f"User {original_username} anonymized",
            details=summary,
        )
        
        db.commit()
        
        logger.info(f"Anonymized user {original_username}: {summary}")
        
        return summary


# Global service instance
_data_privacy_service: Optional[DataPrivacyService] = None


def get_data_privacy_service() -> DataPrivacyService:
    """
    Get or create the global data privacy service instance.
    
    Returns:
        DataPrivacyService instance
    """
    global _data_privacy_service
    if _data_privacy_service is None:
        _data_privacy_service = DataPrivacyService()
    return _data_privacy_service
