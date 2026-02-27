"""
Audit log model for tracking user actions (Task 16.2).

SQLAlchemy model for comprehensive audit logging of all user actions
and system events for security and compliance.
"""

from sqlalchemy import Column, String, DateTime, Text, JSON, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from .database import Base


class AuditAction(str, enum.Enum):
    """Audit action types."""
    # Authentication actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_REGISTER = "user_register"
    PASSWORD_CHANGE = "password_change"
    SESSION_REVOKED = "session_revoked"
    
    # Design actions
    DESIGN_CREATE = "design_create"
    DESIGN_READ = "design_read"
    DESIGN_UPDATE = "design_update"
    DESIGN_DELETE = "design_delete"
    DESIGN_DOWNLOAD = "design_download"
    DESIGN_EXPORT = "design_export"
    
    # File actions
    FILE_UPLOAD = "file_upload"
    FILE_DOWNLOAD = "file_download"
    FILE_DELETE = "file_delete"
    FILE_ENCRYPT = "file_encrypt"
    FILE_DECRYPT = "file_decrypt"
    
    # Data privacy actions
    DATA_EXPORT_REQUEST = "data_export_request"
    DATA_DELETE_REQUEST = "data_delete_request"
    DATA_DELETED = "data_deleted"
    
    # Security actions
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # System actions
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"


class AuditSeverity(str, enum.Enum):
    """Audit log severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLog(Base):
    """
    Audit log entry for tracking all user actions and system events.
    
    Provides comprehensive audit trail for security, compliance, and debugging.
    Immutable records that should never be modified after creation.
    """
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Who performed the action
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    username = Column(String(255), nullable=True)  # Denormalized for retention after user deletion
    
    # What action was performed
    action = Column(Enum(AuditAction), nullable=False, index=True)
    severity = Column(Enum(AuditSeverity), default=AuditSeverity.INFO, nullable=False)
    
    # Where and when
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Details about the action
    resource_type = Column(String(100), nullable=True)  # e.g., "design", "file", "user"
    resource_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    resource_name = Column(String(255), nullable=True)
    
    # Action details and context
    description = Column(Text, nullable=True)
    details = Column(JSON, nullable=True)  # Additional structured data
    
    # Result of the action
    success = Column(String(50), default="success", nullable=False)  # success, failure, partial
    error_message = Column(Text, nullable=True)
    
    # Relationship
    user = relationship("User", foreign_keys=[user_id])
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, user={self.username}, timestamp={self.timestamp})>"
