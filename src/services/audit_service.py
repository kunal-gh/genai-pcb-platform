"""
Audit logging service (Task 16.2).

Provides comprehensive audit logging for all user actions and system events
to support security monitoring, compliance, and debugging.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import and_, or_

from ..models.audit_log import AuditLog, AuditAction, AuditSeverity
from ..models.user import User

logger = logging.getLogger(__name__)


def log_audit_event(
    db: DBSession,
    action: AuditAction,
    user_id: Optional[UUID] = None,
    username: Optional[str] = None,
    severity: AuditSeverity = AuditSeverity.INFO,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[UUID] = None,
    resource_name: Optional[str] = None,
    description: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    success: str = "success",
    error_message: Optional[str] = None,
) -> AuditLog:
    """
    Log an audit event.
    
    Args:
        db: Database session
        action: Type of action performed
        user_id: ID of user who performed the action
        username: Username (denormalized for retention)
        severity: Severity level of the event
        ip_address: Client IP address
        user_agent: Client user agent
        resource_type: Type of resource affected (e.g., "design", "file")
        resource_id: ID of affected resource
        resource_name: Name of affected resource
        description: Human-readable description
        details: Additional structured data
        success: Result status ("success", "failure", "partial")
        error_message: Error message if action failed
        
    Returns:
        Created AuditLog entry
    """
    # If user_id provided but no username, fetch it
    if user_id and not username:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            username = user.username
    
    audit_entry = AuditLog(
        user_id=user_id,
        username=username,
        action=action,
        severity=severity,
        ip_address=ip_address,
        user_agent=user_agent,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_name=resource_name,
        description=description,
        details=details,
        success=success,
        error_message=error_message,
    )
    
    db.add(audit_entry)
    db.commit()
    db.refresh(audit_entry)
    
    # Also log to application logger for real-time monitoring
    log_level = {
        AuditSeverity.INFO: logging.INFO,
        AuditSeverity.WARNING: logging.WARNING,
        AuditSeverity.ERROR: logging.ERROR,
        AuditSeverity.CRITICAL: logging.CRITICAL,
    }.get(severity, logging.INFO)
    
    logger.log(
        log_level,
        f"AUDIT: {action.value} by {username or 'system'} - {description or 'No description'}"
    )
    
    return audit_entry


def get_user_audit_logs(
    db: DBSession,
    user_id: UUID,
    limit: int = 100,
    offset: int = 0,
    action_filter: Optional[List[AuditAction]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[AuditLog]:
    """
    Get audit logs for a specific user.
    
    Args:
        db: Database session
        user_id: User ID to filter by
        limit: Maximum number of records to return
        offset: Number of records to skip
        action_filter: List of actions to filter by
        start_date: Filter logs after this date
        end_date: Filter logs before this date
        
    Returns:
        List of AuditLog entries
    """
    query = db.query(AuditLog).filter(AuditLog.user_id == user_id)
    
    if action_filter:
        query = query.filter(AuditLog.action.in_(action_filter))
    
    if start_date:
        query = query.filter(AuditLog.timestamp >= start_date)
    
    if end_date:
        query = query.filter(AuditLog.timestamp <= end_date)
    
    return query.order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset).all()


def get_resource_audit_logs(
    db: DBSession,
    resource_type: str,
    resource_id: UUID,
    limit: int = 100,
) -> List[AuditLog]:
    """
    Get audit logs for a specific resource.
    
    Args:
        db: Database session
        resource_type: Type of resource (e.g., "design", "file")
        resource_id: ID of the resource
        limit: Maximum number of records to return
        
    Returns:
        List of AuditLog entries
    """
    return (
        db.query(AuditLog)
        .filter(
            and_(
                AuditLog.resource_type == resource_type,
                AuditLog.resource_id == resource_id
            )
        )
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )


def get_security_events(
    db: DBSession,
    severity: Optional[AuditSeverity] = None,
    limit: int = 100,
    hours: int = 24,
) -> List[AuditLog]:
    """
    Get recent security-related events.
    
    Args:
        db: Database session
        severity: Filter by severity level
        limit: Maximum number of records to return
        hours: Look back this many hours
        
    Returns:
        List of AuditLog entries
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    security_actions = [
        AuditAction.UNAUTHORIZED_ACCESS,
        AuditAction.PERMISSION_DENIED,
        AuditAction.SUSPICIOUS_ACTIVITY,
        AuditAction.SESSION_REVOKED,
    ]
    
    query = db.query(AuditLog).filter(
        and_(
            AuditLog.action.in_(security_actions),
            AuditLog.timestamp >= cutoff
        )
    )
    
    if severity:
        query = query.filter(AuditLog.severity == severity)
    
    return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()


def get_failed_actions(
    db: DBSession,
    limit: int = 100,
    hours: int = 24,
) -> List[AuditLog]:
    """
    Get recent failed actions for monitoring.
    
    Args:
        db: Database session
        limit: Maximum number of records to return
        hours: Look back this many hours
        
    Returns:
        List of AuditLog entries
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    return (
        db.query(AuditLog)
        .filter(
            and_(
                AuditLog.success == "failure",
                AuditLog.timestamp >= cutoff
            )
        )
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )


def cleanup_old_audit_logs(
    db: DBSession,
    retention_days: int = 365,
) -> int:
    """
    Clean up audit logs older than retention period.
    
    This should be run periodically to manage database size while
    maintaining compliance with data retention policies.
    
    Args:
        db: Database session
        retention_days: Keep logs for this many days
        
    Returns:
        Number of logs deleted
    """
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    
    count = db.query(AuditLog).filter(
        AuditLog.timestamp < cutoff
    ).delete()
    
    db.commit()
    
    logger.info(f"Cleaned up {count} audit logs older than {retention_days} days")
    return count


def get_audit_statistics(
    db: DBSession,
    user_id: Optional[UUID] = None,
    hours: int = 24,
) -> Dict[str, Any]:
    """
    Get audit log statistics.
    
    Args:
        db: Database session
        user_id: Optional user ID to filter by
        hours: Look back this many hours
        
    Returns:
        Dictionary with statistics
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    query = db.query(AuditLog).filter(AuditLog.timestamp >= cutoff)
    
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    
    total = query.count()
    
    # Count by severity
    severity_counts = {}
    for severity in AuditSeverity:
        count = query.filter(AuditLog.severity == severity).count()
        severity_counts[severity.value] = count
    
    # Count by success status
    success_count = query.filter(AuditLog.success == "success").count()
    failure_count = query.filter(AuditLog.success == "failure").count()
    
    # Count security events
    security_actions = [
        AuditAction.UNAUTHORIZED_ACCESS,
        AuditAction.PERMISSION_DENIED,
        AuditAction.SUSPICIOUS_ACTIVITY,
    ]
    security_count = query.filter(AuditLog.action.in_(security_actions)).count()
    
    return {
        "total_events": total,
        "time_period_hours": hours,
        "severity_counts": severity_counts,
        "success_count": success_count,
        "failure_count": failure_count,
        "security_events": security_count,
    }


# Convenience functions for common audit events

def log_user_login(
    db: DBSession,
    user_id: UUID,
    username: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    success: bool = True,
) -> AuditLog:
    """Log user login event."""
    return log_audit_event(
        db=db,
        action=AuditAction.USER_LOGIN,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        user_agent=user_agent,
        description=f"User {username} logged in",
        success="success" if success else "failure",
    )


def log_user_logout(
    db: DBSession,
    user_id: UUID,
    username: str,
    ip_address: Optional[str] = None,
) -> AuditLog:
    """Log user logout event."""
    return log_audit_event(
        db=db,
        action=AuditAction.USER_LOGOUT,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        description=f"User {username} logged out",
    )


def log_design_access(
    db: DBSession,
    action: AuditAction,
    user_id: UUID,
    username: str,
    design_id: UUID,
    design_name: str,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    """Log design access event."""
    action_descriptions = {
        AuditAction.DESIGN_CREATE: "created",
        AuditAction.DESIGN_READ: "viewed",
        AuditAction.DESIGN_UPDATE: "updated",
        AuditAction.DESIGN_DELETE: "deleted",
        AuditAction.DESIGN_DOWNLOAD: "downloaded",
    }
    
    return log_audit_event(
        db=db,
        action=action,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        resource_type="design",
        resource_id=design_id,
        resource_name=design_name,
        description=f"User {username} {action_descriptions.get(action, 'accessed')} design '{design_name}'",
        details=details,
    )


def log_data_deletion(
    db: DBSession,
    user_id: UUID,
    username: str,
    resource_type: str,
    resource_id: UUID,
    resource_name: str,
    details: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    """Log data deletion event."""
    return log_audit_event(
        db=db,
        action=AuditAction.DATA_DELETED,
        user_id=user_id,
        username=username,
        severity=AuditSeverity.WARNING,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_name=resource_name,
        description=f"User {username} permanently deleted {resource_type} '{resource_name}'",
        details=details,
    )


def log_unauthorized_access(
    db: DBSession,
    user_id: Optional[UUID],
    username: Optional[str],
    ip_address: Optional[str],
    resource_type: str,
    resource_id: UUID,
    details: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    """Log unauthorized access attempt."""
    return log_audit_event(
        db=db,
        action=AuditAction.UNAUTHORIZED_ACCESS,
        user_id=user_id,
        username=username,
        severity=AuditSeverity.WARNING,
        ip_address=ip_address,
        resource_type=resource_type,
        resource_id=resource_id,
        description=f"Unauthorized access attempt to {resource_type} by {username or 'unknown user'}",
        details=details,
        success="failure",
    )
