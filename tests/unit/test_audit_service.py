"""
Unit tests for audit logging service (Task 16.2).

Tests audit event logging, querying, and statistics.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.models.audit_log import AuditLog, AuditAction, AuditSeverity
from src.models.user import User
from src.services import audit_service


class TestAuditLogging:
    """Test audit event logging."""
    
    def test_log_audit_event(self, db_session):
        """Test basic audit event logging."""
        user_id = uuid4()
        username = "testuser"
        
        log = audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.USER_LOGIN,
            user_id=user_id,
            username=username,
            ip_address="192.168.1.1",
            description="User logged in successfully",
        )
        
        assert log.id is not None
        assert log.user_id == user_id
        assert log.username == username
        assert log.action == AuditAction.USER_LOGIN
        assert log.severity == AuditSeverity.INFO
        assert log.ip_address == "192.168.1.1"
        assert log.success == "success"
    
    def test_log_audit_event_with_details(self, db_session):
        """Test audit logging with structured details."""
        details = {
            "design_id": str(uuid4()),
            "file_count": 5,
            "total_size": 1024000,
        }
        
        log = audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.DESIGN_CREATE,
            user_id=uuid4(),
            username="testuser",
            resource_type="design",
            resource_id=uuid4(),
            resource_name="Test Design",
            details=details,
        )
        
        assert log.details == details
        assert log.resource_type == "design"
        assert log.resource_name == "Test Design"
    
    def test_log_failed_action(self, db_session):
        """Test logging failed actions."""
        log = audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.UNAUTHORIZED_ACCESS,
            user_id=uuid4(),
            username="testuser",
            severity=AuditSeverity.WARNING,
            success="failure",
            error_message="Access denied",
        )
        
        assert log.success == "failure"
        assert log.error_message == "Access denied"
        assert log.severity == AuditSeverity.WARNING


class TestAuditQuerying:
    """Test audit log querying."""
    
    def test_get_user_audit_logs(self, db_session):
        """Test retrieving audit logs for a user."""
        user_id = uuid4()
        
        # Create multiple audit logs
        for i in range(5):
            audit_service.log_audit_event(
                db=db_session,
                action=AuditAction.DESIGN_READ,
                user_id=user_id,
                username="testuser",
                description=f"Action {i}",
            )
        
        # Query logs
        logs = audit_service.get_user_audit_logs(db_session, user_id, limit=10)
        
        assert len(logs) == 5
        assert all(log.user_id == user_id for log in logs)
    
    def test_get_user_audit_logs_with_filter(self, db_session):
        """Test filtering audit logs by action."""
        user_id = uuid4()
        
        # Create different types of logs
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.USER_LOGIN,
            user_id=user_id,
            username="testuser",
        )
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.DESIGN_CREATE,
            user_id=user_id,
            username="testuser",
        )
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.DESIGN_CREATE,
            user_id=user_id,
            username="testuser",
        )
        
        # Filter for design creation only
        logs = audit_service.get_user_audit_logs(
            db_session,
            user_id,
            action_filter=[AuditAction.DESIGN_CREATE],
        )
        
        assert len(logs) == 2
        assert all(log.action == AuditAction.DESIGN_CREATE for log in logs)
    
    def test_get_resource_audit_logs(self, db_session):
        """Test retrieving audit logs for a specific resource."""
        resource_id = uuid4()
        
        # Create logs for the resource
        for i in range(3):
            audit_service.log_audit_event(
                db=db_session,
                action=AuditAction.DESIGN_UPDATE,
                user_id=uuid4(),
                username=f"user{i}",
                resource_type="design",
                resource_id=resource_id,
            )
        
        logs = audit_service.get_resource_audit_logs(
            db_session,
            resource_type="design",
            resource_id=resource_id,
        )
        
        assert len(logs) == 3
        assert all(log.resource_id == resource_id for log in logs)
    
    def test_get_security_events(self, db_session):
        """Test retrieving security-related events."""
        # Create security events
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.UNAUTHORIZED_ACCESS,
            severity=AuditSeverity.WARNING,
            username="attacker",
        )
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.PERMISSION_DENIED,
            severity=AuditSeverity.WARNING,
            username="testuser",
        )
        
        # Create non-security event
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.DESIGN_READ,
            username="testuser",
        )
        
        security_events = audit_service.get_security_events(db_session, hours=24)
        
        assert len(security_events) == 2
        assert all(
            log.action in [
                AuditAction.UNAUTHORIZED_ACCESS,
                AuditAction.PERMISSION_DENIED,
            ]
            for log in security_events
        )
    
    def test_get_failed_actions(self, db_session):
        """Test retrieving failed actions."""
        # Create successful and failed actions
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.DESIGN_CREATE,
            username="testuser",
            success="success",
        )
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.DESIGN_CREATE,
            username="testuser",
            success="failure",
            error_message="Validation failed",
        )
        
        failed = audit_service.get_failed_actions(db_session, hours=24)
        
        assert len(failed) == 1
        assert failed[0].success == "failure"


class TestAuditStatistics:
    """Test audit statistics."""
    
    def test_get_audit_statistics(self, db_session):
        """Test audit statistics calculation."""
        user_id = uuid4()
        
        # Create various audit events
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.USER_LOGIN,
            user_id=user_id,
            username="testuser",
            severity=AuditSeverity.INFO,
        )
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.DESIGN_CREATE,
            user_id=user_id,
            username="testuser",
            severity=AuditSeverity.INFO,
            success="success",
        )
        audit_service.log_audit_event(
            db=db_session,
            action=AuditAction.UNAUTHORIZED_ACCESS,
            user_id=user_id,
            username="testuser",
            severity=AuditSeverity.WARNING,
            success="failure",
        )
        
        stats = audit_service.get_audit_statistics(db_session, user_id=user_id, hours=24)
        
        assert stats["total_events"] == 3
        assert stats["success_count"] == 2
        assert stats["failure_count"] == 1
        assert stats["security_events"] == 1
        assert stats["severity_counts"]["info"] == 2
        assert stats["severity_counts"]["warning"] == 1


class TestConvenienceFunctions:
    """Test convenience functions for common audit events."""
    
    def test_log_user_login(self, db_session):
        """Test user login logging."""
        user_id = uuid4()
        
        log = audit_service.log_user_login(
            db=db_session,
            user_id=user_id,
            username="testuser",
            ip_address="192.168.1.1",
            success=True,
        )
        
        assert log.action == AuditAction.USER_LOGIN
        assert log.success == "success"
    
    def test_log_user_logout(self, db_session):
        """Test user logout logging."""
        user_id = uuid4()
        
        log = audit_service.log_user_logout(
            db=db_session,
            user_id=user_id,
            username="testuser",
        )
        
        assert log.action == AuditAction.USER_LOGOUT
    
    def test_log_design_access(self, db_session):
        """Test design access logging."""
        user_id = uuid4()
        design_id = uuid4()
        
        log = audit_service.log_design_access(
            db=db_session,
            action=AuditAction.DESIGN_CREATE,
            user_id=user_id,
            username="testuser",
            design_id=design_id,
            design_name="Test Design",
        )
        
        assert log.action == AuditAction.DESIGN_CREATE
        assert log.resource_type == "design"
        assert log.resource_id == design_id
    
    def test_log_unauthorized_access(self, db_session):
        """Test unauthorized access logging."""
        resource_id = uuid4()
        
        log = audit_service.log_unauthorized_access(
            db=db_session,
            user_id=uuid4(),
            username="testuser",
            ip_address="192.168.1.1",
            resource_type="design",
            resource_id=resource_id,
        )
        
        assert log.action == AuditAction.UNAUTHORIZED_ACCESS
        assert log.severity == AuditSeverity.WARNING
        assert log.success == "failure"


class TestAuditCleanup:
    """Test audit log cleanup."""
    
    def test_cleanup_old_audit_logs(self, db_session):
        """Test cleanup of old audit logs."""
        # Create old log (manually set timestamp)
        old_log = AuditLog(
            action=AuditAction.USER_LOGIN,
            username="testuser",
            timestamp=datetime.utcnow() - timedelta(days=400),
        )
        db_session.add(old_log)
        
        # Create recent log
        recent_log = AuditLog(
            action=AuditAction.USER_LOGIN,
            username="testuser",
            timestamp=datetime.utcnow(),
        )
        db_session.add(recent_log)
        db_session.commit()
        
        # Cleanup logs older than 365 days
        deleted = audit_service.cleanup_old_audit_logs(db_session, retention_days=365)
        
        assert deleted == 1
        
        # Verify recent log still exists
        remaining = db_session.query(AuditLog).all()
        assert len(remaining) == 1
        assert remaining[0].id == recent_log.id
