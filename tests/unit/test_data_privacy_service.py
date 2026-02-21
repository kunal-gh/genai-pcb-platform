"""
Unit tests for data privacy service (Task 16.2).

Tests data deletion, export, and anonymization features.
"""

import pytest
import json
from uuid import uuid4
from pathlib import Path
from datetime import datetime, timedelta

from src.models.user import User
from src.models.design import DesignProject, DesignStatus, DesignFile, FileType
from src.models.session import Session
from src.models.audit_log import AuditLog, AuditAction
from src.services.data_privacy_service import get_data_privacy_service


@pytest.fixture
def privacy_service():
    """Get data privacy service instance."""
    return get_data_privacy_service()


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password_123",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_design(db_session, test_user):
    """Create a test design."""
    design = DesignProject(
        user_id=test_user.id,
        name="Test Design",
        description="Test description",
        natural_language_prompt="Create a simple LED circuit",
        status=DesignStatus.COMPLETED,
    )
    db_session.add(design)
    db_session.commit()
    db_session.refresh(design)
    return design


class TestDataDeletion:
    """Test data deletion features."""
    
    def test_delete_design(self, db_session, privacy_service, test_user, test_design):
        """Test deleting a single design."""
        design_id = test_design.id
        
        summary = privacy_service.delete_design(
            db=db_session,
            design_id=design_id,
            user_id=test_user.id,
            username=test_user.username,
        )
        
        assert summary["design_id"] == str(design_id)
        assert summary["design_name"] == "Test Design"
        
        # Verify design is deleted
        deleted_design = db_session.query(DesignProject).filter(
            DesignProject.id == design_id
        ).first()
        assert deleted_design is None
    
    def test_delete_design_unauthorized(self, db_session, privacy_service, test_design):
        """Test that unauthorized users cannot delete designs."""
        other_user_id = uuid4()
        
        with pytest.raises(PermissionError):
            privacy_service.delete_design(
                db=db_session,
                design_id=test_design.id,
                user_id=other_user_id,
                username="otheruser",
            )
    
    def test_delete_user_data(self, db_session, privacy_service, test_user, test_design):
        """Test complete user data deletion."""
        user_id = test_user.id
        
        # Create a session
        session = Session(
            user_id=user_id,
            token_jti="test_token_123",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db_session.add(session)
        db_session.commit()
        
        summary = privacy_service.delete_user_data(
            db=db_session,
            user_id=user_id,
            requesting_user_id=user_id,
        )
        
        assert summary["designs_deleted"] == 1
        assert summary["sessions_deleted"] == 1
        
        # Verify user is deleted
        deleted_user = db_session.query(User).filter(User.id == user_id).first()
        assert deleted_user is None
        
        # Verify designs are deleted
        designs = db_session.query(DesignProject).filter(
            DesignProject.user_id == user_id
        ).all()
        assert len(designs) == 0


class TestDataExport:
    """Test data export features."""
    
    def test_export_user_data(self, db_session, privacy_service, test_user, test_design):
        """Test exporting user data."""
        export_data = privacy_service.export_user_data(
            db=db_session,
            user_id=test_user.id,
        )
        
        assert export_data["user"]["id"] == str(test_user.id)
        assert export_data["user"]["username"] == test_user.username
        assert export_data["user"]["email"] == test_user.email
        assert len(export_data["designs"]) == 1
        assert export_data["designs"][0]["name"] == "Test Design"
    
    def test_export_includes_sessions(self, db_session, privacy_service, test_user):
        """Test that export includes session data."""
        # Create a session
        session = Session(
            user_id=test_user.id,
            token_jti="test_token_456",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            ip_address="192.168.1.1",
        )
        db_session.add(session)
        db_session.commit()
        
        export_data = privacy_service.export_user_data(
            db=db_session,
            user_id=test_user.id,
        )
        
        assert len(export_data["sessions"]) == 1
        assert export_data["sessions"][0]["ip_address"] == "192.168.1.1"


class TestDataAnonymization:
    """Test data anonymization features."""
    
    def test_anonymize_user_data(self, db_session, privacy_service, test_user, test_design):
        """Test anonymizing user data while preserving designs."""
        user_id = test_user.id
        original_username = test_user.username
        
        summary = privacy_service.anonymize_user_data(
            db=db_session,
            user_id=user_id,
            requesting_user_id=user_id,
        )
        
        assert summary["original_username"] == original_username
        assert summary["designs_preserved"] == 1
        
        # Verify user is anonymized
        db_session.refresh(test_user)
        assert test_user.username == f"anonymous_{user_id}"
        assert test_user.is_active is False
        
        # Verify design still exists
        design = db_session.query(DesignProject).filter(
            DesignProject.id == test_design.id
        ).first()
        assert design is not None