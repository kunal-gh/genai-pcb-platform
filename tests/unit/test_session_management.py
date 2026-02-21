"""
Unit tests for session management (Task 16.1).

Tests session creation, validation, revocation, and cleanup.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.services import session_service
from src.models.user import User
from src.models.session import Session
from src.models.database import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# In-memory SQLite for session tests
SQLITE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(db):
    """Create a test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class TestSessionCreation:
    """Test session creation and retrieval."""
    
    def test_generate_jti(self):
        """Test JWT ID generation."""
        jti1 = session_service.generate_jti()
        jti2 = session_service.generate_jti()
        
        assert isinstance(jti1, str)
        assert len(jti1) == 32  # 16 bytes = 32 hex chars
        assert jti1 != jti2  # Should be unique
    
    def test_create_session(self, db, test_user):
        """Test creating a new session."""
        jti = session_service.generate_jti()
        
        session = session_service.create_session(
            db,
            test_user.id,
            jti,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            device_info="Desktop",
        )
        
        assert session.id is not None
        assert session.user_id == test_user.id
        assert session.token_jti == jti
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Mozilla/5.0"
        assert session.device_info == "Desktop"
        assert session.is_active is True
        assert session.expires_at > datetime.utcnow()
    
    def test_get_session_by_jti(self, db, test_user):
        """Test retrieving session by JWT ID."""
        jti = session_service.generate_jti()
        created = session_service.create_session(db, test_user.id, jti)
        
        retrieved = session_service.get_session_by_jti(db, jti)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.token_jti == jti
    
    def test_get_nonexistent_session_returns_none(self, db):
        """Test that getting nonexistent session returns None."""
        result = session_service.get_session_by_jti(db, "nonexistent-jti")
        assert result is None


class TestSessionValidation:
    """Test session validation logic."""
    
    def test_validate_active_session(self, db, test_user):
        """Test validating an active session."""
        jti = session_service.generate_jti()
        session_service.create_session(db, test_user.id, jti)
        
        is_valid = session_service.validate_session(db, jti)
        assert is_valid is True
    
    def test_validate_nonexistent_session(self, db):
        """Test validating nonexistent session returns False."""
        is_valid = session_service.validate_session(db, "nonexistent")
        assert is_valid is False
    
    def test_validate_revoked_session(self, db, test_user):
        """Test that revoked session is invalid."""
        jti = session_service.generate_jti()
        session_service.create_session(db, test_user.id, jti)
        session_service.revoke_session(db, jti, "Test revocation")
        
        is_valid = session_service.validate_session(db, jti)
        assert is_valid is False
    
    def test_validate_expired_session(self, db, test_user):
        """Test that expired session is invalid."""
        jti = session_service.generate_jti()
        session = session_service.create_session(db, test_user.id, jti)
        
        # Manually expire the session
        session.expires_at = datetime.utcnow() - timedelta(hours=1)
        db.commit()
        
        is_valid = session_service.validate_session(db, jti)
        assert is_valid is False
    
    def test_validate_updates_last_activity(self, db, test_user):
        """Test that validation updates last activity timestamp."""
        jti = session_service.generate_jti()
        session = session_service.create_session(db, test_user.id, jti)
        
        original_activity = session.last_activity
        
        # Wait a tiny bit and validate
        import time
        time.sleep(0.01)
        
        session_service.validate_session(db, jti)
        db.refresh(session)
        
        assert session.last_activity > original_activity


class TestSessionRevocation:
    """Test session revocation."""
    
    def test_revoke_session(self, db, test_user):
        """Test revoking a session."""
        jti = session_service.generate_jti()
        session_service.create_session(db, test_user.id, jti)
        
        result = session_service.revoke_session(db, jti, "User logout")
        assert result is True
        
        session = session_service.get_session_by_jti(db, jti)
        assert session.is_active is False
        assert session.revoked_at is not None
        assert session.revocation_reason == "User logout"
    
    def test_revoke_nonexistent_session(self, db):
        """Test revoking nonexistent session returns False."""
        result = session_service.revoke_session(db, "nonexistent", "Test")
        assert result is False
    
    def test_revoke_all_user_sessions(self, db, test_user):
        """Test revoking all sessions for a user."""
        # Create multiple sessions
        jti1 = session_service.generate_jti()
        jti2 = session_service.generate_jti()
        jti3 = session_service.generate_jti()
        
        session_service.create_session(db, test_user.id, jti1)
        session_service.create_session(db, test_user.id, jti2)
        session_service.create_session(db, test_user.id, jti3)
        
        count = session_service.revoke_all_user_sessions(
            db, test_user.id, "Security measure"
        )
        
        assert count == 3
        
        # Verify all are revoked
        assert session_service.validate_session(db, jti1) is False
        assert session_service.validate_session(db, jti2) is False
        assert session_service.validate_session(db, jti3) is False


class TestSessionQueries:
    """Test session query functions."""
    
    def test_get_user_sessions_active_only(self, db, test_user):
        """Test getting only active sessions."""
        jti1 = session_service.generate_jti()
        jti2 = session_service.generate_jti()
        
        session_service.create_session(db, test_user.id, jti1)
        session_service.create_session(db, test_user.id, jti2)
        session_service.revoke_session(db, jti2, "Test")
        
        sessions = session_service.get_user_sessions(db, test_user.id, active_only=True)
        
        assert len(sessions) == 1
        assert sessions[0].token_jti == jti1
    
    def test_get_user_sessions_all(self, db, test_user):
        """Test getting all sessions including revoked."""
        jti1 = session_service.generate_jti()
        jti2 = session_service.generate_jti()
        
        session_service.create_session(db, test_user.id, jti1)
        session_service.create_session(db, test_user.id, jti2)
        session_service.revoke_session(db, jti2, "Test")
        
        sessions = session_service.get_user_sessions(db, test_user.id, active_only=False)
        
        assert len(sessions) == 2
    
    def test_get_session_stats(self, db, test_user):
        """Test getting session statistics."""
        jti1 = session_service.generate_jti()
        jti2 = session_service.generate_jti()
        jti3 = session_service.generate_jti()
        
        session_service.create_session(db, test_user.id, jti1)
        session_service.create_session(db, test_user.id, jti2)
        session_service.create_session(db, test_user.id, jti3)
        session_service.revoke_session(db, jti3, "Test")
        
        stats = session_service.get_session_stats(db, test_user.id)
        
        assert stats["total_sessions"] == 3
        assert stats["active_sessions"] == 2
        assert stats["revoked_sessions"] == 1


class TestSessionCleanup:
    """Test session cleanup functionality."""
    
    def test_cleanup_expired_sessions(self, db, test_user):
        """Test cleaning up old expired sessions."""
        jti1 = session_service.generate_jti()
        jti2 = session_service.generate_jti()
        
        # Create sessions
        session1 = session_service.create_session(db, test_user.id, jti1)
        session2 = session_service.create_session(db, test_user.id, jti2)
        
        # Make session1 expired 8 days ago (should be deleted)
        session1.expires_at = datetime.utcnow() - timedelta(days=8)
        
        # Make session2 expired 5 days ago (should be kept)
        session2.expires_at = datetime.utcnow() - timedelta(days=5)
        
        db.commit()
        
        count = session_service.cleanup_expired_sessions(db)
        
        assert count == 1
        
        # Verify session1 is deleted
        assert session_service.get_session_by_jti(db, jti1) is None
        
        # Verify session2 still exists
        assert session_service.get_session_by_jti(db, jti2) is not None


class TestSessionModel:
    """Test Session model methods."""
    
    def test_is_expired(self, db, test_user):
        """Test is_expired method."""
        jti = session_service.generate_jti()
        session = session_service.create_session(db, test_user.id, jti)
        
        assert session.is_expired() is False
        
        # Expire the session
        session.expires_at = datetime.utcnow() - timedelta(hours=1)
        assert session.is_expired() is True
    
    def test_is_valid(self, db, test_user):
        """Test is_valid method."""
        jti = session_service.generate_jti()
        session = session_service.create_session(db, test_user.id, jti)
        
        assert session.is_valid() is True
        
        # Revoke the session
        session.revoke("Test")
        assert session.is_valid() is False
    
    def test_revoke_method(self, db, test_user):
        """Test revoke method on Session model."""
        jti = session_service.generate_jti()
        session = session_service.create_session(db, test_user.id, jti)
        
        session.revoke("Manual revocation")
        
        assert session.is_active is False
        assert session.revoked_at is not None
        assert session.revocation_reason == "Manual revocation"
    
    def test_update_activity(self, db, test_user):
        """Test update_activity method."""
        jti = session_service.generate_jti()
        session = session_service.create_session(db, test_user.id, jti)
        
        original = session.last_activity
        
        import time
        time.sleep(0.01)
        
        session.update_activity()
        
        assert session.last_activity > original
