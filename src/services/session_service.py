"""
Session management service (Task 16.1).

Manages user sessions with security controls including session tracking,
revocation, and cleanup of expired sessions.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID
import secrets

from sqlalchemy.orm import Session as DBSession

from ..models.session import Session
from ..models.user import User
from ..config import settings

logger = logging.getLogger(__name__)


def generate_jti() -> str:
    """
    Generate a unique JWT ID (jti) for session tracking.
    
    Returns:
        Random 32-character hex string
    """
    return secrets.token_hex(16)


def create_session(
    db: DBSession,
    user_id: UUID,
    token_jti: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    device_info: Optional[str] = None,
) -> Session:
    """
    Create a new user session.
    
    Args:
        db: Database session
        user_id: User ID
        token_jti: JWT ID for this session
        ip_address: Client IP address
        user_agent: Client user agent string
        device_info: Device information
        
    Returns:
        Created Session object
    """
    expires_at = datetime.utcnow() + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    
    session = Session(
        user_id=user_id,
        token_jti=token_jti,
        ip_address=ip_address,
        user_agent=user_agent,
        device_info=device_info,
        expires_at=expires_at,
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    logger.info(f"Created session {session.id} for user {user_id}")
    return session


def get_session_by_jti(db: DBSession, token_jti: str) -> Optional[Session]:
    """
    Get session by JWT ID.
    
    Args:
        db: Database session
        token_jti: JWT ID
        
    Returns:
        Session object or None
    """
    return db.query(Session).filter(Session.token_jti == token_jti).first()


def validate_session(db: DBSession, token_jti: str) -> bool:
    """
    Validate if a session is active and not expired.
    
    Args:
        db: Database session
        token_jti: JWT ID
        
    Returns:
        True if session is valid, False otherwise
    """
    session = get_session_by_jti(db, token_jti)
    if session is None:
        return False
    
    if not session.is_valid():
        return False
    
    # Update last activity
    session.update_activity()
    db.commit()
    
    return True


def revoke_session(
    db: DBSession,
    token_jti: str,
    reason: str = "User logout"
) -> bool:
    """
    Revoke a session.
    
    Args:
        db: Database session
        token_jti: JWT ID
        reason: Reason for revocation
        
    Returns:
        True if session was revoked, False if not found
    """
    session = get_session_by_jti(db, token_jti)
    if session is None:
        return False
    
    session.revoke(reason)
    db.commit()
    
    logger.info(f"Revoked session {session.id}: {reason}")
    return True


def revoke_all_user_sessions(
    db: DBSession,
    user_id: UUID,
    reason: str = "Revoke all sessions"
) -> int:
    """
    Revoke all active sessions for a user.
    
    Args:
        db: Database session
        user_id: User ID
        reason: Reason for revocation
        
    Returns:
        Number of sessions revoked
    """
    sessions = db.query(Session).filter(
        Session.user_id == user_id,
        Session.is_active == True
    ).all()
    
    count = 0
    for session in sessions:
        session.revoke(reason)
        count += 1
    
    db.commit()
    
    logger.info(f"Revoked {count} sessions for user {user_id}: {reason}")
    return count


def get_user_sessions(
    db: DBSession,
    user_id: UUID,
    active_only: bool = True
) -> List[Session]:
    """
    Get all sessions for a user.
    
    Args:
        db: Database session
        user_id: User ID
        active_only: If True, only return active sessions
        
    Returns:
        List of Session objects
    """
    query = db.query(Session).filter(Session.user_id == user_id)
    
    if active_only:
        query = query.filter(Session.is_active == True)
    
    return query.order_by(Session.created_at.desc()).all()


def cleanup_expired_sessions(db: DBSession) -> int:
    """
    Clean up expired sessions from the database.
    
    This should be run periodically (e.g., daily cron job) to remove
    expired sessions and keep the database clean.
    
    Args:
        db: Database session
        
    Returns:
        Number of sessions deleted
    """
    now = datetime.utcnow()
    
    # Delete sessions that expired more than 7 days ago
    cutoff = now - timedelta(days=7)
    
    count = db.query(Session).filter(
        Session.expires_at < cutoff
    ).delete()
    
    db.commit()
    
    logger.info(f"Cleaned up {count} expired sessions")
    return count


def get_session_stats(db: DBSession, user_id: UUID) -> dict:
    """
    Get session statistics for a user.
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        Dictionary with session statistics
    """
    total = db.query(Session).filter(Session.user_id == user_id).count()
    active = db.query(Session).filter(
        Session.user_id == user_id,
        Session.is_active == True
    ).count()
    
    return {
        "total_sessions": total,
        "active_sessions": active,
        "revoked_sessions": total - active,
    }
