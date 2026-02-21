"""
Authentication service (Task 16.1).

Password hashing, user CRUD, and JWT create/verify for registration and login.
Includes session management and security controls.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import bcrypt
import jwt
from sqlalchemy.orm import Session

from ..config import settings
from ..models.user import User
from . import session_service

logger = logging.getLogger(__name__)

JWT_SUB_CLAIM = "sub"  # user_id (str)
JWT_JTI_CLAIM = "jti"  # JWT ID for session tracking
JWT_EXP_CLAIM = "exp"
JWT_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """Hash a plain password for storage."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a stored hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def create_user(
    db: Session,
    username: str,
    email: str,
    password: str,
) -> User:
    """
    Create a new user. Raises ValueError if username or email already exists.
    """
    if get_user_by_username(db, username) is not None:
        raise ValueError("Username already registered")
    if get_user_by_email(db, email) is not None:
        raise ValueError("Email already registered")
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Created user: %s", username)
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Return user by username or None."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Return user by email or None."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: UUID) -> Optional[User]:
    """Return user by id or None."""
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(
    db: Session,
    username: str,
    password: str,
) -> Optional[User]:
    """
    Authenticate by username and password. Returns User if valid, else None.
    """
    user = get_user_by_username(db, username)
    if user is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user


def create_access_token(
    user_id: str,
    db: Optional[Session] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> tuple[str, int, str]:
    """
    Create a JWT access token for the given user_id with session tracking.
    
    Args:
        user_id: User ID to encode in token
        db: Database session for session tracking (optional)
        ip_address: Client IP address for session tracking
        user_agent: Client user agent for session tracking
    
    Returns:
        Tuple of (token_string, expires_in_seconds, jti)
    """
    # Generate unique JWT ID for session tracking
    jti = session_service.generate_jti()
    
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        JWT_SUB_CLAIM: str(user_id),
        JWT_JTI_CLAIM: jti,
        JWT_EXP_CLAIM: expire
    }
    token = jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    
    # Create session record if database provided
    if db is not None:
        try:
            session_service.create_session(
                db,
                UUID(user_id),
                jti,
                ip_address=ip_address,
                user_agent=user_agent,
            )
        except Exception as e:
            logger.error(f"Failed to create session record: {e}")
            # Continue anyway - session tracking is not critical for token creation
    
    return token, settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60, jti


def decode_access_token(token: str, db: Optional[Session] = None) -> Optional[str]:
    """
    Decode and validate JWT; return user_id (sub) or None if invalid/expired/revoked.
    
    Args:
        token: JWT token string
        db: Database session for session validation (optional)
    
    Returns:
        User ID string or None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
        sub = payload.get(JWT_SUB_CLAIM)
        jti = payload.get(JWT_JTI_CLAIM)
        
        # If database provided, validate session is not revoked
        if db is not None and jti is not None:
            if not session_service.validate_session(db, jti):
                logger.warning(f"Token with jti {jti} has revoked or expired session")
                return None
        
        return str(sub) if sub is not None else None
    except jwt.PyJWTError:
        return None
