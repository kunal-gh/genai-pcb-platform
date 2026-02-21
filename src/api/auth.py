"""
Authentication API (Task 16.1).

Endpoints for login (JWT), register, logout, and session management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List

from ..models.database import get_db
from ..services import auth_service, session_service
from .deps import get_current_user_required

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool

    class Config:
        from_attributes = True


class SessionInfo(BaseModel):
    id: str
    created_at: str
    last_activity: str
    expires_at: str
    ip_address: str | None
    device_info: str | None
    is_active: bool


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]
    total: int
    active: int


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and return a JWT with session tracking."""
    user = auth_service.authenticate_user(db, request.username, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    
    # Create token with session tracking
    # In production, extract real IP and user agent from request headers
    token, expires_in, jti = auth_service.create_access_token(
        str(user.id),
        db=db,
        ip_address=None,  # TODO: Extract from request
        user_agent=None,  # TODO: Extract from request
    )
    
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        user = auth_service.create_user(
            db, request.username, request.email, request.password
        )
        return UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            is_active=user.is_active,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/me", response_model=UserResponse)
async def me(
    user_id: str = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Return current user from JWT."""
    from uuid import UUID
    user = auth_service.get_user_by_id(db, UUID(user_id))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        is_active=user.is_active,
    )


@router.post("/logout")
async def logout(
    user_id: str = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """
    Logout current user by revoking the current session.
    
    Note: This requires extracting the JWT ID from the token.
    For now, this is a placeholder that revokes all user sessions.
    """
    from uuid import UUID
    count = session_service.revoke_all_user_sessions(
        db,
        UUID(user_id),
        reason="User logout"
    )
    return {"message": f"Logged out successfully. {count} session(s) revoked."}


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user_id: str = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """List all active sessions for the current user."""
    from uuid import UUID
    sessions = session_service.get_user_sessions(db, UUID(user_id), active_only=True)
    
    session_infos = [
        SessionInfo(
            id=str(s.id),
            created_at=s.created_at.isoformat(),
            last_activity=s.last_activity.isoformat(),
            expires_at=s.expires_at.isoformat(),
            ip_address=s.ip_address,
            device_info=s.device_info,
            is_active=s.is_active,
        )
        for s in sessions
    ]
    
    stats = session_service.get_session_stats(db, UUID(user_id))
    
    return SessionListResponse(
        sessions=session_infos,
        total=stats["total_sessions"],
        active=stats["active_sessions"],
    )


@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    user_id: str = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Revoke a specific session."""
    from uuid import UUID
    
    # Get the session and verify it belongs to the current user
    session = db.query(session_service.Session).filter(
        session_service.Session.id == UUID(session_id)
    ).first()
    
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if str(session.user_id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot revoke another user's session"
        )
    
    session.revoke("Revoked by user")
    db.commit()
    
    return {"message": "Session revoked successfully"}


@router.post("/sessions/revoke-all")
async def revoke_all_sessions(
    user_id: str = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Revoke all sessions for the current user (useful for security)."""
    from uuid import UUID
    count = session_service.revoke_all_user_sessions(
        db,
        UUID(user_id),
        reason="Revoke all sessions by user"
    )
    return {"message": f"All sessions revoked. {count} session(s) affected."}
