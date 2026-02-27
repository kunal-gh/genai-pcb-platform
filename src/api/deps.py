"""
FastAPI dependencies for the GenAI PCB Design Platform.

Provides shared dependencies such as auth (get_current_user) and DB session.
Task 16.1: Auth scaffolding — replace with real JWT validation when implemented.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..services import auth_service

http_bearer = HTTPBearer(auto_error=False)


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
) -> Optional[str]:
    """
    Optional auth: returns user_id if valid token present, else None.
    """
    if not credentials:
        return None
    return auth_service.decode_access_token(credentials.credentials)


def get_current_user_required(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
) -> str:
    """
    Required auth: returns user_id or raises 401.
    Use for endpoints that must be authenticated.
    """
    user_id = get_current_user_optional(credentials)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


def get_current_user_id_for_designs(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
) -> str:
    """
    Returns current user ID for design ownership. Until auth is implemented,
    returns a placeholder so existing design CRUD keeps working.
    """
    user_id = get_current_user_optional(credentials)
    if user_id:
        return user_id
    # Placeholder until Task 16.1 JWT is implemented
    return "00000000-0000-0000-0000-000000000000"
