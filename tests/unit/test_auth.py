"""
Unit tests for authentication (Task 16.1).

Tests auth_service (password hashing, JWT) and auth API (register, login, me).
"""

import pytest
from uuid import uuid4

from src.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    create_user,
    get_user_by_username,
    authenticate_user,
)
from src.models.user import User
from src.models.database import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# In-memory SQLite for auth tests
SQLITE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
# Use plain String for id in SQLite-compatible way - we only test auth_service logic
# So we create tables from Base; User has UUID column which SQLite may map to blob/string
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


class TestPasswordHashing:
    def test_hash_password_returns_string(self):
        h = hash_password("secret123")
        assert isinstance(h, str)
        assert len(h) > 20

    def test_verify_password_correct(self):
        h = hash_password("secret123")
        assert verify_password("secret123", h) is True

    def test_verify_password_wrong(self):
        h = hash_password("secret123")
        assert verify_password("wrong", h) is False


class TestJWT:
    def test_create_and_decode_token(self):
        user_id = str(uuid4())
        token, expires_in, jti = create_access_token(user_id)
        assert isinstance(token, str)
        assert isinstance(jti, str)
        assert expires_in > 0
        assert decode_access_token(token) == user_id

    def test_decode_invalid_token_returns_none(self):
        assert decode_access_token("invalid") is None
        assert decode_access_token("") is None


class TestAuthServiceWithDB:
    def test_create_user_and_get_by_username(self, db):
        user = create_user(db, "alice", "alice@example.com", "password123")
        assert user.username == "alice"
        assert user.email == "alice@example.com"
        assert user.hashed_password != "password123"
        found = get_user_by_username(db, "alice")
        assert found is not None
        assert found.id == user.id

    def test_create_user_duplicate_username_raises(self, db):
        create_user(db, "bob", "bob1@example.com", "pass")
        with pytest.raises(ValueError, match="Username already registered"):
            create_user(db, "bob", "bob2@example.com", "pass")

    def test_create_user_duplicate_email_raises(self, db):
        create_user(db, "c1", "same@example.com", "pass")
        with pytest.raises(ValueError, match="Email already registered"):
            create_user(db, "c2", "same@example.com", "pass")

    def test_authenticate_user_success(self, db):
        create_user(db, "u1", "u1@example.com", "mypass")
        user = authenticate_user(db, "u1", "mypass")
        assert user is not None
        assert user.username == "u1"

    def test_authenticate_user_wrong_password(self, db):
        create_user(db, "u2", "u2@example.com", "mypass")
        assert authenticate_user(db, "u2", "wrong") is None

    def test_authenticate_user_unknown_username(self, db):
        assert authenticate_user(db, "nonexistent", "any") is None
