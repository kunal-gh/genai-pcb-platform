"""
Unit tests for data privacy service (Task 16.2).

Tests data deletion, export, and anonymization features.
"""

import pytest
import json
from uuid import uuid4
from pathlib import Path

from src.models.user import User
from src.models.design import DesignProject, DesignStatus, DesignFile, FileType
from src.models.session import Session
from src.models.audit_log import AuditLog, AuditAction
from src.services.data_privacy_service import get_data_privacy_service


@pytest.fixture
def privacy_service():
    "