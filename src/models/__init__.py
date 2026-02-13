"""
Data models for the GenAI PCB Design Platform.

This package contains SQLAlchemy models, Pydantic schemas, and data structures
for the stuff-made-easy platform.
"""

from .database import Base, get_db, get_db_context, init_db, drop_db
from .design import (
    DesignProject,
    DesignFile,
    VerificationResult,
    SimulationResult,
    DesignStatus,
    FileType
)
from .component import (
    Component,
    Manufacturer,
    ComponentCategory,
    PackageType
)

__all__ = [
    # Database
    "Base",
    "get_db",
    "get_db_context",
    "init_db",
    "drop_db",
    # Design models
    "DesignProject",
    "DesignFile",
    "VerificationResult",
    "SimulationResult",
    "DesignStatus",
    "FileType",
    # Component models
    "Component",
    "Manufacturer",
    "ComponentCategory",
    "PackageType",
]