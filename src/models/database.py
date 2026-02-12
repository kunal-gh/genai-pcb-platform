"""
Database configuration and session management.

Provides SQLAlchemy engine, session factory, and base model class.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging

from ..config import settings

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db():
    """
    Dependency for FastAPI routes to get database session.
    
    Yields:
        Session: SQLAlchemy database session
        
    Example:
        @app.get("/designs")
        def get_designs(db: Session = Depends(get_db)):
            return db.query(Design).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions outside of FastAPI.
    
    Example:
        with get_db_context() as db:
            design = db.query(Design).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}", exc_info=True)
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database by creating all tables.
    
    Should be called on application startup.
    """
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def drop_db():
    """
    Drop all database tables.
    
    WARNING: This will delete all data! Use only for testing.
    """
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables dropped")