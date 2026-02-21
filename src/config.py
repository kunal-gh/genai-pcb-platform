"""
Configuration settings for the GenAI PCB Design Platform.

This module defines all configuration settings using Pydantic Settings
for type validation and environment variable loading.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/stuff_made_easy",
        description="PostgreSQL database connection URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for caching and message queue"
    )
    
    # AI Service Configuration
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM services"
    )
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude models"
    )
    
    # Application Settings
    SECRET_KEY: str = Field(
        default="change-this-in-production",
        description="Secret key for JWT token signing"
    )
    ALGORITHM: str = Field(
        default="HS256",
        description="Algorithm for JWT token signing"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="JWT token expiration time in minutes"
    )
    
    # Environment Configuration
    ENVIRONMENT: str = Field(
        default="development",
        description="Application environment (development, staging, production)"
    )
    DEBUG: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # File Storage Configuration
    UPLOAD_DIR: str = Field(
        default="./uploads",
        description="Directory for uploaded files"
    )
    GENERATED_DESIGNS_DIR: str = Field(
        default="./generated_designs",
        description="Directory for generated design files"
    )
    MAX_FILE_SIZE_MB: int = Field(
        default=50,
        description="Maximum file size in megabytes"
    )
    
    # KiCad Configuration
    KICAD_PATH: str = Field(
        default="/usr/bin/kicad",
        description="Path to KiCad executable"
    )
    KICAD_LIBRARIES_PATH: str = Field(
        default="/usr/share/kicad/library",
        description="Path to KiCad libraries"
    )
    
    # Simulation Configuration
    SPICE_SIMULATOR: str = Field(
        default="ngspice",
        description="SPICE simulator to use (ngspice, ltspice)"
    )
    SIMULATION_TIMEOUT_SECONDS: int = Field(
        default=300,
        description="Timeout for simulation operations in seconds"
    )
    
    # Component Database Configuration
    COMPONENT_DB_UPDATE_INTERVAL_HOURS: int = Field(
        default=24,
        description="Interval for updating component database in hours"
    )
    OCTOPART_API_KEY: Optional[str] = Field(
        default=None,
        description="Octopart API key for component data"
    )
    DIGIKEY_API_KEY: Optional[str] = Field(
        default=None,
        description="DigiKey API key for component data"
    )
    
    # Manufacturing Integration
    JLCPCB_API_KEY: Optional[str] = Field(
        default=None,
        description="JLCPCB API key for prototype ordering"
    )
    PCBWAY_API_KEY: Optional[str] = Field(
        default=None,
        description="PCBWay API key for prototype ordering"
    )
    
    # Monitoring Configuration
    PROMETHEUS_ENABLED: bool = Field(
        default=False,
        description="Enable Prometheus metrics collection"
    )
    GRAFANA_ENABLED: bool = Field(
        default=False,
        description="Enable Grafana dashboard"
    )
    SENTRY_DSN: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    
    # Rate Limiting Configuration
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        description="Rate limit per minute per user"
    )
    RATE_LIMIT_PER_HOUR: int = Field(
        default=1000,
        description="Rate limit per hour per user"
    )
    
    # Email Configuration
    SMTP_HOST: str = Field(
        default="smtp.gmail.com",
        description="SMTP server host"
    )
    SMTP_PORT: int = Field(
        default=587,
        description="SMTP server port"
    )
    SMTP_USERNAME: Optional[str] = Field(
        default=None,
        description="SMTP username"
    )
    SMTP_PASSWORD: Optional[str] = Field(
        default=None,
        description="SMTP password"
    )
    FROM_EMAIL: str = Field(
        default="noreply@stuff-made-easy.com",
        description="From email address"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # ignore REACT_APP_* and other extra vars from .env
    
    def __init__(self, **kwargs):
        """Initialize settings and create required directories."""
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            self.UPLOAD_DIR,
            self.GENERATED_DESIGNS_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"


# Global settings instance
settings = Settings()