"""
Pydantic schemas for API request/response validation.

Defines data validation models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DesignCreateRequest(BaseModel):
    """Request schema for creating a new design."""
    name: str = Field(..., min_length=1, max_length=255, description="Design project name")
    description: Optional[str] = Field(None, description="Optional design description")
    prompt: str = Field(..., min_length=10, max_length=10000, description="Natural language design prompt")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "LED Circuit",
                "description": "Simple LED indicator circuit",
                "prompt": "Design a 40x20mm PCB with a 9V battery connector, a 5mm LED indicator, and a 220-ohm resistor inline"
            }
        }


class DesignResponse(BaseModel):
    """Response schema for design project details."""
    id: str
    name: str
    description: Optional[str]
    status: str
    prompt: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DesignListResponse(BaseModel):
    """Response schema for design list items."""
    id: str
    name: str
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    environment: str
    debug: bool
    database: str = "connected"
    redis: str = "connected"