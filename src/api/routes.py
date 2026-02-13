"""
Main API routes for the GenAI PCB Design Platform.

Defines endpoints for design creation, processing, status checking, and file downloads.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import logging

from ..models.database import get_db
from ..models.design import DesignProject, DesignStatus
from .schemas import DesignCreateRequest, DesignResponse, DesignListResponse
from ..services.pipeline_orchestrator import get_pipeline_orchestrator
from ..services.request_queue import RequestPriority

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["designs"])


@router.post("/designs", response_model=DesignResponse, status_code=status.HTTP_201_CREATED)
async def create_design(
    request: DesignCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new PCB design from natural language prompt.
    
    Args:
        request: Design creation request with natural language prompt
        db: Database session
        
    Returns:
        DesignResponse: Created design project with ID and status
        
    Example:
        POST /api/v1/designs
        {
            "name": "LED Circuit",
            "description": "Simple LED with resistor",
            "prompt": "Design a 40x20mm PCB with 9V battery, LED, and 220-ohm resistor"
        }
    """
    try:
        # Create design project
        design = DesignProject(
            user_id="00000000-0000-0000-0000-000000000000",  # TODO: Get from auth
            name=request.name,
            description=request.description,
            natural_language_prompt=request.prompt,
            status=DesignStatus.DRAFT
        )
        
        db.add(design)
        db.commit()
        db.refresh(design)
        
        logger.info(f"Created design project: {design.id}")
        
        return DesignResponse(
            id=str(design.id),
            name=design.name,
            description=design.description,
            status=design.status.value,
            created_at=design.created_at,
            updated_at=design.updated_at
        )
        
    except Exception as e:
        logger.error(f"Error creating design: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create design: {str(e)}"
        )


@router.get("/designs", response_model=List[DesignListResponse])
async def list_designs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all design projects.
    
    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        List[DesignListResponse]: List of design projects
    """
    try:
        designs = db.query(DesignProject).offset(skip).limit(limit).all()
        
        return [
            DesignListResponse(
                id=str(design.id),
                name=design.name,
                status=design.status.value,
                created_at=design.created_at,
                updated_at=design.updated_at
            )
            for design in designs
        ]
        
    except Exception as e:
        logger.error(f"Error listing designs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list designs: {str(e)}"
        )


@router.get("/designs/{design_id}", response_model=DesignResponse)
async def get_design(
    design_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a specific design project by ID.
    
    Args:
        design_id: UUID of the design project
        db: Database session
        
    Returns:
        DesignResponse: Design project details
    """
    try:
        design = db.query(DesignProject).filter(DesignProject.id == design_id).first()
        
        if not design:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Design {design_id} not found"
            )
        
        return DesignResponse(
            id=str(design.id),
            name=design.name,
            description=design.description,
            status=design.status.value,
            prompt=design.natural_language_prompt,
            created_at=design.created_at,
            updated_at=design.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting design: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get design: {str(e)}"
        )


@router.delete("/designs/{design_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_design(
    design_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a design project.
    
    Args:
        design_id: UUID of the design project
        db: Database session
    """
    try:
        design = db.query(DesignProject).filter(DesignProject.id == design_id).first()
        
        if not design:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Design {design_id} not found"
            )
        
        db.delete(design)
        db.commit()
        
        logger.info(f"Deleted design project: {design_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting design: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete design: {str(e)}"
        )


@router.post("/designs/{design_id}/process")
async def process_design(
    design_id: str,
    priority: str = "normal",
    db: Session = Depends(get_db)
):
    """
    Start processing a design through the complete pipeline.
    
    Args:
        design_id: UUID of the design project
        priority: Processing priority (low, normal, high, critical)
        db: Database session
        
    Returns:
        Dict with request_id for tracking progress
    """
    try:
        # Get design project
        design = db.query(DesignProject).filter(DesignProject.id == design_id).first()
        
        if not design:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Design {design_id} not found"
            )
        
        # Map priority string to enum
        priority_map = {
            "low": RequestPriority.LOW,
            "normal": RequestPriority.NORMAL,
            "high": RequestPriority.HIGH,
            "critical": RequestPriority.CRITICAL
        }
        
        request_priority = priority_map.get(priority.lower(), RequestPriority.NORMAL)
        
        # Start pipeline processing
        orchestrator = get_pipeline_orchestrator()
        request_id = orchestrator.process_design_request(
            prompt=design.natural_language_prompt,
            user_id=str(design.user_id),
            priority=request_priority
        )
        
        # Update design status
        design.status = DesignStatus.PROCESSING
        db.commit()
        
        logger.info(f"Started processing design {design_id} with request {request_id}")
        
        return {
            "request_id": request_id,
            "design_id": design_id,
            "status": "processing_started",
            "message": "Design processing has been queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing design: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start design processing: {str(e)}"
        )


@router.get("/processing/{request_id}/status")
async def get_processing_status(request_id: str):
    """
    Get the status of a design processing request.
    
    Args:
        request_id: Request ID from process_design endpoint
        
    Returns:
        Dict with processing status, progress, and results
    """
    try:
        orchestrator = get_pipeline_orchestrator()
        status_info = orchestrator.get_pipeline_status(request_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing request {request_id} not found"
            )
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing status: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        Dict with service status and component health
    """
    try:
        orchestrator = get_pipeline_orchestrator()
        
        # Get queue information
        queue_info = orchestrator.request_queue.get_queue_info()
        
        return {
            "status": "healthy",
            "timestamp": "2026-02-13T22:30:00Z",
            "services": {
                "api": "healthy",
                "database": "healthy",
                "pipeline": "healthy",
                "queue": {
                    "status": "healthy",
                    "active_workers": queue_info["active_workers"],
                    "max_workers": queue_info["max_workers"],
                    "queued_requests": queue_info["queued_requests"],
                    "processing_requests": queue_info["processing_requests"]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": "2026-02-13T22:30:00Z",
            "error": str(e)
        }