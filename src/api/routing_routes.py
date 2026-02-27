"""
Routing optimizer API routes for the GenAI PCB Design Platform.

Integrates the routing optimizer into the main application API.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
import logging

from ..models.database import get_db
from ..models.design import DesignProject, DesignStatus
from .deps import get_current_user_id_for_designs
from routing_optimizer.api.job_queue import JobQueueManager, JobPriority
from routing_optimizer.models import RoutingConfig, RoutingRequest
from routing_optimizer.orchestrator import RoutingOrchestrator
from src.models.pcb_state import PCBState
from src.models.circuit_graph import CircuitGraph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/routing", tags=["routing"])

# Initialize job queue manager (singleton)
_job_queue = None

def get_job_queue() -> JobQueueManager:
    """Get or create job queue manager singleton."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueueManager(num_workers=2)
        _job_queue.start()
        logger.info("Routing job queue manager started")
    return _job_queue


@router.post("/designs/{design_id}/route")
async def route_design(
    design_id: str,
    algorithm: str = "astar",
    use_gpu: bool = False,
    nets_to_route: Optional[list[str]] = None,
    db: Session = Depends(get_db)
):
    """
    Start routing optimization for a design.
    
    This endpoint creates a routing job for the specified design and returns
    a job ID for tracking progress.
    
    Args:
        design_id: UUID of the design project
        algorithm: Routing algorithm ("astar" or "rl")
        use_gpu: Whether to use GPU for RL routing
        nets_to_route: Optional list of specific nets to route
        db: Database session
        
    Returns:
        Dict with job_id and status
    """
    try:
        # Get design project
        design = db.query(DesignProject).filter(DesignProject.id == design_id).first()
        
        if not design:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Design {design_id} not found"
            )
        
        # Check if design has PCB state
        if not hasattr(design, 'pcb_state') or design.pcb_state is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Design {design_id} does not have a PCB state. Please run placement first."
            )
        
        # Create routing configuration
        config = RoutingConfig(
            algorithm=algorithm,
            grid_resolution=0.5,
            max_steps_per_net=10000,
            use_gpu=use_gpu,
            checkpoint_path="checkpoints/rl_routing/best_model.pt" if algorithm == "rl" else None
        )
        
        # Create routing request
        routing_request = RoutingRequest(
            pcb_state=design.pcb_state,
            config=config,
            nets_to_route=nets_to_route,
            incremental=False
        )
        
        # Submit to job queue
        job_queue = get_job_queue()
        job_id = job_queue.create_job(routing_request, priority=JobPriority.NORMAL)
        
        # Update design status
        design.status = DesignStatus.ROUTING
        db.commit()
        
        logger.info(f"Started routing for design {design_id} with job {job_id}")
        
        return {
            "job_id": job_id,
            "design_id": design_id,
            "status": "queued",
            "algorithm": algorithm,
            "message": "Routing job has been queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting routing: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start routing: {str(e)}"
        )


@router.get("/jobs/{job_id}/status")
async def get_routing_status(job_id: str):
    """
    Get the status of a routing job.
    
    Args:
        job_id: Job ID from route_design endpoint
        
    Returns:
        JobStatus with current status and progress
    """
    try:
        job_queue = get_job_queue()
        job_status = job_queue.get_job_status(job_id)
        
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Routing job {job_id} not found"
            )
        
        return {
            "job_id": job_status.job_id,
            "status": job_status.status,
            "progress": job_status.progress,
            "current_net": job_status.current_net,
            "nets_completed": job_status.nets_completed,
            "nets_total": job_status.nets_total,
            "start_time": job_status.start_time.isoformat(),
            "metrics": {
                "total_trace_length": job_status.metrics.total_trace_length if job_status.metrics else 0,
                "via_count": job_status.metrics.via_count if job_status.metrics else 0,
                "drc_violations": job_status.metrics.drc_violation_count if job_status.metrics else 0
            } if job_status.metrics else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting routing status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get routing status: {str(e)}"
        )


@router.get("/jobs/{job_id}/result")
async def get_routing_result(job_id: str, db: Session = Depends(get_db)):
    """
    Get the result of a completed routing job.
    
    Args:
        job_id: Job ID from route_design endpoint
        db: Database session
        
    Returns:
        RoutingResult with routed traces and metrics
    """
    try:
        job_queue = get_job_queue()
        
        # Check job status first
        job_status = job_queue.get_job_status(job_id)
        
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Routing job {job_id} not found"
            )
        
        if job_status.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=f"Routing job {job_id} is still {job_status.status}"
            )
        
        # Get result
        result = job_queue.get_job_result(job_id)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Routing job {job_id} completed but result not found"
            )
        
        return {
            "success": result.success,
            "routing_stats": {
                "total_trace_length": result.routing_stats.total_trace_length,
                "via_count": result.routing_stats.via_count,
                "layer_transitions": result.routing_stats.layer_transitions,
                "nets_routed": result.routing_stats.nets_routed,
                "nets_total": result.routing_stats.nets_total,
                "drc_violation_count": result.routing_stats.drc_violation_count,
                "routing_time": result.routing_stats.routing_time,
                "per_layer_usage": result.routing_stats.per_layer_usage
            },
            "routing_time": result.routing_time,
            "routing_success_rate": result.routing_success_rate,
            "unrouted_nets": result.unrouted_nets,
            "drc_violations": [
                {
                    "type": v.violation_type,
                    "location": v.location,
                    "severity": v.severity,
                    "description": v.description,
                    "affected_nets": v.affected_nets
                }
                for v in result.drc_violations
            ],
            "error_message": result.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting routing result: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get routing result: {str(e)}"
        )


@router.post("/jobs/{job_id}/cancel")
async def cancel_routing_job(job_id: str):
    """
    Cancel a running routing job.
    
    Args:
        job_id: Job ID from route_design endpoint
        
    Returns:
        Dict with cancellation status
    """
    try:
        job_queue = get_job_queue()
        cancelled = job_queue.cancel_job(job_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Routing job {job_id} cannot be cancelled (not found or already completed)"
            )
        
        logger.info(f"Cancelled routing job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Routing job has been cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling routing job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel routing job: {str(e)}"
        )


@router.post("/optimize")
async def optimize_routing_direct(
    pcb_state: dict,
    config: dict,
    nets_to_route: Optional[list[str]] = None
):
    """
    Direct routing optimization without job queue (synchronous).
    
    This endpoint performs routing synchronously and returns the result immediately.
    Use this for small designs or when you need immediate results.
    
    Args:
        pcb_state: PCB state dictionary
        config: Routing configuration dictionary
        nets_to_route: Optional list of specific nets to route
        
    Returns:
        RoutingResult with routed traces and metrics
    """
    try:
        # Create orchestrator
        orchestrator = RoutingOrchestrator()
        
        # Parse configuration
        routing_config = RoutingConfig(**config)
        
        # Create PCB state (simplified - in practice would deserialize properly)
        # This is a placeholder - actual implementation would convert dict to PCBState
        import numpy as np
        grid = np.array(pcb_state.get("grid", []))
        pcb_state_obj = PCBState(grid=grid)
        
        # Execute routing
        result = orchestrator.optimize_routing(
            pcb_state=pcb_state_obj,
            config=routing_config,
            nets_to_route=nets_to_route
        )
        
        return {
            "success": result.success,
            "routing_stats": {
                "total_trace_length": result.routing_stats.total_trace_length,
                "via_count": result.routing_stats.via_count,
                "nets_routed": result.routing_stats.nets_routed,
                "nets_total": result.routing_stats.nets_total,
                "drc_violation_count": result.routing_stats.drc_violation_count,
                "routing_time": result.routing_stats.routing_time
            },
            "routing_time": result.routing_time,
            "routing_success_rate": result.routing_success_rate,
            "unrouted_nets": result.unrouted_nets,
            "error_message": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Error in direct routing optimization: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize routing: {str(e)}"
        )
