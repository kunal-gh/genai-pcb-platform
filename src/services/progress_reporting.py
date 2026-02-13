"""
Progress reporting service for long-running operations.

Provides progress tracking, status updates, and completion estimates
for the GenAI PCB Design Platform.

Requirements: 13.5
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ProgressStatus(Enum):
    """Progress status for operations."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressStep:
    """A step in a multi-step operation."""
    name: str
    description: str
    weight: float = 1.0  # Relative weight for progress calculation
    status: ProgressStatus = ProgressStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percent: float = 0.0
    message: Optional[str] = None


@dataclass
class OperationProgress:
    """Progress tracking for an operation."""
    operation_id: str
    operation_name: str
    status: ProgressStatus = ProgressStatus.NOT_STARTED
    overall_progress: float = 0.0
    current_step: Optional[str] = None
    steps: List[ProgressStep] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "status": self.status.value,
            "overall_progress": self.overall_progress,
            "current_step": self.current_step,
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "status": step.status.value,
                    "progress_percent": step.progress_percent,
                    "message": step.message
                }
                for step in self.steps
            ],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "message": self.message,
            "metadata": self.metadata
        }


class ProgressReporter:
    """
    Progress reporting service for tracking long-running operations.
    
    Provides:
    - Multi-step progress tracking
    - Progress percentage calculation
    - Estimated completion time
    - Status updates and messages
    """
    
    def __init__(self):
        """Initialize progress reporter."""
        self._operations: Dict[str, OperationProgress] = {}
        self._lock = threading.RLock()  # Use reentrant lock to avoid deadlocks
        logger.info("Progress reporter initialized")
    
    def create_operation(
        self,
        operation_id: str,
        operation_name: str,
        steps: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OperationProgress:
        """
        Create a new operation for progress tracking.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_name: Name of the operation
            steps: Optional list of step definitions with name, description, weight
            metadata: Optional metadata
            
        Returns:
            OperationProgress object
        """
        progress_steps = []
        if steps:
            for step_def in steps:
                progress_steps.append(ProgressStep(
                    name=step_def["name"],
                    description=step_def.get("description", ""),
                    weight=step_def.get("weight", 1.0)
                ))
        
        operation = OperationProgress(
            operation_id=operation_id,
            operation_name=operation_name,
            steps=progress_steps,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._operations[operation_id] = operation
        
        logger.info(f"Created progress tracking for operation {operation_id} ({operation_name})")
        return operation
    
    def start_operation(self, operation_id: str, message: Optional[str] = None) -> None:
        """
        Mark an operation as started.
        
        Args:
            operation_id: Operation identifier
            message: Optional status message
        """
        with self._lock:
            operation = self._operations.get(operation_id)
            if not operation:
                raise ValueError(f"Operation {operation_id} not found")
            
            operation.status = ProgressStatus.IN_PROGRESS
            operation.started_at = datetime.now()
            if message:
                operation.message = message
        
        logger.info(f"Started operation {operation_id}")
    
    def start_step(self, operation_id: str, step_name: str, message: Optional[str] = None) -> None:
        """
        Mark a step as started.
        
        Args:
            operation_id: Operation identifier
            step_name: Name of the step
            message: Optional status message
        """
        with self._lock:
            operation = self._operations.get(operation_id)
            if not operation:
                raise ValueError(f"Operation {operation_id} not found")
            
            step = self._find_step(operation, step_name)
            if not step:
                raise ValueError(f"Step {step_name} not found in operation {operation_id}")
            
            step.status = ProgressStatus.IN_PROGRESS
            step.started_at = datetime.now()
            if message:
                step.message = message
            
            operation.current_step = step_name
            self._update_overall_progress(operation)
        
        logger.info(f"Started step {step_name} in operation {operation_id}")
    
    def update_step_progress(
        self,
        operation_id: str,
        step_name: str,
        progress_percent: float,
        message: Optional[str] = None
    ) -> None:
        """
        Update progress for a specific step.
        
        Args:
            operation_id: Operation identifier
            step_name: Name of the step
            progress_percent: Progress percentage (0-100)
            message: Optional status message
        """
        with self._lock:
            operation = self._operations.get(operation_id)
            if not operation:
                raise ValueError(f"Operation {operation_id} not found")
            
            step = self._find_step(operation, step_name)
            if not step:
                raise ValueError(f"Step {step_name} not found in operation {operation_id}")
            
            step.progress_percent = max(0.0, min(100.0, progress_percent))
            if message:
                step.message = message
            
            self._update_overall_progress(operation)
        
        logger.debug(f"Updated step {step_name} progress to {progress_percent}% in operation {operation_id}")
    
    def complete_step(self, operation_id: str, step_name: str, message: Optional[str] = None) -> None:
        """
        Mark a step as completed.
        
        Args:
            operation_id: Operation identifier
            step_name: Name of the step
            message: Optional status message
        """
        with self._lock:
            operation = self._operations.get(operation_id)
            if not operation:
                raise ValueError(f"Operation {operation_id} not found")
            
            step = self._find_step(operation, step_name)
            if not step:
                raise ValueError(f"Step {step_name} not found in operation {operation_id}")
            
            step.status = ProgressStatus.COMPLETED
            step.progress_percent = 100.0
            step.completed_at = datetime.now()
            if message:
                step.message = message
            
            self._update_overall_progress(operation)
        
        logger.info(f"Completed step {step_name} in operation {operation_id}")
    
    def complete_operation(self, operation_id: str, message: Optional[str] = None) -> None:
        """
        Mark an operation as completed.
        
        Args:
            operation_id: Operation identifier
            message: Optional status message
        """
        with self._lock:
            operation = self._operations.get(operation_id)
            if not operation:
                raise ValueError(f"Operation {operation_id} not found")
            
            operation.status = ProgressStatus.COMPLETED
            operation.overall_progress = 100.0
            operation.completed_at = datetime.now()
            operation.current_step = None
            if message:
                operation.message = message
            
            # Mark all steps as completed
            for step in operation.steps:
                if step.status != ProgressStatus.COMPLETED:
                    step.status = ProgressStatus.COMPLETED
                    step.progress_percent = 100.0
        
        logger.info(f"Completed operation {operation_id}")
    
    def fail_operation(self, operation_id: str, error_message: str) -> None:
        """
        Mark an operation as failed.
        
        Args:
            operation_id: Operation identifier
            error_message: Error message
        """
        with self._lock:
            operation = self._operations.get(operation_id)
            if not operation:
                raise ValueError(f"Operation {operation_id} not found")
            
            operation.status = ProgressStatus.FAILED
            operation.completed_at = datetime.now()
            operation.message = error_message
        
        logger.error(f"Operation {operation_id} failed: {error_message}")
    
    def cancel_operation(self, operation_id: str, message: Optional[str] = None) -> None:
        """
        Mark an operation as cancelled.
        
        Args:
            operation_id: Operation identifier
            message: Optional cancellation message
        """
        with self._lock:
            operation = self._operations.get(operation_id)
            if not operation:
                raise ValueError(f"Operation {operation_id} not found")
            
            operation.status = ProgressStatus.CANCELLED
            operation.completed_at = datetime.now()
            if message:
                operation.message = message
        
        logger.info(f"Cancelled operation {operation_id}")
    
    def get_progress(self, operation_id: str) -> Optional[OperationProgress]:
        """
        Get progress for an operation.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            OperationProgress if found, None otherwise
        """
        with self._lock:
            return self._operations.get(operation_id)
    
    def _find_step(self, operation: OperationProgress, step_name: str) -> Optional[ProgressStep]:
        """Find a step by name in an operation."""
        for step in operation.steps:
            if step.name == step_name:
                return step
        return None
    
    def _update_overall_progress(self, operation: OperationProgress) -> None:
        """Update overall progress based on step progress."""
        if not operation.steps:
            return
        
        total_weight = sum(step.weight for step in operation.steps)
        if total_weight == 0:
            return
        
        weighted_progress = sum(
            step.progress_percent * step.weight
            for step in operation.steps
        )
        
        operation.overall_progress = weighted_progress / total_weight


# Global progress reporter instance
_progress_reporter: Optional[ProgressReporter] = None


def get_progress_reporter() -> ProgressReporter:
    """Get the global progress reporter instance."""
    global _progress_reporter
    if _progress_reporter is None:
        _progress_reporter = ProgressReporter()
    return _progress_reporter
