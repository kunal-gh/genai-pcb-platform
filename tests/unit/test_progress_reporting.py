"""
Unit tests for progress reporting service.

Tests progress tracking, status updates, and completion estimates.
"""

import pytest
import time
from datetime import datetime

from src.services.progress_reporting import (
    ProgressReporter,
    ProgressStatus,
    ProgressStep,
    OperationProgress,
    get_progress_reporter
)


class TestProgressReporter:
    """Test suite for ProgressReporter."""
    
    def test_create_operation(self):
        """Test creating an operation for tracking."""
        reporter = ProgressReporter()
        
        steps = [
            {"name": "step1", "description": "First step", "weight": 1.0},
            {"name": "step2", "description": "Second step", "weight": 2.0}
        ]
        
        operation = reporter.create_operation(
            "op1",
            "test_operation",
            steps=steps,
            metadata={"key": "value"}
        )
        
        assert operation.operation_id == "op1"
        assert operation.operation_name == "test_operation"
        assert len(operation.steps) == 2
        assert operation.steps[0].name == "step1"
        assert operation.steps[1].weight == 2.0
        assert operation.metadata == {"key": "value"}
    
    def test_start_operation(self):
        """Test starting an operation."""
        reporter = ProgressReporter()
        reporter.create_operation("op1", "test_op")
        
        reporter.start_operation("op1", "Starting operation")
        
        progress = reporter.get_progress("op1")
        assert progress.status == ProgressStatus.IN_PROGRESS
        assert progress.started_at is not None
        assert progress.message == "Starting operation"
    
    def test_start_nonexistent_operation(self):
        """Test starting an operation that doesn't exist."""
        reporter = ProgressReporter()
        
        with pytest.raises(ValueError, match="Operation .* not found"):
            reporter.start_operation("nonexistent")
    
    def test_start_step(self):
        """Test starting a step."""
        reporter = ProgressReporter()
        steps = [{"name": "step1", "description": "Test step"}]
        reporter.create_operation("op1", "test_op", steps=steps)
        reporter.start_operation("op1")
        
        reporter.start_step("op1", "step1", "Processing step 1")
        
        progress = reporter.get_progress("op1")
        assert progress.current_step == "step1"
        step = progress.steps[0]
        assert step.status == ProgressStatus.IN_PROGRESS
        assert step.started_at is not None
        assert step.message == "Processing step 1"
    
    def test_start_nonexistent_step(self):
        """Test starting a step that doesn't exist."""
        reporter = ProgressReporter()
        reporter.create_operation("op1", "test_op")
        reporter.start_operation("op1")
        
        with pytest.raises(ValueError, match="Step .* not found"):
            reporter.start_step("op1", "nonexistent")
    
    def test_update_step_progress(self):
        """Test updating step progress."""
        reporter = ProgressReporter()
        steps = [{"name": "step1", "description": "Test step"}]
        reporter.create_operation("op1", "test_op", steps=steps)
        reporter.start_operation("op1")
        reporter.start_step("op1", "step1")
        
        reporter.update_step_progress("op1", "step1", 50.0, "Halfway done")
        
        progress = reporter.get_progress("op1")
        step = progress.steps[0]
        assert step.progress_percent == 50.0
        assert step.message == "Halfway done"
    
    def test_update_step_progress_clamping(self):
        """Test that progress is clamped to 0-100 range."""
        reporter = ProgressReporter()
        steps = [{"name": "step1", "description": "Test step"}]
        reporter.create_operation("op1", "test_op", steps=steps)
        reporter.start_operation("op1")
        reporter.start_step("op1", "step1")
        
        # Test over 100
        reporter.update_step_progress("op1", "step1", 150.0)
        progress = reporter.get_progress("op1")
        assert progress.steps[0].progress_percent == 100.0
        
        # Test under 0
        reporter.update_step_progress("op1", "step1", -10.0)
        progress = reporter.get_progress("op1")
        assert progress.steps[0].progress_percent == 0.0
    
    def test_complete_step(self):
        """Test completing a step."""
        reporter = ProgressReporter()
        steps = [{"name": "step1", "description": "Test step"}]
        reporter.create_operation("op1", "test_op", steps=steps)
        reporter.start_operation("op1")
        reporter.start_step("op1", "step1")
        
        reporter.complete_step("op1", "step1", "Step completed")
        
        progress = reporter.get_progress("op1")
        step = progress.steps[0]
        assert step.status == ProgressStatus.COMPLETED
        assert step.progress_percent == 100.0
        assert step.completed_at is not None
        assert step.message == "Step completed"
    
    def test_complete_operation(self):
        """Test completing an operation."""
        reporter = ProgressReporter()
        steps = [
            {"name": "step1", "description": "Step 1"},
            {"name": "step2", "description": "Step 2"}
        ]
        reporter.create_operation("op1", "test_op", steps=steps)
        reporter.start_operation("op1")
        
        reporter.complete_operation("op1", "All done")
        
        progress = reporter.get_progress("op1")
        assert progress.status == ProgressStatus.COMPLETED
        assert progress.overall_progress == 100.0
        assert progress.completed_at is not None
        assert progress.message == "All done"
        assert progress.current_step is None
        
        # All steps should be marked completed
        for step in progress.steps:
            assert step.status == ProgressStatus.COMPLETED
            assert step.progress_percent == 100.0
    
    def test_fail_operation(self):
        """Test failing an operation."""
        reporter = ProgressReporter()
        reporter.create_operation("op1", "test_op")
        reporter.start_operation("op1")
        
        reporter.fail_operation("op1", "Something went wrong")
        
        progress = reporter.get_progress("op1")
        assert progress.status == ProgressStatus.FAILED
        assert progress.completed_at is not None
        assert progress.message == "Something went wrong"
    
    def test_cancel_operation(self):
        """Test cancelling an operation."""
        reporter = ProgressReporter()
        reporter.create_operation("op1", "test_op")
        reporter.start_operation("op1")
        
        reporter.cancel_operation("op1", "User cancelled")
        
        progress = reporter.get_progress("op1")
        assert progress.status == ProgressStatus.CANCELLED
        assert progress.completed_at is not None
        assert progress.message == "User cancelled"
    
    def test_overall_progress_calculation(self):
        """Test overall progress calculation with weighted steps."""
        reporter = ProgressReporter()
        steps = [
            {"name": "step1", "description": "Step 1", "weight": 1.0},
            {"name": "step2", "description": "Step 2", "weight": 2.0},
            {"name": "step3", "description": "Step 3", "weight": 1.0}
        ]
        reporter.create_operation("op1", "test_op", steps=steps)
        reporter.start_operation("op1")
        
        # Complete first step (weight 1.0)
        reporter.start_step("op1", "step1")
        reporter.complete_step("op1", "step1")
        
        progress = reporter.get_progress("op1")
        # 1.0 / (1.0 + 2.0 + 1.0) * 100 = 25%
        assert progress.overall_progress == 25.0
        
        # Update second step to 50% (weight 2.0)
        reporter.start_step("op1", "step2")
        reporter.update_step_progress("op1", "step2", 50.0)
        
        progress = reporter.get_progress("op1")
        # (1.0 * 100 + 2.0 * 50 + 1.0 * 0) / 4.0 = 50%
        assert progress.overall_progress == 50.0
    
    def test_get_nonexistent_progress(self):
        """Test getting progress for nonexistent operation."""
        reporter = ProgressReporter()
        
        progress = reporter.get_progress("nonexistent")
        assert progress is None
    
    def test_operation_to_dict(self):
        """Test converting operation progress to dictionary."""
        reporter = ProgressReporter()
        steps = [{"name": "step1", "description": "Test step"}]
        reporter.create_operation("op1", "test_op", steps=steps, metadata={"key": "value"})
        reporter.start_operation("op1", "Starting")
        reporter.start_step("op1", "step1")
        reporter.update_step_progress("op1", "step1", 75.0)
        
        progress = reporter.get_progress("op1")
        data = progress.to_dict()
        
        assert data["operation_id"] == "op1"
        assert data["operation_name"] == "test_op"
        assert data["status"] == "in_progress"
        assert data["overall_progress"] == 75.0
        assert data["current_step"] == "step1"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["name"] == "step1"
        assert data["steps"][0]["progress_percent"] == 75.0
        assert data["message"] == "Starting"
        assert data["metadata"] == {"key": "value"}
        assert data["started_at"] is not None
    
    def test_multiple_operations(self):
        """Test tracking multiple operations simultaneously."""
        reporter = ProgressReporter()
        
        reporter.create_operation("op1", "operation_1")
        reporter.create_operation("op2", "operation_2")
        reporter.create_operation("op3", "operation_3")
        
        reporter.start_operation("op1")
        reporter.start_operation("op2")
        
        progress1 = reporter.get_progress("op1")
        progress2 = reporter.get_progress("op2")
        progress3 = reporter.get_progress("op3")
        
        assert progress1.status == ProgressStatus.IN_PROGRESS
        assert progress2.status == ProgressStatus.IN_PROGRESS
        assert progress3.status == ProgressStatus.NOT_STARTED


class TestGetProgressReporter:
    """Test global progress reporter instance."""
    
    def test_get_progress_reporter_singleton(self):
        """Test that get_progress_reporter returns singleton."""
        reporter1 = get_progress_reporter()
        reporter2 = get_progress_reporter()
        
        assert reporter1 is reporter2
    
    def test_get_progress_reporter_returns_instance(self):
        """Test that get_progress_reporter returns ProgressReporter."""
        reporter = get_progress_reporter()
        assert isinstance(reporter, ProgressReporter)


class TestProgressStep:
    """Test ProgressStep dataclass."""
    
    def test_create_progress_step(self):
        """Test creating a progress step."""
        step = ProgressStep(
            name="test_step",
            description="Test description",
            weight=2.0
        )
        
        assert step.name == "test_step"
        assert step.description == "Test description"
        assert step.weight == 2.0
        assert step.status == ProgressStatus.NOT_STARTED
        assert step.progress_percent == 0.0
    
    def test_progress_step_defaults(self):
        """Test progress step with default values."""
        step = ProgressStep(
            name="test_step",
            description="Test"
        )
        
        assert step.weight == 1.0
        assert step.status == ProgressStatus.NOT_STARTED
        assert step.started_at is None
        assert step.completed_at is None


class TestOperationProgress:
    """Test OperationProgress dataclass."""
    
    def test_create_operation_progress(self):
        """Test creating operation progress."""
        steps = [
            ProgressStep(name="step1", description="Step 1"),
            ProgressStep(name="step2", description="Step 2")
        ]
        
        progress = OperationProgress(
            operation_id="op1",
            operation_name="test_op",
            steps=steps,
            metadata={"key": "value"}
        )
        
        assert progress.operation_id == "op1"
        assert progress.operation_name == "test_op"
        assert len(progress.steps) == 2
        assert progress.status == ProgressStatus.NOT_STARTED
        assert progress.overall_progress == 0.0
        assert progress.metadata == {"key": "value"}
