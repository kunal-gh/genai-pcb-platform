"""
Tests for Error Management Service
"""

import pytest
from src.services.error_management import (
    ErrorManager,
    ErrorSeverity,
    ErrorCategory,
    ErrorRecord,
    GracefulDegradation,
    PartialResultRecovery,
    get_error_manager
)


@pytest.fixture
def error_manager(tmp_path):
    """Create ErrorManager instance with temp log file"""
    log_file = tmp_path / "test_errors.log"
    return ErrorManager(log_file=str(log_file))


@pytest.fixture
def graceful_degradation(error_manager):
    """Create GracefulDegradation instance"""
    return GracefulDegradation(error_manager)


@pytest.fixture
def partial_recovery(error_manager):
    """Create PartialResultRecovery instance"""
    return PartialResultRecovery(error_manager)


def test_log_error_basic(error_manager):
    """Test basic error logging"""
    error_record = error_manager.log_error(
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.NLP_PARSING,
        message="Test error message"
    )
    
    assert error_record.severity == "error"
    assert error_record.category == "nlp_parsing"
    assert error_record.message == "Test error message"
    assert error_record.recoverable is True


def test_log_error_with_details(error_manager):
    """Test error logging with details"""
    error_record = error_manager.log_error(
        severity=ErrorSeverity.CRITICAL,
        category=ErrorCategory.DATABASE,
        message="Database connection failed",
        details="Connection timeout after 30 seconds",
        user_id="user123",
        design_id="design456"
    )
    
    assert error_record.details == "Connection timeout after 30 seconds"
    assert error_record.user_id == "user123"
    assert error_record.design_id == "design456"


def test_log_error_with_exception(error_manager):
    """Test error logging with exception"""
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        error_record = error_manager.log_error(
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.CODE_GENERATION,
            message="Code generation failed",
            exception=e
        )
    
    assert error_record.stack_trace is not None
    assert "ValueError" in error_record.stack_trace
    assert "Test exception" in error_record.stack_trace


def test_error_history(error_manager):
    """Test error history tracking"""
    # Log multiple errors
    for i in range(5):
        error_manager.log_error(
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.VERIFICATION,
            message=f"Warning {i}"
        )
    
    assert len(error_manager.error_history) == 5


def test_error_history_max_limit(error_manager):
    """Test error history respects max limit"""
    error_manager.max_history = 10
    
    # Log more than max
    for i in range(15):
        error_manager.log_error(
            severity=ErrorSeverity.INFO,
            category=ErrorCategory.SYSTEM,
            message=f"Info {i}"
        )
    
    assert len(error_manager.error_history) == 10


def test_attempt_recovery_success(error_manager):
    """Test successful error recovery"""
    error_record = error_manager.log_error(
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.SIMULATION,
        message="Simulation failed",
        recoverable=True
    )
    
    def recovery_func():
        return "recovered"
    
    success, result = error_manager.attempt_recovery(error_record, recovery_func)
    
    assert success is True
    assert result == "recovered"
    assert error_record.recovery_attempted is True
    assert error_record.recovery_successful is True


def test_attempt_recovery_failure(error_manager):
    """Test failed error recovery"""
    error_record = error_manager.log_error(
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.PCB_LAYOUT,
        message="Layout failed",
        recoverable=True
    )
    
    def recovery_func():
        raise Exception("Recovery failed")
    
    success, result = error_manager.attempt_recovery(error_record, recovery_func)
    
    assert success is False
    assert result is None
    assert error_record.recovery_attempted is True
    assert error_record.recovery_successful is False


def test_attempt_recovery_non_recoverable(error_manager):
    """Test recovery attempt on non-recoverable error"""
    error_record = error_manager.log_error(
        severity=ErrorSeverity.CRITICAL,
        category=ErrorCategory.SYSTEM,
        message="System failure",
        recoverable=False
    )
    
    def recovery_func():
        return "recovered"
    
    success, result = error_manager.attempt_recovery(error_record, recovery_func)
    
    assert success is False
    assert result is None


def test_get_error_statistics(error_manager):
    """Test error statistics generation"""
    # Log various errors
    error_manager.log_error(ErrorSeverity.ERROR, ErrorCategory.NLP_PARSING, "Error 1")
    error_manager.log_error(ErrorSeverity.ERROR, ErrorCategory.NLP_PARSING, "Error 2")
    error_manager.log_error(ErrorSeverity.WARNING, ErrorCategory.VERIFICATION, "Warning 1")
    error_manager.log_error(ErrorSeverity.CRITICAL, ErrorCategory.DATABASE, "Critical 1")
    
    stats = error_manager.get_error_statistics()
    
    assert stats['total_errors'] == 4
    assert stats['by_severity']['error'] == 2
    assert stats['by_severity']['warning'] == 1
    assert stats['by_severity']['critical'] == 1
    assert stats['by_category']['nlp_parsing'] == 2


def test_get_error_statistics_empty(error_manager):
    """Test error statistics with no errors"""
    stats = error_manager.get_error_statistics()
    
    assert stats['total_errors'] == 0
    assert stats['by_severity'] == {}
    assert stats['by_category'] == {}
    assert stats['recovery_rate'] == 0.0


def test_get_recent_errors(error_manager):
    """Test getting recent errors"""
    for i in range(15):
        error_manager.log_error(
            severity=ErrorSeverity.INFO,
            category=ErrorCategory.SYSTEM,
            message=f"Error {i}"
        )
    
    recent = error_manager.get_recent_errors(count=5)
    
    assert len(recent) == 5
    assert recent[-1].message == "Error 14"


def test_get_errors_by_design(error_manager):
    """Test getting errors for specific design"""
    error_manager.log_error(
        ErrorSeverity.ERROR,
        ErrorCategory.SCHEMATIC_GENERATION,
        "Error 1",
        design_id="design123"
    )
    error_manager.log_error(
        ErrorSeverity.ERROR,
        ErrorCategory.PCB_LAYOUT,
        "Error 2",
        design_id="design123"
    )
    error_manager.log_error(
        ErrorSeverity.ERROR,
        ErrorCategory.VERIFICATION,
        "Error 3",
        design_id="design456"
    )
    
    design_errors = error_manager.get_errors_by_design("design123")
    
    assert len(design_errors) == 2
    assert all(e.design_id == "design123" for e in design_errors)


def test_clear_history(error_manager):
    """Test clearing error history"""
    error_manager.log_error(ErrorSeverity.ERROR, ErrorCategory.SYSTEM, "Error 1")
    error_manager.log_error(ErrorSeverity.ERROR, ErrorCategory.SYSTEM, "Error 2")
    
    assert len(error_manager.error_history) == 2
    
    error_manager.clear_history()
    
    assert len(error_manager.error_history) == 0


def test_graceful_degradation_register_fallback(graceful_degradation):
    """Test registering fallback strategy"""
    def fallback_func():
        return "fallback"
    
    graceful_degradation.register_fallback("test_service", fallback_func)
    
    assert "test_service" in graceful_degradation.fallback_strategies


def test_graceful_degradation_primary_success(graceful_degradation):
    """Test execution with successful primary function"""
    def primary_func():
        return "primary"
    
    used_fallback, result = graceful_degradation.execute_with_fallback(
        "test_service",
        primary_func
    )
    
    assert used_fallback is False
    assert result == "primary"


def test_graceful_degradation_fallback_used(graceful_degradation):
    """Test execution with fallback when primary fails"""
    def primary_func():
        raise Exception("Primary failed")
    
    def fallback_func():
        return "fallback"
    
    graceful_degradation.register_fallback("test_service", fallback_func)
    
    used_fallback, result = graceful_degradation.execute_with_fallback(
        "test_service",
        primary_func
    )
    
    assert used_fallback is True
    assert result == "fallback"


def test_graceful_degradation_no_fallback(graceful_degradation):
    """Test execution without fallback raises exception"""
    def primary_func():
        raise ValueError("Primary failed")
    
    with pytest.raises(ValueError, match="Primary failed"):
        graceful_degradation.execute_with_fallback(
            "test_service",
            primary_func
        )


def test_partial_result_save(partial_recovery):
    """Test saving partial results"""
    partial_recovery.save_partial_result(
        design_id="design123",
        stage="schematic",
        result={"data": "schematic_data"},
        metadata={"version": "1.0"}
    )
    
    assert partial_recovery.has_partial_results("design123")


def test_partial_result_get(partial_recovery):
    """Test getting partial results"""
    partial_recovery.save_partial_result(
        design_id="design123",
        stage="schematic",
        result={"data": "schematic_data"}
    )
    partial_recovery.save_partial_result(
        design_id="design123",
        stage="pcb",
        result={"data": "pcb_data"}
    )
    
    results = partial_recovery.get_partial_results("design123")
    
    assert "schematic" in results
    assert "pcb" in results
    assert results["schematic"]["result"]["data"] == "schematic_data"


def test_partial_result_completed_stages(partial_recovery):
    """Test getting completed stages"""
    partial_recovery.save_partial_result("design123", "stage1", {"data": "1"})
    partial_recovery.save_partial_result("design123", "stage2", {"data": "2"})
    partial_recovery.save_partial_result("design123", "stage3", {"data": "3"})
    
    stages = partial_recovery.get_completed_stages("design123")
    
    assert len(stages) == 3
    assert "stage1" in stages
    assert "stage2" in stages
    assert "stage3" in stages


def test_partial_result_clear(partial_recovery):
    """Test clearing partial results"""
    partial_recovery.save_partial_result("design123", "stage1", {"data": "1"})
    
    assert partial_recovery.has_partial_results("design123")
    
    partial_recovery.clear_partial_results("design123")
    
    assert not partial_recovery.has_partial_results("design123")


def test_partial_result_no_results(partial_recovery):
    """Test getting partial results for non-existent design"""
    results = partial_recovery.get_partial_results("nonexistent")
    
    assert results == {}
    assert not partial_recovery.has_partial_results("nonexistent")


def test_get_error_manager_singleton():
    """Test global error manager singleton"""
    manager1 = get_error_manager()
    manager2 = get_error_manager()
    
    assert manager1 is manager2


def test_error_record_dataclass():
    """Test ErrorRecord dataclass"""
    record = ErrorRecord(
        error_id="test123",
        timestamp="2024-01-01T00:00:00",
        severity="error",
        category="system",
        message="Test error",
        recoverable=True
    )
    
    assert record.error_id == "test123"
    assert record.severity == "error"
    assert record.recoverable is True


def test_recovery_rate_calculation(error_manager):
    """Test recovery rate calculation in statistics"""
    # Create errors and attempt recovery
    for i in range(10):
        error = error_manager.log_error(
            ErrorSeverity.ERROR,
            ErrorCategory.SYSTEM,
            f"Error {i}",
            recoverable=True
        )
        
        # Recover half of them successfully
        if i < 5:
            error_manager.attempt_recovery(error, lambda: "success")
        else:
            error_manager.attempt_recovery(error, lambda: (_ for _ in ()).throw(Exception("fail")))
    
    stats = error_manager.get_error_statistics()
    
    assert stats['recovery_attempts'] == 10
    assert stats['recovery_successes'] == 5
    assert stats['recovery_rate'] == 50.0
