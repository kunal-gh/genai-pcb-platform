"""
Error Management Service

Provides centralized error logging, monitoring, and graceful degradation
for the GenAI PCB Design Platform.
"""

import logging
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import json


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for classification"""
    NLP_PARSING = "nlp_parsing"
    CODE_GENERATION = "code_generation"
    COMPONENT_SELECTION = "component_selection"
    SCHEMATIC_GENERATION = "schematic_generation"
    PCB_LAYOUT = "pcb_layout"
    VERIFICATION = "verification"
    SIMULATION = "simulation"
    FILE_EXPORT = "file_export"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"


@dataclass
class ErrorRecord:
    """Structured error record"""
    error_id: str
    timestamp: str
    severity: str
    category: str
    message: str
    details: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    design_id: Optional[str] = None
    recoverable: bool = True
    recovery_attempted: bool = False
    recovery_successful: Optional[bool] = None


class ErrorManager:
    """Centralized error management system"""
    
    def __init__(self, log_file: str = "logs/errors.log"):
        self.log_file = log_file
        self.logger = self._setup_logger()
        self.error_history: List[ErrorRecord] = []
        self.max_history = 1000
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger("error_manager")
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else "logs", exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_error(
        self,
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: str,
        details: Optional[str] = None,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        design_id: Optional[str] = None,
        recoverable: bool = True
    ) -> ErrorRecord:
        """
        Log an error with full context
        
        Args:
            severity: Error severity level
            category: Error category
            message: Human-readable error message
            details: Additional error details
            exception: Exception object if available
            context: Additional context information
            user_id: User ID if applicable
            design_id: Design ID if applicable
            recoverable: Whether error is recoverable
            
        Returns:
            ErrorRecord object
        """
        # Generate error ID
        error_id = f"{category.value}_{datetime.utcnow().timestamp()}"
        
        # Get stack trace if exception provided
        stack_trace = None
        if exception:
            stack_trace = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.utcnow().isoformat(),
            severity=severity.value,
            category=category.value,
            message=message,
            details=details,
            stack_trace=stack_trace,
            context=context,
            user_id=user_id,
            design_id=design_id,
            recoverable=recoverable,
            recovery_attempted=False,
            recovery_successful=None
        )
        
        # Add to history
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log to file
        log_message = f"[{severity.value.upper()}] [{category.value}] {message}"
        if details:
            log_message += f" | Details: {details}"
        if design_id:
            log_message += f" | Design: {design_id}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        if stack_trace:
            self.logger.debug(f"Stack trace: {stack_trace}")
        
        return error_record
    
    def attempt_recovery(
        self,
        error_record: ErrorRecord,
        recovery_function: callable,
        *args,
        **kwargs
    ) -> tuple[bool, Any]:
        """
        Attempt to recover from an error
        
        Args:
            error_record: Error record to recover from
            recovery_function: Function to attempt recovery
            *args, **kwargs: Arguments for recovery function
            
        Returns:
            Tuple of (success, result)
        """
        if not error_record.recoverable:
            self.logger.warning(f"Error {error_record.error_id} is not recoverable")
            return False, None
        
        error_record.recovery_attempted = True
        
        try:
            result = recovery_function(*args, **kwargs)
            error_record.recovery_successful = True
            self.logger.info(f"Successfully recovered from error {error_record.error_id}")
            return True, result
        except Exception as e:
            error_record.recovery_successful = False
            self.logger.error(f"Recovery failed for error {error_record.error_id}: {str(e)}")
            return False, None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {
                'total_errors': 0,
                'by_severity': {},
                'by_category': {},
                'recovery_rate': 0.0
            }
        
        by_severity = {}
        by_category = {}
        recovery_attempts = 0
        recovery_successes = 0
        
        for error in self.error_history:
            # Count by severity
            by_severity[error.severity] = by_severity.get(error.severity, 0) + 1
            
            # Count by category
            by_category[error.category] = by_category.get(error.category, 0) + 1
            
            # Track recovery
            if error.recovery_attempted:
                recovery_attempts += 1
                if error.recovery_successful:
                    recovery_successes += 1
        
        recovery_rate = (recovery_successes / recovery_attempts * 100) if recovery_attempts > 0 else 0.0
        
        return {
            'total_errors': len(self.error_history),
            'by_severity': by_severity,
            'by_category': by_category,
            'recovery_attempts': recovery_attempts,
            'recovery_successes': recovery_successes,
            'recovery_rate': recovery_rate
        }
    
    def get_recent_errors(self, count: int = 10) -> List[ErrorRecord]:
        """Get most recent errors"""
        return self.error_history[-count:]
    
    def get_errors_by_design(self, design_id: str) -> List[ErrorRecord]:
        """Get all errors for a specific design"""
        return [e for e in self.error_history if e.design_id == design_id]
    
    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.logger.info("Error history cleared")


class GracefulDegradation:
    """Handles graceful degradation for service failures"""
    
    def __init__(self, error_manager: ErrorManager):
        self.error_manager = error_manager
        self.fallback_strategies: Dict[str, callable] = {}
    
    def register_fallback(self, service_name: str, fallback_function: callable):
        """Register a fallback strategy for a service"""
        self.fallback_strategies[service_name] = fallback_function
        self.error_manager.logger.info(f"Registered fallback for {service_name}")
    
    def execute_with_fallback(
        self,
        service_name: str,
        primary_function: callable,
        *args,
        **kwargs
    ) -> tuple[bool, Any]:
        """
        Execute function with fallback on failure
        
        Args:
            service_name: Name of the service
            primary_function: Primary function to execute
            *args, **kwargs: Arguments for function
            
        Returns:
            Tuple of (used_fallback, result)
        """
        try:
            result = primary_function(*args, **kwargs)
            return False, result
        except Exception as e:
            # Log error
            self.error_manager.log_error(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                message=f"Service {service_name} failed",
                details=str(e),
                exception=e,
                recoverable=True
            )
            
            # Try fallback
            if service_name in self.fallback_strategies:
                try:
                    fallback_result = self.fallback_strategies[service_name](*args, **kwargs)
                    self.error_manager.logger.info(f"Used fallback for {service_name}")
                    return True, fallback_result
                except Exception as fallback_error:
                    self.error_manager.log_error(
                        severity=ErrorSeverity.CRITICAL,
                        category=ErrorCategory.SYSTEM,
                        message=f"Fallback failed for {service_name}",
                        details=str(fallback_error),
                        exception=fallback_error,
                        recoverable=False
                    )
                    raise
            else:
                self.error_manager.logger.warning(f"No fallback registered for {service_name}")
                raise


class PartialResultRecovery:
    """Handles partial result recovery and download"""
    
    def __init__(self, error_manager: ErrorManager):
        self.error_manager = error_manager
        self.partial_results: Dict[str, Dict[str, Any]] = {}
    
    def save_partial_result(
        self,
        design_id: str,
        stage: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save partial result for a design stage"""
        if design_id not in self.partial_results:
            self.partial_results[design_id] = {}
        
        self.partial_results[design_id][stage] = {
            'result': result,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        self.error_manager.logger.info(
            f"Saved partial result for design {design_id}, stage {stage}"
        )
    
    def get_partial_results(self, design_id: str) -> Dict[str, Any]:
        """Get all partial results for a design"""
        return self.partial_results.get(design_id, {})
    
    def has_partial_results(self, design_id: str) -> bool:
        """Check if design has any partial results"""
        return design_id in self.partial_results and len(self.partial_results[design_id]) > 0
    
    def get_completed_stages(self, design_id: str) -> List[str]:
        """Get list of completed stages for a design"""
        if design_id not in self.partial_results:
            return []
        return list(self.partial_results[design_id].keys())
    
    def clear_partial_results(self, design_id: str):
        """Clear partial results for a design"""
        if design_id in self.partial_results:
            del self.partial_results[design_id]
            self.error_manager.logger.info(f"Cleared partial results for design {design_id}")


# Global error manager instance
_error_manager = None


def get_error_manager() -> ErrorManager:
    """Get global error manager instance"""
    global _error_manager
    if _error_manager is None:
        _error_manager = ErrorManager()
    return _error_manager
