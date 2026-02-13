"""
Tests for User Error Communication Service
"""

import pytest
import json
from src.services.user_error_communication import (
    UserErrorCommunicator,
    ErrorType,
    UserErrorMessage,
    ProgressiveDisclosure
)


@pytest.fixture
def communicator():
    """Create UserErrorCommunicator instance"""
    return UserErrorCommunicator()


@pytest.fixture
def progressive_disclosure():
    """Create ProgressiveDisclosure instance"""
    return ProgressiveDisclosure()


def test_generate_invalid_input_error(communicator):
    """Test generating invalid input error message"""
    error_msg = communicator.generate_error_message(ErrorType.INVALID_INPUT)
    
    assert error_msg.title == "Invalid Input"
    assert "couldn't understand" in error_msg.message
    assert len(error_msg.corrective_actions) > 0
    assert error_msg.can_retry is False


def test_generate_component_not_found_error(communicator):
    """Test generating component not found error"""
    error_msg = communicator.generate_error_message(
        ErrorType.COMPONENT_NOT_FOUND,
        context={'component_name': 'XYZ123'}
    )
    
    assert error_msg.title == "Component Not Available"
    assert "XYZ123" in error_msg.message
    assert len(error_msg.corrective_actions) > 0


def test_generate_generation_failed_error(communicator):
    """Test generating generation failed error"""
    error_msg = communicator.generate_error_message(ErrorType.GENERATION_FAILED)
    
    assert error_msg.title == "Design Generation Failed"
    assert len(error_msg.corrective_actions) > 0
    assert error_msg.can_retry is True


def test_generate_verification_failed_error(communicator):
    """Test generating verification failed error"""
    error_msg = communicator.generate_error_message(ErrorType.VERIFICATION_FAILED)
    
    assert error_msg.title == "Design Verification Issues"
    assert "verification errors" in error_msg.message
    assert len(error_msg.corrective_actions) > 0


def test_generate_simulation_failed_error(communicator):
    """Test generating simulation failed error"""
    error_msg = communicator.generate_error_message(ErrorType.SIMULATION_FAILED)
    
    assert error_msg.title == "Simulation Error"
    assert "simulation" in error_msg.message.lower()
    assert len(error_msg.corrective_actions) > 0


def test_generate_export_failed_error(communicator):
    """Test generating export failed error"""
    error_msg = communicator.generate_error_message(ErrorType.EXPORT_FAILED)
    
    assert error_msg.title == "File Export Error"
    assert "export" in error_msg.message.lower()
    assert len(error_msg.corrective_actions) > 0


def test_generate_service_unavailable_error(communicator):
    """Test generating service unavailable error"""
    error_msg = communicator.generate_error_message(ErrorType.SERVICE_UNAVAILABLE)
    
    assert error_msg.title == "Service Temporarily Unavailable"
    assert "unavailable" in error_msg.message.lower()
    assert error_msg.can_retry is True


def test_generate_timeout_error(communicator):
    """Test generating timeout error"""
    error_msg = communicator.generate_error_message(ErrorType.TIMEOUT)
    
    assert error_msg.title == "Request Timeout"
    assert "longer than expected" in error_msg.message
    assert error_msg.can_retry is True


def test_generate_resource_limit_error(communicator):
    """Test generating resource limit error"""
    error_msg = communicator.generate_error_message(ErrorType.RESOURCE_LIMIT)
    
    assert error_msg.title == "Resource Limit Exceeded"
    assert "exceeds" in error_msg.message
    assert error_msg.can_retry is False


def test_error_with_technical_details(communicator):
    """Test error message with technical details"""
    error_msg = communicator.generate_error_message(
        ErrorType.GENERATION_FAILED,
        technical_details="ValueError: Invalid component specification"
    )
    
    assert error_msg.technical_details == "ValueError: Invalid component specification"


def test_error_with_partial_results(communicator):
    """Test error message with partial results flag"""
    error_msg = communicator.generate_error_message(
        ErrorType.GENERATION_FAILED,
        partial_results=True
    )
    
    assert error_msg.partial_results_available is True


def test_error_with_context(communicator):
    """Test error message personalization with context"""
    error_msg = communicator.generate_error_message(
        ErrorType.GENERATION_FAILED,
        context={'stage': 'PCB Layout', 'component_name': 'U1'}
    )
    
    assert "PCB Layout" in error_msg.message
    assert "U1" in error_msg.message


def test_support_link_generation(communicator):
    """Test support link generation"""
    error_msg = communicator.generate_error_message(ErrorType.INVALID_INPUT)
    
    assert error_msg.support_link is not None
    assert "invalid_input" in error_msg.support_link


def test_categorize_invalid_input_error(communicator):
    """Test categorizing invalid input errors"""
    error_type = communicator.categorize_error(
        "Invalid prompt format",
        "nlp_parsing"
    )
    
    assert error_type == ErrorType.INVALID_INPUT


def test_categorize_component_error(communicator):
    """Test categorizing component errors"""
    error_type = communicator.categorize_error(
        "Component not found in database",
        "component_selection"
    )
    
    assert error_type == ErrorType.COMPONENT_NOT_FOUND


def test_categorize_generation_error(communicator):
    """Test categorizing generation errors"""
    error_type = communicator.categorize_error(
        "Code generation failed",
        "code_generation"
    )
    
    assert error_type == ErrorType.GENERATION_FAILED


def test_categorize_verification_error(communicator):
    """Test categorizing verification errors"""
    error_type = communicator.categorize_error(
        "ERC check failed",
        "verification"
    )
    
    assert error_type == ErrorType.VERIFICATION_FAILED


def test_categorize_simulation_error(communicator):
    """Test categorizing simulation errors"""
    error_type = communicator.categorize_error(
        "SPICE simulation error",
        "simulation"
    )
    
    assert error_type == ErrorType.SIMULATION_FAILED


def test_categorize_timeout_error(communicator):
    """Test categorizing timeout errors"""
    error_type = communicator.categorize_error(
        "Request timeout after 30 seconds",
        "system"
    )
    
    assert error_type == ErrorType.TIMEOUT


def test_categorize_resource_limit_error(communicator):
    """Test categorizing resource limit errors"""
    error_type = communicator.categorize_error(
        "Design exceeds maximum component limit",
        "system"
    )
    
    assert error_type == ErrorType.RESOURCE_LIMIT


def test_format_html(communicator):
    """Test HTML formatting"""
    error_msg = communicator.generate_error_message(ErrorType.INVALID_INPUT)
    html = communicator.format_for_display(error_msg, 'html')
    
    assert '<div class="error-message">' in html
    assert error_msg.title in html
    assert error_msg.message in html
    assert '<ul>' in html


def test_format_text(communicator):
    """Test plain text formatting"""
    error_msg = communicator.generate_error_message(ErrorType.INVALID_INPUT)
    text = communicator.format_for_display(error_msg, 'text')
    
    assert error_msg.title in text
    assert error_msg.message in text
    assert "What you can do:" in text


def test_format_json(communicator):
    """Test JSON formatting"""
    error_msg = communicator.generate_error_message(ErrorType.INVALID_INPUT)
    json_str = communicator.format_for_display(error_msg, 'json')
    
    data = json.loads(json_str)
    assert data['title'] == error_msg.title
    assert data['message'] == error_msg.message
    assert 'corrective_actions' in data


def test_format_with_partial_results(communicator):
    """Test formatting with partial results"""
    error_msg = communicator.generate_error_message(
        ErrorType.GENERATION_FAILED,
        partial_results=True
    )
    html = communicator.format_for_display(error_msg, 'html')
    
    assert "partial results" in html.lower()


def test_format_with_recovery_steps(communicator):
    """Test formatting with recovery steps"""
    error_msg = communicator.generate_error_message(ErrorType.GENERATION_FAILED)
    html = communicator.format_for_display(error_msg, 'html')
    
    if error_msg.recovery_steps:
        assert "Recovery options:" in html


def test_progressive_disclosure_basic(progressive_disclosure, communicator):
    """Test basic level progressive disclosure"""
    error_msg = communicator.generate_error_message(
        ErrorType.GENERATION_FAILED,
        technical_details="Technical error details"
    )
    
    details = progressive_disclosure.get_error_details(error_msg, 'basic')
    
    assert 'title' in details
    assert 'message' in details
    assert 'corrective_actions' in details
    assert 'technical_details' not in details


def test_progressive_disclosure_intermediate(progressive_disclosure, communicator):
    """Test intermediate level progressive disclosure"""
    error_msg = communicator.generate_error_message(
        ErrorType.GENERATION_FAILED,
        technical_details="Technical error details"
    )
    
    details = progressive_disclosure.get_error_details(error_msg, 'intermediate')
    
    assert 'title' in details
    assert 'message' in details
    assert 'corrective_actions' in details
    assert 'recovery_steps' in details
    assert 'can_retry' in details
    assert 'technical_details' not in details


def test_progressive_disclosure_advanced(progressive_disclosure, communicator):
    """Test advanced level progressive disclosure"""
    error_msg = communicator.generate_error_message(
        ErrorType.GENERATION_FAILED,
        technical_details="Technical error details"
    )
    
    details = progressive_disclosure.get_error_details(error_msg, 'advanced')
    
    assert 'title' in details
    assert 'message' in details
    assert 'corrective_actions' in details
    assert 'recovery_steps' in details
    assert 'technical_details' in details
    assert 'support_link' in details


def test_progressive_disclosure_invalid_level(progressive_disclosure, communicator):
    """Test progressive disclosure with invalid level defaults to basic"""
    error_msg = communicator.generate_error_message(ErrorType.GENERATION_FAILED)
    
    details = progressive_disclosure.get_error_details(error_msg, 'invalid')
    
    assert 'title' in details
    assert 'message' in details
    assert 'technical_details' not in details


def test_user_error_message_dataclass():
    """Test UserErrorMessage dataclass"""
    msg = UserErrorMessage(
        title="Test Error",
        message="Test message",
        corrective_actions=["Action 1", "Action 2"],
        can_retry=True,
        partial_results_available=False
    )
    
    assert msg.title == "Test Error"
    assert msg.message == "Test message"
    assert len(msg.corrective_actions) == 2
    assert msg.can_retry is True


def test_all_error_types_have_templates(communicator):
    """Test that all error types have templates"""
    for error_type in ErrorType:
        error_msg = communicator.generate_error_message(error_type)
        assert error_msg.title is not None
        assert error_msg.message is not None
        assert len(error_msg.corrective_actions) > 0


def test_generic_error_fallback(communicator):
    """Test generic error message generation"""
    # This tests the fallback when error type is not in templates
    error_msg = communicator._generate_generic_error("Unknown error")
    
    assert error_msg.title == "Unexpected Error"
    assert "unexpected error" in error_msg.message.lower()
    assert len(error_msg.corrective_actions) > 0
