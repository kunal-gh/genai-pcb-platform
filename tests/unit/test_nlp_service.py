"""
Unit tests for NLP Service.

Tests the natural language prompt parsing functionality.
"""

import pytest
from src.services.nlp_service import NLPService, StructuredRequirements


@pytest.fixture
def nlp_service():
    """Create NLP service instance for testing."""
    return NLPService()


def test_parse_simple_led_circuit(nlp_service):
    """Test parsing a simple LED circuit prompt."""
    prompt = "Design a 40x20mm PCB with a 9V battery connector, a 5mm LED indicator, and a 220-ohm resistor inline"
    
    result = nlp_service.parse_prompt(prompt)
    
    assert isinstance(result, StructuredRequirements)
    assert result.board.width_mm == 40.0
    assert result.board.height_mm == 20.0
    assert result.power.battery_type == "9V"
    assert result.power.voltage == 9.0
    assert len(result.components) >= 2  # LED and resistor
    assert result.confidence_score > 0.5


def test_parse_board_dimensions(nlp_service):
    """Test parsing board dimensions."""
    prompt = "Create a 100x50mm PCB"
    
    result = nlp_service.parse_prompt(prompt)
    
    assert result.board.width_mm == 100.0
    assert result.board.height_mm == 50.0


def test_parse_multilayer_board(nlp_service):
    """Test parsing multilayer board specification."""
    prompt = "Design a 4-layer PCB with components"
    
    result = nlp_service.parse_prompt(prompt)
    
    assert result.board.layers == 4


def test_parse_resistor_value(nlp_service):
    """Test parsing resistor with value."""
    prompt = "Add a 220-ohm resistor and a 1k resistor"
    
    result = nlp_service.parse_prompt(prompt)
    
    resistors = [c for c in result.components if c.type == "RESISTOR"]
    assert len(resistors) >= 1
    # Check that at least one resistor has a value
    assert any(c.value for c in resistors)


def test_parse_usb_power(nlp_service):
    """Test parsing USB power supply."""
    prompt = "Design a USB-powered circuit"
    
    result = nlp_service.parse_prompt(prompt)
    
    assert result.power.type == "usb"
    assert result.power.voltage == 5.0


def test_parse_multiple_components(nlp_service):
    """Test parsing multiple different components."""
    prompt = "PCB with LED, resistor, capacitor, and transistor"
    
    result = nlp_service.parse_prompt(prompt)
    
    component_types = {c.type for c in result.components}
    assert "LED" in component_types
    assert "RESISTOR" in component_types
    assert "CAPACITOR" in component_types
    assert "TRANSISTOR" in component_types


def test_parse_compact_priority(nlp_service):
    """Test parsing compact design priority."""
    prompt = "Design a compact PCB with minimal size"
    
    result = nlp_service.parse_prompt(prompt)
    
    assert result.constraints.priority == "compact"


def test_parse_cost_priority(nlp_service):
    """Test parsing cost-optimized design."""
    prompt = "Design a low cost budget PCB"
    
    result = nlp_service.parse_prompt(prompt)
    
    assert result.constraints.priority == "cost"


def test_detect_missing_dimensions(nlp_service):
    """Test detection of missing board dimensions."""
    prompt = "Design a PCB with LED and resistor"
    
    result = nlp_service.parse_prompt(prompt)
    
    assert "Board dimensions not specified" in result.ambiguities


def test_detect_missing_component_values(nlp_service):
    """Test detection of missing component values."""
    prompt = "Add a resistor to the circuit"
    
    result = nlp_service.parse_prompt(prompt)
    
    # Should detect that resistor value is not specified
    assert any("RESISTOR value not specified" in amb for amb in result.ambiguities)


def test_confidence_score_complete_prompt(nlp_service):
    """Test confidence score for complete prompt."""
    prompt = "Design a 50x30mm PCB with 9V battery, 220-ohm resistor, and LED"
    
    result = nlp_service.parse_prompt(prompt)
    
    # Should have high confidence with all info present
    assert result.confidence_score > 0.7


def test_confidence_score_incomplete_prompt(nlp_service):
    """Test confidence score for incomplete prompt."""
    prompt = "Add some components"
    
    result = nlp_service.parse_prompt(prompt)
    
    # Should have low confidence with missing info
    assert result.confidence_score < 0.5


def test_validate_prompt_too_short(nlp_service):
    """Test validation rejects too-short prompts."""
    is_valid, error = nlp_service.validate_prompt("short")
    
    assert not is_valid
    assert "too short" in error.lower()


def test_validate_prompt_too_long(nlp_service):
    """Test validation rejects too-long prompts."""
    long_prompt = "x" * 10001
    is_valid, error = nlp_service.validate_prompt(long_prompt)
    
    assert not is_valid
    assert "too long" in error.lower()


def test_validate_prompt_no_components(nlp_service):
    """Test validation rejects prompts without components."""
    is_valid, error = nlp_service.validate_prompt("This is just random text without any components")
    
    assert not is_valid
    assert "no recognizable components" in error.lower()


def test_validate_prompt_valid(nlp_service):
    """Test validation accepts valid prompts."""
    is_valid, error = nlp_service.validate_prompt("Design a PCB with LED and resistor")
    
    assert is_valid
    assert error is None


def test_to_dict_conversion(nlp_service):
    """Test conversion of structured requirements to dictionary."""
    prompt = "Design a 40x20mm PCB with LED"
    
    result = nlp_service.parse_prompt(prompt)
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert "board" in result_dict
    assert "power" in result_dict
    assert "components" in result_dict
    assert "constraints" in result_dict
    assert result_dict["original_prompt"] == prompt


def test_parse_smd_package(nlp_service):
    """Test parsing SMD package specifications."""
    prompt = "Use 0805 resistors and 0603 capacitors"
    
    result = nlp_service.parse_prompt(prompt)
    
    # Check that packages were detected
    packages = [c.package for c in result.components if c.package]
    assert len(packages) > 0
    assert any(pkg in ["0805", "0603"] for pkg in packages)


def test_parse_special_requirements(nlp_service):
    """Test parsing special requirements."""
    prompt = "Design a waterproof PCB for high temperature environment"
    
    result = nlp_service.parse_prompt(prompt)
    
    assert "waterproof" in result.constraints.special_requirements
    assert "high_temperature" in result.constraints.special_requirements


def test_case_insensitive_parsing(nlp_service):
    """Test that parsing is case-insensitive."""
    prompt1 = "Design with LED and RESISTOR"
    prompt2 = "design with led and resistor"
    
    result1 = nlp_service.parse_prompt(prompt1)
    result2 = nlp_service.parse_prompt(prompt2)
    
    # Both should find the same component types
    types1 = {c.type for c in result1.components}
    types2 = {c.type for c in result2.components}
    assert types1 == types2


# ============================================================================
# Task 2.3: Input Validation and Error Handling Tests
# ============================================================================

def test_clarification_requests_missing_dimensions(nlp_service):
    """Test clarification requests when board dimensions are missing."""
    prompt = "Design a PCB with LED and resistor powered by 9V battery"
    result = nlp_service.parse_prompt(prompt)
    
    # Should have clarification request for board dimensions
    clarifications = result.clarification_requests
    assert len(clarifications) > 0
    
    dimension_clarification = next(
        (c for c in clarifications if c.field == "board_dimensions"),
        None
    )
    assert dimension_clarification is not None
    assert "dimensions" in dimension_clarification.message.lower()
    assert len(dimension_clarification.suggestions) > 0


def test_clarification_requests_missing_power(nlp_service):
    """Test clarification requests when power supply is missing."""
    prompt = "Design a 40x20mm PCB with LED and 220-ohm resistor"
    result = nlp_service.parse_prompt(prompt)
    
    # Should have clarification request for power supply
    clarifications = result.clarification_requests
    power_clarification = next(
        (c for c in clarifications if c.field == "power_supply"),
        None
    )
    assert power_clarification is not None
    assert "power" in power_clarification.message.lower()
    assert "9V battery" in power_clarification.suggestions


def test_clarification_requests_missing_component_values(nlp_service):
    """Test clarification requests for components without values."""
    prompt = "Design a 40x20mm PCB with 9V battery, LED, and resistor"
    result = nlp_service.parse_prompt(prompt)
    
    # Should have clarification for resistor value
    clarifications = result.clarification_requests
    value_clarifications = [c for c in clarifications if "value" in c.field]
    assert len(value_clarifications) > 0
    
    # Check that suggestions include common values
    resistor_clarification = next(
        (c for c in value_clarifications if "resistance" in c.message.lower()),
        None
    )
    if resistor_clarification:
        assert any("Ω" in s or "ohm" in s.lower() for s in resistor_clarification.suggestions)


def test_validation_error_prompt_too_short(nlp_service):
    """Test validation error for prompts that are too short."""
    prompt = "LED"
    result = nlp_service.parse_prompt(prompt)
    
    # Should have validation error for short prompt
    errors = result.validation_errors
    short_error = next(
        (e for e in errors if e.error_code == "PROMPT_TOO_SHORT"),
        None
    )
    assert short_error is not None
    assert "too short" in short_error.message.lower()
    assert short_error.suggestion is not None


def test_validation_error_no_components(nlp_service):
    """Test validation error when no components are detected."""
    prompt = "Design a PCB that is 40x20mm with power supply"
    result = nlp_service.parse_prompt(prompt)
    
    # Should have validation error for no components
    errors = result.validation_errors
    no_comp_error = next(
        (e for e in errors if e.error_code == "NO_COMPONENTS"),
        None
    )
    assert no_comp_error is not None
    assert "no components" in no_comp_error.message.lower()


def test_validation_error_board_too_small(nlp_service):
    """Test validation error for unreasonably small board dimensions."""
    prompt = "Design a 5x5mm PCB with LED and resistor powered by 9V battery"
    result = nlp_service.parse_prompt(prompt)
    
    # Should have validation error for small board
    errors = result.validation_errors
    small_board_error = next(
        (e for e in errors if e.error_code == "BOARD_TOO_SMALL"),
        None
    )
    assert small_board_error is not None
    assert "too small" in small_board_error.message.lower()
    assert "10x10mm" in small_board_error.suggestion


def test_validation_error_board_too_large(nlp_service):
    """Test validation error for unreasonably large board dimensions."""
    prompt = "Design a 1000x1000mm PCB with LED and resistor"
    result = nlp_service.parse_prompt(prompt)
    
    # Should have validation error for large board
    errors = result.validation_errors
    large_board_error = next(
        (e for e in errors if e.error_code == "BOARD_TOO_LARGE"),
        None
    )
    assert large_board_error is not None
    assert "large" in large_board_error.message.lower()


def test_validation_error_negative_voltage(nlp_service):
    """Test validation error for negative voltage."""
    prompt = "Design a 40x20mm PCB with LED and resistor powered by -9V"
    result = nlp_service.parse_prompt(prompt)
    
    # Should have validation error for negative voltage
    errors = result.validation_errors
    voltage_error = next(
        (e for e in errors if e.error_code == "INVALID_VOLTAGE"),
        None
    )
    assert voltage_error is not None
    assert "negative" in voltage_error.message.lower()


def test_validation_error_high_voltage_warning(nlp_service):
    """Test validation warning for high voltage."""
    prompt = "Design a 40x20mm PCB with LED and resistor powered by 100V"
    result = nlp_service.parse_prompt(prompt)
    
    # Should have validation warning for high voltage
    errors = result.validation_errors
    high_voltage_error = next(
        (e for e in errors if e.error_code == "HIGH_VOLTAGE_WARNING"),
        None
    )
    assert high_voltage_error is not None
    assert "100" in high_voltage_error.message or "100.0" in high_voltage_error.message


def test_validate_prompt_empty(nlp_service):
    """Test validation of empty prompt."""
    is_valid, error_msg = nlp_service.validate_prompt("")
    assert not is_valid
    assert "empty" in error_msg.lower()
    assert "example" in error_msg.lower()


def test_validate_prompt_whitespace_only(nlp_service):
    """Test validation of whitespace-only prompt."""
    is_valid, error_msg = nlp_service.validate_prompt("   \n\t  ")
    assert not is_valid
    assert "empty" in error_msg.lower()


def test_validate_prompt_too_short_descriptive(nlp_service):
    """Test validation of too-short prompt with descriptive error."""
    is_valid, error_msg = nlp_service.validate_prompt("LED")
    assert not is_valid
    assert "too short" in error_msg.lower()
    assert "board dimensions" in error_msg.lower()
    assert "power source" in error_msg.lower()


def test_validate_prompt_too_long(nlp_service):
    """Test validation of too-long prompt."""
    long_prompt = "Design a PCB " + "with many components " * 1000
    is_valid, error_msg = nlp_service.validate_prompt(long_prompt)
    assert not is_valid
    assert "too long" in error_msg.lower()
    assert "10,000" in error_msg


def test_validate_prompt_no_components_descriptive(nlp_service):
    """Test validation when no components found with descriptive error."""
    is_valid, error_msg = nlp_service.validate_prompt(
        "Design a PCB that is very nice and works well"
    )
    assert not is_valid
    assert "no recognizable components" in error_msg.lower()
    assert "LED" in error_msg or "RESISTOR" in error_msg


def test_validate_prompt_suspicious_patterns(nlp_service):
    """Test validation rejects suspicious patterns."""
    is_valid, error_msg = nlp_service.validate_prompt(
        "Design a PCB with <script>alert('xss')</script> LED"
    )
    assert not is_valid
    assert "invalid" in error_msg.lower()


def test_validate_prompt_valid_returns_none(nlp_service):
    """Test that valid prompt returns None for error message."""
    is_valid, error_msg = nlp_service.validate_prompt(
        "Design a 40x20mm PCB with 9V battery, LED, and 220-ohm resistor"
    )
    assert is_valid
    assert error_msg is None


def test_structured_requirements_to_dict_includes_new_fields(nlp_service):
    """Test that to_dict includes clarification_requests and validation_errors."""
    prompt = "Design a PCB with LED"  # Will trigger clarifications and errors
    result = nlp_service.parse_prompt(prompt)
    
    result_dict = result.to_dict()
    assert "clarification_requests" in result_dict
    assert "validation_errors" in result_dict
    assert isinstance(result_dict["clarification_requests"], list)
    assert isinstance(result_dict["validation_errors"], list)


def test_clarification_severity_levels(nlp_service):
    """Test that clarifications have appropriate severity levels."""
    prompt = "Design a PCB with LED and resistor"
    result = nlp_service.parse_prompt(prompt)
    
    clarifications = result.clarification_requests
    assert len(clarifications) > 0
    
    # Check that severity is one of the expected values
    for clarification in clarifications:
        assert clarification.severity in ["warning", "error", "info"]


def test_validation_error_resistor_value_low(nlp_service):
    """Test validation error for unusually low resistor value."""
    prompt = "Design a 40x20mm PCB with 9V battery, LED, and 0.1-ohm resistor"
    result = nlp_service.parse_prompt(prompt)
    
    errors = result.validation_errors
    low_value_error = next(
        (e for e in errors if e.error_code == "RESISTOR_VALUE_LOW"),
        None
    )
    assert low_value_error is not None
    assert "unusually low" in low_value_error.message.lower()


def test_validation_error_resistor_value_high(nlp_service):
    """Test validation error for unusually high resistor value."""
    prompt = "Design a 40x20mm PCB with 9V battery, LED, and 100M-ohm resistor"
    result = nlp_service.parse_prompt(prompt)
    
    errors = result.validation_errors
    high_value_error = next(
        (e for e in errors if e.error_code == "RESISTOR_VALUE_HIGH"),
        None
    )
    assert high_value_error is not None
    assert "unusually high" in high_value_error.message.lower()


def test_multiple_clarifications_and_errors(nlp_service):
    """Test that multiple clarifications and errors can coexist."""
    prompt = "Design a PCB with LED"  # Missing dimensions, power, values
    result = nlp_service.parse_prompt(prompt)
    
    # Should have multiple clarifications
    assert len(result.clarification_requests) >= 2
    
    # Should have validation error for short prompt
    assert len(result.validation_errors) >= 1
    
    # Confidence should be low
    assert result.confidence_score < 0.5


def test_clarification_suggestions_are_helpful(nlp_service):
    """Test that clarification suggestions provide helpful options."""
    prompt = "Design a 40x20mm PCB with LED and resistor"
    result = nlp_service.parse_prompt(prompt)
    
    # Find power supply clarification
    power_clarification = next(
        (c for c in result.clarification_requests if c.field == "power_supply"),
        None
    )
    
    if power_clarification:
        # Should have multiple suggestions
        assert len(power_clarification.suggestions) >= 3
        # Should include common options
        suggestions_text = " ".join(power_clarification.suggestions).lower()
        assert "battery" in suggestions_text or "usb" in suggestions_text
