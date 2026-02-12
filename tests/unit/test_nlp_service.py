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