"""
Property-based tests for NLP Service.

Uses Hypothesis to generate thousands of test cases and validate
universal properties of the natural language parser.

Feature: genai-pcb-platform
Property 1: Natural Language Parsing Completeness
Property 2: Input Validation and Error Handling
Property 3: Prompt Length Handling
Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.strategies import composite
import re

from src.services.nlp_service import NLPService, StructuredRequirements


# Custom strategies for generating valid PCB design prompts

@composite
def board_dimensions(draw):
    """Generate valid board dimensions."""
    width = draw(st.floats(min_value=10.0, max_value=500.0))
    height = draw(st.floats(min_value=10.0, max_value=500.0))
    return f"{width:.1f}x{height:.1f}mm"


@composite
def component_types(draw):
    """Generate component type mentions."""
    components = ["LED", "resistor", "capacitor", "transistor", "diode", "IC", "connector"]
    return draw(st.sampled_from(components))


@composite
def resistor_values(draw):
    """Generate resistor values."""
    value = draw(st.floats(min_value=1.0, max_value=10000.0))
    unit = draw(st.sampled_from(["ohm", "Ω", "R"]))
    prefix = draw(st.sampled_from(["", "k", "K", "M"]))
    return f"{value:.0f}{prefix}{unit}"


@composite
def voltage_values(draw):
    """Generate voltage values."""
    value = draw(st.floats(min_value=1.0, max_value=48.0))
    return f"{value:.1f}V"


@composite
def valid_pcb_prompt(draw):
    """Generate a valid PCB design prompt with all required elements."""
    dimensions = draw(board_dimensions())
    component = draw(component_types())
    voltage = draw(voltage_values())
    
    templates = [
        f"Design a {dimensions} PCB with {voltage} power and {component}",
        f"Create a {dimensions} board with {component} powered by {voltage}",
        f"Build a {dimensions} PCB including {component} using {voltage}",
    ]
    
    return draw(st.sampled_from(templates))


@composite
def prompt_with_components(draw):
    """Generate prompts with various components."""
    dimensions = draw(board_dimensions())
    num_components = draw(st.integers(min_value=1, max_value=5))
    components = draw(st.lists(component_types(), min_size=num_components, max_size=num_components))
    
    component_list = ", ".join(components[:-1]) + f" and {components[-1]}" if len(components) > 1 else components[0]
    
    return f"Design a {dimensions} PCB with {component_list}"


# Property 1: Natural Language Parsing Completeness
# For any valid natural language prompt describing PCB requirements,
# parsing should produce structured JSON containing component requirements,
# connections, and design constraints with all essential fields populated.

@pytest.mark.property
@given(prompt=valid_pcb_prompt())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_parsing_produces_structured_output(prompt):
    """
    Property 1: Natural Language Parsing Completeness
    
    For any valid prompt, parsing should produce a StructuredRequirements
    object with all essential fields populated.
    
    Validates: Requirements 1.1, 1.2
    """
    nlp = NLPService()
    
    result = nlp.parse_prompt(prompt)
    
    # Property: Result must be StructuredRequirements instance
    assert isinstance(result, StructuredRequirements)
    
    # Property: Must have board specification
    assert result.board is not None
    
    # Property: Must have power specification
    assert result.power is not None
    
    # Property: Must have components list (even if empty)
    assert isinstance(result.components, list)
    
    # Property: Must have constraints
    assert result.constraints is not None
    
    # Property: Must preserve original prompt
    assert result.original_prompt == prompt
    
    # Property: Confidence score must be between 0 and 1
    assert 0.0 <= result.confidence_score <= 1.0
    
    # Property: Ambiguities must be a list
    assert isinstance(result.ambiguities, list)


@pytest.mark.property
@given(prompt=prompt_with_components())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_component_extraction(prompt):
    """
    Property 1: Natural Language Parsing Completeness
    
    For any prompt containing component keywords, at least one component
    should be extracted.
    
    Validates: Requirements 1.2
    """
    nlp = NLPService()
    
    result = nlp.parse_prompt(prompt)
    
    # Property: If prompt contains component keywords, components should be found
    component_keywords = ["led", "resistor", "capacitor", "transistor", "diode", "ic", "connector"]
    prompt_lower = prompt.lower()
    
    if any(keyword in prompt_lower for keyword in component_keywords):
        # At least one component should be extracted
        assert len(result.components) > 0, f"No components extracted from: {prompt}"


@pytest.mark.property
@given(
    width=st.floats(min_value=10.0, max_value=500.0),
    height=st.floats(min_value=10.0, max_value=500.0)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_board_dimension_extraction(width, height):
    """
    Property 1: Natural Language Parsing Completeness
    
    For any prompt with explicit board dimensions in WxH format,
    those dimensions should be correctly extracted.
    
    Validates: Requirements 1.2
    """
    nlp = NLPService()
    
    prompt = f"Design a {width:.1f}x{height:.1f}mm PCB with components"
    result = nlp.parse_prompt(prompt)
    
    # Property: Dimensions should be extracted correctly (within floating point tolerance)
    if result.board.width_mm is not None:
        assert abs(result.board.width_mm - width) < 0.2, \
            f"Width mismatch: expected {width}, got {result.board.width_mm}"
    
    if result.board.height_mm is not None:
        assert abs(result.board.height_mm - height) < 0.2, \
            f"Height mismatch: expected {height}, got {result.board.height_mm}"


# Property 2: Input Validation and Error Handling
# For any invalid or ambiguous natural language prompt, the system should
# either request clarification for ambiguous inputs or return descriptive
# error messages for invalid inputs, never proceeding with incomplete information.

@pytest.mark.property
@given(prompt=st.text(min_size=0, max_size=9))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_reject_too_short_prompts(prompt):
    """
    Property 2: Input Validation and Error Handling
    
    For any prompt shorter than 10 characters, validation should fail
    with a descriptive error message.
    
    Validates: Requirements 1.3, 1.4
    """
    nlp = NLPService()
    
    is_valid, error = nlp.validate_prompt(prompt)
    
    # Property: Short prompts must be rejected
    assert not is_valid, f"Short prompt was accepted: '{prompt}'"
    
    # Property: Error message must be provided
    assert error is not None
    assert "short" in error.lower()


@pytest.mark.property
@given(prompt=st.text(min_size=10001, max_size=11000))
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_reject_too_long_prompts(prompt):
    """
    Property 3: Prompt Length Handling
    
    For any prompt longer than 10,000 characters, validation should fail
    with a descriptive error message.
    
    Validates: Requirements 1.5
    """
    nlp = NLPService()
    
    is_valid, error = nlp.validate_prompt(prompt)
    
    # Property: Long prompts must be rejected
    assert not is_valid, f"Long prompt was accepted (length: {len(prompt)})"
    
    # Property: Error message must be provided
    assert error is not None
    assert "long" in error.lower()


@pytest.mark.property
@given(prompt=st.text(min_size=10, max_size=100).filter(
    lambda s: not any(word in s.lower() for word in 
                     ["led", "resistor", "capacitor", "transistor", "diode", "ic", "connector", 
                      "battery", "switch", "inductor", "crystal", "fuse"])
))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_reject_prompts_without_components(prompt):
    """
    Property 2: Input Validation and Error Handling
    
    For any prompt without recognizable components, validation should fail
    with a descriptive error message.
    
    Validates: Requirements 1.4
    """
    nlp = NLPService()
    
    is_valid, error = nlp.validate_prompt(prompt)
    
    # Property: Prompts without components must be rejected
    assert not is_valid, f"Prompt without components was accepted: '{prompt}'"
    
    # Property: Error message must mention components
    assert error is not None
    assert "component" in error.lower()


@pytest.mark.property
@given(prompt=valid_pcb_prompt())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_valid_prompts_accepted(prompt):
    """
    Property 2: Input Validation and Error Handling
    
    For any valid prompt (10-10000 chars with components), validation
    should succeed.
    
    Validates: Requirements 1.3, 1.5
    """
    nlp = NLPService()
    
    is_valid, error = nlp.validate_prompt(prompt)
    
    # Property: Valid prompts must be accepted
    assert is_valid, f"Valid prompt was rejected: '{prompt}' - Error: {error}"
    
    # Property: No error message for valid prompts
    assert error is None


# Property 3: Prompt Length Handling
# For any natural language prompt between 10 and 1000 words, the system
# should successfully parse and process the input without length-related failures.

@pytest.mark.property
@given(
    num_words=st.integers(min_value=10, max_value=1000),
    component=component_types()
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_handle_various_prompt_lengths(num_words, component):
    """
    Property 3: Prompt Length Handling
    
    For any prompt between 10 and 1000 words, parsing should succeed
    without length-related failures.
    
    Validates: Requirements 1.5
    """
    nlp = NLPService()
    
    # Generate a prompt with approximately num_words words
    words = ["Design", "a", "PCB", "with", component] + ["component"] * (num_words - 5)
    prompt = " ".join(words)
    
    # Ensure it's within character limits
    assume(10 <= len(prompt) <= 10000)
    
    # Property: Parsing should not raise exceptions
    try:
        result = nlp.parse_prompt(prompt)
        
        # Property: Should return valid StructuredRequirements
        assert isinstance(result, StructuredRequirements)
        
        # Property: Should preserve original prompt
        assert result.original_prompt == prompt
        
    except Exception as e:
        pytest.fail(f"Parsing failed for {num_words}-word prompt: {str(e)}")


@pytest.mark.property
@given(prompt=valid_pcb_prompt())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_confidence_score_bounds(prompt):
    """
    Property: Confidence Score Bounds
    
    For any prompt, the confidence score must always be between 0.0 and 1.0.
    
    Validates: Requirements 1.2
    """
    nlp = NLPService()
    
    result = nlp.parse_prompt(prompt)
    
    # Property: Confidence must be in valid range
    assert 0.0 <= result.confidence_score <= 1.0, \
        f"Confidence score {result.confidence_score} out of bounds for prompt: {prompt}"


@pytest.mark.property
@given(prompt=valid_pcb_prompt())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_to_dict_serialization(prompt):
    """
    Property: Serialization Completeness
    
    For any parsed result, to_dict() should produce a valid dictionary
    with all required fields.
    
    Validates: Requirements 1.2
    """
    nlp = NLPService()
    
    result = nlp.parse_prompt(prompt)
    result_dict = result.to_dict()
    
    # Property: Must be a dictionary
    assert isinstance(result_dict, dict)
    
    # Property: Must contain all required top-level keys
    required_keys = ["board", "power", "components", "constraints", 
                     "connections", "original_prompt", "confidence_score", "ambiguities"]
    
    for key in required_keys:
        assert key in result_dict, f"Missing required key: {key}"
    
    # Property: Original prompt must be preserved
    assert result_dict["original_prompt"] == prompt


@pytest.mark.property
@given(
    prompt1=valid_pcb_prompt(),
    prompt2=valid_pcb_prompt()
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_parsing_determinism(prompt1, prompt2):
    """
    Property: Parsing Determinism
    
    For any prompt, parsing it multiple times should produce identical results.
    
    Validates: Requirements 1.1, 1.2
    """
    nlp = NLPService()
    
    # Parse the same prompt twice
    result1a = nlp.parse_prompt(prompt1)
    result1b = nlp.parse_prompt(prompt1)
    
    # Property: Results should be identical
    assert result1a.to_dict() == result1b.to_dict(), \
        f"Non-deterministic parsing for prompt: {prompt1}"


@pytest.mark.property
@given(prompt=valid_pcb_prompt())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_case_insensitivity(prompt):
    """
    Property: Case Insensitivity
    
    For any prompt, parsing should be case-insensitive (uppercase, lowercase,
    or mixed case should produce equivalent results).
    
    Validates: Requirements 1.1
    """
    nlp = NLPService()
    
    result_original = nlp.parse_prompt(prompt)
    result_lower = nlp.parse_prompt(prompt.lower())
    result_upper = nlp.parse_prompt(prompt.upper())
    
    # Property: Component types should be the same regardless of case
    types_original = {c.type for c in result_original.components}
    types_lower = {c.type for c in result_lower.components}
    types_upper = {c.type for c in result_upper.components}
    
    assert types_original == types_lower == types_upper, \
        f"Case-sensitive parsing detected for prompt: {prompt}"