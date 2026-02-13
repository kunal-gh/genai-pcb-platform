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



# ============================================================================
# Property 2: Input Validation and Error Handling
# Feature: genai-pcb-platform
# Validates: Requirements 1.3, 1.4, 1.5
# ============================================================================

@given(st.text(min_size=1, max_size=50))
@settings(max_examples=100)
def test_property_validation_errors_have_required_fields(prompt):
    """
    Property 2: Input Validation and Error Handling
    
    For any prompt that triggers validation errors, each error SHALL have:
    - A field identifier
    - A descriptive message
    - An optional suggestion
    - An error code
    
    Validates: Requirements 1.3, 1.4
    """
    nlp = NLPService()
    result = nlp.parse_prompt(prompt)
    
    # Check all validation errors have required fields
    for error in result.validation_errors:
        assert hasattr(error, 'field'), "Validation error must have 'field'"
        assert hasattr(error, 'message'), "Validation error must have 'message'"
        assert hasattr(error, 'suggestion'), "Validation error must have 'suggestion'"
        assert hasattr(error, 'error_code'), "Validation error must have 'error_code'"
        
        # Field and message must be non-empty strings
        assert isinstance(error.field, str) and len(error.field) > 0
        assert isinstance(error.message, str) and len(error.message) > 0
        
        # Error code should be uppercase with underscores if present
        if error.error_code:
            assert isinstance(error.error_code, str)
            assert error.error_code.isupper() or '_' in error.error_code


@given(st.text(min_size=1, max_size=50))
@settings(max_examples=100)
def test_property_clarification_requests_have_required_fields(prompt):
    """
    Property 2: Input Validation and Error Handling
    
    For any prompt that triggers clarification requests, each request SHALL have:
    - A field identifier
    - A user-friendly message
    - A list of suggestions (may be empty)
    - A severity level (warning, error, info)
    
    Validates: Requirements 1.3, 1.4
    """
    nlp = NLPService()
    result = nlp.parse_prompt(prompt)
    
    # Check all clarification requests have required fields
    for clarification in result.clarification_requests:
        assert hasattr(clarification, 'field'), "Clarification must have 'field'"
        assert hasattr(clarification, 'message'), "Clarification must have 'message'"
        assert hasattr(clarification, 'suggestions'), "Clarification must have 'suggestions'"
        assert hasattr(clarification, 'severity'), "Clarification must have 'severity'"
        
        # Field and message must be non-empty strings
        assert isinstance(clarification.field, str) and len(clarification.field) > 0
        assert isinstance(clarification.message, str) and len(clarification.message) > 0
        
        # Suggestions must be a list
        assert isinstance(clarification.suggestions, list)
        
        # Severity must be one of the expected values
        assert clarification.severity in ['warning', 'error', 'info']


@given(st.integers(min_value=-1000, max_value=1000))
@settings(max_examples=100)
def test_property_negative_voltage_always_rejected(voltage):
    """
    Property 2: Input Validation and Error Handling
    
    For any negative voltage value in a prompt, the validation SHALL:
    - Detect the negative voltage
    - Generate a validation error with code INVALID_VOLTAGE
    - Provide a suggestion for positive voltage values
    
    Validates: Requirements 1.3, 1.4
    """
    if voltage >= 0:
        return  # Skip non-negative voltages
    
    nlp = NLPService()
    prompt = f"Design a 40x20mm PCB with LED and resistor powered by {voltage}V"
    result = nlp.parse_prompt(prompt)
    
    # Should have at least one INVALID_VOLTAGE error
    voltage_errors = [e for e in result.validation_errors if e.error_code == "INVALID_VOLTAGE"]
    assert len(voltage_errors) > 0, f"Negative voltage {voltage}V should trigger INVALID_VOLTAGE error"
    
    # Error message should mention negative or invalid
    for error in voltage_errors:
        assert "negative" in error.message.lower() or "invalid" in error.message.lower()
        assert error.suggestion is not None


@given(st.integers(min_value=5, max_value=20), st.integers(min_value=5, max_value=20))
@settings(max_examples=100)
def test_property_small_board_dimensions_flagged(width, height):
    """
    Property 2: Input Validation and Error Handling
    
    For any board dimensions below 10mm, the validation SHALL:
    - Detect the small dimensions
    - Generate a validation error with code BOARD_TOO_SMALL
    - Suggest minimum board size of 10x10mm
    
    Validates: Requirements 1.3, 1.4
    """
    if width >= 10 and height >= 10:
        return  # Skip valid dimensions
    
    nlp = NLPService()
    prompt = f"Design a {width}x{height}mm PCB with LED and resistor powered by 9V battery"
    result = nlp.parse_prompt(prompt)
    
    # Should have BOARD_TOO_SMALL error
    small_board_errors = [e for e in result.validation_errors if e.error_code == "BOARD_TOO_SMALL"]
    assert len(small_board_errors) > 0, f"Board {width}x{height}mm should trigger BOARD_TOO_SMALL error"
    
    # Error should mention minimum size
    for error in small_board_errors:
        assert "10" in error.suggestion or "10x10" in error.suggestion


# ============================================================================
# Property 3: Prompt Length Handling
# Feature: genai-pcb-platform
# Validates: Requirements 1.5
# ============================================================================

@given(st.integers(min_value=0, max_value=5000))
@settings(max_examples=100)
def test_property_prompt_length_validation(word_count):
    """
    Property 3: Prompt Length Handling
    
    For any prompt:
    - Prompts with <10 words SHALL be rejected with PROMPT_TOO_SHORT error
    - Prompts with >1000 words SHALL be rejected with PROMPT_TOO_LONG error
    - Prompts with 10-1000 words SHALL pass length validation
    
    Validates: Requirement 1.5
    """
    nlp = NLPService()
    
    # Generate a prompt with the specified word count
    # Use "LED" as a component word to pass component validation
    words = ["LED"] + ["component"] * (word_count - 1) if word_count > 0 else []
    prompt = " ".join(words)
    
    result = nlp.parse_prompt(prompt)
    
    if word_count < 10:
        # Should have PROMPT_TOO_SHORT error
        short_errors = [e for e in result.validation_errors if e.error_code == "PROMPT_TOO_SHORT"]
        assert len(short_errors) > 0, f"Prompt with {word_count} words should trigger PROMPT_TOO_SHORT"
        
        # Error should provide helpful suggestion
        for error in short_errors:
            assert error.suggestion is not None
            assert "example" in error.suggestion.lower() or "include" in error.suggestion.lower()
    
    elif word_count > 1000:
        # Should have PROMPT_TOO_LONG error
        long_errors = [e for e in result.validation_errors if e.error_code == "PROMPT_TOO_LONG"]
        assert len(long_errors) > 0, f"Prompt with {word_count} words should trigger PROMPT_TOO_LONG"
        
        # Error should suggest breaking down the design
        for error in long_errors:
            assert error.suggestion is not None
            assert "break" in error.suggestion.lower() or "smaller" in error.suggestion.lower()
    
    else:
        # Should NOT have prompt length errors
        length_errors = [e for e in result.validation_errors 
                        if e.error_code in ["PROMPT_TOO_SHORT", "PROMPT_TOO_LONG"]]
        assert len(length_errors) == 0, f"Prompt with {word_count} words should not trigger length errors"


@given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Z')), 
               min_size=0, max_size=100))
@settings(max_examples=100)
def test_property_empty_prompts_rejected(text):
    """
    Property 3: Prompt Length Handling
    
    For any prompt that is empty or contains only whitespace:
    - validate_prompt() SHALL return False
    - Error message SHALL explain that prompt cannot be empty
    - Error message SHALL provide an example
    
    Validates: Requirement 1.5
    """
    if text.strip():  # Skip non-empty prompts
        return
    
    nlp = NLPService()
    is_valid, error_msg = nlp.validate_prompt(text)
    
    assert not is_valid, "Empty or whitespace-only prompt should be invalid"
    assert error_msg is not None
    assert "empty" in error_msg.lower()
    assert "example" in error_msg.lower()


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=100)
def test_property_validation_errors_are_actionable(prompt):
    """
    Property 2: Input Validation and Error Handling
    
    For any validation error generated:
    - The error message SHALL clearly explain what is wrong
    - The suggestion SHALL provide actionable guidance on how to fix it
    - The error code SHALL be descriptive and consistent
    
    Validates: Requirements 1.3, 1.4
    """
    nlp = NLPService()
    result = nlp.parse_prompt(prompt)
    
    for error in result.validation_errors:
        # Message should be descriptive (not just error code)
        assert len(error.message) > 10, "Error message should be descriptive"
        
        # If suggestion exists, it should be helpful (not empty)
        if error.suggestion:
            assert len(error.suggestion) > 10, "Suggestion should be helpful"
            
            # Suggestion should contain actionable words
            actionable_words = ['specify', 'provide', 'include', 'add', 'use', 'try', 
                              'consider', 'verify', 'check', 'ensure', 'example']
            has_actionable = any(word in error.suggestion.lower() for word in actionable_words)
            assert has_actionable, f"Suggestion should contain actionable guidance: {error.suggestion}"


@given(st.floats(min_value=0.01, max_value=0.99))
@settings(max_examples=100)
def test_property_low_resistor_values_flagged(value):
    """
    Property 2: Input Validation and Error Handling
    
    For any resistor value below 1 ohm:
    - The validation SHALL detect the unusually low value
    - Generate a validation error with code RESISTOR_VALUE_LOW
    - Suggest typical resistor range (1Ω to 10MΩ)
    
    Validates: Requirements 1.3, 1.4
    """
    nlp = NLPService()
    prompt = f"Design a 40x20mm PCB with 9V battery, LED, and {value}-ohm resistor"
    result = nlp.parse_prompt(prompt)
    
    # Should have RESISTOR_VALUE_LOW error
    low_value_errors = [e for e in result.validation_errors if e.error_code == "RESISTOR_VALUE_LOW"]
    assert len(low_value_errors) > 0, f"Resistor value {value}Ω should trigger RESISTOR_VALUE_LOW error"
    
    # Error should mention typical range
    for error in low_value_errors:
        assert "1" in error.suggestion and ("10M" in error.suggestion or "10 M" in error.suggestion)


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=100)
def test_property_confidence_score_bounds(prompt):
    """
    Property 2: Input Validation and Error Handling
    
    For any prompt, the confidence score SHALL:
    - Be a float between 0.0 and 1.0 (inclusive)
    - Decrease when validation errors or clarifications are present
    - Be deterministic (same prompt = same confidence)
    
    Validates: Requirements 1.3
    """
    nlp = NLPService()
    result = nlp.parse_prompt(prompt)
    
    # Confidence must be between 0 and 1
    assert 0.0 <= result.confidence_score <= 1.0, \
        f"Confidence score {result.confidence_score} must be between 0.0 and 1.0"
    
    # If there are many errors/clarifications, confidence should be lower
    total_issues = len(result.validation_errors) + len(result.clarification_requests)
    if total_issues >= 3:
        assert result.confidence_score < 0.8, \
            f"Confidence should be low (<0.8) when there are {total_issues} issues"
    
    # Determinism: same prompt should give same confidence
    result2 = nlp.parse_prompt(prompt)
    assert result.confidence_score == result2.confidence_score, \
        "Confidence score should be deterministic"


@given(st.text(min_size=10, max_size=200))
@settings(max_examples=100)
def test_property_to_dict_serialization(prompt):
    """
    Property 2: Input Validation and Error Handling
    
    For any parsed requirements:
    - to_dict() SHALL return a valid dictionary
    - Dictionary SHALL contain all required fields
    - Dictionary SHALL be JSON-serializable
    - Clarification requests and validation errors SHALL be included
    
    Validates: Requirements 1.3, 1.4
    """
    nlp = NLPService()
    result = nlp.parse_prompt(prompt)
    
    # Convert to dictionary
    result_dict = result.to_dict()
    
    # Must be a dictionary
    assert isinstance(result_dict, dict)
    
    # Must contain required fields
    required_fields = ['board', 'power', 'components', 'constraints', 
                      'original_prompt', 'confidence_score', 'ambiguities',
                      'clarification_requests', 'validation_errors']
    for field in required_fields:
        assert field in result_dict, f"to_dict() must include '{field}'"
    
    # Clarification requests and validation errors must be lists
    assert isinstance(result_dict['clarification_requests'], list)
    assert isinstance(result_dict['validation_errors'], list)
    
    # Try to serialize to JSON (will raise exception if not serializable)
    import json
    try:
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0
    except (TypeError, ValueError) as e:
        pytest.fail(f"to_dict() result should be JSON-serializable: {e}")



# ============================================================================
# Property Tests for Simulation Engine (Task 11.3)
# ============================================================================

"""
Property 17: Simulation Capability
Property 18: Simulation Error Handling

These properties validate that the simulation engine correctly handles
various circuit configurations and provides appropriate error handling.
"""

from hypothesis import given, strategies as st, assume
from src.services.simulation_engine import (
    SimulationEngine,
    SimulationType,
    SimulationStatus
)


# Strategy for generating valid component values
@st.composite
def component_value(draw):
    """Generate valid component values (resistance, capacitance, etc.)."""
    value = draw(st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False))
    unit = draw(st.sampled_from(["", "k", "M", "m", "u", "n", "p"]))
    return f"{value:.2f}{unit}"


# Strategy for generating valid node names
node_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=10
).filter(lambda x: x not in ["0", "GND", "gnd"])


@st.composite
def simple_circuit_components(draw):
    """Generate components for a simple valid circuit."""
    num_resistors = draw(st.integers(min_value=1, max_value=5))
    
    components = [
        {
            "type": "VOLTAGE_SOURCE",
            "reference": "V1",
            "value": "5V",
            "pins": ["1", "0"]
        }
    ]
    
    for i in range(num_resistors):
        value = draw(component_value())
        node1 = str(i + 1)
        node2 = "0" if i == num_resistors - 1 else str(i + 2)
        
        components.append({
            "type": "RESISTOR",
            "reference": f"R{i+1}",
            "value": value,
            "pins": [node1, node2]
        })
    
    return components


@given(simple_circuit_components())
def test_property_17_simulation_capability_netlist_generation(components):
    """
    Property 17: Simulation Capability
    
    **Validates: Requirements 10.1, 10.2, 10.4**
    
    Property: For any valid circuit with components and ground connection,
    the simulation engine should generate a valid SPICE netlist.
    
    Invariants:
    - Generated netlist contains all components
    - Netlist has proper .end statement
    - Netlist includes ground node (0)
    - Component references are preserved
    """
    engine = SimulationEngine()
    
    try:
        netlist = engine.generate_spice_netlist(components, [])
        
        # Invariant 1: Netlist contains all components
        for comp in components:
            ref = comp.get("reference", "")
            assert ref in netlist, f"Component {ref} not in netlist"
        
        # Invariant 2: Netlist has .end statement
        assert ".end" in netlist.lower(), "Netlist missing .end statement"
        
        # Invariant 3: Netlist includes ground node
        assert " 0 " in netlist or " 0\n" in netlist, "Netlist missing ground node"
        
        # Invariant 4: Component references preserved
        for comp in components:
            ref = comp.get("reference", "")
            if ref:
                assert netlist.count(ref) >= 1, f"Component reference {ref} not found"
    
    finally:
        engine.cleanup()


@given(simple_circuit_components())
def test_property_17_simulation_capability_dc_analysis(components):
    """
    Property 17: Simulation Capability (DC Analysis)
    
    **Validates: Requirements 10.1, 10.2, 10.4**
    
    Property: For any valid circuit, DC analysis should either succeed
    or fail with a descriptive error message.
    
    Invariants:
    - Result has valid status (SUCCESS or FAILED)
    - If successful, dc_voltages is not None
    - If failed, error_message is not None
    - Simulation type is DC
    """
    engine = SimulationEngine()
    
    try:
        netlist = engine.generate_spice_netlist(components, [])
        result = engine.run_dc_analysis(netlist)
        
        # Invariant 1: Valid status
        assert result.status in [SimulationStatus.SUCCESS, SimulationStatus.FAILED, 
                                SimulationStatus.INVALID_NETLIST], \
            f"Invalid status: {result.status}"
        
        # Invariant 2: Success implies dc_voltages
        if result.status == SimulationStatus.SUCCESS:
            assert result.dc_voltages is not None, "Success but no DC voltages"
            assert len(result.dc_voltages) > 0, "Success but empty DC voltages"
        
        # Invariant 3: Failure implies error message
        if result.status == SimulationStatus.FAILED:
            assert result.error_message is not None, "Failed but no error message"
            assert len(result.error_message) > 0, "Failed but empty error message"
        
        # Invariant 4: Simulation type is DC
        assert result.simulation_type == SimulationType.DC, \
            f"Wrong simulation type: {result.simulation_type}"
    
    finally:
        engine.cleanup()


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        min_size=1,
        max_size=20
    )
)
def test_property_17_simulation_capability_ac_analysis(freq_mag_pairs):
    """
    Property 17: Simulation Capability (AC Analysis)
    
    **Validates: Requirements 10.1, 10.2, 10.4**
    
    Property: AC analysis should handle various frequency ranges and
    return properly formatted results.
    
    Invariants:
    - Frequencies are in ascending order
    - Magnitudes are non-negative
    - Result contains ac_response data
    """
    # Create simple RC circuit for AC analysis
    components = [
        {"type": "VOLTAGE_SOURCE", "reference": "V1", "value": "1V", "pins": ["1", "0"]},
        {"type": "RESISTOR", "reference": "R1", "value": "1k", "pins": ["1", "2"]},
        {"type": "CAPACITOR", "reference": "C1", "value": "1uF", "pins": ["2", "0"]}
    ]
    
    engine = SimulationEngine()
    
    try:
        netlist = engine.generate_spice_netlist(components, [])
        
        # Use first and last frequency from generated data
        start_freq = min(f for f, _ in freq_mag_pairs)
        stop_freq = max(f for f, _ in freq_mag_pairs)
        
        assume(start_freq < stop_freq)
        assume(start_freq >= 0.1)
        
        result = engine.run_ac_analysis(netlist, start_freq=start_freq, stop_freq=stop_freq)
        
        # Invariant 1: Valid status
        assert result.status in [SimulationStatus.SUCCESS, SimulationStatus.FAILED,
                                SimulationStatus.INVALID_NETLIST]
        
        # Invariant 2: Success implies ac_response
        if result.status == SimulationStatus.SUCCESS:
            assert result.ac_response is not None, "Success but no AC response"
            
            # Invariant 3: Frequencies in ascending order
            for node, data in result.ac_response.items():
                if len(data) > 1:
                    frequencies = [f for f, _ in data]
                    for i in range(len(frequencies) - 1):
                        assert frequencies[i] <= frequencies[i+1], \
                            "Frequencies not in ascending order"
                
                # Invariant 4: Magnitudes non-negative
                for _, mag in data:
                    assert mag >= 0, f"Negative magnitude: {mag}"
    
    finally:
        engine.cleanup()


@given(st.text(min_size=1, max_size=100))
def test_property_18_simulation_error_handling_invalid_netlist(invalid_text):
    """
    Property 18: Simulation Error Handling
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
    
    Property: Invalid netlists should be rejected with appropriate error messages.
    
    Invariants:
    - Invalid netlist returns INVALID_NETLIST status
    - Error message is descriptive
    - No crash or exception propagation
    """
    engine = SimulationEngine()
    
    try:
        # Assume this is not a valid netlist
        assume(".end" not in invalid_text.lower())
        
        result = engine.run_dc_analysis(invalid_text)
        
        # Invariant 1: Invalid status
        assert result.status == SimulationStatus.INVALID_NETLIST, \
            f"Expected INVALID_NETLIST, got {result.status}"
        
        # Invariant 2: Error message exists
        assert result.error_message is not None, "No error message for invalid netlist"
        assert len(result.error_message) > 0, "Empty error message"
        
        # Invariant 3: No dc_voltages for invalid netlist
        assert result.dc_voltages is None or len(result.dc_voltages) == 0, \
            "Invalid netlist should not have DC voltages"
    
    finally:
        engine.cleanup()


@given(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10)
)
def test_property_18_simulation_error_handling_validation(num_components, num_nets):
    """
    Property 18: Simulation Error Handling (Validation)
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
    
    Property: Netlist validation should catch common errors before simulation.
    
    Invariants:
    - Validation returns boolean and optional error message
    - Invalid netlists have descriptive errors
    - Valid netlists pass validation
    """
    engine = SimulationEngine()
    
    try:
        # Test 1: Missing .end
        netlist_no_end = "V1 1 0 5V\nR1 1 0 1k"
        is_valid, error = engine.validate_netlist(netlist_no_end)
        assert not is_valid, "Should reject netlist without .end"
        assert error is not None, "Should have error message"
        assert ".end" in error.lower(), "Error should mention .end"
        
        # Test 2: No components
        netlist_no_comp = "* Comment\n.op\n.end"
        is_valid, error = engine.validate_netlist(netlist_no_comp)
        assert not is_valid, "Should reject netlist without components"
        assert error is not None, "Should have error message"
        
        # Test 3: No ground
        netlist_no_gnd = "V1 1 2 5V\nR1 1 2 1k\n.end"
        is_valid, error = engine.validate_netlist(netlist_no_gnd)
        assert not is_valid, "Should reject netlist without ground"
        assert error is not None, "Should have error message"
        
        # Test 4: Valid netlist
        netlist_valid = "V1 1 0 5V\nR1 1 0 1k\n.op\n.end"
        is_valid, error = engine.validate_netlist(netlist_valid)
        assert is_valid, f"Should accept valid netlist: {error}"
        assert error is None, "Valid netlist should have no error"
    
    finally:
        engine.cleanup()


@given(st.integers(min_value=1, max_value=100))
def test_property_18_simulation_error_handling_cleanup(num_iterations):
    """
    Property 18: Simulation Error Handling (Resource Cleanup)
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
    
    Property: Simulation engine should properly clean up resources
    even after multiple operations.
    
    Invariants:
    - Cleanup can be called multiple times safely
    - Work directory exists before cleanup
    - No exceptions during cleanup
    """
    engine = SimulationEngine()
    
    try:
        # Invariant 1: Work directory exists
        assert engine.work_dir.exists(), "Work directory should exist"
        
        # Perform some operations
        components = [
            {"type": "VOLTAGE_SOURCE", "reference": "V1", "value": "5V", "pins": ["1", "0"]},
            {"type": "RESISTOR", "reference": "R1", "value": "1k", "pins": ["1", "0"]}
        ]
        
        for _ in range(min(num_iterations, 10)):  # Limit to 10 for performance
            netlist = engine.generate_spice_netlist(components, [])
            _ = engine.run_dc_analysis(netlist)
        
        # Invariant 2: Cleanup succeeds
        engine.cleanup()
        
        # Invariant 3: Multiple cleanups are safe
        engine.cleanup()
        engine.cleanup()
    
    except Exception as e:
        # Invariant 4: No exceptions during cleanup
        assert False, f"Cleanup raised exception: {e}"
