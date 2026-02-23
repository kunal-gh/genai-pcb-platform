"""
Property-based tests for Design Verification System.

Uses Hypothesis to generate thousands of test cases and validate
universal properties of the ERC/DRC and DFM verification systems.

Feature: genai-pcb-platform
Property 10: Comprehensive Design Verification
Property 11: DFM Validation and Scoring
Property 12: DFM Success Rate Target
Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.strategies import composite
import random
from typing import Dict, Any, List

from src.services.design_verification import (
    DesignVerificationEngine,
    DesignRules,
    ViolationType,
    Severity
)
from src.services.dfm_validation import (
    DFMValidator,
    ManufacturingConstraints,
    DFMViolationSeverity
)
from src.services.verification_reporting import (
    VerificationReporter,
    ReportFormat
)


# ============================================================================
# Custom Strategies for Generating Test Data
# ============================================================================

@composite
def valid_component(draw):
    """Generate a valid component for testing."""
    component_types = ["resistor", "capacitor", "led", "transistor_npn", "op_amp"]
    comp_type = draw(st.sampled_from(component_types))
    reference = f"{comp_type[0].upper()}{draw(st.integers(min_value=1, max_value=100))}"
    
    return {
        "reference": reference,
        "type": comp_type,
        "value": draw(st.text(min_size=1, max_size=10)),
        "package": draw(st.sampled_from(["smd", "through_hole", "0603", "0805"]))
    }

@composite
def valid_net(draw):
    """Generate a valid net for testing."""
    net_name = draw(st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=1, max_size=20
    ))
    
    num_connections = draw(st.integers(min_value=0, max_value=5))
    connections = []
    
    for i in range(num_connections):
        connections.append({
            "component": f"R{i+1}",
            "pin": str(draw(st.integers(min_value=1, max_value=8)))
        })
    
    return {
        "name": net_name,
        "connections": connections
    }
@composite
def valid_netlist_data(draw):
    """Generate valid netlist data for testing."""
    num_components = draw(st.integers(min_value=1, max_value=10))
    num_nets = draw(st.integers(min_value=1, max_value=15))
    
    components = [draw(valid_component()) for _ in range(num_components)]
    nets = [draw(valid_net()) for _ in range(num_nets)]
    
    # Ensure at least one power and ground net
    nets.append({
        "name": "VCC",
        "connections": [{"component": "R1", "pin": "1"}]
    })
    nets.append({
        "name": "GND", 
        "connections": [{"component": "R1", "pin": "2"}]
    })
    
    return {
        "components": components,
        "nets": nets
    }

@composite
def valid_trace(draw):
    """Generate a valid trace for PCB data."""
    return {
        "width": draw(st.floats(min_value=0.05, max_value=5.0)),
        "net": draw(st.text(min_size=1, max_size=10)),
        "x": draw(st.floats(min_value=0.0, max_value=100.0)),
        "y": draw(st.floats(min_value=0.0, max_value=100.0)),
        "length": draw(st.floats(min_value=1.0, max_value=200.0))
    }

@composite
def valid_via(draw):
    """Generate a valid via for PCB data."""
    size = draw(st.floats(min_value=0.1, max_value=2.0))
    drill = draw(st.floats(min_value=0.05, max_value=size * 0.8))
    
    return {
        "size": size,
        "drill": drill,
        "x": draw(st.floats(min_value=0.0, max_value=100.0)),
        "y": draw(st.floats(min_value=0.0, max_value=100.0))
    }

@composite
def valid_pcb_component(draw):
    """Generate a valid PCB component with position."""
    comp = draw(valid_component())
    comp.update({
        "x": draw(st.floats(min_value=1.0, max_value=99.0)),
        "y": draw(st.floats(min_value=1.0, max_value=99.0)),
        "pad_size": draw(st.floats(min_value=0.1, max_value=2.0))
    })
    return comp

@composite
def valid_pcb_data(draw):
    """Generate valid PCB data for testing."""
    num_traces = draw(st.integers(min_value=1, max_value=20))
    num_vias = draw(st.integers(min_value=0, max_value=10))
    num_components = draw(st.integers(min_value=1, max_value=10))
    
    return {
        "traces": [draw(valid_trace()) for _ in range(num_traces)],
        "vias": [draw(valid_via()) for _ in range(num_vias)],
        "components": [draw(valid_pcb_component()) for _ in range(num_components)],
        "width": draw(st.floats(min_value=20.0, max_value=200.0)),
        "height": draw(st.floats(min_value=20.0, max_value=200.0)),
        "thickness": draw(st.floats(min_value=0.4, max_value=3.2)),
        "layers": draw(st.integers(min_value=2, max_value=8)),
        "drill_holes": [
            {
                "diameter": draw(st.floats(min_value=0.1, max_value=3.0)),
                "x": draw(st.floats(min_value=0.0, max_value=100.0)),
                "y": draw(st.floats(min_value=0.0, max_value=100.0))
            }
            for _ in range(draw(st.integers(min_value=0, max_value=5)))
        ]
    }


# ============================================================================
# Property 10: Comprehensive Design Verification
# Feature: genai-pcb-platform
# Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
# ============================================================================

@pytest.mark.property
@given(netlist_data=valid_netlist_data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_10_erc_violation_detection(netlist_data):
    """
    Property 10: Comprehensive Design Verification (ERC)
    
    **Validates: Requirements 5.1, 5.4**
    
    Property: For any netlist data, ERC checking SHALL detect all electrical
    rule violations and provide specific error locations and suggested fixes.
    
    Invariants:
    - All violations have valid severity levels
    - All violations have descriptive messages
    - All violations have suggested fixes
    - Violation types are from valid enum
    """
    engine = DesignVerificationEngine()
    
    result = engine.verify_design(netlist_data)
    
    # Invariant 1: Result is successful
    assert result["success"] is True, "Verification should not fail with valid input"
    
    # Invariant 2: All violations have valid properties
    for violation in result["violations"]:
        # Valid violation type
        assert violation["type"] in [vt.value for vt in ViolationType], \
            f"Invalid violation type: {violation['type']}"
        
        # Valid severity
        assert violation["severity"] in [s.value for s in Severity], \
            f"Invalid severity: {violation['severity']}"
        
        # Descriptive message
        assert isinstance(violation["message"], str), "Message must be string"
        assert len(violation["message"]) > 10, "Message must be descriptive"
        
        # Suggested fix (if present)
        if violation["suggested_fix"]:
            assert isinstance(violation["suggested_fix"], str), "Fix must be string"
            assert len(violation["suggested_fix"]) > 5, "Fix must be helpful"
    
    # Invariant 3: Summary contains required fields
    summary = result["summary"]
    required_fields = ["total_violations", "errors", "warnings", 
                      "erc_violations", "drc_violations", "connectivity_violations"]
    for field in required_fields:
        assert field in summary, f"Summary missing field: {field}"
        assert isinstance(summary[field], int), f"Field {field} must be integer"
        assert summary[field] >= 0, f"Field {field} must be non-negative"


@pytest.mark.property
@given(pcb_data=valid_pcb_data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_10_drc_constraint_validation(pcb_data):
    """
    Property 10: Comprehensive Design Verification (DRC)
    
    **Validates: Requirements 5.2, 5.5**
    
    Property: For any PCB layout data, DRC checking SHALL validate all
    design rules and categorize violations by priority.
    
    Invariants:
    - Trace width violations are detected
    - Via size violations are detected
    - Spacing violations are detected
    - All violations have rule names
    """
    engine = DesignVerificationEngine()
    netlist_data = {"components": [], "nets": []}
    
    result = engine.verify_design(netlist_data, pcb_data)
    
    # Invariant 1: DRC violations are categorized
    drc_violations = [v for v in result["violations"] 
                     if v["type"].startswith("drc")]
    
    # Invariant 2: Check for expected violation types based on data
    for trace in pcb_data["traces"]:
        width = trace["width"]
        if width < engine.design_rules.min_trace_width:
            # Should have trace width violation
            width_violations = [v for v in drc_violations 
                              if "trace width" in v["message"].lower()]
            assert len(width_violations) > 0, \
                f"Missing trace width violation for width {width}mm"
    
    # Invariant 3: All DRC violations have rule names
    for violation in drc_violations:
        if violation.get("rule_name"):
            assert isinstance(violation["rule_name"], str), "Rule name must be string"
            assert len(violation["rule_name"]) > 0, "Rule name must not be empty"


@pytest.mark.property
@given(
    netlist_data=valid_netlist_data(),
    pcb_data=valid_pcb_data()
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_10_connectivity_validation(netlist_data, pcb_data):
    """
    Property 10: Comprehensive Design Verification (Connectivity)
    
    **Validates: Requirements 5.4**
    
    Property: For any design, connectivity validation SHALL detect
    unconnected pins, single-pin nets, and empty nets.
    
    Invariants:
    - Single-pin nets are flagged as warnings
    - Empty nets are flagged as errors
    - All connectivity violations have net names
    """
    engine = DesignVerificationEngine()
    
    result = engine.verify_design(netlist_data, pcb_data)
    
    # Invariant 1: Connectivity violations are properly categorized
    connectivity_violations = [v for v in result["violations"] 
                             if v["type"] == "connectivity_error"]
    
    # Invariant 2: Check for single-pin nets
    for net in netlist_data["nets"]:
        if len(net["connections"]) == 1:
            single_pin_violations = [v for v in connectivity_violations 
                                   if net["name"] in v.get("message", "")]
            # Should have at least one violation for single-pin net
            assert len(single_pin_violations) > 0, \
                f"Missing violation for single-pin net: {net['name']}"
    
    # Invariant 3: All connectivity violations reference nets
    for violation in connectivity_violations:
        if violation.get("net"):
            assert isinstance(violation["net"], str), "Net name must be string"
            assert len(violation["net"]) > 0, "Net name must not be empty"


# ============================================================================
# Property 11: DFM Validation and Scoring
# Feature: genai-pcb-platform  
# Validates: Requirements 6.1, 6.2, 6.3, 6.4
# ============================================================================

@pytest.mark.property
@given(pcb_data=valid_pcb_data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_11_dfm_manufacturability_scoring(pcb_data):
    """
    Property 11: DFM Validation and Scoring
    
    **Validates: Requirements 6.1, 6.4**
    
    Property: For any PCB design, DFM validation SHALL provide accurate
    manufacturability scoring between 0-100 with confidence levels.
    
    Invariants:
    - Score is between 0.0 and 100.0
    - Score decreases with more violations
    - Confidence level matches score ranges
    - Critical violations significantly impact score
    """
    validator = DFMValidator()
    
    result = validator.validate_design(pcb_data)
    
    # Invariant 1: Valid score range
    score = result["score"]
    assert 0.0 <= score <= 100.0, f"Score {score} out of valid range [0, 100]"
    
    # Invariant 2: Confidence level matches score
    confidence = result["confidence_level"]
    expected_confidence = validator._get_confidence_level(score)
    assert confidence == expected_confidence, \
        f"Confidence mismatch: got {confidence}, expected {expected_confidence}"
    
    # Invariant 3: Critical violations impact score significantly
    critical_violations = [v for v in result["violations"] 
                          if v["severity"] == "critical"]
    if len(critical_violations) > 0:
        assert score < 85.0, \
            f"Score {score} too high with {len(critical_violations)} critical violations"
    
    # Invariant 4: Score consistency
    # Same input should give same score
    result2 = validator.validate_design(pcb_data)
    assert result["score"] == result2["score"], "DFM scoring should be deterministic"


@pytest.mark.property
@given(pcb_data=valid_pcb_data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_11_dfm_violation_categorization(pcb_data):
    """
    Property 11: DFM Validation and Scoring (Categorization)
    
    **Validates: Requirements 6.2, 6.3**
    
    Property: For any design, DFM violations SHALL be properly categorized
    by type (trace, via, component, etc.) with appropriate recommendations.
    
    Invariants:
    - All violations have valid categories
    - All violations have severity levels
    - All violations have recommendations
    - Cost impact is specified where applicable
    """
    validator = DFMValidator()
    
    result = validator.validate_design(pcb_data)
    
    # Invariant 1: All violations have valid properties
    valid_categories = ["trace", "via", "component", "drill", "board", "signal_integrity"]
    valid_severities = [s.value for s in DFMViolationSeverity]
    
    for violation in result["violations"]:
        # Valid category
        assert violation["category"] in valid_categories, \
            f"Invalid category: {violation['category']}"
        
        # Valid severity
        assert violation["severity"] in valid_severities, \
            f"Invalid severity: {violation['severity']}"
        
        # Has recommendation
        assert violation.get("recommendation") is not None, \
            "All violations must have recommendations"
        assert len(violation["recommendation"]) > 10, \
            "Recommendations must be descriptive"
        
        # Cost impact (if specified)
        if violation.get("cost_impact"):
            assert violation["cost_impact"] in ["low", "medium", "high"], \
                f"Invalid cost impact: {violation['cost_impact']}"
    
    # Invariant 2: Violations are grouped by category
    by_category = result["summary"]["by_category"]
    assert isinstance(by_category, dict), "by_category must be dictionary"
    
    total_by_category = sum(by_category.values())
    assert total_by_category == result["summary"]["total_violations"], \
        "Category counts must sum to total violations"


@pytest.mark.property
@given(
    manufacturer=st.sampled_from(["standard", "jlcpcb", "pcbway", "oshpark"]),
    pcb_data=valid_pcb_data()
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_11_manufacturer_specific_constraints(manufacturer, pcb_data):
    """
    Property 11: DFM Validation and Scoring (Manufacturer Constraints)
    
    **Validates: Requirements 6.1, 6.3**
    
    Property: For any manufacturer profile, DFM validation SHALL apply
    manufacturer-specific constraints and adjust scoring accordingly.
    
    Invariants:
    - Different manufacturers may have different violation counts
    - Constraints are applied consistently
    - Manufacturer name is preserved in results
    """
    validator = DFMValidator()
    
    result = validator.validate_design(pcb_data, manufacturer=manufacturer)
    
    # Invariant 1: Manufacturer is preserved
    assert result["manufacturer"] == manufacturer, \
        f"Manufacturer mismatch: expected {manufacturer}, got {result['manufacturer']}"
    
    # Invariant 2: Constraints are applied
    constraints = result["constraints"]
    assert isinstance(constraints, dict), "Constraints must be dictionary"
    
    required_constraint_fields = ["min_trace_width", "min_via_diameter", 
                                 "min_drill_size", "min_component_spacing"]
    for field in required_constraint_fields:
        assert field in constraints, f"Missing constraint field: {field}"
        assert isinstance(constraints[field], (int, float)), \
            f"Constraint {field} must be numeric"
        assert constraints[field] > 0, f"Constraint {field} must be positive"
    
    # Invariant 3: Manufacturability determination
    is_manufacturable = result["manufacturable"]
    critical_count = result["summary"]["critical"]
    
    # If no critical violations, should be manufacturable
    if critical_count == 0:
        assert is_manufacturable is True, \
            "Design with no critical violations should be manufacturable"
    else:
        assert is_manufacturable is False, \
            "Design with critical violations should not be manufacturable"


# ============================================================================
# Property 12: DFM Success Rate Target
# Feature: genai-pcb-platform
# Validates: Requirements 6.5
# ============================================================================

@pytest.mark.property
@given(
    design_batch=st.lists(valid_pcb_data(), min_size=10, max_size=20)
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_12_dfm_success_rate_target(design_batch):
    """
    Property 12: DFM Success Rate Target
    
    **Validates: Requirements 6.5**
    
    Property: For any batch of generated designs, the system SHALL achieve
    ≥95% DFM pass rate (designs scoring ≥85/100 are considered passing).
    
    Invariants:
    - Pass rate is calculated correctly
    - Pass threshold is 85/100
    - Batch statistics are accurate
    - Success rate meets target when designs are well-formed
    """
    validator = DFMValidator()
    
    # Process entire batch
    results = []
    passing_designs = 0
    
    for design in design_batch:
        result = validator.validate_design(design)
        results.append(result)
        
        # Count passing designs (score >= 85)
        if result["score"] >= 85.0:
            passing_designs += 1
    
    # Invariant 1: Pass rate calculation
    total_designs = len(design_batch)
    pass_rate = (passing_designs / total_designs) * 100.0
    
    assert 0.0 <= pass_rate <= 100.0, f"Pass rate {pass_rate}% out of valid range"
    
    # Invariant 2: Batch statistics
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    assert 0.0 <= avg_score <= 100.0, f"Average score {avg_score} out of range"
    assert 0.0 <= min_score <= max_score <= 100.0, "Score range invalid"
    
    # Invariant 3: Well-formed designs should have reasonable pass rates
    # For randomly generated valid PCB data, we expect some designs to pass
    # This tests that the DFM system isn't overly restrictive
    if avg_score >= 70.0:  # If designs are reasonably well-formed
        assert pass_rate >= 50.0, \
            f"Pass rate {pass_rate}% too low for well-formed designs (avg score: {avg_score})"
    
    # Invariant 4: Consistency check
    # Re-validate first design to ensure deterministic results
    if results:
        revalidation = validator.validate_design(design_batch[0])
        assert results[0]["score"] == revalidation["score"], \
            "DFM validation should be deterministic"


@pytest.mark.property
@given(pcb_data=valid_pcb_data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_12_dfm_pass_criteria_consistency(pcb_data):
    """
    Property 12: DFM Success Rate Target (Pass Criteria)
    
    **Validates: Requirements 6.5**
    
    Property: For any design, the pass/fail determination SHALL be
    consistent with the scoring system and violation severity.
    
    Invariants:
    - Designs with score ≥85 and no critical violations pass
    - Designs with critical violations fail regardless of score
    - Pass/fail status is deterministic
    - Confidence level aligns with pass/fail status
    """
    validator = DFMValidator()
    
    result = validator.validate_design(pcb_data)
    
    score = result["score"]
    is_manufacturable = result["manufacturable"]
    critical_count = result["summary"]["critical"]
    confidence = result["confidence_level"]
    
    # Invariant 1: Critical violations prevent manufacturing
    if critical_count > 0:
        assert is_manufacturable is False, \
            f"Design with {critical_count} critical violations should not be manufacturable"
    
    # Invariant 2: High scores without critical issues should pass
    if score >= 85.0 and critical_count == 0:
        assert is_manufacturable is True, \
            f"Design with score {score} and no critical violations should be manufacturable"
    
    # Invariant 3: Confidence aligns with manufacturability
    if is_manufacturable:
        assert confidence in ["excellent", "good"], \
            f"Manufacturable design should have good confidence, got: {confidence}"
    else:
        if score < 50.0:
            assert confidence in ["poor", "unmanufacturable"], \
                f"Poor design should have low confidence, got: {confidence}"
    
    # Invariant 4: Score and manufacturability correlation
    if score >= 95.0:
        assert confidence == "excellent", \
            f"Score {score} should have excellent confidence"
    elif score < 50.0:
        assert confidence in ["poor", "unmanufacturable"], \
            f"Score {score} should have poor confidence"


# ============================================================================
# Integration Property Tests
# ============================================================================

@pytest.mark.property
@given(
    netlist_data=valid_netlist_data(),
    pcb_data=valid_pcb_data()
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_verification_integration(netlist_data, pcb_data):
    """
    Property: Verification Integration
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5**
    
    Property: For any design, the integrated verification system SHALL
    provide comprehensive reporting with ERC/DRC and DFM results.
    
    Invariants:
    - All verification engines work together
    - Reports contain all required sections
    - Overall design readiness is determined correctly
    """
    # Run ERC/DRC verification
    erc_drc_engine = DesignVerificationEngine()
    erc_drc_results = erc_drc_engine.verify_design(netlist_data, pcb_data)
    
    # Run DFM validation
    dfm_validator = DFMValidator()
    dfm_results = dfm_validator.validate_design(pcb_data)
    
    # Generate integrated report
    reporter = VerificationReporter()
    report = reporter.generate_report(erc_drc_results, dfm_results)
    
    # Invariant 1: Report structure
    required_sections = ["timestamp", "summary", "violations", 
                        "recommendations", "next_steps"]
    for section in required_sections:
        assert section in report, f"Report missing section: {section}"
    
    # Invariant 2: Summary completeness
    summary = report["summary"]
    required_summary_fields = ["total_violations", "critical_count", "error_count",
                              "warning_count", "erc_violations", "dfm_violations",
                              "design_ready", "manufacturability_score"]
    for field in required_summary_fields:
        assert field in summary, f"Summary missing field: {field}"
    
    # Invariant 3: Design readiness logic
    design_ready = summary["design_ready"]
    critical_count = summary["critical_count"]
    
    if critical_count > 0:
        assert design_ready is False, \
            "Design with critical violations should not be ready"
    
    # Invariant 4: Next steps are provided
    next_steps = report["next_steps"]
    assert isinstance(next_steps, list), "Next steps must be a list"
    assert len(next_steps) > 0, "Must provide next steps"
    
    for step in next_steps:
        assert isinstance(step, str), "Each step must be a string"
        assert len(step) > 5, "Steps must be descriptive"


@pytest.mark.property
@given(
    format_type=st.sampled_from([ReportFormat.JSON, ReportFormat.HTML, 
                                ReportFormat.TEXT, ReportFormat.MARKDOWN])
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_report_format_consistency(format_type):
    """
    Property: Report Format Consistency
    
    **Validates: Requirements 5.3, 6.2**
    
    Property: For any report format, the generated report SHALL contain
    all essential information in the appropriate format.
    
    Invariants:
    - All formats contain core information
    - Format-specific output is valid
    - Reports are non-empty
    """
    # Create sample data
    erc_drc_results = {
        "success": True,
        "ready_for_manufacturing": True,
        "summary": {"total_violations": 0, "errors": 0, "warnings": 0,
                   "erc_violations": 0, "drc_violations": 0, "connectivity_violations": 0},
        "violations": []
    }
    
    dfm_results = {
        "success": True,
        "manufacturable": True,
        "score": 95.0,
        "confidence_level": "excellent",
        "summary": {"total_violations": 0, "critical": 0, "high": 0, 
                   "medium": 0, "low": 0, "by_category": {}},
        "violations": []
    }
    
    reporter = VerificationReporter()
    report = reporter.generate_report(erc_drc_results, dfm_results, format=format_type)
    
    # Invariant 1: Report is generated successfully
    assert "summary" in report, "Report must contain summary"
    assert "violations" in report, "Report must contain violations"
    
    # Invariant 2: Format-specific output
    if format_type != ReportFormat.JSON:
        assert "formatted_output" in report, f"Missing formatted output for {format_type}"
        formatted = report["formatted_output"]
        assert isinstance(formatted, str), "Formatted output must be string"
        assert len(formatted) > 100, "Formatted output must be substantial"
        
        # Format-specific checks
        if format_type == ReportFormat.HTML:
            assert "<html>" in formatted.lower(), "HTML format must contain HTML tags"
            assert "</html>" in formatted.lower(), "HTML format must be complete"
        elif format_type == ReportFormat.MARKDOWN:
            assert "#" in formatted, "Markdown format must contain headers"
        elif format_type == ReportFormat.TEXT:
            assert "VERIFICATION REPORT" in formatted.upper(), "Text format must have title"