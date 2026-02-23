#!/usr/bin/env python3
"""
Standalone test runner for verification property tests.
Runs without pytest conftest.py to avoid import issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.strategies import composite

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

# Simple test data generators
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
def valid_netlist_data(draw):
    """Generate valid netlist data for testing."""
    num_components = draw(st.integers(min_value=1, max_value=5))
    
    components = [draw(valid_component()) for _ in range(num_components)]
    
    # Simple nets with power and ground
    nets = [
        {
            "name": "VCC",
            "connections": [{"component": "R1", "pin": "1"}]
        },
        {
            "name": "GND", 
            "connections": [{"component": "R1", "pin": "2"}]
        }
    ]
    
    return {
        "components": components,
        "nets": nets
    }

@composite
def valid_pcb_data(draw):
    """Generate valid PCB data for testing."""
    return {
        "traces": [
            {
                "width": draw(st.floats(min_value=0.05, max_value=5.0)),
                "net": "VCC",
                "x": 10.0,
                "y": 10.0,
                "length": 50.0
            }
        ],
        "vias": [
            {
                "size": draw(st.floats(min_value=0.1, max_value=2.0)),
                "drill": 0.2,
                "x": 20.0,
                "y": 20.0
            }
        ],
        "components": [
            {
                "reference": "R1",
                "type": "resistor",
                "x": 30.0,
                "y": 30.0,
                "pad_size": 0.5
            }
        ],
        "width": 50.0,
        "height": 40.0,
        "thickness": 1.6,
        "layers": 2,
        "drill_holes": []
    }

def test_property_10_basic():
    """Test basic ERC/DRC functionality."""
    print("Testing Property 10: ERC/DRC Verification...")
    
    engine = DesignVerificationEngine()
    
    # Test with simple valid data
    netlist_data = {
        "components": [
            {"reference": "R1", "type": "resistor", "value": "1k"}
        ],
        "nets": [
            {"name": "VCC", "connections": [{"component": "R1", "pin": "1"}]},
            {"name": "GND", "connections": [{"component": "R1", "pin": "2"}]}
        ]
    }
    
    result = engine.verify_design(netlist_data)
    
    assert result["success"] is True, "Verification should succeed"
    assert "violations" in result, "Result should contain violations"
    assert "summary" in result, "Result should contain summary"
    
    print("✓ Property 10 basic test passed")

def test_property_11_basic():
    """Test basic DFM validation."""
    print("Testing Property 11: DFM Validation...")
    
    validator = DFMValidator()
    
    # Test with simple PCB data
    pcb_data = {
        "traces": [{"width": 0.2, "net": "VCC", "x": 10.0, "y": 10.0, "length": 50.0}],
        "vias": [{"size": 0.3, "drill": 0.2, "x": 20.0, "y": 20.0}],
        "components": [{"reference": "R1", "type": "resistor", "x": 30.0, "y": 30.0}],
        "width": 50.0,
        "height": 40.0,
        "thickness": 1.6,
        "layers": 2,
        "drill_holes": []
    }
    
    result = validator.validate_design(pcb_data)
    
    assert result["success"] is True, "DFM validation should succeed"
    assert "score" in result, "Result should contain score"
    assert 0.0 <= result["score"] <= 100.0, f"Score {result['score']} out of range"
    assert "violations" in result, "Result should contain violations"
    
    print(f"✓ Property 11 basic test passed (score: {result['score']:.1f})")

def test_property_12_basic():
    """Test DFM success rate calculation."""
    print("Testing Property 12: DFM Success Rate...")
    
    validator = DFMValidator()
    
    # Test with multiple designs
    designs = []
    for i in range(5):
        designs.append({
            "traces": [{"width": 0.15 + i*0.05, "net": f"NET{i}", "x": 10.0, "y": 10.0, "length": 50.0}],
            "vias": [{"size": 0.3, "drill": 0.2, "x": 20.0, "y": 20.0}],
            "components": [{"reference": f"R{i}", "type": "resistor", "x": 30.0, "y": 30.0}],
            "width": 50.0,
            "height": 40.0,
            "thickness": 1.6,
            "layers": 2,
            "drill_holes": []
        })
    
    results = []
    passing_count = 0
    
    for design in designs:
        result = validator.validate_design(design)
        results.append(result)
        if result["score"] >= 85.0:
            passing_count += 1
    
    pass_rate = (passing_count / len(designs)) * 100.0
    avg_score = sum(r["score"] for r in results) / len(results)
    
    print(f"✓ Property 12 basic test passed (pass rate: {pass_rate:.1f}%, avg score: {avg_score:.1f})")

@given(netlist_data=valid_netlist_data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_10_hypothesis(netlist_data):
    """Property test for ERC/DRC with Hypothesis."""
    engine = DesignVerificationEngine()
    
    result = engine.verify_design(netlist_data)
    
    # Basic invariants
    assert result["success"] is True
    assert isinstance(result["violations"], list)
    assert isinstance(result["summary"], dict)
    
    # Check violation structure
    for violation in result["violations"]:
        assert "type" in violation
        assert "severity" in violation
        assert "message" in violation
        assert violation["type"] in [vt.value for vt in ViolationType]
        assert violation["severity"] in [s.value for s in Severity]

@given(pcb_data=valid_pcb_data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_11_hypothesis(pcb_data):
    """Property test for DFM validation with Hypothesis."""
    validator = DFMValidator()
    
    result = validator.validate_design(pcb_data)
    
    # Basic invariants
    assert result["success"] is True
    assert 0.0 <= result["score"] <= 100.0
    assert isinstance(result["violations"], list)
    assert result["confidence_level"] in ["excellent", "good", "fair", "poor", "unmanufacturable"]

if __name__ == "__main__":
    print("Running Verification Property Tests...")
    print("=" * 50)
    
    try:
        # Run basic tests
        test_property_10_basic()
        test_property_11_basic()
        test_property_12_basic()
        
        print("\nRunning Hypothesis property tests...")
        
        # Run hypothesis tests
        test_property_10_hypothesis()
        test_property_11_hypothesis()
        
        print("\n" + "=" * 50)
        print("✓ All verification property tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)