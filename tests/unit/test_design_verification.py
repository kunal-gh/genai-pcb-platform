"""
Unit tests for design verification engine.

Tests DesignVerificationEngine class functionality.
"""

import pytest
from src.services.design_verification import (
    DesignVerificationEngine,
    DesignRules,
    DesignViolation,
    ViolationType,
    Severity
)


@pytest.fixture
def default_engine():
    """Create verification engine with default rules."""
    return DesignVerificationEngine()


@pytest.fixture
def custom_rules():
    """Create custom design rules."""
    return DesignRules(
        min_trace_width=0.15,
        min_via_size=0.25,
        min_trace_spacing=0.15,
        min_component_spacing=1.0
    )


@pytest.fixture
def custom_engine(custom_rules):
    """Create verification engine with custom rules."""
    return DesignVerificationEngine(custom_rules)


@pytest.fixture
def sample_netlist():
    """Sample netlist data for testing."""
    return {
        "components": [
            {"reference": "R1", "type": "resistor", "value": "10k"},
            {"reference": "R2", "type": "resistor", "value": "1k"},
            {"reference": "C1", "type": "capacitor", "value": "100nF"},
            {"reference": "U1", "type": "op_amp", "value": "LM358"},
            {"reference": "LED1", "type": "led", "value": "red"}
        ],
        "nets": [
            {
                "name": "VCC",
                "connections": [
                    {"component": "U1", "pin": "4"},
                    {"component": "R1", "pin": "1"}
                ]
            },
            {
                "name": "GND",
                "connections": [
                    {"component": "U1", "pin": "5"},
                    {"component": "C1", "pin": "2"},
                    {"component": "LED1", "pin": "2"}
                ]
            },
            {
                "name": "OUTPUT",
                "connections": [
                    {"component": "U1", "pin": "1"},
                    {"component": "R2", "pin": "1"},
                    {"component": "LED1", "pin": "1"}
                ]
            },
            {
                "name": "INPUT",
                "connections": [
                    {"component": "U1", "pin": "2"},
                    {"component": "R1", "pin": "2"}
                ]
            },
            {
                "name": "FEEDBACK",
                "connections": [
                    {"component": "U1", "pin": "3"},
                    {"component": "R2", "pin": "2"},
                    {"component": "C1", "pin": "1"}
                ]
            }
        ]
    }


@pytest.fixture
def sample_pcb_data():
    """Sample PCB data for testing."""
    return {
        "traces": [
            {"width": 0.2, "net": "VCC", "x": 10.0, "y": 20.0},
            {"width": 0.05, "net": "SIGNAL", "x": 15.0, "y": 25.0},  # Too narrow
            {"width": 0.3, "net": "GND", "x": 20.0, "y": 30.0}
        ],
        "vias": [
            {"size": 0.3, "x": 10.0, "y": 10.0},
            {"size": 0.1, "x": 20.0, "y": 20.0},  # Too small
            {"size": 0.4, "x": 30.0, "y": 30.0}
        ],
        "components": [
            {"reference": "R1", "x": 10.0, "y": 10.0},
            {"reference": "R2", "x": 10.2, "y": 10.2},  # Too close
            {"reference": "C1", "x": 20.0, "y": 20.0}
        ],
        "drill_holes": [
            {"diameter": 0.2, "x": 10.0, "y": 10.0},
            {"diameter": 0.1, "x": 20.0, "y": 20.0},  # Too small
            {"diameter": 0.8, "x": 30.0, "y": 30.0}
        ]
    }


@pytest.fixture
def problematic_netlist():
    """Netlist with various issues for testing."""
    return {
        "components": [
            {"reference": "R1", "type": "resistor", "value": "10k"},
            {"reference": "U1", "type": "op_amp", "value": "LM358"}
        ],
        "nets": [
            {
                "name": "SINGLE_PIN",
                "connections": [
                    {"component": "R1", "pin": "1"}
                ]
            },
            {
                "name": "EMPTY_NET",
                "connections": []
            },
            {
                "name": "MULTI_OUTPUT",
                "connections": [
                    {"component": "U1", "pin": "1"},  # Output
                    {"component": "U1", "pin": "2"}   # Another output (conflict)
                ]
            }
        ]
    }


class TestDesignRules:
    """Tests for DesignRules dataclass."""
    
    def test_default_rules(self):
        """Test default design rules."""
        rules = DesignRules()
        
        assert rules.min_trace_width == 0.1
        assert rules.min_via_size == 0.2
        assert rules.min_trace_spacing == 0.1
        assert rules.min_component_spacing == 0.5
        assert rules.max_fanout == 50
        assert rules.min_impedance == 25.0
        assert rules.max_impedance == 120.0
    
    def test_custom_rules(self):
        """Test custom design rules."""
        rules = DesignRules(
            min_trace_width=0.15,
            min_via_size=0.25,
            max_fanout=100
        )
        
        assert rules.min_trace_width == 0.15
        assert rules.min_via_size == 0.25
        assert rules.max_fanout == 100
        # Defaults should still apply
        assert rules.min_trace_spacing == 0.1


class TestDesignViolation:
    """Tests for DesignViolation dataclass."""
    
    def test_basic_violation(self):
        """Test basic violation creation."""
        violation = DesignViolation(
            violation_type=ViolationType.ERC_ERROR,
            severity=Severity.ERROR,
            message="Test violation"
        )
        
        assert violation.violation_type == ViolationType.ERC_ERROR
        assert violation.severity == Severity.ERROR
        assert violation.message == "Test violation"
        assert violation.component is None
        assert violation.net is None
    
    def test_detailed_violation(self):
        """Test violation with all fields."""
        violation = DesignViolation(
            violation_type=ViolationType.DRC_ERROR,
            severity=Severity.CRITICAL,
            message="Trace too narrow",
            component="R1",
            net="VCC",
            location=(10.0, 20.0),
            suggested_fix="Increase trace width",
            rule_name="min_trace_width"
        )
        
        assert violation.violation_type == ViolationType.DRC_ERROR
        assert violation.severity == Severity.CRITICAL
        assert violation.component == "R1"
        assert violation.net == "VCC"
        assert violation.location == (10.0, 20.0)
        assert violation.suggested_fix == "Increase trace width"
        assert violation.rule_name == "min_trace_width"


class TestDesignVerificationEngine:
    """Tests for DesignVerificationEngine class."""
    
    def test_init_default_rules(self, default_engine):
        """Test initialization with default rules."""
        assert default_engine.design_rules.min_trace_width == 0.1
        assert len(default_engine.violations) == 0
        assert len(default_engine.component_pins) > 0
    
    def test_init_custom_rules(self, custom_engine, custom_rules):
        """Test initialization with custom rules."""
        assert custom_engine.design_rules == custom_rules
        assert custom_engine.design_rules.min_trace_width == 0.15
    
    def test_verify_design_success(self, default_engine, sample_netlist):
        """Test successful design verification."""
        result = default_engine.verify_design(sample_netlist)
        
        assert result["success"] is True
        assert "summary" in result
        assert "violations" in result
        assert "design_rules" in result
    
    def test_verify_design_with_pcb(self, default_engine, sample_netlist, sample_pcb_data):
        """Test design verification with PCB data."""
        result = default_engine.verify_design(sample_netlist, sample_pcb_data)
        
        assert result["success"] is True
        # Should have DRC violations due to narrow traces and small vias
        assert result["summary"]["drc_violations"] > 0
    
    def test_component_pin_config(self, default_engine):
        """Test component pin configuration lookup."""
        # Test exact match
        config = default_engine._get_component_pin_config("resistor")
        assert config["pins"] == 2
        assert config["types"] == ["passive", "passive"]
        
        # Test partial match
        config = default_engine._get_component_pin_config("npn_transistor")
        assert config["pins"] == 3
        
        # Test unknown component
        config = default_engine._get_component_pin_config("unknown_component")
        assert config is None
    
    def test_component_connections(self, default_engine, sample_netlist):
        """Test component connection extraction."""
        connections = default_engine._get_component_connections("U1", sample_netlist["nets"])
        
        # U1 should be connected to pins 1, 2, 3, 4, 5
        assert len(connections) == 5
        assert "1" in connections
        assert "4" in connections  # VCC
        assert "5" in connections  # GND
    
    def test_erc_power_connections(self, default_engine, sample_netlist):
        """Test ERC power connection checking."""
        default_engine._perform_erc(sample_netlist)
        
        # Should not have power connection warnings (VCC and GND present)
        power_violations = [v for v in default_engine.violations 
                          if "power" in v.message.lower() or "ground" in v.message.lower()]
        
        # Should have ground connection (GND net exists)
        ground_violations = [v for v in power_violations if "ground" in v.message.lower()]
        assert len(ground_violations) == 0
    
    def test_erc_unconnected_pins(self, default_engine):
        """Test ERC unconnected pin detection."""
        # Create netlist with unconnected pins
        netlist = {
            "components": [
                {"reference": "R1", "type": "resistor", "value": "10k"}
            ],
            "nets": [
                {
                    "name": "NET1",
                    "connections": [
                        {"component": "R1", "pin": "1"}
                        # Pin 2 is unconnected
                    ]
                }
            ]
        }
        
        default_engine._perform_erc(netlist)
        
        # Should detect unconnected pin
        unconnected_violations = [v for v in default_engine.violations 
                                if "unconnected" in v.message.lower()]
        assert len(unconnected_violations) > 0
    
    def test_erc_component_connections(self, default_engine):
        """Test ERC component connection validation."""
        # Create netlist with wrong number of connections
        netlist = {
            "components": [
                {"reference": "R1", "type": "resistor", "value": "10k"}
            ],
            "nets": [
                {
                    "name": "NET1",
                    "connections": [
                        {"component": "R1", "pin": "1"},
                        {"component": "R1", "pin": "2"},
                        {"component": "R1", "pin": "3"}  # Resistor should only have 2 pins
                    ]
                }
            ]
        }
        
        default_engine._perform_erc(netlist)
        
        # Should detect incorrect number of connections
        connection_violations = [v for v in default_engine.violations 
                               if v.component == "R1" and "connections" in v.message]
        assert len(connection_violations) > 0
    
    def test_drc_trace_widths(self, default_engine, sample_pcb_data):
        """Test DRC trace width checking."""
        default_engine._perform_drc(sample_pcb_data)
        
        # Should detect narrow trace violation
        trace_violations = [v for v in default_engine.violations 
                          if v.violation_type == ViolationType.DRC_ERROR 
                          and "trace width" in v.message.lower()]
        assert len(trace_violations) > 0
        
        # Check specific violation
        narrow_violation = next((v for v in trace_violations 
                               if "0.05" in v.message), None)
        assert narrow_violation is not None
        assert narrow_violation.net == "SIGNAL"
    
    def test_drc_via_sizes(self, default_engine, sample_pcb_data):
        """Test DRC via size checking."""
        default_engine._perform_drc(sample_pcb_data)
        
        # Should detect small via violation
        via_violations = [v for v in default_engine.violations 
                         if v.violation_type == ViolationType.DRC_ERROR 
                         and "via size" in v.message.lower()]
        assert len(via_violations) > 0
        
        # Check specific violation
        small_violation = next((v for v in via_violations 
                              if "0.1" in v.message), None)
        assert small_violation is not None
        assert small_violation.location == (20.0, 20.0)
    
    def test_drc_component_spacing(self, default_engine, sample_pcb_data):
        """Test DRC component spacing checking."""
        default_engine._perform_drc(sample_pcb_data)
        
        # Should detect close component spacing
        spacing_violations = [v for v in default_engine.violations 
                            if "spacing" in v.message.lower() 
                            and "component" in v.message.lower()]
        assert len(spacing_violations) > 0
    
    def test_drc_manufacturing_rules(self, default_engine, sample_pcb_data):
        """Test DRC manufacturing rule checking."""
        default_engine._perform_drc(sample_pcb_data)
        
        # Should detect small drill hole
        drill_violations = [v for v in default_engine.violations 
                          if "drill size" in v.message.lower()]
        assert len(drill_violations) > 0
    
    def test_connectivity_validation(self, default_engine, problematic_netlist):
        """Test connectivity validation."""
        default_engine._validate_connectivity(problematic_netlist)
        
        # Should detect single-pin net
        single_pin_violations = [v for v in default_engine.violations 
                               if v.violation_type == ViolationType.CONNECTIVITY_ERROR 
                               and "one connection" in v.message]
        assert len(single_pin_violations) > 0
        
        # Should detect empty net
        empty_net_violations = [v for v in default_engine.violations 
                              if v.violation_type == ViolationType.CONNECTIVITY_ERROR 
                              and "no connections" in v.message]
        assert len(empty_net_violations) > 0
    
    def test_verification_report_generation(self, default_engine, sample_netlist, sample_pcb_data):
        """Test verification report generation."""
        result = default_engine.verify_design(sample_netlist, sample_pcb_data)
        
        # Check report structure
        assert "success" in result
        assert "ready_for_manufacturing" in result
        assert "summary" in result
        assert "violations" in result
        assert "design_rules" in result
        
        # Check summary structure
        summary = result["summary"]
        assert "total_violations" in summary
        assert "errors" in summary
        assert "warnings" in summary
        assert "erc_violations" in summary
        assert "drc_violations" in summary
        assert "connectivity_violations" in summary
        
        # Check violation structure
        if result["violations"]:
            violation = result["violations"][0]
            assert "type" in violation
            assert "severity" in violation
            assert "message" in violation
    
    def test_get_violations_by_severity(self, default_engine, sample_pcb_data):
        """Test filtering violations by severity."""
        # Generate some violations
        default_engine._perform_drc(sample_pcb_data)
        
        errors = default_engine.get_violations_by_severity(Severity.ERROR)
        warnings = default_engine.get_violations_by_severity(Severity.WARNING)
        
        # Should have some errors and warnings
        assert len(errors) > 0
        assert all(v.severity == Severity.ERROR for v in errors)
        
        if warnings:
            assert all(v.severity == Severity.WARNING for v in warnings)
    
    def test_get_violations_by_type(self, default_engine, sample_netlist, sample_pcb_data):
        """Test filtering violations by type."""
        # Generate violations
        default_engine.verify_design(sample_netlist, sample_pcb_data)
        
        erc_violations = default_engine.get_violations_by_type(ViolationType.ERC_ERROR)
        drc_violations = default_engine.get_violations_by_type(ViolationType.DRC_ERROR)
        
        # Check that filtering works
        if erc_violations:
            assert all(v.violation_type == ViolationType.ERC_ERROR for v in erc_violations)
        
        if drc_violations:
            assert all(v.violation_type == ViolationType.DRC_ERROR for v in drc_violations)
    
    def test_clear_violations(self, default_engine, sample_pcb_data):
        """Test clearing violations."""
        # Generate some violations
        default_engine._perform_drc(sample_pcb_data)
        assert len(default_engine.violations) > 0
        
        # Clear violations
        default_engine.clear_violations()
        assert len(default_engine.violations) == 0
    
    def test_custom_design_rules(self, custom_engine, sample_pcb_data):
        """Test verification with custom design rules."""
        result = custom_engine.verify_design({}, sample_pcb_data)
        
        # Should use custom rules (more strict)
        assert result["design_rules"]["min_trace_width"] == 0.15
        assert result["design_rules"]["min_via_size"] == 0.25
        
        # Should have more violations due to stricter rules
        violations = result["violations"]
        trace_violations = [v for v in violations if "trace width" in v["message"]]
        
        # 0.2mm trace should now violate 0.15mm minimum
        strict_violations = [v for v in trace_violations if "0.2" in v["message"]]
        assert len(strict_violations) > 0
    
    def test_error_handling(self, default_engine):
        """Test error handling in verification."""
        # Test with invalid data
        result = default_engine.verify_design(None)
        
        assert result["success"] is False
        assert "error" in result
        assert result["violations"] == []
    
    def test_is_output_pin(self, default_engine):
        """Test output pin detection heuristic."""
        # Test output pin detection
        assert default_engine._is_output_pin("U1", "out") is True
        assert default_engine._is_output_pin("U1", "output") is True
        assert default_engine._is_output_pin("U1", "q") is True
        assert default_engine._is_output_pin("U1", "y") is True
        
        # Test non-output pins
        assert default_engine._is_output_pin("U1", "in") is False
        assert default_engine._is_output_pin("U1", "input") is False
        assert default_engine._is_output_pin("U1", "1") is False
    
    def test_spacing_calculations(self, default_engine):
        """Test spacing calculation methods."""
        trace1 = {"x": 0.0, "y": 0.0}
        trace2 = {"x": 3.0, "y": 4.0}
        
        spacing = default_engine._calculate_trace_spacing(trace1, trace2)
        assert spacing == 5.0  # 3-4-5 triangle
        
        comp1 = {"x": 0.0, "y": 0.0}
        comp2 = {"x": 1.0, "y": 1.0}
        
        spacing = default_engine._calculate_component_spacing(comp1, comp2)
        assert spacing == pytest.approx(1.414, rel=1e-3)  # sqrt(2)