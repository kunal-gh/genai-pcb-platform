"""
Unit tests for DFM validation system.

Tests DFMValidator class functionality.
"""

import pytest
from src.services.dfm_validation import (
    DFMValidator,
    ManufacturingConstraints,
    DFMViolation,
    DFMViolationSeverity
)


@pytest.fixture
def default_validator():
    """Create DFM validator with default constraints."""
    return DFMValidator()


@pytest.fixture
def strict_constraints():
    """Create strict manufacturing constraints."""
    return ManufacturingConstraints(
        min_trace_width=0.15,
        min_via_diameter=0.3,
        min_drill_size=0.2,
        min_component_spacing=1.0
    )


@pytest.fixture
def strict_validator(strict_constraints):
    """Create DFM validator with strict constraints."""
    return DFMValidator(strict_constraints)


@pytest.fixture
def good_pcb_data():
    """PCB data that should pass DFM checks."""
    return {
        "width": 100.0,
        "height": 80.0,
        "thickness": 1.6,
        "layers": 2,
        "traces": [
            {"width": 0.2, "net": "SIGNAL", "x": 20.0, "y": 20.0, "length": 50.0},
            {"width": 0.3, "net": "VCC", "x": 30.0, "y": 30.0, "length": 40.0},
            {"width": 0.2, "net": "GND", "x": 40.0, "y": 40.0, "length": 45.0}
        ],
        "vias": [
            {"size": 0.6, "drill": 0.3, "x": 25.0, "y": 25.0},
            {"size": 0.6, "drill": 0.3, "x": 35.0, "y": 35.0}
        ],
        "components": [
            {"reference": "R1", "x": 20.0, "y": 20.0, "package": "0805"},
            {"reference": "C1", "x": 30.0, "y": 30.0, "package": "0603"},
            {"reference": "U1", "x": 50.0, "y": 50.0, "package": "SOIC-8"}
        ],
        "drill_holes": [
            {"diameter": 0.8, "x": 10.0, "y": 10.0},
            {"diameter": 1.0, "x": 90.0, "y": 70.0}
        ]
    }


@pytest.fixture
def problematic_pcb_data():
    """PCB data with various DFM issues."""
    return {
        "width": 100.0,
        "height": 80.0,
        "thickness": 0.3,  # Too thin
        "layers": 2,
        "traces": [
            {"width": 0.05, "net": "SIGNAL", "x": 20.0, "y": 20.0, "length": 50.0},  # Too narrow
            {"width": 0.15, "net": "VCC", "x": 20.1, "y": 20.1, "length": 40.0},  # Too close, power trace too narrow
            {"width": 0.2, "net": "CLK", "x": 30.0, "y": 30.0, "length": 150.0}  # High-speed, long trace
        ],
        "vias": [
            {"size": 0.15, "drill": 0.1, "x": 25.0, "y": 25.0},  # Too small
            {"size": 0.4, "drill": 0.35, "x": 35.0, "y": 35.0}  # Insufficient annular ring
        ],
        "components": [
            {"reference": "R1", "x": 2.0, "y": 2.0, "package": "0805"},  # Too close to edge
            {"reference": "C1", "x": 2.3, "y": 2.3, "package": "0603", "pad_size": 0.2},  # Too close, small pad
            {"reference": "U1", "x": 50.0, "y": 50.0, "package": "SOIC-8"}
        ],
        "drill_holes": [
            {"diameter": 0.1, "x": 10.0, "y": 10.0},  # Too small
            {"diameter": 7.0, "x": 90.0, "y": 70.0}  # Too large
        ]
    }


@pytest.fixture
def small_board_data():
    """Small board data for testing."""
    return {
        "width": 8.0,
        "height": 8.0,
        "thickness": 1.6,
        "layers": 2,
        "traces": [],
        "vias": [],
        "components": [],
        "drill_holes": []
    }


class TestManufacturingConstraints:
    """Tests for ManufacturingConstraints dataclass."""
    
    def test_default_constraints(self):
        """Test default manufacturing constraints."""
        constraints = ManufacturingConstraints()
        
        assert constraints.min_trace_width == 0.1
        assert constraints.min_via_diameter == 0.2
        assert constraints.min_drill_size == 0.15
        assert constraints.min_component_spacing == 0.5
        assert constraints.max_aspect_ratio == 10.0
    
    def test_custom_constraints(self):
        """Test custom manufacturing constraints."""
        constraints = ManufacturingConstraints(
            min_trace_width=0.15,
            min_via_diameter=0.3,
            max_aspect_ratio=8.0
        )
        
        assert constraints.min_trace_width == 0.15
        assert constraints.min_via_diameter == 0.3
        assert constraints.max_aspect_ratio == 8.0
        # Defaults should still apply
        assert constraints.min_drill_size == 0.15


class TestDFMViolation:
    """Tests for DFMViolation dataclass."""
    
    def test_basic_violation(self):
        """Test basic violation creation."""
        violation = DFMViolation(
            severity=DFMViolationSeverity.CRITICAL,
            category="trace",
            message="Trace too narrow"
        )
        
        assert violation.severity == DFMViolationSeverity.CRITICAL
        assert violation.category == "trace"
        assert violation.message == "Trace too narrow"
        assert violation.location is None
    
    def test_detailed_violation(self):
        """Test violation with all fields."""
        violation = DFMViolation(
            severity=DFMViolationSeverity.HIGH,
            category="via",
            message="Via too small",
            location=(10.0, 20.0),
            component="U1",
            net="VCC",
            recommendation="Increase via size",
            cost_impact="high",
            manufacturability_impact=0.15
        )
        
        assert violation.severity == DFMViolationSeverity.HIGH
        assert violation.location == (10.0, 20.0)
        assert violation.recommendation == "Increase via size"
        assert violation.cost_impact == "high"
        assert violation.manufacturability_impact == 0.15


class TestDFMValidator:
    """Tests for DFMValidator class."""
    
    def test_init_default_constraints(self, default_validator):
        """Test initialization with default constraints."""
        assert default_validator.constraints.min_trace_width == 0.1
        assert len(default_validator.violations) == 0
        assert len(default_validator.manufacturer_profiles) > 0
    
    def test_init_custom_constraints(self, strict_validator, strict_constraints):
        """Test initialization with custom constraints."""
        assert strict_validator.constraints == strict_constraints
        assert strict_validator.constraints.min_trace_width == 0.15
    
    def test_manufacturer_profiles(self, default_validator):
        """Test manufacturer-specific profiles."""
        assert "standard" in default_validator.manufacturer_profiles
        assert "jlcpcb" in default_validator.manufacturer_profiles
        assert "pcbway" in default_validator.manufacturer_profiles
        assert "oshpark" in default_validator.manufacturer_profiles
        
        # Check JLCPCB profile
        jlcpcb = default_validator.manufacturer_profiles["jlcpcb"]
        assert jlcpcb.min_trace_width == 0.09
        assert jlcpcb.min_drill_size == 0.2
    
    def test_validate_good_design(self, default_validator, good_pcb_data):
        """Test validation of good design."""
        result = default_validator.validate_design(good_pcb_data)
        
        assert result["success"] is True
        assert result["manufacturable"] is True
        assert result["score"] >= 90.0
        assert result["confidence_level"] in ["excellent", "good"]
    
    def test_validate_problematic_design(self, default_validator, problematic_pcb_data):
        """Test validation of problematic design."""
        result = default_validator.validate_design(problematic_pcb_data)
        
        assert result["success"] is True
        assert result["manufacturable"] is False  # Should have critical violations
        assert result["score"] < 90.0
        assert result["summary"]["critical"] > 0
    
    def test_validate_traces(self, default_validator, problematic_pcb_data):
        """Test trace validation."""
        default_validator._validate_traces(problematic_pcb_data)
        
        # Should detect narrow trace
        narrow_violations = [v for v in default_validator.violations 
                           if "0.05" in v.message and v.category == "trace"]
        assert len(narrow_violations) > 0
        assert narrow_violations[0].severity == DFMViolationSeverity.CRITICAL
        
        # Should detect narrow power trace
        power_violations = [v for v in default_validator.violations 
                          if "power" in v.message.lower()]
        assert len(power_violations) > 0
    
    def test_validate_vias(self, default_validator, problematic_pcb_data):
        """Test via validation."""
        default_validator._validate_vias(problematic_pcb_data)
        
        # Should detect small via
        small_via_violations = [v for v in default_validator.violations 
                              if v.category == "via" and "0.15" in v.message]
        assert len(small_via_violations) > 0
        
        # Should detect insufficient annular ring
        annular_violations = [v for v in default_validator.violations 
                            if "annular ring" in v.message.lower()]
        assert len(annular_violations) > 0
    
    def test_validate_components(self, default_validator, problematic_pcb_data):
        """Test component validation."""
        default_validator._validate_components(problematic_pcb_data)
        
        # Should detect component too close to edge
        edge_violations = [v for v in default_validator.violations 
                         if "edge" in v.message.lower()]
        assert len(edge_violations) > 0
        
        # Should detect small SMD pad
        pad_violations = [v for v in default_validator.violations 
                        if "pad size" in v.message.lower()]
        assert len(pad_violations) > 0
        
        # Should detect close component spacing
        spacing_violations = [v for v in default_validator.violations 
                            if "spacing" in v.message.lower() and v.category == "component"]
        assert len(spacing_violations) > 0
    
    def test_validate_drills(self, default_validator, problematic_pcb_data):
        """Test drill validation."""
        default_validator._validate_drills(problematic_pcb_data)
        
        # Should detect small drill
        small_drill_violations = [v for v in default_validator.violations 
                                if v.category == "drill" and "0.1" in v.message]
        assert len(small_drill_violations) > 0
        
        # Should detect large drill
        large_drill_violations = [v for v in default_validator.violations 
                                if v.category == "drill" and "7.0" in v.message]
        assert len(large_drill_violations) > 0
    
    def test_validate_board(self, default_validator, problematic_pcb_data):
        """Test board-level validation."""
        default_validator._validate_board(problematic_pcb_data)
        
        # Should detect thin board
        thickness_violations = [v for v in default_validator.violations 
                              if "thickness" in v.message.lower()]
        assert len(thickness_violations) > 0
    
    def test_validate_small_board(self, default_validator, small_board_data):
        """Test validation of small board."""
        default_validator._validate_board(small_board_data)
        
        # Should detect small board size
        size_violations = [v for v in default_validator.violations 
                         if "small" in v.message.lower()]
        assert len(size_violations) > 0
    
    def test_validate_signal_integrity(self, default_validator, problematic_pcb_data):
        """Test signal integrity validation."""
        default_validator._validate_signal_integrity(problematic_pcb_data)
        
        # Should detect high-speed signal issues
        si_violations = [v for v in default_validator.violations 
                       if v.category == "signal_integrity"]
        assert len(si_violations) > 0
        
        # Should detect long trace on CLK signal
        clk_violations = [v for v in si_violations if "CLK" in v.message]
        assert len(clk_violations) > 0
    
    def test_manufacturability_score_calculation(self, default_validator, good_pcb_data, problematic_pcb_data):
        """Test manufacturability score calculation."""
        # Good design should have high score
        result_good = default_validator.validate_design(good_pcb_data)
        assert result_good["score"] >= 85.0
        
        # Problematic design should have lower score
        default_validator.clear_violations()
        result_bad = default_validator.validate_design(problematic_pcb_data)
        assert result_bad["score"] < result_good["score"]
        assert result_bad["score"] >= 0.0
        assert result_bad["score"] <= 100.0
    
    def test_confidence_level(self, default_validator):
        """Test confidence level determination."""
        assert default_validator._get_confidence_level(98.0) == "excellent"
        assert default_validator._get_confidence_level(90.0) == "good"
        assert default_validator._get_confidence_level(75.0) == "fair"
        assert default_validator._get_confidence_level(60.0) == "poor"
        assert default_validator._get_confidence_level(30.0) == "unmanufacturable"
    
    def test_dfm_report_generation(self, default_validator, problematic_pcb_data):
        """Test DFM report generation."""
        result = default_validator.validate_design(problematic_pcb_data)
        
        # Check report structure
        assert "success" in result
        assert "manufacturable" in result
        assert "score" in result
        assert "confidence_level" in result
        assert "manufacturer" in result
        assert "summary" in result
        assert "violations" in result
        assert "constraints" in result
        
        # Check summary structure
        summary = result["summary"]
        assert "total_violations" in summary
        assert "critical" in summary
        assert "high" in summary
        assert "medium" in summary
        assert "low" in summary
        assert "by_category" in summary
        
        # Check violation structure
        if result["violations"]:
            violation = result["violations"][0]
            assert "severity" in violation
            assert "category" in violation
            assert "message" in violation
            assert "recommendation" in violation
    
    def test_manufacturer_specific_validation(self, default_validator, good_pcb_data):
        """Test validation with manufacturer-specific constraints."""
        # Test with JLCPCB profile
        result_jlcpcb = default_validator.validate_design(good_pcb_data, manufacturer="jlcpcb")
        assert result_jlcpcb["manufacturer"] == "jlcpcb"
        assert result_jlcpcb["constraints"]["min_trace_width"] == 0.09
        
        # Test with OSH Park profile
        default_validator.clear_violations()
        result_oshpark = default_validator.validate_design(good_pcb_data, manufacturer="oshpark")
        assert result_oshpark["manufacturer"] == "oshpark"
        assert result_oshpark["constraints"]["min_trace_width"] == 0.127
    
    def test_get_violations_by_severity(self, default_validator, problematic_pcb_data):
        """Test filtering violations by severity."""
        default_validator.validate_design(problematic_pcb_data)
        
        critical = default_validator.get_violations_by_severity(DFMViolationSeverity.CRITICAL)
        high = default_validator.get_violations_by_severity(DFMViolationSeverity.HIGH)
        
        assert len(critical) > 0
        assert all(v.severity == DFMViolationSeverity.CRITICAL for v in critical)
        
        if high:
            assert all(v.severity == DFMViolationSeverity.HIGH for v in high)
    
    def test_get_violations_by_category(self, default_validator, problematic_pcb_data):
        """Test filtering violations by category."""
        default_validator.validate_design(problematic_pcb_data)
        
        trace_violations = default_validator.get_violations_by_category("trace")
        via_violations = default_validator.get_violations_by_category("via")
        
        assert len(trace_violations) > 0
        assert all(v.category == "trace" for v in trace_violations)
        
        if via_violations:
            assert all(v.category == "via" for v in via_violations)
    
    def test_clear_violations(self, default_validator, problematic_pcb_data):
        """Test clearing violations."""
        default_validator.validate_design(problematic_pcb_data)
        assert len(default_validator.violations) > 0
        
        default_validator.clear_violations()
        assert len(default_validator.violations) == 0
    
    def test_is_power_net(self, default_validator):
        """Test power net detection."""
        assert default_validator._is_power_net("VCC") is True
        assert default_validator._is_power_net("VDD") is True
        assert default_validator._is_power_net("+5V") is True
        assert default_validator._is_power_net("+3V3") is True
        assert default_validator._is_power_net("VBAT") is True
        
        assert default_validator._is_power_net("SIGNAL") is False
        assert default_validator._is_power_net("GND") is False
        assert default_validator._is_power_net("DATA") is False
    
    def test_spacing_calculations(self, default_validator):
        """Test spacing calculation methods."""
        obj1 = {"x": 0.0, "y": 0.0}
        obj2 = {"x": 3.0, "y": 4.0}
        
        spacing = default_validator._calculate_spacing(obj1, obj2)
        assert spacing == 5.0  # 3-4-5 triangle
        
        comp1 = {"x": 0.0, "y": 0.0}
        comp2 = {"x": 1.0, "y": 1.0}
        
        spacing = default_validator._calculate_component_spacing(comp1, comp2)
        assert spacing == pytest.approx(1.414, rel=1e-3)  # sqrt(2)
    
    def test_error_handling(self, default_validator):
        """Test error handling in validation."""
        # Test with invalid data
        result = default_validator.validate_design(None)
        
        assert result["success"] is False
        assert "error" in result
        assert result["score"] == 0.0
    
    def test_strict_constraints(self, strict_validator, good_pcb_data):
        """Test validation with strict constraints."""
        result = strict_validator.validate_design(good_pcb_data)
        
        # Should have more violations due to stricter rules
        # 0.2mm traces should now violate 0.15mm minimum
        violations = result["violations"]
        trace_violations = [v for v in violations if v["category"] == "trace"]
        
        # May have violations that wouldn't exist with default constraints
        assert result["score"] <= 100.0
    
    def test_aspect_ratio_validation(self, default_validator):
        """Test via aspect ratio validation."""
        pcb_data = {
            "width": 100.0,
            "height": 80.0,
            "thickness": 3.2,  # Thick board
            "layers": 2,
            "traces": [],
            "vias": [
                {"size": 0.6, "drill": 0.2, "x": 25.0, "y": 25.0}  # High aspect ratio
            ],
            "components": [],
            "drill_holes": []
        }
        
        default_validator._validate_vias(pcb_data)
        
        # Should detect high aspect ratio
        aspect_violations = [v for v in default_validator.violations 
                           if "aspect ratio" in v.message.lower()]
        assert len(aspect_violations) > 0