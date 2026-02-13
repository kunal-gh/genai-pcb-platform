"""
Unit tests for component library integration.

Tests ComponentLibrary class functionality.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.database import Base
from src.models.component import Component, Manufacturer, ComponentCategory, PackageType
from src.services.component_library import ComponentLibrary


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    
    Manufacturer.__table__.create(engine, checkfirst=True)
    Component.__table__.create(engine, checkfirst=True)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()


@pytest.fixture
def sample_manufacturer(db_session):
    """Create a sample manufacturer."""
    manufacturer = Manufacturer(name="Test Manufacturer")
    db_session.add(manufacturer)
    db_session.commit()
    return manufacturer


@pytest.fixture
def sample_components(db_session, sample_manufacturer):
    """Create sample components."""
    components = [
        Component(
            part_number="R1K-0805",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            package_name="0805",
            electrical_parameters={
                "resistance": {"value": 1000, "unit": "ohm"},
                "power_rating": {"value": 0.125, "unit": "W"}
            },
            symbol_id="Device:R",
            footprint_id="Resistor_SMD:R_0805_2012Metric",
            in_stock=True,
            lifecycle_status="active"
        ),
        Component(
            part_number="C100N-0603",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.CAPACITOR,
            package_type=PackageType.SMD,
            package_name="0603",
            electrical_parameters={
                "capacitance": {"value": 100e-9, "unit": "F"},
                "voltage_rating": {"value": 16, "unit": "V"}
            },
            symbol_id="Device:C",
            footprint_id="Capacitor_SMD:C_0603_1608Metric",
            in_stock=True,
            lifecycle_status="active"
        ),
        Component(
            part_number="LED-RED-0805",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.LED,
            package_type=PackageType.SMD,
            package_name="0805",
            electrical_parameters={
                "forward_voltage": {"value": 2.0, "unit": "V"},
                "forward_current": {"value": 0.020, "unit": "A"}
            },
            symbol_id="Device:LED",
            footprint_id="LED_SMD:LED_0805_2012Metric",
            in_stock=True,
            lifecycle_status="active"
        )
    ]
    db_session.add_all(components)
    db_session.commit()
    return components


class TestComponentLibrary:
    """Tests for ComponentLibrary class."""
    
    def test_init(self, db_session):
        """Test initialization."""
        library = ComponentLibrary(db_session)
        
        assert library.db == db_session
        assert library.selector is not None
        assert len(library.library_mappings) > 0
    
    def test_lookup_symbol_by_part_number(self, db_session, sample_components):
        """Test looking up symbol by part number."""
        library = ComponentLibrary(db_session)
        
        result = library.lookup_symbol(part_number="R1K-0805")
        
        assert result["library"] == "Device"
        assert result["symbol"] == "R"
        assert result["footprint"] == "Resistor_SMD:R_0805_2012Metric"
        assert result["component"] is not None
        assert result["component"].part_number == "R1K-0805"
    
    def test_lookup_symbol_by_category(self, db_session, sample_components):
        """Test looking up symbol by category."""
        library = ComponentLibrary(db_session)
        
        result = library.lookup_symbol(category=ComponentCategory.CAPACITOR)
        
        assert result["library"] == "Device"
        assert result["symbol"] == "C"
        assert result["component"] is not None
    
    def test_lookup_symbol_not_found(self, db_session):
        """Test looking up non-existent symbol."""
        library = ComponentLibrary(db_session)
        
        result = library.lookup_symbol(part_number="NONEXISTENT")
        
        # Should return default symbol
        assert result["library"] is not None
        assert result["symbol"] is not None
        assert result["component"] is None
    
    def test_lookup_symbol_with_alternatives(self, db_session, sample_components):
        """Test that lookup includes alternatives."""
        library = ComponentLibrary(db_session)
        
        result = library.lookup_symbol(part_number="R1K-0805")
        
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)
    
    def test_get_library_name(self, db_session, sample_components):
        """Test getting library name from component."""
        library = ComponentLibrary(db_session)
        component = sample_components[0]
        
        lib_name = library._get_library_name(component)
        
        assert lib_name == "Device"
    
    def test_get_symbol_name(self, db_session, sample_components):
        """Test getting symbol name from component."""
        library = ComponentLibrary(db_session)
        component = sample_components[0]
        
        symbol_name = library._get_symbol_name(component)
        
        assert symbol_name == "R"
    
    def test_get_default_footprint_resistor(self, db_session, sample_components):
        """Test getting default footprint for resistor."""
        library = ComponentLibrary(db_session)
        component = sample_components[0]
        
        footprint = library._get_default_footprint(component)
        
        assert "Resistor_SMD" in footprint
        assert "0805" in footprint
    
    def test_get_default_symbol(self, db_session):
        """Test getting default symbol for category."""
        library = ComponentLibrary(db_session)
        
        result = library._get_default_symbol(ComponentCategory.RESISTOR)
        
        assert result["library"] == "Device"
        assert result["symbol"] == "R"
    
    def test_validate_component_valid(self, db_session):
        """Test validating valid component."""
        library = ComponentLibrary(db_session)
        
        result = library.validate_component("Device", "R", "10k")
        
        assert result["valid"] is True
        assert len(result["warnings"]) == 0
    
    def test_validate_component_non_standard_library(self, db_session):
        """Test validating component with non-standard library."""
        library = ComponentLibrary(db_session)
        
        result = library.validate_component("CustomLib", "R", "10k")
        
        assert result["valid"] is True
        assert any("Non-standard" in w for w in result["warnings"])
    
    def test_validate_component_invalid_symbol(self, db_session):
        """Test validating component with invalid symbol."""
        library = ComponentLibrary(db_session)
        
        result = library.validate_component("Device", "", "10k")
        
        assert result["valid"] is False
        assert any("Invalid symbol" in w for w in result["warnings"])
    
    def test_validate_value_format_resistor(self, db_session):
        """Test validating resistor value format."""
        library = ComponentLibrary(db_session)
        
        assert library._validate_value_format("R", "10k") is True
        assert library._validate_value_format("R", "1M") is True
        assert library._validate_value_format("R", "100ohm") is True
        assert library._validate_value_format("R", "4.7k") is True
    
    def test_validate_value_format_capacitor(self, db_session):
        """Test validating capacitor value format."""
        library = ComponentLibrary(db_session)
        
        assert library._validate_value_format("C", "100nF") is True
        assert library._validate_value_format("C", "10uF") is True
        assert library._validate_value_format("C", "1pF") is True
    
    def test_validate_value_format_inductor(self, db_session):
        """Test validating inductor value format."""
        library = ComponentLibrary(db_session)
        
        assert library._validate_value_format("L", "10uH") is True
        assert library._validate_value_format("L", "1mH") is True
    
    def test_find_missing_components(self, db_session, sample_components):
        """Test finding missing components in SKiDL code."""
        library = ComponentLibrary(db_session)
        
        code = """
from skidl import Part

r1 = Part('Device', 'R', value='10k')
unknown = Part('CustomLib', 'UnknownPart')
"""
        
        missing = library.find_missing_components(code)
        
        # Should find the unknown part
        assert len(missing) >= 1
        assert any(m["part"] == "UnknownPart" for m in missing)
    
    def test_suggest_alternatives_resistor(self, db_session, sample_components):
        """Test suggesting alternatives for resistor."""
        library = ComponentLibrary(db_session)
        
        alternatives = library._suggest_alternatives("Device", "R")
        
        assert isinstance(alternatives, list)
        # Should find the resistor in database
        if alternatives:
            assert any("R1K" in alt.get("part_number", "") for alt in alternatives)
    
    def test_get_component_info(self, db_session, sample_components):
        """Test getting component information."""
        library = ComponentLibrary(db_session)
        
        info = library.get_component_info("R1K-0805")
        
        assert info is not None
        assert info["part_number"] == "R1K-0805"
        assert info["category"] == "resistor"
        assert info["package"] == "0805"
        assert "electrical_parameters" in info
    
    def test_get_component_info_not_found(self, db_session):
        """Test getting info for non-existent component."""
        library = ComponentLibrary(db_session)
        
        info = library.get_component_info("NONEXISTENT")
        
        assert info is None
    
    def test_library_mappings(self, db_session):
        """Test that library mappings are defined."""
        library = ComponentLibrary(db_session)
        
        assert ComponentCategory.RESISTOR in library.library_mappings
        assert ComponentCategory.CAPACITOR in library.library_mappings
        assert ComponentCategory.LED in library.library_mappings
    
    def test_lookup_symbol_with_electrical_params(self, db_session, sample_components):
        """Test looking up symbol with electrical parameters."""
        library = ComponentLibrary(db_session)
        
        result = library.lookup_symbol(
            category=ComponentCategory.RESISTOR,
            electrical_params={"resistance": {"value": 1000, "tolerance": 0.1}}
        )
        
        assert result["component"] is not None
        assert result["library"] == "Device"
        assert result["symbol"] == "R"
    
    def test_validate_component_with_suggestions(self, db_session):
        """Test that validation provides suggestions."""
        library = ComponentLibrary(db_session)
        
        result = library.validate_component("Device", "R", "invalid_value")
        
        assert "suggestions" in result
        assert isinstance(result["suggestions"], list)
