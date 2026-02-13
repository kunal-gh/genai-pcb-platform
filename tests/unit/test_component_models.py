"""
Unit tests for component database models.

Tests Component, Manufacturer, and related models.
"""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid

from src.models.database import Base
from src.models.component import (
    Component,
    Manufacturer,
    ComponentCategory,
    PackageType
)


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    
    # Only create tables we need for component tests
    Manufacturer.__table__.create(engine, checkfirst=True)
    Component.__table__.create(engine, checkfirst=True)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()


@pytest.fixture
def sample_manufacturer(db_session):
    """Create a sample manufacturer for testing."""
    manufacturer = Manufacturer(
        name="Texas Instruments",
        website="https://www.ti.com",
        description="Leading semiconductor manufacturer"
    )
    db_session.add(manufacturer)
    db_session.commit()
    return manufacturer


class TestManufacturer:
    """Tests for Manufacturer model."""
    
    def test_create_manufacturer(self, db_session):
        """Test creating a manufacturer."""
        manufacturer = Manufacturer(
            name="Analog Devices",
            website="https://www.analog.com",
            description="High-performance analog ICs"
        )
        db_session.add(manufacturer)
        db_session.commit()
        
        assert manufacturer.id is not None
        assert manufacturer.name == "Analog Devices"
        assert manufacturer.website == "https://www.analog.com"
        assert manufacturer.created_at is not None
    
    def test_manufacturer_unique_name(self, db_session, sample_manufacturer):
        """Test that manufacturer names must be unique."""
        duplicate = Manufacturer(name="Texas Instruments")
        db_session.add(duplicate)
        
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()
    
    def test_manufacturer_repr(self, sample_manufacturer):
        """Test manufacturer string representation."""
        repr_str = repr(sample_manufacturer)
        assert "Manufacturer" in repr_str
        assert "Texas Instruments" in repr_str


class TestComponent:
    """Tests for Component model."""
    
    def test_create_resistor(self, db_session, sample_manufacturer):
        """Test creating a resistor component."""
        component = Component(
            part_number="RC0805FR-0710KL",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            description="10K ohm resistor, 1%, 0805",
            package_type=PackageType.SMD,
            package_name="0805",
            electrical_parameters={
                "resistance": {"value": 10000, "unit": "ohm", "tolerance": 0.01},
                "power_rating": {"value": 0.125, "unit": "W"},
                "temperature_coefficient": {"value": 100, "unit": "ppm/C"}
            },
            footprint_id="Resistor_SMD:R_0805_2012Metric",
            symbol_id="Device:R",
            in_stock=True,
            lifecycle_status="active",
            pricing=[
                {"quantity": 1, "price": 0.10},
                {"quantity": 100, "price": 0.05},
                {"quantity": 1000, "price": 0.02}
            ]
        )
        db_session.add(component)
        db_session.commit()
        
        assert component.id is not None
        assert component.part_number == "RC0805FR-0710KL"
        assert component.category == ComponentCategory.RESISTOR
        assert component.package_type == PackageType.SMD
    
    def test_create_capacitor(self, db_session, sample_manufacturer):
        """Test creating a capacitor component."""
        component = Component(
            part_number="GRM188R71C104KA01D",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.CAPACITOR,
            description="100nF ceramic capacitor, X7R, 0603",
            package_type=PackageType.SMD,
            package_name="0603",
            electrical_parameters={
                "capacitance": {"value": 100e-9, "unit": "F", "tolerance": 0.10},
                "voltage_rating": {"value": 16, "unit": "V"},
                "dielectric": "X7R"
            },
            footprint_id="Capacitor_SMD:C_0603_1608Metric",
            symbol_id="Device:C",
            in_stock=True,
            lifecycle_status="active"
        )
        db_session.add(component)
        db_session.commit()
        
        assert component.category == ComponentCategory.CAPACITOR
        assert component.electrical_parameters["capacitance"]["value"] == 100e-9
    
    def test_create_ic(self, db_session, sample_manufacturer):
        """Test creating an IC component."""
        component = Component(
            part_number="LM358DR",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.IC,
            description="Dual operational amplifier",
            package_type=PackageType.SOIC,
            package_name="SOIC-8",
            electrical_parameters={
                "supply_voltage_min": {"value": 3.0, "unit": "V"},
                "supply_voltage_max": {"value": 32, "unit": "V"},
                "input_offset_voltage": {"value": 7e-3, "unit": "V"},
                "gain_bandwidth": {"value": 1e6, "unit": "Hz"}
            },
            footprint_id="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
            symbol_id="Amplifier_Operational:LM358",
            in_stock=True,
            lifecycle_status="active"
        )
        db_session.add(component)
        db_session.commit()
        
        assert component.category == ComponentCategory.IC
        assert component.package_type == PackageType.SOIC
    
    def test_component_unique_part_number(self, db_session, sample_manufacturer):
        """Test that part numbers must be unique."""
        component1 = Component(
            part_number="TEST123",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={}
        )
        db_session.add(component1)
        db_session.commit()
        
        component2 = Component(
            part_number="TEST123",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.CAPACITOR,
            package_type=PackageType.SMD,
            electrical_parameters={}
        )
        db_session.add(component2)
        
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()
    
    def test_get_parameter(self, db_session, sample_manufacturer):
        """Test getting electrical parameters."""
        component = Component(
            part_number="TEST_PARAM",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={
                "resistance": {"value": 1000, "unit": "ohm"},
                "power_rating": {"value": 0.25, "unit": "W"}
            }
        )
        
        resistance = component.get_parameter("resistance")
        assert resistance is not None
        assert resistance["value"] == 1000
        assert resistance["unit"] == "ohm"
        
        power = component.get_parameter("power_rating")
        assert power["value"] == 0.25
        
        missing = component.get_parameter("nonexistent")
        assert missing is None
    
    def test_get_parameter_empty(self, db_session, sample_manufacturer):
        """Test getting parameters when none exist."""
        component = Component(
            part_number="TEST_EMPTY",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters=None
        )
        
        result = component.get_parameter("resistance")
        assert result is None
    
    def test_get_price_for_quantity(self, db_session, sample_manufacturer):
        """Test price calculation for different quantities."""
        component = Component(
            part_number="TEST_PRICE",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={},
            pricing=[
                {"quantity": 1, "price": 0.10},
                {"quantity": 100, "price": 0.05},
                {"quantity": 1000, "price": 0.02}
            ]
        )
        
        # Test different quantity tiers
        assert component.get_price_for_quantity(1) == 0.10
        assert component.get_price_for_quantity(50) == 0.10
        assert component.get_price_for_quantity(100) == 0.05
        assert component.get_price_for_quantity(500) == 0.05
        assert component.get_price_for_quantity(1000) == 0.02
        assert component.get_price_for_quantity(5000) == 0.02
    
    def test_get_price_no_pricing(self, db_session, sample_manufacturer):
        """Test price calculation when no pricing available."""
        component = Component(
            part_number="TEST_NO_PRICE",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={},
            pricing=None
        )
        
        assert component.get_price_for_quantity(100) == 0.0
    
    def test_component_with_suppliers(self, db_session, sample_manufacturer):
        """Test component with supplier information."""
        component = Component(
            part_number="TEST_SUPPLIERS",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={},
            suppliers=[
                {"name": "DigiKey", "sku": "123-456", "url": "https://digikey.com/..."},
                {"name": "Mouser", "sku": "789-012", "url": "https://mouser.com/..."}
            ]
        )
        db_session.add(component)
        db_session.commit()
        
        assert len(component.suppliers) == 2
        assert component.suppliers[0]["name"] == "DigiKey"
        assert component.suppliers[1]["name"] == "Mouser"
    
    def test_component_lifecycle_status(self, db_session, sample_manufacturer):
        """Test component lifecycle status."""
        active = Component(
            part_number="ACTIVE_PART",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={},
            lifecycle_status="active",
            in_stock=True
        )
        
        obsolete = Component(
            part_number="OBSOLETE_PART",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={},
            lifecycle_status="obsolete",
            in_stock=False
        )
        
        db_session.add_all([active, obsolete])
        db_session.commit()
        
        assert active.lifecycle_status == "active"
        assert active.in_stock is True
        assert obsolete.lifecycle_status == "obsolete"
        assert obsolete.in_stock is False
    
    def test_component_repr(self, db_session, sample_manufacturer):
        """Test component string representation."""
        component = Component(
            part_number="TEST_REPR",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.LED,
            package_type=PackageType.SMD,
            electrical_parameters={}
        )
        
        repr_str = repr(component)
        assert "Component" in repr_str
        assert "TEST_REPR" in repr_str
        assert "led" in repr_str.lower()
    
    def test_component_relationship(self, db_session, sample_manufacturer):
        """Test component-manufacturer relationship."""
        component = Component(
            part_number="TEST_REL",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={}
        )
        db_session.add(component)
        db_session.commit()
        
        # Test relationship
        assert component.manufacturer.name == "Texas Instruments"
        assert component in sample_manufacturer.components
    
    def test_query_by_category(self, db_session, sample_manufacturer):
        """Test querying components by category."""
        resistor = Component(
            part_number="R1",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={}
        )
        capacitor = Component(
            part_number="C1",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.CAPACITOR,
            package_type=PackageType.SMD,
            electrical_parameters={}
        )
        db_session.add_all([resistor, capacitor])
        db_session.commit()
        
        resistors = db_session.query(Component).filter(
            Component.category == ComponentCategory.RESISTOR
        ).all()
        
        assert len(resistors) == 1
        assert resistors[0].part_number == "R1"
    
    def test_query_by_package_type(self, db_session, sample_manufacturer):
        """Test querying components by package type."""
        smd = Component(
            part_number="SMD1",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={}
        )
        through_hole = Component(
            part_number="TH1",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.THROUGH_HOLE,
            electrical_parameters={}
        )
        db_session.add_all([smd, through_hole])
        db_session.commit()
        
        smd_components = db_session.query(Component).filter(
            Component.package_type == PackageType.SMD
        ).all()
        
        assert len(smd_components) == 1
        assert smd_components[0].part_number == "SMD1"
