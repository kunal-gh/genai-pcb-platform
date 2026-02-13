"""
Unit tests for component selection and recommendation engine.

Tests ComponentSelector class functionality.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.database import Base
from src.models.component import Component, Manufacturer, ComponentCategory, PackageType
from src.services.component_selector import ComponentSelector


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    
    # Only create tables we need
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
def sample_resistors(db_session, sample_manufacturer):
    """Create sample resistor components."""
    resistors = [
        Component(
            part_number="R1K-0805",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            package_name="0805",
            electrical_parameters={
                "resistance": {"value": 1000, "unit": "ohm", "tolerance": 0.01},
                "power_rating": {"value": 0.125, "unit": "W"}
            },
            in_stock=True,
            lifecycle_status="active",
            pricing=[{"quantity": 1, "price": 0.10}]
        ),
        Component(
            part_number="R10K-0805",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            package_name="0805",
            electrical_parameters={
                "resistance": {"value": 10000, "unit": "ohm", "tolerance": 0.01},
                "power_rating": {"value": 0.125, "unit": "W"}
            },
            in_stock=True,
            lifecycle_status="active",
            pricing=[{"quantity": 1, "price": 0.12}]
        ),
        Component(
            part_number="R10K-TH",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.THROUGH_HOLE,
            package_name="TH",
            electrical_parameters={
                "resistance": {"value": 10000, "unit": "ohm", "tolerance": 0.05},
                "power_rating": {"value": 0.25, "unit": "W"}
            },
            in_stock=True,
            lifecycle_status="active",
            pricing=[{"quantity": 1, "price": 0.08}]
        )
    ]
    db_session.add_all(resistors)
    db_session.commit()
    return resistors


@pytest.fixture
def sample_capacitors(db_session, sample_manufacturer):
    """Create sample capacitor components."""
    capacitors = [
        Component(
            part_number="C100N-0603",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.CAPACITOR,
            package_type=PackageType.SMD,
            package_name="0603",
            electrical_parameters={
                "capacitance": {"value": 100e-9, "unit": "F", "tolerance": 0.10},
                "voltage_rating": {"value": 16, "unit": "V"}
            },
            in_stock=True,
            lifecycle_status="active",
            pricing=[{"quantity": 1, "price": 0.05}]
        ),
        Component(
            part_number="C100N-0805",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.CAPACITOR,
            package_type=PackageType.SMD,
            package_name="0805",
            electrical_parameters={
                "capacitance": {"value": 100e-9, "unit": "F", "tolerance": 0.10},
                "voltage_rating": {"value": 50, "unit": "V"}
            },
            in_stock=True,
            lifecycle_status="active",
            pricing=[{"quantity": 1, "price": 0.08}]
        )
    ]
    db_session.add_all(capacitors)
    db_session.commit()
    return capacitors


class TestComponentSelector:
    """Tests for ComponentSelector class."""
    
    def test_select_resistor_by_value(self, db_session, sample_resistors):
        """Test selecting resistor by resistance value."""
        selector = ComponentSelector(db_session)
        
        # Select 1K resistor
        component = selector.select_resistor(resistance=1000, package_type=PackageType.SMD)
        
        assert component is not None
        assert component.part_number == "R1K-0805"
        assert component.electrical_parameters["resistance"]["value"] == 1000
    
    def test_select_resistor_10k(self, db_session, sample_resistors):
        """Test selecting 10K resistor."""
        selector = ComponentSelector(db_session)
        
        component = selector.select_resistor(resistance=10000, package_type=PackageType.SMD)
        
        assert component is not None
        assert component.part_number == "R10K-0805"
    
    def test_select_resistor_through_hole(self, db_session, sample_resistors):
        """Test selecting through-hole resistor."""
        selector = ComponentSelector(db_session)
        
        component = selector.select_resistor(
            resistance=10000,
            package_type=PackageType.THROUGH_HOLE
        )
        
        assert component is not None
        assert component.part_number == "R10K-TH"
        assert component.package_type == PackageType.THROUGH_HOLE
    
    def test_select_resistor_not_found(self, db_session, sample_resistors):
        """Test selecting non-existent resistor."""
        selector = ComponentSelector(db_session)
        
        component = selector.select_resistor(resistance=999999)
        
        assert component is None
    
    def test_select_capacitor_by_value(self, db_session, sample_capacitors):
        """Test selecting capacitor by capacitance."""
        selector = ComponentSelector(db_session)
        
        component = selector.select_capacitor(
            capacitance=100e-9,
            voltage_rating=16,
            package_type=PackageType.SMD
        )
        
        assert component is not None
        assert component.electrical_parameters["capacitance"]["value"] == 100e-9
    
    def test_select_capacitor_higher_voltage(self, db_session, sample_capacitors):
        """Test selecting capacitor with higher voltage rating."""
        selector = ComponentSelector(db_session)
        
        # Request 30V rating, should get 50V component
        component = selector.select_capacitor(
            capacitance=100e-9,
            voltage_rating=30
        )
        
        assert component is not None
        assert component.part_number == "C100N-0805"
        assert component.electrical_parameters["voltage_rating"]["value"] == 50
    
    def test_select_capacitor_not_found(self, db_session, sample_capacitors):
        """Test selecting non-existent capacitor."""
        selector = ComponentSelector(db_session)
        
        component = selector.select_capacitor(
            capacitance=1e-3,  # 1mF - not in database
            voltage_rating=16
        )
        
        assert component is None
    
    def test_find_alternatives(self, db_session, sample_resistors):
        """Test finding alternative components."""
        selector = ComponentSelector(db_session)
        
        # Get alternatives for R10K-0805
        reference = sample_resistors[1]
        alternatives = selector.find_alternatives(reference, max_alternatives=5)
        
        assert len(alternatives) >= 0
        # Should not include the reference component itself
        assert reference not in alternatives
    
    def test_find_alternatives_same_category(self, db_session, sample_resistors):
        """Test that alternatives are from same category."""
        selector = ComponentSelector(db_session)
        
        reference = sample_resistors[0]
        alternatives = selector.find_alternatives(reference)
        
        for alt in alternatives:
            assert alt.category == reference.category
    
    def test_check_availability(self, db_session, sample_resistors):
        """Test checking component availability."""
        selector = ComponentSelector(db_session)
        
        component = sample_resistors[0]
        availability = selector.check_availability(component)
        
        assert availability["part_number"] == component.part_number
        assert availability["in_stock"] is True
        assert availability["lifecycle_status"] == "active"
        assert "pricing" in availability
    
    def test_recommend_replacement_for_obsolete(self, db_session, sample_manufacturer):
        """Test recommending replacement for obsolete component."""
        # Create obsolete component
        obsolete = Component(
            part_number="R10K-OBSOLETE",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={
                "resistance": {"value": 10000, "unit": "ohm"},
                "power_rating": {"value": 0.125, "unit": "W"}
            },
            in_stock=False,
            lifecycle_status="obsolete",
            pricing=[{"quantity": 1, "price": 0.15}]
        )
        
        # Create active replacement
        replacement = Component(
            part_number="R10K-NEW",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={
                "resistance": {"value": 10000, "unit": "ohm"},
                "power_rating": {"value": 0.125, "unit": "W"}
            },
            in_stock=True,
            lifecycle_status="active",
            pricing=[{"quantity": 1, "price": 0.10}]
        )
        
        db_session.add_all([obsolete, replacement])
        db_session.commit()
        
        selector = ComponentSelector(db_session)
        recommended = selector.recommend_replacement(obsolete)
        
        assert recommended is not None
        assert recommended.lifecycle_status == "active"
        assert recommended.in_stock is True
    
    def test_select_by_category(self, db_session, sample_resistors):
        """Test selecting components by category."""
        selector = ComponentSelector(db_session)
        
        components = selector.select_by_category(
            category=ComponentCategory.RESISTOR,
            electrical_params={
                "resistance": {"value": 10000, "tolerance": 0.1}
            }
        )
        
        assert len(components) > 0
        for comp in components:
            assert comp.category == ComponentCategory.RESISTOR
    
    def test_select_by_category_with_package(self, db_session, sample_resistors):
        """Test selecting components by category and package type."""
        selector = ComponentSelector(db_session)
        
        components = selector.select_by_category(
            category=ComponentCategory.RESISTOR,
            electrical_params={
                "resistance": {"value": 10000, "tolerance": 0.1}
            },
            package_type=PackageType.SMD
        )
        
        assert len(components) > 0
        for comp in components:
            assert comp.package_type == PackageType.SMD
    
    def test_matches_parameters_exact(self, db_session, sample_resistors):
        """Test parameter matching with exact values."""
        selector = ComponentSelector(db_session)
        
        component_params = {
            "resistance": {"value": 1000, "unit": "ohm"},
            "power_rating": {"value": 0.125, "unit": "W"}
        }
        
        required_params = {
            "resistance": {"value": 1000, "tolerance": 0.01}
        }
        
        assert selector._matches_parameters(component_params, required_params)
    
    def test_matches_parameters_range(self, db_session, sample_resistors):
        """Test parameter matching with range."""
        selector = ComponentSelector(db_session)
        
        component_params = {
            "voltage_rating": {"value": 50, "unit": "V"}
        }
        
        required_params = {
            "voltage_rating": {"min": 30, "max": 100}
        }
        
        assert selector._matches_parameters(component_params, required_params)
    
    def test_matches_parameters_fails(self, db_session, sample_resistors):
        """Test parameter matching failure."""
        selector = ComponentSelector(db_session)
        
        component_params = {
            "resistance": {"value": 1000, "unit": "ohm"}
        }
        
        required_params = {
            "resistance": {"value": 10000, "tolerance": 0.01}
        }
        
        assert not selector._matches_parameters(component_params, required_params)
    
    def test_calculate_similarity_score(self, db_session):
        """Test similarity score calculation."""
        selector = ComponentSelector(db_session)
        
        params1 = {
            "resistance": {"value": 1000},
            "power_rating": {"value": 0.125}
        }
        
        params2 = {
            "resistance": {"value": 1000},
            "power_rating": {"value": 0.125}
        }
        
        score = selector._calculate_similarity_score(params1, params2)
        assert score == 1.0
    
    def test_calculate_similarity_score_partial(self, db_session):
        """Test similarity score with partial match."""
        selector = ComponentSelector(db_session)
        
        params1 = {
            "resistance": {"value": 1000},
            "power_rating": {"value": 0.125}
        }
        
        params2 = {
            "resistance": {"value": 1000},
            "power_rating": {"value": 0.25}
        }
        
        score = selector._calculate_similarity_score(params1, params2)
        assert 0.0 < score < 1.0
    
    def test_select_cheapest_component(self, db_session, sample_manufacturer):
        """Test that selector chooses cheapest component."""
        # Create two identical components with different prices
        expensive = Component(
            part_number="R1K-EXPENSIVE",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={
                "resistance": {"value": 1000, "unit": "ohm", "tolerance": 0.01},
                "power_rating": {"value": 0.125, "unit": "W"}
            },
            in_stock=True,
            lifecycle_status="active",
            pricing=[{"quantity": 1, "price": 0.50}]
        )
        
        cheap = Component(
            part_number="R1K-CHEAP",
            manufacturer_id=sample_manufacturer.id,
            category=ComponentCategory.RESISTOR,
            package_type=PackageType.SMD,
            electrical_parameters={
                "resistance": {"value": 1000, "unit": "ohm", "tolerance": 0.01},
                "power_rating": {"value": 0.125, "unit": "W"}
            },
            in_stock=True,
            lifecycle_status="active",
            pricing=[{"quantity": 1, "price": 0.05}]
        )
        
        db_session.add_all([expensive, cheap])
        db_session.commit()
        
        selector = ComponentSelector(db_session)
        component = selector.select_resistor(resistance=1000)
        
        assert component is not None
        assert component.part_number == "R1K-CHEAP"
