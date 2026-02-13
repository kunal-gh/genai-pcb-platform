"""
Unit tests for BOM generator.

Tests BOM generation, component extraction, pricing, and export functionality.
"""

import pytest
import os
import json
import csv
from datetime import datetime
from sqlalchemy.orm import Session

from src.services.bom_generator import (
    BOMGenerator,
    BOMItem,
    BOMSummary,
    BOMGenerationError
)
from src.models.component import Component, Manufacturer, ComponentCategory, PackageType


@pytest.fixture
def sample_netlist():
    """Sample netlist for testing."""
    return {
        "components": [
            {
                "reference": "R1",
                "part_number": "RC0805FR-0710KL",
                "value": "10K",
                "package": "0805",
                "manufacturer": "Yageo",
                "description": "10K ohm resistor"
            },
            {
                "reference": "R2",
                "part_number": "RC0805FR-0710KL",
                "value": "10K",
                "package": "0805",
                "manufacturer": "Yageo",
                "description": "10K ohm resistor"
            },
            {
                "reference": "C1",
                "part_number": "CL21B104KBCNNNC",
                "value": "100nF",
                "package": "0805",
                "manufacturer": "Samsung",
                "description": "100nF ceramic capacitor"
            },
            {
                "reference": "U1",
                "part_number": "ATMEGA328P-AU",
                "value": "",
                "package": "TQFP-32",
                "manufacturer": "Microchip",
                "description": "8-bit microcontroller"
            }
        ]
    }


@pytest.fixture
def sample_components(db_session):
    """Create sample components in database."""
    # Create manufacturers
    yageo = Manufacturer(name="Yageo", website="https://www.yageo.com")
    samsung = Manufacturer(name="Samsung", website="https://www.samsung.com")
    microchip = Manufacturer(name="Microchip", website="https://www.microchip.com")
    
    db_session.add_all([yageo, samsung, microchip])
    db_session.flush()
    
    # Create components
    resistor = Component(
        part_number="RC0805FR-0710KL",
        manufacturer_id=yageo.id,
        category=ComponentCategory.RESISTOR,
        package_type=PackageType.SMD,
        description="10K ohm resistor, 1%, 1/8W",
        electrical_parameters={"resistance": "10000", "tolerance": "1%", "power": "0.125W"},
        pricing=[{"quantity": 1, "price": 0.10}, {"quantity": 100, "price": 0.05}],
        suppliers=[{"name": "DigiKey", "sku": "311-10.0KCRCT-ND"}],
        in_stock=True,
        lifecycle_status="active",
        footprint_id="Resistor_SMD:R_0805_2012Metric",
        symbol_id="Device:R"
    )
    
    capacitor = Component(
        part_number="CL21B104KBCNNNC",
        manufacturer_id=samsung.id,
        category=ComponentCategory.CAPACITOR,
        package_type=PackageType.SMD,
        description="100nF ceramic capacitor, X7R, 50V",
        electrical_parameters={"capacitance": "100e-9", "voltage": "50V", "tolerance": "10%"},
        pricing=[{"quantity": 1, "price": 0.15}, {"quantity": 100, "price": 0.08}],
        suppliers=[{"name": "DigiKey", "sku": "1276-1003-1-ND"}],
        in_stock=True,
        lifecycle_status="active",
        footprint_id="Capacitor_SMD:C_0805_2012Metric",
        symbol_id="Device:C"
    )
    
    mcu = Component(
        part_number="ATMEGA328P-AU",
        manufacturer_id=microchip.id,
        category=ComponentCategory.IC,
        package_type=PackageType.QFP,
        description="8-bit AVR microcontroller, 32KB Flash",
        electrical_parameters={"flash": "32KB", "ram": "2KB", "speed": "20MHz"},
        pricing=[{"quantity": 1, "price": 2.50}, {"quantity": 100, "price": 1.80}],
        suppliers=[{"name": "DigiKey", "sku": "ATMEGA328P-AU-ND"}],
        in_stock=True,
        lifecycle_status="active",
        footprint_id="Package_QFP:TQFP-32_7x7mm_P0.8mm",
        symbol_id="MCU_Microchip_ATmega:ATmega328P-A"
    )
    
    db_session.add_all([resistor, capacitor, mcu])
    db_session.commit()
    
    return [resistor, capacitor, mcu]


def test_bom_generator_initialization(db_session):
    """Test BOM generator initialization."""
    generator = BOMGenerator(db_session)
    assert generator.db == db_session


def test_generate_bom_basic(db_session, sample_netlist, sample_components):
    """Test basic BOM generation."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    assert "items" in bom_data
    assert "summary" in bom_data
    assert "warnings" in bom_data
    assert len(bom_data["items"]) == 3  # 3 unique parts


def test_bom_item_grouping(db_session, sample_netlist, sample_components):
    """Test that components are grouped correctly."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    # Find resistor item (R1 and R2 should be grouped)
    resistor_item = next(
        (item for item in bom_data["items"] if item.part_number == "RC0805FR-0710KL"),
        None
    )
    
    assert resistor_item is not None
    assert resistor_item.quantity == 2
    assert "R1" in resistor_item.reference_designators
    assert "R2" in resistor_item.reference_designators


def test_bom_item_details(db_session, sample_netlist, sample_components):
    """Test BOM item details are populated correctly."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    resistor_item = next(
        (item for item in bom_data["items"] if item.part_number == "RC0805FR-0710KL"),
        None
    )
    
    assert resistor_item.manufacturer == "Yageo"
    assert resistor_item.value == "10K"
    assert resistor_item.package == "0805"
    assert resistor_item.description == "10K ohm resistor, 1%, 1/8W"


def test_bom_pricing_included(db_session, sample_netlist, sample_components):
    """Test that pricing information is included."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist, include_pricing=True)
    
    resistor_item = next(
        (item for item in bom_data["items"] if item.part_number == "RC0805FR-0710KL"),
        None
    )
    
    assert resistor_item.unit_price is not None
    assert resistor_item.extended_price is not None
    assert resistor_item.supplier == "DigiKey"
    assert resistor_item.availability == "In Stock"


def test_bom_pricing_excluded(db_session, sample_netlist, sample_components):
    """Test BOM generation without pricing."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist, include_pricing=False)
    
    resistor_item = next(
        (item for item in bom_data["items"] if item.part_number == "RC0805FR-0710KL"),
        None
    )
    
    assert resistor_item.unit_price is None
    assert resistor_item.extended_price is None


def test_bom_summary_generation(db_session, sample_netlist, sample_components):
    """Test BOM summary statistics."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    summary = bom_data["summary"]
    assert isinstance(summary, BOMSummary)
    assert summary.total_unique_parts == 3
    assert summary.total_components == 4  # R1, R2, C1, U1
    assert summary.estimated_total_cost > 0
    assert isinstance(summary.generated_at, datetime)


def test_obsolete_component_detection(db_session, sample_netlist):
    """Test detection of obsolete components."""
    # Create obsolete component
    manufacturer = Manufacturer(name="Test Mfg")
    db_session.add(manufacturer)
    db_session.flush()
    
    obsolete_comp = Component(
        part_number="RC0805FR-0710KL",
        manufacturer_id=manufacturer.id,
        category=ComponentCategory.RESISTOR,
        package_type=PackageType.SMD,
        description="Obsolete resistor",
        electrical_parameters={},
        lifecycle_status="obsolete",
        footprint_id="Resistor_SMD:R_0805_2012Metric",
        symbol_id="Device:R"
    )
    db_session.add(obsolete_comp)
    db_session.commit()
    
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    resistor_item = next(
        (item for item in bom_data["items"] if item.part_number == "RC0805FR-0710KL"),
        None
    )
    
    assert resistor_item.is_obsolete is True
    assert any("obsolete" in warning.lower() for warning in bom_data["warnings"])


def test_hard_to_source_detection(db_session, sample_netlist):
    """Test detection of hard-to-source components."""
    # Create component with out of stock
    manufacturer = Manufacturer(name="Test Mfg")
    db_session.add(manufacturer)
    db_session.flush()
    
    out_of_stock_comp = Component(
        part_number="RC0805FR-0710KL",
        manufacturer_id=manufacturer.id,
        category=ComponentCategory.RESISTOR,
        package_type=PackageType.SMD,
        description="Out of stock resistor",
        electrical_parameters={},
        in_stock=False,  # Out of stock
        lifecycle_status="active",
        footprint_id="Resistor_SMD:R_0805_2012Metric",
        symbol_id="Device:R"
    )
    db_session.add(out_of_stock_comp)
    db_session.commit()
    
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    resistor_item = next(
        (item for item in bom_data["items"] if item.part_number == "RC0805FR-0710KL"),
        None
    )
    
    assert resistor_item.is_hard_to_source is True


def test_alternative_parts_suggestion(db_session, sample_netlist):
    """Test alternative parts suggestion for obsolete components."""
    # Create obsolete component and alternatives
    manufacturer = Manufacturer(name="Test Mfg")
    db_session.add(manufacturer)
    db_session.flush()
    
    obsolete_comp = Component(
        part_number="RC0805FR-0710KL",
        manufacturer_id=manufacturer.id,
        category=ComponentCategory.RESISTOR,
        package_type=PackageType.SMD,
        description="Obsolete resistor",
        electrical_parameters={},
        lifecycle_status="obsolete",
        footprint_id="Resistor_SMD:R_0805_2012Metric",
        symbol_id="Device:R"
    )
    
    alternative1 = Component(
        part_number="RC0805FR-0710KL-ALT1",
        manufacturer_id=manufacturer.id,
        category=ComponentCategory.RESISTOR,
        package_type=PackageType.SMD,
        description="Alternative resistor 1",
        electrical_parameters={},
        in_stock=True,
        lifecycle_status="active",
        footprint_id="Resistor_SMD:R_0805_2012Metric",
        symbol_id="Device:R"
    )
    
    alternative2 = Component(
        part_number="RC0805FR-0710KL-ALT2",
        manufacturer_id=manufacturer.id,
        category=ComponentCategory.RESISTOR,
        package_type=PackageType.SMD,
        description="Alternative resistor 2",
        electrical_parameters={},
        in_stock=True,
        lifecycle_status="active",
        footprint_id="Resistor_SMD:R_0805_2012Metric",
        symbol_id="Device:R"
    )
    
    db_session.add_all([obsolete_comp, alternative1, alternative2])
    db_session.commit()
    
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist, include_alternatives=True)
    
    resistor_item = next(
        (item for item in bom_data["items"] if item.part_number == "RC0805FR-0710KL"),
        None
    )
    
    assert len(resistor_item.alternative_parts) > 0


def test_export_bom_csv(db_session, sample_netlist, sample_components, tmp_path):
    """Test BOM export to CSV format."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    output_path = tmp_path / "bom.csv"
    generator.export_bom_csv(bom_data, str(output_path))
    
    assert output_path.exists()
    
    # Read and verify CSV
    with open(output_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
        assert len(rows) > 1  # Header + data rows
        assert rows[0][0] == "Reference Designators"
        assert len(rows) == 4  # Header + 3 items


def test_export_bom_json(db_session, sample_netlist, sample_components, tmp_path):
    """Test BOM export to JSON format."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    output_path = tmp_path / "bom.json"
    generator.export_bom_json(bom_data, str(output_path))
    
    assert output_path.exists()
    
    # Read and verify JSON
    with open(output_path, 'r') as f:
        data = json.load(f)
        
        assert "items" in data
        assert "summary" in data
        assert "warnings" in data
        assert len(data["items"]) == 3


def test_empty_netlist(db_session):
    """Test BOM generation with empty netlist."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom({"components": []})
    
    assert len(bom_data["items"]) == 0
    assert bom_data["summary"].total_unique_parts == 0
    assert bom_data["summary"].total_components == 0


def test_component_not_in_database(db_session):
    """Test BOM generation for components not in database."""
    netlist = {
        "components": [
            {
                "reference": "R1",
                "part_number": "UNKNOWN-PART",
                "value": "10K",
                "package": "0805",
                "manufacturer": "Unknown",
                "description": "Unknown resistor"
            }
        ]
    }
    
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(netlist)
    
    assert len(bom_data["items"]) == 1
    item = bom_data["items"][0]
    assert item.part_number == "UNKNOWN-PART"
    assert item.manufacturer == "Unknown"
    assert item.unit_price is None  # No pricing available


def test_extended_price_calculation(db_session, sample_netlist, sample_components):
    """Test extended price calculation."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    resistor_item = next(
        (item for item in bom_data["items"] if item.part_number == "RC0805FR-0710KL"),
        None
    )
    
    assert resistor_item.extended_price == resistor_item.unit_price * resistor_item.quantity


def test_quantity_break_pricing(db_session, sample_netlist, sample_components):
    """Test that quantity break pricing is applied."""
    generator = BOMGenerator(db_session)
    bom_data = generator.generate_bom(sample_netlist)
    
    # Resistor has quantity 2, should get lower price tier
    resistor_item = next(
        (item for item in bom_data["items"] if item.part_number == "RC0805FR-0710KL"),
        None
    )
    
    # Price should be calculated based on quantity
    assert resistor_item.unit_price is not None


def test_bom_item_dataclass():
    """Test BOMItem dataclass initialization."""
    item = BOMItem(
        reference_designators=["R1", "R2"],
        quantity=2,
        part_number="TEST-123",
        manufacturer="Test Mfg",
        description="Test component",
        unit_price=1.50
    )
    
    assert item.quantity == 2
    assert item.extended_price == 3.00  # Auto-calculated
    assert item.alternative_parts == []  # Default empty list


def test_bom_summary_dataclass():
    """Test BOMSummary dataclass."""
    summary = BOMSummary(
        total_unique_parts=10,
        total_components=25,
        estimated_total_cost=50.00,
        obsolete_parts_count=1,
        hard_to_source_count=2,
        generated_at=datetime.utcnow()
    )
    
    assert summary.total_unique_parts == 10
    assert summary.total_components == 25
    assert summary.estimated_total_cost == 50.00


def test_bom_generation_error_handling(db_session):
    """Test error handling in BOM generation."""
    generator = BOMGenerator(db_session)
    
    # Test with invalid netlist
    with pytest.raises(BOMGenerationError):
        generator.generate_bom(None)
