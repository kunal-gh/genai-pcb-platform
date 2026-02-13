"""
Unit tests for manufacturing file export system.

Tests ManufacturingExporter class functionality.
"""

import pytest
import os
import tempfile
import csv
from pathlib import Path

from src.services.manufacturing_export import (
    ManufacturingExporter,
    ManufacturingExportError,
    ComponentPlacement,
    DrillHole
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_manufacturing_")
    yield temp_dir
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_pcb_data():
    """Sample PCB data for testing."""
    return {
        "width": 100.0,
        "height": 80.0,
        "layers": 2,
        "components": [
            {"reference": "R1", "value": "10k", "package": "0805"},
            {"reference": "C1", "value": "100nF", "package": "0603"},
            {"reference": "U1", "value": "LM358", "package": "SOIC-8"}
        ]
    }


@pytest.fixture
def sample_components():
    """Sample component placements."""
    return [
        ComponentPlacement("R1", "10k", "0805", 10.0, 20.0, 0.0, "top"),
        ComponentPlacement("R2", "1k", "0805", 30.0, 20.0, 90.0, "top"),
        ComponentPlacement("C1", "100nF", "0603", 50.0, 20.0, 0.0, "top"),
        ComponentPlacement("C2", "10uF", "1206", 10.0, 60.0, 180.0, "bottom"),
        ComponentPlacement("U1", "LM358", "SOIC-8", 70.0, 40.0, 0.0, "top")
    ]


@pytest.fixture
def sample_drill_holes():
    """Sample drill holes."""
    return [
        DrillHole(10.0, 20.0, 0.8, True),   # Via
        DrillHole(30.0, 20.0, 0.8, True),   # Via
        DrillHole(50.0, 20.0, 1.0, True),   # Component hole
        DrillHole(70.0, 40.0, 0.6, True),   # Small via
        DrillHole(90.0, 70.0, 3.2, True)    # Mounting hole
    ]


class TestManufacturingExporter:
    """Tests for ManufacturingExporter class."""
    
    def test_init(self, temp_dir):
        """Test initialization."""
        exporter = ManufacturingExporter("test_project", temp_dir)
        
        assert exporter.project_name == "test_project"
        assert exporter.output_dir == temp_dir
        assert os.path.exists(temp_dir)
        assert len(exporter.gerber_extensions) > 0
    
    def test_export_gerber_files(self, temp_dir, sample_pcb_data):
        """Test Gerber file export."""
        exporter = ManufacturingExporter("test_gerber", temp_dir)
        
        result = exporter.export_gerber_files(sample_pcb_data)
        
        assert result["success"] is True
        assert "files" in result
        assert result["layer_count"] > 0
        
        # Check that files were created
        for layer, filepath in result["files"].items():
            if layer != "apertures":  # Skip aperture file
                assert os.path.exists(filepath)
                
                # Check file content
                with open(filepath, 'r') as f:
                    content = f.read()
                assert "G04" in content  # Gerber comment
                assert "M02" in content  # End of file
    
    def test_export_gerber_files_custom_layers(self, temp_dir, sample_pcb_data):
        """Test Gerber export with custom layer list."""
        exporter = ManufacturingExporter("test_custom", temp_dir)
        
        custom_layers = ["top_copper", "bottom_copper", "outline"]
        result = exporter.export_gerber_files(sample_pcb_data, custom_layers)
        
        assert result["success"] is True
        assert result["layer_count"] == 3
        assert len(result["files"]) >= 3  # May include aperture file
    
    def test_generate_gerber_content(self, temp_dir, sample_pcb_data):
        """Test Gerber content generation."""
        exporter = ManufacturingExporter("test_content", temp_dir)
        
        content = exporter._generate_gerber_content("top_copper", sample_pcb_data)
        
        assert "G04" in content  # Comments
        assert "TF.FileFunction" in content
        assert "M02" in content  # End of file
        assert "test_content" in content  # Project name
    
    def test_get_file_function(self, temp_dir):
        """Test file function attribute generation."""
        exporter = ManufacturingExporter("test_function", temp_dir)
        
        assert exporter._get_file_function("top_copper") == "Copper,L1,Top"
        assert exporter._get_file_function("bottom_copper") == "Copper,L2,Bot"
        assert exporter._get_file_function("outline") == "Profile,NP"
        assert exporter._get_file_function("unknown") == "Other"
    
    def test_generate_outline_content(self, temp_dir, sample_pcb_data):
        """Test board outline generation."""
        exporter = ManufacturingExporter("test_outline", temp_dir)
        
        content = exporter._generate_outline_content(sample_pcb_data)
        
        assert len(content) > 0
        assert "G01" in content  # Linear interpolation
        assert "D02" in content  # Move
        assert "D01" in content  # Draw
        assert "100000" in "\n".join(content)  # Width in micrometers
    
    def test_export_drill_files(self, temp_dir, sample_drill_holes):
        """Test drill file export."""
        exporter = ManufacturingExporter("test_drill", temp_dir)
        
        result = exporter.export_drill_files(sample_drill_holes)
        
        assert result["success"] is True
        assert result["hole_count"] == 5
        assert result["tool_count"] > 0
        assert os.path.exists(result["drill_file"])
        assert os.path.exists(result["report_file"])
        
        # Check drill file content
        with open(result["drill_file"], 'r') as f:
            content = f.read()
        assert "M48" in content  # Excellon header
        assert "M30" in content  # End of program
        assert "T01" in content  # Tool definition
    
    def test_generate_drill_report(self, temp_dir, sample_drill_holes):
        """Test drill report generation."""
        exporter = ManufacturingExporter("test_report", temp_dir)
        
        # Group holes by diameter
        holes_by_diameter = {}
        for hole in sample_drill_holes:
            diameter = hole.diameter
            if diameter not in holes_by_diameter:
                holes_by_diameter[diameter] = []
            holes_by_diameter[diameter].append(hole)
        
        report_file = exporter._generate_drill_report(sample_drill_holes, holes_by_diameter)
        
        assert os.path.exists(report_file)
        
        with open(report_file, 'r') as f:
            content = f.read()
        assert "Drill Report" in content
        assert "Total holes: 5" in content
        assert "T01:" in content
    
    def test_export_pick_and_place(self, temp_dir, sample_components):
        """Test pick-and-place file export."""
        exporter = ManufacturingExporter("test_pnp", temp_dir)
        
        result = exporter.export_pick_and_place(sample_components)
        
        assert result["success"] is True
        assert result["component_count"] == 5
        assert result["top_count"] == 4
        assert result["bottom_count"] == 1
        assert "files" in result
        
        # Check that files were created
        if "top" in result["files"]:
            assert os.path.exists(result["files"]["top"])
        if "bottom" in result["files"]:
            assert os.path.exists(result["files"]["bottom"])
        assert os.path.exists(result["files"]["report"])
    
    def test_export_pnp_file(self, temp_dir, sample_components):
        """Test individual pick-and-place file export."""
        exporter = ManufacturingExporter("test_pnp_file", temp_dir)
        
        top_components = [c for c in sample_components if c.layer == "top"]
        filepath = exporter._export_pnp_file(top_components, "top")
        
        assert os.path.exists(filepath)
        
        # Check CSV content
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            
            assert len(rows) > 1  # Header + data
            assert rows[0][0] == "Designator"  # Header check
            assert rows[1][0] == "R1"  # First component
    
    def test_generate_assembly_report(self, temp_dir, sample_components):
        """Test assembly report generation."""
        exporter = ManufacturingExporter("test_assembly", temp_dir)
        
        report_file = exporter._generate_assembly_report(sample_components)
        
        assert os.path.exists(report_file)
        
        with open(report_file, 'r') as f:
            content = f.read()
        assert "Assembly Report" in content
        assert "Total components: 5" in content
        assert "0805:" in content  # Package count
    
    def test_export_step_model(self, temp_dir, sample_pcb_data, sample_components):
        """Test STEP model export."""
        exporter = ManufacturingExporter("test_step", temp_dir)
        
        result = exporter.export_step_model(sample_pcb_data, sample_components)
        
        assert result["success"] is True
        assert result["component_count"] == 5
        assert os.path.exists(result["step_file"])
        
        # Check STEP file content
        with open(result["step_file"], 'r') as f:
            content = f.read()
        assert "ISO-10303-21" in content
        assert "STEP" in content.upper() or "AUTOMOTIVE_DESIGN" in content
    
    def test_generate_step_content(self, temp_dir, sample_pcb_data, sample_components):
        """Test STEP content generation."""
        exporter = ManufacturingExporter("test_step_content", temp_dir)
        
        content = exporter._generate_step_content(sample_pcb_data, sample_components)
        
        assert "ISO-10303-21" in content
        assert "HEADER" in content
        assert "DATA" in content
        assert "END-ISO-10303-21" in content
    
    def test_generate_manufacturing_package(self, temp_dir, sample_pcb_data, sample_components, sample_drill_holes):
        """Test complete manufacturing package generation."""
        exporter = ManufacturingExporter("test_package", temp_dir)
        
        result = exporter.generate_manufacturing_package(
            sample_pcb_data, sample_components, sample_drill_holes
        )
        
        assert result["success"] is True
        assert result["output_dir"] == temp_dir
        assert "results" in result
        
        results = result["results"]
        assert "gerbers" in results
        assert "drill" in results
        assert "pick_and_place" in results
        assert "step" in results
        assert "summary" in results
        
        # Check summary file
        assert os.path.exists(results["summary"])
    
    def test_generate_package_summary(self, temp_dir):
        """Test package summary generation."""
        exporter = ManufacturingExporter("test_summary", temp_dir)
        
        mock_results = {
            "gerbers": {"files": {"top_copper": "test.gtl", "bottom_copper": "test.gbl"}},
            "drill": {"hole_count": 10, "tool_count": 3},
            "pick_and_place": {"component_count": 15, "top_count": 12, "bottom_count": 3},
            "step": {"step_file": "test.step"}
        }
        
        summary_file = exporter._generate_package_summary(mock_results)
        
        assert os.path.exists(summary_file)
        
        with open(summary_file, 'r') as f:
            content = f.read()
        assert "Manufacturing Package Summary" in content
        assert "Gerber Files (2)" in content
        assert "Holes: 10" in content
        assert "Components: 15" in content
    
    def test_get_timestamp(self, temp_dir):
        """Test timestamp generation."""
        exporter = ManufacturingExporter("test_timestamp", temp_dir)
        
        timestamp = exporter._get_timestamp()
        
        assert len(timestamp) > 10  # Should be a reasonable timestamp
        assert "-" in timestamp  # Date format
        assert ":" in timestamp  # Time format
    
    def test_component_placement_dataclass(self):
        """Test ComponentPlacement dataclass."""
        comp = ComponentPlacement("R1", "10k", "0805", 10.0, 20.0, 90.0, "top")
        
        assert comp.reference == "R1"
        assert comp.value == "10k"
        assert comp.package == "0805"
        assert comp.x == 10.0
        assert comp.y == 20.0
        assert comp.rotation == 90.0
        assert comp.layer == "top"
    
    def test_drill_hole_dataclass(self):
        """Test DrillHole dataclass."""
        hole = DrillHole(5.0, 10.0, 0.8, False)
        
        assert hole.x == 5.0
        assert hole.y == 10.0
        assert hole.diameter == 0.8
        assert hole.plated is False
        
        # Test default value
        hole_default = DrillHole(0.0, 0.0, 1.0)
        assert hole_default.plated is True
    
    def test_gerber_extensions(self, temp_dir):
        """Test Gerber file extensions mapping."""
        exporter = ManufacturingExporter("test_ext", temp_dir)
        
        assert exporter.gerber_extensions["top_copper"] == "GTL"
        assert exporter.gerber_extensions["bottom_copper"] == "GBL"
        assert exporter.gerber_extensions["outline"] == "GKO"
        assert exporter.gerber_extensions["drill"] == "TXT"
    
    def test_error_handling(self, temp_dir):
        """Test error handling in export functions."""
        exporter = ManufacturingExporter("test_error", temp_dir)
        
        # Test with invalid data that might cause errors
        with pytest.raises(ManufacturingExportError):
            # This should fail due to invalid drill data
            exporter.export_drill_files("invalid_data")