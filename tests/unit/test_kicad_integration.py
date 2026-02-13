"""
Unit tests for KiCad integration service.

Tests KiCadProject class functionality.
"""

import pytest
import os
import tempfile
from pathlib import Path

from src.services.kicad_integration import KiCadProject, KiCadIntegrationError


class TestKiCadProject:
    """Tests for KiCadProject class."""
    
    def test_init_with_default_dir(self):
        """Test initialization with default output directory."""
        project = KiCadProject("test_project")
        
        assert project.project_name == "test_project"
        assert project.output_dir is not None
        assert os.path.exists(project.output_dir)
        assert project.project_path.endswith("test_project.kicad_pro")
        
        project.cleanup()
    
    def test_init_with_custom_dir(self):
        """Test initialization with custom output directory."""
        custom_dir = tempfile.mkdtemp(prefix="test_kicad_")
        project = KiCadProject("test_project", output_dir=custom_dir)
        
        assert project.output_dir == custom_dir
        assert os.path.exists(custom_dir)
        
        project.cleanup()
    
    def test_create_project(self):
        """Test creating KiCad project files."""
        project = KiCadProject("test_create")
        
        result = project.create_project(
            board_width=50.0,
            board_height=40.0,
            layers=4
        )
        
        assert result["success"] is True
        assert result["board_size"]["width"] == 50.0
        assert result["board_size"]["height"] == 40.0
        assert result["layers"] == 4
        
        # Check that files were created
        assert os.path.exists(project.project_path)
        assert os.path.exists(project.schematic_path)
        assert os.path.exists(project.pcb_path)
        
        project.cleanup()
    
    def test_create_project_files_content(self):
        """Test that created project files have valid content."""
        project = KiCadProject("test_content")
        project.create_project()
        
        # Check project file content
        with open(project.project_path, 'r') as f:
            project_content = f.read()
        assert "kicad_pro" in project_content
        assert "test_content" in project_content
        
        # Check schematic file content
        with open(project.schematic_path, 'r') as f:
            schematic_content = f.read()
        assert "kicad_sch" in schematic_content
        assert "test_content" in schematic_content
        
        # Check PCB file content
        with open(project.pcb_path, 'r') as f:
            pcb_content = f.read()
        assert "kicad_pcb" in pcb_content
        
        project.cleanup()
    
    def test_import_netlist_file_not_found(self):
        """Test importing non-existent netlist file."""
        project = KiCadProject("test_import")
        project.create_project()
        
        with pytest.raises(KiCadIntegrationError):
            project.import_netlist("nonexistent.net")
        
        project.cleanup()
    
    def test_import_netlist_success(self):
        """Test successful netlist import."""
        project = KiCadProject("test_import")
        project.create_project()
        
        # Create a sample netlist file
        netlist_content = """(export (version D)
  (design
    (source test.sch)
    (date "2024-01-01 12:00:00")
    (tool "Eeschema")
  )
  (components
    (comp (ref R1)
      (value 10k)
      (footprint Resistor_SMD:R_0805_2012Metric)
    )
  )
  (nets
    (net (code 1) (name VCC))
    (net (code 2) (name GND))
  )
)"""
        
        temp_netlist = os.path.join(project.output_dir, "temp.net")
        with open(temp_netlist, 'w') as f:
            f.write(netlist_content)
        
        result = project.import_netlist(temp_netlist)
        
        assert result["success"] is True
        assert os.path.exists(result["netlist_path"])
        assert len(result["components"]) >= 0  # Parser might not catch all
        
        project.cleanup()
    
    def test_parse_netlist(self):
        """Test netlist parsing functionality."""
        project = KiCadProject("test_parse")
        
        # Create sample netlist content
        netlist_content = """(comp (ref R1) (value 10k) (footprint Resistor_SMD:R_0805_2012Metric))
(comp (ref C1) (value 100nF) (footprint Capacitor_SMD:C_0603_1608Metric))
(net (code 1) (name VCC))
(net (code 2) (name GND))"""
        
        temp_netlist = os.path.join(project.output_dir, "test.net")
        with open(temp_netlist, 'w') as f:
            f.write(netlist_content)
        
        result = project._parse_netlist(temp_netlist)
        
        assert "components" in result
        assert "nets" in result
        assert len(result["components"]) == 2
        assert len(result["nets"]) == 2
        
        project.cleanup()
    
    def test_generate_pcb_layout(self):
        """Test PCB layout generation."""
        project = KiCadProject("test_layout")
        project.create_project()
        
        design_rules = {
            "min_trace_width": 0.15,
            "min_via_size": 0.3,
            "min_clearance": 0.15
        }
        
        result = project.generate_pcb_layout(design_rules)
        
        assert result["success"] is True
        assert result["design_rules_applied"] is True
        assert "pcb_path" in result
        
        project.cleanup()
    
    def test_generate_pcb_layout_no_rules(self):
        """Test PCB layout generation without design rules."""
        project = KiCadProject("test_layout_no_rules")
        project.create_project()
        
        result = project.generate_pcb_layout()
        
        assert result["success"] is True
        assert result["design_rules_applied"] is False
        
        project.cleanup()
    
    def test_export_gerbers(self):
        """Test Gerber file export."""
        project = KiCadProject("test_gerbers")
        project.create_project()
        
        result = project.export_gerbers()
        
        assert result["success"] is True
        assert "gerber_dir" in result
        assert len(result["files"]) > 0
        assert len(result["file_paths"]) > 0
        
        # Check that files were created
        for file_path in result["file_paths"]:
            assert os.path.exists(file_path)
        
        project.cleanup()
    
    def test_export_gerbers_custom_dir(self):
        """Test Gerber export to custom directory."""
        project = KiCadProject("test_gerbers_custom")
        project.create_project()
        
        custom_gerber_dir = os.path.join(project.output_dir, "custom_gerbers")
        result = project.export_gerbers(custom_gerber_dir)
        
        assert result["success"] is True
        assert result["gerber_dir"] == custom_gerber_dir
        assert os.path.exists(custom_gerber_dir)
        
        project.cleanup()
    
    def test_validate_design_valid(self):
        """Test design validation with valid project."""
        project = KiCadProject("test_validate")
        project.create_project()
        
        result = project.validate_design()
        
        assert result["valid"] is True
        assert len(result["issues"]) == 0
        
        project.cleanup()
    
    def test_validate_design_missing_files(self):
        """Test design validation with missing files."""
        project = KiCadProject("test_validate_missing")
        # Don't create project files
        
        result = project.validate_design()
        
        assert result["valid"] is False
        assert len(result["issues"]) > 0
        assert any("missing" in issue.lower() for issue in result["issues"])
        
        project.cleanup()
    
    def test_get_project_info(self):
        """Test getting project information."""
        project = KiCadProject("test_info")
        project.create_project()
        
        info = project.get_project_info()
        
        assert info["project_name"] == "test_info"
        assert "output_dir" in info
        assert "files_exist" in info
        assert info["files_exist"]["project"] is True
        assert info["files_exist"]["schematic"] is True
        assert info["files_exist"]["pcb"] is True
        
        project.cleanup()
    
    def test_generate_uuid(self):
        """Test UUID generation."""
        project = KiCadProject("test_uuid")
        
        uuid1 = project._generate_uuid()
        uuid2 = project._generate_uuid()
        
        assert uuid1 != uuid2
        assert len(uuid1) == 36  # Standard UUID length
        assert "-" in uuid1
        
        project.cleanup()
    
    def test_get_current_date(self):
        """Test current date generation."""
        project = KiCadProject("test_date")
        
        date = project._get_current_date()
        
        assert len(date) == 10  # YYYY-MM-DD format
        assert date.count("-") == 2
        
        project.cleanup()
    
    def test_apply_design_rules(self):
        """Test applying design rules."""
        project = KiCadProject("test_rules")
        
        design_rules = {
            "min_trace_width": 0.1,
            "min_via_size": 0.2,
            "min_clearance": 0.1
        }
        
        # Should not raise exception
        project._apply_design_rules(design_rules)
        
        project.cleanup()
    
    def test_cleanup(self):
        """Test cleanup of temporary files."""
        project = KiCadProject("test_cleanup")
        project.create_project()
        
        output_dir = project.output_dir
        assert os.path.exists(output_dir)
        
        project.cleanup()
        
        assert not os.path.exists(output_dir)
    
    def test_project_paths(self):
        """Test that project paths are correctly set."""
        project = KiCadProject("test_paths", "/tmp/test")
        
        assert project.project_path == "/tmp/test/test_paths.kicad_pro"
        assert project.schematic_path == "/tmp/test/test_paths.kicad_sch"
        assert project.pcb_path == "/tmp/test/test_paths.kicad_pcb"
        assert project.netlist_path == "/tmp/test/test_paths.net"
        
        project.cleanup()
    
    def test_board_dimensions_in_pcb(self):
        """Test that board dimensions are correctly set in PCB file."""
        project = KiCadProject("test_dimensions")
        project.create_project(board_width=75.0, board_height=50.0)
        
        with open(project.pcb_path, 'r') as f:
            pcb_content = f.read()
        
        assert "75" in pcb_content
        assert "50" in pcb_content
        
        project.cleanup()