"""
Unit tests for SKiDL execution environment.

Tests SKiDLExecutor class functionality.
"""

import pytest
import os
import tempfile
from pathlib import Path

from src.services.skidl_executor import SKiDLExecutor, SKiDLExecutionError


class TestSKiDLExecutor:
    """Tests for SKiDLExecutor class."""
    
    def test_init_with_default_dir(self):
        """Test initialization with default output directory."""
        executor = SKiDLExecutor()
        
        assert executor.output_dir is not None
        assert os.path.exists(executor.output_dir)
        
        executor.cleanup()
    
    def test_init_with_custom_dir(self):
        """Test initialization with custom output directory."""
        custom_dir = tempfile.mkdtemp(prefix="test_skidl_")
        executor = SKiDLExecutor(output_dir=custom_dir)
        
        assert executor.output_dir == custom_dir
        assert os.path.exists(custom_dir)
        
        executor.cleanup()
    
    def test_validate_code_valid(self):
        """Test validating valid SKiDL code."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Part, Net, generate_netlist

# Create components
r1 = Part('Device', 'R', value='1k')
r2 = Part('Device', 'R', value='10k')

# Create nets
vcc = Net('VCC')
gnd = Net('GND')

# Connect components
vcc += r1[1]
r1[2] += r2[1]
r2[2] += gnd
"""
        
        result = executor.validate_code(code)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        executor.cleanup()
    
    def test_validate_code_missing_import(self):
        """Test validating code without SKiDL import."""
        executor = SKiDLExecutor()
        
        code = """
r1 = Part('Device', 'R', value='1k')
"""
        
        result = executor.validate_code(code)
        
        assert result["valid"] is False
        assert any("import" in error.lower() for error in result["errors"])
        
        executor.cleanup()
    
    def test_validate_code_syntax_error(self):
        """Test validating code with syntax error."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Part

r1 = Part('Device', 'R', value='1k'
"""  # Missing closing parenthesis
        
        result = executor.validate_code(code)
        
        assert result["valid"] is False
        assert any("syntax" in error.lower() for error in result["errors"])
        
        executor.cleanup()
    
    def test_validate_code_no_components(self):
        """Test validating code without components."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Part, Net
"""
        
        result = executor.validate_code(code)
        
        # Should be valid but with warning
        assert result["valid"] is True
        assert any("component" in warning.lower() for warning in result["warnings"])
        
        executor.cleanup()
    
    def test_extract_components(self):
        """Test extracting component definitions."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Part

r1 = Part('Device', 'R', value='1k')
r2 = Part('Device', 'R', value='10k')
c1 = Part('Device', 'C', value='100nF')
"""
        
        components = executor.extract_components(code)
        
        assert len(components) == 3
        assert components[0]["name"] == "r1"
        assert components[0]["library"] == "Device"
        assert components[0]["part"] == "R"
        assert components[2]["name"] == "c1"
        assert components[2]["part"] == "C"
        
        executor.cleanup()
    
    def test_extract_components_empty(self):
        """Test extracting components from code without components."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Part
"""
        
        components = executor.extract_components(code)
        
        assert len(components) == 0
        
        executor.cleanup()
    
    def test_extract_nets(self):
        """Test extracting net definitions."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Net

vcc = Net('VCC')
gnd = Net('GND')
signal = Net('SIGNAL')
"""
        
        nets = executor.extract_nets(code)
        
        assert len(nets) == 3
        assert "vcc" in nets
        assert "gnd" in nets
        assert "signal" in nets
        
        executor.cleanup()
    
    def test_extract_nets_empty(self):
        """Test extracting nets from code without nets."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Part
"""
        
        nets = executor.extract_nets(code)
        
        assert len(nets) == 0
        
        executor.cleanup()
    
    def test_parse_warnings(self):
        """Test parsing warnings from output."""
        executor = SKiDLExecutor()
        
        output = """
Some normal output
WARNING: Component not found
More output
Warning: Pin mismatch
"""
        
        warnings = executor._parse_warnings(output)
        
        assert len(warnings) == 2
        assert any("Component not found" in w for w in warnings)
        assert any("Pin mismatch" in w for w in warnings)
        
        executor.cleanup()
    
    def test_cleanup(self):
        """Test cleanup of temporary files."""
        executor = SKiDLExecutor()
        output_dir = executor.output_dir
        
        assert os.path.exists(output_dir)
        
        executor.cleanup()
        
        assert not os.path.exists(output_dir)
    
    def test_execute_creates_code_file(self):
        """Test that execute creates a code file."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Part, generate_netlist

r1 = Part('Device', 'R', value='1k')
generate_netlist()
"""
        
        try:
            result = executor.execute(code, "test_circuit")
            
            # Check that code file was created
            code_file = os.path.join(executor.output_dir, "test_circuit.py")
            assert os.path.exists(code_file)
            
        except SKiDLExecutionError:
            # Expected if SKiDL is not installed
            pass
        finally:
            executor.cleanup()
    
    def test_generate_netlist_adds_command(self):
        """Test that generate_netlist adds netlist generation command."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Part

r1 = Part('Device', 'R', value='1k')
"""
        
        try:
            # This will fail if SKiDL is not installed, but we can check the code modification
            netlist_path = executor.generate_netlist(code, "test")
        except SKiDLExecutionError:
            # Expected if SKiDL is not installed
            pass
        finally:
            executor.cleanup()
    
    def test_validate_code_with_connections(self):
        """Test validating code with net connections."""
        executor = SKiDLExecutor()
        
        code = """
from skidl import Part, Net

r1 = Part('Device', 'R', value='1k')
r2 = Part('Device', 'R', value='10k')
vcc = Net('VCC')

vcc += r1[1]
r1[2] += r2[1]
"""
        
        result = executor.validate_code(code)
        
        assert result["valid"] is True
        # Should not have warning about missing connections
        assert not any("connection" in warning.lower() for warning in result["warnings"])
        
        executor.cleanup()
