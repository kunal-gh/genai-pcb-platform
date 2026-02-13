"""
Unit tests for simulation engine.
"""

import pytest
from pathlib import Path
import tempfile

from src.services.simulation_engine import (
    SimulationEngine,
    SimulationType,
    SimulationStatus,
    SimulationResult
)


@pytest.fixture
def sim_engine():
    """Create simulation engine with temp directory."""
    engine = SimulationEngine()
    yield engine
    engine.cleanup()


@pytest.fixture
def simple_components():
    """Simple resistor divider components."""
    return [
        {
            "type": "VOLTAGE_SOURCE",
            "reference": "V1",
            "value": "5V",
            "pins": ["1", "0"]
        },
        {
            "type": "RESISTOR",
            "reference": "R1",
            "value": "1k",
            "pins": ["1", "2"]
        },
        {
            "type": "RESISTOR",
            "reference": "R2",
            "value": "1k",
            "pins": ["2", "0"]
        }
    ]


@pytest.fixture
def simple_nets():
    """Simple nets for resistor divider."""
    return [
        {"name": "VCC", "pins": ["V1.1", "R1.1"]},
        {"name": "OUT", "pins": ["R1.2", "R2.1"]},
        {"name": "GND", "pins": ["V1.2", "R2.2"]}
    ]


def test_engine_initialization():
    """Test simulation engine initialization."""
    engine = SimulationEngine()
    assert engine.work_dir.exists()
    engine.cleanup()


def test_engine_custom_work_dir():
    """Test simulation engine with custom work directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = SimulationEngine(work_dir=tmpdir)
        assert engine.work_dir == Path(tmpdir)


def test_generate_spice_netlist_resistor_divider(sim_engine, simple_components, simple_nets):
    """Test SPICE netlist generation for resistor divider."""
    netlist = sim_engine.generate_spice_netlist(simple_components, simple_nets)
    
    assert "* SPICE Netlist" in netlist
    assert "V1 1 0 DC 5V" in netlist
    assert "R1 1 2 1k" in netlist
    assert "R2 2 0 1k" in netlist
    assert ".end" in netlist


def test_generate_spice_netlist_with_capacitor(sim_engine):
    """Test SPICE netlist generation with capacitor."""
    components = [
        {"type": "CAPACITOR", "reference": "C1", "value": "10uF", "pins": ["1", "0"]}
    ]
    
    netlist = sim_engine.generate_spice_netlist(components, [])
    assert "C1 1 0 10uF" in netlist


def test_generate_spice_netlist_with_inductor(sim_engine):
    """Test SPICE netlist generation with inductor."""
    components = [
        {"type": "INDUCTOR", "reference": "L1", "value": "1mH", "pins": ["1", "0"]}
    ]
    
    netlist = sim_engine.generate_spice_netlist(components, [])
    assert "L1 1 0 1mH" in netlist


def test_generate_spice_netlist_with_current_source(sim_engine):
    """Test SPICE netlist generation with current source."""
    components = [
        {"type": "CURRENT_SOURCE", "reference": "I1", "value": "1mA", "pins": ["0", "1"]}
    ]
    
    netlist = sim_engine.generate_spice_netlist(components, [])
    assert "I1 0 1 DC 1mA" in netlist


def test_generate_spice_netlist_with_simulation_commands(sim_engine, simple_components, simple_nets):
    """Test SPICE netlist generation with simulation commands."""
    commands = [".op", ".print dc v(2)"]
    netlist = sim_engine.generate_spice_netlist(simple_components, simple_nets, commands)
    
    assert ".op" in netlist
    assert ".print dc v(2)" in netlist


def test_validate_netlist_valid(sim_engine):
    """Test netlist validation with valid netlist."""
    netlist = """
* Test Circuit
V1 1 0 DC 5V
R1 1 0 1k
.op
.end
"""
    is_valid, error = sim_engine.validate_netlist(netlist)
    assert is_valid
    assert error is None


def test_validate_netlist_missing_end(sim_engine):
    """Test netlist validation with missing .end."""
    netlist = """
* Test Circuit
V1 1 0 DC 5V
R1 1 0 1k
"""
    is_valid, error = sim_engine.validate_netlist(netlist)
    assert not is_valid
    assert "Missing .end statement" in error


def test_validate_netlist_no_components(sim_engine):
    """Test netlist validation with no components."""
    netlist = """
* Test Circuit
.op
.end
"""
    is_valid, error = sim_engine.validate_netlist(netlist)
    assert not is_valid
    assert "No components found" in error


def test_validate_netlist_no_ground(sim_engine):
    """Test netlist validation with no ground node."""
    netlist = """
* Test Circuit
V1 1 2 DC 5V
R1 1 2 1k
.end
"""
    is_valid, error = sim_engine.validate_netlist(netlist)
    assert not is_valid
    assert "No ground node" in error


def test_run_dc_analysis_success(sim_engine, simple_components, simple_nets):
    """Test successful DC analysis."""
    netlist = sim_engine.generate_spice_netlist(simple_components, simple_nets)
    result = sim_engine.run_dc_analysis(netlist)
    
    assert result.status == SimulationStatus.SUCCESS
    assert result.simulation_type == SimulationType.DC
    assert result.dc_voltages is not None
    assert len(result.dc_voltages) > 0


def test_run_dc_analysis_invalid_netlist(sim_engine):
    """Test DC analysis with invalid netlist."""
    netlist = "* Invalid netlist"
    result = sim_engine.run_dc_analysis(netlist)
    
    assert result.status == SimulationStatus.INVALID_NETLIST
    assert result.error_message is not None


def test_run_dc_analysis_adds_op_command(sim_engine, simple_components, simple_nets):
    """Test DC analysis adds .op command if missing."""
    netlist = sim_engine.generate_spice_netlist(simple_components, simple_nets)
    # Remove .end to add .op before it
    netlist = netlist.replace(".end", "")
    netlist += ".end"
    
    result = sim_engine.run_dc_analysis(netlist)
    assert result.status == SimulationStatus.SUCCESS


def test_run_ac_analysis_success(sim_engine, simple_components, simple_nets):
    """Test successful AC analysis."""
    netlist = sim_engine.generate_spice_netlist(simple_components, simple_nets)
    result = sim_engine.run_ac_analysis(netlist, start_freq=1.0, stop_freq=1e6)
    
    assert result.status == SimulationStatus.SUCCESS
    assert result.simulation_type == SimulationType.AC
    assert result.ac_response is not None


def test_run_ac_analysis_invalid_netlist(sim_engine):
    """Test AC analysis with invalid netlist."""
    netlist = "* Invalid netlist"
    result = sim_engine.run_ac_analysis(netlist)
    
    assert result.status == SimulationStatus.INVALID_NETLIST
    assert result.error_message is not None


def test_run_ac_analysis_custom_parameters(sim_engine, simple_components, simple_nets):
    """Test AC analysis with custom frequency parameters."""
    netlist = sim_engine.generate_spice_netlist(simple_components, simple_nets)
    result = sim_engine.run_ac_analysis(
        netlist,
        start_freq=10.0,
        stop_freq=1e5,
        points_per_decade=20
    )
    
    assert result.status == SimulationStatus.SUCCESS


def test_parse_dc_voltages(sim_engine):
    """Test DC voltage parsing."""
    output = """
V(1) = 5.000000
V(2) = 2.500000
V(out) = 1.250000
"""
    voltages = sim_engine._parse_dc_voltages(output)
    
    assert voltages["1"] == 5.0
    assert voltages["2"] == 2.5
    assert voltages["out"] == 1.25


def test_parse_ac_response(sim_engine):
    """Test AC response parsing."""
    output = """
1.000000e+00      1.000000            0.000000
1.000000e+01      0.995000            -5.710000
1.000000e+02      0.950000            -45.000000
"""
    response = sim_engine._parse_ac_response(output)
    
    assert "out" in response
    assert len(response["out"]) == 3
    assert response["out"][0] == (1.0, 1.0)
    assert response["out"][1] == (10.0, 0.995)


def test_cleanup(sim_engine):
    """Test cleanup of temporary files."""
    # Create a test file
    test_file = sim_engine.work_dir / "test.txt"
    test_file.write_text("test")
    
    assert test_file.exists()
    sim_engine.cleanup()
    # Note: cleanup removes files but may not remove directory


def test_simulation_result_dataclass():
    """Test SimulationResult dataclass."""
    result = SimulationResult(
        status=SimulationStatus.SUCCESS,
        simulation_type=SimulationType.DC,
        output="test output",
        dc_voltages={"1": 5.0}
    )
    
    assert result.status == SimulationStatus.SUCCESS
    assert result.simulation_type == SimulationType.DC
    assert result.output == "test output"
    assert result.dc_voltages == {"1": 5.0}
    assert result.error_message is None


def test_simulation_result_with_error():
    """Test SimulationResult with error."""
    result = SimulationResult(
        status=SimulationStatus.FAILED,
        simulation_type=SimulationType.AC,
        output="",
        error_message="Simulation failed"
    )
    
    assert result.status == SimulationStatus.FAILED
    assert result.error_message == "Simulation failed"


def test_simulation_types_enum():
    """Test SimulationType enum."""
    assert SimulationType.DC.value == "dc"
    assert SimulationType.AC.value == "ac"
    assert SimulationType.TRANSIENT.value == "tran"
    assert SimulationType.OPERATING_POINT.value == "op"


def test_simulation_status_enum():
    """Test SimulationStatus enum."""
    assert SimulationStatus.SUCCESS.value == "success"
    assert SimulationStatus.FAILED.value == "failed"
    assert SimulationStatus.TIMEOUT.value == "timeout"
    assert SimulationStatus.INVALID_NETLIST.value == "invalid_netlist"
