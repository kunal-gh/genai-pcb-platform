"""
Unit tests for simulation visualization.
"""

import pytest
from pathlib import Path
import tempfile
import json

from src.services.simulation_visualization import (
    SimulationVisualizer,
    PlotType,
    PlotData,
    SimulationDiagnostic
)
from src.services.simulation_engine import (
    SimulationResult,
    SimulationType,
    SimulationStatus
)


@pytest.fixture
def visualizer():
    """Create simulation visualizer."""
    return SimulationVisualizer()


@pytest.fixture
def dc_result():
    """Create sample DC simulation result."""
    return SimulationResult(
        status=SimulationStatus.SUCCESS,
        simulation_type=SimulationType.DC,
        output="DC analysis output",
        dc_voltages={"1": 5.0, "2": 2.5, "out": 2.5}
    )


@pytest.fixture
def ac_result():
    """Create sample AC simulation result."""
    return SimulationResult(
        status=SimulationStatus.SUCCESS,
        simulation_type=SimulationType.AC,
        output="AC analysis output",
        ac_response={
            "out": [
                (1.0, 1.0),
                (10.0, 0.995),
                (100.0, 0.95),
                (1000.0, 0.707),
                (10000.0, 0.1)
            ]
        }
    )


@pytest.fixture
def failed_result():
    """Create sample failed simulation result."""
    return SimulationResult(
        status=SimulationStatus.FAILED,
        simulation_type=SimulationType.DC,
        output="",
        error_message="Convergence failure"
    )


def test_visualizer_initialization(visualizer):
    """Test visualizer initialization."""
    assert visualizer is not None


def test_create_dc_voltage_plot(visualizer, dc_result):
    """Test DC voltage plot creation."""
    plot = visualizer.create_dc_voltage_plot(dc_result.dc_voltages)
    
    assert plot.plot_type == PlotType.DC_VOLTAGE
    assert plot.title == "DC Operating Point Voltages"
    assert plot.x_label == "Node"
    assert plot.y_label == "Voltage (V)"
    assert len(plot.x_data) == 3
    assert len(plot.y_data) == 3
    assert plot.x_scale == "linear"
    assert plot.y_scale == "linear"


def test_create_dc_voltage_plot_custom_title(visualizer, dc_result):
    """Test DC voltage plot with custom title."""
    plot = visualizer.create_dc_voltage_plot(
        dc_result.dc_voltages,
        title="Custom Title"
    )
    assert plot.title == "Custom Title"


def test_create_ac_magnitude_plot(visualizer, ac_result):
    """Test AC magnitude plot creation."""
    plot = visualizer.create_ac_magnitude_plot(ac_result.ac_response)
    
    assert plot.plot_type == PlotType.AC_MAGNITUDE
    assert plot.title == "AC Magnitude Response"
    assert plot.x_label == "Frequency (Hz)"
    assert plot.y_label == "Magnitude"
    assert len(plot.x_data) == 5
    assert len(plot.y_data) == 5
    assert plot.x_scale == "log"
    assert plot.y_scale == "linear"


def test_create_ac_magnitude_plot_specific_node(visualizer, ac_result):
    """Test AC magnitude plot for specific node."""
    plot = visualizer.create_ac_magnitude_plot(ac_result.ac_response, node="out")
    assert len(plot.x_data) == 5


def test_create_ac_magnitude_plot_empty_response(visualizer):
    """Test AC magnitude plot with empty response."""
    plot = visualizer.create_ac_magnitude_plot({})
    assert len(plot.x_data) == 0
    assert len(plot.y_data) == 0


def test_create_bode_plot(visualizer, ac_result):
    """Test Bode plot creation."""
    mag_plot, phase_plot = visualizer.create_bode_plot(ac_result.ac_response)
    
    assert mag_plot.plot_type == PlotType.AC_MAGNITUDE
    assert phase_plot.plot_type == PlotType.AC_PHASE
    assert "Bode Plot" in mag_plot.title
    assert "Bode Plot" in phase_plot.title
    assert len(mag_plot.x_data) == len(phase_plot.x_data)


def test_diagnose_convergence_failure(visualizer):
    """Test diagnosis of convergence failure."""
    diagnostics = visualizer.diagnose_simulation_failure(
        "Convergence failure at node 5",
        "V1 1 0 5V\nR1 1 0 1k\n.end"
    )
    
    assert len(diagnostics) > 0
    assert any("Convergence" in d.error_type for d in diagnostics)
    assert any("converge" in d.message.lower() for d in diagnostics)


def test_diagnose_singular_matrix(visualizer):
    """Test diagnosis of singular matrix error."""
    diagnostics = visualizer.diagnose_simulation_failure(
        "Singular matrix detected",
        "V1 1 0 5V\n.end"
    )
    
    assert len(diagnostics) > 0
    assert any("Singular" in d.error_type for d in diagnostics)


def test_diagnose_timestep_error(visualizer):
    """Test diagnosis of timestep error."""
    diagnostics = visualizer.diagnose_simulation_failure(
        "Timestep too small",
        "V1 1 0 5V\nR1 1 0 1k\n.end"
    )
    
    assert len(diagnostics) > 0
    assert any("Timestep" in d.error_type for d in diagnostics)


def test_diagnose_floating_nodes(visualizer):
    """Test detection of floating nodes."""
    netlist = """
V1 1 0 5V
R1 1 2 1k
R2 3 0 1k
.end
"""
    diagnostics = visualizer.diagnose_simulation_failure("Error", netlist)
    
    # Should detect node 2 and 3 as potentially floating
    assert len(diagnostics) > 0


def test_diagnose_generic_error(visualizer):
    """Test diagnosis of generic error."""
    diagnostics = visualizer.diagnose_simulation_failure(
        "Unknown error occurred",
        "V1 1 0 5V\nR1 1 0 1k\n.end"
    )
    
    assert len(diagnostics) > 0
    # Should have at least one diagnostic (could be floating nodes or generic)
    assert any(d.error_type in ["Simulation Error", "Floating Nodes"] for d in diagnostics)


def test_detect_floating_nodes(visualizer):
    """Test floating node detection."""
    netlist = """
V1 1 0 5V
R1 1 2 1k
R2 2 3 1k
C1 4 0 10uF
.end
"""
    floating = visualizer._detect_floating_nodes(netlist)
    
    # Node 3 and 4 appear only once (besides ground)
    assert len(floating) >= 1


def test_generate_simulation_report_dc(visualizer, dc_result):
    """Test simulation report generation for DC analysis."""
    report = visualizer.generate_simulation_report(dc_result)
    
    assert report["status"] == "success"
    assert report["simulation_type"] == "dc"
    assert report["success"] is True
    assert "dc_voltages" in report
    assert "dc_summary" in report
    assert report["dc_summary"]["num_nodes"] == 3


def test_generate_simulation_report_ac(visualizer, ac_result):
    """Test simulation report generation for AC analysis."""
    report = visualizer.generate_simulation_report(ac_result)
    
    assert report["status"] == "success"
    assert report["simulation_type"] == "ac"
    assert "ac_response" in report
    assert "ac_summary_out" in report


def test_generate_simulation_report_with_plots(visualizer, dc_result):
    """Test simulation report with plot data."""
    report = visualizer.generate_simulation_report(dc_result, include_plots=True)
    
    assert "dc_plot" in report
    assert report["dc_plot"]["type"] == "dc_voltage"


def test_generate_simulation_report_without_plots(visualizer, dc_result):
    """Test simulation report without plot data."""
    report = visualizer.generate_simulation_report(dc_result, include_plots=False)
    
    assert "dc_plot" not in report


def test_generate_simulation_report_failed(visualizer, failed_result):
    """Test simulation report for failed simulation."""
    report = visualizer.generate_simulation_report(failed_result)
    
    assert report["status"] == "failed"
    assert report["success"] is False
    assert "error" in report


def test_calculate_bandwidth(visualizer):
    """Test bandwidth calculation."""
    freq_response = [
        (1.0, 1.0),
        (10.0, 0.9),
        (100.0, 0.7),
        (1000.0, 0.5),
        (10000.0, 0.1)
    ]
    
    bandwidth = visualizer._calculate_bandwidth(freq_response)
    assert bandwidth is not None
    assert bandwidth > 0


def test_calculate_bandwidth_empty(visualizer):
    """Test bandwidth calculation with empty data."""
    bandwidth = visualizer._calculate_bandwidth([])
    assert bandwidth is None


def test_plot_to_dict(visualizer):
    """Test plot data conversion to dictionary."""
    plot = PlotData(
        plot_type=PlotType.DC_VOLTAGE,
        title="Test Plot",
        x_label="X",
        y_label="Y",
        x_data=[1, 2, 3],
        y_data=[4, 5, 6]
    )
    
    plot_dict = visualizer._plot_to_dict(plot)
    
    assert plot_dict["type"] == "dc_voltage"
    assert plot_dict["title"] == "Test Plot"
    assert plot_dict["x_data"] == [1, 2, 3]
    assert plot_dict["y_data"] == [4, 5, 6]


def test_export_results_json(visualizer, dc_result):
    """Test JSON export of simulation results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.json"
        visualizer.export_results_json(dc_result, output_path)
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["status"] == "success"
        assert "dc_voltages" in data


def test_export_results_csv_dc(visualizer, dc_result):
    """Test CSV export of DC simulation results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.csv"
        visualizer.export_results_csv(dc_result, output_path)
        
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "Node,Voltage" in content
        assert "1,5.0" in content


def test_export_results_csv_ac(visualizer, ac_result):
    """Test CSV export of AC simulation results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.csv"
        visualizer.export_results_csv(ac_result, output_path)
        
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "Frequency" in content
        assert "Magnitude" in content


def test_format_diagnostic_report(visualizer):
    """Test diagnostic report formatting."""
    diagnostics = [
        SimulationDiagnostic(
            error_type="Test Error",
            message="Test message",
            suggestion="Test suggestion",
            affected_components=["R1", "C1"]
        )
    ]
    
    report = visualizer.format_diagnostic_report(diagnostics)
    
    assert "Test Error" in report
    assert "Test message" in report
    assert "Test suggestion" in report
    assert "R1" in report


def test_format_diagnostic_report_empty(visualizer):
    """Test diagnostic report formatting with no diagnostics."""
    report = visualizer.format_diagnostic_report([])
    assert "No diagnostics" in report


def test_plot_data_dataclass():
    """Test PlotData dataclass."""
    plot = PlotData(
        plot_type=PlotType.AC_MAGNITUDE,
        title="Test",
        x_label="X",
        y_label="Y",
        x_data=[1, 2],
        y_data=[3, 4],
        x_scale="log",
        y_scale="linear"
    )
    
    assert plot.plot_type == PlotType.AC_MAGNITUDE
    assert plot.x_scale == "log"


def test_simulation_diagnostic_dataclass():
    """Test SimulationDiagnostic dataclass."""
    diag = SimulationDiagnostic(
        error_type="Test",
        message="Message",
        suggestion="Suggestion",
        affected_components=["R1"]
    )
    
    assert diag.error_type == "Test"
    assert diag.affected_components == ["R1"]


def test_plot_type_enum():
    """Test PlotType enum."""
    assert PlotType.DC_VOLTAGE.value == "dc_voltage"
    assert PlotType.AC_MAGNITUDE.value == "ac_magnitude"
    assert PlotType.AC_PHASE.value == "ac_phase"
    assert PlotType.BODE.value == "bode"
