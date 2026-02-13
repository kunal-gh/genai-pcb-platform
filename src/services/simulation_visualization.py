"""
Simulation result visualization and reporting.

This module provides visualization and reporting capabilities for SPICE simulation results,
including graphical displays, failure diagnostics, and result export.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class PlotType(Enum):
    """Types of simulation plots."""
    DC_VOLTAGE = "dc_voltage"
    AC_MAGNITUDE = "ac_magnitude"
    AC_PHASE = "ac_phase"
    BODE = "bode"


@dataclass
class PlotData:
    """Data for a simulation plot."""
    plot_type: PlotType
    title: str
    x_label: str
    y_label: str
    x_data: List[float]
    y_data: List[float]
    x_scale: str = "linear"  # "linear" or "log"
    y_scale: str = "linear"


@dataclass
class SimulationDiagnostic:
    """Diagnostic information for simulation failures."""
    error_type: str
    message: str
    suggestion: str
    affected_components: Optional[List[str]] = None


class SimulationVisualizer:
    """
    Visualization and reporting for simulation results.
    
    Provides methods to create plots, diagnose failures, and export results
    in various formats.
    """
    
    def __init__(self):
        """Initialize simulation visualizer."""
        pass
    
    def create_dc_voltage_plot(
        self,
        dc_voltages: Dict[str, float],
        title: str = "DC Operating Point Voltages"
    ) -> PlotData:
        """
        Create plot data for DC voltage results.
        
        Args:
            dc_voltages: Dictionary of node names to voltages
            title: Plot title
        
        Returns:
            PlotData for DC voltages
        """
        nodes = sorted(dc_voltages.keys())
        voltages = [dc_voltages[node] for node in nodes]
        
        return PlotData(
            plot_type=PlotType.DC_VOLTAGE,
            title=title,
            x_label="Node",
            y_label="Voltage (V)",
            x_data=list(range(len(nodes))),
            y_data=voltages,
            x_scale="linear",
            y_scale="linear"
        )
    
    def create_ac_magnitude_plot(
        self,
        ac_response: Dict[str, List[Tuple[float, float]]],
        node: str = "out",
        title: str = "AC Magnitude Response"
    ) -> PlotData:
        """
        Create plot data for AC magnitude response.
        
        Args:
            ac_response: Dictionary of node names to (frequency, magnitude) tuples
            node: Node to plot
            title: Plot title
        
        Returns:
            PlotData for AC magnitude
        """
        if node not in ac_response:
            node = list(ac_response.keys())[0] if ac_response else "out"
        
        data = ac_response.get(node, [])
        frequencies = [freq for freq, _ in data]
        magnitudes = [mag for _, mag in data]
        
        return PlotData(
            plot_type=PlotType.AC_MAGNITUDE,
            title=title,
            x_label="Frequency (Hz)",
            y_label="Magnitude",
            x_data=frequencies,
            y_data=magnitudes,
            x_scale="log",
            y_scale="linear"
        )
    
    def create_bode_plot(
        self,
        ac_response: Dict[str, List[Tuple[float, float]]],
        node: str = "out"
    ) -> Tuple[PlotData, PlotData]:
        """
        Create Bode plot data (magnitude and phase).
        
        Args:
            ac_response: Dictionary of node names to (frequency, magnitude) tuples
            node: Node to plot
        
        Returns:
            Tuple of (magnitude_plot, phase_plot)
        """
        magnitude_plot = self.create_ac_magnitude_plot(
            ac_response,
            node,
            f"Bode Plot - Magnitude ({node})"
        )
        
        # Phase plot would need phase data from simulation
        # For now, create placeholder
        phase_plot = PlotData(
            plot_type=PlotType.AC_PHASE,
            title=f"Bode Plot - Phase ({node})",
            x_label="Frequency (Hz)",
            y_label="Phase (degrees)",
            x_data=magnitude_plot.x_data,
            y_data=[0.0] * len(magnitude_plot.x_data),
            x_scale="log",
            y_scale="linear"
        )
        
        return magnitude_plot, phase_plot
    
    def diagnose_simulation_failure(
        self,
        error_message: str,
        netlist: str
    ) -> List[SimulationDiagnostic]:
        """
        Diagnose simulation failure and provide suggestions.
        
        Args:
            error_message: Error message from simulation
            netlist: SPICE netlist that failed
        
        Returns:
            List of diagnostic information
        """
        diagnostics = []
        
        # Check for common issues
        if "convergence" in error_message.lower():
            diagnostics.append(SimulationDiagnostic(
                error_type="Convergence Failure",
                message="Simulation failed to converge to a solution",
                suggestion="Try adding initial conditions, reducing step size, or checking for floating nodes"
            ))
        
        if "singular matrix" in error_message.lower():
            diagnostics.append(SimulationDiagnostic(
                error_type="Singular Matrix",
                message="Circuit matrix is singular (not invertible)",
                suggestion="Check for floating nodes, voltage source loops, or current source cutsets"
            ))
        
        if "timestep too small" in error_message.lower():
            diagnostics.append(SimulationDiagnostic(
                error_type="Timestep Error",
                message="Simulation timestep became too small",
                suggestion="Check for stiff equations, reduce tolerances, or simplify the circuit"
            ))
        
        # Check netlist for potential issues
        if "floating" in error_message.lower() or not diagnostics:
            floating_nodes = self._detect_floating_nodes(netlist)
            if floating_nodes:
                diagnostics.append(SimulationDiagnostic(
                    error_type="Floating Nodes",
                    message=f"Detected potentially floating nodes: {', '.join(floating_nodes)}",
                    suggestion="Connect all nodes to ground through high-value resistors or voltage sources",
                    affected_components=floating_nodes
                ))
        
        # Generic diagnostic if no specific issue found
        if not diagnostics:
            diagnostics.append(SimulationDiagnostic(
                error_type="Simulation Error",
                message=error_message,
                suggestion="Check netlist syntax, component values, and circuit topology"
            ))
        
        return diagnostics
    
    def _detect_floating_nodes(self, netlist: str) -> List[str]:
        """Detect potentially floating nodes in netlist."""
        # Simple heuristic: find nodes that appear only once
        node_counts = {}
        
        for line in netlist.split("\n"):
            line = line.strip()
            if not line or line.startswith("*") or line.startswith("."):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                # Extract node names (skip component reference)
                for node in parts[1:]:
                    if node.replace(".", "").replace("-", "").isalnum():
                        node_counts[node] = node_counts.get(node, 0) + 1
        
        # Nodes appearing only once might be floating
        floating = [node for node, count in node_counts.items() if count == 1 and node != "0"]
        return floating[:5]  # Limit to 5 nodes
    
    def generate_simulation_report(
        self,
        simulation_result,
        include_plots: bool = True
    ) -> Dict:
        """
        Generate comprehensive simulation report.
        
        Args:
            simulation_result: SimulationResult object
            include_plots: Whether to include plot data
        
        Returns:
            Dictionary with report data
        """
        report = {
            "status": simulation_result.status.value,
            "simulation_type": simulation_result.simulation_type.value,
            "success": simulation_result.status.value == "success"
        }
        
        if simulation_result.error_message:
            report["error"] = simulation_result.error_message
        
        if simulation_result.warnings:
            report["warnings"] = simulation_result.warnings
        
        # Add DC results
        if simulation_result.dc_voltages:
            report["dc_voltages"] = simulation_result.dc_voltages
            report["dc_summary"] = {
                "num_nodes": len(simulation_result.dc_voltages),
                "max_voltage": max(simulation_result.dc_voltages.values()),
                "min_voltage": min(simulation_result.dc_voltages.values())
            }
            
            if include_plots:
                plot = self.create_dc_voltage_plot(simulation_result.dc_voltages)
                report["dc_plot"] = self._plot_to_dict(plot)
        
        # Add AC results
        if simulation_result.ac_response:
            report["ac_response"] = {
                node: [(f, m) for f, m in data]
                for node, data in simulation_result.ac_response.items()
            }
            
            # Calculate frequency response metrics
            for node, data in simulation_result.ac_response.items():
                if data:
                    magnitudes = [m for _, m in data]
                    report[f"ac_summary_{node}"] = {
                        "max_magnitude": max(magnitudes),
                        "min_magnitude": min(magnitudes),
                        "bandwidth": self._calculate_bandwidth(data)
                    }
            
            if include_plots:
                node = list(simulation_result.ac_response.keys())[0]
                mag_plot, phase_plot = self.create_bode_plot(simulation_result.ac_response, node)
                report["bode_magnitude_plot"] = self._plot_to_dict(mag_plot)
                report["bode_phase_plot"] = self._plot_to_dict(phase_plot)
        
        return report
    
    def _calculate_bandwidth(self, freq_response: List[Tuple[float, float]]) -> Optional[float]:
        """Calculate -3dB bandwidth from frequency response."""
        if not freq_response:
            return None
        
        magnitudes = [m for _, m in freq_response]
        max_mag = max(magnitudes)
        cutoff = max_mag / 1.414  # -3dB point
        
        # Find first frequency below cutoff
        for freq, mag in freq_response:
            if mag < cutoff:
                return freq
        
        return None
    
    def _plot_to_dict(self, plot: PlotData) -> Dict:
        """Convert PlotData to dictionary."""
        return {
            "type": plot.plot_type.value,
            "title": plot.title,
            "x_label": plot.x_label,
            "y_label": plot.y_label,
            "x_data": plot.x_data,
            "y_data": plot.y_data,
            "x_scale": plot.x_scale,
            "y_scale": plot.y_scale
        }
    
    def export_results_json(
        self,
        simulation_result,
        output_path: Path
    ):
        """
        Export simulation results to JSON file.
        
        Args:
            simulation_result: SimulationResult object
            output_path: Path to output JSON file
        """
        report = self.generate_simulation_report(simulation_result, include_plots=True)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    
    def export_results_csv(
        self,
        simulation_result,
        output_path: Path
    ):
        """
        Export simulation results to CSV file.
        
        Args:
            simulation_result: SimulationResult object
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            if simulation_result.dc_voltages:
                f.write("Node,Voltage (V)\n")
                for node, voltage in sorted(simulation_result.dc_voltages.items()):
                    f.write(f"{node},{voltage}\n")
            
            elif simulation_result.ac_response:
                f.write("Frequency (Hz),Magnitude\n")
                node = list(simulation_result.ac_response.keys())[0]
                for freq, mag in simulation_result.ac_response[node]:
                    f.write(f"{freq},{mag}\n")
    
    def format_diagnostic_report(
        self,
        diagnostics: List[SimulationDiagnostic]
    ) -> str:
        """
        Format diagnostic information as human-readable text.
        
        Args:
            diagnostics: List of diagnostic information
        
        Returns:
            Formatted diagnostic report
        """
        if not diagnostics:
            return "No diagnostics available."
        
        lines = ["Simulation Failure Diagnostics", "=" * 40, ""]
        
        for i, diag in enumerate(diagnostics, 1):
            lines.append(f"{i}. {diag.error_type}")
            lines.append(f"   Message: {diag.message}")
            lines.append(f"   Suggestion: {diag.suggestion}")
            
            if diag.affected_components:
                lines.append(f"   Affected: {', '.join(diag.affected_components)}")
            
            lines.append("")
        
        return "\n".join(lines)
