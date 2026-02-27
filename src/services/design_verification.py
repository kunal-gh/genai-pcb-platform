"""
Design verification engine for ERC and DRC checking.

Implements electrical rule checking (ERC) and design rule checking (DRC)
with customizable rules and comprehensive error reporting.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of design rule violations."""
    ERC_ERROR = "erc_error"
    ERC_WARNING = "erc_warning"
    DRC_ERROR = "drc_error"
    DRC_WARNING = "drc_warning"
    CONNECTIVITY_ERROR = "connectivity_error"


class Severity(Enum):
    """Severity levels for violations."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class DesignViolation:
    """Represents a design rule or electrical rule violation."""
    violation_type: ViolationType
    severity: Severity
    message: str
    component: Optional[str] = None
    net: Optional[str] = None
    location: Optional[Tuple[float, float]] = None
    suggested_fix: Optional[str] = None
    rule_name: Optional[str] = None


@dataclass
class DesignRules:
    """Design rules configuration."""
    # Trace rules
    min_trace_width: float = 0.1  # mm
    max_trace_width: float = 10.0  # mm
    min_via_size: float = 0.2  # mm
    max_via_size: float = 2.0  # mm
    
    # Spacing rules
    min_trace_spacing: float = 0.1  # mm
    min_via_spacing: float = 0.2  # mm
    min_component_spacing: float = 0.5  # mm
    
    # Electrical rules
    max_fanout: int = 50
    min_impedance: float = 25.0  # ohms
    max_impedance: float = 120.0  # ohms
    
    # Power rules
    min_power_trace_width: float = 0.3  # mm
    max_current_density: float = 35.0  # A/mm²
    
    # Manufacturing rules
    min_drill_size: float = 0.15  # mm
    max_drill_size: float = 6.0  # mm
    min_annular_ring: float = 0.05  # mm


class DesignVerificationEngine:
    """
    Design verification engine for ERC and DRC checking.
    
    Performs comprehensive electrical and design rule checking
    with customizable rules and detailed error reporting.
    """
    
    def __init__(self, design_rules: Optional[DesignRules] = None):
        """
        Initialize verification engine.
        
        Args:
            design_rules: Custom design rules (uses defaults if None)
        """
        self.design_rules = design_rules or DesignRules()
        self.violations: List[DesignViolation] = []
        
        # Component pin definitions for ERC
        self.component_pins = {
            "resistor": {"pins": 2, "types": ["passive", "passive"]},
            "capacitor": {"pins": 2, "types": ["passive", "passive"]},
            "inductor": {"pins": 2, "types": ["passive", "passive"]},
            "led": {"pins": 2, "types": ["anode", "cathode"]},
            "diode": {"pins": 2, "types": ["anode", "cathode"]},
            "transistor_npn": {"pins": 3, "types": ["collector", "base", "emitter"]},
            "transistor_pnp": {"pins": 3, "types": ["collector", "base", "emitter"]},
            "op_amp": {"pins": 8, "types": ["out", "in-", "in+", "vcc", "vee", "nc", "nc", "nc"]},
            "microcontroller": {"pins": "variable", "types": ["io", "power", "ground"]},
        }
    
    def verify_design(
        self,
        netlist_data: Dict[str, Any],
        pcb_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive design verification.
        
        Args:
            netlist_data: Netlist information
            pcb_data: PCB layout data (optional)
            
        Returns:
            Verification results with violations and summary
        """
        self.violations = []
        
        try:
            # Perform ERC checks
            self._perform_erc(netlist_data)
            
            # Perform DRC checks if PCB data available
            if pcb_data:
                self._perform_drc(pcb_data)
            
            # Perform connectivity validation
            self._validate_connectivity(netlist_data)
            
            # Generate verification report
            return self._generate_verification_report()
            
        except Exception as e:
            logger.error(f"Design verification failed: {str(e)}")
            return {
                "success": False,
                "error": f"Verification failed: {str(e)}",
                "violations": []
            }
    
    def _perform_erc(self, netlist_data: Dict[str, Any]) -> None:
        """Perform electrical rule checking."""
        components = netlist_data.get("components", [])
        nets = netlist_data.get("nets", [])
        
        # Check component connections
        self._check_component_connections(components, nets)
        
        # Check power connections
        self._check_power_connections(components, nets)
        
        # Check unconnected pins
        self._check_unconnected_pins(components, nets)
        
        # Check pin conflicts
        self._check_pin_conflicts(nets)
    
    def _check_component_connections(
        self,
        components: List[Dict[str, Any]],
        nets: List[Dict[str, Any]]
    ) -> None:
        """Check component connection validity."""
        for component in components:
            comp_ref = component.get("reference", "")
            comp_type = component.get("type", "").lower()
            
            # Get expected pin configuration
            pin_config = self._get_component_pin_config(comp_type)
            if not pin_config:
                continue
            
            # Check if component has correct number of connections
            connected_pins = self._get_component_connections(comp_ref, nets)
            expected_pins = pin_config.get("pins", 0)
            
            if isinstance(expected_pins, int) and len(connected_pins) != expected_pins:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.ERC_ERROR,
                    severity=Severity.ERROR,
                    message=f"Component {comp_ref} has {len(connected_pins)} connections, expected {expected_pins}",
                    component=comp_ref,
                    suggested_fix=f"Check connections to {comp_ref} and ensure all pins are properly connected"
                ))
    
    def _check_power_connections(
        self,
        components: List[Dict[str, Any]],
        nets: List[Dict[str, Any]]
    ) -> None:
        """Check power supply connections."""
        power_nets = set()
        ground_nets = set()
        
        # Identify power and ground nets
        for net in nets:
            net_name = net.get("name", "").lower()
            if any(power in net_name for power in ["vcc", "vdd", "power", "+5v", "+3v3", "+12v"]):
                power_nets.add(net.get("name"))
            elif any(gnd in net_name for gnd in ["gnd", "ground", "vss", "vee"]):
                ground_nets.add(net.get("name"))
        
        # Check if design has power connections
        if not power_nets:
            self.violations.append(DesignViolation(
                violation_type=ViolationType.ERC_WARNING,
                severity=Severity.WARNING,
                message="No power nets detected in design",
                suggested_fix="Add power supply connections (VCC, VDD, etc.)"
            ))
        
        if not ground_nets:
            self.violations.append(DesignViolation(
                violation_type=ViolationType.ERC_ERROR,
                severity=Severity.ERROR,
                message="No ground nets detected in design",
                suggested_fix="Add ground connections (GND, VSS, etc.)"
            ))
    
    def _check_unconnected_pins(
        self,
        components: List[Dict[str, Any]],
        nets: List[Dict[str, Any]]
    ) -> None:
        """Check for unconnected pins."""
        all_connections = set()
        
        # Collect all pin connections
        for net in nets:
            connections = net.get("connections", [])
            for conn in connections:
                all_connections.add(f"{conn.get('component')}.{conn.get('pin')}")
        
        # Check each component for unconnected pins
        for component in components:
            comp_ref = component.get("reference", "")
            comp_type = component.get("type", "").lower()
            
            pin_config = self._get_component_pin_config(comp_type)
            if not pin_config or pin_config.get("pins") == "variable":
                continue
            
            expected_pins = pin_config.get("pins", 0)
            for pin_num in range(1, expected_pins + 1):
                pin_id = f"{comp_ref}.{pin_num}"
                if pin_id not in all_connections:
                    self.violations.append(DesignViolation(
                        violation_type=ViolationType.ERC_WARNING,
                        severity=Severity.WARNING,
                        message=f"Unconnected pin {pin_num} on component {comp_ref}",
                        component=comp_ref,
                        suggested_fix=f"Connect pin {pin_num} of {comp_ref} or mark as no-connect"
                    ))
    
    def _check_pin_conflicts(self, nets: List[Dict[str, Any]]) -> None:
        """Check for pin connection conflicts."""
        for net in nets:
            net_name = net.get("name", "")
            connections = net.get("connections", [])
            
            # Check for multiple outputs on same net
            output_count = 0
            for conn in connections:
                comp_ref = conn.get("component", "")
                pin_num = conn.get("pin", "")
                
                # Simple heuristic: assume pin 1 is often output for active components
                if self._is_output_pin(comp_ref, pin_num):
                    output_count += 1
            
            if output_count > 1:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.ERC_ERROR,
                    severity=Severity.ERROR,
                    message=f"Multiple outputs connected to net {net_name}",
                    net=net_name,
                    suggested_fix=f"Check connections on net {net_name} - multiple outputs may cause conflicts"
                ))
    
    def _perform_drc(self, pcb_data: Dict[str, Any]) -> None:
        """Perform design rule checking."""
        traces = pcb_data.get("traces", [])
        vias = pcb_data.get("vias", [])
        components = pcb_data.get("components", [])
        
        # Check trace width rules
        self._check_trace_widths(traces)
        
        # Check via size rules
        self._check_via_sizes(vias)
        
        # Check spacing rules
        self._check_spacing_rules(traces, vias, components)
        
        # Check manufacturing rules
        self._check_manufacturing_rules(pcb_data)
    
    def _check_trace_widths(self, traces: List[Dict[str, Any]]) -> None:
        """Check trace width design rules."""
        for trace in traces:
            width = trace.get("width", 0.0)
            net_name = trace.get("net", "")
            
            if width < self.design_rules.min_trace_width:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.DRC_ERROR,
                    severity=Severity.ERROR,
                    message=f"Trace width {width}mm below minimum {self.design_rules.min_trace_width}mm",
                    net=net_name,
                    suggested_fix=f"Increase trace width to at least {self.design_rules.min_trace_width}mm",
                    rule_name="min_trace_width"
                ))
            
            if width > self.design_rules.max_trace_width:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.DRC_WARNING,
                    severity=Severity.WARNING,
                    message=f"Trace width {width}mm exceeds maximum {self.design_rules.max_trace_width}mm",
                    net=net_name,
                    suggested_fix=f"Consider reducing trace width below {self.design_rules.max_trace_width}mm",
                    rule_name="max_trace_width"
                ))
    
    def _check_via_sizes(self, vias: List[Dict[str, Any]]) -> None:
        """Check via size design rules."""
        for via in vias:
            size = via.get("size", 0.0)
            location = (via.get("x", 0.0), via.get("y", 0.0))
            
            if size < self.design_rules.min_via_size:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.DRC_ERROR,
                    severity=Severity.ERROR,
                    message=f"Via size {size}mm below minimum {self.design_rules.min_via_size}mm",
                    location=location,
                    suggested_fix=f"Increase via size to at least {self.design_rules.min_via_size}mm",
                    rule_name="min_via_size"
                ))
            
            if size > self.design_rules.max_via_size:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.DRC_WARNING,
                    severity=Severity.WARNING,
                    message=f"Via size {size}mm exceeds maximum {self.design_rules.max_via_size}mm",
                    location=location,
                    suggested_fix=f"Consider reducing via size below {self.design_rules.max_via_size}mm",
                    rule_name="max_via_size"
                ))
    
    def _check_spacing_rules(
        self,
        traces: List[Dict[str, Any]],
        vias: List[Dict[str, Any]],
        components: List[Dict[str, Any]]
    ) -> None:
        """Check spacing design rules."""
        # Check trace-to-trace spacing
        for i, trace1 in enumerate(traces):
            for trace2 in traces[i+1:]:
                spacing = self._calculate_trace_spacing(trace1, trace2)
                if spacing < self.design_rules.min_trace_spacing:
                    self.violations.append(DesignViolation(
                        violation_type=ViolationType.DRC_ERROR,
                        severity=Severity.ERROR,
                        message=f"Trace spacing {spacing:.3f}mm below minimum {self.design_rules.min_trace_spacing}mm",
                        suggested_fix=f"Increase spacing between traces to at least {self.design_rules.min_trace_spacing}mm",
                        rule_name="min_trace_spacing"
                    ))
        
        # Check component spacing
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                spacing = self._calculate_component_spacing(comp1, comp2)
                if spacing < self.design_rules.min_component_spacing:
                    comp1_ref = comp1.get("reference", "")
                    comp2_ref = comp2.get("reference", "")
                    self.violations.append(DesignViolation(
                        violation_type=ViolationType.DRC_WARNING,
                        severity=Severity.WARNING,
                        message=f"Components {comp1_ref} and {comp2_ref} spacing {spacing:.3f}mm below recommended {self.design_rules.min_component_spacing}mm",
                        suggested_fix=f"Increase spacing between components to at least {self.design_rules.min_component_spacing}mm",
                        rule_name="min_component_spacing"
                    ))
    
    def _check_manufacturing_rules(self, pcb_data: Dict[str, Any]) -> None:
        """Check manufacturing-specific design rules."""
        drill_holes = pcb_data.get("drill_holes", [])
        
        for hole in drill_holes:
            diameter = hole.get("diameter", 0.0)
            location = (hole.get("x", 0.0), hole.get("y", 0.0))
            
            if diameter < self.design_rules.min_drill_size:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.DRC_ERROR,
                    severity=Severity.ERROR,
                    message=f"Drill size {diameter}mm below minimum {self.design_rules.min_drill_size}mm",
                    location=location,
                    suggested_fix=f"Increase drill size to at least {self.design_rules.min_drill_size}mm",
                    rule_name="min_drill_size"
                ))
            
            if diameter > self.design_rules.max_drill_size:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.DRC_WARNING,
                    severity=Severity.WARNING,
                    message=f"Drill size {diameter}mm exceeds maximum {self.design_rules.max_drill_size}mm",
                    location=location,
                    suggested_fix=f"Consider reducing drill size below {self.design_rules.max_drill_size}mm",
                    rule_name="max_drill_size"
                ))
    
    def _validate_connectivity(self, netlist_data: Dict[str, Any]) -> None:
        """Validate net connectivity."""
        nets = netlist_data.get("nets", [])
        
        for net in nets:
            net_name = net.get("name", "")
            connections = net.get("connections", [])
            
            # Check for single-pin nets (usually errors)
            if len(connections) == 1:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.CONNECTIVITY_ERROR,
                    severity=Severity.WARNING,
                    message=f"Net {net_name} has only one connection",
                    net=net_name,
                    suggested_fix=f"Check if net {net_name} should connect to additional components"
                ))
            
            # Check for empty nets
            if len(connections) == 0:
                self.violations.append(DesignViolation(
                    violation_type=ViolationType.CONNECTIVITY_ERROR,
                    severity=Severity.ERROR,
                    message=f"Net {net_name} has no connections",
                    net=net_name,
                    suggested_fix=f"Remove unused net {net_name} or add connections"
                ))
    
    def _get_component_pin_config(self, comp_type: str) -> Optional[Dict[str, Any]]:
        """Get pin configuration for component type."""
        # Normalize component type
        comp_type = comp_type.lower().replace("-", "_")
        
        # Check for partial matches
        for known_type, config in self.component_pins.items():
            if known_type in comp_type or comp_type in known_type:
                return config
        
        return None
    
    def _get_component_connections(
        self,
        comp_ref: str,
        nets: List[Dict[str, Any]]
    ) -> List[str]:
        """Get all connections for a component."""
        connections = []
        
        for net in nets:
            net_connections = net.get("connections", [])
            for conn in net_connections:
                if conn.get("component") == comp_ref:
                    connections.append(conn.get("pin", ""))
        
        return connections
    
    def _is_output_pin(self, comp_ref: str, pin_num: str) -> bool:
        """Determine if a pin is likely an output pin."""
        # Simple heuristic - in practice would need component database
        output_indicators = ["out", "output", "q", "y"]
        pin_str = str(pin_num).lower()
        
        return any(indicator in pin_str for indicator in output_indicators)
    
    def _calculate_trace_spacing(
        self,
        trace1: Dict[str, Any],
        trace2: Dict[str, Any]
    ) -> float:
        """Calculate spacing between two traces."""
        # Simplified calculation - in practice would need full geometry
        x1, y1 = trace1.get("x", 0.0), trace1.get("y", 0.0)
        x2, y2 = trace2.get("x", 0.0), trace2.get("y", 0.0)
        
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def _calculate_component_spacing(
        self,
        comp1: Dict[str, Any],
        comp2: Dict[str, Any]
    ) -> float:
        """Calculate spacing between two components."""
        x1, y1 = comp1.get("x", 0.0), comp1.get("y", 0.0)
        x2, y2 = comp2.get("x", 0.0), comp2.get("y", 0.0)
        
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def _generate_verification_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report."""
        # Categorize violations
        errors = [v for v in self.violations if v.severity in [Severity.CRITICAL, Severity.ERROR]]
        warnings = [v for v in self.violations if v.severity == Severity.WARNING]
        
        # Count by type
        erc_violations = [v for v in self.violations if v.violation_type.value.startswith("erc")]
        drc_violations = [v for v in self.violations if v.violation_type.value.startswith("drc")]
        connectivity_violations = [v for v in self.violations if v.violation_type == ViolationType.CONNECTIVITY_ERROR]
        
        # Determine overall status
        has_errors = len(errors) > 0
        is_ready_for_manufacturing = not has_errors
        
        return {
            "success": True,
            "ready_for_manufacturing": is_ready_for_manufacturing,
            "summary": {
                "total_violations": len(self.violations),
                "errors": len(errors),
                "warnings": len(warnings),
                "erc_violations": len(erc_violations),
                "drc_violations": len(drc_violations),
                "connectivity_violations": len(connectivity_violations)
            },
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "message": v.message,
                    "component": v.component,
                    "net": v.net,
                    "location": v.location,
                    "suggested_fix": v.suggested_fix,
                    "rule_name": v.rule_name
                }
                for v in self.violations
            ],
            "design_rules": {
                "min_trace_width": self.design_rules.min_trace_width,
                "min_via_size": self.design_rules.min_via_size,
                "min_trace_spacing": self.design_rules.min_trace_spacing,
                "min_component_spacing": self.design_rules.min_component_spacing
            }
        }
    
    def get_violations_by_severity(self, severity: Severity) -> List[DesignViolation]:
        """Get violations filtered by severity."""
        return [v for v in self.violations if v.severity == severity]
    
    def get_violations_by_type(self, violation_type: ViolationType) -> List[DesignViolation]:
        """Get violations filtered by type."""
        return [v for v in self.violations if v.violation_type == violation_type]
    
    def clear_violations(self) -> None:
        """Clear all violations."""
        self.violations = []