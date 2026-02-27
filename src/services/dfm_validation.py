"""
Design for Manufacturing (DFM) validation system.

Implements comprehensive DFM checking with manufacturability scoring,
constraint validation, and specific recommendations for resolution.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DFMViolationSeverity(Enum):
    """Severity levels for DFM violations."""
    CRITICAL = "critical"  # Will prevent manufacturing
    HIGH = "high"  # Likely to cause manufacturing issues
    MEDIUM = "medium"  # May cause issues or increase cost
    LOW = "low"  # Best practice recommendations


@dataclass
class ManufacturingConstraints:
    """Manufacturing constraints for DFM validation."""
    # Trace constraints
    min_trace_width: float = 0.1  # mm (standard: 0.1mm / 4mil)
    min_trace_spacing: float = 0.1  # mm
    min_power_trace_width: float = 0.3  # mm
    
    # Via constraints
    min_via_diameter: float = 0.2  # mm
    min_via_drill: float = 0.15  # mm
    min_annular_ring: float = 0.05  # mm
    
    # Component constraints
    min_component_spacing: float = 0.5  # mm
    min_pad_to_pad_spacing: float = 0.2  # mm
    min_smd_pad_size: float = 0.3  # mm
    
    # Board constraints
    min_board_thickness: float = 0.4  # mm
    max_board_thickness: float = 3.2  # mm
    min_edge_clearance: float = 0.5  # mm
    
    # Drill constraints
    min_drill_size: float = 0.15  # mm
    max_drill_size: float = 6.35  # mm
    max_aspect_ratio: float = 10.0  # drill depth / diameter
    
    # Copper constraints
    min_copper_weight: float = 0.5  # oz (0.5oz = 17.5μm)
    max_copper_weight: float = 4.0  # oz
    
    # Soldermask constraints
    min_soldermask_bridge: float = 0.1  # mm
    min_soldermask_expansion: float = 0.05  # mm
    
    # Silkscreen constraints
    min_silkscreen_width: float = 0.15  # mm
    min_silkscreen_text_height: float = 1.0  # mm


@dataclass
class DFMViolation:
    """Represents a DFM violation."""
    severity: DFMViolationSeverity
    category: str  # trace, via, component, drill, etc.
    message: str
    location: Optional[Tuple[float, float]] = None
    component: Optional[str] = None
    net: Optional[str] = None
    recommendation: Optional[str] = None
    cost_impact: Optional[str] = None  # low, medium, high
    manufacturability_impact: float = 0.0  # 0-1 score impact


class DFMValidator:
    """
    Design for Manufacturing validation system.
    
    Performs comprehensive DFM checks and provides manufacturability scoring.
    """
    
    def __init__(self, constraints: Optional[ManufacturingConstraints] = None):
        """
        Initialize DFM validator.
        
        Args:
            constraints: Manufacturing constraints (uses defaults if None)
        """
        self.constraints = constraints or ManufacturingConstraints()
        self.violations: List[DFMViolation] = []
        
        # Manufacturer-specific constraints
        self.manufacturer_profiles = {
            "standard": ManufacturingConstraints(),
            "jlcpcb": ManufacturingConstraints(
                min_trace_width=0.09,
                min_trace_spacing=0.09,
                min_drill_size=0.2
            ),
            "pcbway": ManufacturingConstraints(
                min_trace_width=0.1,
                min_trace_spacing=0.1,
                min_drill_size=0.15
            ),
            "oshpark": ManufacturingConstraints(
                min_trace_width=0.127,
                min_trace_spacing=0.127,
                min_drill_size=0.254
            )
        }
    
    def validate_design(
        self,
        pcb_data: Dict[str, Any],
        gerber_files: Optional[Dict[str, str]] = None,
        manufacturer: str = "standard"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive DFM validation.
        
        Args:
            pcb_data: PCB layout data
            gerber_files: Generated Gerber files (optional)
            manufacturer: Target manufacturer profile
            
        Returns:
            DFM validation results with score and violations
        """
        self.violations = []
        
        # Use manufacturer-specific constraints if available
        if manufacturer in self.manufacturer_profiles:
            self.constraints = self.manufacturer_profiles[manufacturer]
        
        try:
            # Validate trace constraints
            self._validate_traces(pcb_data)
            
            # Validate via constraints
            self._validate_vias(pcb_data)
            
            # Validate component placement
            self._validate_components(pcb_data)
            
            # Validate drill constraints
            self._validate_drills(pcb_data)
            
            # Validate board constraints
            self._validate_board(pcb_data)
            
            # Validate signal integrity
            self._validate_signal_integrity(pcb_data)
            
            # Calculate manufacturability score
            score = self._calculate_manufacturability_score()
            
            # Generate DFM report
            return self._generate_dfm_report(score, manufacturer)
            
        except Exception as e:
            logger.error(f"DFM validation failed: {str(e)}")
            return {
                "success": False,
                "error": f"DFM validation failed: {str(e)}",
                "score": 0.0
            }
    
    def _validate_traces(self, pcb_data: Dict[str, Any]) -> None:
        """Validate trace width and spacing constraints."""
        traces = pcb_data.get("traces", [])
        
        for trace in traces:
            width = trace.get("width", 0.0)
            net_name = trace.get("net", "")
            location = (trace.get("x", 0.0), trace.get("y", 0.0))
            
            # Check minimum trace width
            if width < self.constraints.min_trace_width:
                self.violations.append(DFMViolation(
                    severity=DFMViolationSeverity.CRITICAL,
                    category="trace",
                    message=f"Trace width {width:.3f}mm below minimum {self.constraints.min_trace_width}mm",
                    location=location,
                    net=net_name,
                    recommendation=f"Increase trace width to at least {self.constraints.min_trace_width}mm. "
                                 f"Most manufacturers cannot reliably produce traces below this width.",
                    cost_impact="high",
                    manufacturability_impact=0.15
                ))
            
            # Check power trace width
            if self._is_power_net(net_name) and width < self.constraints.min_power_trace_width:
                self.violations.append(DFMViolation(
                    severity=DFMViolationSeverity.HIGH,
                    category="trace",
                    message=f"Power trace {net_name} width {width:.3f}mm below recommended {self.constraints.min_power_trace_width}mm",
                    location=location,
                    net=net_name,
                    recommendation=f"Increase power trace width to at least {self.constraints.min_power_trace_width}mm "
                                 f"to reduce voltage drop and improve current handling.",
                    cost_impact="low",
                    manufacturability_impact=0.05
                ))
        
        # Check trace spacing
        for i, trace1 in enumerate(traces):
            for trace2 in traces[i+1:]:
                spacing = self._calculate_spacing(trace1, trace2)
                if spacing < self.constraints.min_trace_spacing:
                    self.violations.append(DFMViolation(
                        severity=DFMViolationSeverity.CRITICAL,
                        category="trace",
                        message=f"Trace spacing {spacing:.3f}mm below minimum {self.constraints.min_trace_spacing}mm",
                        recommendation=f"Increase spacing between traces to at least {self.constraints.min_trace_spacing}mm. "
                                     f"Insufficient spacing can cause shorts during manufacturing.",
                        cost_impact="high",
                        manufacturability_impact=0.2
                    ))
    
    def _validate_vias(self, pcb_data: Dict[str, Any]) -> None:
        """Validate via size and annular ring constraints."""
        vias = pcb_data.get("vias", [])
        board_thickness = pcb_data.get("thickness", 1.6)
        
        for via in vias:
            diameter = via.get("size", 0.0)
            drill = via.get("drill", diameter * 0.6)
            location = (via.get("x", 0.0), via.get("y", 0.0))
            
            # Check minimum via diameter
            if diameter < self.constraints.min_via_diameter:
                self.violations.append(DFMViolation(
                    severity=DFMViolationSeverity.CRITICAL,
                    category="via",
                    message=f"Via diameter {diameter:.3f}mm below minimum {self.constraints.min_via_diameter}mm",
                    location=location,
                    recommendation=f"Increase via diameter to at least {self.constraints.min_via_diameter}mm.",
                    cost_impact="high",
                    manufacturability_impact=0.15
                ))
            
            # Check minimum drill size
            if drill < self.constraints.min_via_drill:
                self.violations.append(DFMViolation(
                    severity=DFMViolationSeverity.CRITICAL,
                    category="via",
                    message=f"Via drill {drill:.3f}mm below minimum {self.constraints.min_via_drill}mm",
                    location=location,
                    recommendation=f"Increase via drill size to at least {self.constraints.min_via_drill}mm.",
                    cost_impact="high",
                    manufacturability_impact=0.15
                ))
            
            # Check annular ring
            annular_ring = (diameter - drill) / 2
            if annular_ring < self.constraints.min_annular_ring:
                self.violations.append(DFMViolation(
                    severity=DFMViolationSeverity.HIGH,
                    category="via",
                    message=f"Via annular ring {annular_ring:.3f}mm below minimum {self.constraints.min_annular_ring}mm",
                    location=location,
                    recommendation=f"Increase via pad size or decrease drill size to achieve minimum annular ring of {self.constraints.min_annular_ring}mm. "
                                 f"Insufficient annular ring can cause via breakout during drilling.",
                    cost_impact="medium",
                    manufacturability_impact=0.1
                ))
            
            # Check aspect ratio
            aspect_ratio = board_thickness / drill
            if aspect_ratio > self.constraints.max_aspect_ratio:
                self.violations.append(DFMViolation(
                    severity=DFMViolationSeverity.HIGH,
                    category="via",
                    message=f"Via aspect ratio {aspect_ratio:.1f} exceeds maximum {self.constraints.max_aspect_ratio}",
                    location=location,
                    recommendation=f"Increase drill size or reduce board thickness. High aspect ratio vias are difficult to plate reliably.",
                    cost_impact="high",
                    manufacturability_impact=0.12
                ))
    
    def _validate_components(self, pcb_data: Dict[str, Any]) -> None:
        """Validate component placement and spacing."""
        components = pcb_data.get("components", [])
        board_width = pcb_data.get("width", 100.0)
        board_height = pcb_data.get("height", 80.0)
        
        for component in components:
            x = component.get("x", 0.0)
            y = component.get("y", 0.0)
            reference = component.get("reference", "")
            package = component.get("package", "")
            
            # Check edge clearance
            edge_distance = min(x, y, board_width - x, board_height - y)
            if edge_distance < self.constraints.min_edge_clearance:
                self.violations.append(DFMViolation(
                    severity=DFMViolationSeverity.MEDIUM,
                    category="component",
                    message=f"Component {reference} too close to board edge ({edge_distance:.3f}mm)",
                    location=(x, y),
                    component=reference,
                    recommendation=f"Move component at least {self.constraints.min_edge_clearance}mm from board edge. "
                                 f"Components near edges are at risk during depaneling.",
                    cost_impact="low",
                    manufacturability_impact=0.05
                ))
            
            # Check SMD pad size
            if "smd" in package.lower() or any(pkg in package.lower() for pkg in ["0402", "0603", "0805"]):
                pad_size = component.get("pad_size", 0.0)
                if pad_size > 0 and pad_size < self.constraints.min_smd_pad_size:
                    self.violations.append(DFMViolation(
                        severity=DFMViolationSeverity.HIGH,
                        category="component",
                        message=f"SMD pad size {pad_size:.3f}mm below minimum {self.constraints.min_smd_pad_size}mm",
                        location=(x, y),
                        component=reference,
                        recommendation=f"Increase pad size to at least {self.constraints.min_smd_pad_size}mm for reliable assembly.",
                        cost_impact="medium",
                        manufacturability_impact=0.08
                    ))
        
        # Check component spacing
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                spacing = self._calculate_component_spacing(comp1, comp2)
                if spacing < self.constraints.min_component_spacing:
                    self.violations.append(DFMViolation(
                        severity=DFMViolationSeverity.MEDIUM,
                        category="component",
                        message=f"Components {comp1.get('reference')} and {comp2.get('reference')} spacing {spacing:.3f}mm below recommended {self.constraints.min_component_spacing}mm",
                        recommendation=f"Increase spacing to at least {self.constraints.min_component_spacing}mm for easier assembly and rework.",
                        cost_impact="low",
                        manufacturability_impact=0.03
                    ))
    
    def _validate_drills(self, pcb_data: Dict[str, Any]) -> None:
        """Validate drill size constraints."""
        drill_holes = pcb_data.get("drill_holes", [])
        
        for hole in drill_holes:
            diameter = hole.get("diameter", 0.0)
            location = (hole.get("x", 0.0), hole.get("y", 0.0))
            
            # Check minimum drill size
            if diameter < self.constraints.min_drill_size:
                self.violations.append(DFMViolation(
                    severity=DFMViolationSeverity.CRITICAL,
                    category="drill",
                    message=f"Drill size {diameter:.3f}mm below minimum {self.constraints.min_drill_size}mm",
                    location=location,
                    recommendation=f"Increase drill size to at least {self.constraints.min_drill_size}mm. "
                                 f"Smaller drills are prone to breakage and may not be available.",
                    cost_impact="high",
                    manufacturability_impact=0.15
                ))
            
            # Check maximum drill size
            if diameter > self.constraints.max_drill_size:
                self.violations.append(DFMViolation(
                    severity=DFMViolationSeverity.MEDIUM,
                    category="drill",
                    message=f"Drill size {diameter:.3f}mm exceeds maximum {self.constraints.max_drill_size}mm",
                    location=location,
                    recommendation=f"Reduce drill size below {self.constraints.max_drill_size}mm or use mechanical cutting.",
                    cost_impact="medium",
                    manufacturability_impact=0.05
                ))
    
    def _validate_board(self, pcb_data: Dict[str, Any]) -> None:
        """Validate board-level constraints."""
        thickness = pcb_data.get("thickness", 1.6)
        width = pcb_data.get("width", 100.0)
        height = pcb_data.get("height", 80.0)
        layers = pcb_data.get("layers", 2)
        
        # Check board thickness
        if thickness < self.constraints.min_board_thickness:
            self.violations.append(DFMViolation(
                severity=DFMViolationSeverity.HIGH,
                category="board",
                message=f"Board thickness {thickness:.2f}mm below minimum {self.constraints.min_board_thickness}mm",
                recommendation=f"Increase board thickness to at least {self.constraints.min_board_thickness}mm. "
                             f"Thin boards are fragile and difficult to handle.",
                cost_impact="medium",
                manufacturability_impact=0.1
            ))
        
        if thickness > self.constraints.max_board_thickness:
            self.violations.append(DFMViolation(
                severity=DFMViolationSeverity.MEDIUM,
                category="board",
                message=f"Board thickness {thickness:.2f}mm exceeds maximum {self.constraints.max_board_thickness}mm",
                recommendation=f"Reduce board thickness below {self.constraints.max_board_thickness}mm or use special manufacturing process.",
                cost_impact="high",
                manufacturability_impact=0.08
            ))
        
        # Check board size
        if width < 10.0 or height < 10.0:
            self.violations.append(DFMViolation(
                severity=DFMViolationSeverity.MEDIUM,
                category="board",
                message=f"Board size {width}x{height}mm is very small",
                recommendation="Small boards may require panelization for manufacturing. Consider adding tooling holes.",
                cost_impact="medium",
                manufacturability_impact=0.05
            ))
    
    def _validate_signal_integrity(self, pcb_data: Dict[str, Any]) -> None:
        """Validate signal integrity requirements."""
        traces = pcb_data.get("traces", [])
        
        # Check for high-speed signals
        for trace in traces:
            net_name = trace.get("net", "")
            width = trace.get("width", 0.0)
            length = trace.get("length", 0.0)
            
            # Detect potential high-speed signals
            if any(hs in net_name.lower() for hs in ["clk", "clock", "data", "usb", "eth"]):
                # Check trace width consistency for impedance control
                if width < 0.15:
                    self.violations.append(DFMViolation(
                        severity=DFMViolationSeverity.MEDIUM,
                        category="signal_integrity",
                        message=f"High-speed signal {net_name} may have impedance control issues",
                        net=net_name,
                        recommendation="Consider using controlled impedance traces for high-speed signals. "
                                     "Consult with manufacturer for stackup and trace width calculations.",
                        cost_impact="medium",
                        manufacturability_impact=0.05
                    ))
                
                # Check trace length for high-speed signals
                if length > 100.0:  # mm
                    self.violations.append(DFMViolation(
                        severity=DFMViolationSeverity.LOW,
                        category="signal_integrity",
                        message=f"High-speed signal {net_name} trace length {length:.1f}mm may cause signal integrity issues",
                        net=net_name,
                        recommendation="Consider shortening trace length or adding termination resistors.",
                        cost_impact="low",
                        manufacturability_impact=0.02
                    ))
    
    def _calculate_manufacturability_score(self) -> float:
        """
        Calculate overall manufacturability score (0-100).
        
        Higher score = more manufacturable design.
        """
        # Start with perfect score
        score = 100.0
        
        # Deduct points based on violations
        for violation in self.violations:
            # Apply impact based on severity
            if violation.severity == DFMViolationSeverity.CRITICAL:
                score -= violation.manufacturability_impact * 100
            elif violation.severity == DFMViolationSeverity.HIGH:
                score -= violation.manufacturability_impact * 70
            elif violation.severity == DFMViolationSeverity.MEDIUM:
                score -= violation.manufacturability_impact * 40
            elif violation.severity == DFMViolationSeverity.LOW:
                score -= violation.manufacturability_impact * 20
        
        # Ensure score is in valid range
        return max(0.0, min(100.0, score))
    
    def _generate_dfm_report(self, score: float, manufacturer: str) -> Dict[str, Any]:
        """Generate comprehensive DFM report."""
        # Categorize violations
        critical = [v for v in self.violations if v.severity == DFMViolationSeverity.CRITICAL]
        high = [v for v in self.violations if v.severity == DFMViolationSeverity.HIGH]
        medium = [v for v in self.violations if v.severity == DFMViolationSeverity.MEDIUM]
        low = [v for v in self.violations if v.severity == DFMViolationSeverity.LOW]
        
        # Count by category
        by_category = {}
        for violation in self.violations:
            category = violation.category
            if category not in by_category:
                by_category[category] = 0
            by_category[category] += 1
        
        # Determine manufacturing readiness
        is_manufacturable = len(critical) == 0
        confidence_level = self._get_confidence_level(score)
        
        return {
            "success": True,
            "manufacturable": is_manufacturable,
            "score": round(score, 2),
            "confidence_level": confidence_level,
            "manufacturer": manufacturer,
            "summary": {
                "total_violations": len(self.violations),
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(low),
                "by_category": by_category
            },
            "violations": [
                {
                    "severity": v.severity.value,
                    "category": v.category,
                    "message": v.message,
                    "location": v.location,
                    "component": v.component,
                    "net": v.net,
                    "recommendation": v.recommendation,
                    "cost_impact": v.cost_impact
                }
                for v in self.violations
            ],
            "constraints": {
                "min_trace_width": self.constraints.min_trace_width,
                "min_via_diameter": self.constraints.min_via_diameter,
                "min_drill_size": self.constraints.min_drill_size,
                "min_component_spacing": self.constraints.min_component_spacing
            }
        }
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level description based on score."""
        if score >= 95:
            return "excellent"
        elif score >= 85:
            return "good"
        elif score >= 70:
            return "fair"
        elif score >= 50:
            return "poor"
        else:
            return "unmanufacturable"
    
    def _is_power_net(self, net_name: str) -> bool:
        """Check if net is a power net."""
        power_indicators = ["vcc", "vdd", "power", "+5v", "+3v3", "+12v", "vbat"]
        return any(indicator in net_name.lower() for indicator in power_indicators)
    
    def _calculate_spacing(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> float:
        """Calculate spacing between two objects."""
        x1, y1 = obj1.get("x", 0.0), obj1.get("y", 0.0)
        x2, y2 = obj2.get("x", 0.0), obj2.get("y", 0.0)
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def _calculate_component_spacing(self, comp1: Dict[str, Any], comp2: Dict[str, Any]) -> float:
        """Calculate spacing between two components."""
        return self._calculate_spacing(comp1, comp2)
    
    def get_violations_by_severity(self, severity: DFMViolationSeverity) -> List[DFMViolation]:
        """Get violations filtered by severity."""
        return [v for v in self.violations if v.severity == severity]
    
    def get_violations_by_category(self, category: str) -> List[DFMViolation]:
        """Get violations filtered by category."""
        return [v for v in self.violations if v.category == category]
    
    def clear_violations(self) -> None:
        """Clear all violations."""
        self.violations = []