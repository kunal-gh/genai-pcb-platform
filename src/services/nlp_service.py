"""
Natural Language Processing Service.

Parses natural language prompts into structured JSON requirements
for PCB design generation.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BoardSpecification:
    """Board physical specifications."""
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    thickness_mm: float = 1.6  # Standard PCB thickness
    layers: int = 1  # Default to single layer


@dataclass
class PowerSpecification:
    """Power supply specifications."""
    type: Optional[str] = None  # battery, usb, ac_adapter, etc.
    voltage: Optional[float] = None
    current_max_a: Optional[float] = None
    battery_type: Optional[str] = None  # 9V, AA, lithium, etc.


@dataclass
class ComponentRequirement:
    """Individual component requirement."""
    type: str  # LED, RESISTOR, CAPACITOR, IC, etc.
    value: Optional[str] = None  # resistance, capacitance, part number
    package: Optional[str] = None  # 0805, 5mm, DIP-8, etc.
    quantity: int = 1
    reference: Optional[str] = None  # R1, C1, U1, etc.
    description: Optional[str] = None


@dataclass
class DesignConstraints:
    """Design constraints and preferences."""
    max_power_w: Optional[float] = None
    priority: str = "balanced"  # compact, cost, performance, balanced
    operating_temp_c: Optional[tuple] = None  # (min, max)
    special_requirements: List[str] = None
    
    def __post_init__(self):
        if self.special_requirements is None:
            self.special_requirements = []


@dataclass
class StructuredRequirements:
    """Complete structured requirements from natural language prompt."""
    board: BoardSpecification
    power: PowerSpecification
    components: List[ComponentRequirement]
    constraints: DesignConstraints
    connections: List[tuple] = None  # List of (from, to) connections
    original_prompt: str = ""
    confidence_score: float = 1.0
    ambiguities: List[str] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []
        if self.ambiguities is None:
            self.ambiguities = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "board": asdict(self.board),
            "power": asdict(self.power),
            "components": [asdict(c) for c in self.components],
            "constraints": asdict(self.constraints),
            "connections": self.connections,
            "original_prompt": self.original_prompt,
            "confidence_score": self.confidence_score,
            "ambiguities": self.ambiguities
        }


class NLPService:
    """
    Natural Language Processing service for PCB design prompts.
    
    Parses natural language descriptions into structured requirements
    using pattern matching and keyword extraction.
    """
    
    # Component type patterns
    COMPONENT_PATTERNS = {
        "LED": r"\b(led|light[\s-]?emitting[\s-]?diode)\b",
        "RESISTOR": r"\b(resistor|resistance|ohm)\b",
        "CAPACITOR": r"\b(capacitor|capacitance|farad)\b",
        "INDUCTOR": r"\b(inductor|inductance|henry)\b",
        "DIODE": r"\b(diode|rectifier)\b",
        "TRANSISTOR": r"\b(transistor|bjt|mosfet|fet)\b",
        "IC": r"\b(ic|chip|microcontroller|mcu|processor)\b",
        "CONNECTOR": r"\b(connector|header|socket|jack|plug)\b",
        "SWITCH": r"\b(switch|button|toggle)\b",
        "BATTERY": r"\b(battery|cell|power[\s-]?source)\b",
        "CRYSTAL": r"\b(crystal|oscillator|resonator)\b",
        "FUSE": r"\b(fuse|protection)\b",
    }
    
    # Value patterns
    VALUE_PATTERNS = {
        "resistance": r"(\d+(?:\.\d+)?)\s*(?:k|K|M)?(?:ohm|Ω|R)",
        "capacitance": r"(\d+(?:\.\d+)?)\s*(?:p|n|u|µ|m)?(?:F|farad)",
        "voltage": r"(\d+(?:\.\d+)?)\s*(?:V|volt)",
        "current": r"(\d+(?:\.\d+)?)\s*(?:m)?(?:A|amp)",
        "frequency": r"(\d+(?:\.\d+)?)\s*(?:k|K|M|G)?(?:Hz|hertz)",
    }
    
    # Board dimension patterns
    DIMENSION_PATTERNS = {
        "width_height": r"(\d+(?:\.\d+)?)\s*(?:x|X|×)\s*(\d+(?:\.\d+)?)\s*mm",
        "single_dimension": r"(\d+(?:\.\d+)?)\s*mm",
    }
    
    # Package patterns
    PACKAGE_PATTERNS = {
        "smd": r"\b(0402|0603|0805|1206|1210|2512)\b",
        "through_hole": r"\b(\d+mm|DIP-?\d+|TO-?\d+)\b",
    }
    
    def __init__(self):
        """Initialize the NLP service."""
        logger.info("Initializing NLP Service")
    
    def parse_prompt(self, prompt: str) -> StructuredRequirements:
        """
        Parse natural language prompt into structured requirements.
        
        Args:
            prompt: Natural language description of PCB design
            
        Returns:
            StructuredRequirements: Structured design requirements
            
        Example:
            >>> nlp = NLPService()
            >>> req = nlp.parse_prompt("Design a 40x20mm PCB with 9V battery, LED, and 220-ohm resistor")
            >>> req.board.width_mm
            40.0
        """
        logger.info(f"Parsing prompt: {prompt[:100]}...")
        
        prompt_lower = prompt.lower()
        
        # Parse board specifications
        board = self._parse_board_specs(prompt_lower)
        
        # Parse power specifications
        power = self._parse_power_specs(prompt_lower)
        
        # Parse components
        components = self._parse_components(prompt_lower)
        
        # Parse constraints
        constraints = self._parse_constraints(prompt_lower)
        
        # Detect ambiguities
        ambiguities = self._detect_ambiguities(prompt_lower, components)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(board, power, components, ambiguities)
        
        requirements = StructuredRequirements(
            board=board,
            power=power,
            components=components,
            constraints=constraints,
            original_prompt=prompt,
            confidence_score=confidence,
            ambiguities=ambiguities
        )
        
        logger.info(f"Parsed {len(components)} components with confidence {confidence:.2f}")
        
        return requirements
    
    def _parse_board_specs(self, prompt: str) -> BoardSpecification:
        """Extract board physical specifications."""
        board = BoardSpecification()
        
        # Try to find width x height pattern
        match = re.search(self.DIMENSION_PATTERNS["width_height"], prompt)
        if match:
            board.width_mm = float(match.group(1))
            board.height_mm = float(match.group(2))
            logger.debug(f"Found board dimensions: {board.width_mm}x{board.height_mm}mm")
        
        # Check for layer count
        layer_match = re.search(r"(\d+)[\s-]?layer", prompt)
        if layer_match:
            board.layers = int(layer_match.group(1))
            logger.debug(f"Found layer count: {board.layers}")
        
        # Check for thickness
        thickness_match = re.search(r"(\d+(?:\.\d+)?)\s*mm\s+thick", prompt)
        if thickness_match:
            board.thickness_mm = float(thickness_match.group(1))
        
        return board
    
    def _parse_power_specs(self, prompt: str) -> PowerSpecification:
        """Extract power supply specifications."""
        power = PowerSpecification()
        
        # Check for battery
        if re.search(r"\bbattery\b", prompt):
            power.type = "battery"
            
            # Try to identify battery type
            if "9v" in prompt or "9 v" in prompt:
                power.battery_type = "9V"
                power.voltage = 9.0
            elif "aa" in prompt:
                power.battery_type = "AA"
                power.voltage = 1.5
            elif "aaa" in prompt:
                power.battery_type = "AAA"
                power.voltage = 1.5
            elif "lithium" in prompt or "li-ion" in prompt:
                power.battery_type = "lithium"
                power.voltage = 3.7
        
        # Check for USB power
        if re.search(r"\busb\b", prompt):
            power.type = "usb"
            power.voltage = 5.0
        
        # Extract voltage if not already set
        if power.voltage is None:
            voltage_match = re.search(self.VALUE_PATTERNS["voltage"], prompt)
            if voltage_match:
                power.voltage = float(voltage_match.group(1))
        
        # Extract current
        current_match = re.search(self.VALUE_PATTERNS["current"], prompt)
        if current_match:
            current_val = float(current_match.group(1))
            # Check if it's milliamps
            if "ma" in prompt[current_match.start():current_match.end()].lower():
                current_val /= 1000
            power.current_max_a = current_val
        
        return power
    
    def _parse_components(self, prompt: str) -> List[ComponentRequirement]:
        """Extract component requirements."""
        components = []
        
        for comp_type, pattern in self.COMPONENT_PATTERNS.items():
            matches = list(re.finditer(pattern, prompt, re.IGNORECASE))
            
            for match in matches:
                component = ComponentRequirement(type=comp_type)
                
                # Extract context around the match (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(prompt), match.end() + 50)
                context = prompt[start:end]
                
                # Try to extract value
                if comp_type == "RESISTOR":
                    value_match = re.search(self.VALUE_PATTERNS["resistance"], context)
                    if value_match:
                        component.value = value_match.group(0)
                
                elif comp_type == "CAPACITOR":
                    value_match = re.search(self.VALUE_PATTERNS["capacitance"], context)
                    if value_match:
                        component.value = value_match.group(0)
                
                # Try to extract package
                for pkg_type, pkg_pattern in self.PACKAGE_PATTERNS.items():
                    pkg_match = re.search(pkg_pattern, context)
                    if pkg_match:
                        component.package = pkg_match.group(1)
                        break
                
                # Add description from context
                component.description = match.group(0)
                
                components.append(component)
                logger.debug(f"Found component: {comp_type} - {component.value}")
        
        return components
    
    def _parse_constraints(self, prompt: str) -> DesignConstraints:
        """Extract design constraints."""
        constraints = DesignConstraints()
        
        # Check for priority keywords
        if any(word in prompt for word in ["compact", "small", "tiny", "miniature"]):
            constraints.priority = "compact"
        elif any(word in prompt for word in ["cheap", "low cost", "budget", "economical"]):
            constraints.priority = "cost"
        elif any(word in prompt for word in ["fast", "high performance", "optimized"]):
            constraints.priority = "performance"
        
        # Extract max power
        power_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:W|watt)", prompt)
        if power_match:
            constraints.max_power_w = float(power_match.group(1))
        
        # Check for special requirements
        if "waterproof" in prompt or "water resistant" in prompt:
            constraints.special_requirements.append("waterproof")
        if "high temperature" in prompt or "hot environment" in prompt:
            constraints.special_requirements.append("high_temperature")
        if "low noise" in prompt or "quiet" in prompt:
            constraints.special_requirements.append("low_noise")
        
        return constraints
    
    def _detect_ambiguities(self, prompt: str, components: List[ComponentRequirement]) -> List[str]:
        """Detect ambiguous or missing information."""
        ambiguities = []
        
        # Check if board dimensions are missing
        if not re.search(self.DIMENSION_PATTERNS["width_height"], prompt):
            ambiguities.append("Board dimensions not specified")
        
        # Check for components without values
        for comp in components:
            if comp.type in ["RESISTOR", "CAPACITOR", "INDUCTOR"] and not comp.value:
                ambiguities.append(f"{comp.type} value not specified")
        
        # Check for power supply
        if not any(word in prompt for word in ["battery", "power", "supply", "usb", "voltage"]):
            ambiguities.append("Power supply not specified")
        
        return ambiguities
    
    def _calculate_confidence(
        self,
        board: BoardSpecification,
        power: PowerSpecification,
        components: List[ComponentRequirement],
        ambiguities: List[str]
    ) -> float:
        """Calculate confidence score for parsed requirements."""
        score = 1.0
        
        # Reduce score for missing board dimensions
        if board.width_mm is None or board.height_mm is None:
            score -= 0.2
        
        # Reduce score for missing power specs
        if power.type is None:
            score -= 0.2
        
        # Reduce score if no components found
        if len(components) == 0:
            score -= 0.3
        
        # Reduce score for each ambiguity
        score -= len(ambiguities) * 0.1
        
        return max(0.0, min(1.0, score))
    
    def validate_prompt(self, prompt: str) -> tuple[bool, Optional[str]]:
        """
        Validate prompt meets basic requirements.
        
        Args:
            prompt: Natural language prompt to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check length
        if len(prompt) < 10:
            return False, "Prompt is too short (minimum 10 characters)"
        
        if len(prompt) > 10000:
            return False, "Prompt is too long (maximum 10,000 characters)"
        
        # Check for at least one component keyword
        has_component = any(
            re.search(pattern, prompt, re.IGNORECASE)
            for pattern in self.COMPONENT_PATTERNS.values()
        )
        
        if not has_component:
            return False, "No recognizable components found in prompt"
        
        return True, None