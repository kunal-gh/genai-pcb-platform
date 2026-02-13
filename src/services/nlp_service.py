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
class ClarificationRequest:
    """Request for clarification on ambiguous input."""
    field: str  # What needs clarification
    message: str  # User-friendly question
    suggestions: List[str] = None  # Suggested values
    severity: str = "warning"  # warning, error, info
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class ValidationError:
    """Validation error with descriptive message."""
    field: str
    message: str
    suggestion: Optional[str] = None
    error_code: Optional[str] = None


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
    clarification_requests: List[ClarificationRequest] = None
    validation_errors: List[ValidationError] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []
        if self.ambiguities is None:
            self.ambiguities = []
        if self.clarification_requests is None:
            self.clarification_requests = []
        if self.validation_errors is None:
            self.validation_errors = []
    
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
            "ambiguities": self.ambiguities,
            "clarification_requests": [asdict(c) for c in self.clarification_requests],
            "validation_errors": [asdict(e) for e in self.validation_errors]
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
        "resistance": r"(\d+(?:\.\d+)?)\s*-?\s*([kKMm])?\s*-?\s*(?:ohm|Ω|R)\b",
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
        
        # Generate clarification requests
        clarifications = self._generate_clarification_requests(board, power, components, prompt_lower)
        
        # Validate requirements
        validation_errors = self._validate_requirements(board, power, components, prompt)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(board, power, components, ambiguities)
        
        requirements = StructuredRequirements(
            board=board,
            power=power,
            components=components,
            constraints=constraints,
            original_prompt=prompt,
            confidence_score=confidence,
            ambiguities=ambiguities,
            clarification_requests=clarifications,
            validation_errors=validation_errors
        )
        
        logger.info(f"Parsed {len(components)} components with confidence {confidence:.2f}, "
                   f"{len(clarifications)} clarifications, {len(validation_errors)} errors")
        
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
    
    def _generate_clarification_requests(
        self,
        board: BoardSpecification,
        power: PowerSpecification,
        components: List[ComponentRequirement],
        prompt: str
    ) -> List[ClarificationRequest]:
        """Generate user-friendly clarification requests for ambiguous inputs."""
        clarifications = []
        
        # Board dimensions clarification
        if board.width_mm is None or board.height_mm is None:
            clarifications.append(ClarificationRequest(
                field="board_dimensions",
                message="What are the desired board dimensions?",
                suggestions=["40x20mm", "50x50mm", "100x80mm", "Custom size"],
                severity="warning"
            ))
        
        # Power supply clarification
        if power.type is None:
            clarifications.append(ClarificationRequest(
                field="power_supply",
                message="What power source will the board use?",
                suggestions=["9V battery", "USB (5V)", "AA batteries", "AC adapter", "Other"],
                severity="warning"
            ))
        
        # Component value clarifications
        for i, comp in enumerate(components):
            if comp.type == "RESISTOR" and not comp.value:
                clarifications.append(ClarificationRequest(
                    field=f"component_{i}_value",
                    message=f"What resistance value for the {comp.type.lower()}?",
                    suggestions=["220Ω", "1kΩ", "10kΩ", "100kΩ"],
                    severity="warning"
                ))
            elif comp.type == "CAPACITOR" and not comp.value:
                clarifications.append(ClarificationRequest(
                    field=f"component_{i}_value",
                    message=f"What capacitance value for the {comp.type.lower()}?",
                    suggestions=["100nF", "1µF", "10µF", "100µF"],
                    severity="warning"
                ))
            elif comp.type == "LED" and not comp.package:
                clarifications.append(ClarificationRequest(
                    field=f"component_{i}_package",
                    message=f"What package type for the LED?",
                    suggestions=["5mm through-hole", "0805 SMD", "0603 SMD"],
                    severity="info"
                ))
        
        # Layer count clarification for complex designs
        if len(components) > 20 and board.layers == 1:
            clarifications.append(ClarificationRequest(
                field="board_layers",
                message="This design has many components. Would you like a multi-layer board?",
                suggestions=["2-layer", "4-layer", "Keep single-layer"],
                severity="info"
            ))
        
        return clarifications
    
    def _validate_requirements(
        self,
        board: BoardSpecification,
        power: PowerSpecification,
        components: List[ComponentRequirement],
        prompt: str
    ) -> List[ValidationError]:
        """Validate requirements and generate descriptive error messages."""
        errors = []
        
        # Validate prompt length (10-1000 words)
        word_count = len(prompt.split())
        if word_count < 10:
            errors.append(ValidationError(
                field="prompt",
                message=f"Prompt is too short ({word_count} words). Please provide more details.",
                suggestion="Include component types, connections, and board specifications. "
                          "Example: 'Design a 40x20mm PCB with a 9V battery, LED, and 220-ohm resistor in series.'",
                error_code="PROMPT_TOO_SHORT"
            ))
        elif word_count > 1000:
            errors.append(ValidationError(
                field="prompt",
                message=f"Prompt is too long ({word_count} words). Maximum is 1000 words.",
                suggestion="Break down your design into smaller, focused descriptions. "
                          "Focus on essential components and connections first.",
                error_code="PROMPT_TOO_LONG"
            ))
        
        # Validate board dimensions are reasonable
        if board.width_mm is not None and board.height_mm is not None:
            if board.width_mm < 10 or board.height_mm < 10:
                errors.append(ValidationError(
                    field="board_dimensions",
                    message=f"Board dimensions ({board.width_mm}x{board.height_mm}mm) are too small.",
                    suggestion="Minimum board size is 10x10mm. Consider increasing dimensions to fit components.",
                    error_code="BOARD_TOO_SMALL"
                ))
            elif board.width_mm > 500 or board.height_mm > 500:
                errors.append(ValidationError(
                    field="board_dimensions",
                    message=f"Board dimensions ({board.width_mm}x{board.height_mm}mm) are unusually large.",
                    suggestion="Maximum recommended board size is 500x500mm. Verify dimensions are correct.",
                    error_code="BOARD_TOO_LARGE"
                ))
        
        # Validate power specifications
        if power.voltage is not None:
            if power.voltage < 0:
                errors.append(ValidationError(
                    field="power_voltage",
                    message=f"Voltage cannot be negative ({power.voltage}V).",
                    suggestion="Specify a positive voltage value (e.g., 5V, 9V, 12V).",
                    error_code="INVALID_VOLTAGE"
                ))
            elif power.voltage > 48:
                errors.append(ValidationError(
                    field="power_voltage",
                    message=f"Voltage ({power.voltage}V) exceeds typical low-voltage range.",
                    suggestion="Voltages above 48V require special safety considerations. "
                              "Verify this is correct for your application.",
                    error_code="HIGH_VOLTAGE_WARNING"
                ))
        
        # Also check for negative voltage in the prompt text
        negative_voltage_match = re.search(r"-\s*(\d+(?:\.\d+)?)\s*(?:V|volt)", prompt, re.IGNORECASE)
        if negative_voltage_match:
            errors.append(ValidationError(
                field="power_voltage",
                message=f"Voltage cannot be negative (-{negative_voltage_match.group(1)}V detected).",
                suggestion="Specify a positive voltage value (e.g., 5V, 9V, 12V).",
                error_code="INVALID_VOLTAGE"
            ))
        
        # Check for high voltage in prompt if not already detected
        if power.voltage is None or power.voltage <= 48:
            high_voltage_match = re.search(r"(\d+)\s*(?:V|volt)", prompt, re.IGNORECASE)
            if high_voltage_match:
                voltage_val = float(high_voltage_match.group(1))
                if voltage_val > 48:
                    errors.append(ValidationError(
                        field="power_voltage",
                        message=f"Voltage ({voltage_val}V) exceeds typical low-voltage range.",
                        suggestion="Voltages above 48V require special safety considerations. "
                                  "Verify this is correct for your application.",
                        error_code="HIGH_VOLTAGE_WARNING"
                    ))
        
        # Validate component count
        if len(components) == 0:
            errors.append(ValidationError(
                field="components",
                message="No components detected in the prompt.",
                suggestion="Specify at least one component (e.g., LED, resistor, capacitor, IC). "
                          "Example: 'Add a 220-ohm resistor and an LED.'",
                error_code="NO_COMPONENTS"
            ))
        elif len(components) > 100:
            errors.append(ValidationError(
                field="components",
                message=f"Design has {len(components)} components, which may be too complex for automated generation.",
                suggestion="Consider breaking the design into smaller modules or subsystems.",
                error_code="TOO_MANY_COMPONENTS"
            ))
        
        # Validate component values are reasonable
        for i, comp in enumerate(components):
            if comp.type == "RESISTOR" and comp.value:
                # Extract numeric value from the component value string
                value_str = comp.value
                value_match = re.search(r"(\d+(?:\.\d+)?)", value_str)
                if value_match:
                    value = float(value_match.group(1))
                    
                    # Apply multipliers - check for multiplier letter
                    multiplier_match = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*([kKMm])", value_str)
                    if multiplier_match:
                        multiplier = multiplier_match.group(2)
                        if multiplier in ['k', 'K']:
                            value *= 1000
                        elif multiplier == 'M':
                            value *= 1000000
                        elif multiplier == 'm' and 'mohm' not in value_str.lower():
                            # Only treat as milliohm if explicitly "mohm", otherwise might be typo for M
                            # For "100m-ohm" we'll assume it's meant to be 100M (megaohm)
                            if re.search(r'\d+\s*m\s*-?\s*ohm', value_str, re.IGNORECASE):
                                # Check context - if value is large (>10), likely meant Megaohm
                                if value >= 10:
                                    value *= 1000000
                                else:
                                    value /= 1000
                    
                    # Check for unreasonable values
                    if value < 1:
                        errors.append(ValidationError(
                            field=f"component_{i}_value",
                            message=f"Resistor value ({comp.value}) is unusually low.",
                            suggestion="Typical resistor values range from 1Ω to 10MΩ. Verify the value is correct.",
                            error_code="RESISTOR_VALUE_LOW"
                        ))
                    elif value > 10000000:
                        errors.append(ValidationError(
                            field=f"component_{i}_value",
                            message=f"Resistor value ({comp.value}) is unusually high.",
                            suggestion="Typical resistor values range from 1Ω to 10MΩ. Verify the value is correct.",
                            error_code="RESISTOR_VALUE_HIGH"
                        ))
        
        return errors
    
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
        # Check for empty or whitespace-only prompt
        if not prompt or not prompt.strip():
            return False, (
                "Prompt cannot be empty. Please provide a description of your PCB design.\n"
                "Example: 'Design a 40x20mm PCB with a 9V battery, LED, and 220-ohm resistor in series.'"
            )
        
        # Check minimum length (characters)
        if len(prompt) < 10:
            return False, (
                f"Prompt is too short ({len(prompt)} characters). Minimum is 10 characters.\n"
                "Please provide more details about your design, including:\n"
                "  • Board dimensions (e.g., 40x20mm)\n"
                "  • Power source (e.g., 9V battery, USB)\n"
                "  • Components (e.g., LED, resistor, capacitor)\n"
                "  • Connections between components"
            )
        
        # Check maximum length (characters)
        if len(prompt) > 10000:
            return False, (
                f"Prompt is too long ({len(prompt)} characters). Maximum is 10,000 characters.\n"
                "Please break down your design into smaller, focused descriptions.\n"
                "Consider describing one subsystem or module at a time."
            )
        
        # Check for at least one component keyword
        has_component = any(
            re.search(pattern, prompt, re.IGNORECASE)
            for pattern in self.COMPONENT_PATTERNS.values()
        )
        
        if not has_component:
            component_examples = ", ".join(list(self.COMPONENT_PATTERNS.keys())[:8])
            return False, (
                "No recognizable components found in prompt.\n"
                f"Please include at least one component type such as: {component_examples}.\n"
                "Example: 'Design a simple LED circuit with a resistor and 9V battery.'"
            )
        
        # Check for suspicious patterns that might indicate invalid input
        if re.search(r"<script|javascript:|onerror=", prompt, re.IGNORECASE):
            return False, (
                "Prompt contains invalid characters or patterns.\n"
                "Please provide a plain text description of your PCB design."
            )
        
        return True, None