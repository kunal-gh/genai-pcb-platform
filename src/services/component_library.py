"""
Component library integration service.

Connects SKiDL code generation with component knowledge graph for
symbol lookup, validation, and alternative suggestions.
"""

from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
import logging

from ..models.component import Component, ComponentCategory, Manufacturer
from .component_selector import ComponentSelector

logger = logging.getLogger(__name__)


class ComponentLibraryError(Exception):
    """Exception raised for component library errors."""
    pass


class ComponentLibrary:
    """
    Component library integration service.
    
    Provides component symbol lookup, validation, and alternative
    suggestions for SKiDL code generation.
    """
    
    def __init__(self, db: Session):
        """
        Initialize component library.
        
        Args:
            db: Database session
        """
        self.db = db
        self.selector = ComponentSelector(db)
        
        # Standard KiCad library mappings
        self.library_mappings = {
            ComponentCategory.RESISTOR: "Device:R",
            ComponentCategory.CAPACITOR: "Device:C",
            ComponentCategory.INDUCTOR: "Device:L",
            ComponentCategory.LED: "Device:LED",
            ComponentCategory.DIODE: "Device:D",
            ComponentCategory.TRANSISTOR: "Device:Q_NPN_BCE",
            ComponentCategory.IC: "Device:U",
            ComponentCategory.CONNECTOR: "Connector:Conn",
            ComponentCategory.SWITCH: "Switch:SW_Push",
            ComponentCategory.CRYSTAL: "Device:Crystal",
            ComponentCategory.FUSE: "Device:Fuse"
        }
    
    def lookup_symbol(
        self,
        part_number: Optional[str] = None,
        category: Optional[ComponentCategory] = None,
        electrical_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Look up component symbol information.
        
        Args:
            part_number: Specific part number to look up
            category: Component category
            electrical_params: Electrical parameters for matching
            
        Returns:
            Dictionary with symbol information:
                - library: KiCad library name
                - symbol: Symbol name
                - footprint: Footprint reference
                - component: Component object (if found)
                - alternatives: List of alternative components
        """
        component = None
        
        # Look up by part number
        if part_number:
            component = self.db.query(Component).filter(
                Component.part_number == part_number
            ).first()
        
        # Look up by category and parameters
        elif category and electrical_params:
            components = self.selector.select_by_category(
                category=category,
                electrical_params=electrical_params,
                max_results=1
            )
            if components:
                component = components[0]
        
        # Look up by category only
        elif category:
            component = self.db.query(Component).filter(
                Component.category == category,
                Component.in_stock == True,
                Component.lifecycle_status == "active"
            ).first()
        
        if not component:
            # Return default symbol for category
            return self._get_default_symbol(category)
        
        # Get alternatives
        alternatives = self.selector.find_alternatives(component, max_alternatives=3)
        
        return {
            "library": self._get_library_name(component),
            "symbol": self._get_symbol_name(component),
            "footprint": component.footprint_id or self._get_default_footprint(component),
            "component": component,
            "alternatives": alternatives
        }
    
    def _get_library_name(self, component: Component) -> str:
        """
        Get KiCad library name for component.
        
        Args:
            component: Component object
            
        Returns:
            Library name
        """
        if component.symbol_id:
            # Extract library from symbol_id (format: "Library:Symbol")
            if ":" in component.symbol_id:
                return component.symbol_id.split(":")[0]
        
        # Use default mapping
        return self.library_mappings.get(component.category, "Device")
    
    def _get_symbol_name(self, component: Component) -> str:
        """
        Get symbol name for component.
        
        Args:
            component: Component object
            
        Returns:
            Symbol name
        """
        if component.symbol_id:
            # Extract symbol from symbol_id (format: "Library:Symbol")
            if ":" in component.symbol_id:
                return component.symbol_id.split(":")[1]
            return component.symbol_id
        
        # Use default symbol based on category
        category_symbols = {
            ComponentCategory.RESISTOR: "R",
            ComponentCategory.CAPACITOR: "C",
            ComponentCategory.INDUCTOR: "L",
            ComponentCategory.LED: "LED",
            ComponentCategory.DIODE: "D",
            ComponentCategory.TRANSISTOR: "Q_NPN_BCE",
            ComponentCategory.IC: "U",
            ComponentCategory.CONNECTOR: "Conn",
            ComponentCategory.SWITCH: "SW_Push",
            ComponentCategory.CRYSTAL: "Crystal",
            ComponentCategory.FUSE: "Fuse"
        }
        
        return category_symbols.get(component.category, "U")
    
    def _get_default_footprint(self, component: Component) -> str:
        """
        Get default footprint for component.
        
        Args:
            component: Component object
            
        Returns:
            Footprint reference
        """
        # Use package name if available
        if component.package_name:
            if component.category == ComponentCategory.RESISTOR:
                return f"Resistor_SMD:R_{component.package_name}"
            elif component.category == ComponentCategory.CAPACITOR:
                return f"Capacitor_SMD:C_{component.package_name}"
        
        # Default footprints by category
        defaults = {
            ComponentCategory.RESISTOR: "Resistor_SMD:R_0805_2012Metric",
            ComponentCategory.CAPACITOR: "Capacitor_SMD:C_0805_2012Metric",
            ComponentCategory.LED: "LED_SMD:LED_0805_2012Metric",
            ComponentCategory.DIODE: "Diode_SMD:D_SOD-123",
            ComponentCategory.IC: "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"
        }
        
        return defaults.get(component.category, "")
    
    def _get_default_symbol(self, category: Optional[ComponentCategory]) -> Dict[str, Any]:
        """
        Get default symbol information for category.
        
        Args:
            category: Component category
            
        Returns:
            Default symbol information
        """
        if not category:
            return {
                "library": "Device",
                "symbol": "U",
                "footprint": "",
                "component": None,
                "alternatives": []
            }
        
        library_symbol = self.library_mappings.get(category, "Device:U")
        library, symbol = library_symbol.split(":") if ":" in library_symbol else ("Device", library_symbol)
        
        return {
            "library": library,
            "symbol": symbol,
            "footprint": "",
            "component": None,
            "alternatives": []
        }
    
    def validate_component(
        self,
        library: str,
        symbol: str,
        value: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate component symbol usage.
        
        Args:
            library: KiCad library name
            symbol: Symbol name
            value: Component value
            
        Returns:
            Validation results:
                - valid: bool
                - warnings: List[str]
                - suggestions: List[str]
        """
        result = {
            "valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        # Check if library is standard
        standard_libraries = ["Device", "Connector", "Switch", "Amplifier_Operational"]
        if library not in standard_libraries:
            result["warnings"].append(f"Non-standard library: {library}")
        
        # Check if symbol follows naming conventions
        if not symbol or len(symbol) < 1:
            result["valid"] = False
            result["warnings"].append("Invalid symbol name")
        
        # Validate value format for passive components
        if symbol in ["R", "C", "L"] and value:
            if not self._validate_value_format(symbol, value):
                result["warnings"].append(f"Unusual value format: {value}")
                result["suggestions"].append(f"Consider standard format (e.g., '10k' for resistors)")
        
        return result
    
    def _validate_value_format(self, symbol: str, value: str) -> bool:
        """
        Validate component value format.
        
        Args:
            symbol: Component symbol
            value: Component value
            
        Returns:
            True if format is valid
        """
        import re
        
        if symbol == "R":
            # Resistor: should be number with optional k, M, ohm
            return bool(re.match(r'^\d+\.?\d*[kMG]?(ohm)?$', value, re.IGNORECASE))
        elif symbol == "C":
            # Capacitor: should be number with optional n, u, p, F
            return bool(re.match(r'^\d+\.?\d*[nupmkM]?F?$', value, re.IGNORECASE))
        elif symbol == "L":
            # Inductor: should be number with optional n, u, m, H
            return bool(re.match(r'^\d+\.?\d*[numkM]?H?$', value, re.IGNORECASE))
        
        return True
    
    def find_missing_components(
        self,
        skidl_code: str
    ) -> List[Dict[str, Any]]:
        """
        Find components in SKiDL code that are not in the database.
        
        Args:
            skidl_code: SKiDL Python code
            
        Returns:
            List of missing component information
        """
        import re
        
        missing = []
        
        # Extract Part() calls
        pattern = r'Part\([\'"]([^\'"]+)[\'"]\s*,\s*[\'"]([^\'"]+)[\'"]'
        
        for match in re.finditer(pattern, skidl_code):
            library = match.group(1)
            part = match.group(2)
            
            # Check if component exists in database
            # For now, we'll check by symbol name
            component = self.db.query(Component).filter(
                Component.symbol_id.like(f"%{part}%")
            ).first()
            
            if not component:
                missing.append({
                    "library": library,
                    "part": part,
                    "alternatives": self._suggest_alternatives(library, part)
                })
        
        return missing
    
    def _suggest_alternatives(
        self,
        library: str,
        part: str
    ) -> List[Dict[str, str]]:
        """
        Suggest alternative components.
        
        Args:
            library: Library name
            part: Part name
            
        Returns:
            List of alternative suggestions
        """
        alternatives = []
        
        # Map common part names to categories
        part_category_map = {
            "R": ComponentCategory.RESISTOR,
            "C": ComponentCategory.CAPACITOR,
            "L": ComponentCategory.INDUCTOR,
            "LED": ComponentCategory.LED,
            "D": ComponentCategory.DIODE,
            "Q": ComponentCategory.TRANSISTOR,
            "U": ComponentCategory.IC
        }
        
        # Try to find category
        category = None
        for key, cat in part_category_map.items():
            if part.startswith(key):
                category = cat
                break
        
        if category:
            # Find components in this category
            components = self.db.query(Component).filter(
                Component.category == category,
                Component.in_stock == True,
                Component.lifecycle_status == "active"
            ).limit(3).all()
            
            for comp in components:
                alternatives.append({
                    "part_number": comp.part_number,
                    "library": self._get_library_name(comp),
                    "symbol": self._get_symbol_name(comp),
                    "description": comp.description or ""
                })
        
        return alternatives
    
    def get_component_info(self, part_number: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed component information.
        
        Args:
            part_number: Component part number
            
        Returns:
            Component information dictionary or None
        """
        component = self.db.query(Component).filter(
            Component.part_number == part_number
        ).first()
        
        if not component:
            return None
        
        return {
            "part_number": component.part_number,
            "category": component.category.value,
            "description": component.description,
            "manufacturer": component.manufacturer.name if component.manufacturer else None,
            "package": component.package_name,
            "electrical_parameters": component.electrical_parameters,
            "footprint": component.footprint_id,
            "symbol": component.symbol_id,
            "in_stock": component.in_stock,
            "lifecycle_status": component.lifecycle_status,
            "pricing": component.pricing
        }
