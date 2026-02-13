"""
Component selection and recommendation engine.

Selects components based on electrical parameters, availability, and pricing.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import logging

from ..models.component import Component, ComponentCategory, PackageType, Manufacturer

logger = logging.getLogger(__name__)


class ComponentSelector:
    """
    Component selection and recommendation engine.
    
    Selects optimal components based on electrical requirements,
    availability, pricing, and design constraints.
    """
    
    def __init__(self, db: Session):
        """
        Initialize component selector.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def select_resistor(
        self,
        resistance: float,
        tolerance: float = 0.01,
        power_rating: float = 0.125,
        package_type: Optional[PackageType] = None
    ) -> Optional[Component]:
        """
        Select resistor based on electrical parameters.
        
        Args:
            resistance: Resistance value in ohms
            tolerance: Tolerance (e.g., 0.01 for 1%)
            power_rating: Power rating in watts
            package_type: Preferred package type
            
        Returns:
            Selected component or None if not found
        """
        query = self.db.query(Component).filter(
            Component.category == ComponentCategory.RESISTOR,
            Component.in_stock == True,
            Component.lifecycle_status == "active"
        )
        
        if package_type:
            query = query.filter(Component.package_type == package_type)
        
        # Find components matching electrical parameters
        candidates = []
        for component in query.all():
            params = component.electrical_parameters or {}
            
            # Check resistance value
            res_param = params.get("resistance", {})
            if not res_param:
                continue
            
            res_value = res_param.get("value")
            res_tolerance = res_param.get("tolerance", 0.05)
            
            if res_value and abs(res_value - resistance) / resistance < res_tolerance:
                # Check power rating
                power_param = params.get("power_rating", {})
                power_value = power_param.get("value", 0)
                
                if power_value >= power_rating:
                    candidates.append(component)
        
        # Sort by price and return cheapest
        if candidates:
            candidates.sort(key=lambda c: c.get_price_for_quantity(1))
            return candidates[0]
        
        return None
    
    def select_capacitor(
        self,
        capacitance: float,
        voltage_rating: float,
        tolerance: float = 0.10,
        package_type: Optional[PackageType] = None
    ) -> Optional[Component]:
        """
        Select capacitor based on electrical parameters.
        
        Args:
            capacitance: Capacitance value in farads
            voltage_rating: Voltage rating in volts
            tolerance: Tolerance (e.g., 0.10 for 10%)
            package_type: Preferred package type
            
        Returns:
            Selected component or None if not found
        """
        query = self.db.query(Component).filter(
            Component.category == ComponentCategory.CAPACITOR,
            Component.in_stock == True,
            Component.lifecycle_status == "active"
        )
        
        if package_type:
            query = query.filter(Component.package_type == package_type)
        
        candidates = []
        for component in query.all():
            params = component.electrical_parameters or {}
            
            # Check capacitance value
            cap_param = params.get("capacitance", {})
            if not cap_param:
                continue
            
            cap_value = cap_param.get("value")
            cap_tolerance = cap_param.get("tolerance", 0.20)
            
            if cap_value and abs(cap_value - capacitance) / capacitance < cap_tolerance:
                # Check voltage rating
                voltage_param = params.get("voltage_rating", {})
                voltage_value = voltage_param.get("value", 0)
                
                if voltage_value >= voltage_rating:
                    candidates.append(component)
        
        if candidates:
            candidates.sort(key=lambda c: c.get_price_for_quantity(1))
            return candidates[0]
        
        return None
    
    def find_alternatives(
        self,
        component: Component,
        max_alternatives: int = 5
    ) -> List[Component]:
        """
        Find alternative components with similar specifications.
        
        Args:
            component: Reference component
            max_alternatives: Maximum number of alternatives to return
            
        Returns:
            List of alternative components
        """
        query = self.db.query(Component).filter(
            Component.category == component.category,
            Component.package_type == component.package_type,
            Component.id != component.id,
            Component.in_stock == True,
            Component.lifecycle_status == "active"
        )
        
        # Get all candidates
        candidates = query.all()
        
        # Score alternatives based on parameter similarity
        scored_alternatives = []
        ref_params = component.electrical_parameters or {}
        
        for candidate in candidates:
            cand_params = candidate.electrical_parameters or {}
            score = self._calculate_similarity_score(ref_params, cand_params)
            scored_alternatives.append((score, candidate))
        
        # Sort by score (descending) and return top alternatives
        scored_alternatives.sort(key=lambda x: x[0], reverse=True)
        return [comp for _, comp in scored_alternatives[:max_alternatives]]
    
    def _calculate_similarity_score(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity score between two parameter sets.
        
        Args:
            params1: First parameter set
            params2: Second parameter set
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not params1 or not params2:
            return 0.0
        
        # Find common parameters
        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 0.0
        
        total_score = 0.0
        for key in common_keys:
            val1 = params1[key].get("value") if isinstance(params1[key], dict) else params1[key]
            val2 = params2[key].get("value") if isinstance(params2[key], dict) else params2[key]
            
            if val1 and val2 and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Calculate relative difference
                if val1 == val2:
                    total_score += 1.0
                else:
                    diff = abs(val1 - val2) / max(abs(val1), abs(val2))
                    total_score += max(0, 1.0 - diff)
        
        return total_score / len(common_keys)
    
    def check_availability(self, component: Component) -> Dict[str, Any]:
        """
        Check component availability and pricing.
        
        Args:
            component: Component to check
            
        Returns:
            Availability information
        """
        return {
            "part_number": component.part_number,
            "in_stock": component.in_stock,
            "lifecycle_status": component.lifecycle_status,
            "min_order_quantity": component.min_order_quantity,
            "pricing": component.pricing,
            "suppliers": component.suppliers
        }
    
    def recommend_replacement(
        self,
        obsolete_component: Component
    ) -> Optional[Component]:
        """
        Recommend replacement for obsolete component.
        
        Args:
            obsolete_component: Obsolete component
            
        Returns:
            Recommended replacement or None
        """
        # Find active alternatives
        alternatives = self.find_alternatives(obsolete_component, max_alternatives=10)
        
        # Filter for active components
        active_alternatives = [
            comp for comp in alternatives
            if comp.lifecycle_status == "active" and comp.in_stock
        ]
        
        if active_alternatives:
            # Return cheapest active alternative
            active_alternatives.sort(key=lambda c: c.get_price_for_quantity(1))
            return active_alternatives[0]
        
        return None
    
    def select_by_category(
        self,
        category: ComponentCategory,
        electrical_params: Dict[str, Any],
        package_type: Optional[PackageType] = None,
        max_results: int = 10
    ) -> List[Component]:
        """
        Select components by category and electrical parameters.
        
        Args:
            category: Component category
            electrical_params: Required electrical parameters
            package_type: Preferred package type
            max_results: Maximum number of results
            
        Returns:
            List of matching components
        """
        query = self.db.query(Component).filter(
            Component.category == category,
            Component.in_stock == True,
            Component.lifecycle_status == "active"
        )
        
        if package_type:
            query = query.filter(Component.package_type == package_type)
        
        # Get all candidates
        candidates = []
        for component in query.all():
            if self._matches_parameters(component.electrical_parameters, electrical_params):
                candidates.append(component)
        
        # Sort by price
        candidates.sort(key=lambda c: c.get_price_for_quantity(1))
        return candidates[:max_results]
    
    def _matches_parameters(
        self,
        component_params: Optional[Dict[str, Any]],
        required_params: Dict[str, Any]
    ) -> bool:
        """
        Check if component parameters match requirements.
        
        Args:
            component_params: Component electrical parameters
            required_params: Required parameters
            
        Returns:
            True if parameters match
        """
        if not component_params:
            return False
        
        for key, required_value in required_params.items():
            if key not in component_params:
                return False
            
            comp_param = component_params[key]
            comp_value = comp_param.get("value") if isinstance(comp_param, dict) else comp_param
            
            # Handle different comparison types
            if isinstance(required_value, dict):
                req_value = required_value.get("value")
                req_min = required_value.get("min")
                req_max = required_value.get("max")
                
                if req_value is not None:
                    tolerance = required_value.get("tolerance", 0.1)
                    if abs(comp_value - req_value) / req_value > tolerance:
                        return False
                
                if req_min is not None and comp_value < req_min:
                    return False
                
                if req_max is not None and comp_value > req_max:
                    return False
            else:
                # Direct value comparison
                if comp_value != required_value:
                    return False
        
        return True
