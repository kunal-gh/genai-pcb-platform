"""
BOM (Bill of Materials) generation system.

Generates comprehensive BOMs with live component sourcing, pricing, and availability.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm import Session

from ..models.component import Component, ComponentCategory

logger = logging.getLogger(__name__)


class BOMGenerationError(Exception):
    """Exception raised when BOM generation fails."""
    pass


@dataclass
class BOMItem:
    """
    Bill of Materials item.
    
    Represents a single line item in the BOM with component details,
    quantity, and sourcing information.
    """
    reference_designators: List[str]  # e.g., ["R1", "R2", "R3"]
    quantity: int
    part_number: str
    manufacturer: str
    description: str
    value: Optional[str] = None  # e.g., "10K", "100nF"
    package: Optional[str] = None  # e.g., "0805", "SOT-23"
    
    # Sourcing information
    supplier: Optional[str] = None
    supplier_part_number: Optional[str] = None
    unit_price: Optional[float] = None
    extended_price: Optional[float] = None
    availability: Optional[str] = None  # e.g., "In Stock", "2 weeks"
    minimum_order_quantity: int = 1
    
    # Status flags
    is_obsolete: bool = False
    is_hard_to_source: bool = False
    alternative_parts: List[str] = None  # Alternative part numbers
    
    def __post_init__(self):
        if self.alternative_parts is None:
            self.alternative_parts = []
        if self.extended_price is None and self.unit_price is not None:
            self.extended_price = self.unit_price * self.quantity


@dataclass
class BOMSummary:
    """BOM summary with cost and sourcing statistics."""
    total_unique_parts: int
    total_components: int
    estimated_total_cost: float
    obsolete_parts_count: int
    hard_to_source_count: int
    generated_at: datetime


class BOMGenerator:
    """
    Comprehensive BOM generator.
    
    Extracts component information from schematics and generates
    detailed BOMs with live pricing and availability data.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize BOM generator.
        
        Args:
            db_session: Database session for component lookup
        """
        self.db = db_session
    
    def generate_bom(
        self,
        netlist: Dict[str, Any],
        include_pricing: bool = True,
        include_alternatives: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive BOM from netlist.
        
        Args:
            netlist: Netlist dictionary with components and connections
            include_pricing: Whether to include pricing information
            include_alternatives: Whether to suggest alternative parts
            
        Returns:
            Dictionary with BOM items and summary:
                - items: List[BOMItem]
                - summary: BOMSummary
                - warnings: List[str]
                
        Raises:
            BOMGenerationError: If BOM generation fails
        """
        logger.info("Generating BOM from netlist")
        
        try:
            # Extract components from netlist
            components = self._extract_components(netlist)
            
            # Group components by part number
            grouped_components = self._group_components(components)
            
            # Create BOM items
            bom_items = []
            warnings = []
            
            for part_number, comp_list in grouped_components.items():
                try:
                    bom_item = self._create_bom_item(
                        comp_list,
                        include_pricing=include_pricing,
                        include_alternatives=include_alternatives
                    )
                    bom_items.append(bom_item)
                    
                    # Check for warnings
                    if bom_item.is_obsolete:
                        warnings.append(
                            f"{part_number} is obsolete. "
                            f"Consider alternatives: {', '.join(bom_item.alternative_parts[:3])}"
                        )
                    if bom_item.is_hard_to_source:
                        warnings.append(
                            f"{part_number} is hard to source. "
                            f"Lead time: {bom_item.availability}"
                        )
                        
                except Exception as e:
                    logger.warning(f"Failed to create BOM item for {part_number}: {str(e)}")
                    warnings.append(f"Could not retrieve full information for {part_number}")
            
            # Generate summary
            summary = self._generate_summary(bom_items)
            
            logger.info(f"BOM generated successfully with {len(bom_items)} unique parts")
            
            return {
                "items": bom_items,
                "summary": summary,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"BOM generation failed: {str(e)}", exc_info=True)
            raise BOMGenerationError(f"Failed to generate BOM: {str(e)}")
    
    def _extract_components(self, netlist: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract component information from netlist.
        
        Args:
            netlist: Netlist dictionary
            
        Returns:
            List of component dictionaries
        """
        components = []
        
        # Extract from netlist components section
        if "components" in netlist:
            for comp in netlist["components"]:
                components.append({
                    "reference": comp.get("reference", ""),
                    "part_number": comp.get("part_number", ""),
                    "value": comp.get("value"),
                    "package": comp.get("package"),
                    "manufacturer": comp.get("manufacturer", ""),
                    "description": comp.get("description", "")
                })
        
        return components
    
    def _group_components(
        self,
        components: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group components by part number.
        
        Args:
            components: List of component dictionaries
            
        Returns:
            Dictionary mapping part numbers to component lists
        """
        grouped = {}
        
        for comp in components:
            part_number = comp.get("part_number", "UNKNOWN")
            if part_number not in grouped:
                grouped[part_number] = []
            grouped[part_number].append(comp)
        
        return grouped
    
    def _create_bom_item(
        self,
        components: List[Dict[str, Any]],
        include_pricing: bool = True,
        include_alternatives: bool = True
    ) -> BOMItem:
        """
        Create BOM item from grouped components.
        
        Args:
            components: List of components with same part number
            include_pricing: Whether to include pricing
            include_alternatives: Whether to suggest alternatives
            
        Returns:
            BOMItem with complete information
        """
        # Get reference designators
        references = [comp["reference"] for comp in components]
        quantity = len(components)
        
        # Get component details from first component
        first_comp = components[0]
        part_number = first_comp["part_number"]
        
        # Look up component in database
        component = self.db.query(Component).filter(
            Component.part_number == part_number
        ).first()
        
        if component:
            # Use database information
            manufacturer = component.manufacturer.name if component.manufacturer else first_comp.get("manufacturer", "")
            description = component.description or first_comp.get("description", "")
            value = first_comp.get("value") or component.get_parameter("value")
            package = first_comp.get("package") or component.package_type.value if component.package_type else None
            
            # Get pricing information
            unit_price = None
            supplier = None
            supplier_part_number = None
            availability = "Unknown"
            
            if include_pricing and component.pricing:
                # Get price for quantity
                unit_price = self._get_price_for_quantity(component.pricing, quantity)
                if component.suppliers:
                    # Get first supplier
                    first_supplier = component.suppliers[0] if isinstance(component.suppliers, list) else component.suppliers
                    supplier = first_supplier.get("name")
                    supplier_part_number = first_supplier.get("sku")
                    availability = "In Stock" if component.in_stock else "Out of Stock"
            
            # Check lifecycle status
            is_obsolete = component.lifecycle_status in ["obsolete", "discontinued"]
            is_hard_to_source = not component.in_stock
            
            # Get alternatives
            alternative_parts = []
            if include_alternatives and (is_obsolete or is_hard_to_source):
                alternatives = self._find_alternative_parts(component, quantity)
                alternative_parts = [alt.part_number for alt in alternatives]
            
        else:
            # Use netlist information only
            manufacturer = first_comp.get("manufacturer", "")
            description = first_comp.get("description", "")
            value = first_comp.get("value")
            package = first_comp.get("package")
            unit_price = None
            supplier = None
            supplier_part_number = None
            availability = "Unknown"
            is_obsolete = False
            is_hard_to_source = False
            alternative_parts = []
        
        return BOMItem(
            reference_designators=references,
            quantity=quantity,
            part_number=part_number,
            manufacturer=manufacturer,
            description=description,
            value=value,
            package=package,
            supplier=supplier,
            supplier_part_number=supplier_part_number,
            unit_price=unit_price,
            availability=availability,
            is_obsolete=is_obsolete,
            is_hard_to_source=is_hard_to_source,
            alternative_parts=alternative_parts
        )
    
    def _find_alternative_parts(
        self,
        component: Component,
        quantity: int,
        max_alternatives: int = 5
    ) -> List[Component]:
        """
        Find alternative parts for a component.
        
        Args:
            component: Component to find alternatives for
            quantity: Required quantity
            max_alternatives: Maximum number of alternatives to return
            
        Returns:
            List of alternative components
        """
        # Find components in same category with similar parameters
        alternatives = self.db.query(Component).filter(
            Component.category == component.category,
            Component.id != component.id,
            Component.lifecycle_status == "active",
            Component.in_stock == True
        ).limit(max_alternatives * 2).all()
        
        # Filter and rank alternatives
        ranked_alternatives = []
        for alt in alternatives:
            # Calculate similarity score (simple version)
            score = 0
            if alt.package_type == component.package_type:
                score += 1
            if alt.manufacturer_id == component.manufacturer_id:
                score += 0.5
            
            ranked_alternatives.append((score, alt))
        
        # Sort by score and return top alternatives
        ranked_alternatives.sort(key=lambda x: x[0], reverse=True)
        return [alt for score, alt in ranked_alternatives[:max_alternatives]]
    
    def _get_price_for_quantity(self, pricing_tiers: List[Dict], quantity: int) -> float:
        """
        Get unit price for given quantity from pricing tiers.
        
        Args:
            pricing_tiers: List of pricing tier dictionaries
            quantity: Quantity to price
            
        Returns:
            Unit price for the quantity
        """
        if not pricing_tiers:
            return None
        
        # Sort tiers by quantity
        sorted_tiers = sorted(pricing_tiers, key=lambda x: x.get("quantity", 0))
        
        # Find applicable tier
        applicable_price = sorted_tiers[0].get("price")
        for tier in sorted_tiers:
            if quantity >= tier.get("quantity", 0):
                applicable_price = tier.get("price")
            else:
                break
        
        return applicable_price
    
    def _generate_summary(self, bom_items: List[BOMItem]) -> BOMSummary:
        """
        Generate BOM summary statistics.
        
        Args:
            bom_items: List of BOM items
            
        Returns:
            BOMSummary with statistics
        """
        total_unique_parts = len(bom_items)
        total_components = sum(item.quantity for item in bom_items)
        
        # Calculate total cost
        estimated_total_cost = 0.0
        for item in bom_items:
            if item.extended_price is not None:
                estimated_total_cost += item.extended_price
        
        # Count issues
        obsolete_parts_count = sum(1 for item in bom_items if item.is_obsolete)
        hard_to_source_count = sum(1 for item in bom_items if item.is_hard_to_source)
        
        return BOMSummary(
            total_unique_parts=total_unique_parts,
            total_components=total_components,
            estimated_total_cost=estimated_total_cost,
            obsolete_parts_count=obsolete_parts_count,
            hard_to_source_count=hard_to_source_count,
            generated_at=datetime.utcnow()
        )
    
    def export_bom_csv(self, bom_data: Dict[str, Any], output_path: str):
        """
        Export BOM to CSV format.
        
        Args:
            bom_data: BOM data from generate_bom()
            output_path: Path to output CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Reference Designators",
                "Quantity",
                "Part Number",
                "Manufacturer",
                "Description",
                "Value",
                "Package",
                "Supplier",
                "Supplier Part Number",
                "Unit Price",
                "Extended Price",
                "Availability",
                "Status"
            ])
            
            # Write items
            for item in bom_data["items"]:
                status_flags = []
                if item.is_obsolete:
                    status_flags.append("OBSOLETE")
                if item.is_hard_to_source:
                    status_flags.append("HARD TO SOURCE")
                status = ", ".join(status_flags) if status_flags else "OK"
                
                writer.writerow([
                    ", ".join(item.reference_designators),
                    item.quantity,
                    item.part_number,
                    item.manufacturer,
                    item.description,
                    item.value or "",
                    item.package or "",
                    item.supplier or "",
                    item.supplier_part_number or "",
                    f"${item.unit_price:.2f}" if item.unit_price else "",
                    f"${item.extended_price:.2f}" if item.extended_price else "",
                    item.availability or "",
                    status
                ])
        
        logger.info(f"BOM exported to CSV: {output_path}")
    
    def export_bom_json(self, bom_data: Dict[str, Any], output_path: str):
        """
        Export BOM to JSON format.
        
        Args:
            bom_data: BOM data from generate_bom()
            output_path: Path to output JSON file
        """
        import json
        from dataclasses import asdict
        
        # Convert dataclasses to dictionaries
        export_data = {
            "items": [asdict(item) for item in bom_data["items"]],
            "summary": asdict(bom_data["summary"]),
            "warnings": bom_data["warnings"]
        }
        
        # Convert datetime to string
        export_data["summary"]["generated_at"] = export_data["summary"]["generated_at"].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"BOM exported to JSON: {output_path}")
