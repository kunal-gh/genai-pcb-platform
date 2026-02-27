"""
Manufacturing file export system.

Generates Gerber files, drill files, pick-and-place files, and STEP models
for PCB manufacturing and assembly.
"""

import os
import csv
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ManufacturingExportError(Exception):
    """Exception raised for manufacturing export errors."""
    pass


@dataclass
class ComponentPlacement:
    """Component placement information for pick-and-place."""
    reference: str
    value: str
    package: str
    x: float
    y: float
    rotation: float
    layer: str  # "top" or "bottom"


@dataclass
class DrillHole:
    """Drill hole information."""
    x: float
    y: float
    diameter: float
    plated: bool = True


class ManufacturingExporter:
    """
    Manufacturing file export system.
    
    Generates all files needed for PCB manufacturing and assembly.
    """
    
    def __init__(self, project_name: str, output_dir: str):
        """
        Initialize manufacturing exporter.
        
        Args:
            project_name: Name of the project
            output_dir: Output directory for files
        """
        self.project_name = project_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Standard Gerber file extensions
        self.gerber_extensions = {
            "top_copper": "GTL",
            "bottom_copper": "GBL",
            "top_soldermask": "GTS",
            "bottom_soldermask": "GBS",
            "top_silkscreen": "GTO",
            "bottom_silkscreen": "GBO",
            "top_paste": "GTP",
            "bottom_paste": "GBP",
            "outline": "GKO",
            "drill": "TXT"
        }
    
    def export_gerber_files(
        self,
        pcb_data: Dict[str, Any],
        layers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export Gerber files for PCB manufacturing.
        
        Args:
            pcb_data: PCB layout data
            layers: List of layers to export (default: all standard layers)
            
        Returns:
            Export results with file paths
        """
        if layers is None:
            layers = [
                "top_copper", "bottom_copper",
                "top_soldermask", "bottom_soldermask", 
                "top_silkscreen", "bottom_silkscreen",
                "outline"
            ]
        
        try:
            gerber_files = {}
            
            for layer in layers:
                filename = f"{self.project_name}.{self.gerber_extensions[layer]}"
                filepath = os.path.join(self.output_dir, filename)
                
                # Generate Gerber content for layer
                gerber_content = self._generate_gerber_content(layer, pcb_data)
                
                with open(filepath, 'w') as f:
                    f.write(gerber_content)
                
                gerber_files[layer] = filepath
                logger.info(f"Generated Gerber file: {filename}")
            
            # Generate aperture list
            aperture_file = self._generate_aperture_list(pcb_data)
            if aperture_file:
                gerber_files["apertures"] = aperture_file
            
            return {
                "success": True,
                "files": gerber_files,
                "layer_count": len(layers)
            }
            
        except Exception as e:
            logger.error(f"Failed to export Gerber files: {str(e)}")
            raise ManufacturingExportError(f"Gerber export failed: {str(e)}")
    
    def _generate_gerber_content(self, layer: str, pcb_data: Dict[str, Any]) -> str:
        """
        Generate Gerber file content for a specific layer.
        
        Args:
            layer: Layer name
            pcb_data: PCB layout data
            
        Returns:
            Gerber file content
        """
        # Gerber file header
        content = [
            "G04 #@! TF.GenerationSoftware,GenAI PCB Platform*",
            f"G04 #@! TF.CreationDate,{self._get_timestamp()}*",
            f"G04 #@! TF.ProjectId,{self.project_name},1,rev1*",
            f"G04 #@! TF.FileFunction,{self._get_file_function(layer)}*",
            "G04 #@! TF.FilePolarity,Positive*",
            "%FSLAX36Y36*%",
            "%MOMM*%",
            "%LN{layer}*%"
        ]
        
        # Add aperture definitions
        content.extend(self._get_aperture_definitions(layer))
        
        # Add layer-specific content
        if layer == "outline":
            content.extend(self._generate_outline_content(pcb_data))
        elif "copper" in layer:
            content.extend(self._generate_copper_content(layer, pcb_data))
        elif "soldermask" in layer:
            content.extend(self._generate_soldermask_content(layer, pcb_data))
        elif "silkscreen" in layer:
            content.extend(self._generate_silkscreen_content(layer, pcb_data))
        
        # Gerber file footer
        content.extend([
            "M02*"
        ])
        
        return "\n".join(content)
    
    def _get_file_function(self, layer: str) -> str:
        """Get Gerber file function attribute."""
        functions = {
            "top_copper": "Copper,L1,Top",
            "bottom_copper": "Copper,L2,Bot",
            "top_soldermask": "Soldermask,Top",
            "bottom_soldermask": "Soldermask,Bot",
            "top_silkscreen": "Legend,Top",
            "bottom_silkscreen": "Legend,Bot",
            "outline": "Profile,NP"
        }
        return functions.get(layer, "Other")
    
    def _get_aperture_definitions(self, layer: str) -> List[str]:
        """Get aperture definitions for layer."""
        apertures = [
            "%ADD10C,0.152400*%",  # 0.15mm circle
            "%ADD11C,0.203200*%",  # 0.20mm circle
            "%ADD12R,1.600000X0.800000*%",  # Rectangle for pads
            "%ADD13C,0.100000*%"   # Small circle for vias
        ]
        return apertures
    
    def _generate_outline_content(self, pcb_data: Dict[str, Any]) -> List[str]:
        """Generate board outline content."""
        width = pcb_data.get("width", 100.0)
        height = pcb_data.get("height", 80.0)
        
        # Convert mm to Gerber units (micrometers)
        w_units = int(width * 1000)
        h_units = int(height * 1000)
        
        return [
            "G01*",
            "D10*",
            "X0Y0D02*",
            f"X{w_units}Y0D01*",
            f"X{w_units}Y{h_units}D01*",
            f"X0Y{h_units}D01*",
            "X0Y0D01*"
        ]
    
    def _generate_copper_content(self, layer: str, pcb_data: Dict[str, Any]) -> List[str]:
        """Generate copper layer content."""
        content = ["G01*", "D11*"]
        
        # Add sample traces and pads
        components = pcb_data.get("components", [])
        for i, comp in enumerate(components[:5]):  # Limit for demo
            x = int((10 + i * 20) * 1000)  # Convert to micrometers
            y = int(40 * 1000)
            content.extend([
                f"X{x}Y{y}D03*",  # Flash pad
            ])
        
        return content
    
    def _generate_soldermask_content(self, layer: str, pcb_data: Dict[str, Any]) -> List[str]:
        """Generate soldermask layer content."""
        content = ["G01*", "D12*"]
        
        # Add soldermask openings for pads
        components = pcb_data.get("components", [])
        for i, comp in enumerate(components[:5]):
            x = int((10 + i * 20) * 1000)
            y = int(40 * 1000)
            content.append(f"X{x}Y{y}D03*")
        
        return content
    
    def _generate_silkscreen_content(self, layer: str, pcb_data: Dict[str, Any]) -> List[str]:
        """Generate silkscreen layer content."""
        content = ["G01*", "D10*"]
        
        # Add component reference designators
        components = pcb_data.get("components", [])
        for i, comp in enumerate(components[:5]):
            x = int((10 + i * 20) * 1000)
            y = int(50 * 1000)
            # In real implementation, would add text rendering
            content.append(f"X{x}Y{y}D02*")
        
        return content
    
    def _generate_aperture_list(self, pcb_data: Dict[str, Any]) -> Optional[str]:
        """Generate aperture list file."""
        aperture_file = os.path.join(self.output_dir, f"{self.project_name}.apr")
        
        apertures = [
            "D10: 0.152400mm Circle",
            "D11: 0.203200mm Circle", 
            "D12: 1.600000x0.800000mm Rectangle",
            "D13: 0.100000mm Circle"
        ]
        
        with open(aperture_file, 'w') as f:
            f.write("Aperture List\n")
            f.write("=============\n\n")
            for aperture in apertures:
                f.write(f"{aperture}\n")
        
        return aperture_file
    
    def export_drill_files(
        self,
        drill_data: List[DrillHole]
    ) -> Dict[str, Any]:
        """
        Export drill files for PCB manufacturing.
        
        Args:
            drill_data: List of drill holes
            
        Returns:
            Export results
        """
        try:
            # Excellon drill file
            drill_file = os.path.join(self.output_dir, f"{self.project_name}.drl")
            
            # Group holes by diameter
            holes_by_diameter = {}
            for hole in drill_data:
                diameter = hole.diameter
                if diameter not in holes_by_diameter:
                    holes_by_diameter[diameter] = []
                holes_by_diameter[diameter].append(hole)
            
            # Generate Excellon content
            content = [
                "M48",
                "INCH",
                "VER,1",
                "FMAT,2",
                "TCST,OFF"
            ]
            
            # Tool definitions
            tool_num = 1
            tool_map = {}
            for diameter in sorted(holes_by_diameter.keys()):
                content.append(f"T{tool_num:02d}C{diameter:.4f}")
                tool_map[diameter] = tool_num
                tool_num += 1
            
            content.append("%")
            content.append("G90")
            content.append("G05")
            
            # Drill coordinates
            for diameter, holes in holes_by_diameter.items():
                tool_num = tool_map[diameter]
                content.append(f"T{tool_num:02d}")
                
                for hole in holes:
                    x_inch = hole.x / 25.4  # Convert mm to inches
                    y_inch = hole.y / 25.4
                    content.append(f"X{x_inch:.4f}Y{y_inch:.4f}")
            
            content.append("T00")
            content.append("M30")
            
            with open(drill_file, 'w') as f:
                f.write("\n".join(content))
            
            # Drill report
            report_file = self._generate_drill_report(drill_data, holes_by_diameter)
            
            logger.info(f"Generated drill file with {len(drill_data)} holes")
            
            return {
                "success": True,
                "drill_file": drill_file,
                "report_file": report_file,
                "hole_count": len(drill_data),
                "tool_count": len(holes_by_diameter)
            }
            
        except Exception as e:
            logger.error(f"Failed to export drill files: {str(e)}")
            raise ManufacturingExportError(f"Drill export failed: {str(e)}")
    
    def _generate_drill_report(
        self,
        drill_data: List[DrillHole],
        holes_by_diameter: Dict[float, List[DrillHole]]
    ) -> str:
        """Generate drill report file."""
        report_file = os.path.join(self.output_dir, f"{self.project_name}_drill_report.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"Drill Report for {self.project_name}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total holes: {len(drill_data)}\n")
            f.write(f"Tool count: {len(holes_by_diameter)}\n\n")
            
            f.write("Tools:\n")
            for i, diameter in enumerate(sorted(holes_by_diameter.keys()), 1):
                hole_count = len(holes_by_diameter[diameter])
                f.write(f"T{i:02d}: {diameter:.3f}mm ({hole_count} holes)\n")
        
        return report_file
    
    def export_pick_and_place(
        self,
        components: List[ComponentPlacement]
    ) -> Dict[str, Any]:
        """
        Export pick-and-place files for assembly.
        
        Args:
            components: List of component placements
            
        Returns:
            Export results
        """
        try:
            # Separate top and bottom components
            top_components = [c for c in components if c.layer == "top"]
            bottom_components = [c for c in components if c.layer == "bottom"]
            
            files = {}
            
            # Export top side
            if top_components:
                top_file = self._export_pnp_file(top_components, "top")
                files["top"] = top_file
            
            # Export bottom side
            if bottom_components:
                bottom_file = self._export_pnp_file(bottom_components, "bottom")
                files["bottom"] = bottom_file
            
            # Generate assembly report
            report_file = self._generate_assembly_report(components)
            files["report"] = report_file
            
            logger.info(f"Generated pick-and-place files for {len(components)} components")
            
            return {
                "success": True,
                "files": files,
                "component_count": len(components),
                "top_count": len(top_components),
                "bottom_count": len(bottom_components)
            }
            
        except Exception as e:
            logger.error(f"Failed to export pick-and-place files: {str(e)}")
            raise ManufacturingExportError(f"Pick-and-place export failed: {str(e)}")
    
    def _export_pnp_file(
        self,
        components: List[ComponentPlacement],
        side: str
    ) -> str:
        """Export pick-and-place file for one side."""
        filename = f"{self.project_name}_{side}_pnp.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow([
                "Designator", "Value", "Package", "Mid X", "Mid Y", "Rotation", "Layer"
            ])
            
            # Component data
            for comp in components:
                writer.writerow([
                    comp.reference,
                    comp.value,
                    comp.package,
                    f"{comp.x:.3f}",
                    f"{comp.y:.3f}",
                    f"{comp.rotation:.1f}",
                    comp.layer
                ])
        
        return filepath
    
    def _generate_assembly_report(self, components: List[ComponentPlacement]) -> str:
        """Generate assembly report."""
        report_file = os.path.join(self.output_dir, f"{self.project_name}_assembly_report.txt")
        
        # Count components by package
        package_counts = {}
        for comp in components:
            package = comp.package
            if package not in package_counts:
                package_counts[package] = 0
            package_counts[package] += 1
        
        with open(report_file, 'w') as f:
            f.write(f"Assembly Report for {self.project_name}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total components: {len(components)}\n")
            f.write(f"Unique packages: {len(package_counts)}\n\n")
            
            f.write("Package Summary:\n")
            for package, count in sorted(package_counts.items()):
                f.write(f"{package}: {count} components\n")
        
        return report_file
    
    def export_step_model(
        self,
        pcb_data: Dict[str, Any],
        components: List[ComponentPlacement]
    ) -> Dict[str, Any]:
        """
        Export STEP 3D model file.
        
        Args:
            pcb_data: PCB layout data
            components: Component placements
            
        Returns:
            Export results
        """
        try:
            step_file = os.path.join(self.output_dir, f"{self.project_name}.step")
            
            # Generate basic STEP file content
            # In real implementation, this would use OpenCASCADE or similar
            step_content = self._generate_step_content(pcb_data, components)
            
            with open(step_file, 'w') as f:
                f.write(step_content)
            
            logger.info(f"Generated STEP model: {step_file}")
            
            return {
                "success": True,
                "step_file": step_file,
                "component_count": len(components)
            }
            
        except Exception as e:
            logger.error(f"Failed to export STEP model: {str(e)}")
            raise ManufacturingExportError(f"STEP export failed: {str(e)}")
    
    def _generate_step_content(
        self,
        pcb_data: Dict[str, Any],
        components: List[ComponentPlacement]
    ) -> str:
        """Generate STEP file content."""
        # Basic STEP file header
        content = [
            "ISO-10303-21;",
            "HEADER;",
            f"FILE_DESCRIPTION(('PCB Model for {self.project_name}'),'2;1');",
            f"FILE_NAME('{self.project_name}.step','','','','','','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));",
            "ENDSEC;",
            "DATA;",
            "/* PCB Board */",
            "#1 = CARTESIAN_POINT('Origin',(0.0,0.0,0.0));",
            f"#2 = DIRECTION('Z-Axis',(0.0,0.0,1.0));",
            f"#3 = DIRECTION('X-Axis',(1.0,0.0,0.0));",
            "ENDSEC;",
            "END-ISO-10303-21;"
        ]
        
        return "\n".join(content)
    
    def generate_manufacturing_package(
        self,
        pcb_data: Dict[str, Any],
        components: List[ComponentPlacement],
        drill_holes: List[DrillHole]
    ) -> Dict[str, Any]:
        """
        Generate complete manufacturing package.
        
        Args:
            pcb_data: PCB layout data
            components: Component placements
            drill_holes: Drill hole data
            
        Returns:
            Complete package information
        """
        try:
            results = {}
            
            # Export Gerber files
            gerber_result = self.export_gerber_files(pcb_data)
            results["gerbers"] = gerber_result
            
            # Export drill files
            drill_result = self.export_drill_files(drill_holes)
            results["drill"] = drill_result
            
            # Export pick-and-place
            pnp_result = self.export_pick_and_place(components)
            results["pick_and_place"] = pnp_result
            
            # Export STEP model
            step_result = self.export_step_model(pcb_data, components)
            results["step"] = step_result
            
            # Generate package summary
            summary_file = self._generate_package_summary(results)
            results["summary"] = summary_file
            
            logger.info(f"Generated complete manufacturing package in {self.output_dir}")
            
            return {
                "success": True,
                "output_dir": self.output_dir,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to generate manufacturing package: {str(e)}")
            raise ManufacturingExportError(f"Package generation failed: {str(e)}")
    
    def _generate_package_summary(self, results: Dict[str, Any]) -> str:
        """Generate manufacturing package summary."""
        summary_file = os.path.join(self.output_dir, f"{self.project_name}_manufacturing_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"Manufacturing Package Summary\n")
            f.write(f"Project: {self.project_name}\n")
            f.write(f"Generated: {self._get_timestamp()}\n")
            f.write("=" * 50 + "\n\n")
            
            # Gerber files
            if "gerbers" in results:
                gerber_files = results["gerbers"].get("files", {})
                f.write(f"Gerber Files ({len(gerber_files)}):\n")
                for layer, filepath in gerber_files.items():
                    filename = os.path.basename(filepath)
                    f.write(f"  {layer}: {filename}\n")
                f.write("\n")
            
            # Drill files
            if "drill" in results:
                drill_info = results["drill"]
                f.write(f"Drill Files:\n")
                f.write(f"  Holes: {drill_info.get('hole_count', 0)}\n")
                f.write(f"  Tools: {drill_info.get('tool_count', 0)}\n\n")
            
            # Pick-and-place
            if "pick_and_place" in results:
                pnp_info = results["pick_and_place"]
                f.write(f"Assembly Files:\n")
                f.write(f"  Components: {pnp_info.get('component_count', 0)}\n")
                f.write(f"  Top side: {pnp_info.get('top_count', 0)}\n")
                f.write(f"  Bottom side: {pnp_info.get('bottom_count', 0)}\n\n")
            
            # STEP model
            if "step" in results:
                f.write(f"3D Model: {self.project_name}.step\n")
        
        return summary_file
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")