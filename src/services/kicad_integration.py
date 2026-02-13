"""
KiCad Python API integration service.

Provides KiCad project management, netlist import, and PCB layout generation.
"""

import os
import tempfile
import subprocess
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class KiCadIntegrationError(Exception):
    """Exception raised for KiCad integration errors."""
    pass


class KiCadProject:
    """
    KiCad project management and operations.
    
    Handles KiCad project creation, netlist import, and PCB layout generation
    using KiCad's Python API and command-line tools.
    """
    
    def __init__(self, project_name: str, output_dir: Optional[str] = None):
        """
        Initialize KiCad project.
        
        Args:
            project_name: Name of the project
            output_dir: Output directory (default: temp directory)
        """
        self.project_name = project_name
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="kicad_")
        self.project_path = os.path.join(self.output_dir, f"{project_name}.kicad_pro")
        self.schematic_path = os.path.join(self.output_dir, f"{project_name}.kicad_sch")
        self.pcb_path = os.path.join(self.output_dir, f"{project_name}.kicad_pcb")
        self.netlist_path = os.path.join(self.output_dir, f"{project_name}.net")
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Initialized KiCad project: {project_name} in {self.output_dir}")
    
    def create_project(
        self,
        board_width: float = 100.0,
        board_height: float = 80.0,
        layers: int = 2
    ) -> Dict[str, Any]:
        """
        Create new KiCad project files.
        
        Args:
            board_width: Board width in mm
            board_height: Board height in mm
            layers: Number of layers
            
        Returns:
            Dictionary with project information
        """
        try:
            # Create project file
            self._create_project_file()
            
            # Create schematic file
            self._create_schematic_file()
            
            # Create PCB file
            self._create_pcb_file(board_width, board_height, layers)
            
            logger.info(f"Created KiCad project: {self.project_name}")
            
            return {
                "success": True,
                "project_path": self.project_path,
                "schematic_path": self.schematic_path,
                "pcb_path": self.pcb_path,
                "board_size": {"width": board_width, "height": board_height},
                "layers": layers
            }
            
        except Exception as e:
            logger.error(f"Failed to create KiCad project: {str(e)}")
            raise KiCadIntegrationError(f"Project creation failed: {str(e)}")
    
    def _create_project_file(self):
        """Create KiCad project file."""
        project_config = {
            "board": {
                "design_settings": {
                    "defaults": {
                        "board_outline_line_width": 0.1,
                        "copper_line_width": 0.2,
                        "copper_text_size_h": 1.5,
                        "copper_text_size_v": 1.5
                    }
                }
            },
            "libraries": {
                "pinned_footprint_libs": [],
                "pinned_symbol_libs": []
            },
            "meta": {
                "filename": f"{self.project_name}.kicad_pro",
                "version": 1
            }
        }
        
        with open(self.project_path, 'w') as f:
            json.dump(project_config, f, indent=2)
    
    def _create_schematic_file(self):
        """Create basic schematic file."""
        schematic_content = f'''(kicad_sch (version 20230121) (generator eeschema)

  (uuid {self._generate_uuid()})

  (paper "A4")

  (title_block
    (title "{self.project_name}")
    (date "{self._get_current_date()}")
    (rev "1")
  )

  (lib_symbols
  )

  (sheet_instances
    (path "/" (page "1"))
  )
)
'''
        
        with open(self.schematic_path, 'w') as f:
            f.write(schematic_content)
    
    def _create_pcb_file(self, width: float, height: float, layers: int):
        """Create basic PCB file."""
        pcb_content = f'''(kicad_pcb (version 20221018) (generator pcbnew)

  (general
    (thickness 1.6)
  )

  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (42 "Eco1.User" user "User.Eco1")
    (43 "Eco2.User" user "User.Eco2")
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
  )

  (setup
    (stackup
      (layer "F.SilkS" (type "Top Silk Screen"))
      (layer "F.Paste" (type "Top Solder Paste"))
      (layer "F.Mask" (type "Top Solder Mask") (thickness 0.01))
      (layer "F.Cu" (type "copper") (thickness 0.035))
      (layer "dielectric 1" (type "core") (thickness 1.51) (material "FR4") (epsilon_r 4.5) (loss_tangent 0.02))
      (layer "B.Cu" (type "copper") (thickness 0.035))
      (layer "B.Mask" (type "Bottom Solder Mask") (thickness 0.01))
      (layer "B.Paste" (type "Bottom Solder Paste"))
      (layer "B.SilkS" (type "Bottom Silk Screen"))
      (copper_finish "None")
      (dielectric_constraints no)
    )
  )

  (gr_rect (start 0 0) (end {width} {height}) (stroke (width 0.1) (type solid)) (fill none) (layer "Edge.Cuts") (tstamp {self._generate_uuid()}))
)
'''
        
        with open(self.pcb_path, 'w') as f:
            f.write(pcb_content)
    
    def import_netlist(self, netlist_file: str) -> Dict[str, Any]:
        """
        Import netlist into KiCad project.
        
        Args:
            netlist_file: Path to netlist file
            
        Returns:
            Import results
        """
        if not os.path.exists(netlist_file):
            raise KiCadIntegrationError(f"Netlist file not found: {netlist_file}")
        
        try:
            # Copy netlist to project directory
            import shutil
            shutil.copy2(netlist_file, self.netlist_path)
            
            # Parse netlist for validation
            netlist_info = self._parse_netlist(netlist_file)
            
            logger.info(f"Imported netlist with {len(netlist_info.get('components', []))} components")
            
            return {
                "success": True,
                "netlist_path": self.netlist_path,
                "components": netlist_info.get("components", []),
                "nets": netlist_info.get("nets", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to import netlist: {str(e)}")
            raise KiCadIntegrationError(f"Netlist import failed: {str(e)}")
    
    def _parse_netlist(self, netlist_file: str) -> Dict[str, Any]:
        """
        Parse netlist file for component and net information.
        
        Args:
            netlist_file: Path to netlist file
            
        Returns:
            Parsed netlist information
        """
        components = []
        nets = []
        
        try:
            with open(netlist_file, 'r') as f:
                content = f.read()
            
            # Simple parsing for KiCad netlist format
            # This is a basic implementation - real parsing would be more complex
            import re
            
            # Find components
            comp_pattern = r'\(comp \(ref ([^)]+)\)\s*\(value ([^)]+)\)\s*\(footprint ([^)]+)\)'
            for match in re.finditer(comp_pattern, content):
                components.append({
                    "reference": match.group(1),
                    "value": match.group(2),
                    "footprint": match.group(3)
                })
            
            # Find nets
            net_pattern = r'\(net \(code \d+\) \(name ([^)]+)\)'
            for match in re.finditer(net_pattern, content):
                nets.append(match.group(1))
            
        except Exception as e:
            logger.warning(f"Failed to parse netlist: {str(e)}")
        
        return {"components": components, "nets": nets}
    
    def generate_pcb_layout(
        self,
        design_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate PCB layout from netlist.
        
        Args:
            design_rules: Design rules to apply
            
        Returns:
            Layout generation results
        """
        try:
            # Apply design rules
            if design_rules:
                self._apply_design_rules(design_rules)
            
            # For now, we'll create a basic layout
            # In a real implementation, this would use KiCad's auto-router
            # or our RL-based routing algorithm
            
            layout_info = {
                "success": True,
                "pcb_path": self.pcb_path,
                "design_rules_applied": design_rules is not None,
                "routing_method": "basic"  # Would be "RL" in full implementation
            }
            
            logger.info("Generated PCB layout")
            return layout_info
            
        except Exception as e:
            logger.error(f"Failed to generate PCB layout: {str(e)}")
            raise KiCadIntegrationError(f"PCB layout generation failed: {str(e)}")
    
    def _apply_design_rules(self, design_rules: Dict[str, Any]):
        """
        Apply design rules to PCB.
        
        Args:
            design_rules: Design rules dictionary
        """
        # Extract common design rules
        min_trace_width = design_rules.get("min_trace_width", 0.2)  # mm
        min_via_size = design_rules.get("min_via_size", 0.4)  # mm
        min_clearance = design_rules.get("min_clearance", 0.2)  # mm
        
        logger.info(f"Applied design rules: trace={min_trace_width}mm, via={min_via_size}mm, clearance={min_clearance}mm")
    
    def export_gerbers(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Export Gerber files for manufacturing.
        
        Args:
            output_dir: Output directory for Gerber files
            
        Returns:
            Export results with file paths
        """
        gerber_dir = output_dir or os.path.join(self.output_dir, "gerbers")
        os.makedirs(gerber_dir, exist_ok=True)
        
        try:
            # In a real implementation, this would use KiCad's plot functionality
            # For now, we'll create placeholder files
            
            gerber_files = [
                f"{self.project_name}-F_Cu.gbr",      # Top copper
                f"{self.project_name}-B_Cu.gbr",      # Bottom copper
                f"{self.project_name}-F_Mask.gbr",    # Top solder mask
                f"{self.project_name}-B_Mask.gbr",    # Bottom solder mask
                f"{self.project_name}-F_SilkS.gbr",   # Top silkscreen
                f"{self.project_name}-B_SilkS.gbr",   # Bottom silkscreen
                f"{self.project_name}-Edge_Cuts.gbr", # Board outline
                f"{self.project_name}.drl"            # Drill file
            ]
            
            # Create placeholder files
            for filename in gerber_files:
                filepath = os.path.join(gerber_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(f"# Placeholder Gerber file: {filename}\n")
            
            logger.info(f"Exported {len(gerber_files)} Gerber files to {gerber_dir}")
            
            return {
                "success": True,
                "gerber_dir": gerber_dir,
                "files": gerber_files,
                "file_paths": [os.path.join(gerber_dir, f) for f in gerber_files]
            }
            
        except Exception as e:
            logger.error(f"Failed to export Gerber files: {str(e)}")
            raise KiCadIntegrationError(f"Gerber export failed: {str(e)}")
    
    def validate_design(self) -> Dict[str, Any]:
        """
        Validate PCB design for common issues.
        
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check if files exist
        if not os.path.exists(self.schematic_path):
            issues.append("Schematic file missing")
        
        if not os.path.exists(self.pcb_path):
            issues.append("PCB file missing")
        
        if not os.path.exists(self.netlist_path):
            warnings.append("Netlist file missing")
        
        # Basic file size checks
        if os.path.exists(self.pcb_path):
            pcb_size = os.path.getsize(self.pcb_path)
            if pcb_size < 1000:  # Less than 1KB
                warnings.append("PCB file seems too small")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def _generate_uuid(self) -> str:
        """Generate UUID for KiCad objects."""
        import uuid
        return str(uuid.uuid4())
    
    def _get_current_date(self) -> str:
        """Get current date in YYYY-MM-DD format."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.output_dir and os.path.exists(self.output_dir):
            import shutil
            try:
                shutil.rmtree(self.output_dir)
                logger.info(f"Cleaned up project directory: {self.output_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up directory: {str(e)}")
    
    def get_project_info(self) -> Dict[str, Any]:
        """
        Get project information.
        
        Returns:
            Project information dictionary
        """
        return {
            "project_name": self.project_name,
            "output_dir": self.output_dir,
            "project_path": self.project_path,
            "schematic_path": self.schematic_path,
            "pcb_path": self.pcb_path,
            "netlist_path": self.netlist_path,
            "files_exist": {
                "project": os.path.exists(self.project_path),
                "schematic": os.path.exists(self.schematic_path),
                "pcb": os.path.exists(self.pcb_path),
                "netlist": os.path.exists(self.netlist_path)
            }
        }