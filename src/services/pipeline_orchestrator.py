"""
End-to-end pipeline orchestration service.

Orchestrates the complete workflow from natural language prompt to 
downloadable PCB design files (schematics, netlist, PCB layout, Gerber files).

Requirements: All requirements integration
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .nlp_service import NLPService
from .llm_service import LLMService
from .skidl_generator import SKiDLGenerator
from .skidl_executor import SKiDLExecutor
from .component_selector import ComponentSelector
from .component_library import ComponentLibrary
from .kicad_integration import KiCadProject
from .manufacturing_export import ManufacturingExporter
from .design_verification import DesignVerificationEngine
from .dfm_validation import DFMValidator
from .verification_reporting import VerificationReporter
from .bom_generator import BOMGenerator
from .simulation_engine import SimulationEngine
from .file_packaging import FilePackager
from .error_management import ErrorManager
from .progress_reporting import get_progress_reporter
from .performance_monitoring import get_performance_monitor
from .request_queue import get_request_queue, RequestPriority

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""
    NLP_PARSING = "nlp_parsing"
    CODE_GENERATION = "code_generation"
    SCHEMATIC_GENERATION = "schematic_generation"
    COMPONENT_SELECTION = "component_selection"
    PCB_LAYOUT = "pcb_layout"
    DESIGN_VERIFICATION = "design_verification"
    DFM_VALIDATION = "dfm_validation"
    BOM_GENERATION = "bom_generation"
    SIMULATION = "simulation"
    MANUFACTURING_EXPORT = "manufacturing_export"
    FILE_PACKAGING = "file_packaging"
    COMPLETED = "completed"


@dataclass
class PipelineResult:
    """Result of pipeline processing."""
    design_id: str
    status: str  # success, failed, partial
    stage: PipelineStage
    files: Dict[str, str] = None  # file_type -> file_path
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.files is None:
            self.files = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class PipelineOrchestrator:
    """
    End-to-end pipeline orchestrator.
    
    Coordinates all services to process a natural language prompt
    into a complete PCB design with all deliverables.
    """
    
    def __init__(self):
        """Initialize pipeline orchestrator with all services."""
        # Core services
        self.nlp_service = NLPService()
        self.llm_service = LLMService()
        self.skidl_generator = SKiDLGenerator(self.llm_service)
        self.skidl_executor = SKiDLExecutor()
        
        # Component services (will be created with DB session when needed)
        self.component_selector = None  # Created with DB session when needed
        self.component_library = None  # Created with DB session when needed
        
        # PCB services (will be created per-design)
        self.kicad_integration = None  # Created per design
        self.manufacturing_export = None  # Created per design
        
        # Verification services
        self.design_verification = DesignVerificationEngine()
        self.dfm_validation = DFMValidator()
        self.verification_reporting = VerificationReporter()
        
        # Analysis services
        self.bom_generator = None  # Will be created with DB session when needed
        self.simulation_engine = SimulationEngine()
        
        # Output services
        self.file_packaging = FilePackager()
        
        # Infrastructure services
        self.error_manager = ErrorManager()
        self.progress_reporter = get_progress_reporter()
        self.performance_monitor = get_performance_monitor()
        self.request_queue = get_request_queue()
        
        logger.info("Pipeline orchestrator initialized with all services")
    
    def process_design_request(
        self,
        prompt: str,
        user_id: str = "anonymous",
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> str:
        """
        Process a design request asynchronously.
        
        Args:
            prompt: Natural language design prompt
            user_id: User identifier
            priority: Request priority
            
        Returns:
            Request ID for tracking progress
        """
        design_id = str(uuid.uuid4())
        
        # Enqueue the processing request
        request_id = self.request_queue.enqueue(
            operation_name="design_processing",
            handler=self._process_design_pipeline,
            args=(design_id, prompt, user_id),
            priority=priority,
            metadata={
                "design_id": design_id,
                "user_id": user_id,
                "prompt_length": len(prompt)
            }
        )
        
        logger.info(f"Enqueued design request {design_id} for user {user_id}")
        return request_id
    
    def _process_design_pipeline(
        self,
        design_id: str,
        prompt: str,
        user_id: str
    ) -> PipelineResult:
        """
        Execute the complete design pipeline.
        
        Args:
            design_id: Unique design identifier
            prompt: Natural language prompt
            user_id: User identifier
            
        Returns:
            PipelineResult with all generated files and metadata
        """
        # Start performance monitoring
        self.performance_monitor.start_operation(design_id, "design_pipeline", {
            "user_id": user_id,
            "prompt_length": len(prompt)
        })
        
        # Create progress tracking
        pipeline_steps = [
            {"name": "nlp_parsing", "description": "Parsing natural language prompt", "weight": 1.0},
            {"name": "code_generation", "description": "Generating SKiDL code", "weight": 2.0},
            {"name": "schematic_generation", "description": "Creating schematic", "weight": 2.0},
            {"name": "component_selection", "description": "Selecting components", "weight": 1.5},
            {"name": "pcb_layout", "description": "Generating PCB layout", "weight": 3.0},
            {"name": "design_verification", "description": "Verifying design rules", "weight": 2.0},
            {"name": "dfm_validation", "description": "Validating manufacturability", "weight": 2.0},
            {"name": "bom_generation", "description": "Generating bill of materials", "weight": 1.0},
            {"name": "simulation", "description": "Running circuit simulation", "weight": 2.5},
            {"name": "manufacturing_export", "description": "Exporting manufacturing files", "weight": 2.0},
            {"name": "file_packaging", "description": "Packaging deliverables", "weight": 1.0}
        ]
        
        self.progress_reporter.create_operation(
            design_id,
            "PCB Design Generation",
            steps=pipeline_steps,
            metadata={"user_id": user_id}
        )
        self.progress_reporter.start_operation(design_id, "Starting PCB design pipeline")
        
        result = PipelineResult(
            design_id=design_id,
            status="processing",
            stage=PipelineStage.NLP_PARSING
        )
        
        try:
            # Stage 1: Natural Language Processing
            result = self._stage_nlp_parsing(result, prompt)
            if result.status == "failed":
                return result
            
            # Stage 2: Code Generation
            result = self._stage_code_generation(result, prompt)
            if result.status == "failed":
                return result
            
            # Stage 3: Schematic Generation
            result = self._stage_schematic_generation(result)
            if result.status == "failed":
                return result
            
            # Stage 4: Component Selection
            result = self._stage_component_selection(result)
            if result.status == "failed":
                return result
            
            # Stage 5: PCB Layout
            result = self._stage_pcb_layout(result)
            if result.status == "failed":
                return result
            
            # Stage 6: Design Verification
            result = self._stage_design_verification(result)
            if result.status == "failed":
                return result
            
            # Stage 7: DFM Validation
            result = self._stage_dfm_validation(result)
            if result.status == "failed":
                return result
            
            # Stage 8: BOM Generation
            result = self._stage_bom_generation(result)
            if result.status == "failed":
                return result
            
            # Stage 9: Simulation
            result = self._stage_simulation(result)
            if result.status == "failed":
                return result
            
            # Stage 10: Manufacturing Export
            result = self._stage_manufacturing_export(result)
            if result.status == "failed":
                return result
            
            # Stage 11: File Packaging
            result = self._stage_file_packaging(result)
            if result.status == "failed":
                return result
            
            # Pipeline completed successfully
            result.status = "success"
            result.stage = PipelineStage.COMPLETED
            
            self.progress_reporter.complete_operation(design_id, "PCB design completed successfully")
            self.performance_monitor.end_operation(design_id, "completed")
            
            logger.info(f"Design pipeline completed successfully for {design_id}")
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Pipeline failed with unexpected error: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            
            self.progress_reporter.fail_operation(design_id, error_msg)
            self.performance_monitor.end_operation(design_id, "failed")
            
            logger.error(f"Design pipeline failed for {design_id}: {e}", exc_info=True)
        
        return result
    
    def _stage_nlp_parsing(self, result: PipelineResult, prompt: str) -> PipelineResult:
        """Stage 1: Parse natural language prompt."""
        try:
            self.progress_reporter.start_step(result.design_id, "nlp_parsing", "Parsing natural language prompt")
            
            # Parse the prompt
            structured_requirements = self.nlp_service.parse_prompt(prompt)
            
            # Store parsed requirements
            result.metadata["structured_requirements"] = structured_requirements
            result.stage = PipelineStage.CODE_GENERATION
            
            self.progress_reporter.complete_step(result.design_id, "nlp_parsing", "Prompt parsed successfully")
            logger.info(f"NLP parsing completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"NLP parsing failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"NLP parsing failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_code_generation(self, result: PipelineResult, prompt: str) -> PipelineResult:
        """Stage 2: Generate SKiDL code."""
        try:
            self.progress_reporter.start_step(result.design_id, "code_generation", "Generating SKiDL code")
            
            # Generate SKiDL code
            structured_requirements = result.metadata["structured_requirements"]
            skidl_code = self.skidl_generator.generate_code(structured_requirements)
            
            # Store generated code
            result.metadata["skidl_code"] = skidl_code
            result.stage = PipelineStage.SCHEMATIC_GENERATION
            
            self.progress_reporter.complete_step(result.design_id, "code_generation", "SKiDL code generated")
            logger.info(f"Code generation completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"Code generation failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"Code generation failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_schematic_generation(self, result: PipelineResult) -> PipelineResult:
        """Stage 3: Generate schematic from SKiDL code."""
        try:
            self.progress_reporter.start_step(result.design_id, "schematic_generation", "Creating schematic")
            
            # Execute SKiDL code to generate netlist
            skidl_code = result.metadata["skidl_code"]
            execution_result = self.skidl_executor.execute_code(skidl_code)
            
            # Store execution results
            result.metadata["netlist"] = execution_result.netlist
            result.metadata["components"] = execution_result.components
            result.metadata["nets"] = execution_result.nets
            result.files["netlist"] = execution_result.netlist_file
            result.stage = PipelineStage.COMPONENT_SELECTION
            
            self.progress_reporter.complete_step(result.design_id, "schematic_generation", "Schematic created")
            logger.info(f"Schematic generation completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"Schematic generation failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"Schematic generation failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_component_selection(self, result: PipelineResult) -> PipelineResult:
        """Stage 4: Select and validate components."""
        try:
            self.progress_reporter.start_step(result.design_id, "component_selection", "Selecting components")
            
            # Get components from netlist
            components = result.metadata["components"]
            
            # Create component selector with database session
            from ..models.database import get_db_session
            with get_db_session() as db_session:
                component_selector = ComponentSelector(db_session)
                
                # Select actual components for each requirement
                selected_components = []
                for component in components:
                    # Use component selector to find suitable parts
                    if component.get("type") == "resistor":
                        selected = component_selector.select_resistor(
                            resistance=component.get("value", 1000),
                            package_type=component.get("package", "SMD")
                        )
                    elif component.get("type") == "capacitor":
                        selected = component_selector.select_capacitor(
                            capacitance=component.get("value", 1e-6),
                            voltage_rating=component.get("voltage", 25)
                        )
                    else:
                        # Generic component selection
                        selected = component_selector.select_by_category(
                            category=component.get("type", "generic"),
                            electrical_parameters=component
                        )
                    
                    if selected:
                        selected_components.extend(selected)
                
                # Store selected components
                result.metadata["selected_components"] = selected_components
                result.stage = PipelineStage.PCB_LAYOUT
            
            self.progress_reporter.complete_step(result.design_id, "component_selection", "Components selected")
            logger.info(f"Component selection completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"Component selection failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"Component selection failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_pcb_layout(self, result: PipelineResult) -> PipelineResult:
        """Stage 5: Generate PCB layout."""
        try:
            self.progress_reporter.start_step(result.design_id, "pcb_layout", "Generating PCB layout")
            
            # Create KiCad project and generate PCB
            netlist_file = result.files["netlist"]
            
            # Create a new KiCad project for this design
            kicad_project = KiCadProject(result.design_id)
            
            # Create project files
            project_result = kicad_project.create_project(
                board_width=50.0,  # Default board size
                board_height=50.0,
                layers=2
            )
            
            # Import netlist
            import_result = kicad_project.import_netlist(netlist_file)
            
            # Generate PCB layout
            layout_result = kicad_project.generate_pcb_layout()
            
            # Store PCB files
            result.files["kicad_project"] = project_result["project_path"]
            result.files["pcb_layout"] = project_result["pcb_path"]
            result.files["schematic"] = project_result["schematic_path"]
            result.metadata["pcb_info"] = {
                "board_size": project_result["board_size"],
                "layers": project_result["layers"],
                "components": import_result.get("components", []),
                "nets": import_result.get("nets", [])
            }
            result.stage = PipelineStage.DESIGN_VERIFICATION
            
            self.progress_reporter.complete_step(result.design_id, "pcb_layout", "PCB layout generated")
            logger.info(f"PCB layout completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"PCB layout failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"PCB layout failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_design_verification(self, result: PipelineResult) -> PipelineResult:
        """Stage 6: Verify design rules."""
        try:
            self.progress_reporter.start_step(result.design_id, "design_verification", "Verifying design rules")
            
            # Run ERC and DRC checks
            netlist_data = {
                "components": result.metadata.get("components", []),
                "nets": result.metadata.get("nets", [])
            }
            pcb_data = result.metadata.get("pcb_info", {})
            
            verification_result = self.design_verification.verify_design(netlist_data, pcb_data)
            
            # Store verification results
            result.metadata["verification_result"] = verification_result
            if verification_result.get("violations"):
                errors = [v for v in verification_result["violations"] if v.get("severity") in ["critical", "error"]]
                warnings = [v for v in verification_result["violations"] if v.get("severity") == "warning"]
                result.errors.extend([v["message"] for v in errors])
                result.warnings.extend([v["message"] for v in warnings])
            
            result.stage = PipelineStage.DFM_VALIDATION
            
            self.progress_reporter.complete_step(result.design_id, "design_verification", "Design verified")
            logger.info(f"Design verification completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"Design verification failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"Design verification failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_dfm_validation(self, result: PipelineResult) -> PipelineResult:
        """Stage 7: Validate design for manufacturing."""
        try:
            self.progress_reporter.start_step(result.design_id, "dfm_validation", "Validating manufacturability")
            
            # Run DFM validation
            pcb_file = result.files["pcb_layout"]
            dfm_result = self.dfm_validation.validate_design(pcb_file)
            
            # Store DFM results
            result.metadata["dfm_result"] = dfm_result
            result.metadata["dfm_score"] = dfm_result.get("confidence_score", 0.0)
            
            violations = dfm_result.get("violations", [])
            if violations:
                result.warnings.extend([v.get("description", str(v)) for v in violations])
            
            result.stage = PipelineStage.BOM_GENERATION
            
            self.progress_reporter.complete_step(result.design_id, "dfm_validation", "DFM validation completed")
            logger.info(f"DFM validation completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"DFM validation failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"DFM validation failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_bom_generation(self, result: PipelineResult) -> PipelineResult:
        """Stage 8: Generate bill of materials."""
        try:
            self.progress_reporter.start_step(result.design_id, "bom_generation", "Generating bill of materials")
            
            # Create BOM generator with database session
            from ..models.database import get_db_session
            with get_db_session() as db_session:
                bom_generator = BOMGenerator(db_session)
                
                # Generate BOM from netlist data
                netlist_data = {
                    "components": result.metadata.get("components", []),
                    "nets": result.metadata.get("nets", [])
                }
                bom_result = bom_generator.generate_bom(netlist_data)
                
                # Store BOM
                result.files["bom_csv"] = bom_result.get("csv_file", "")
                result.files["bom_xlsx"] = bom_result.get("xlsx_file", "")
                result.metadata["bom_summary"] = bom_result.get("summary", {})
                result.stage = PipelineStage.SIMULATION
            
            self.progress_reporter.complete_step(result.design_id, "bom_generation", "BOM generated")
            logger.info(f"BOM generation completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"BOM generation failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"BOM generation failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_simulation(self, result: PipelineResult) -> PipelineResult:
        """Stage 9: Run circuit simulation."""
        try:
            self.progress_reporter.start_step(result.design_id, "simulation", "Running circuit simulation")
            
            # Run simulation
            netlist_data = {
                "components": result.metadata.get("components", []),
                "nets": result.metadata.get("nets", [])
            }
            
            # Generate SPICE netlist
            spice_netlist = self.simulation_engine.generate_spice_netlist(netlist_data)
            
            # Run DC analysis
            dc_result = self.simulation_engine.run_dc_analysis(spice_netlist)
            
            # Store simulation results
            simulation_result = {
                "dc_analysis": dc_result,
                "spice_netlist": spice_netlist
            }
            result.metadata["simulation_result"] = simulation_result
            
            # Store any generated plots
            plots = dc_result.get("plots", {})
            if plots:
                result.files.update(plots)
            
            result.stage = PipelineStage.MANUFACTURING_EXPORT
            
            self.progress_reporter.complete_step(result.design_id, "simulation", "Simulation completed")
            logger.info(f"Simulation completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"Simulation failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"Simulation failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_manufacturing_export(self, result: PipelineResult) -> PipelineResult:
        """Stage 10: Export manufacturing files."""
        try:
            self.progress_reporter.start_step(result.design_id, "manufacturing_export", "Exporting manufacturing files")
            
            # Create manufacturing exporter for this design
            import tempfile
            export_dir = tempfile.mkdtemp(prefix=f"manufacturing_{result.design_id}_")
            manufacturing_exporter = ManufacturingExporter(result.design_id, export_dir)
            
            # Get PCB data and components
            pcb_data = result.metadata.get("pcb_info", {})
            selected_components = result.metadata.get("selected_components", [])
            
            # Convert components to ComponentPlacement format
            from .manufacturing_export import ComponentPlacement, DrillHole
            component_placements = []
            for i, comp in enumerate(selected_components[:10]):  # Limit for demo
                component_placements.append(ComponentPlacement(
                    reference=comp.get("reference", f"R{i+1}"),
                    value=comp.get("value", "1k"),
                    package=comp.get("package", "0805"),
                    x=float(10 + i * 5),  # Simple placement
                    y=float(10),
                    rotation=0.0,
                    layer="top"
                ))
            
            # Create sample drill holes
            drill_holes = [
                DrillHole(x=5.0, y=5.0, diameter=0.3),
                DrillHole(x=45.0, y=5.0, diameter=0.3),
                DrillHole(x=5.0, y=45.0, diameter=0.3),
                DrillHole(x=45.0, y=45.0, diameter=0.3)
            ]
            
            # Generate complete manufacturing package
            export_result = manufacturing_exporter.generate_manufacturing_package(
                pcb_data, component_placements, drill_holes
            )
            
            # Store manufacturing files
            if export_result.get("success"):
                results = export_result.get("results", {})
                
                # Gerber files
                gerber_files = results.get("gerbers", {}).get("files", {})
                if gerber_files:
                    result.files["gerber_zip"] = export_dir  # Directory containing all files
                
                # Drill files
                drill_info = results.get("drill", {})
                if drill_info.get("drill_file"):
                    result.files["drill_file"] = drill_info["drill_file"]
                
                # Pick and place
                pnp_info = results.get("pick_and_place", {})
                pnp_files = pnp_info.get("files", {})
                if pnp_files.get("top"):
                    result.files["pick_place"] = pnp_files["top"]
                
                # STEP model
                step_info = results.get("step", {})
                if step_info.get("step_file"):
                    result.files["step_3d"] = step_info["step_file"]
            
            result.stage = PipelineStage.FILE_PACKAGING
            
            self.progress_reporter.complete_step(result.design_id, "manufacturing_export", "Manufacturing files exported")
            logger.info(f"Manufacturing export completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"Manufacturing export failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"Manufacturing export failed for {result.design_id}: {e}")
        
        return result
    
    def _stage_file_packaging(self, result: PipelineResult) -> PipelineResult:
        """Stage 11: Package all deliverables."""
        try:
            self.progress_reporter.start_step(result.design_id, "file_packaging", "Packaging deliverables")
            
            # Package all files
            package_result = self.file_packaging.create_package(
                design_id=result.design_id,
                files=result.files,
                metadata=result.metadata
            )
            
            # Store final package
            result.files["design_package"] = package_result.get("package_path", "")
            result.metadata["package_info"] = package_result.get("manifest", {})
            result.stage = PipelineStage.COMPLETED
            
            self.progress_reporter.complete_step(result.design_id, "file_packaging", "Files packaged")
            logger.info(f"File packaging completed for {result.design_id}")
            
        except Exception as e:
            error_msg = f"File packaging failed: {str(e)}"
            result.status = "failed"
            result.errors.append(error_msg)
            self.progress_reporter.fail_operation(result.design_id, error_msg)
            logger.error(f"File packaging failed for {result.design_id}: {e}")
        
        return result
    
    def get_pipeline_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a pipeline request.
        
        Args:
            request_id: Request ID from process_design_request
            
        Returns:
            Status information including progress and results
        """
        # Get request status from queue
        queue_status = self.request_queue.get_status(request_id)
        if not queue_status:
            return None
        
        status_info = {
            "request_id": request_id,
            "queue_status": queue_status.status,
            "queued_at": queue_status.queued_at.isoformat(),
            "started_at": queue_status.started_at.isoformat() if queue_status.started_at else None,
            "completed_at": queue_status.completed_at.isoformat() if queue_status.completed_at else None,
            "position": queue_status.position,
            "estimated_wait_time": queue_status.estimated_wait_time
        }
        
        # Get progress information if available
        if queue_status.status in ["processing", "completed"]:
            design_id = queue_status.metadata.get("design_id") if hasattr(queue_status, 'metadata') else None
            if design_id:
                progress = self.progress_reporter.get_progress(design_id)
                if progress:
                    status_info["progress"] = progress.to_dict()
        
        # Get result if completed
        if queue_status.status == "completed" and queue_status.result:
            status_info["result"] = {
                "design_id": queue_status.result.design_id,
                "status": queue_status.result.status,
                "stage": queue_status.result.stage.value,
                "files": queue_status.result.files,
                "errors": queue_status.result.errors,
                "warnings": queue_status.result.warnings,
                "processing_time": queue_status.result.processing_time
            }
        
        return status_info


# Global pipeline orchestrator instance
_pipeline_orchestrator: Optional[PipelineOrchestrator] = None


def get_pipeline_orchestrator() -> PipelineOrchestrator:
    """Get the global pipeline orchestrator instance."""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is None:
        _pipeline_orchestrator = PipelineOrchestrator()
    return _pipeline_orchestrator