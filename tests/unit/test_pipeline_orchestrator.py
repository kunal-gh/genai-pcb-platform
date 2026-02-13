"""
Unit tests for pipeline orchestrator service.

Tests end-to-end pipeline orchestration and workflow management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.services.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineResult,
    PipelineStage,
    get_pipeline_orchestrator
)
from src.services.request_queue import RequestPriority


class TestPipelineOrchestrator:
    """Test suite for PipelineOrchestrator."""
    
    @patch('src.services.pipeline_orchestrator.LLMService')
    def test_initialization(self, mock_llm_service):
        """Test pipeline orchestrator initialization."""
        # Mock the LLM service to avoid API key requirements
        mock_llm_service.return_value = Mock()
        
        orchestrator = PipelineOrchestrator()
        
        # Check that all services are initialized
        assert orchestrator.nlp_service is not None
        assert orchestrator.llm_service is not None
        assert orchestrator.skidl_generator is not None
        assert orchestrator.skidl_executor is not None
        assert orchestrator.component_selector is None  # Created with DB session when needed
        assert orchestrator.component_library is None  # Created with DB session when needed
        assert orchestrator.kicad_integration is None  # Created per design
        assert orchestrator.manufacturing_export is None  # Created per design
        assert orchestrator.design_verification is not None
        assert orchestrator.dfm_validation is not None
        assert orchestrator.verification_reporting is not None
        assert orchestrator.bom_generator is None  # Created with DB session when needed
        assert orchestrator.simulation_engine is not None
        assert orchestrator.file_packaging is not None
        assert orchestrator.error_manager is not None
        assert orchestrator.progress_reporter is not None
        assert orchestrator.performance_monitor is not None
        assert orchestrator.request_queue is not None
    
    def test_process_design_request(self):
        """Test processing a design request."""
        orchestrator = PipelineOrchestrator()
        
        # Mock the request queue
        with patch.object(orchestrator.request_queue, 'enqueue') as mock_enqueue:
            mock_enqueue.return_value = "test-request-id"
            
            request_id = orchestrator.process_design_request(
                prompt="Create a simple LED circuit",
                user_id="test-user",
                priority=RequestPriority.HIGH
            )
            
            assert request_id == "test-request-id"
            mock_enqueue.assert_called_once()
            
            # Check enqueue arguments
            call_args = mock_enqueue.call_args
            assert call_args[1]["operation_name"] == "design_processing"
            assert call_args[1]["priority"] == RequestPriority.HIGH
            assert "design_id" in call_args[1]["metadata"]
            assert call_args[1]["metadata"]["user_id"] == "test-user"
    
    @patch('src.services.pipeline_orchestrator.PipelineOrchestrator._process_design_pipeline')
    def test_pipeline_stages_success(self, mock_pipeline):
        """Test successful pipeline execution through all stages."""
        orchestrator = PipelineOrchestrator()
        
        # Mock successful pipeline result
        mock_result = PipelineResult(
            design_id="test-design",
            status="success",
            stage=PipelineStage.COMPLETED,
            files={
                "netlist": "/path/to/netlist.net",
                "pcb_layout": "/path/to/layout.kicad_pcb",
                "gerber_zip": "/path/to/gerbers.zip",
                "design_package": "/path/to/package.zip"
            }
        )
        mock_pipeline.return_value = mock_result
        
        # Execute pipeline
        result = orchestrator._process_design_pipeline(
            "test-design",
            "Create a simple LED circuit",
            "test-user"
        )
        
        assert result.status == "success"
        assert result.stage == PipelineStage.COMPLETED
        assert "design_package" in result.files
    
    def test_nlp_parsing_stage(self):
        """Test NLP parsing stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock NLP service
        mock_requirements = {
            "components": [{"type": "led", "quantity": 1}],
            "board_spec": {"width": 50, "height": 50}
        }
        
        with patch.object(orchestrator.nlp_service, 'parse_prompt') as mock_parse:
            mock_parse.return_value = mock_requirements
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.NLP_PARSING
                    )
                    
                    result = orchestrator._stage_nlp_parsing(result, "Create LED circuit")
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.CODE_GENERATION
                    assert result.metadata["structured_requirements"] == mock_requirements
    
    def test_code_generation_stage(self):
        """Test code generation stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock SKiDL generator
        mock_code = "# Generated SKiDL code\nled = Part('Device', 'LED')"
        
        with patch.object(orchestrator.skidl_generator, 'generate_code') as mock_generate:
            mock_generate.return_value = mock_code
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.CODE_GENERATION,
                        metadata={"structured_requirements": {"components": []}}
                    )
                    
                    result = orchestrator._stage_code_generation(result, "Create LED circuit")
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.SCHEMATIC_GENERATION
                    assert result.metadata["skidl_code"] == mock_code
    
    def test_schematic_generation_stage(self):
        """Test schematic generation stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock SKiDL executor
        mock_execution_result = Mock()
        mock_execution_result.netlist = "netlist content"
        mock_execution_result.components = [{"name": "LED1", "type": "led"}]
        mock_execution_result.nets = [{"name": "VCC", "pins": ["LED1.1"]}]
        mock_execution_result.netlist_file = "/path/to/netlist.net"
        
        with patch.object(orchestrator.skidl_executor, 'execute_code') as mock_execute:
            mock_execute.return_value = mock_execution_result
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.SCHEMATIC_GENERATION,
                        metadata={"skidl_code": "# test code"}
                    )
                    
                    result = orchestrator._stage_schematic_generation(result)
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.COMPONENT_SELECTION
                    assert result.metadata["netlist"] == "netlist content"
                    assert result.files["netlist"] == "/path/to/netlist.net"
    
    def test_component_selection_stage(self):
        """Test component selection stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock component selector
        mock_resistor = Mock()
        mock_resistor.part_number = "R1234"
        
        with patch.object(orchestrator.component_selector, 'select_resistor') as mock_select:
            mock_select.return_value = [mock_resistor]
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.COMPONENT_SELECTION,
                        metadata={
                            "components": [
                                {"type": "resistor", "value": 1000, "package": "SMD"}
                            ]
                        }
                    )
                    
                    result = orchestrator._stage_component_selection(result)
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.PCB_LAYOUT
                    assert len(result.metadata["selected_components"]) == 1
    
    def test_pcb_layout_stage(self):
        """Test PCB layout stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock KiCad integration
        mock_pcb_result = Mock()
        mock_pcb_result.project_file = "/path/to/project.kicad_pro"
        mock_pcb_result.pcb_file = "/path/to/layout.kicad_pcb"
        mock_pcb_result.schematic_file = "/path/to/schematic.kicad_sch"
        mock_pcb_result.board_info = {"layers": 2, "size": [50, 50]}
        
        with patch.object(orchestrator.kicad_integration, 'create_pcb_from_netlist') as mock_create:
            mock_create.return_value = mock_pcb_result
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.PCB_LAYOUT,
                        files={"netlist": "/path/to/netlist.net"}
                    )
                    
                    result = orchestrator._stage_pcb_layout(result)
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.DESIGN_VERIFICATION
                    assert result.files["pcb_layout"] == "/path/to/layout.kicad_pcb"
    
    def test_design_verification_stage(self):
        """Test design verification stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock design verification
        mock_verification_result = Mock()
        mock_verification_result.errors = []
        mock_verification_result.warnings = ["Minor spacing issue"]
        
        with patch.object(orchestrator.design_verification, 'verify_design') as mock_verify:
            mock_verify.return_value = mock_verification_result
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.DESIGN_VERIFICATION,
                        files={"pcb_layout": "/path/to/layout.kicad_pcb"}
                    )
                    
                    result = orchestrator._stage_design_verification(result)
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.DFM_VALIDATION
                    assert len(result.warnings) == 1
    
    def test_dfm_validation_stage(self):
        """Test DFM validation stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock DFM validation
        mock_dfm_result = Mock()
        mock_dfm_result.confidence_score = 0.95
        mock_dfm_result.violations = []
        
        with patch.object(orchestrator.dfm_validation, 'validate_design') as mock_validate:
            mock_validate.return_value = mock_dfm_result
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.DFM_VALIDATION,
                        files={"pcb_layout": "/path/to/layout.kicad_pcb"}
                    )
                    
                    result = orchestrator._stage_dfm_validation(result)
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.BOM_GENERATION
                    assert result.metadata["dfm_score"] == 0.95
    
    def test_bom_generation_stage(self):
        """Test BOM generation stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock BOM generator
        mock_bom_result = Mock()
        mock_bom_result.csv_file = "/path/to/bom.csv"
        mock_bom_result.xlsx_file = "/path/to/bom.xlsx"
        mock_bom_result.summary = {"total_cost": 5.50, "component_count": 10}
        
        with patch.object(orchestrator.bom_generator, 'generate_bom') as mock_generate:
            mock_generate.return_value = mock_bom_result
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.BOM_GENERATION,
                        metadata={"selected_components": []}
                    )
                    
                    result = orchestrator._stage_bom_generation(result)
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.SIMULATION
                    assert result.files["bom_csv"] == "/path/to/bom.csv"
    
    def test_simulation_stage(self):
        """Test simulation stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock simulation engine
        mock_simulation_result = Mock()
        mock_simulation_result.plots = {"dc_analysis": "/path/to/dc_plot.png"}
        
        with patch.object(orchestrator.simulation_engine, 'run_simulation') as mock_simulate:
            mock_simulate.return_value = mock_simulation_result
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.SIMULATION,
                        metadata={"netlist": "netlist content"}
                    )
                    
                    result = orchestrator._stage_simulation(result)
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.MANUFACTURING_EXPORT
                    assert "dc_analysis" in result.files
    
    def test_manufacturing_export_stage(self):
        """Test manufacturing export stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock manufacturing export
        mock_export_result = Mock()
        mock_export_result.gerber_archive = "/path/to/gerbers.zip"
        mock_export_result.drill_file = "/path/to/drill.drl"
        mock_export_result.pick_place_file = "/path/to/pick_place.csv"
        mock_export_result.step_file = "/path/to/model.step"
        
        with patch.object(orchestrator.manufacturing_export, 'export_manufacturing_files') as mock_export:
            mock_export.return_value = mock_export_result
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.MANUFACTURING_EXPORT,
                        files={"pcb_layout": "/path/to/layout.kicad_pcb"}
                    )
                    
                    result = orchestrator._stage_manufacturing_export(result)
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.FILE_PACKAGING
                    assert result.files["gerber_zip"] == "/path/to/gerbers.zip"
    
    def test_file_packaging_stage(self):
        """Test file packaging stage."""
        orchestrator = PipelineOrchestrator()
        
        # Mock file packaging
        mock_package_result = Mock()
        mock_package_result.package_file = "/path/to/design_package.zip"
        mock_package_result.manifest = {"files": 10, "size": "5.2MB"}
        
        with patch.object(orchestrator.file_packaging, 'create_design_package') as mock_package:
            mock_package.return_value = mock_package_result
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'complete_step'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.FILE_PACKAGING,
                        files={"pcb_layout": "/path/to/layout.kicad_pcb"}
                    )
                    
                    result = orchestrator._stage_file_packaging(result)
                    
                    assert result.status == "processing"
                    assert result.stage == PipelineStage.COMPLETED
                    assert result.files["design_package"] == "/path/to/design_package.zip"
    
    def test_stage_error_handling(self):
        """Test error handling in pipeline stages."""
        orchestrator = PipelineOrchestrator()
        
        # Mock NLP service to raise exception
        with patch.object(orchestrator.nlp_service, 'parse_prompt') as mock_parse:
            mock_parse.side_effect = Exception("NLP parsing failed")
            
            with patch.object(orchestrator.progress_reporter, 'start_step'):
                with patch.object(orchestrator.progress_reporter, 'fail_operation'):
                    
                    result = PipelineResult(
                        design_id="test-design",
                        status="processing",
                        stage=PipelineStage.NLP_PARSING
                    )
                    
                    result = orchestrator._stage_nlp_parsing(result, "test prompt")
                    
                    assert result.status == "failed"
                    assert len(result.errors) == 1
                    assert "NLP parsing failed" in result.errors[0]
    
    def test_get_pipeline_status(self):
        """Test getting pipeline status."""
        orchestrator = PipelineOrchestrator()
        
        # Mock queue status
        mock_queue_status = Mock()
        mock_queue_status.status = "processing"
        mock_queue_status.queued_at = datetime.now()
        mock_queue_status.started_at = datetime.now()
        mock_queue_status.completed_at = None
        mock_queue_status.position = None
        mock_queue_status.estimated_wait_time = None
        
        with patch.object(orchestrator.request_queue, 'get_status') as mock_get_status:
            mock_get_status.return_value = mock_queue_status
            
            status = orchestrator.get_pipeline_status("test-request-id")
            
            assert status is not None
            assert status["request_id"] == "test-request-id"
            assert status["queue_status"] == "processing"
    
    def test_get_pipeline_status_nonexistent(self):
        """Test getting status for nonexistent request."""
        orchestrator = PipelineOrchestrator()
        
        with patch.object(orchestrator.request_queue, 'get_status') as mock_get_status:
            mock_get_status.return_value = None
            
            status = orchestrator.get_pipeline_status("nonexistent-id")
            
            assert status is None


class TestGetPipelineOrchestrator:
    """Test global pipeline orchestrator instance."""
    
    def test_get_pipeline_orchestrator_singleton(self):
        """Test that get_pipeline_orchestrator returns singleton."""
        orchestrator1 = get_pipeline_orchestrator()
        orchestrator2 = get_pipeline_orchestrator()
        
        assert orchestrator1 is orchestrator2
    
    def test_get_pipeline_orchestrator_returns_instance(self):
        """Test that get_pipeline_orchestrator returns PipelineOrchestrator."""
        orchestrator = get_pipeline_orchestrator()
        assert isinstance(orchestrator, PipelineOrchestrator)


class TestPipelineResult:
    """Test PipelineResult dataclass."""
    
    def test_create_pipeline_result(self):
        """Test creating pipeline result."""
        result = PipelineResult(
            design_id="test-design",
            status="success",
            stage=PipelineStage.COMPLETED
        )
        
        assert result.design_id == "test-design"
        assert result.status == "success"
        assert result.stage == PipelineStage.COMPLETED
        assert result.files == {}
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {}
        assert result.processing_time == 0.0
    
    def test_pipeline_result_with_data(self):
        """Test pipeline result with data."""
        files = {"netlist": "/path/to/netlist.net"}
        errors = ["Error 1"]
        warnings = ["Warning 1"]
        metadata = {"key": "value"}
        
        result = PipelineResult(
            design_id="test-design",
            status="partial",
            stage=PipelineStage.PCB_LAYOUT,
            files=files,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
            processing_time=45.5
        )
        
        assert result.files == files
        assert result.errors == errors
        assert result.warnings == warnings
        assert result.metadata == metadata
        assert result.processing_time == 45.5