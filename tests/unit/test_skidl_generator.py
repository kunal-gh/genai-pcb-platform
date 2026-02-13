"""
Unit tests for SKiDL Generator.

Tests SKiDL code generation, validation, and error recovery.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.services.skidl_generator import (
    SKiDLGenerator,
    SKiDLGenerationResult,
    SKiDLGenerationError
)
from src.services.nlp_service import (
    StructuredRequirements,
    BoardSpecification,
    PowerSpecification,
    ComponentRequirement,
    DesignConstraints
)
from src.services.llm_service import LLMResponse


@pytest.fixture
def sample_requirements():
    """Sample structured requirements for testing."""
    return StructuredRequirements(
        board=BoardSpecification(width_mm=50.0, height_mm=30.0, layers=1),
        power=PowerSpecification(type="battery", voltage=9.0),
        components=[
            ComponentRequirement(type="LED", value="red", package="0805"),
            ComponentRequirement(type="RESISTOR", value="330", package="0805")
        ],
        constraints=DesignConstraints(priority="compact"),
        original_prompt="Design a simple LED circuit"
    )


@pytest.fixture
def valid_skidl_code():
    """Valid SKiDL code for testing."""
    return """from skidl import Part, Net, generate_netlist

# Create components
led = Part('Device', 'LED', footprint='LED_SMD:LED_0805_2012Metric')
resistor = Part('Device', 'R', value='330', footprint='Resistor_SMD:R_0805_2012Metric')

# Create nets
vcc = Net('VCC')
gnd = Net('GND')
led_anode = Net('LED_ANODE')

# Connect components
vcc += resistor[1]
resistor[2] += led_anode
led_anode += led['A']
led['K'] += gnd

# Generate netlist
generate_netlist()
"""


@pytest.fixture
def invalid_skidl_code():
    """Invalid SKiDL code for testing."""
    return """from skidl import Part

# Syntax error - missing closing parenthesis
led = Part('Device', 'LED'
"""


@pytest.fixture
def mock_llm_response(valid_skidl_code):
    """Mock LLM response."""
    return LLMResponse(
        content=valid_skidl_code,
        provider="openai",
        model="gpt-4o",
        tokens_used=250,
        latency_ms=500,
        finish_reason="stop"
    )


class TestSKiDLGeneratorInitialization:
    """Test SKiDL generator initialization."""
    
    def test_init_with_default_llm(self):
        """Test initialization with default LLM service."""
        with patch('src.services.skidl_generator.LLMService'):
            generator = SKiDLGenerator()
            
            assert generator.max_retries == 3
            assert generator.llm_service is not None
    
    def test_init_with_custom_llm(self):
        """Test initialization with custom LLM service."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        
        assert generator.llm_service == mock_llm
        assert generator.max_retries == 3
    
    def test_init_with_custom_retries(self):
        """Test initialization with custom retry count."""
        with patch('src.services.skidl_generator.LLMService'):
            generator = SKiDLGenerator(max_retries=5)
            
            assert generator.max_retries == 5


class TestCodeGeneration:
    """Test SKiDL code generation."""
    
    def test_successful_generation(self, sample_requirements, mock_llm_response):
        """Test successful code generation on first attempt."""
        mock_llm = Mock()
        mock_llm.generate_skidl_code.return_value = mock_llm_response
        
        generator = SKiDLGenerator(llm_service=mock_llm)
        result = generator.generate(sample_requirements)
        
        assert result.success is True
        assert result.code == mock_llm_response.content
        assert len(result.validation_errors) == 0
        assert result.metadata["attempts"] == 1
        assert result.metadata["provider"] == "openai"
        assert result.metadata["tokens_used"] == 250
    
    def test_generation_with_design_context(self, sample_requirements, mock_llm_response):
        """Test generation with additional design context."""
        mock_llm = Mock()
        mock_llm.generate_skidl_code.return_value = mock_llm_response
        
        generator = SKiDLGenerator(llm_service=mock_llm)
        result = generator.generate(
            sample_requirements,
            design_context="This is a simple LED blinker"
        )
        
        assert result.success is True
        # Verify design context was passed to LLM
        call_args = mock_llm.generate_skidl_code.call_args[0][0]
        assert call_args.design_context == "This is a simple LED blinker"
    
    def test_generation_with_retry(self, sample_requirements, invalid_skidl_code, valid_skidl_code):
        """Test generation with retry on validation failure."""
        mock_llm = Mock()
        
        # First attempt returns invalid code, second returns valid
        invalid_response = LLMResponse(
            content=invalid_skidl_code,
            provider="openai",
            model="gpt-4o",
            tokens_used=200,
            latency_ms=400,
            finish_reason="stop"
        )
        
        valid_response = LLMResponse(
            content=valid_skidl_code,
            provider="openai",
            model="gpt-4o",
            tokens_used=250,
            latency_ms=500,
            finish_reason="stop"
        )
        
        mock_llm.generate_skidl_code.side_effect = [invalid_response, valid_response]
        
        generator = SKiDLGenerator(llm_service=mock_llm, max_retries=3)
        result = generator.generate(sample_requirements)
        
        assert result.success is True
        assert result.metadata["attempts"] == 2
        assert mock_llm.generate_skidl_code.call_count == 2
    
    def test_generation_max_retries_exceeded(self, sample_requirements, invalid_skidl_code):
        """Test generation failure after max retries."""
        mock_llm = Mock()
        
        invalid_response = LLMResponse(
            content=invalid_skidl_code,
            provider="openai",
            model="gpt-4o",
            tokens_used=200,
            latency_ms=400,
            finish_reason="stop"
        )
        
        # Always return invalid code
        mock_llm.generate_skidl_code.return_value = invalid_response
        
        generator = SKiDLGenerator(llm_service=mock_llm, max_retries=2)
        result = generator.generate(sample_requirements)
        
        assert result.success is False
        assert len(result.validation_errors) > 0
        assert result.metadata["attempts"] == 2
        assert mock_llm.generate_skidl_code.call_count == 2


class TestCodeValidation:
    """Test SKiDL code validation."""
    
    def test_validate_valid_code(self, valid_skidl_code):
        """Test validation of valid SKiDL code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        result = generator._validate_code(valid_skidl_code)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_empty_code(self):
        """Test validation fails on empty code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        result = generator._validate_code("")
        
        assert result["valid"] is False
        assert "empty" in result["errors"][0].lower()
    
    def test_validate_syntax_error(self, invalid_skidl_code):
        """Test validation fails on syntax error."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        result = generator._validate_code(invalid_skidl_code)
        
        assert result["valid"] is False
        assert any("syntax error" in err.lower() for err in result["errors"])
    
    def test_validate_missing_part_instantiation(self):
        """Test validation fails without Part() calls."""
        code = """from skidl import Net

# No Part() instantiation
vcc = Net('VCC')
"""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        result = generator._validate_code(code)
        
        assert result["valid"] is False
        assert any("component instantiation" in err.lower() for err in result["errors"])
    
    def test_validate_warnings_for_missing_imports(self):
        """Test warnings for missing recommended imports."""
        code = """# Missing imports
led = Part('Device', 'LED')
"""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        result = generator._validate_code(code)
        
        # Should have warnings but might still be valid
        assert len(result["warnings"]) > 0
    
    def test_validate_warning_for_missing_generate_netlist(self):
        """Test warning for missing generate_netlist() call."""
        code = """from skidl import Part, Net

led = Part('Device', 'LED')
"""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        result = generator._validate_code(code)
        
        assert any("generate_netlist" in warn.lower() for warn in result["warnings"])


class TestSyntaxValidation:
    """Test syntax-only validation."""
    
    def test_syntax_valid(self, valid_skidl_code):
        """Test syntax validation on valid code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        assert generator.validate_syntax_only(valid_skidl_code) is True
    
    def test_syntax_invalid(self, invalid_skidl_code):
        """Test syntax validation on invalid code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        assert generator.validate_syntax_only(invalid_skidl_code) is False
    
    def test_syntax_empty(self):
        """Test syntax validation on empty code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        # Empty code is technically valid Python
        assert generator.validate_syntax_only("") is True


class TestDocumentation:
    """Test code documentation."""
    
    def test_add_documentation(self, valid_skidl_code, sample_requirements):
        """Test adding documentation to code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        documented = generator.add_documentation(valid_skidl_code, sample_requirements)
        
        assert '"""' in documented
        assert "SKiDL PCB Design" in documented
        assert "50.0x30.0mm" in documented
        assert "battery" in documented
        assert "9.0V" in documented
        assert valid_skidl_code in documented
    
    def test_documentation_includes_requirements(self, valid_skidl_code, sample_requirements):
        """Test documentation includes all requirement details."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        documented = generator.add_documentation(valid_skidl_code, sample_requirements)
        
        assert str(sample_requirements.board.width_mm) in documented
        assert str(sample_requirements.board.height_mm) in documented
        assert str(sample_requirements.board.layers) in documented
        assert sample_requirements.power.type in documented
        assert sample_requirements.constraints.priority in documented


class TestComponentExtraction:
    """Test component extraction from code."""
    
    def test_extract_components(self, valid_skidl_code):
        """Test extracting components from SKiDL code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        components = generator.extract_components(valid_skidl_code)
        
        assert len(components) == 2
        
        # Check LED component
        led = next((c for c in components if c["variable"] == "led"), None)
        assert led is not None
        assert led["library"] == "Device"
        assert led["type"] == "LED"
        
        # Check resistor component
        resistor = next((c for c in components if c["variable"] == "resistor"), None)
        assert resistor is not None
        assert resistor["library"] == "Device"
        assert resistor["type"] == "R"
    
    def test_extract_components_empty_code(self):
        """Test extracting components from empty code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        components = generator.extract_components("")
        
        assert len(components) == 0
    
    def test_extract_components_no_parts(self):
        """Test extracting components from code without Part() calls."""
        code = """from skidl import Net

vcc = Net('VCC')
"""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        components = generator.extract_components(code)
        
        assert len(components) == 0


class TestNetExtraction:
    """Test net extraction from code."""
    
    def test_extract_nets(self, valid_skidl_code):
        """Test extracting nets from SKiDL code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        nets = generator.extract_nets(valid_skidl_code)
        
        assert len(nets) == 3
        assert "VCC" in nets
        assert "GND" in nets
        assert "LED_ANODE" in nets
    
    def test_extract_nets_empty_code(self):
        """Test extracting nets from empty code."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        nets = generator.extract_nets("")
        
        assert len(nets) == 0
    
    def test_extract_nets_no_nets(self):
        """Test extracting nets from code without Net() calls."""
        code = """from skidl import Part

led = Part('Device', 'LED')
"""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        nets = generator.extract_nets(code)
        
        assert len(nets) == 0


class TestErrorFormatting:
    """Test error message formatting."""
    
    def test_format_validation_errors(self):
        """Test formatting validation errors for LLM feedback."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        errors = [
            "Python syntax error at line 5",
            "No component instantiation found"
        ]
        
        formatted = generator._format_validation_errors(errors)
        
        assert "validation errors" in formatted.lower()
        assert "1. Python syntax error" in formatted
        assert "2. No component instantiation" in formatted
        assert "fix these errors" in formatted.lower()
    
    def test_format_empty_errors(self):
        """Test formatting empty error list."""
        mock_llm = Mock()
        generator = SKiDLGenerator(llm_service=mock_llm)
        formatted = generator._format_validation_errors([])
        
        assert formatted == ""


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_generation_with_llm_error(self, sample_requirements):
        """Test handling LLM service errors."""
        mock_llm = Mock()
        mock_llm.generate_skidl_code.side_effect = Exception("LLM API error")
        
        generator = SKiDLGenerator(llm_service=mock_llm, max_retries=2)
        result = generator.generate(sample_requirements)
        
        assert result.success is False
        assert len(result.validation_errors) > 0
        assert mock_llm.generate_skidl_code.call_count == 2
    
    def test_generation_with_minimal_requirements(self):
        """Test generation with minimal requirements."""
        minimal_reqs = StructuredRequirements(
            board=BoardSpecification(),
            power=PowerSpecification(),
            components=[ComponentRequirement(type="LED")],
            constraints=DesignConstraints()
        )
        
        mock_llm = Mock()
        mock_llm.generate_skidl_code.return_value = LLMResponse(
            content="from skidl import Part\nled = Part('Device', 'LED')",
            provider="openai",
            model="gpt-4o",
            tokens_used=50,
            latency_ms=200,
            finish_reason="stop"
        )
        
        generator = SKiDLGenerator(llm_service=mock_llm)
        result = generator.generate(minimal_reqs)
        
        # Should succeed even with minimal requirements
        assert result.success is True
