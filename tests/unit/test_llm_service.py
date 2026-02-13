"""
Unit tests for LLM Service.

Tests LLM integration, prompt templates, retry logic, and response validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Mock the availability flags before importing the service
with patch('src.services.llm_service.OPENAI_AVAILABLE', True):
    with patch('src.services.llm_service.ANTHROPIC_AVAILABLE', True):
        from src.services.llm_service import (
            LLMService,
            LLMResponse,
            CodeGenerationRequest,
            LLMServiceError,
            LLMAPIError,
            LLMValidationError
        )


# Patch at module level for all tests
pytestmark = [
    pytest.mark.usefixtures("mock_llm_packages")
]


@pytest.fixture(autouse=True)
def mock_llm_packages():
    """Mock LLM packages for all tests."""
    with patch('src.services.llm_service.OPENAI_AVAILABLE', True):
        with patch('src.services.llm_service.ANTHROPIC_AVAILABLE', True):
            with patch('src.services.llm_service.openai') as mock_openai:
                with patch('src.services.llm_service.Anthropic') as mock_anthropic:
                    # Setup mock OpenAI
                    mock_openai.OpenAI = Mock(return_value=Mock())
                    mock_openai.APIError = type('APIError', (Exception,), {})
                    
                    # Setup mock Anthropic
                    mock_anthropic.return_value = Mock()
                    
                    yield {
                        'openai': mock_openai,
                        'anthropic': mock_anthropic
                    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = """```python
from skidl import Part, Net, generate_netlist

# Create LED
led = Part('Device', 'LED', footprint='LED_SMD:LED_0805_2012Metric')

# Create resistor
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
```"""
    response.choices[0].finish_reason = "stop"
    response.usage = Mock()
    response.usage.total_tokens = 250
    return response


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    response = Mock()
    response.content = [Mock()]
    response.content[0].text = """```python
from skidl import Part, Net, generate_netlist

# Simple LED circuit
led = Part('Device', 'LED')
resistor = Part('Device', 'R', value='330')

vcc = Net('VCC')
gnd = Net('GND')

vcc += resistor[1]
resistor[2] += led['A']
led['K'] += gnd

generate_netlist()
```"""
    response.stop_reason = "end_turn"
    response.usage = Mock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 150
    return response


@pytest.fixture
def sample_requirements():
    """Sample structured requirements for testing."""
    return {
        "board": {
            "width_mm": 50.0,
            "height_mm": 30.0,
            "layers": 1
        },
        "power": {
            "type": "battery",
            "voltage": 9.0
        },
        "components": [
            {
                "type": "LED",
                "value": "red",
                "package": "0805",
                "quantity": 1
            },
            {
                "type": "RESISTOR",
                "value": "330",
                "package": "0805",
                "quantity": 1
            }
        ],
        "connections": [
            ("VCC", "R1"),
            ("R1", "LED1"),
            ("LED1", "GND")
        ],
        "constraints": {
            "priority": "compact"
        }
    }


class TestLLMServiceInitialization:
    """Test LLM service initialization."""
    
    @patch('src.services.llm_service.settings')
    def test_init_openai_provider(self, mock_settings):
        """Test initialization with OpenAI provider."""
        mock_settings.OPENAI_API_KEY = "test-key"
        
        service = LLMService(provider="openai")
        
        assert service.provider == "openai"
        assert service.model == "gpt-4o"
        assert service.temperature == 0.2
        assert service.max_tokens == 4000
    
    @patch('src.services.llm_service.settings')
    def test_init_anthropic_provider(self, mock_settings):
        """Test initialization with Anthropic provider."""
        mock_settings.ANTHROPIC_API_KEY = "test-key"
        
        service = LLMService(provider="anthropic")
        
        assert service.provider == "anthropic"
        assert service.model == "claude-3-5-sonnet-20241022"
    
    @patch('src.services.llm_service.settings')
    def test_init_custom_model(self, mock_settings):
        """Test initialization with custom model."""
        mock_settings.OPENAI_API_KEY = "test-key"
        
        service = LLMService(provider="openai", model="gpt-4-turbo")
        
        assert service.model == "gpt-4-turbo"
    
    @patch('src.services.llm_service.settings')
    def test_init_custom_parameters(self, mock_settings):
        """Test initialization with custom parameters."""
        mock_settings.OPENAI_API_KEY = "test-key"
        
        service = LLMService(
            provider="openai",
            temperature=0.5,
            max_tokens=2000,
            timeout=30
        )
        
        assert service.temperature == 0.5
        assert service.max_tokens == 2000
        assert service.timeout == 30
    
    @patch('src.services.llm_service.settings')
    def test_init_missing_api_key(self, mock_settings):
        """Test initialization fails without API key."""
        mock_settings.OPENAI_API_KEY = None
        
        with pytest.raises(LLMServiceError, match="OPENAI_API_KEY not configured"):
            LLMService(provider="openai")


class TestPromptBuilding:
    """Test prompt template building."""
    
    @patch('src.services.llm_service.settings')
    def test_build_basic_prompt(self, mock_settings, sample_requirements):
        """Test building basic SKiDL prompt."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        request = CodeGenerationRequest(
            structured_requirements=sample_requirements
        )
        
        prompt = service._build_skidl_prompt(request)
        
        # Check prompt contains key sections
        assert "SKiDL" in prompt
        assert "Board Specifications" in prompt
        assert "50mm x 30mm" in prompt
        assert "Power Supply" in prompt
        assert "9.0V" in prompt
        assert "Components" in prompt
        assert "LED" in prompt
        assert "RESISTOR" in prompt
        assert "Connections" in prompt
        assert "Design Constraints" in prompt
    
    @patch('src.services.llm_service.settings')
    def test_build_prompt_with_previous_attempts(self, mock_settings, sample_requirements):
        """Test prompt includes previous attempts."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        request = CodeGenerationRequest(
            structured_requirements=sample_requirements,
            previous_attempts=["# Previous code attempt 1", "# Previous code attempt 2"]
        )
        
        prompt = service._build_skidl_prompt(request)
        
        assert "Previous Attempts" in prompt
        assert "Previous code attempt" in prompt
    
    @patch('src.services.llm_service.settings')
    def test_build_prompt_with_error_feedback(self, mock_settings, sample_requirements):
        """Test prompt includes error feedback."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        request = CodeGenerationRequest(
            structured_requirements=sample_requirements,
            error_feedback="SyntaxError: invalid syntax on line 10"
        )
        
        prompt = service._build_skidl_prompt(request)
        
        assert "Error Feedback" in prompt
        assert "SyntaxError" in prompt
    
    @patch('src.services.llm_service.settings')
    def test_build_prompt_with_design_context(self, mock_settings, sample_requirements):
        """Test prompt includes design context."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        request = CodeGenerationRequest(
            structured_requirements=sample_requirements,
            design_context="This is a simple LED blinker circuit"
        )
        
        prompt = service._build_skidl_prompt(request)
        
        assert "Additional Context" in prompt
        assert "LED blinker" in prompt


class TestCodeGeneration:
    """Test SKiDL code generation."""
    
    @patch('src.services.llm_service.settings')
    def test_generate_skidl_code_openai(
        self, mock_settings, sample_requirements, mock_openai_response
    ):
        """Test successful code generation with OpenAI."""
        mock_settings.OPENAI_API_KEY = "test-key"
        
        service = LLMService(provider="openai")
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        service.client = mock_client
        
        request = CodeGenerationRequest(
            structured_requirements=sample_requirements
        )
        
        response = service.generate_skidl_code(request)
        
        assert isinstance(response, LLMResponse)
        assert response.provider == "openai"
        assert response.model == "gpt-4o"
        assert response.tokens_used == 250
        assert "from skidl import" in response.content
        assert "Part" in response.content
        assert "generate_netlist" in response.content
        assert response.latency_ms > 0
    
    @patch('src.services.llm_service.settings')
    def test_generate_skidl_code_anthropic(
        self, mock_settings, sample_requirements, mock_anthropic_response
    ):
        """Test successful code generation with Anthropic."""
        mock_settings.ANTHROPIC_API_KEY = "test-key"
        
        service = LLMService(provider="anthropic")
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        service.client = mock_client
        
        request = CodeGenerationRequest(
            structured_requirements=sample_requirements
        )
        
        response = service.generate_skidl_code(request)
        
        assert isinstance(response, LLMResponse)
        assert response.provider == "anthropic"
        assert response.tokens_used == 250
        assert "from skidl import" in response.content
    
    @patch('src.services.llm_service.settings')
    def test_generate_code_with_retry(
        self, mock_settings, sample_requirements, mock_openai_response
    ):
        """Test retry logic on API failure."""
        mock_settings.OPENAI_API_KEY = "test-key"
        
        service = LLMService(provider="openai")
        mock_client = Mock()
        
        # Create a mock APIError
        api_error = Exception("Temporary error")
        api_error.__class__.__name__ = "APIError"
        
        # Fail twice, then succeed
        mock_client.chat.completions.create.side_effect = [
            api_error,
            api_error,
            mock_openai_response
        ]
        service.client = mock_client
        
        request = CodeGenerationRequest(
            structured_requirements=sample_requirements
        )
        
        response = service.generate_skidl_code(request)
        
        # Should succeed after retries
        assert isinstance(response, LLMResponse)
        assert mock_client.chat.completions.create.call_count == 3
    
    @patch('src.services.llm_service.settings')
    def test_generate_code_max_retries_exceeded(
        self, mock_settings, sample_requirements
    ):
        """Test failure after max retries."""
        mock_settings.OPENAI_API_KEY = "test-key"
        
        service = LLMService(provider="openai")
        mock_client = Mock()
        
        # Create a mock APIError
        api_error = Exception("Persistent error")
        api_error.__class__.__name__ = "APIError"
        
        # Always fail
        mock_client.chat.completions.create.side_effect = api_error
        service.client = mock_client
        
        request = CodeGenerationRequest(
            structured_requirements=sample_requirements
        )
        
        with pytest.raises(LLMAPIError):
            service.generate_skidl_code(request)


class TestResponseValidation:
    """Test LLM response validation."""
    
    @patch('src.services.llm_service.settings')
    def test_validate_valid_response(self, mock_settings):
        """Test validation of valid SKiDL response."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        response = LLMResponse(
            content="""```python
from skidl import Part, Net, generate_netlist

led = Part('Device', 'LED')
generate_netlist()
```""",
            provider="openai",
            model="gpt-4o",
            tokens_used=100,
            latency_ms=500,
            finish_reason="stop"
        )
        
        # Should not raise
        service._validate_skidl_response(response)
        
        # Should extract code from markdown
        assert "```python" not in response.content
        assert "from skidl import" in response.content
    
    @patch('src.services.llm_service.settings')
    def test_validate_response_without_markdown(self, mock_settings):
        """Test validation of response without markdown."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        response = LLMResponse(
            content="""from skidl import Part, Net, generate_netlist

led = Part('Device', 'LED')
generate_netlist()""",
            provider="openai",
            model="gpt-4o",
            tokens_used=100,
            latency_ms=500,
            finish_reason="stop"
        )
        
        # Should not raise
        service._validate_skidl_response(response)
    
    @patch('src.services.llm_service.settings')
    def test_validate_empty_response(self, mock_settings):
        """Test validation fails on empty response."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        response = LLMResponse(
            content="",
            provider="openai",
            model="gpt-4o",
            tokens_used=0,
            latency_ms=500,
            finish_reason="stop"
        )
        
        with pytest.raises(LLMValidationError, match="Empty response"):
            service._validate_skidl_response(response)
    
    @patch('src.services.llm_service.settings')
    def test_validate_non_code_response(self, mock_settings):
        """Test validation fails on non-code response."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        response = LLMResponse(
            content="I cannot generate code for this request.",
            provider="openai",
            model="gpt-4o",
            tokens_used=20,
            latency_ms=500,
            finish_reason="stop"
        )
        
        with pytest.raises(LLMValidationError, match="does not appear to contain Python code"):
            service._validate_skidl_response(response)
    
    @patch('src.services.llm_service.settings')
    def test_validate_malformed_code_block(self, mock_settings):
        """Test validation fails on malformed code block."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        response = LLMResponse(
            content="```python\nfrom skidl import Part\n# Missing closing backticks",
            provider="openai",
            model="gpt-4o",
            tokens_used=50,
            latency_ms=500,
            finish_reason="stop"
        )
        
        with pytest.raises(LLMValidationError, match="Malformed code block"):
            service._validate_skidl_response(response)


class TestRequirementsFormatting:
    """Test requirements formatting for prompts."""
    
    @patch('src.services.llm_service.settings')
    def test_format_complete_requirements(self, mock_settings, sample_requirements):
        """Test formatting complete requirements."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        formatted = service._format_requirements(sample_requirements)
        
        assert "Board Specifications" in formatted
        assert "50mm x 30mm" in formatted
        assert "Layers: 1" in formatted
        assert "Power Supply" in formatted
        assert "battery" in formatted
        assert "9.0V" in formatted
        assert "Components" in formatted
        assert "LED" in formatted
        assert "RESISTOR" in formatted
        assert "Connections" in formatted
        assert "Design Constraints" in formatted
        assert "compact" in formatted
    
    @patch('src.services.llm_service.settings')
    def test_format_minimal_requirements(self, mock_settings):
        """Test formatting minimal requirements."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        minimal_reqs = {
            "components": [
                {"type": "LED"}
            ]
        }
        
        formatted = service._format_requirements(minimal_reqs)
        
        assert "Components" in formatted
        assert "LED" in formatted
    
    @patch('src.services.llm_service.settings')
    def test_format_multiple_components(self, mock_settings):
        """Test formatting multiple components with quantities."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        reqs = {
            "components": [
                {
                    "type": "RESISTOR",
                    "value": "10k",
                    "package": "0805",
                    "quantity": 3,
                    "description": "Pull-up resistors"
                }
            ]
        }
        
        formatted = service._format_requirements(reqs)
        
        assert "RESISTOR" in formatted
        assert "10k" in formatted
        assert "0805" in formatted
        assert "x3" in formatted
        assert "Pull-up resistors" in formatted


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('src.services.llm_service.settings')
    def test_unsupported_provider(self, mock_settings, sample_requirements):
        """Test error on unsupported provider."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        service.provider = "unsupported"
        
        request = CodeGenerationRequest(
            structured_requirements=sample_requirements
        )
        
        with pytest.raises(LLMServiceError, match="Unsupported provider"):
            service.generate_skidl_code(request)
    
    @patch('src.services.llm_service.settings')
    def test_empty_requirements(self, mock_settings):
        """Test handling empty requirements."""
        mock_settings.OPENAI_API_KEY = "test-key"
        service = LLMService(provider="openai")
        
        request = CodeGenerationRequest(
            structured_requirements={}
        )
        
        prompt = service._build_skidl_prompt(request)
        
        # Should still generate valid prompt
        assert "SKiDL" in prompt
        assert "Requirements" in prompt
