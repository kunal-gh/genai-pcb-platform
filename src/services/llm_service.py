"""
LLM Service Integration.

Provides integration with OpenAI and Anthropic APIs for SKiDL code generation
with retry logic, prompt templates, and response validation.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Literal, TYPE_CHECKING
from dataclasses import dataclass

try:
    import openai
    from anthropic import Anthropic
    OPENAI_AVAILABLE = True
    ANTHROPIC_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False
    # Create mock classes for type checking
    if TYPE_CHECKING:
        import openai
        from anthropic import Anthropic

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from src.config import settings

logger = logging.getLogger(__name__)


# LLM Provider Types
LLMProvider = Literal["openai", "anthropic", "local"]


@dataclass
class LLMResponse:
    """Response from LLM service."""
    content: str
    provider: str
    model: str
    tokens_used: int
    latency_ms: float
    finish_reason: str
    confidence_score: float = 1.0


@dataclass
class CodeGenerationRequest:
    """Request for SKiDL code generation."""
    structured_requirements: Dict[str, Any]
    design_context: Optional[str] = None
    previous_attempts: List[str] = None
    error_feedback: Optional[str] = None
    
    def __post_init__(self):
        if self.previous_attempts is None:
            self.previous_attempts = []


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""
    pass


class LLMAPIError(LLMServiceError):
    """Exception for LLM API errors."""
    pass


class LLMValidationError(LLMServiceError):
    """Exception for LLM response validation errors."""
    pass


class LLMService:
    """
    LLM Service for SKiDL code generation.
    
    Supports multiple LLM providers (OpenAI, Anthropic) with automatic
    retry logic, prompt templates, and response validation.
    """
    
    def __init__(
        self,
        provider: LLMProvider = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        timeout: int = 60
    ):
        """
        Initialize LLM service.
        
        Args:
            provider: LLM provider to use (openai, anthropic, local)
            model: Specific model to use (defaults to provider's best model)
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Set default models
        if model is None:
            if provider == "openai":
                self.model = "gpt-4o"
            elif provider == "anthropic":
                self.model = "claude-3-5-sonnet-20241022"
            else:
                self.model = "llama-3-70b"
        else:
            self.model = model
        
        # Initialize clients
        self._init_clients()
        
        logger.info(
            f"Initialized LLM service: provider={provider}, "
            f"model={self.model}, temperature={temperature}"
        )
    
    def _init_clients(self):
        """Initialize LLM provider clients."""
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise LLMServiceError("openai package not installed. Install with: pip install openai")
            if not settings.OPENAI_API_KEY:
                raise LLMServiceError("OPENAI_API_KEY not configured")
            openai.api_key = settings.OPENAI_API_KEY
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise LLMServiceError("anthropic package not installed. Install with: pip install anthropic")
            if not settings.ANTHROPIC_API_KEY:
                raise LLMServiceError("ANTHROPIC_API_KEY not configured")
            self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        else:
            # Local model support (future implementation)
            logger.warning(f"Local model support not yet implemented")
            self.client = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMAPIError, Exception)),
        reraise=True
    )
    def generate_skidl_code(
        self,
        request: CodeGenerationRequest
    ) -> LLMResponse:
        """
        Generate SKiDL code from structured requirements.
        
        Args:
            request: Code generation request with requirements
            
        Returns:
            LLMResponse with generated SKiDL code
            
        Raises:
            LLMAPIError: If API call fails after retries
            LLMValidationError: If response validation fails
        """
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self._build_skidl_prompt(request)
            
            # Call LLM
            if self.provider == "openai":
                response = self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                raise LLMServiceError(f"Unsupported provider: {self.provider}")
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            response.latency_ms = latency_ms
            
            # Validate response
            self._validate_skidl_response(response)
            
            logger.info(
                f"Generated SKiDL code: provider={self.provider}, "
                f"model={self.model}, tokens={response.tokens_used}, "
                f"latency={latency_ms:.0f}ms"
            )
            
            return response
            
        except Exception as e:
            # Check if it's an OpenAI API error
            if OPENAI_AVAILABLE and isinstance(e, openai.APIError):
                logger.error(f"OpenAI API error: {e}")
                raise LLMAPIError(f"OpenAI API error: {e}") from e
            logger.error(f"LLM service error: {e}")
            raise LLMServiceError(f"LLM service error: {e}") from e
    
    def _build_skidl_prompt(self, request: CodeGenerationRequest) -> str:
        """
        Build prompt for SKiDL code generation.
        
        Args:
            request: Code generation request
            
        Returns:
            Formatted prompt string
        """
        requirements = request.structured_requirements
        
        prompt = f"""You are an expert PCB design engineer specializing in SKiDL (Python-based schematic capture).

Generate complete, valid SKiDL Python code for the following PCB design requirements:

## Requirements
{self._format_requirements(requirements)}

## Instructions
1. Import necessary SKiDL modules (skidl, Part, Net, generate_netlist)
2. Create all required components with proper part numbers and values
3. Define all nets and connections
4. Add comprehensive comments explaining the design logic
5. Include design rule annotations where appropriate
6. Call generate_netlist() at the end
7. Ensure all syntax is valid Python and SKiDL

## Code Quality Requirements
- Use descriptive variable names (e.g., led_red, resistor_current_limit)
- Add comments for each major section
- Include component datasheets references in comments where applicable
- Follow Python PEP 8 style guidelines
- Ensure all connections are explicit and clear

"""
        
        # Add context from previous attempts if available
        if request.previous_attempts:
            prompt += "\n## Previous Attempts\n"
            prompt += "The following code had errors. Please fix them:\n\n"
            for i, attempt in enumerate(request.previous_attempts[-2:], 1):
                prompt += f"### Attempt {i}\n```python\n{attempt}\n```\n\n"
        
        if request.error_feedback:
            prompt += f"\n## Error Feedback\n{request.error_feedback}\n\n"
        
        if request.design_context:
            prompt += f"\n## Additional Context\n{request.design_context}\n\n"
        
        prompt += "\nGenerate the complete SKiDL code now:"
        
        return prompt
    
    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format structured requirements for prompt."""
        formatted = []
        
        # Board specifications
        if "board" in requirements:
            board = requirements["board"]
            formatted.append("### Board Specifications")
            if board.get("width_mm") and board.get("height_mm"):
                formatted.append(
                    f"- Dimensions: {board['width_mm']}mm x {board['height_mm']}mm"
                )
            formatted.append(f"- Layers: {board.get('layers', 1)}")
            formatted.append("")
        
        # Power specifications
        if "power" in requirements:
            power = requirements["power"]
            formatted.append("### Power Supply")
            if power.get("type"):
                formatted.append(f"- Type: {power['type']}")
            if power.get("voltage"):
                formatted.append(f"- Voltage: {power['voltage']}V")
            if power.get("current_max_a"):
                formatted.append(f"- Max Current: {power['current_max_a']}A")
            formatted.append("")
        
        # Components
        if "components" in requirements:
            components = requirements["components"]
            formatted.append("### Components")
            for comp in components:
                comp_str = f"- {comp['type']}"
                if comp.get("value"):
                    comp_str += f" ({comp['value']})"
                if comp.get("package"):
                    comp_str += f" [{comp['package']}]"
                if comp.get("quantity", 1) > 1:
                    comp_str += f" x{comp['quantity']}"
                if comp.get("description"):
                    comp_str += f" - {comp['description']}"
                formatted.append(comp_str)
            formatted.append("")
        
        # Connections
        if "connections" in requirements and requirements["connections"]:
            formatted.append("### Connections")
            for conn in requirements["connections"]:
                formatted.append(f"- Connect {conn[0]} to {conn[1]}")
            formatted.append("")
        
        # Constraints
        if "constraints" in requirements:
            constraints = requirements["constraints"]
            formatted.append("### Design Constraints")
            if constraints.get("priority"):
                formatted.append(f"- Priority: {constraints['priority']}")
            if constraints.get("max_power_w"):
                formatted.append(f"- Max Power: {constraints['max_power_w']}W")
            if constraints.get("special_requirements"):
                for req in constraints["special_requirements"]:
                    formatted.append(f"- {req}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _call_openai(self, prompt: str) -> LLMResponse:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert PCB design engineer specializing in SKiDL."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider="openai",
                model=self.model,
                tokens_used=response.usage.total_tokens,
                latency_ms=0,  # Will be set by caller
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _call_anthropic(self, prompt: str) -> LLMResponse:
        """Call Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                timeout=self.timeout
            )
            
            # Extract text content
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
            
            return LLMResponse(
                content=content,
                provider="anthropic",
                model=self.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=0,  # Will be set by caller
                finish_reason=response.stop_reason
            )
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise
    
    def _validate_skidl_response(self, response: LLMResponse):
        """
        Validate SKiDL code response.
        
        Args:
            response: LLM response to validate
            
        Raises:
            LLMValidationError: If validation fails
        """
        content = response.content.strip()
        
        # Check if response is empty
        if not content:
            raise LLMValidationError("Empty response from LLM")
        
        # Check if response contains Python code
        if "```python" not in content and "import" not in content:
            raise LLMValidationError(
                "Response does not appear to contain Python code"
            )
        
        # Extract code from markdown if present
        if "```python" in content:
            # Extract code between ```python and ```
            code_start = content.find("```python") + len("```python")
            code_end = content.find("```", code_start)
            if code_end == -1:
                raise LLMValidationError("Malformed code block in response")
            code = content[code_start:code_end].strip()
        else:
            code = content
        
        # Basic SKiDL validation
        required_imports = ["skidl", "Part", "Net"]
        for imp in required_imports:
            if imp not in code:
                logger.warning(f"Missing expected import: {imp}")
        
        # Check for generate_netlist call
        if "generate_netlist" not in code:
            logger.warning("Missing generate_netlist() call")
        
        # Update response content with extracted code
        response.content = code
        
        logger.debug(f"Validated SKiDL response: {len(code)} characters")
