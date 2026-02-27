"""
SKiDL Code Generation Engine.

Generates SKiDL Python code from structured requirements using LLM services,
with component instantiation, net connections, code commenting, and syntax validation.
"""

import logging
import ast
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.services.llm_service import (
    LLMService,
    CodeGenerationRequest,
    LLMResponse,
    LLMServiceError
)
from src.services.nlp_service import StructuredRequirements

logger = logging.getLogger(__name__)


@dataclass
class SKiDLGenerationResult:
    """Result of SKiDL code generation."""
    code: str
    success: bool
    validation_errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class SKiDLGenerationError(Exception):
    """Exception for SKiDL generation errors."""
    pass


class SKiDLGenerator:
    """
    SKiDL Code Generation Engine.
    
    Generates valid SKiDL Python code from structured requirements,
    with automatic validation and error recovery.
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        max_retries: int = 3
    ):
        """
        Initialize SKiDL generator.
        
        Args:
            llm_service: LLM service for code generation (creates default if None)
            max_retries: Maximum retry attempts for code generation
        """
        self.llm_service = llm_service or LLMService(provider="openai")
        self.max_retries = max_retries
        
        logger.info(f"Initialized SKiDL generator with max_retries={max_retries}")
    
    def generate(
        self,
        requirements: StructuredRequirements,
        design_context: Optional[str] = None
    ) -> SKiDLGenerationResult:
        """
        Generate SKiDL code from structured requirements.
        
        Args:
            requirements: Structured requirements from NLP service
            design_context: Optional additional context for generation
            
        Returns:
            SKiDLGenerationResult with generated code and validation status
            
        Raises:
            SKiDLGenerationError: If generation fails after all retries
        """
        logger.info("Starting SKiDL code generation")
        
        # Convert requirements to dict for LLM
        requirements_dict = requirements.to_dict()
        
        previous_attempts = []
        error_feedback = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Generation attempt {attempt + 1}/{self.max_retries}")
                
                # Create generation request
                request = CodeGenerationRequest(
                    structured_requirements=requirements_dict,
                    design_context=design_context,
                    previous_attempts=previous_attempts,
                    error_feedback=error_feedback
                )
                
                # Generate code using LLM
                llm_response = self.llm_service.generate_skidl_code(request)
                
                # Validate generated code
                validation_result = self._validate_code(llm_response.content)
                
                if validation_result["valid"]:
                    # Success!
                    logger.info(
                        f"Successfully generated SKiDL code on attempt {attempt + 1}"
                    )
                    
                    return SKiDLGenerationResult(
                        code=llm_response.content,
                        success=True,
                        validation_errors=[],
                        warnings=validation_result.get("warnings", []),
                        metadata={
                            "attempts": attempt + 1,
                            "provider": llm_response.provider,
                            "model": llm_response.model,
                            "tokens_used": llm_response.tokens_used,
                            "latency_ms": llm_response.latency_ms
                        }
                    )
                else:
                    # Validation failed, prepare for retry
                    errors = validation_result.get("errors", [])
                    logger.warning(
                        f"Code validation failed on attempt {attempt + 1}: {errors}"
                    )
                    
                    previous_attempts.append(llm_response.content)
                    error_feedback = self._format_validation_errors(errors)
                    
            except LLMServiceError as e:
                logger.error(f"LLM service error on attempt {attempt + 1}: {e}")
                error_feedback = f"LLM service error: {str(e)}"
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                error_feedback = f"Unexpected error: {str(e)}"
        
        # All attempts failed
        logger.error(f"Failed to generate valid SKiDL code after {self.max_retries} attempts")
        
        return SKiDLGenerationResult(
            code=previous_attempts[-1] if previous_attempts else "",
            success=False,
            validation_errors=[f"Failed after {self.max_retries} attempts: {error_feedback}"],
            warnings=[],
            metadata={"attempts": self.max_retries}
        )
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate generated SKiDL code.
        
        Args:
            code: Generated SKiDL code
            
        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []
        
        # Check if code is empty
        if not code or not code.strip():
            errors.append("Generated code is empty")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Validate Python syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Python syntax error at line {e.lineno}: {e.msg}")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check for required SKiDL imports
        required_imports = ["skidl", "Part", "Net"]
        for imp in required_imports:
            if imp not in code:
                warnings.append(f"Missing recommended import: {imp}")
        
        # Check for generate_netlist call
        if "generate_netlist" not in code:
            warnings.append("Missing generate_netlist() call - netlist won't be generated")
        
        # Check for component instantiation
        # Look for actual Part() calls, not just the string "Part("
        part_pattern = r'\w+\s*=\s*Part\s*\('
        if not re.search(part_pattern, code):
            errors.append("No component instantiation found (Part() calls)")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check for net definitions
        if "Net(" not in code:
            warnings.append("No explicit Net() definitions found")
        
        # Validate component connections (basic check)
        if "+=" not in code and "connect" not in code.lower():
            warnings.append("No component connections found (no += or connect() calls)")
        
        # Check for comments
        if "#" not in code:
            warnings.append("No comments found - consider adding design documentation")
        
        # All checks passed
        return {"valid": True, "errors": [], "warnings": warnings}
    
    def _format_validation_errors(self, errors: List[str]) -> str:
        """Format validation errors for feedback to LLM."""
        if not errors:
            return ""
        
        feedback = "The generated code has the following validation errors:\n\n"
        for i, error in enumerate(errors, 1):
            feedback += f"{i}. {error}\n"
        
        feedback += "\nPlease fix these errors and regenerate the code."
        
        return feedback
    
    def add_documentation(self, code: str, requirements: StructuredRequirements) -> str:
        """
        Add comprehensive documentation to SKiDL code.
        
        Args:
            code: Generated SKiDL code
            requirements: Original requirements
            
        Returns:
            Code with added documentation
        """
        # Create header documentation
        header = f'''"""
SKiDL PCB Design - Auto-generated

Design Requirements:
- Board: {requirements.board.width_mm}x{requirements.board.height_mm}mm, {requirements.board.layers} layer(s)
- Power: {requirements.power.type or "Not specified"} ({requirements.power.voltage}V)
- Components: {len(requirements.components)} component(s)
- Priority: {requirements.constraints.priority}

Generated by GenAI PCB Design Platform
"""

'''
        
        # Add header to code
        documented_code = header + code
        
        return documented_code
    
    def validate_syntax_only(self, code: str) -> bool:
        """
        Quick syntax-only validation.
        
        Args:
            code: SKiDL code to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def extract_components(self, code: str) -> List[Dict[str, str]]:
        """
        Extract component information from SKiDL code.
        
        Args:
            code: SKiDL code
            
        Returns:
            List of component dictionaries
        """
        components = []
        
        # Pattern to match Part() instantiations
        # Example: led = Part('Device', 'LED', footprint='LED_SMD:LED_0805')
        part_pattern = r"(\w+)\s*=\s*Part\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]"
        
        matches = re.finditer(part_pattern, code)
        
        for match in matches:
            var_name = match.group(1)
            library = match.group(2)
            part_type = match.group(3)
            
            components.append({
                "variable": var_name,
                "library": library,
                "type": part_type
            })
        
        logger.debug(f"Extracted {len(components)} components from code")
        
        return components
    
    def extract_nets(self, code: str) -> List[str]:
        """
        Extract net names from SKiDL code.
        
        Args:
            code: SKiDL code
            
        Returns:
            List of net names
        """
        nets = []
        
        # Pattern to match Net() instantiations
        # Example: vcc = Net('VCC')
        net_pattern = r"(\w+)\s*=\s*Net\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
        
        matches = re.finditer(net_pattern, code)
        
        for match in matches:
            var_name = match.group(1)
            net_name = match.group(2)
            
            nets.append(net_name)
        
        logger.debug(f"Extracted {len(nets)} nets from code")
        
        return nets
