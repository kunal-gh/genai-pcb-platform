"""
User-Facing Error Communication Service

Provides clear, actionable error messages with corrective actions
and recovery guidance for end users.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ErrorType(Enum):
    """User-facing error types"""
    INVALID_INPUT = "invalid_input"
    COMPONENT_NOT_FOUND = "component_not_found"
    GENERATION_FAILED = "generation_failed"
    VERIFICATION_FAILED = "verification_failed"
    SIMULATION_FAILED = "simulation_failed"
    EXPORT_FAILED = "export_failed"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class UserErrorMessage:
    """User-friendly error message with guidance"""
    title: str
    message: str
    corrective_actions: List[str]
    recovery_steps: Optional[List[str]] = None
    technical_details: Optional[str] = None
    support_link: Optional[str] = None
    can_retry: bool = True
    partial_results_available: bool = False


class UserErrorCommunicator:
    """Generates user-friendly error messages"""
    
    def __init__(self):
        self.error_templates = self._initialize_templates()
        self.support_base_url = "https://support.genai-pcb.com"
    
    def _initialize_templates(self) -> Dict[ErrorType, Dict]:
        """Initialize error message templates"""
        return {
            ErrorType.INVALID_INPUT: {
                'title': 'Invalid Input',
                'message': 'We couldn\'t understand your design requirements.',
                'actions': [
                    'Check that your prompt includes component types (e.g., LED, resistor)',
                    'Specify connections between components',
                    'Include power supply information if needed',
                    'Keep your prompt between 10 and 10,000 characters'
                ],
                'recovery': [
                    'Review the example prompts in the help section',
                    'Try simplifying your design description',
                    'Break complex designs into smaller parts'
                ]
            },
            ErrorType.COMPONENT_NOT_FOUND: {
                'title': 'Component Not Available',
                'message': 'One or more components in your design are not available.',
                'actions': [
                    'Check component specifications for typos',
                    'Use standard component values (e.g., 1K, 10K for resistors)',
                    'Consider alternative components with similar specifications'
                ],
                'recovery': [
                    'View suggested alternative components',
                    'Modify component specifications',
                    'Contact support for custom component requests'
                ]
            },
            ErrorType.GENERATION_FAILED: {
                'title': 'Design Generation Failed',
                'message': 'We encountered an issue while generating your PCB design.',
                'actions': [
                    'Simplify your design requirements',
                    'Reduce the number of components',
                    'Check for conflicting specifications'
                ],
                'recovery': [
                    'Try again with a simpler design',
                    'Download any partial results available',
                    'Contact support if the issue persists'
                ]
            },
            ErrorType.VERIFICATION_FAILED: {
                'title': 'Design Verification Issues',
                'message': 'Your design has verification errors that need attention.',
                'actions': [
                    'Review the verification report for specific issues',
                    'Check for unconnected pins or nets',
                    'Verify power supply connections',
                    'Ensure design rules are met'
                ],
                'recovery': [
                    'Fix reported issues and regenerate',
                    'Adjust design constraints',
                    'Download design files for manual review'
                ]
            },
            ErrorType.SIMULATION_FAILED: {
                'title': 'Simulation Error',
                'message': 'Circuit simulation encountered an error.',
                'actions': [
                    'Check component values for validity',
                    'Verify circuit topology is correct',
                    'Ensure all nodes are properly connected'
                ],
                'recovery': [
                    'Skip simulation and proceed to layout',
                    'Modify circuit for simulation compatibility',
                    'Download netlist for external simulation'
                ]
            },
            ErrorType.EXPORT_FAILED: {
                'title': 'File Export Error',
                'message': 'We couldn\'t export your design files.',
                'actions': [
                    'Check available disk space',
                    'Try exporting individual file types',
                    'Verify file format compatibility'
                ],
                'recovery': [
                    'Retry export operation',
                    'Download available files separately',
                    'Contact support for assistance'
                ]
            },
            ErrorType.SERVICE_UNAVAILABLE: {
                'title': 'Service Temporarily Unavailable',
                'message': 'A required service is currently unavailable.',
                'actions': [
                    'Wait a few moments and try again',
                    'Check system status page',
                    'Save your work and retry later'
                ],
                'recovery': [
                    'Retry your request',
                    'Use cached results if available',
                    'Contact support if issue persists'
                ]
            },
            ErrorType.TIMEOUT: {
                'title': 'Request Timeout',
                'message': 'Your design is taking longer than expected to process.',
                'actions': [
                    'Simplify your design to reduce processing time',
                    'Reduce the number of components',
                    'Try again during off-peak hours'
                ],
                'recovery': [
                    'Retry with a simpler design',
                    'Break design into smaller modules',
                    'Contact support for complex designs'
                ]
            },
            ErrorType.RESOURCE_LIMIT: {
                'title': 'Resource Limit Exceeded',
                'message': 'Your design exceeds current resource limits.',
                'actions': [
                    'Reduce design complexity',
                    'Decrease number of components',
                    'Simplify board layout requirements'
                ],
                'recovery': [
                    'Upgrade to a higher tier plan',
                    'Split design into multiple boards',
                    'Contact sales for enterprise options'
                ]
            }
        }
    
    def generate_error_message(
        self,
        error_type: ErrorType,
        context: Optional[Dict] = None,
        technical_details: Optional[str] = None,
        partial_results: bool = False
    ) -> UserErrorMessage:
        """
        Generate user-friendly error message
        
        Args:
            error_type: Type of error
            context: Additional context for personalization
            technical_details: Technical error details
            partial_results: Whether partial results are available
            
        Returns:
            UserErrorMessage object
        """
        template = self.error_templates.get(error_type)
        if not template:
            return self._generate_generic_error(technical_details)
        
        # Personalize message with context
        message = template['message']
        if context:
            message = self._personalize_message(message, context)
        
        # Determine if retry is possible
        can_retry = error_type not in [
            ErrorType.INVALID_INPUT,
            ErrorType.RESOURCE_LIMIT
        ]
        
        return UserErrorMessage(
            title=template['title'],
            message=message,
            corrective_actions=template['actions'],
            recovery_steps=template.get('recovery'),
            technical_details=technical_details,
            support_link=f"{self.support_base_url}/errors/{error_type.value}",
            can_retry=can_retry,
            partial_results_available=partial_results
        )
    
    def _personalize_message(self, message: str, context: Dict) -> str:
        """Personalize error message with context"""
        if 'component_name' in context:
            message += f" Component: {context['component_name']}"
        if 'stage' in context:
            message += f" Stage: {context['stage']}"
        return message
    
    def _generate_generic_error(self, technical_details: Optional[str] = None) -> UserErrorMessage:
        """Generate generic error message"""
        return UserErrorMessage(
            title="Unexpected Error",
            message="An unexpected error occurred while processing your design.",
            corrective_actions=[
                "Try your request again",
                "Simplify your design if possible",
                "Contact support if the issue persists"
            ],
            recovery_steps=[
                "Retry your request",
                "Check system status",
                "Contact support with error details"
            ],
            technical_details=technical_details,
            support_link=f"{self.support_base_url}/errors/general",
            can_retry=True,
            partial_results_available=False
        )
    
    def categorize_error(self, error_message: str, error_category: str) -> ErrorType:
        """
        Categorize technical error into user-facing type
        
        Args:
            error_message: Technical error message
            error_category: Error category from error manager
            
        Returns:
            ErrorType enum
        """
        error_lower = error_message.lower()
        
        # Check for resource/limit errors first (before component check)
        if any(word in error_lower for word in ['limit', 'quota', 'exceeded', 'too large']):
            return ErrorType.RESOURCE_LIMIT
        
        # Input validation errors
        if any(word in error_lower for word in ['invalid', 'validation', 'parse', 'format']):
            return ErrorType.INVALID_INPUT
        
        # Component errors
        if any(word in error_lower for word in ['component', 'part', 'not found', 'unavailable']):
            return ErrorType.COMPONENT_NOT_FOUND
        
        # Generation errors
        if error_category in ['code_generation', 'schematic_generation']:
            return ErrorType.GENERATION_FAILED
        
        # Verification errors
        if error_category == 'verification':
            return ErrorType.VERIFICATION_FAILED
        
        # Simulation errors
        if error_category == 'simulation':
            return ErrorType.SIMULATION_FAILED
        
        # Export errors
        if error_category == 'file_export':
            return ErrorType.EXPORT_FAILED
        
        # Timeout errors
        if 'timeout' in error_lower:
            return ErrorType.TIMEOUT
        
        # Service errors
        if any(word in error_lower for word in ['unavailable', 'connection', 'service']):
            return ErrorType.SERVICE_UNAVAILABLE
        
        # Default to generation failed
        return ErrorType.GENERATION_FAILED
    
    def format_for_display(self, error_msg: UserErrorMessage, format_type: str = 'html') -> str:
        """
        Format error message for display
        
        Args:
            error_msg: UserErrorMessage object
            format_type: Output format (html, text, json)
            
        Returns:
            Formatted error message
        """
        if format_type == 'html':
            return self._format_html(error_msg)
        elif format_type == 'text':
            return self._format_text(error_msg)
        elif format_type == 'json':
            import json
            return json.dumps({
                'title': error_msg.title,
                'message': error_msg.message,
                'corrective_actions': error_msg.corrective_actions,
                'recovery_steps': error_msg.recovery_steps,
                'technical_details': error_msg.technical_details,
                'support_link': error_msg.support_link,
                'can_retry': error_msg.can_retry,
                'partial_results_available': error_msg.partial_results_available
            }, indent=2)
        else:
            return self._format_text(error_msg)
    
    def _format_html(self, error_msg: UserErrorMessage) -> str:
        """Format error message as HTML"""
        html = f"""
<div class="error-message">
    <h3>{error_msg.title}</h3>
    <p>{error_msg.message}</p>
    
    <div class="corrective-actions">
        <h4>What you can do:</h4>
        <ul>
"""
        for action in error_msg.corrective_actions:
            html += f"            <li>{action}</li>\n"
        
        html += """        </ul>
    </div>
"""
        
        if error_msg.recovery_steps:
            html += """    <div class="recovery-steps">
        <h4>Recovery options:</h4>
        <ul>
"""
            for step in error_msg.recovery_steps:
                html += f"            <li>{step}</li>\n"
            html += """        </ul>
    </div>
"""
        
        if error_msg.partial_results_available:
            html += """    <div class="partial-results">
        <p><strong>Note:</strong> Partial results are available for download.</p>
    </div>
"""
        
        if error_msg.support_link:
            html += f"""    <div class="support">
        <p><a href="{error_msg.support_link}">Learn more</a> or <a href="{self.support_base_url}/contact">contact support</a></p>
    </div>
"""
        
        html += "</div>"
        return html
    
    def _format_text(self, error_msg: UserErrorMessage) -> str:
        """Format error message as plain text"""
        text = f"{error_msg.title}\n"
        text += "=" * len(error_msg.title) + "\n\n"
        text += f"{error_msg.message}\n\n"
        
        text += "What you can do:\n"
        for i, action in enumerate(error_msg.corrective_actions, 1):
            text += f"{i}. {action}\n"
        text += "\n"
        
        if error_msg.recovery_steps:
            text += "Recovery options:\n"
            for i, step in enumerate(error_msg.recovery_steps, 1):
                text += f"{i}. {step}\n"
            text += "\n"
        
        if error_msg.partial_results_available:
            text += "Note: Partial results are available for download.\n\n"
        
        if error_msg.support_link:
            text += f"Learn more: {error_msg.support_link}\n"
            text += f"Contact support: {self.support_base_url}/contact\n"
        
        return text


class ProgressiveDisclosure:
    """Handles progressive disclosure of error details"""
    
    def __init__(self):
        self.detail_levels = ['basic', 'intermediate', 'advanced']
    
    def get_error_details(
        self,
        error_msg: UserErrorMessage,
        level: str = 'basic'
    ) -> Dict:
        """
        Get error details at specified disclosure level
        
        Args:
            error_msg: UserErrorMessage object
            level: Detail level (basic, intermediate, advanced)
            
        Returns:
            Dictionary with appropriate level of detail
        """
        if level not in self.detail_levels:
            level = 'basic'
        
        details = {
            'title': error_msg.title,
            'message': error_msg.message
        }
        
        if level in ['basic', 'intermediate', 'advanced']:
            details['corrective_actions'] = error_msg.corrective_actions
        
        if level in ['intermediate', 'advanced']:
            details['recovery_steps'] = error_msg.recovery_steps
            details['can_retry'] = error_msg.can_retry
            details['partial_results_available'] = error_msg.partial_results_available
        
        if level == 'advanced':
            details['technical_details'] = error_msg.technical_details
            details['support_link'] = error_msg.support_link
        
        return details
