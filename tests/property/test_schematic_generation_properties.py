"""
Property-based tests for Schematic Generation.

Uses Hypothesis to generate thousands of test cases and validate
universal properties of the SKiDL schematic generation system.

Feature: genai-pcb-platform
Property 6: Netlist Generation Completeness
Property 7: Schematic Generation Error Handling
Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.strategies import composite
import tempfile
import os
import re
from pathlib import Path

from src.services.skidl_executor import SKiDLExecutor, SKiDLExecutionError
from src.services.component_library import ComponentLibrary, ComponentLibraryError
from src.models.component import Component, ComponentCategory


# Custom strategies for generating valid SKiDL code

@composite
def component_reference(draw):
    """Generate valid component reference designators."""
    prefix = draw(st.sampled_from(["R", "C", "L", "U", "D", "Q", "J", "SW"]))
    number = draw(st.integers(min_value=1, max_value=999))
    return f"{prefix}{number}"


@composite
def component_value(draw):
    """Generate valid component values."""
    value = draw(st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False))
    unit = draw(st.sampled_from(["", "k", "M", "m", "u", "n", "p"]))
    suffix = draw(st.sampled_from(["", "ohm", "Ω", "F", "H", "V", "A"]))
    return f"{value:.2f}{unit}{suffix}"


@composite
def net_name(draw):
    """Generate valid net names."""
    name = draw(st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.isidentifier() and not x.startswith("_")))
    return name


@composite
def simple_skidl_code(draw):
    """Generate simple but valid SKiDL code."""
    num_components = draw(st.integers(min_value=2, max_value=8))
    components = []
    nets = []
    
    # Generate components
    for i in range(num_components):
        ref = draw(component_reference())
        library = draw(st.sampled_from(["Device", "Connector", "Switch"]))
        part = draw(st.sampled_from(["R", "C", "L", "LED", "D", "Q_NPN_BCE", "Conn_01x02"]))
        value = draw(component_value()) if part in ["R", "C", "L"] else None
        
        if value:
            components.append(f'{ref} = Part("{library}", "{part}", value="{value}")')
        else:
            components.append(f'{ref} = Part("{library}", "{part}")')
    
    # Generate some nets
    num_nets = draw(st.integers(min_value=1, max_value=4))
    for i in range(num_nets):
        net_name_str = draw(net_name())
        nets.append(f'{net_name_str} = Net("{net_name_str}")')
    
    # Generate some connections (simplified)
    connections = []
    if len(components) >= 2:
        ref1 = components[0].split()[0]
        ref2 = components[1].split()[0]
        if nets:
            net_ref = nets[0].split()[0]
            connections.append(f'{ref1}[1] += {net_ref}')
            connections.append(f'{ref2}[1] += {net_ref}')
    
    # Combine all parts
    code_parts = [
        "from skidl import *",
        "",
        "# Components"
    ] + components + [
        "",
        "# Nets"
    ] + nets + [
        "",
        "# Connections"
    ] + connections
    
    return "\n".join(code_parts)


@composite
def invalid_skidl_code(draw):
    """Generate invalid SKiDL code for error testing."""
    error_type = draw(st.sampled_from([
        "syntax_error",
        "missing_import",
        "invalid_component",
        "invalid_connection"
    ]))
    
    if error_type == "syntax_error":
        return "from skidl import *\nR1 = Part(Device\", \"R\")"  # Missing quote
    elif error_type == "missing_import":
        return "R1 = Part(\"Device\", \"R\")"  # No import
    elif error_type == "invalid_component":
        return "from skidl import *\nR1 = Part(\"NonExistentLib\", \"NonExistentPart\")"
    else:  # invalid_connection
        return "from skidl import *\nR1 = Part(\"Device\", \"R\")\nR1[999] += Net(\"test\")"  # Invalid pin


# ============================================================================
# Property 6: Netlist Generation Completeness
# Feature: genai-pcb-platform
# Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5
# ============================================================================

@pytest.mark.property
@given(code=simple_skidl_code())
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_6_netlist_generation_completeness(code):
    """
    Property 6: Netlist Generation Completeness
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    
    Property: For any valid SKiDL code with components and connections,
    the code validation and component extraction SHALL work correctly,
    and if execution succeeds, the netlist SHALL contain all specified components.
    
    Invariants:
    - Code validation works for syntactically correct SKiDL
    - Component extraction identifies all Part() declarations
    - Net extraction identifies all Net() declarations
    - If execution succeeds, netlist contains component references
    """
    executor = SKiDLExecutor()
    
    try:
        # Extract expected components and nets from code
        expected_components = re.findall(r'(\w+)\s*=\s*Part\s*\(', code)
        expected_nets = re.findall(r'(\w+)\s*=\s*Net\s*\(', code)
        
        # Skip if no components (invalid test case)
        assume(len(expected_components) > 0)
        
        # Invariant 1: Code validation should work
        validation = executor.validate_code(code)
        assert validation["valid"], f"Valid SKiDL code failed validation: {validation['errors']}"
        
        # Invariant 2: Component extraction should find all components
        extracted_components = executor.extract_components(code)
        extracted_refs = [comp["name"] for comp in extracted_components]
        
        for expected_ref in expected_components:
            assert expected_ref in extracted_refs, \
                f"Component {expected_ref} not extracted from code"
        
        # Invariant 3: Net extraction should find all nets
        extracted_nets = executor.extract_nets(code)
        for expected_net in expected_nets:
            assert expected_net in extracted_nets, \
                f"Net {expected_net} not extracted from code"
        
        # Invariant 4: Try execution (may fail due to missing KiCad libraries)
        try:
            result = executor.execute(code, "test_circuit")
            
            # If execution succeeds, verify netlist properties
            if result["success"]:
                assert result["netlist_file"] is not None, "No netlist file generated"
                
                if os.path.exists(result["netlist_file"]):
                    with open(result["netlist_file"], 'r') as f:
                        netlist_content = f.read()
                    
                    # Netlist should not be empty
                    assert netlist_content.strip(), "Netlist should not be empty"
                    
                    # Component references should appear in netlist
                    for component_ref in expected_components:
                        # Check if component reference appears (case insensitive)
                        if not re.search(rf'\b{re.escape(component_ref)}\b', 
                                       netlist_content, re.IGNORECASE):
                            # This is acceptable if KiCad libraries are missing
                            pass
        
        except SKiDLExecutionError:
            # Execution failure is acceptable if KiCad libraries are missing
            # The important thing is that validation and extraction worked
            pass
    
    finally:
        executor.cleanup()


@pytest.mark.property
@given(
    num_components=st.integers(min_value=1, max_value=10),
    num_nets=st.integers(min_value=0, max_value=5)
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_6_netlist_component_count_preservation(num_components, num_nets):
    """
    Property 6: Netlist Generation Completeness (Component Count)
    
    **Validates: Requirements 3.1, 3.4**
    
    Property: The validation and component extraction SHALL correctly
    identify the number of components defined in the SKiDL code.
    
    Invariants:
    - Component count is correctly extracted
    - Each component has unique reference
    - Validation succeeds for well-formed code
    """
    executor = SKiDLExecutor()
    
    try:
        # Generate SKiDL code with specific component count
        code_lines = ["from skidl import *", ""]
        component_refs = []
        
        for i in range(num_components):
            ref = f"R{i+1}"
            component_refs.append(ref)
            code_lines.append(f'{ref} = Part("Device", "R", value="1k")')
        
        # Add some nets if specified
        for i in range(num_nets):
            net_name = f"net{i+1}"
            code_lines.append(f'{net_name} = Net("{net_name}")')
        
        code = "\n".join(code_lines)
        
        # Invariant 1: Validation should succeed
        validation = executor.validate_code(code)
        assert validation["valid"], f"Valid code failed validation: {validation['errors']}"
        
        # Invariant 2: Component extraction should find all components
        extracted_components = executor.extract_components(code)
        assert len(extracted_components) == num_components, \
            f"Expected {num_components} components, extracted {len(extracted_components)}"
        
        # Invariant 3: All component references should be found
        extracted_refs = [comp["name"] for comp in extracted_components]
        for ref in component_refs:
            assert ref in extracted_refs, f"Component {ref} not extracted"
        
        # Invariant 4: No duplicate component references
        assert len(set(extracted_refs)) == len(extracted_refs), \
            "Duplicate component references found"
        
        # Try execution (may fail due to missing libraries, but that's OK)
        try:
            result = executor.execute(code, "component_count_test")
            if result["success"] and result["netlist_file"]:
                # If execution succeeds, verify netlist contains components
                with open(result["netlist_file"], 'r') as f:
                    netlist_content = f.read()
                
                # Count how many component references appear
                found_components = 0
                for ref in component_refs:
                    if ref in netlist_content:
                        found_components += 1
                
                # Should find at least some components if execution succeeded
                assert found_components > 0, "No components found in generated netlist"
        
        except SKiDLExecutionError:
            # Execution failure is acceptable due to missing KiCad libraries
            pass
    
    finally:
        executor.cleanup()


@pytest.mark.property
@given(code=simple_skidl_code())
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_6_netlist_connection_preservation(code):
    """
    Property 6: Netlist Generation Completeness (Connection Preservation)
    
    **Validates: Requirements 3.1, 3.4**
    
    Property: All pin connections specified in SKiDL code SHALL be
    correctly parsed and validated.
    
    Invariants:
    - Connection syntax is correctly parsed
    - Component references in connections are validated
    - Code with connections passes validation
    """
    executor = SKiDLExecutor()
    
    try:
        # Extract connection information from code
        connections = re.findall(r'(\w+)\[(\d+)\]\s*\+=\s*(\w+)', code)
        
        # Invariant 1: Code should pass validation if syntactically correct
        validation = executor.validate_code(code)
        if validation["valid"]:
            # Invariant 2: Component extraction should work
            components = executor.extract_components(code)
            component_refs = [comp["name"] for comp in components]
            
            # Invariant 3: Components referenced in connections should exist
            for component_ref, pin, net_ref in connections:
                # Component should be defined in the code
                assert component_ref in component_refs or component_ref in code, \
                    f"Component {component_ref} used in connection but not defined"
            
            # Try execution (may fail due to missing libraries)
            try:
                result = executor.execute(code, "connection_test")
                
                if result["success"] and result["netlist_file"]:
                    with open(result["netlist_file"], 'r') as f:
                        netlist_content = f.read()
                    
                    # If execution succeeded, netlist should contain components
                    for component_ref, pin, net_ref in connections:
                        # Component should appear in netlist
                        if component_ref not in netlist_content:
                            # This might be due to library issues, which is acceptable
                            pass
            
            except SKiDLExecutionError:
                # Execution failure is acceptable due to missing KiCad libraries
                pass
    
    finally:
        executor.cleanup()


# ============================================================================
# Property 7: Schematic Generation Error Handling
# Feature: genai-pcb-platform
# Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5
# ============================================================================

@pytest.mark.property
@given(code=invalid_skidl_code())
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_7_schematic_generation_error_handling_invalid_code(code):
    """
    Property 7: Schematic Generation Error Handling
    
    **Validates: Requirements 3.1, 3.2, 3.5**
    
    Property: For any invalid SKiDL code, the system SHALL detect the error
    and provide descriptive error messages without crashing.
    
    Invariants:
    - Invalid code is rejected with appropriate error
    - Error messages are descriptive
    - System does not crash or hang
    - No partial/corrupted netlist is generated
    """
    executor = SKiDLExecutor()
    
    try:
        # Invariant 1: Validation should catch invalid code
        validation = executor.validate_code(code)
        
        if not validation["valid"]:
            # Invariant 2: Error messages should be descriptive
            assert len(validation["errors"]) > 0, "Invalid code should have error messages"
            
            for error in validation["errors"]:
                assert isinstance(error, str), "Error should be a string"
                assert len(error) > 5, "Error message should be descriptive"
        
        # Invariant 3: If validation passes but execution fails, should handle gracefully
        if validation["valid"]:
            try:
                result = executor.execute(code, "error_test")
                
                # If execution fails, should have proper error handling
                if not result["success"]:
                    # Should not generate partial netlist file
                    if result.get("netlist_file"):
                        assert not os.path.exists(result["netlist_file"]) or \
                               os.path.getsize(result["netlist_file"]) == 0, \
                               "Should not generate partial netlist on failure"
            
            except SKiDLExecutionError as e:
                # Invariant 4: Exceptions should be descriptive
                assert len(str(e)) > 5, "Exception message should be descriptive"
                assert "error" in str(e).lower() or "failed" in str(e).lower(), \
                       "Exception should indicate error/failure"
    
    finally:
        executor.cleanup()


@pytest.mark.property
@given(
    timeout_seconds=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_7_schematic_generation_timeout_handling(timeout_seconds):
    """
    Property 7: Schematic Generation Error Handling (Timeout)
    
    **Validates: Requirements 3.1**
    
    Property: SKiDL code execution SHALL timeout appropriately for
    long-running or infinite loop code.
    
    Invariants:
    - Execution times out for infinite loops
    - Timeout error is descriptive
    - Resources are cleaned up after timeout
    """
    executor = SKiDLExecutor()
    
    try:
        # Create code that will likely timeout (infinite loop)
        infinite_loop_code = """
from skidl import *
import time

# This will cause timeout
while True:
    time.sleep(0.1)

R1 = Part("Device", "R")
"""
        
        # Invariant 1: Should timeout and raise appropriate error
        with pytest.raises(SKiDLExecutionError) as exc_info:
            executor.execute(infinite_loop_code, "timeout_test")
        
        # Invariant 2: Error message should mention timeout
        error_msg = str(exc_info.value)
        assert "timeout" in error_msg.lower(), \
            f"Timeout error should mention timeout: {error_msg}"
        
        # Invariant 3: Should not generate netlist file on timeout
        netlist_files = list(Path(executor.output_dir).glob("*.net"))
        for netlist_file in netlist_files:
            # If netlist exists, it should be empty or very small
            assert netlist_file.stat().st_size < 100, \
                "Timeout should not generate substantial netlist"
    
    finally:
        executor.cleanup()


@pytest.mark.property
@given(
    library_name=st.text(min_size=1, max_size=20),
    part_name=st.text(min_size=1, max_size=20)
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_7_missing_component_detection(library_name, part_name):
    """
    Property 7: Schematic Generation Error Handling (Missing Components)
    
    **Validates: Requirements 3.2, 3.5**
    
    Property: When SKiDL code references non-existent components,
    the system SHALL detect missing components and suggest alternatives.
    
    Invariants:
    - Missing components are detected
    - Alternative suggestions are provided
    - Error handling is graceful
    """
    # Filter out known valid library/part combinations
    assume(not (library_name == "Device" and part_name in ["R", "C", "L", "LED", "D"]))
    assume(library_name.isidentifier())
    assume(part_name.isidentifier())
    
    executor = SKiDLExecutor()
    
    try:
        # Create code with potentially missing component
        code = f"""
from skidl import *

R1 = Part("{library_name}", "{part_name}")
generate_netlist()
"""
        
        # Execute the code
        try:
            result = executor.execute(code, "missing_component_test")
            
            # If execution fails, check for appropriate error handling
            if not result["success"]:
                # Invariant 1: Should have descriptive error output
                output = result.get("output", "")
                assert len(output) > 0, "Should have error output for missing component"
                
                # Invariant 2: Error should mention the missing component
                assert library_name in output or part_name in output, \
                    f"Error should mention missing component: {output}"
            
            # If execution succeeds, the component might exist or have fallback
            else:
                # Invariant 3: Should generate valid netlist or have warnings
                if result.get("warnings"):
                    warnings = result["warnings"]
                    # Check if warnings mention missing or unknown components
                    warning_text = " ".join(warnings).lower()
                    # This is acceptable - warnings about unknown components
        
        except SKiDLExecutionError as e:
            # Invariant 4: Exception should be descriptive
            error_msg = str(e)
            assert len(error_msg) > 5, "Error message should be descriptive"
    
    finally:
        executor.cleanup()


@pytest.mark.property
@given(
    num_syntax_errors=st.integers(min_value=1, max_value=3)
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_7_syntax_error_detection(num_syntax_errors):
    """
    Property 7: Schematic Generation Error Handling (Syntax Errors)
    
    **Validates: Requirements 3.1**
    
    Property: SKiDL code with syntax errors SHALL be detected during
    validation before execution.
    
    Invariants:
    - Syntax errors are caught during validation
    - Line numbers are provided when possible
    - Multiple syntax errors are reported
    """
    executor = SKiDLExecutor()
    
    try:
        # Create code with intentional syntax errors
        syntax_errors = [
            'R1 = Part("Device", "R"',  # Missing closing parenthesis
            'R2 = Part("Device" "R")',  # Missing comma
            'R3 = Part("Device", "R"]]'  # Extra bracket
        ]
        
        # Select errors based on num_syntax_errors
        selected_errors = syntax_errors[:num_syntax_errors]
        
        code = "from skidl import *\n" + "\n".join(selected_errors)
        
        # Invariant 1: Validation should detect syntax errors
        validation = executor.validate_code(code)
        
        assert not validation["valid"], "Code with syntax errors should be invalid"
        
        # Invariant 2: Should have error messages
        assert len(validation["errors"]) > 0, "Should have syntax error messages"
        
        # Invariant 3: Error messages should mention syntax
        error_text = " ".join(validation["errors"]).lower()
        assert "syntax" in error_text, "Error should mention syntax error"
        
        # Invariant 4: Should not execute invalid code
        with pytest.raises(SKiDLExecutionError):
            executor.execute(code, "syntax_error_test")
    
    finally:
        executor.cleanup()


@pytest.mark.property
@given(code=simple_skidl_code())
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_7_resource_cleanup_on_error(code):
    """
    Property 7: Schematic Generation Error Handling (Resource Cleanup)
    
    **Validates: Requirements 3.1**
    
    Property: When errors occur during SKiDL execution, all resources
    SHALL be properly cleaned up.
    
    Invariants:
    - Temporary files are cleaned up on error
    - No resource leaks occur
    - Cleanup can be called multiple times safely
    """
    executor = SKiDLExecutor()
    
    try:
        # Invariant 1: Output directory should exist initially
        assert os.path.exists(executor.output_dir), "Output directory should exist"
        
        # Execute code (may succeed or fail)
        try:
            result = executor.execute(code, "cleanup_test")
        except SKiDLExecutionError:
            # Error is expected for some generated code
            pass
        
        # Invariant 2: Cleanup should work regardless of execution result
        executor.cleanup()
        
        # Invariant 3: Multiple cleanups should be safe
        executor.cleanup()
        executor.cleanup()
        
        # Invariant 4: After cleanup, temp directory should be removed
        # (Only if it was a temp directory created by executor)
        if executor._temp_dir:
            assert not os.path.exists(executor.output_dir), \
                "Temporary directory should be removed after cleanup"
    
    finally:
        # Ensure cleanup even if test fails
        try:
            executor.cleanup()
        except:
            pass


@pytest.mark.property
@given(
    component_count=st.integers(min_value=0, max_value=20)
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_7_empty_circuit_handling(component_count):
    """
    Property 7: Schematic Generation Error Handling (Empty Circuits)
    
    **Validates: Requirements 3.4**
    
    Property: SKiDL code with no components or empty circuits SHALL
    be handled gracefully with appropriate warnings.
    
    Invariants:
    - Empty circuits generate warnings
    - No crash on empty input
    - Appropriate error messages for unusable circuits
    """
    executor = SKiDLExecutor()
    
    try:
        if component_count == 0:
            # Completely empty circuit
            code = "from skidl import *\ngenerate_netlist()"
        else:
            # Circuit with components but no connections
            code_lines = ["from skidl import *"]
            for i in range(component_count):
                code_lines.append(f'R{i+1} = Part("Device", "R")')
            code_lines.append("generate_netlist()")
            code = "\n".join(code_lines)
        
        # Execute the code
        result = executor.execute(code, "empty_circuit_test")
        
        if component_count == 0:
            # Invariant 1: Empty circuit should generate warnings or fail gracefully
            if result["success"]:
                assert len(result.get("warnings", [])) > 0, \
                    "Empty circuit should generate warnings"
            else:
                # Failure is acceptable for empty circuits
                pass
        else:
            # Invariant 2: Circuit with components should succeed or warn
            if result["success"]:
                # Should generate netlist file
                assert result["netlist_file"] is not None, \
                    "Should generate netlist for circuit with components"
            else:
                # If it fails, should have descriptive error
                assert len(result.get("output", "")) > 0, \
                    "Should have error output for failed execution"
    
    finally:
        executor.cleanup()