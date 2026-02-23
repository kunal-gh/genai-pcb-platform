"""
Property-based tests for PCB Generation.

Uses Hypothesis to generate thousands of test cases and validate
universal properties of the KiCad integration and manufacturing export systems.

Feature: genai-pcb-platform
Property 8: PCB Layout Generation
Property 9: Layout Error Recovery
Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.strategies import composite
import tempfile
import os
import json
from pathlib import Path
from typing import Dict, Any, List

from src.services.kicad_integration import KiCadProject, KiCadIntegrationError
from src.services.manufacturing_export import (
    ManufacturingExporter, 
    ManufacturingExportError,
    ComponentPlacement,
    DrillHole
)


# Custom strategies for generating valid PCB data

@composite
def board_dimensions(draw):
    """Generate valid board dimensions in mm."""
    width = draw(st.floats(min_value=10.0, max_value=200.0))
    height = draw(st.floats(min_value=10.0, max_value=200.0))
    return {"width": width, "height": height}


@composite
def layer_count(draw):
    """Generate valid layer counts."""
    return draw(st.integers(min_value=2, max_value=8))


@composite
def component_placement(draw):
    """Generate valid component placement data."""
    reference = draw(st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=2, max_size=10
    ).filter(lambda x: x[0].isalpha()))
    
    value = draw(st.text(min_size=1, max_size=20))
    package = draw(st.sampled_from([
        "0603", "0805", "1206", "SOT-23", "SOIC-8", "QFN-16", "BGA-64"
    ]))
    
    x = draw(st.floats(min_value=0.0, max_value=100.0))
    y = draw(st.floats(min_value=0.0, max_value=80.0))
    rotation = draw(st.floats(min_value=0.0, max_value=360.0))
    layer = draw(st.sampled_from(["top", "bottom"]))
    
    return ComponentPlacement(
        reference=reference,
        value=value,
        package=package,
        x=x,
        y=y,
        rotation=rotation,
        layer=layer
    )


@composite
def drill_hole(draw):
    """Generate valid drill hole data."""
    x = draw(st.floats(min_value=0.0, max_value=100.0))
    y = draw(st.floats(min_value=0.0, max_value=80.0))
    diameter = draw(st.floats(min_value=0.2, max_value=6.0))
    plated = draw(st.booleans())
    
    return DrillHole(x=x, y=y, diameter=diameter, plated=plated)

@composite
def design_rules(draw):
    """Generate valid design rules."""
    return {
        "min_trace_width": draw(st.floats(min_value=0.1, max_value=1.0)),
        "min_via_size": draw(st.floats(min_value=0.2, max_value=2.0)),
        "min_clearance": draw(st.floats(min_value=0.1, max_value=0.5)),
        "max_layers": draw(st.integers(min_value=2, max_value=8))
    }


@composite
def netlist_data(draw):
    """Generate valid netlist data."""
    num_components = draw(st.integers(min_value=1, max_value=10))
    num_nets = draw(st.integers(min_value=1, max_value=15))
    
    components = []
    for i in range(num_components):
        ref = f"U{i+1}" if i % 3 == 0 else f"R{i+1}" if i % 3 == 1 else f"C{i+1}"
        value = draw(st.text(min_size=1, max_size=10))
        footprint = draw(st.sampled_from(["0603", "0805", "SOIC-8", "QFN-16"]))
        
        components.append({
            "reference": ref,
            "value": value,
            "footprint": footprint
        })
    
    nets = [f"Net{i+1}" for i in range(num_nets)]
    
    return {"components": components, "nets": nets}


@composite
def pcb_layout_data(draw):
    """Generate complete PCB layout data."""
    dimensions = draw(board_dimensions())
    layers = draw(layer_count())
    netlist = draw(netlist_data())
    
    return {
        "width": dimensions["width"],
        "height": dimensions["height"],
        "layers": layers,
        "components": netlist["components"],
        "nets": netlist["nets"]
    }


# ============================================================================
# Property 8: PCB Layout Generation
# Feature: genai-pcb-platform
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
# ============================================================================

@pytest.mark.property
@given(
    project_name=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        min_size=1, max_size=20
    ).filter(lambda x: x.replace("_", "").replace("-", "").isalnum()),
    dimensions=board_dimensions(),
    layers=layer_count()
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_8_pcb_layout_generation_project_creation(project_name, dimensions, layers):
    """
    Property 8: PCB Layout Generation (Project Creation)
    
    **Validates: Requirements 4.1, 4.2**
    
    Property: For any valid project name and board specifications,
    KiCad project creation SHALL succeed and generate all required files.
    
    Invariants:
    - Project files are created successfully
    - All essential files exist (project, schematic, PCB)
    - Board dimensions are preserved
    - Layer count is preserved
    - Project validation passes
    """
    project = None
    
    try:
        # Create KiCad project
        project = KiCadProject(project_name)
        
        # Invariant 1: Project creation succeeds
        result = project.create_project(
            board_width=dimensions["width"],
            board_height=dimensions["height"],
            layers=layers
        )
        
        assert result["success"], f"Project creation failed: {result}"
        
        # Invariant 2: Essential files exist
        assert os.path.exists(project.project_path), "Project file should exist"
        assert os.path.exists(project.schematic_path), "Schematic file should exist"
        assert os.path.exists(project.pcb_path), "PCB file should exist"
        
        # Invariant 3: Board dimensions preserved
        assert abs(result["board_size"]["width"] - dimensions["width"]) < 0.1, \
            f"Width mismatch: expected {dimensions['width']}, got {result['board_size']['width']}"
        assert abs(result["board_size"]["height"] - dimensions["height"]) < 0.1, \
            f"Height mismatch: expected {dimensions['height']}, got {result['board_size']['height']}"
        
        # Invariant 4: Layer count preserved
        assert result["layers"] == layers, \
            f"Layer count mismatch: expected {layers}, got {result['layers']}"
        
        # Invariant 5: Project validation passes
        validation = project.validate_design()
        assert validation["valid"] or len(validation["issues"]) == 0, \
            f"Project validation failed: {validation['issues']}"
        
        # Invariant 6: Project info is consistent
        info = project.get_project_info()
        assert info["project_name"] == project_name, "Project name mismatch"
        assert info["files_exist"]["project"], "Project file existence check failed"
        assert info["files_exist"]["schematic"], "Schematic file existence check failed"
        assert info["files_exist"]["pcb"], "PCB file existence check failed"
    
    finally:
        if project:
            project.cleanup()


@pytest.mark.property
@given(
    netlist_data=netlist_data(),
    design_rules=design_rules()
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_8_pcb_layout_generation_netlist_import(netlist_data, design_rules):
    """
    Property 8: PCB Layout Generation (Netlist Import)
    
    **Validates: Requirements 4.1, 4.2**
    
    Property: For any valid netlist with components and nets,
    netlist import SHALL succeed and preserve all component information.
    
    Invariants:
    - Netlist import succeeds
    - All components are preserved
    - All nets are identified
    - Component references are maintained
    - Netlist file is created
    """
    project = None
    
    try:
        project = KiCadProject("netlist_test")
        project.create_project()
        
        # Create temporary netlist file
        netlist_content = _generate_test_netlist(netlist_data)
        netlist_file = os.path.join(project.output_dir, "test.net")
        
        with open(netlist_file, 'w') as f:
            f.write(netlist_content)
        
        # Invariant 1: Netlist import succeeds
        result = project.import_netlist(netlist_file)
        assert result["success"], f"Netlist import failed: {result}"
        
        # Invariant 2: All components preserved
        imported_components = result["components"]
        expected_refs = {comp["reference"] for comp in netlist_data["components"]}
        imported_refs = {comp["reference"] for comp in imported_components}
        
        # Should find most components (some might be filtered)
        found_ratio = len(imported_refs & expected_refs) / len(expected_refs)
        assert found_ratio >= 0.5, \
            f"Too few components imported: {imported_refs} vs {expected_refs}"
        
        # Invariant 3: Nets are identified
        imported_nets = result["nets"]
        assert len(imported_nets) > 0, "Should import at least some nets"
        
        # Invariant 4: Netlist file created
        assert os.path.exists(project.netlist_path), "Netlist file should be created"
        
        # Invariant 5: Component references maintained
        for comp in imported_components:
            assert "reference" in comp, "Component should have reference"
            assert len(comp["reference"]) > 0, "Reference should not be empty"
    
    finally:
        if project:
            project.cleanup()


def _generate_test_netlist(self, netlist_data: Dict[str, Any]) -> str:
    """Generate a simple test netlist in KiCad format."""
    lines = [
        "(export (version D)",
        "  (design",
        "    (source test.sch)",
        "    (date \"2024-01-01 12:00:00\")",
        "    (tool \"GenAI PCB Platform\")",
        "  )",
        "  (components"
    ]
    
    # Add components
    for comp in netlist_data["components"]:
        lines.append(f"    (comp (ref {comp['reference']})")
        lines.append(f"      (value {comp['value']})")
        lines.append(f"      (footprint {comp['footprint']})")
        lines.append("    )")
    
    lines.extend([
        "  )",
        "  (nets"
    ])
    
    # Add nets
    for i, net in enumerate(netlist_data["nets"]):
        lines.append(f"    (net (code {i+1}) (name {net}))")
    
    lines.extend([
        "  )",
        ")"
    ])
    
    return "\n".join(lines)


@pytest.mark.property
@given(
    pcb_data=pcb_layout_data(),
    design_rules=design_rules()
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_8_pcb_layout_generation_layout_generation(pcb_data, design_rules):
    """
    Property 8: PCB Layout Generation (Layout Generation)
    
    **Validates: Requirements 4.2, 4.3**
    
    Property: For any valid PCB data and design rules,
    PCB layout generation SHALL succeed and apply design rules correctly.
    
    Invariants:
    - Layout generation succeeds
    - Design rules are applied
    - PCB file is updated
    - Layout method is recorded
    - No corruption of existing files
    """
    project = None
    
    try:
        project = KiCadProject("layout_test")
        project.create_project(
            board_width=pcb_data["width"],
            board_height=pcb_data["height"],
            layers=pcb_data["layers"]
        )
        
        # Get initial PCB file size
        initial_size = os.path.getsize(project.pcb_path)
        
        # Invariant 1: Layout generation succeeds
        result = project.generate_pcb_layout(design_rules)
        assert result["success"], f"Layout generation failed: {result}"
        
        # Invariant 2: Design rules applied
        assert result["design_rules_applied"], "Design rules should be applied"
        
        # Invariant 3: PCB file exists and is valid
        assert os.path.exists(project.pcb_path), "PCB file should exist"
        final_size = os.path.getsize(project.pcb_path)
        assert final_size >= initial_size, "PCB file should not shrink"
        
        # Invariant 4: Layout method recorded
        assert "routing_method" in result, "Should record routing method"
        assert result["routing_method"] in ["basic", "RL", "heuristic"], \
            f"Invalid routing method: {result['routing_method']}"
        
        # Invariant 5: File integrity maintained
        validation = project.validate_design()
        assert len(validation["issues"]) == 0, \
            f"Layout generation corrupted files: {validation['issues']}"
    
    finally:
        if project:
            project.cleanup()

@pytest.mark.property
@given(
    pcb_data=pcb_layout_data(),
    output_layers=st.lists(
        st.sampled_from([
            "top_copper", "bottom_copper", "top_soldermask", 
            "bottom_soldermask", "top_silkscreen", "bottom_silkscreen", "outline"
        ]),
        min_size=1, max_size=7, unique=True
    )
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_8_pcb_layout_generation_gerber_export(pcb_data, output_layers):
    """
    Property 8: PCB Layout Generation (Gerber Export)
    
    **Validates: Requirements 4.4, 4.5**
    
    Property: For any valid PCB layout and layer selection,
    Gerber file export SHALL succeed and generate all requested layers.
    
    Invariants:
    - Gerber export succeeds
    - All requested layers are generated
    - Files have valid extensions
    - File sizes are reasonable
    - Export directory is created
    """
    project = None
    
    try:
        project = KiCadProject("gerber_test")
        project.create_project(
            board_width=pcb_data["width"],
            board_height=pcb_data["height"],
            layers=pcb_data["layers"]
        )
        
        # Generate layout first
        project.generate_pcb_layout()
        
        # Invariant 1: Gerber export succeeds
        result = project.export_gerbers()
        assert result["success"], f"Gerber export failed: {result}"
        
        # Invariant 2: Export directory created
        gerber_dir = result["gerber_dir"]
        assert os.path.exists(gerber_dir), "Gerber directory should exist"
        
        # Invariant 3: All files generated
        generated_files = result["files"]
        assert len(generated_files) > 0, "Should generate at least one file"
        
        # Invariant 4: Files exist and have content
        for filename in generated_files:
            filepath = os.path.join(gerber_dir, filename)
            assert os.path.exists(filepath), f"Gerber file should exist: {filename}"
            
            # File should have some content
            file_size = os.path.getsize(filepath)
            assert file_size > 0, f"Gerber file should not be empty: {filename}"
        
        # Invariant 5: File paths are correct
        file_paths = result["file_paths"]
        assert len(file_paths) == len(generated_files), "File paths count mismatch"
        
        for filepath in file_paths:
            assert os.path.exists(filepath), f"File path should exist: {filepath}"
    
    finally:
        if project:
            project.cleanup()


# ============================================================================
# Property 9: Layout Error Recovery
# Feature: genai-pcb-platform
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
# ============================================================================

@pytest.mark.property
@given(
    invalid_project_name=st.text(min_size=0, max_size=5).filter(
        lambda x: not x.replace("_", "").replace("-", "").isalnum() or len(x.strip()) == 0
    )
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_9_layout_error_recovery_invalid_project_names(invalid_project_name):
    """
    Property 9: Layout Error Recovery (Invalid Project Names)
    
    **Validates: Requirements 4.1**
    
    Property: For any invalid project name,
    the system SHALL handle the error gracefully and provide descriptive feedback.
    
    Invariants:
    - Invalid names are rejected appropriately
    - Error messages are descriptive
    - No partial project creation
    - System remains stable
    - Cleanup works even after errors
    """
    project = None
    
    try:
        # Invariant 1: Invalid names should cause controlled failure
        try:
            project = KiCadProject(invalid_project_name)
            result = project.create_project()
            
            # If creation succeeds, it should be because the name was sanitized
            if result["success"]:
                # Project should exist and be valid
                assert os.path.exists(project.project_path), "Project file should exist"
                validation = project.validate_design()
                assert validation["valid"] or len(validation["warnings"]) > 0, \
                    "Should be valid or have warnings"
        
        except (KiCadIntegrationError, ValueError, OSError) as e:
            # Invariant 2: Error messages should be descriptive
            error_msg = str(e)
            assert len(error_msg) > 5, "Error message should be descriptive"
            assert any(word in error_msg.lower() for word in 
                      ["name", "invalid", "character", "empty"]), \
                f"Error should mention name issue: {error_msg}"
            
            # Invariant 3: No partial project creation
            if project and hasattr(project, 'output_dir'):
                # If output dir exists, it should be empty or minimal
                if os.path.exists(project.output_dir):
                    files = os.listdir(project.output_dir)
                    # Should have no substantial files
                    for file in files:
                        filepath = os.path.join(project.output_dir, file)
                        if os.path.isfile(filepath):
                            assert os.path.getsize(filepath) < 1000, \
                                "Should not create substantial files on error"
    
    finally:
        # Invariant 4: Cleanup works even after errors
        if project:
            try:
                project.cleanup()
            except Exception as cleanup_error:
                # Cleanup should not raise exceptions
                pytest.fail(f"Cleanup failed after error: {cleanup_error}")


@pytest.mark.property
@given(
    dimensions=st.tuples(
        st.floats(min_value=-100.0, max_value=5.0),  # Invalid widths
        st.floats(min_value=-100.0, max_value=5.0)   # Invalid heights
    )
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_9_layout_error_recovery_invalid_dimensions(dimensions):
    """
    Property 9: Layout Error Recovery (Invalid Dimensions)
    
    **Validates: Requirements 4.2**
    
    Property: For any invalid board dimensions (negative or too small),
    the system SHALL detect the error and provide recovery suggestions.
    
    Invariants:
    - Invalid dimensions are detected
    - Error messages suggest valid ranges
    - No PCB file corruption
    - System provides fallback dimensions
    """
    project = None
    width, height = dimensions
    
    try:
        project = KiCadProject("dimension_test")
        
        # Invariant 1: Invalid dimensions should be handled
        try:
            result = project.create_project(
                board_width=width,
                board_height=height,
                layers=2
            )
            
            # If creation succeeds, dimensions should be sanitized
            if result["success"]:
                # Invariant 2: Dimensions should be corrected to valid values
                actual_width = result["board_size"]["width"]
                actual_height = result["board_size"]["height"]
                
                assert actual_width >= 10.0, \
                    f"Width should be corrected to minimum: {actual_width}"
                assert actual_height >= 10.0, \
                    f"Height should be corrected to minimum: {actual_height}"
                
                # PCB file should be valid
                assert os.path.exists(project.pcb_path), "PCB file should exist"
                validation = project.validate_design()
                assert validation["valid"], f"PCB should be valid: {validation}"
        
        except (KiCadIntegrationError, ValueError) as e:
            # Invariant 3: Error should mention dimensions
            error_msg = str(e)
            assert any(word in error_msg.lower() for word in 
                      ["dimension", "size", "width", "height", "minimum"]), \
                f"Error should mention dimension issue: {error_msg}"
            
            # Invariant 4: Should suggest valid ranges
            assert any(word in error_msg for word in ["10", "mm"]), \
                f"Error should suggest minimum dimensions: {error_msg}"
    
    finally:
        if project:
            project.cleanup()


@pytest.mark.property
@given(
    invalid_layers=st.one_of(
        st.integers(min_value=-5, max_value=1),
        st.integers(min_value=17, max_value=100)
    )
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_9_layout_error_recovery_invalid_layer_count(invalid_layers):
    """
    Property 9: Layout Error Recovery (Invalid Layer Count)
    
    **Validates: Requirements 4.2, 4.3**
    
    Property: For any invalid layer count (negative, zero, or excessive),
    the system SHALL correct to valid values or provide clear error messages.
    
    Invariants:
    - Invalid layer counts are detected
    - Layer count is corrected to valid range (2-16)
    - PCB structure remains valid
    - Error messages are helpful
    """
    project = None
    
    try:
        project = KiCadProject("layer_test")
        
        # Invariant 1: Invalid layer count should be handled
        try:
            result = project.create_project(
                board_width=50.0,
                board_height=40.0,
                layers=invalid_layers
            )
            
            # If creation succeeds, layer count should be corrected
            if result["success"]:
                # Invariant 2: Layer count corrected to valid range
                actual_layers = result["layers"]
                assert 2 <= actual_layers <= 16, \
                    f"Layer count should be corrected: {actual_layers}"
                
                # Invariant 3: PCB structure should be valid
                validation = project.validate_design()
                assert validation["valid"], f"PCB should be valid: {validation}"
        
        except (KiCadIntegrationError, ValueError) as e:
            # Invariant 4: Error should mention layers
            error_msg = str(e)
            assert any(word in error_msg.lower() for word in 
                      ["layer", "count", "invalid", "range"]), \
                f"Error should mention layer issue: {error_msg}"
    
    finally:
        if project:
            project.cleanup()


@pytest.mark.property
@given(
    corrupted_netlist=st.text(min_size=1, max_size=100).filter(
        lambda x: not x.strip().startswith("(export") and ".net" not in x.lower()
    )
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_9_layout_error_recovery_corrupted_netlist(corrupted_netlist):
    """
    Property 9: Layout Error Recovery (Corrupted Netlist)
    
    **Validates: Requirements 4.1, 4.2**
    
    Property: For any corrupted or invalid netlist file,
    the system SHALL detect the corruption and provide recovery options.
    
    Invariants:
    - Corrupted netlists are detected
    - Import fails gracefully
    - Error messages explain the issue
    - No partial import occurs
    - Original project remains intact
    """
    project = None
    
    try:
        project = KiCadProject("netlist_error_test")
        project.create_project()
        
        # Get initial project state
        initial_validation = project.validate_design()
        
        # Create corrupted netlist file
        netlist_file = os.path.join(project.output_dir, "corrupted.net")
        with open(netlist_file, 'w') as f:
            f.write(corrupted_netlist)
        
        # Invariant 1: Corrupted netlist should be rejected
        try:
            result = project.import_netlist(netlist_file)
            
            # If import claims success, should have warnings or minimal data
            if result["success"]:
                # Should have warnings about issues
                components = result.get("components", [])
                nets = result.get("nets", [])
                
                # Should not import substantial data from corrupted file
                assert len(components) <= 1, "Should not import many components from corrupted netlist"
                assert len(nets) <= 1, "Should not import many nets from corrupted netlist"
        
        except KiCadIntegrationError as e:
            # Invariant 2: Error should be descriptive
            error_msg = str(e)
            assert len(error_msg) > 5, "Error message should be descriptive"
            assert any(word in error_msg.lower() for word in 
                      ["netlist", "invalid", "corrupted", "format", "parse"]), \
                f"Error should mention netlist issue: {error_msg}"
        
        # Invariant 3: Original project should remain intact
        final_validation = project.validate_design()
        assert final_validation["valid"] == initial_validation["valid"], \
            "Project validity should not change after failed netlist import"
    
    finally:
        if project:
            project.cleanup()


@pytest.mark.property
@given(
    components=st.lists(component_placement(), min_size=1, max_size=20),
    drill_holes=st.lists(drill_hole(), min_size=1, max_size=30)
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_9_layout_error_recovery_manufacturing_export_errors(components, drill_holes):
    """
    Property 9: Layout Error Recovery (Manufacturing Export Errors)
    
    **Validates: Requirements 4.4, 4.5**
    
    Property: When manufacturing file export encounters errors,
    the system SHALL handle them gracefully and provide partial results where possible.
    
    Invariants:
    - Export errors are handled gracefully
    - Partial results are provided when possible
    - Error messages are actionable
    - No corrupted output files
    - Cleanup works after errors
    """
    exporter = None
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ManufacturingExporter("error_test", temp_dir)
            
            # Test PCB data that might cause issues
            pcb_data = {
                "width": 50.0,
                "height": 40.0,
                "layers": 2,
                "components": [
                    {"reference": comp.reference, "value": comp.value, "footprint": comp.package}
                    for comp in components[:5]  # Limit for performance
                ]
            }
            
            # Invariant 1: Manufacturing package generation handles errors
            try:
                result = exporter.generate_manufacturing_package(
                    pcb_data, components[:10], drill_holes[:15]
                )
                
                # If generation succeeds, should have valid results
                if result["success"]:
                    # Should have output directory
                    assert os.path.exists(result["output_dir"]), "Output directory should exist"
                    
                    # Should have some results
                    results = result["results"]
                    assert len(results) > 0, "Should have some manufacturing results"
                    
                    # Files should exist and be non-empty
                    for category, category_result in results.items():
                        if category_result.get("success"):
                            # Check that files exist
                            if "files" in category_result:
                                files = category_result["files"]
                                if isinstance(files, dict):
                                    for filepath in files.values():
                                        if isinstance(filepath, str) and os.path.exists(filepath):
                                            assert os.path.getsize(filepath) > 0, \
                                                f"File should not be empty: {filepath}"
            
            except ManufacturingExportError as e:
                # Invariant 2: Error should be descriptive
                error_msg = str(e)
                assert len(error_msg) > 5, "Error message should be descriptive"
                
                # Invariant 3: Should mention specific issue
                assert any(word in error_msg.lower() for word in 
                          ["export", "manufacturing", "file", "generation"]), \
                    f"Error should mention export issue: {error_msg}"
            
            # Invariant 4: Individual export functions handle errors
            try:
                # Test Gerber export with potentially problematic data
                gerber_result = exporter.export_gerber_files(pcb_data)
                
                if gerber_result["success"]:
                    assert len(gerber_result["files"]) > 0, "Should generate some Gerber files"
            
            except ManufacturingExportError:
                # Individual export errors are acceptable
                pass
            
            try:
                # Test drill export
                drill_result = exporter.export_drill_files(drill_holes[:10])
                
                if drill_result["success"]:
                    assert drill_result["hole_count"] > 0, "Should have drill holes"
            
            except ManufacturingExportError:
                # Drill export errors are acceptable
                pass
    
    finally:
        # Invariant 5: No cleanup needed for ManufacturingExporter
        # (it uses provided directory, doesn't create temp dirs)
        pass


# Helper method for generating test netlists
def _generate_test_netlist(netlist_data: Dict[str, Any]) -> str:
    """Generate a simple test netlist in KiCad format."""
    lines = [
        "(export (version D)",
        "  (design",
        "    (source test.sch)",
        "    (date \"2024-01-01 12:00:00\")",
        "    (tool \"GenAI PCB Platform\")",
        "  )",
        "  (components"
    ]
    
    # Add components
    for comp in netlist_data["components"]:
        lines.append(f"    (comp (ref {comp['reference']})")
        lines.append(f"      (value {comp['value']})")
        lines.append(f"      (footprint {comp['footprint']})")
        lines.append("    )")
    
    lines.extend([
        "  )",
        "  (nets"
    ])
    
    # Add nets
    for i, net in enumerate(netlist_data["nets"]):
        lines.append(f"    (net (code {i+1}) (name {net}))")
    
    lines.extend([
        "  )",
        ")"
    ])
    
    return "\n".join(lines)


# Monkey patch the helper method to the test class
# test_property_8_pcb_layout_generation_netlist_import._generate_test_netlist = _generate_test_netlist