"""
Tests for File Packaging Service
"""

import pytest
import json
import zipfile
from pathlib import Path
from src.services.file_packaging import FilePackager, PackageManifest


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_files(tmp_path):
    """Create sample files for testing"""
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    
    # Create sample files
    schematic = files_dir / "design.kicad_sch"
    schematic.write_text("# Schematic content")
    
    pcb = files_dir / "design.kicad_pcb"
    pcb.write_text("# PCB content")
    
    gerber = files_dir / "design-F_Cu.gbr"
    gerber.write_text("# Gerber content")
    
    return [
        {'path': str(schematic), 'type': 'schematic', 'name': 'design.kicad_sch'},
        {'path': str(pcb), 'type': 'pcb', 'name': 'design.kicad_pcb'},
        {'path': str(gerber), 'type': 'gerber', 'name': 'design-F_Cu.gbr'},
    ]


@pytest.fixture
def packager(temp_output_dir):
    """Create FilePackager instance"""
    return FilePackager(output_dir=str(temp_output_dir))


def test_create_package(packager, sample_files):
    """Test creating a complete design package"""
    zip_path = packager.create_package(
        design_id="test123",
        project_name="Test Project",
        files=sample_files,
        metadata={'author': 'Test User'}
    )
    
    assert Path(zip_path).exists()
    assert zip_path.endswith('.zip')


def test_package_contains_manifest(packager, sample_files):
    """Test that package contains manifest file"""
    zip_path = packager.create_package(
        design_id="test123",
        project_name="Test Project",
        files=sample_files
    )
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        namelist = zipf.namelist()
        assert any('manifest.json' in name for name in namelist)


def test_package_contains_readme(packager, sample_files):
    """Test that package contains README file"""
    zip_path = packager.create_package(
        design_id="test123",
        project_name="Test Project",
        files=sample_files
    )
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        namelist = zipf.namelist()
        assert any('README.md' in name for name in namelist)


def test_files_organized_by_type(packager, sample_files):
    """Test that files are organized into correct subdirectories"""
    zip_path = packager.create_package(
        design_id="test123",
        project_name="Test Project",
        files=sample_files
    )
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        namelist = zipf.namelist()
        
        # Check for organized directories
        assert any('schematics/' in name for name in namelist)
        assert any('pcb/' in name for name in namelist)
        assert any('manufacturing/gerbers/' in name for name in namelist)


def test_manifest_content(packager, sample_files):
    """Test manifest file content"""
    zip_path = packager.create_package(
        design_id="test123",
        project_name="Test Project",
        files=sample_files,
        metadata={'author': 'Test User'}
    )
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Find and read manifest
        manifest_name = [n for n in zipf.namelist() if 'manifest.json' in n][0]
        manifest_data = json.loads(zipf.read(manifest_name))
        
        assert manifest_data['project_name'] == "Test Project"
        assert manifest_data['version'] == "1.0"
        assert 'created_at' in manifest_data
        assert len(manifest_data['files']) == 3
        assert manifest_data['metadata']['author'] == 'Test User'


def test_readme_content(packager, sample_files):
    """Test README file content"""
    zip_path = packager.create_package(
        design_id="test123",
        project_name="Test Project",
        files=sample_files
    )
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Find and read README
        readme_name = [n for n in zipf.namelist() if 'README.md' in n][0]
        readme_content = zipf.read(readme_name).decode('utf-8')
        
        assert "Test Project" in readme_content
        assert "Schematic Files" in readme_content
        assert "PCB Layout Files" in readme_content
        assert "Manufacturing" in readme_content


def test_sanitize_filename(packager):
    """Test filename sanitization"""
    result = packager._sanitize_filename("test<>:file.txt")
    assert '<' not in result
    assert '>' not in result
    assert ':' not in result


def test_export_to_altium(packager, sample_files, temp_output_dir):
    """Test export to Altium format"""
    output_path = packager.export_to_format(
        source_files=[f['path'] for f in sample_files],
        target_format='altium',
        output_path=str(temp_output_dir / "altium")
    )
    
    assert Path(output_path).exists()
    assert output_path.endswith('.PrjPcb')


def test_export_to_eagle(packager, sample_files, temp_output_dir):
    """Test export to Eagle format"""
    output_path = packager.export_to_format(
        source_files=[f['path'] for f in sample_files],
        target_format='eagle',
        output_path=str(temp_output_dir / "eagle")
    )
    
    assert Path(output_path).exists()
    assert output_path.endswith('.brd')


def test_export_to_orcad(packager, sample_files, temp_output_dir):
    """Test export to OrCAD format"""
    output_path = packager.export_to_format(
        source_files=[f['path'] for f in sample_files],
        target_format='orcad',
        output_path=str(temp_output_dir / "orcad")
    )
    
    assert Path(output_path).exists()
    assert output_path.endswith('.dsn')


def test_export_to_ipc2581(packager, sample_files, temp_output_dir):
    """Test export to IPC-2581 format"""
    output_path = packager.export_to_format(
        source_files=[f['path'] for f in sample_files],
        target_format='ipc2581',
        output_path=str(temp_output_dir / "ipc2581")
    )
    
    assert Path(output_path).exists()
    assert output_path.endswith('.xml')
    
    # Verify XML structure
    content = Path(output_path).read_text()
    assert '<?xml version="1.0"' in content
    assert 'IPC-2581' in content


def test_export_to_odbpp(packager, sample_files, temp_output_dir):
    """Test export to ODB++ format"""
    output_path = packager.export_to_format(
        source_files=[f['path'] for f in sample_files],
        target_format='odbpp',
        output_path=str(temp_output_dir / "odbpp")
    )
    
    assert Path(output_path).exists()
    assert Path(output_path).is_dir()
    
    # Verify ODB++ directory structure
    assert (Path(output_path) / "steps").exists()
    assert (Path(output_path) / "fonts").exists()
    assert (Path(output_path) / "symbols").exists()
    assert (Path(output_path) / "matrix" / "matrix").exists()


def test_export_unsupported_format(packager, sample_files, temp_output_dir):
    """Test export with unsupported format raises error"""
    with pytest.raises(ValueError, match="Unsupported format"):
        packager.export_to_format(
            source_files=[f['path'] for f in sample_files],
            target_format='unsupported',
            output_path=str(temp_output_dir / "unsupported")
        )


def test_add_documentation(packager, temp_output_dir):
    """Test adding documentation to package"""
    package_dir = temp_output_dir / "test_package"
    package_dir.mkdir()
    
    packager.add_documentation(
        package_dir=str(package_dir),
        design_notes="These are test design notes",
        specifications={'voltage': '5V', 'layers': 2}
    )
    
    # Check design notes
    notes_file = package_dir / "docs" / "design_notes.md"
    assert notes_file.exists()
    content = notes_file.read_text()
    assert "These are test design notes" in content
    
    # Check specifications
    specs_file = package_dir / "docs" / "specifications.json"
    assert specs_file.exists()
    specs = json.loads(specs_file.read_text())
    assert specs['voltage'] == '5V'
    assert specs['layers'] == 2


def test_package_with_empty_files(packager):
    """Test creating package with no files"""
    zip_path = packager.create_package(
        design_id="test123",
        project_name="Empty Project",
        files=[]
    )
    
    assert Path(zip_path).exists()
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Should still have manifest and README
        namelist = zipf.namelist()
        assert any('manifest.json' in name for name in namelist)
        assert any('README.md' in name for name in namelist)


def test_package_with_special_characters(packager, sample_files):
    """Test package creation with special characters in project name"""
    zip_path = packager.create_package(
        design_id="test123",
        project_name="Test: Project <2024>",
        files=sample_files
    )
    
    assert Path(zip_path).exists()
    # Special characters should be sanitized in filename
    filename = Path(zip_path).name
    assert '<' not in filename
    assert '>' not in filename
    # Note: ':' is allowed in full path (drive letter on Windows)


def test_package_manifest_dataclass():
    """Test PackageManifest dataclass"""
    manifest = PackageManifest(
        project_name="Test",
        created_at="2024-01-01T00:00:00",
        version="1.0",
        description="Test description",
        files=[],
        metadata={}
    )
    
    assert manifest.project_name == "Test"
    assert manifest.version == "1.0"


def test_file_size_in_manifest(packager, sample_files):
    """Test that file sizes are included in manifest"""
    zip_path = packager.create_package(
        design_id="test123",
        project_name="Test Project",
        files=sample_files
    )
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        manifest_name = [n for n in zipf.namelist() if 'manifest.json' in n][0]
        manifest_data = json.loads(zipf.read(manifest_name))
        
        # Check that all files have size information
        for file_info in manifest_data['files']:
            assert 'size' in file_info
            assert file_info['size'] > 0
