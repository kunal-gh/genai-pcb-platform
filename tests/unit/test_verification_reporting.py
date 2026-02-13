"""
Unit tests for verification reporting system.

Tests VerificationReporter class functionality.
"""

import pytest
import os
import tempfile
import json
from src.services.verification_reporting import (
    VerificationReporter,
    ReportFormat,
    VerificationSummary
)


@pytest.fixture
def reporter():
    """Create verification reporter."""
    return VerificationReporter()


@pytest.fixture
def sample_erc_drc_results():
    """Sample ERC/DRC results."""
    return {
        "success": True,
        "ready_for_manufacturing": False,
        "summary": {
            "total_violations": 3,
            "errors": 2,
            "warnings": 1
        },
        "violations": [
            {
                "type": "erc_error",
                "severity": "error",
                "message": "Unconnected pin on component R1",
                "component": "R1",
                "suggested_fix": "Connect pin 2 of R1"
            },
            {
                "type": "drc_error",
                "severity": "error",
                "message": "Trace width below minimum",
                "net": "VCC",
                "rule_name": "min_trace_width",
                "suggested_fix": "Increase trace width to 0.1mm"
            },
            {
                "type": "erc_warning",
                "severity": "warning",
                "message": "No ground net detected",
                "suggested_fix": "Add ground connections"
            }
        ]
    }


@pytest.fixture
def sample_dfm_results():
    """Sample DFM results."""
    return {
        "success": True,
        "manufacturable": False,
        "score": 75.5,
        "confidence_level": "fair",
        "summary": {
            "total_violations": 4,
            "critical": 1,
            "high": 2,
            "medium": 1
        },
        "violations": [
            {
                "severity": "critical",
                "category": "trace",
                "message": "Trace width 0.05mm below minimum 0.1mm",
                "net": "SIGNAL",
                "recommendation": "Increase trace width to at least 0.1mm",
                "cost_impact": "high"
            },
            {
                "severity": "high",
                "category": "via",
                "message": "Via annular ring insufficient",
                "location": (10.0, 20.0),
                "recommendation": "Increase via pad size",
                "cost_impact": "medium"
            },
            {
                "severity": "high",
                "category": "component",
                "message": "Component too close to board edge",
                "component": "U1",
                "recommendation": "Move component away from edge",
                "cost_impact": "low"
            },
            {
                "severity": "medium",
                "category": "signal_integrity",
                "message": "High-speed trace may have impedance issues",
                "net": "CLK",
                "recommendation": "Use controlled impedance traces",
                "cost_impact": "medium"
            }
        ]
    }


@pytest.fixture
def good_erc_drc_results():
    """ERC/DRC results with no violations."""
    return {
        "success": True,
        "ready_for_manufacturing": True,
        "summary": {
            "total_violations": 0,
            "errors": 0,
            "warnings": 0
        },
        "violations": []
    }


@pytest.fixture
def good_dfm_results():
    """DFM results with high score."""
    return {
        "success": True,
        "manufacturable": True,
        "score": 98.5,
        "confidence_level": "excellent",
        "summary": {
            "total_violations": 0,
            "critical": 0,
            "high": 0,
            "medium": 0
        },
        "violations": []
    }


@pytest.fixture
def design_info():
    """Sample design information."""
    return {
        "name": "Test PCB",
        "version": "1.0",
        "designer": "Test User",
        "board_size": "100x80mm",
        "layers": 2
    }


class TestVerificationSummary:
    """Tests for VerificationSummary dataclass."""
    
    def test_summary_creation(self):
        """Test summary creation."""
        summary = VerificationSummary(
            total_violations=10,
            critical_count=2,
            error_count=5,
            warning_count=3,
            info_count=0,
            erc_violations=3,
            drc_violations=3,
            dfm_violations=4,
            design_ready=False,
            manufacturability_score=75.5,
            confidence_level="fair"
        )
        
        assert summary.total_violations == 10
        assert summary.critical_count == 2
        assert summary.design_ready is False
        assert summary.manufacturability_score == 75.5


class TestVerificationReporter:
    """Tests for VerificationReporter class."""
    
    def test_init(self, reporter):
        """Test initialization."""
        assert reporter.report_data == {}
    
    def test_generate_report_basic(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test basic report generation."""
        report = reporter.generate_report(sample_erc_drc_results, sample_dfm_results)
        
        assert "timestamp" in report
        assert "summary" in report
        assert "violations" in report
        assert "recommendations" in report
        assert "next_steps" in report
    
    def test_generate_report_with_design_info(self, reporter, sample_erc_drc_results, sample_dfm_results, design_info):
        """Test report generation with design info."""
        report = reporter.generate_report(
            sample_erc_drc_results,
            sample_dfm_results,
            design_info=design_info
        )
        
        assert report["design_info"] == design_info
        assert report["design_info"]["name"] == "Test PCB"
    
    def test_combine_violations(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test violation combination."""
        combined = reporter._combine_violations(sample_erc_drc_results, sample_dfm_results)
        
        # Should have 3 ERC/DRC + 4 DFM = 7 total
        assert len(combined) == 7
        
        # Check sources
        erc_drc_count = len([v for v in combined if v["source"] == "ERC/DRC"])
        dfm_count = len([v for v in combined if v["source"] == "DFM"])
        assert erc_drc_count == 3
        assert dfm_count == 4
        
        # Check that all violations have required fields
        for v in combined:
            assert "source" in v
            assert "severity" in v
            assert "message" in v
            assert "priority" in v
    
    def test_categorize_violations(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test violation categorization."""
        combined = reporter._combine_violations(sample_erc_drc_results, sample_dfm_results)
        categorized = reporter._categorize_violations(combined)
        
        assert isinstance(categorized, dict)
        assert len(categorized) > 0
        
        # Check that categories exist
        assert "trace" in categorized or "electrical" in categorized
    
    def test_prioritize_violations(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test violation prioritization."""
        combined = reporter._combine_violations(sample_erc_drc_results, sample_dfm_results)
        prioritized = reporter._prioritize_violations(combined)
        
        assert "critical" in prioritized
        assert "high" in prioritized
        assert "medium" in prioritized
        assert "low" in prioritized
        
        # Check that critical violations exist
        assert len(prioritized["critical"]) > 0
    
    def test_calculate_priority(self, reporter):
        """Test priority calculation."""
        assert reporter._calculate_priority({"severity": "critical"}) == "critical"
        assert reporter._calculate_priority({"severity": "error"}) == "critical"
        assert reporter._calculate_priority({"severity": "high"}) == "high"
        assert reporter._calculate_priority({"severity": "warning"}) == "high"
        assert reporter._calculate_priority({"severity": "medium"}) == "medium"
        assert reporter._calculate_priority({"severity": "info"}) == "medium"
        assert reporter._calculate_priority({"severity": "low"}) == "low"
    
    def test_map_category(self, reporter):
        """Test category mapping."""
        assert reporter._map_category("erc_error") == "electrical"
        assert reporter._map_category("drc_warning") == "design_rule"
        assert reporter._map_category("connectivity_error") == "connectivity"
        assert reporter._map_category("unknown") == "other"
    
    def test_generate_summary(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test summary generation."""
        combined = reporter._combine_violations(sample_erc_drc_results, sample_dfm_results)
        summary = reporter._generate_summary(sample_erc_drc_results, sample_dfm_results, combined)
        
        assert isinstance(summary, VerificationSummary)
        assert summary.total_violations == 7
        assert summary.critical_count > 0
        assert summary.design_ready is False
        assert summary.manufacturability_score == 75.5
        assert summary.confidence_level == "fair"
    
    def test_generate_summary_good_design(self, reporter, good_erc_drc_results, good_dfm_results):
        """Test summary for good design."""
        combined = reporter._combine_violations(good_erc_drc_results, good_dfm_results)
        summary = reporter._generate_summary(good_erc_drc_results, good_dfm_results, combined)
        
        assert summary.total_violations == 0
        assert summary.critical_count == 0
        assert summary.design_ready is True
        assert summary.manufacturability_score == 98.5
        assert summary.confidence_level == "excellent"
    
    def test_generate_recommendations(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test recommendation generation."""
        combined = reporter._combine_violations(sample_erc_drc_results, sample_dfm_results)
        recommendations = reporter._generate_recommendations(combined, sample_dfm_results)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert "priority" in rec
            assert "category" in rec
            assert "title" in rec
            assert "description" in rec
    
    def test_generate_next_steps_not_ready(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test next steps for design not ready."""
        combined = reporter._combine_violations(sample_erc_drc_results, sample_dfm_results)
        summary = reporter._generate_summary(sample_erc_drc_results, sample_dfm_results, combined)
        steps = reporter._generate_next_steps(summary, combined)
        
        assert isinstance(steps, list)
        assert len(steps) > 0
        
        # Should include fix instructions
        assert any("Fix" in step or "fix" in step for step in steps)
    
    def test_generate_next_steps_ready(self, reporter, good_erc_drc_results, good_dfm_results):
        """Test next steps for design ready."""
        combined = reporter._combine_violations(good_erc_drc_results, good_dfm_results)
        summary = reporter._generate_summary(good_erc_drc_results, good_dfm_results, combined)
        steps = reporter._generate_next_steps(summary, combined)
        
        assert isinstance(steps, list)
        assert len(steps) > 0
        
        # Should include manufacturing instructions
        assert any("ready" in step.lower() for step in steps)
    
    def test_format_html(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test HTML formatting."""
        report = reporter.generate_report(sample_erc_drc_results, sample_dfm_results, format=ReportFormat.HTML)
        
        assert "formatted_output" in report
        html = report["formatted_output"]
        
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "Verification Report" in html
        assert "</html>" in html
    
    def test_format_text(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test text formatting."""
        report = reporter.generate_report(sample_erc_drc_results, sample_dfm_results, format=ReportFormat.TEXT)
        
        assert "formatted_output" in report
        text = report["formatted_output"]
        
        assert "VERIFICATION REPORT" in text
        assert "SUMMARY" in text
        assert "NEXT STEPS" in text
    
    def test_format_markdown(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test Markdown formatting."""
        report = reporter.generate_report(sample_erc_drc_results, sample_dfm_results, format=ReportFormat.MARKDOWN)
        
        assert "formatted_output" in report
        md = report["formatted_output"]
        
        assert "# PCB Design Verification Report" in md
        assert "## Summary" in md
        assert "## Next Steps" in md
    
    def test_format_json(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test JSON formatting (default)."""
        report = reporter.generate_report(sample_erc_drc_results, sample_dfm_results, format=ReportFormat.JSON)
        
        # JSON format doesn't add formatted_output
        assert "formatted_output" not in report
        assert "summary" in report
        assert "violations" in report
    
    def test_export_report_json(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test JSON report export."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            reporter.generate_report(sample_erc_drc_results, sample_dfm_results)
            success = reporter.export_report(temp_path, ReportFormat.JSON)
            
            assert success is True
            assert os.path.exists(temp_path)
            
            # Verify JSON content
            with open(temp_path, 'r') as f:
                data = json.load(f)
            assert "summary" in data
            assert "violations" in data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_report_html(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test HTML report export."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            temp_path = f.name
        
        try:
            reporter.generate_report(sample_erc_drc_results, sample_dfm_results, format=ReportFormat.HTML)
            success = reporter.export_report(temp_path, ReportFormat.HTML)
            
            assert success is True
            assert os.path.exists(temp_path)
            
            # Verify HTML content
            with open(temp_path, 'r') as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_report_text(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test text report export."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name
        
        try:
            reporter.generate_report(sample_erc_drc_results, sample_dfm_results, format=ReportFormat.TEXT)
            success = reporter.export_report(temp_path, ReportFormat.TEXT)
            
            assert success is True
            assert os.path.exists(temp_path)
            
            # Verify text content
            with open(temp_path, 'r') as f:
                content = f.read()
            assert "VERIFICATION REPORT" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_report_markdown(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test Markdown report export."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name
        
        try:
            reporter.generate_report(sample_erc_drc_results, sample_dfm_results, format=ReportFormat.MARKDOWN)
            success = reporter.export_report(temp_path, ReportFormat.MARKDOWN)
            
            assert success is True
            assert os.path.exists(temp_path)
            
            # Verify Markdown content
            with open(temp_path, 'r') as f:
                content = f.read()
            assert "# PCB Design Verification Report" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_report_no_data(self, reporter):
        """Test export without generating report first."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            success = reporter.export_report(temp_path, ReportFormat.JSON)
            assert success is False
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_violation_statistics(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test violation statistics."""
        reporter.generate_report(sample_erc_drc_results, sample_dfm_results)
        stats = reporter.get_violation_statistics()
        
        assert "total" in stats
        assert "by_source" in stats
        assert "by_category" in stats
        assert "by_severity" in stats
        
        assert stats["total"] == 7
        assert "ERC/DRC" in stats["by_source"]
        assert "DFM" in stats["by_source"]
    
    def test_get_violation_statistics_no_data(self, reporter):
        """Test statistics without report data."""
        stats = reporter.get_violation_statistics()
        assert stats == {}
    
    def test_error_handling(self, reporter):
        """Test error handling in report generation."""
        # Test with invalid data
        result = reporter.generate_report(None, None)
        
        assert result["success"] is False
        assert "error" in result
    
    def test_report_completeness(self, reporter, sample_erc_drc_results, sample_dfm_results):
        """Test that report contains all required sections."""
        report = reporter.generate_report(sample_erc_drc_results, sample_dfm_results)
        
        # Check all required sections
        assert "timestamp" in report
        assert "design_info" in report
        assert "summary" in report
        assert "violations" in report
        assert "recommendations" in report
        assert "next_steps" in report
        
        # Check summary completeness
        summary = report["summary"]
        assert "total_violations" in summary
        assert "critical_count" in summary
        assert "design_ready" in summary
        assert "manufacturability_score" in summary
        
        # Check violations structure
        violations = report["violations"]
        assert "by_priority" in violations
        assert "by_category" in violations
        assert "all" in violations