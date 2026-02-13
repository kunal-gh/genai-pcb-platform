"""
Verification reporting system.

Integrates ERC/DRC and DFM validation results into comprehensive reports
with clear error explanations, suggested fixes, and priority scoring.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    HTML = "html"
    TEXT = "text"
    MARKDOWN = "markdown"


@dataclass
class VerificationSummary:
    """Summary of verification results."""
    total_violations: int
    critical_count: int
    error_count: int
    warning_count: int
    info_count: int
    erc_violations: int
    drc_violations: int
    dfm_violations: int
    design_ready: bool
    manufacturability_score: float
    confidence_level: str


class VerificationReporter:
    """
    Comprehensive verification reporting system.
    
    Integrates ERC/DRC and DFM validation results into unified reports.
    """
    
    def __init__(self):
        """Initialize verification reporter."""
        self.report_data = {}
        
    def generate_report(
        self,
        erc_drc_results: Dict[str, Any],
        dfm_results: Dict[str, Any],
        design_info: Optional[Dict[str, Any]] = None,
        format: ReportFormat = ReportFormat.JSON
    ) -> Dict[str, Any]:
        """
        Generate comprehensive verification report.
        
        Args:
            erc_drc_results: Results from ERC/DRC verification
            dfm_results: Results from DFM validation
            design_info: Optional design metadata
            format: Output format for report
            
        Returns:
            Comprehensive verification report
        """
        try:
            # Combine and analyze results
            combined_violations = self._combine_violations(erc_drc_results, dfm_results)
            
            # Categorize and prioritize violations
            categorized = self._categorize_violations(combined_violations)
            prioritized = self._prioritize_violations(combined_violations)
            
            # Generate summary
            summary = self._generate_summary(erc_drc_results, dfm_results, combined_violations)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(combined_violations, dfm_results)
            
            # Build report
            report = {
                "timestamp": datetime.now().isoformat(),
                "design_info": design_info or {},
                "summary": {
                    "total_violations": summary.total_violations,
                    "critical_count": summary.critical_count,
                    "error_count": summary.error_count,
                    "warning_count": summary.warning_count,
                    "info_count": summary.info_count,
                    "erc_violations": summary.erc_violations,
                    "drc_violations": summary.drc_violations,
                    "dfm_violations": summary.dfm_violations,
                    "design_ready": summary.design_ready,
                    "manufacturability_score": summary.manufacturability_score,
                    "confidence_level": summary.confidence_level
                },
                "violations": {
                    "by_priority": prioritized,
                    "by_category": categorized,
                    "all": combined_violations
                },
                "recommendations": recommendations,
                "next_steps": self._generate_next_steps(summary, combined_violations)
            }
            
            # Format report
            if format == ReportFormat.HTML:
                report["formatted_output"] = self._format_html(report)
            elif format == ReportFormat.TEXT:
                report["formatted_output"] = self._format_text(report)
            elif format == ReportFormat.MARKDOWN:
                report["formatted_output"] = self._format_markdown(report)
            
            self.report_data = report
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {
                "success": False,
                "error": f"Report generation failed: {str(e)}"
            }
    
    def _combine_violations(
        self,
        erc_drc_results: Dict[str, Any],
        dfm_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combine violations from ERC/DRC and DFM results."""
        combined = []
        
        # Add ERC/DRC violations
        if "violations" in erc_drc_results:
            for violation in erc_drc_results["violations"]:
                combined.append({
                    "source": "ERC/DRC",
                    "type": violation.get("type", "unknown"),
                    "severity": violation.get("severity", "error"),
                    "category": self._map_category(violation.get("type", "")),
                    "message": violation.get("message", ""),
                    "component": violation.get("component"),
                    "net": violation.get("net"),
                    "location": violation.get("location"),
                    "suggested_fix": violation.get("suggested_fix"),
                    "rule_name": violation.get("rule_name"),
                    "priority": self._calculate_priority(violation)
                })
        
        # Add DFM violations
        if "violations" in dfm_results:
            for violation in dfm_results["violations"]:
                combined.append({
                    "source": "DFM",
                    "type": "dfm_violation",
                    "severity": violation.get("severity", "medium"),
                    "category": violation.get("category", "manufacturing"),
                    "message": violation.get("message", ""),
                    "component": violation.get("component"),
                    "net": violation.get("net"),
                    "location": violation.get("location"),
                    "suggested_fix": violation.get("recommendation"),
                    "cost_impact": violation.get("cost_impact"),
                    "priority": self._calculate_priority(violation)
                })
        
        return combined
    
    def _categorize_violations(self, violations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize violations by type."""
        categories = {}
        
        for violation in violations:
            category = violation.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append(violation)
        
        return categories
    
    def _prioritize_violations(self, violations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Prioritize violations by severity and impact."""
        priorities = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        for violation in violations:
            priority = violation.get("priority", "medium")
            if priority in priorities:
                priorities[priority].append(violation)
        
        return priorities
    
    def _calculate_priority(self, violation: Dict[str, Any]) -> str:
        """Calculate priority level for a violation."""
        severity = violation.get("severity", "").lower()
        
        # Map severity to priority
        if severity in ["critical", "error"]:
            return "critical"
        elif severity in ["high", "warning"]:
            return "high"
        elif severity in ["medium", "info"]:
            return "medium"
        else:
            return "low"
    
    def _map_category(self, violation_type: str) -> str:
        """Map violation type to category."""
        type_lower = violation_type.lower()
        
        if "erc" in type_lower:
            return "electrical"
        elif "drc" in type_lower:
            return "design_rule"
        elif "connectivity" in type_lower:
            return "connectivity"
        else:
            return "other"
    
    def _generate_summary(
        self,
        erc_drc_results: Dict[str, Any],
        dfm_results: Dict[str, Any],
        combined_violations: List[Dict[str, Any]]
    ) -> VerificationSummary:
        """Generate verification summary."""
        # Count violations by severity
        critical_count = len([v for v in combined_violations if v.get("priority") == "critical"])
        error_count = len([v for v in combined_violations if v.get("severity") in ["error", "critical"]])
        warning_count = len([v for v in combined_violations if v.get("severity") == "warning"])
        info_count = len([v for v in combined_violations if v.get("severity") == "info"])
        
        # Count by source
        erc_drc_count = len([v for v in combined_violations if v.get("source") == "ERC/DRC"])
        dfm_count = len([v for v in combined_violations if v.get("source") == "DFM"])
        
        # Determine design readiness
        design_ready = (
            erc_drc_results.get("ready_for_manufacturing", False) and
            dfm_results.get("manufacturable", False) and
            critical_count == 0
        )
        
        # Get manufacturability score
        manufacturability_score = dfm_results.get("score", 0.0)
        confidence_level = dfm_results.get("confidence_level", "unknown")
        
        return VerificationSummary(
            total_violations=len(combined_violations),
            critical_count=critical_count,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            erc_violations=erc_drc_count,
            drc_violations=erc_drc_count,  # Combined in ERC/DRC
            dfm_violations=dfm_count,
            design_ready=design_ready,
            manufacturability_score=manufacturability_score,
            confidence_level=confidence_level
        )
    
    def _generate_recommendations(
        self,
        violations: List[Dict[str, Any]],
        dfm_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Group violations by category for targeted recommendations
        by_category = self._categorize_violations(violations)
        
        # Generate category-specific recommendations
        for category, category_violations in by_category.items():
            if not category_violations:
                continue
            
            # Get most common issues in category
            critical_in_category = [v for v in category_violations if v.get("priority") == "critical"]
            
            if critical_in_category:
                recommendations.append({
                    "priority": "critical",
                    "category": category,
                    "title": f"Fix {len(critical_in_category)} critical {category} issue(s)",
                    "description": f"Address critical {category} violations before manufacturing",
                    "violations": [v.get("message") for v in critical_in_category[:3]],
                    "action": "Review and fix all critical violations in this category"
                })
        
        # Add manufacturability recommendations
        score = dfm_results.get("score", 0.0)
        if score < 95.0:
            recommendations.append({
                "priority": "high",
                "category": "manufacturability",
                "title": f"Improve manufacturability score (current: {score:.1f}/100)",
                "description": "Address DFM violations to improve manufacturing success rate",
                "action": "Review DFM violations and apply suggested fixes"
            })
        
        # Add general recommendations
        if len(violations) > 10:
            recommendations.append({
                "priority": "medium",
                "category": "general",
                "title": "High violation count detected",
                "description": f"Design has {len(violations)} total violations",
                "action": "Consider design review or simplification"
            })
        
        return recommendations
    
    def _generate_next_steps(
        self,
        summary: VerificationSummary,
        violations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate next steps based on verification results."""
        steps = []
        
        if summary.design_ready:
            steps.append("✓ Design is ready for manufacturing")
            steps.append("Review manufacturability score and consider optimizations")
            steps.append("Generate manufacturing files (Gerber, drill, pick-and-place)")
            steps.append("Submit to manufacturer for quote")
        else:
            if summary.critical_count > 0:
                steps.append(f"⚠ Fix {summary.critical_count} critical violation(s) first")
            
            if summary.error_count > 0:
                steps.append(f"Fix {summary.error_count} error(s)")
            
            if summary.warning_count > 0:
                steps.append(f"Review {summary.warning_count} warning(s)")
            
            steps.append("Re-run verification after fixes")
            
            if summary.manufacturability_score < 85.0:
                steps.append("Improve manufacturability score to at least 85/100")
        
        return steps
    
    def _format_html(self, report: Dict[str, Any]) -> str:
        """Format report as HTML."""
        summary = report["summary"]
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PCB Design Verification Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; }}
        .summary {{ background: #ecf0f1; padding: 15px; margin: 20px 0; }}
        .ready {{ color: #27ae60; font-weight: bold; }}
        .not-ready {{ color: #e74c3c; font-weight: bold; }}
        .violation {{ border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; }}
        .critical {{ border-color: #c0392b; background: #fadbd8; }}
        .high {{ border-color: #e67e22; background: #fdebd0; }}
        .medium {{ border-color: #f39c12; background: #fcf3cf; }}
        .low {{ border-color: #3498db; background: #d6eaf8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PCB Design Verification Report</h1>
        <p>Generated: {report['timestamp']}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Violations: {summary['total_violations']}</p>
        <p>Manufacturability Score: {summary['manufacturability_score']:.1f}/100 ({summary['confidence_level']})</p>
        <p class="{'ready' if summary['design_ready'] else 'not-ready'}">
            {'✓ Design Ready for Manufacturing' if summary['design_ready'] else '⚠ Design Not Ready'}
        </p>
    </div>
    
    <h2>Violations by Priority</h2>
"""
        
        # Add violations
        for priority in ["critical", "high", "medium", "low"]:
            priority_violations = report["violations"]["by_priority"].get(priority, [])
            if priority_violations:
                html += f"<h3>{priority.title()} Priority ({len(priority_violations)})</h3>"
                for v in priority_violations[:10]:  # Limit to 10 per priority
                    html += f"""
    <div class="violation {priority}">
        <strong>{v.get('message', 'No message')}</strong><br>
        Category: {v.get('category', 'unknown')}<br>
        {f"Component: {v['component']}<br>" if v.get('component') else ""}
        {f"Net: {v['net']}<br>" if v.get('net') else ""}
        {f"<em>Fix: {v['suggested_fix']}</em>" if v.get('suggested_fix') else ""}
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def _format_text(self, report: Dict[str, Any]) -> str:
        """Format report as plain text."""
        summary = report["summary"]
        
        text = f"""
PCB DESIGN VERIFICATION REPORT
{'=' * 50}
Generated: {report['timestamp']}

SUMMARY
{'-' * 50}
Total Violations: {summary['total_violations']}
  Critical: {summary['critical_count']}
  Errors: {summary['error_count']}
  Warnings: {summary['warning_count']}
  Info: {summary['info_count']}

ERC/DRC Violations: {summary['erc_violations']}
DFM Violations: {summary['dfm_violations']}

Manufacturability Score: {summary['manufacturability_score']:.1f}/100
Confidence Level: {summary['confidence_level']}

Design Status: {'✓ READY FOR MANUFACTURING' if summary['design_ready'] else '⚠ NOT READY'}

"""
        
        # Add next steps
        text += "\nNEXT STEPS\n" + "-" * 50 + "\n"
        for step in report["next_steps"]:
            text += f"  {step}\n"
        
        # Add critical violations
        critical = report["violations"]["by_priority"].get("critical", [])
        if critical:
            text += f"\nCRITICAL VIOLATIONS ({len(critical)})\n" + "-" * 50 + "\n"
            for v in critical[:10]:
                text += f"\n• {v.get('message', 'No message')}\n"
                if v.get('suggested_fix'):
                    text += f"  Fix: {v['suggested_fix']}\n"
        
        return text
    
    def _format_markdown(self, report: Dict[str, Any]) -> str:
        """Format report as Markdown."""
        summary = report["summary"]
        
        md = f"""# PCB Design Verification Report

**Generated:** {report['timestamp']}

## Summary

- **Total Violations:** {summary['total_violations']}
  - Critical: {summary['critical_count']}
  - Errors: {summary['error_count']}
  - Warnings: {summary['warning_count']}
  - Info: {summary['info_count']}

- **ERC/DRC Violations:** {summary['erc_violations']}
- **DFM Violations:** {summary['dfm_violations']}

- **Manufacturability Score:** {summary['manufacturability_score']:.1f}/100
- **Confidence Level:** {summary['confidence_level']}

**Design Status:** {'✅ READY FOR MANUFACTURING' if summary['design_ready'] else '⚠️ NOT READY'}

## Next Steps

"""
        
        for step in report["next_steps"]:
            md += f"- {step}\n"
        
        # Add violations by priority
        for priority in ["critical", "high", "medium", "low"]:
            priority_violations = report["violations"]["by_priority"].get(priority, [])
            if priority_violations:
                md += f"\n## {priority.title()} Priority Violations ({len(priority_violations)})\n\n"
                for v in priority_violations[:10]:
                    md += f"### {v.get('message', 'No message')}\n\n"
                    md += f"- **Category:** {v.get('category', 'unknown')}\n"
                    if v.get('component'):
                        md += f"- **Component:** {v['component']}\n"
                    if v.get('net'):
                        md += f"- **Net:** {v['net']}\n"
                    if v.get('suggested_fix'):
                        md += f"- **Suggested Fix:** {v['suggested_fix']}\n"
                    md += "\n"
        
        return md
    
    def export_report(self, filepath: str, format: ReportFormat = ReportFormat.JSON) -> bool:
        """
        Export report to file.
        
        Args:
            filepath: Output file path
            format: Export format
            
        Returns:
            Success status
        """
        try:
            if not self.report_data:
                logger.error("No report data to export")
                return False
            
            with open(filepath, 'w') as f:
                if format == ReportFormat.JSON:
                    json.dump(self.report_data, f, indent=2)
                elif format in [ReportFormat.HTML, ReportFormat.TEXT, ReportFormat.MARKDOWN]:
                    f.write(self.report_data.get("formatted_output", ""))
                else:
                    logger.error(f"Unsupported format: {format}")
                    return False
            
            logger.info(f"Report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export report: {str(e)}")
            return False
    
    def get_violation_statistics(self) -> Dict[str, Any]:
        """Get detailed violation statistics."""
        if not self.report_data:
            return {}
        
        violations = self.report_data.get("violations", {}).get("all", [])
        
        # Calculate statistics
        by_source = {}
        by_category = {}
        by_severity = {}
        
        for v in violations:
            source = v.get("source", "unknown")
            category = v.get("category", "unknown")
            severity = v.get("severity", "unknown")
            
            by_source[source] = by_source.get(source, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total": len(violations),
            "by_source": by_source,
            "by_category": by_category,
            "by_severity": by_severity
        }