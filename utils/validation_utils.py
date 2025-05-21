"""
Utility functions for validation rule application and verification.
This module provides tools to analyze validation results, verify rule application,
and generate reports on validation coverage.
"""


from datetime import datetime
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

from models.validation import (
    DocumentValidationResult, ComplianceRuleSet,
)


def get_validation_coverage(validation_result: DocumentValidationResult) -> Dict[str, Any]:
    """
    Calculate the validation coverage metrics from a validation result.
    
    Args:
        validation_result: The validation result to analyze.
        
    Returns:
        A dictionary with coverage metrics.
    """
    total_fields = len(validation_result.field_results)
    total_rules_applied = validation_result.total_rules_applied
    rules_by_severity = {
        "critical": len(validation_result.critical_issues),
        "error": len(validation_result.errors),
        "warning": len(validation_result.warnings),
        "info": len(validation_result.info)
    }
    
    # Calculate percentage of passed rules
    success_rate = 0
    if total_rules_applied > 0:
        success_rate = (validation_result.passed_rules / total_rules_applied) * 100
    
    return {
        "document_id": validation_result.document_id,
        "document_type": validation_result.document_type,
        "fields_validated": total_fields,
        "rules_applied": total_rules_applied,
        "rules_passed": validation_result.passed_rules,
        "rules_failed": validation_result.failed_rules,
        "success_rate": success_rate,
        "rules_by_severity": rules_by_severity,
        "compliance_score": validation_result.compliance_score
    }


def verify_rule_application(
    validation_result: DocumentValidationResult,
    expected_rule_ids: List[str]
) -> Dict[str, Any]:
    """
    Verify that expected rules were applied during validation.
    
    Args:
        validation_result: The validation result to analyze.
        expected_rule_ids: List of rule IDs that were expected to be applied.
        
    Returns:
        A dictionary with verification results.
    """
    rules_applied = set(validation_result.rules_applied)
    rules_expected = set(expected_rule_ids)
    
    # Find missing and unexpected rules
    missing_rules = rules_expected - rules_applied
    unexpected_rules = rules_applied - rules_expected
    
    verification_passed = len(missing_rules) == 0
    
    return {
        "verification_passed": verification_passed,
        "expected_rules": len(expected_rule_ids),
        "applied_rules": len(rules_applied),
        "missing_rules": list(missing_rules),
        "unexpected_rules": list(unexpected_rules),
    }


def analyze_field_validation(
    validation_result: DocumentValidationResult,
    field_name: str
) -> Dict[str, Any]:
    """
    Analyze validation results for a specific field.
    
    Args:
        validation_result: The validation result to analyze.
        field_name: The name of the field to analyze.
        
    Returns:
        A dictionary with field validation analysis.
    """
    field_results = validation_result.field_results.get(field_name, [])
    
    # Collect all errors for this field
    errors = []
    for result in field_results:
        errors.extend(result.errors)
    
    # Count results by validation type
    validation_types = {}
    for result in field_results:
        val_type = result.validation_type
        if val_type not in validation_types:
            validation_types[val_type] = {"total": 0, "passed": 0, "failed": 0}
            
        validation_types[val_type]["total"] += 1
        if result.passed:
            validation_types[val_type]["passed"] += 1
        else:
            validation_types[val_type]["failed"] += 1
    
    return {
        "field_name": field_name,
        "total_validations": len(field_results),
        "passed": all(r.passed for r in field_results),
        "error_count": len(errors),
        "validation_types": validation_types,
        "errors": [e.model_dump() for e in errors]
    }


def generate_validation_summary_chart(validation_result: DocumentValidationResult) -> str:
    """
    Generate a base64-encoded chart image showing validation results.
    
    Args:
        validation_result: The validation result to visualize.
        
    Returns:
        Base64-encoded PNG image data.
    """
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Passed vs Failed rules
    labels = ['Passed', 'Failed']
    sizes = [validation_result.passed_rules, validation_result.failed_rules]
    colors = ['#4CAF50', '#F44336']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Validation Results')
    
    # Plot 2: Issues by severity
    severities = ['Critical', 'Error', 'Warning', 'Info']
    counts = [
        len(validation_result.critical_issues),
        len(validation_result.errors),
        len(validation_result.warnings),
        len(validation_result.info)
    ]
    colors = ['#9C27B0', '#F44336', '#FF9800', '#2196F3']
    
    ax2.bar(severities, counts, color=colors)
    ax2.set_title('Issues by Severity')
    ax2.set_ylabel('Number of Issues')
    
    # Add compliance score as text
    fig.text(0.5, 0.01, f'Compliance Score: {validation_result.compliance_score:.1f}%', 
             ha='center', fontsize=12, bbox=dict(facecolor='#E0E0E0', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic


def export_validation_results_to_csv(validation_result: DocumentValidationResult) -> str:
    """
    Export validation results to CSV format.
    
    Args:
        validation_result: The validation result to export.
        
    Returns:
        CSV data as a string.
    """
    # Create a list of records for the results
    records = []
    
    # Add overall document validation record
    records.append({
        'document_id': validation_result.document_id,
        'field_name': 'DOCUMENT_OVERALL',
        'validation_type': 'SUMMARY',
        'passed': validation_result.passed,
        'severity': 'N/A',
        'message': f"Overall compliance score: {validation_result.compliance_score:.1f}%",
        'rule_id': 'N/A',
        'rule_version': 'N/A'
    })
    
    # Add individual field validation records
    for field_name, field_results in validation_result.field_results.items():
        for result in field_results:
            # Get rule version
            rule_version = validation_result.rule_versions_applied.get(result.rule_id, 'unknown')
            
            if result.passed:
                # Add a passing record
                records.append({
                    'document_id': validation_result.document_id,
                    'field_name': field_name,
                    'validation_type': result.validation_type,
                    'passed': True,
                    'severity': 'N/A',
                    'message': 'Validation passed',
                    'rule_id': result.rule_id,
                    'rule_version': rule_version
                })
            else:
                # Add records for each error
                for error in result.errors:
                    records.append({
                        'document_id': validation_result.document_id,
                        'field_name': field_name,
                        'validation_type': result.validation_type,
                        'passed': False,
                        'severity': error.severity,
                        'message': error.message,
                        'rule_id': result.rule_id,
                        'rule_version': rule_version
                    })
    
    # Convert to a DataFrame and then to CSV
    df = pd.DataFrame(records)
    csv_data = df.to_csv(index=False)
    return csv_data


def compare_validation_results(
    result1: DocumentValidationResult,
    result2: DocumentValidationResult
) -> Dict[str, Any]:
    """
    Compare two validation results to identify differences.
    
    Args:
        result1: First validation result.
        result2: Second validation result.
        
    Returns:
        A dictionary with comparison results.
    """
    # Compare basic metrics
    metrics_diff = {
        "passed_diff": result2.passed != result1.passed,
        "compliance_score_diff": result2.compliance_score - result1.compliance_score,
        "total_rules_diff": result2.total_rules_applied - result1.total_rules_applied,
        "passed_rules_diff": result2.passed_rules - result1.passed_rules,
        "failed_rules_diff": result2.failed_rules - result1.failed_rules,
        "critical_issues_diff": len(result2.critical_issues) - len(result1.critical_issues),
        "errors_diff": len(result2.errors) - len(result1.errors),
        "warnings_diff": len(result2.warnings) - len(result1.warnings),
        "info_diff": len(result2.info) - len(result1.info)
    }
    
    # Compare rules applied
    rules1 = set(result1.rules_applied)
    rules2 = set(result2.rules_applied)
    
    rules_diff = {
        "new_rules": list(rules2 - rules1),
        "removed_rules": list(rules1 - rules2),
        "common_rules": list(rules1.intersection(rules2))
    }
    
    # Compare field results
    fields1 = set(result1.field_results.keys())
    fields2 = set(result2.field_results.keys())
    
    fields_diff = {
        "new_fields": list(fields2 - fields1),
        "removed_fields": list(fields1 - fields2),
        "common_fields": list(fields1.intersection(fields2))
    }
    
    # Detailed comparison of common fields
    field_details = {}
    for field in fields_diff["common_fields"]:
        results1 = result1.field_results.get(field, [])
        results2 = result2.field_results.get(field, [])
        
        # Check if validation results changed for this field
        changed = False
        if len(results1) != len(results2):
            changed = True
        else:
            # Check if any result changed its pass status
            passed1 = [r.passed for r in results1]
            passed2 = [r.passed for r in results2]
            if passed1 != passed2:
                changed = True
        
        if changed:
            field_details[field] = {
                "changed": True,
                "result1_count": len(results1),
                "result2_count": len(results2),
                "result1_passed": all(r.passed for r in results1),
                "result2_passed": all(r.passed for r in results2)
            }
    
    return {
        "metrics_diff": metrics_diff,
        "rules_diff": rules_diff,
        "fields_diff": fields_diff,
        "field_details": field_details
    }


def create_rule_compliance_report(
    validation_results: List[DocumentValidationResult],
    rule_sets: List[ComplianceRuleSet]
) -> Dict[str, Any]:
    """
    Create a compliance report across multiple documents and rule sets.
    
    Args:
        validation_results: List of validation results to analyze.
        rule_sets: List of rule sets to track compliance for.
        
    Returns:
        A dictionary with compliance metrics.
    """
    if not validation_results:
        return {"error": "No validation results provided"}
    
    # Track overall compliance 
    total_documents = len(validation_results)
    compliant_documents = sum(1 for r in validation_results if r.passed)
    
    # Track compliance by rule
    rule_compliance = {}
    rule_application_count = {}
    
    # Track compliance by rule set
    ruleset_compliance = {rs.ruleset_id: {
        "name": rs.name,
        "description": rs.description,
        "category": rs.category,
        "total_documents": 0,
        "compliant_documents": 0,
        "compliance_rate": 0.0
    } for rs in rule_sets}
    
    # Process each validation result
    for result in validation_results:
        # Track rule-level compliance
        for rule_id in result.rules_applied:
            if rule_id not in rule_compliance:
                rule_compliance[rule_id] = {"passed": 0, "failed": 0}
                
            if rule_id not in rule_application_count:
                rule_application_count[rule_id] = 0
                
            rule_application_count[rule_id] += 1
            
            # Check if this rule passed for this document
            passed = True
            for field_results in result.field_results.values():
                for field_result in field_results:
                    if field_result.rule_id == rule_id and not field_result.passed:
                        passed = False
                        break
                if not passed:
                    break
                    
            if passed:
                rule_compliance[rule_id]["passed"] += 1
            else:
                rule_compliance[rule_id]["failed"] += 1
        
        # Track ruleset-level compliance
        for rs in rule_sets:
            # Check if any of the ruleset's rules were applied
            if any(rule_id in result.rules_applied for rule_id in rs.rules):
                ruleset_compliance[rs.ruleset_id]["total_documents"] += 1
                
                # Check if all applied rules from this ruleset passed
                ruleset_rules_applied = [r for r in result.rules_applied if r in rs.rules]
                ruleset_passed = True
                
                for rule_id in ruleset_rules_applied:
                    # Check if this rule passed for this document
                    passed = True
                    for field_results in result.field_results.values():
                        for field_result in field_results:
                            if field_result.rule_id == rule_id and not field_result.passed:
                                passed = False
                                break
                        if not passed:
                            break
                    
                    if not passed:
                        ruleset_passed = False
                        break
                
                if ruleset_passed:
                    ruleset_compliance[rs.ruleset_id]["compliant_documents"] += 1
    
    # Calculate compliance rates for each rule
    rule_compliance_rates = {}
    for rule_id, data in rule_compliance.items():
        total = data["passed"] + data["failed"]
        if total > 0:
            compliance_rate = (data["passed"] / total) * 100
        else:
            compliance_rate = 0
            
        rule_compliance_rates[rule_id] = {
            "compliance_rate": compliance_rate,
            "total_applications": total,
            "passed": data["passed"],
            "failed": data["failed"]
        }
    
    # Calculate compliance rates for each ruleset
    for rs_id, data in ruleset_compliance.items():
        if data["total_documents"] > 0:
            data["compliance_rate"] = (data["compliant_documents"] / data["total_documents"]) * 100
    
    # Calculate overall compliance rate
    overall_compliance_rate = 0
    if total_documents > 0:
        overall_compliance_rate = (compliant_documents / total_documents) * 100
    
    return {
        "report_date": datetime.now().isoformat(),
        "total_documents": total_documents,
        "compliant_documents": compliant_documents,
        "overall_compliance_rate": overall_compliance_rate,
        "rule_compliance": rule_compliance_rates,
        "ruleset_compliance": ruleset_compliance
    }
