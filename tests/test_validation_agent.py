"""
Test the validation agent's functionality.
"""

import asyncio
import json

from agents.validation_agent import ValidationAgent
from utils.validation_utils import (
    get_validation_coverage, verify_rule_application, analyze_field_validation
)


async def test_validation_agent():
    """Test the validation agent with sample documents."""
    agent = ValidationAgent()
    print("Validation Agent initialized")
    
    # Sample invoice document
    invoice = {
        "document_id": "INV-12345",
        "document_type": "invoice",
        "invoice_number": "INV-12345",
        "invoice_date": "2023-05-15",
        "total_amount": 1250.50,
        "supplier": {
            "name": "Acme Corp",
            "tax_id": "123456789"
        },
        "items": [
            {
                "description": "Widget A",
                "quantity": 5,
                "unit_price": 100.0,
                "total": 500.0
            },
            {
                "description": "Widget B",
                "quantity": 3,
                "unit_price": 250.0,
                "total": 750.0
            }
        ]
    }
    
    # Sample receipt with missing fields
    receipt = {
        "document_id": "RCT-98765",
        "document_type": "receipt",
        "receipt_date": "2023-05-16",
        "total_amount": "invalid",  # Invalid type
        "merchant": "Local Store",
        "items": [
            {
                "description": "Item 1",
                "price": 45.99
            },
            {
                "description": "Item 2",
                "price": 12.50
            }
        ]
    }
    
    print("\n=== Testing invoice validation ===")
    invoice_result = await agent.validate_document(invoice)
    print(f"Invoice validation passed: {invoice_result.passed}")
    print(f"Compliance score: {invoice_result.compliance_score:.1f}%")
    print(f"Total rules applied: {invoice_result.total_rules_applied}")
    print(f"Rules passed: {invoice_result.passed_rules}")
    print(f"Rules failed: {invoice_result.failed_rules}")
    
    print("\n=== Testing receipt validation ===")
    receipt_result = await agent.validate_document(receipt)
    print(f"Receipt validation passed: {receipt_result.passed}")
    print(f"Compliance score: {receipt_result.compliance_score:.1f}%")
    print(f"Total rules applied: {receipt_result.total_rules_applied}")
    print(f"Rules passed: {receipt_result.passed_rules}")
    print(f"Rules failed: {receipt_result.failed_rules}")
    
    print("\n=== Testing validation coverage ===")
    coverage = get_validation_coverage(receipt_result)
    print(f"Fields validated: {coverage['fields_validated']}")
    print(f"Success rate: {coverage['success_rate']:.1f}%")
    print(f"Rules by severity: {json.dumps(coverage['rules_by_severity'], indent=2)}")
    
    # Test rule application verification
    print("\n=== Testing rule application verification ===")
    # Get all rule IDs from the agent
    all_rules = agent.rule_store.get_all_rules()
    expected_rule_ids = [rule.rule_id for rule in all_rules]
    
    verification = verify_rule_application(receipt_result, expected_rule_ids)
    print(f"Verification passed: {verification['verification_passed']}")
    print(f"Expected rules: {verification['expected_rules']}")
    print(f"Applied rules: {verification['applied_rules']}")
    if verification['missing_rules']:
        print(f"Missing rules: {len(verification['missing_rules'])}")
    if verification['unexpected_rules']:
        print(f"Unexpected rules: {len(verification['unexpected_rules'])}")
    
    # Test field validation analysis
    print("\n=== Testing field validation analysis ===")
    field_analysis = analyze_field_validation(receipt_result, "total_amount")
    print(f"Field: {field_analysis['field_name']}")
    print(f"Total validations: {field_analysis['total_validations']}")
    print(f"Passed: {field_analysis['passed']}")
    print(f"Error count: {field_analysis['error_count']}")
    
    # Test rule versioning
    print("\n=== Testing rule versioning ===")
    # Get a rule to update
    rule = agent.rule_store.get_rule(all_rules[0].rule_id)
    if rule:
        # Update the rule
        rule_data = rule.model_dump()
        rule_data["message"] = "Updated validation message"
        
        # Update via agent
        new_version = await agent.update_validation_rule(
            rule.rule_id, 
            rule_data, 
            "Testing rule versioning"
        )
        print(f"Updated rule to version: {new_version}")
        
        # Get version history
        versions = await agent.get_rule_version_history(rule.rule_id)
        print(f"Version history: {len(versions)} versions")
        for idx, version in enumerate(versions):
            print(f"  Version {idx+1}: {version['version_number']} - {version['created_at']}")
    
    print("\n=== Validation Agent Testing Complete ===")


if __name__ == "__main__":
    asyncio.run(test_validation_agent())
