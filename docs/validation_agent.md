# ValidationAgent: Advanced Document Validation System

The ValidationAgent module provides a comprehensive system for validating financial documents against business rules and compliance requirements. It features a robust rule engine, versioning system, and detailed error reporting.

## Features

### 1. Field Validation Rules
- **Type Validation**: Ensure fields contain the correct data types
- **Required Fields**: Validate that mandatory fields are present
- **Regex Patterns**: Match field values against regex patterns
- **Range Validation**: Check that numeric values are within acceptable ranges
- **Length Validation**: Verify text length constraints
- **Enumeration**: Ensure values are from a predefined set
- **Field Comparison**: Compare fields against other fields or constants
- **Field Dependencies**: Conditional validation based on related fields
- **Custom Validation**: Use custom Python expressions for complex validation
- **Composite Rules**: Combine multiple validations into a single rule

### 2. Rule Storage & Versioning
- **Rule Repository**: Centralized storage for validation rules
- **Semantic Versioning**: Track changes to rules over time
- **Version History**: Full audit trail of rule modifications
- **Rule Activation**: Switch between rule versions as needed
- **Change Tracking**: Record reasons for rule changes

### 3. Rule Sets & Compliance Management
- **Rule Grouping**: Organize rules into logical sets
- **Document Type Mapping**: Apply different rules based on document type
- **Conditional Rule Application**: Apply rules based on document properties

### 4. Detailed Error Reporting
- **Field-Level Validation**: Track validation status per field
- **Severity Classification**: Categorize issues by severity (Critical, Error, Warning, Info)
- **Contextual Errors**: Provide detailed context with error messages
- **Compliance Scoring**: Calculate overall compliance score based on validation results
- **Rule Application Tracking**: Record which rules were applied and their outcomes

### 5. Rule Application Verification
- **Coverage Analysis**: Verify that all expected rules were applied
- **Result Comparison**: Compare validation results across documents or versions
- **Visualization**: Generate charts and reports for validation outcomes

## Usage Examples

### Basic Document Validation

```python
from agents.validation_agent import ValidationAgent

# Initialize the agent
agent = ValidationAgent()

# Sample document
invoice = {
    "document_id": "INV-12345",
    "document_type": "invoice",
    "invoice_number": "INV-12345",
    "invoice_date": "2023-05-15",
    "total_amount": 1250.50,
    "supplier": {
        "name": "Acme Corp"
    }
}

# Validate the document
result = await agent.validate_document(invoice)

# Check validation outcome
if result.passed:
    print("Document passed all validations")
else:
    print(f"Document failed validation with {result.failed_rules} rule failures")
    
    # Show error details
    for error in result.errors:
        print(f"- {error.severity}: {error.message}")
```

### Creating a Custom Validation Rule

```python
from models.validation import FieldValidationRule, FieldValidationType, RuleSeverity

# Create a rule for validating invoice numbers
invoice_number_rule = FieldValidationRule(
    field_name="invoice_number",
    validation_type=FieldValidationType.REGEX,
    severity=RuleSeverity.ERROR,
    message="Invoice number must follow format: INV-XXXXX",
    regex_pattern=r"INV-\d{5}",
    description="Validates invoice number formatting"
)

# Add rule to the validation agent
rule_id = await agent.add_validation_rule(invoice_number_rule.model_dump())
```

### Creating Rule Sets

```python
from models.validation import ComplianceRuleSet
from models.compliance import RuleCategory

# Create a rule set for invoice validation
invoice_ruleset = {
    "name": "Basic Invoice Validation",
    "description": "Essential validation rules for all invoices",
    "category": RuleCategory.DATA_VALIDATION,
    "rules": [rule1_id, rule2_id, rule3_id]
}

# Add the rule set
ruleset_id = await agent.create_rule_set(invoice_ruleset)
```

### Rule Versioning

```python
# Update an existing rule
updated_rule_data = existing_rule.model_dump()
updated_rule_data["message"] = "Updated validation message"
updated_rule_data["severity"] = RuleSeverity.ERROR

new_version = await agent.update_validation_rule(
    rule_id=existing_rule.rule_id,
    rule_data=updated_rule_data,
    change_reason="Updated error message for clarity"
)

# Get version history
versions = await agent.get_rule_version_history(rule_id)
```

### Analyzing Validation Results

```python
from utils.validation_utils import (
    get_validation_coverage, verify_rule_application, analyze_field_validation
)

# Get validation coverage metrics
coverage = get_validation_coverage(validation_result)
print(f"Fields validated: {coverage['fields_validated']}")
print(f"Success rate: {coverage['success_rate']:.1f}%")

# Verify rule application
verification = verify_rule_application(validation_result, expected_rule_ids)
if verification['verification_passed']:
    print("All expected rules were applied")
else:
    print(f"Missing rules: {verification['missing_rules']}")

# Analyze specific field validation
field_analysis = analyze_field_validation(validation_result, "total_amount")
if not field_analysis['passed']:
    print(f"Field failed validation with {field_analysis['error_count']} errors")
```

## System Architecture

The validation system consists of the following components:

1. **ValidationEngine**: Core validation logic for different validation types
2. **ComplianceRuleStore**: Storage and retrieval system for rules and versions
3. **ValidationAgent**: Public API for document validation and rule management
4. **Validation Models**: Data structures for rules, results, and reports
5. **Validation Utilities**: Tools for analyzing and reporting on validation results

## Production Considerations

In a production environment, additional components should be considered:

1. **Database Storage**: Replace the in-memory rule store with a persistent database
2. **API Authentication**: Secure rule management endpoints
3. **Rule Approval Workflow**: Add review and approval process for rule changes
4. **Caching**: Implement rule caching for improved performance
5. **Monitoring**: Add metrics collection for validation failures and performance
6. **Notifications**: Alert system for critical validation failures
7. **Rule Testing**: Framework for testing rule changes before deployment
8. **Integration**: Hooks into document workflow systems
