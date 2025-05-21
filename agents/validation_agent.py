"""
Validation agent for the FinFlow system.
Responsible for validating documents against business rules, compliance requirements,
and regulatory standards. It also handles versioning and storage of validation rules.
"""

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from agents.base_agent import BaseAgent
from models.compliance import RuleSeverity, RuleCategory
from models.validation import (
    FieldValidationType, FieldValidationRule, ComplianceRuleVersion, ComplianceRuleSet, ValidationError,
    FieldValidationResult, DocumentValidationResult
)

# Type definitions for legacy compatibility
from typing import TypedDict

class ValidationResult(TypedDict):
    rule_id: str
    passed: bool
    message: str
    severity: str

class ValidationReport(TypedDict):
    document_id: str
    passed: bool
    validation_results: List[ValidationResult]
    total_rules: int
    passed_rules: int
    failed_rules: int

class ValidationEngine:
    """
    Core validation engine that performs field-level validation.
    """
    
    @staticmethod
    def validate_required(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate that a required field is present and non-empty."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        passed = field_value is not None and field_value != ""
        
        errors = []
        if not passed:
            errors.append(ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=rule.message or f"Field '{field_name}' is required",
                severity=rule.severity,
                error_code="REQUIRED_FIELD_MISSING"
            ))
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_type(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate that a field is of the expected type."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        expected_type = rule.expected_type
        
        # Skip validation if the field is not present
        if field_value is None:
            return FieldValidationResult(
                field_name=field_name,
                field_value=None,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
        
        passed = False
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        
        if expected_type in type_mapping:
            passed = isinstance(field_value, type_mapping[expected_type])
        else:
            # For more complex types or custom types, we could extend this logic
            passed = False
            
        errors = []
        if not passed:
            errors.append(ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=rule.message or f"Field '{field_name}' must be of type {expected_type}",
                severity=rule.severity,
                error_code="TYPE_MISMATCH",
                context={"expected_type": expected_type, "actual_type": str(type(field_value))}
            ))
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_regex(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate that a field matches a regex pattern."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        
        # Skip validation if the field is not present or not a string
        if field_value is None or not isinstance(field_value, str):
            return FieldValidationResult(
                field_name=field_name,
                field_value=field_value,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
        
        pattern = rule.regex_pattern
        passed = bool(re.match(pattern, field_value)) if pattern else False
            
        errors = []
        if not passed:
            errors.append(ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=rule.message or f"Field '{field_name}' does not match the required pattern",
                severity=rule.severity,
                error_code="REGEX_MISMATCH",
                context={"pattern": pattern, "value": field_value}
            ))
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_range(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate that a numeric field is within a specified range."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        
        # Skip validation if the field is not present
        if field_value is None:
            return FieldValidationResult(
                field_name=field_name,
                field_value=None,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
            
        # Try to convert to float if not already a number
        if not isinstance(field_value, (int, float)):
            try:
                field_value = float(field_value)
            except (ValueError, TypeError):
                return FieldValidationResult(
                    field_name=field_name,
                    field_value=field_value,
                    passed=False,
                    rule_id=rule.rule_id,
                    validation_type=rule.validation_type,
                    errors=[ValidationError(
                        field_name=field_name,
                        rule_id=rule.rule_id,
                        message=f"Field '{field_name}' must be a number",
                        severity=rule.severity,
                        error_code="TYPE_MISMATCH"
                    )]
                )
        
        min_value = rule.min_value
        max_value = rule.max_value
        passed = True
        
        if min_value is not None and field_value < min_value:
            passed = False
        
        if max_value is not None and field_value > max_value:
            passed = False
            
        errors = []
        if not passed:
            # Construct a descriptive message
            if min_value is not None and max_value is not None:
                message = rule.message or f"Field '{field_name}' must be between {min_value} and {max_value}"
            elif min_value is not None:
                message = rule.message or f"Field '{field_name}' must be at least {min_value}"
            else:
                message = rule.message or f"Field '{field_name}' must be at most {max_value}"
                
            errors.append(ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=message,
                severity=rule.severity,
                error_code="RANGE_VIOLATION",
                context={"min_value": min_value, "max_value": max_value, "value": field_value}
            ))
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_length(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate that a field's length is within specified limits."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        
        # Skip validation if the field is not present
        if field_value is None:
            return FieldValidationResult(
                field_name=field_name,
                field_value=None,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
            
        # Convert to string if not already a string or list
        if not isinstance(field_value, (str, list, dict)):
            field_value = str(field_value)
            
        length = len(field_value)
        min_length = rule.min_length
        max_length = rule.max_length
        passed = True
        
        if min_length is not None and length < min_length:
            passed = False
        
        if max_length is not None and length > max_length:
            passed = False
            
        errors = []
        if not passed:
            # Construct a descriptive message
            if min_length is not None and max_length is not None:
                message = rule.message or f"Field '{field_name}' length must be between {min_length} and {max_length}"
            elif min_length is not None:
                message = rule.message or f"Field '{field_name}' length must be at least {min_length}"
            else:
                message = rule.message or f"Field '{field_name}' length must be at most {max_length}"
                
            errors.append(ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=message,
                severity=rule.severity,
                error_code="LENGTH_VIOLATION",
                context={"min_length": min_length, "max_length": max_length, "length": length}
            ))
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_enumeration(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate that a field value is within a set of allowed values."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        
        # Skip validation if the field is not present
        if field_value is None:
            return FieldValidationResult(
                field_name=field_name,
                field_value=None,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
            
        allowed_values = rule.allowed_values
        passed = allowed_values is not None and field_value in allowed_values
            
        errors = []
        if not passed:
            errors.append(ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=rule.message or f"Field '{field_name}' must be one of: {', '.join(map(str, allowed_values or []))}",
                severity=rule.severity,
                error_code="INVALID_VALUE",
                context={"allowed_values": allowed_values, "value": field_value}
            ))
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_comparison(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate a field by comparing it to another field or a constant value."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        
        # Skip validation if the field is not present
        if field_value is None:
            return FieldValidationResult(
                field_name=field_name,
                field_value=None,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
            
        comparison_field = rule.comparison_field
        comparison_value = rule.comparison_value
        comparison_operator = rule.comparison_operator
        
        # Get the value to compare against
        compare_to = None
        if comparison_field is not None:
            compare_to = ValidationEngine._get_field_value(document, comparison_field)
        else:
            compare_to = comparison_value
            
        # Skip if comparison value is not available
        if compare_to is None:
            return FieldValidationResult(
                field_name=field_name,
                field_value=field_value,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
            
        # Perform the comparison
        passed = False
        if comparison_operator == "eq":
            passed = field_value == compare_to
        elif comparison_operator == "ne":
            passed = field_value != compare_to
        elif comparison_operator == "gt":
            passed = field_value > compare_to
        elif comparison_operator == "lt":
            passed = field_value < compare_to
        elif comparison_operator == "ge":
            passed = field_value >= compare_to
        elif comparison_operator == "le":
            passed = field_value <= compare_to
            
        errors = []
        if not passed:
            compare_desc = f"field '{comparison_field}'" if comparison_field else f"value {comparison_value}"
            op_text = {
                "eq": "equal to",
                "ne": "not equal to",
                "gt": "greater than",
                "lt": "less than",
                "ge": "greater than or equal to",
                "le": "less than or equal to"
            }.get(comparison_operator, comparison_operator)
            
            errors.append(ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=rule.message or f"Field '{field_name}' must be {op_text} {compare_desc}",
                severity=rule.severity,
                error_code="COMPARISON_FAILED",
                context={
                    "field_value": field_value, 
                    "comparison_value": compare_to,
                    "operator": comparison_operator
                }
            ))
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_dependency(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate that a field is present when another field has a specific value."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        
        dependency_field = rule.dependency_field
        dependency_value = rule.dependency_value
        
        if dependency_field is None:
            # Can't validate without a dependency field
            return FieldValidationResult(
                field_name=field_name,
                field_value=field_value,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
            
        dependency_field_value = ValidationEngine._get_field_value(document, dependency_field)
        
        # Check if the dependency condition is met
        condition_met = dependency_field_value == dependency_value
        
        # Only validate if the condition is met
        passed = True
        if condition_met and field_value is None:
            passed = False
            
        errors = []
        if not passed:
            errors.append(ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=rule.message or f"Field '{field_name}' is required when '{dependency_field}' is '{dependency_value}'",
                severity=rule.severity,
                error_code="DEPENDENCY_VIOLATION",
                context={
                    "dependency_field": dependency_field,
                    "dependency_value": dependency_value
                }
            ))
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_custom(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate using a custom Python expression."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        
        custom_expression = rule.custom_expression
        if not custom_expression:
            return FieldValidationResult(
                field_name=field_name,
                field_value=field_value,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
            
        # Create a safe execution context with limited variables
        context = {
            "value": field_value,
            "document": document,
            "re": re
        }
        
        try:
            # Evaluate the custom expression
            result = eval(custom_expression, {"__builtins__": {}}, context)
            passed = bool(result)
        except Exception as e:
            # If the expression fails, treat it as a validation failure
            passed = False
            errors = [ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=f"Custom validation error: {str(e)}",
                severity=rule.severity,
                error_code="CUSTOM_VALIDATION_ERROR",
                context={"expression": custom_expression, "error": str(e)}
            )]
            return FieldValidationResult(
                field_name=field_name,
                field_value=field_value,
                passed=passed,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                errors=errors
            )
            
        errors = []
        if not passed:
            errors.append(ValidationError(
                field_name=field_name,
                rule_id=rule.rule_id,
                message=rule.message or f"Field '{field_name}' failed custom validation",
                severity=rule.severity,
                error_code="CUSTOM_VALIDATION_FAILED",
                context={"expression": custom_expression}
            ))
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_composite(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate using multiple sub-rules."""
        field_value = ValidationEngine._get_field_value(document, field_name)
        
        sub_rules = rule.sub_rules or []
        if not sub_rules:
            return FieldValidationResult(
                field_name=field_name,
                field_value=field_value,
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
            
        # Validate against all sub-rules
        all_results = []
        for sub_rule in sub_rules:
            result = ValidationEngine.validate_field(field_name, document, sub_rule)
            all_results.append(result)
            
        # A composite rule passes if all sub-rules pass
        passed = all(result.passed for result in all_results)
        
        # Collect all errors from sub-rules
        errors = []
        for result in all_results:
            errors.extend(result.errors)
            
        return FieldValidationResult(
            field_name=field_name,
            field_value=field_value,
            passed=passed,
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            errors=errors
        )
    
    @staticmethod
    def validate_field(field_name: str, document: Dict[str, Any], rule: FieldValidationRule) -> FieldValidationResult:
        """Validate a field using the appropriate validation method based on the rule type."""
        validation_methods = {
            FieldValidationType.REQUIRED: ValidationEngine.validate_required,
            FieldValidationType.TYPE: ValidationEngine.validate_type,
            FieldValidationType.REGEX: ValidationEngine.validate_regex,
            FieldValidationType.RANGE: ValidationEngine.validate_range,
            FieldValidationType.LENGTH: ValidationEngine.validate_length,
            FieldValidationType.ENUMERATION: ValidationEngine.validate_enumeration,
            FieldValidationType.COMPARISON: ValidationEngine.validate_comparison,
            FieldValidationType.DEPENDENCY: ValidationEngine.validate_dependency,
            FieldValidationType.CUSTOM: ValidationEngine.validate_custom,
            FieldValidationType.COMPOSITE: ValidationEngine.validate_composite
        }
        
        validation_method = validation_methods.get(rule.validation_type)
        if validation_method:
            return validation_method(field_name, document, rule)
        else:
            # If no validation method is found, return a default "passed" result
            return FieldValidationResult(
                field_name=field_name,
                field_value=ValidationEngine._get_field_value(document, field_name),
                passed=True,
                rule_id=rule.rule_id,
                validation_type=rule.validation_type
            )
    
    @staticmethod
    def _get_field_value(document: Dict[str, Any], field_name: str) -> Any:
        """Get a field value from the document, supporting dot notation for nested fields."""
        parts = field_name.split(".")
        value = document
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
                
        return value


class ComplianceRuleStore:
    """
    Storage and retrieval system for compliance rules with versioning support.
    In a production environment, this would be backed by a database.
    """
    
    def __init__(self):
        """Initialize the rule store with in-memory collections."""
        self.rules: Dict[str, FieldValidationRule] = {}
        self.rule_versions: Dict[str, List[ComplianceRuleVersion]] = {}
        self.rule_sets: Dict[str, ComplianceRuleSet] = {}
        self.active_rule_versions: Dict[str, str] = {}  # rule_id -> version_number
    
    def add_rule(self, rule: FieldValidationRule) -> str:
        """Add a new validation rule to the store."""
        rule_id = rule.rule_id
        self.rules[rule_id] = rule
        
        # Create initial version
        version = ComplianceRuleVersion(
            rule_id=rule_id,
            version_number=rule.version,
            description=rule.description or f"Rule for {rule.field_name}",
            rule_data=rule.model_dump()
        )
        
        # Add to versions collection
        if rule_id not in self.rule_versions:
            self.rule_versions[rule_id] = []
        self.rule_versions[rule_id].append(version)
        
        # Set as active version
        self.active_rule_versions[rule_id] = rule.version
        
        return rule_id
    
    def update_rule(self, rule: FieldValidationRule, change_reason: str = None) -> str:
        """Update an existing rule, creating a new version."""
        rule_id = rule.rule_id
        
        # Get the current rule
        current_rule = self.rules.get(rule_id)
        if not current_rule:
            raise ValueError(f"Rule with ID {rule_id} not found")
            
        # Create a new version number (simple increment for this example)
        current_version = current_rule.version
        version_parts = current_version.split('.')
        new_version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"
        
        # Update the rule version
        rule.version = new_version
        rule.updated_at = datetime.now()
        
        # Store the updated rule
        self.rules[rule_id] = rule
        
        # Create a version record
        version = ComplianceRuleVersion(
            rule_id=rule_id,
            version_number=new_version,
            description=rule.description or f"Rule for {rule.field_name}",
            change_reason=change_reason or "Rule updated",
            rule_data=rule.model_dump()
        )
        
        # Add to versions collection
        self.rule_versions[rule_id].append(version)
        
        # Set as active version
        self.active_rule_versions[rule_id] = new_version
        
        return new_version
    
    def get_rule(self, rule_id: str, version: str = None) -> Optional[FieldValidationRule]:
        """Get a rule by ID, optionally specifying a version."""
        if rule_id not in self.rules:
            return None
            
        if version is None:
            # Get active version
            return self.rules[rule_id]
            
        # Find the specific version
        for v in self.rule_versions.get(rule_id, []):
            if v.version_number == version:
                # Reconstruct rule from version data
                return FieldValidationRule(**v.rule_data)
                
        return None
    
    def get_rule_version_history(self, rule_id: str) -> List[ComplianceRuleVersion]:
        """Get the version history for a rule."""
        return self.rule_versions.get(rule_id, [])
    
    def activate_rule_version(self, rule_id: str, version: str) -> bool:
        """Set a specific version as the active version for a rule."""
        if rule_id not in self.rules:
            return False
            
        # Find the version
        version_found = False
        for v in self.rule_versions.get(rule_id, []):
            if v.version_number == version:
                version_found = True
                self.rules[rule_id] = FieldValidationRule(**v.rule_data)
                self.active_rule_versions[rule_id] = version
                break
                
        return version_found
    
    def add_rule_set(self, rule_set: ComplianceRuleSet) -> str:
        """Add a new rule set to the store."""
        ruleset_id = rule_set.ruleset_id
        self.rule_sets[ruleset_id] = rule_set
        return ruleset_id
    
    def get_rule_set(self, ruleset_id: str) -> Optional[ComplianceRuleSet]:
        """Get a rule set by ID."""
        return self.rule_sets.get(ruleset_id)
    
    def get_rules_by_ids(self, rule_ids: List[str], use_active_versions: bool = True) -> List[FieldValidationRule]:
        """Get multiple rules by their IDs."""
        rules = []
        for rule_id in rule_ids:
            if use_active_versions:
                rule = self.get_rule(rule_id)
            else:
                rule = self.get_rule(rule_id, version=self.active_rule_versions.get(rule_id))
                
            if rule:
                rules.append(rule)
                
        return rules
    
    def get_all_rules(self) -> List[FieldValidationRule]:
        """Get all rules in the store."""
        return list(self.rules.values())
    
    def get_all_rule_sets(self) -> List[ComplianceRuleSet]:
        """Get all rule sets in the store."""
        return list(self.rule_sets.values())


class ValidationAgent(BaseAgent):
    """
    Agent responsible for validating documents against business rules and compliance requirements.
    Provides a comprehensive validation framework with support for rule storage, versioning,
    and detailed reporting of validation failures.
    """
    
    # Pre-define fields for Pydantic compliance
    rule_store: 'ComplianceRuleStore' = None
    validation_engine: 'ValidationEngine' = None
    
    def add_tool(self, tool):
        """
        Add a tool to the agent.
        
        Args:
            tool: The tool to add to the agent.
        """
        # Store the tool in the agent's tools dictionary
        if not hasattr(self, "_tools"):
            self.__dict__["_tools"] = {}
        self._tools[tool.name] = tool
    
    def __init__(self):
        """Initialize the validation agent."""
        # Initialize the BaseAgent
        super().__init__(
            name="FinFlow_Validation",
            model="gemini-2.0-pro",
            description="Validates documents against business rules and compliance requirements",
            instruction="""
            You are a validation agent for financial documents.
            Your job is to:
            
            1. Apply compliance rules to processed documents
            2. Validate document structure and required fields
            3. Perform mathematical validation (totals, taxes, etc.)
            4. Check for regulatory compliance based on document type
            5. Generate validation reports with issue details
            
            You should identify issues and categorize them by severity.
            """
        )
        
        # Initialize logger explicitly after parent init
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger("finflow.agents.FinFlow_Validation")
        
        # Initialize the rule store directly in __dict__ to bypass Pydantic validation
        self.__dict__["rule_store"] = ComplianceRuleStore()
        
        # Initialize the validation engine
        self.__dict__["validation_engine"] = ValidationEngine()
        
        # Register validation tools
        self._register_tools()
        
        # Load default rules (in production, these would come from a database)
        self._initialize_default_rules()
    
    def _register_tools(self):
        """Register validation tools."""
        # Create and add tools using direct access to the LlmAgent.add_tool method
        from google.adk.tools import BaseTool
        
        # Define a wrapper class for the validation tools
        class ValidationTool(BaseTool):
            def __init__(self, name, description, func):
                super().__init__(name=name, description=description)
                self.func = func
            
            async def __call__(self, context):
                return await self.func(context)
        
        # Helper function to create and add tools
        def add_validation_tool(name, description, func):
            tool = ValidationTool(name, description, func)
            self.add_tool(tool)  # type: ignore
        
        # Add all required tools
        add_validation_tool(
            name="validate_document",
            description="Validate a document against compliance rules",
            func=self.validate_document,
        )
        
        add_validation_tool(
            name="add_validation_rule",
            description="Add a new field validation rule",
            func=self.add_validation_rule,
        )
        
        add_validation_tool(
            name="update_validation_rule",
            description="Update an existing validation rule",
            func=self.update_validation_rule,
        )
        
        add_validation_tool(
            name="get_rule_version_history",
            description="Get the version history for a rule",
            func=self.get_rule_version_history,
        )
        
        add_validation_tool(
            name="activate_rule_version",
            description="Activate a specific version of a rule",
            func=self.activate_rule_version,
        )
        
        add_validation_tool(
            name="create_rule_set",
            description="Create a new rule set",
            func=self.create_rule_set,
        )
    
    def _initialize_default_rules(self):
        """Initialize default validation rules."""
        # Example rules for invoices
        invoice_rules = [
            FieldValidationRule(
                field_name="invoice_number",
                validation_type=FieldValidationType.REQUIRED,
                severity=RuleSeverity.ERROR,
                message="Invoice number is required",
                description="Validates that the invoice has a number"
            ),
            FieldValidationRule(
                field_name="invoice_date",
                validation_type=FieldValidationType.REQUIRED,
                severity=RuleSeverity.ERROR,
                message="Invoice date is required",
                description="Validates that the invoice has a date"
            ),
            FieldValidationRule(
                field_name="total_amount",
                validation_type=FieldValidationType.COMPOSITE,
                severity=RuleSeverity.ERROR,
                message="Total amount must be a positive number",
                description="Validates that the total amount is valid",
                sub_rules=[
                    FieldValidationRule(
                        field_name="total_amount",
                        validation_type=FieldValidationType.REQUIRED,
                        severity=RuleSeverity.ERROR,
                        message="Total amount is required"
                    ),
                    FieldValidationRule(
                        field_name="total_amount",
                        validation_type=FieldValidationType.TYPE,
                        severity=RuleSeverity.ERROR,
                        message="Total amount must be a number",
                        expected_type="float"
                    ),
                    FieldValidationRule(
                        field_name="total_amount",
                        validation_type=FieldValidationType.RANGE,
                        severity=RuleSeverity.ERROR,
                        message="Total amount must be positive",
                        min_value=0.0
                    )
                ]
            ),
            FieldValidationRule(
                field_name="supplier.name",
                validation_type=FieldValidationType.REQUIRED,
                severity=RuleSeverity.ERROR,
                message="Supplier name is required",
                description="Validates that the invoice has a supplier name"
            ),
        ]
        
        # Example rules for receipts
        receipt_rules = [
            FieldValidationRule(
                field_name="receipt_number",
                validation_type=FieldValidationType.REQUIRED,
                severity=RuleSeverity.WARNING,
                message="Receipt number is recommended",
                description="Validates that the receipt has a number"
            ),
            FieldValidationRule(
                field_name="receipt_date",
                validation_type=FieldValidationType.REQUIRED,
                severity=RuleSeverity.ERROR,
                message="Receipt date is required",
                description="Validates that the receipt has a date"
            ),
        ]
        
        # Add rules to the store
        invoice_rule_ids = []
        for rule in invoice_rules:
            rule_id = self.rule_store.add_rule(rule)
            invoice_rule_ids.append(rule_id)
            
        receipt_rule_ids = []
        for rule in receipt_rules:
            rule_id = self.rule_store.add_rule(rule)
            receipt_rule_ids.append(rule_id)
            
        # Create rule sets
        invoice_ruleset = ComplianceRuleSet(
            name="Invoice Validation",
            description="Basic validation rules for invoices",
            category=RuleCategory.DATA_VALIDATION,
            rules=invoice_rule_ids
        )
        
        receipt_ruleset = ComplianceRuleSet(
            name="Receipt Validation",
            description="Basic validation rules for receipts",
            category=RuleCategory.DATA_VALIDATION,
            rules=receipt_rule_ids
        )
        
        # Add rule sets to the store
        self.rule_store.add_rule_set(invoice_ruleset)
        self.rule_store.add_rule_set(receipt_ruleset)
            
    async def validate_document(
        self, document: Dict[str, Any], rule_ids: List[str] = None, ruleset_id: str = None
    ) -> Union[ValidationReport, DocumentValidationResult]:
        """
        Validate a document against the selected rules or ruleset.
        
        Args:
            document: The document to validate.
            rule_ids: Optional list of specific rule IDs to apply.
            ruleset_id: Optional ruleset ID to use for validation.
            
        Returns:
            A detailed validation report.
        """
        self.logger.info(f"Validating document: {document.get('document_id', 'unknown')}")
        
        # Determine which rules to apply
        rules_to_apply = []
        
        if ruleset_id:
            # Get rules from ruleset
            ruleset = self.rule_store.get_rule_set(ruleset_id)
            if ruleset:
                rules_to_apply = self.rule_store.get_rules_by_ids(ruleset.rules)
        elif rule_ids:
            # Get specified rules
            rules_to_apply = self.rule_store.get_rules_by_ids(rule_ids)
        else:
            # Try to determine rules based on document type
            doc_type = document.get("document_type", "").lower()
            
            if doc_type:
                # Find ruleset that matches the document type
                for ruleset in self.rule_store.get_all_rule_sets():
                    if doc_type in [t.lower() for t in ruleset.name.split()]:
                        rules_to_apply = self.rule_store.get_rules_by_ids(ruleset.rules)
                        break
            
            # If no rules were found, use all rules as a fallback
            if not rules_to_apply:
                rules_to_apply = self.rule_store.get_all_rules()
                
        # Apply validation for each rule
        field_results: Dict[str, List[FieldValidationResult]] = {}
        critical_issues = []
        errors = []
        warnings = []
        info = []
        
        rules_applied = []
        rules_skipped = []
        rule_versions_applied = {}
        
        for rule in rules_to_apply:
            # Keep track of rule application
            rules_applied.append(rule.rule_id)
            rule_versions_applied[rule.rule_id] = rule.version
            
            # Validate the field
            result = ValidationEngine.validate_field(rule.field_name, document, rule)
            
            # Record the result
            if rule.field_name not in field_results:
                field_results[rule.field_name] = []
            field_results[rule.field_name].append(result)
            
            # Categorize issues by severity
            for error in result.errors:
                if error.severity == RuleSeverity.CRITICAL:
                    critical_issues.append(error)
                elif error.severity == RuleSeverity.ERROR:
                    errors.append(error)
                elif error.severity == RuleSeverity.WARNING:
                    warnings.append(error)
                else:
                    info.append(error)
        
        # Calculate passed_rules and failed_rules
        passed_rules = sum(1 for field, results in field_results.items() 
                           for result in results if result.passed)
        failed_rules = sum(1 for field, results in field_results.items() 
                           for result in results if not result.passed)
        
        # Calculate compliance score
        total_rules = passed_rules + failed_rules
        if total_rules > 0:
            compliance_score = (passed_rules / total_rules) * 100
        else:
            compliance_score = 100.0
            
        # Create the detailed validation result
        validation_result = DocumentValidationResult(
            document_id=document.get("document_id", "unknown"),
            document_type=document.get("document_type", "unknown"),
            passed=len(critical_issues) == 0 and len(errors) == 0,
            total_rules_applied=total_rules,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            field_results=field_results,
            critical_issues=critical_issues,
            errors=errors,
            warnings=warnings,
            info=info,
            rules_applied=rules_applied,
            rules_skipped=rules_skipped,
            rule_versions_applied=rule_versions_applied,
            compliance_score=compliance_score
        )
        
        # Log summary
        self.logger.info(
            f"Validation completed with {validation_result.passed_rules} passed rules "
            f"and {validation_result.failed_rules} failed rules. "
            f"Compliance score: {validation_result.compliance_score:.1f}%"
        )
        
        # For backward compatibility, also return a ValidationReport
        validation_results: List[ValidationResult] = []
        
        # Convert to the old format
        for field, results in field_results.items():
            for result in results:
                validation_results.append({
                    "rule_id": result.rule_id,
                    "passed": result.passed,
                    "message": result.errors[0].message if result.errors else "Validation passed",
                    "severity": result.errors[0].severity if result.errors else "info"
                })
        
        validation_report: ValidationReport = {
            "document_id": document.get("document_id", "unknown"),
            "passed": validation_result.passed,
            "validation_results": validation_results,
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
        }
        
        return validation_result
    
    async def add_validation_rule(self, rule_data: Dict[str, Any]) -> str:
        """
        Add a new validation rule to the rule store.
        
        Args:
            rule_data: Data for the new rule.
            
        Returns:
            The ID of the newly created rule.
        """
        try:
            rule = FieldValidationRule(**rule_data)
            rule_id = self.rule_store.add_rule(rule)
            self.logger.info(f"Added new rule: {rule_id}")
            return rule_id
        except Exception as e:
            self.logger.error(f"Error adding rule: {str(e)}")
            raise
    
    async def update_validation_rule(
        self, rule_id: str, rule_data: Dict[str, Any], change_reason: str = None
    ) -> str:
        """
        Update an existing validation rule.
        
        Args:
            rule_id: ID of the rule to update.
            rule_data: Updated rule data.
            change_reason: Reason for the update.
            
        Returns:
            The new version number of the rule.
        """
        try:
            # Get existing rule
            existing_rule = self.rule_store.get_rule(rule_id)
            if not existing_rule:
                raise ValueError(f"Rule with ID {rule_id} not found")
                
            # Update with new data
            rule_data["rule_id"] = rule_id
            rule = FieldValidationRule(**rule_data)
            
            # Store the updated rule
            new_version = self.rule_store.update_rule(rule, change_reason)
            self.logger.info(f"Updated rule {rule_id} to version {new_version}")
            return new_version
        except Exception as e:
            self.logger.error(f"Error updating rule: {str(e)}")
            raise
    
    async def get_rule_version_history(self, rule_id: str) -> List[Dict[str, Any]]:
        """
        Get the version history for a rule.
        
        Args:
            rule_id: ID of the rule.
            
        Returns:
            List of version information.
        """
        try:
            versions = self.rule_store.get_rule_version_history(rule_id)
            return [v.model_dump() for v in versions]
        except Exception as e:
            self.logger.error(f"Error getting rule version history: {str(e)}")
            raise
    
    async def activate_rule_version(self, rule_id: str, version: str) -> bool:
        """
        Activate a specific version of a rule.
        
        Args:
            rule_id: ID of the rule.
            version: Version number to activate.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            success = self.rule_store.activate_rule_version(rule_id, version)
            if success:
                self.logger.info(f"Activated version {version} of rule {rule_id}")
            else:
                self.logger.warning(f"Failed to activate version {version} of rule {rule_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error activating rule version: {str(e)}")
            raise
    
    async def create_rule_set(self, ruleset_data: Dict[str, Any]) -> str:
        """
        Create a new rule set.
        
        Args:
            ruleset_data: Data for the new rule set.
            
        Returns:
            The ID of the newly created rule set.
        """
        try:
            ruleset = ComplianceRuleSet(**ruleset_data)
            ruleset_id = self.rule_store.add_rule_set(ruleset)
            self.logger.info(f"Added new rule set: {ruleset_id}")
            return ruleset_id
        except Exception as e:
            self.logger.error(f"Error creating rule set: {str(e)}")
            raise
