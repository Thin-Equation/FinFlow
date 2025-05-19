"""
Validation models for the FinFlow system.
This module provides the data structures and models for field validation, rule management,
and versioning of compliance rules.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import re
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator

from models.compliance import RuleSeverity, RuleCategory


class FieldValidationType(str, Enum):
    """Types of field validations that can be applied."""
    REQUIRED = "required"  # Field must be present and non-empty
    TYPE = "type"  # Field must be of specific type (str, int, float, bool, etc.)
    REGEX = "regex"  # Field must match regex pattern
    RANGE = "range"  # Field must be within numeric range
    LENGTH = "length"  # Field must have specific length constraints
    ENUMERATION = "enumeration"  # Field value must be in a set of allowed values
    COMPARISON = "comparison"  # Field value compared to another field or constant
    DEPENDENCY = "dependency"  # Field required if another field has specific value
    CUSTOM = "custom"  # Custom python expression to evaluate
    COMPOSITE = "composite"  # Composite of multiple validation types


class FieldValidationRule(BaseModel):
    """
    Defines a validation rule for a specific field in a document.
    """
    # Identifiers
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0.0"
    
    # Core rule properties
    field_name: str
    validation_type: FieldValidationType
    severity: RuleSeverity = RuleSeverity.ERROR
    message: str
    
    # Validation parameters (used differently based on validation_type)
    expected_type: Optional[str] = None
    regex_pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None
    comparison_field: Optional[str] = None
    comparison_operator: Optional[str] = None  # "eq", "ne", "gt", "lt", "ge", "le"
    comparison_value: Optional[Any] = None
    dependency_field: Optional[str] = None
    dependency_value: Optional[Any] = None
    custom_expression: Optional[str] = None
    sub_rules: Optional[List["FieldValidationRule"]] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    description: Optional[str] = None
    
    # Prioritization
    priority: int = 100  # Lower numbers execute first
    
    @field_validator('regex_pattern')
    @classmethod
    def validate_regex(cls, v: Optional[str]) -> Optional[str]:
        """Validate that the regex pattern is valid."""
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        return v
    
    @model_validator(mode='after')
    def validate_required_fields(self) -> 'FieldValidationRule':
        """Validate that the required fields for each validation type are provided."""
        validation_type = self.validation_type
        
        if validation_type == FieldValidationType.TYPE:
            if not self.expected_type:
                raise ValueError("expected_type is required for validation_type=TYPE")
                
        elif validation_type == FieldValidationType.REGEX:
            if not self.regex_pattern:
                raise ValueError("regex_pattern is required for validation_type=REGEX")
                
        elif validation_type == FieldValidationType.RANGE:
            if self.min_value is None and self.max_value is None:
                raise ValueError("Either min_value or max_value must be provided for validation_type=RANGE")
                
        elif validation_type == FieldValidationType.LENGTH:
            if self.min_length is None and self.max_length is None:
                raise ValueError("Either min_length or max_length must be provided for validation_type=LENGTH")
                
        elif validation_type == FieldValidationType.ENUMERATION:
            if not self.allowed_values:
                raise ValueError("allowed_values is required for validation_type=ENUMERATION")
                
        elif validation_type == FieldValidationType.COMPARISON:
            comp_field = self.comparison_field
            comp_value = self.comparison_value
            comp_op = self.comparison_operator
            
            if not comp_op:
                raise ValueError("comparison_operator is required for validation_type=COMPARISON")
            if comp_field is None and comp_value is None:
                raise ValueError("Either comparison_field or comparison_value must be provided")
                
        elif validation_type == FieldValidationType.DEPENDENCY:
            if self.dependency_field is None:
                raise ValueError("dependency_field is required for validation_type=DEPENDENCY")
                
        elif validation_type == FieldValidationType.CUSTOM:
            if not self.custom_expression:
                raise ValueError("custom_expression is required for validation_type=CUSTOM")
                
        elif validation_type == FieldValidationType.COMPOSITE:
            if not self.sub_rules:
                raise ValueError("sub_rules is required for validation_type=COMPOSITE")
                
        return self


class ComplianceRuleVersion(BaseModel):
    """A specific version of a compliance rule."""
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str
    version_number: str  # Semantic versioning (e.g., "1.0.0")
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    description: str
    change_reason: Optional[str] = None
    rule_data: Dict[str, Any]  # Serialized rule data
    is_active: bool = True


class RuleCondition(BaseModel):
    """Condition for when a compliance rule applies to a document."""
    document_type: Optional[str] = None
    field_conditions: Dict[str, Any] = Field(default_factory=dict)
    
    def matches(self, document: Dict[str, Any]) -> bool:
        """Check if the document matches this condition."""
        # Check document type if specified
        if self.document_type and document.get("document_type") != self.document_type:
            return False
            
        # Check field conditions
        for field_name, expected_value in self.field_conditions.items():
            # Handle nested fields using dot notation (e.g., "metadata.source")
            parts = field_name.split(".")
            value = document
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return False  # Field doesn't exist
                    
            # Compare the field value to expected value
            if value != expected_value:
                return False
                
        return True


class ComplianceRuleSet(BaseModel):
    """
    A set of compliance rules that are applied together, typically representing
    a specific regulation or company policy.
    """
    ruleset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    version: str = "1.0.0"
    category: RuleCategory
    rules: List[str]  # List of rule_ids that belong to this set
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None


class ValidationError(BaseModel):
    """Detailed information about a validation error."""
    field_name: str
    rule_id: str
    message: str
    severity: RuleSeverity
    error_code: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class FieldValidationResult(BaseModel):
    """Result of validating a single field."""
    field_name: str
    field_value: Any
    passed: bool
    rule_id: str
    validation_type: FieldValidationType
    errors: List[ValidationError] = []
    

class DocumentValidationResult(BaseModel):
    """Comprehensive validation result for a document."""
    document_id: str
    document_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Summary
    passed: bool
    total_rules_applied: int
    passed_rules: int
    failed_rules: int
    
    # Results by field
    field_results: Dict[str, List[FieldValidationResult]]
    
    # Results by severity
    critical_issues: List[ValidationError] = []
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    info: List[ValidationError] = []
    
    # Rule application tracking
    rules_applied: List[str] = []
    rules_skipped: List[str] = []
    rule_versions_applied: Dict[str, str] = Field(default_factory=dict)  # rule_id -> version
    
    # Overall compliance score (0-100)
    compliance_score: float
    
    @field_validator('compliance_score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure the compliance score is between 0 and 100."""
        if v < 0 or v > 100:
            raise ValueError("Compliance score must be between 0 and 100")
        return v
