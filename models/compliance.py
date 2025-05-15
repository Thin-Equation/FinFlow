"""
Data models for compliance rules and validation in the FinFlow system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class RuleSeverity(str, Enum):
    """Severity level for compliance rules."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleCategory(str, Enum):
    """Categories of compliance rules."""
    DATA_VALIDATION = "data_validation"
    REGULATORY = "regulatory"
    COMPANY_POLICY = "company_policy"
    TAX = "tax"
    ACCOUNTING = "accounting"
    FRAUD_DETECTION = "fraud_detection"
    WORKFLOW = "workflow"


class ComplianceRule(BaseModel):
    """Compliance rule definition."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    category: RuleCategory
    severity: RuleSeverity = RuleSeverity.WARNING
    
    # Document types this rule applies to
    applicable_document_types: List[str]
    
    # Condition expression - to be evaluated by rule engine
    condition: str
    
    # Error message template
    error_message: str
    
    # Reference to regulation or policy (if applicable)
    regulatory_reference: Optional[str] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    enabled: bool = True
    
    # Rule execution order
    priority: int = 100  # Lower numbers execute first


class ValidationResult(BaseModel):
    """Result of validating a document against a rule."""
    rule_id: str
    document_id: str
    passed: bool
    message: Optional[str] = None
    severity: RuleSeverity
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None


class ComplianceCheckReport(BaseModel):
    """Report of all compliance checks for a document."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    document_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Summary
    passed: bool
    total_rules: int
    passed_rules: int
    failed_rules: int
    
    # Detailed results by severity
    critical_issues: List[ValidationResult] = []
    errors: List[ValidationResult] = []
    warnings: List[ValidationResult] = []
    info: List[ValidationResult] = []
    
    # Overall compliance score (0-100)
    compliance_score: float = 100.0
