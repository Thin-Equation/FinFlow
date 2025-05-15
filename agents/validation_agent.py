"""
Validation agent for the FinFlow system.
"""

from typing import Any, Dict, List, TypedDict

from agents.base_agent import BaseAgent

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

class ValidationAgent(BaseAgent):
    """
    Agent responsible for validating documents against business rules and compliance requirements.
    """
    
    def __init__(self):
        """Initialize the validation agent."""
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
        
        # Register validation tools
        self._register_tools()
    
    def _register_tools(self):
        """Register validation tools."""
        # TODO: Implement validation tools
        pass
    
    async def validate_document(
        self, document: Dict[str, Any], rules: List[Dict[str, Any]]
    ) -> ValidationReport:
        """
        Validate a document against the provided rules.
        
        Args:
            document: The document to validate.
            rules: List of rules to apply.
            
        Returns:
            Validation report with results for each rule.
        """
        self.logger.info(f"Validating document: {document.get('document_id', 'unknown')}")
        
        # TODO: Implement validation logic
        # This is a stub implementation
        validation_results: List[ValidationResult] = []
        for rule in rules:
            result: ValidationResult = {
                "rule_id": rule["rule_id"],
                "passed": True,  # Placeholder
                "message": "Validation passed",
                "severity": rule["severity"]
            }
            validation_results.append(result)
        
        # Calculate passed and failed rules
        passed_rules = sum(1 for result in validation_results if result["passed"])
        failed_rules = len(validation_results) - passed_rules
        
        validation_report: ValidationReport = {
            "document_id": document.get("document_id", "unknown"),
            "passed": all(result["passed"] for result in validation_results),
            "validation_results": validation_results,
            "total_rules": len(rules),
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
        }
        
        self.logger.info(f"Validation completed with {validation_report['passed_rules']} passed rules and {validation_report['failed_rules']} failed rules")
        return validation_report
