"""
Tools for the FinFlow agent system.
"""

from typing import Any, Dict, Optional, Callable

# Ignore missing stubs for google.adk.tools
from google.adk.tools import BaseTool, ToolContext  # type: ignore


class FinflowTool(BaseTool):
    """Base class for FinFlow tools."""
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable[[Dict[str, Any], Optional[ToolContext]], Dict[str, Any]],
    ):
        """
        Initialize a FinFlow tool.
        
        Args:
            name: Tool name
            description: Tool description
            function: Function to call when the tool is invoked
        """
        # Ignore type error due to missing stubs for google.adk.tools
        super().__init__(name=name, description=description)  # type: ignore
        self._function = function
    
    def execute(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            parameters: Tool parameters
            tool_context: Optional tool context
        
        Returns:
            Tool execution result
        """
        return self._function(parameters, tool_context)


class DocumentProcessingTool(FinflowTool):
    """Tool for processing financial documents."""
    
    def __init__(self, processor_function: Callable[[str, Optional[ToolContext]], Dict[str, Any]]):
        """
        Initialize document processing tool.
        
        Args:
            processor_function: Function to process documents
        """
        super().__init__(
            name="process_document",
            description="Process a financial document and extract structured information",
            function=self._process_document
        )
        self._processor = processor_function
    
    def _process_document(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Process a document.
        
        Args:
            parameters: Parameters with document path
            tool_context: Tool context
            
        Returns:
            Processing results
        """
        document_path = parameters.get("document_path")
        if not document_path:
            raise ValueError("Document path not provided")
            
        return self._processor(document_path, tool_context)


class DocumentIngestionTool(FinflowTool):
    """Tool for document ingestion operations."""
    
    def __init__(
        self, 
        ingestion_function: Callable[[str, Optional[ToolContext]], Dict[str, Any]],
        name: str = "ingest_document",
        description: str = "Upload and validate a document"
    ):
        """
        Initialize document ingestion tool.
        
        Args:
            ingestion_function: Function for document ingestion operations
            name: Tool name
            description: Tool description
        """
        super().__init__(
            name=name,
            description=description,
            function=self._ingest_document
        )
        self._ingestion = ingestion_function
    
    def _ingest_document(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Perform document ingestion operation.
        
        Args:
            parameters: Parameters with document path and other options
            tool_context: Tool context
            
        Returns:
            Ingestion operation results
        """
        document_path = parameters.get("document_path")
        destination_folder = parameters.get("destination_folder")
        
        if not document_path:
            raise ValueError("Document path not provided")
            
        return self._ingestion(document_path, tool_context)


class InvoiceProcessingTool(DocumentProcessingTool):
    """Tool specifically for processing invoice documents."""
    
    def __init__(self, processor_function: Callable[[str, Optional[ToolContext]], Dict[str, Any]]):
        """
        Initialize invoice processing tool.
        
        Args:
            processor_function: Function to process invoice documents
        """
        # Use the same initialization as DocumentProcessingTool but with a different name and description
        super().__init__(processor_function)
        self.name = "process_invoice"
        self.description = "Process an invoice document and extract structured financial information"


class DocumentIngestionTool(FinflowTool):
    """Tool for document ingestion operations."""
    
    def __init__(
        self, 
        ingestion_function: Callable[[str, Optional[ToolContext]], Dict[str, Any]],
        name: str = "ingest_document",
        description: str = "Upload and validate a document"
    ):
        """
        Initialize document ingestion tool.
        
        Args:
            ingestion_function: Function for document ingestion operations
            name: Tool name
            description: Tool description
        """
        super().__init__(
            name=name,
            description=description,
            function=self._ingest_document
        )
        self._ingestion = ingestion_function
    
    def _ingest_document(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Perform document ingestion operation.
        
        Args:
            parameters: Parameters with document path and other options
            tool_context: Tool context
            
        Returns:
            Ingestion operation results
        """
        document_path = parameters.get("document_path")
        destination_folder = parameters.get("destination_folder")
        
        if not document_path:
            raise ValueError("Document path not provided")
        
        if destination_folder:
            return self._ingestion(document_path, destination_folder, tool_context)
            
        return self._ingestion(document_path, tool_context)


class RuleLookupTool(FinflowTool):
    """Tool for retrieving compliance rules."""
    
    def __init__(self, rule_function: Callable[[str, str, Optional[ToolContext]], Dict[str, Any]]):
        """
        Initialize rule lookup tool.
        
        Args:
            rule_function: Function to retrieve rules
        """
        super().__init__(
            name="lookup_rules",
            description="Retrieve compliance rules for a specific document type and jurisdiction",
            function=self._lookup_rules
        )
        self._rule_lookup = rule_function
    
    def _lookup_rules(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Look up compliance rules.
        
        Args:
            parameters: Parameters with document type and jurisdiction
            tool_context: Tool context
            
        Returns:
            Compliance rules
        """
        document_type = parameters.get("document_type")
        jurisdiction = parameters.get("jurisdiction")
        
        if not document_type or not jurisdiction:
            raise ValueError("Document type and jurisdiction must be provided")
            
        return self._rule_lookup(document_type, jurisdiction, tool_context)


class InvoiceProcessingTool(FinflowTool):
    """Tool specifically for processing invoice documents."""
    
    def __init__(self, processor_function: Callable[[str, Optional[ToolContext]], Dict[str, Any]]):
        """
        Initialize invoice processing tool.
        
        Args:
            processor_function: Function to process invoice documents
        """
        super().__init__(
            name="process_invoice",
            description="Process an invoice document and extract structured financial information such as invoice number, amounts, dates, and vendor details",
            function=self._process_invoice
        )
        self._processor = processor_function
    
    def _process_invoice(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Process an invoice document.
        
        Args:
            parameters: Parameters with invoice document path
            tool_context: Tool context
            
        Returns:
            Invoice processing results with structured financial data
        """
        document_path = parameters.get("document_path")
        if not document_path:
            raise ValueError("Invoice document path not provided")
            
        return self._processor(document_path, tool_context)


class ValidationTool(FinflowTool):
    """Tool for validating documents against rules."""
    
    def __init__(self, validation_function: Callable[[Dict[str, Any], Dict[str, Any], Optional[ToolContext]], Dict[str, Any]]):
        """
        Initialize validation tool.
        
        Args:
            validation_function: Function to validate documents
        """
        super().__init__(
            name="validate_document",
            description="Validate a document against compliance rules",
            function=self._validate_document
        )
        self._validation = validation_function
    
    def _validate_document(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Validate a document.
        
        Args:
            parameters: Parameters with document data and rules
            tool_context: Tool context
            
        Returns:
            Validation results
        """
        document_data = parameters.get("document_data")
        rules = parameters.get("rules")
        
        if not document_data:
            raise ValueError("Document data not provided")
        if not rules:
            raise ValueError("Rules not provided")
            
        return self._validation(document_data, rules, tool_context)


class DataStorageTool(FinflowTool):
    """Tool for storing document data."""
    
    def __init__(self, storage_function: Callable[[str, Dict[str, Any], Optional[ToolContext]], Dict[str, Any]]):
        """
        Initialize storage tool.
        
        Args:
            storage_function: Function to store data
        """
        super().__init__(
            name="store_document",
            description="Store document data in BigQuery",
            function=self._store_document
        )
        self._storage = storage_function
    
    def _store_document(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Store document data.
        
        Args:
            parameters: Parameters with document data
            tool_context: Tool context
            
        Returns:
            Storage results
        """
        document_data = parameters.get("document_data")
        document_id = parameters.get("document_id")
        
        if not document_data:
            raise ValueError("Document data not provided")
        if not document_id:
            raise ValueError("Document ID not provided")
            
        return self._storage(document_id, document_data, tool_context)


class AnalyticsTool(FinflowTool):
    """Tool for analyzing financial data."""
    
    def __init__(self, analytics_function: Callable[[Dict[str, Any], str, Optional[ToolContext]], Dict[str, Any]]):
        """
        Initialize analytics tool.
        
        Args:
            analytics_function: Function to analyze data
        """
        super().__init__(
            name="analyze_document",
            description="Analyze document data and generate financial insights",
            function=self._analyze_document
        )
        self._analytics = analytics_function
    
    def _analyze_document(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Analyze document data.
        
        Args:
            parameters: Parameters with document data
            tool_context: Tool context
            
        Returns:
            Analysis results
        """
        document_data = parameters.get("document_data")
        analysis_type = parameters.get("analysis_type", "basic")
        
        if not document_data:
            raise ValueError("Document data not provided")
            
        return self._analytics(document_data, analysis_type, tool_context)
