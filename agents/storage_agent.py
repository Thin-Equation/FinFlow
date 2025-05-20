"""
Storage agent for the FinFlow system.

This agent is responsible for all database operations including:
1. Document storage and retrieval
2. Entity management
3. Relationship tracking
4. Financial data analysis
5. Data transformation and caching
"""

import json
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, TypedDict, cast

from google.adk.tools import BaseTool, ToolContext  # type: ignore
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from config.config_loader import ConfigLoader
from models.documents import FinancialDocument, DocumentType, DocumentStatus
from models.entities import FinancialEntity, EntityType
from tools import bigquery


class StorageResult(TypedDict):
    """Result of a storage operation."""
    document_id: str
    status: str
    timestamp: str
    details: Optional[Dict[str, Any]]


class StorageQuery(TypedDict):
    """Query parameters for document retrieval."""
    filters: Dict[str, Any]
    limit: Optional[int]
    offset: Optional[int]
    order_by: Optional[str]


class Document(TypedDict):
    """Simplified document model for storage operations."""
    document_id: str
    document_type: str
    status: str
    content: Optional[Dict[str, Any]]


class QueryResult(TypedDict):
    """Result of a query operation."""
    count: int
    results: List[Dict[str, Any]]
    status: str
    execution_time_ms: float


class CacheConfig(BaseModel):
    """Configuration for caching."""
    enable_cache: bool = True
    default_ttl_seconds: int = 300
    max_cache_size: int = 100


class BigQueryConfig(BaseModel):
    """Configuration for BigQuery."""
    project_id: str
    dataset_id: str
    location: str = "US"


class StorageAgent(BaseAgent):
    """
    Agent responsible for managing data persistence in BigQuery and other storage systems.
    
    This agent handles all database operations including document storage, retrieval,
    entity management, relationship tracking, and financial data analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the storage agent.
        
        Args:
            config: Configuration options for the storage agent.
                   If None, will load from config files.
        """
        # Initialize base agent first
        super().__init__(
            name="FinFlow_Storage",
            model="gemini-2.0-flash",
            description="Manages data persistence in BigQuery for financial data",
            instruction="""
            You are a storage agent for financial document data.
            Your job is to:
            
            1. Store processed documents in BigQuery
            2. Create relationships between entities and documents
            3. Implement data versioning and audit trails
            4. Handle data retrieval requests from other agents
            5. Maintain data consistency and integrity
            6. Perform financial data analysis
            7. Transform and cache data for efficient retrieval
            
            You should ensure that data is stored efficiently and can be retrieved quickly.
            Always validate input data before storage and maintain referential integrity.
            """
        )
        
        # Load configuration - add these to __dict__ directly to avoid pydantic validation
        config_loader = ConfigLoader()
        self.__dict__["config_loader"] = config_loader
        self.__dict__["config"] = config if config else config_loader.load_config()
        
        # Initialize logger explicitly 
        import logging
        self.logger = logging.getLogger(f"finflow.agents.FinFlow_Storage")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        # Extract BigQuery configuration
        bigquery_config = BigQueryConfig(
            project_id=self.__dict__["config"].get("bigquery", {}).get("project_id", ""),
            dataset_id=self.__dict__["config"].get("bigquery", {}).get("dataset_id", "finflow"),
            location=self.__dict__["config"].get("bigquery", {}).get("location", "US")
        )
        self.__dict__["bigquery_config"] = bigquery_config
        
        # Initialize cache configuration
        cache_config = CacheConfig(
            enable_cache=self.__dict__["config"].get("storage", {}).get("enable_cache", True),
            default_ttl_seconds=self.__dict__["config"].get("storage", {}).get("cache_ttl_seconds", 300),
            max_cache_size=self.__dict__["config"].get("storage", {}).get("max_cache_size", 100)
        )
        self.__dict__["cache_config"] = cache_config
        
        # Register storage tools
        self._register_tools()
        
        # Initialize database if needed
        self._init_database()
        
        self._log("info", "StorageAgent initialized successfully")
    
    def _init_database(self) -> None:
        """
        Initialize the database if needed.
        Creates the dataset and tables if they don't exist.
        """
        try:
            # Create dataset if it doesn't exist
            bigquery_config = self.__dict__["bigquery_config"]
            result = bigquery.create_dataset(
                project_id=bigquery_config.project_id,
                dataset_id=bigquery_config.dataset_id,
                location=bigquery_config.location
            )
            self._log("info", f"Dataset initialization result: {result['message']}")
            
            # Create tables if they don't exist
            tables_result = bigquery.create_financial_tables(
                project_id=bigquery_config.project_id,
                dataset_id=bigquery_config.dataset_id
            )
            self._log("info", f"Tables created: {', '.join(tables_result['tables'].keys())}")
            
        except Exception as e:
            self._log("error", f"Error initializing database: {str(e)}")
            # Don't raise the exception, let the agent continue to function
            # even if db initialization fails - it might work with existing tables
    
    def _register_tools(self) -> None:
        """Register storage tools for the agent to use."""
        
        # Create a tool using FinflowTool from utils.agent_tools
        from utils.agent_tools import FinflowTool
        
        # Access add_tool through __dict__ to bypass Pydantic validation
        if "add_tool" in self.__dict__:
            add_tool_method = self.__dict__["add_tool"]
        else:
            # If add_tool is not found in __dict__, log warning and return early
            if "logger" in self.__dict__ and self.__dict__["logger"]:
                self._log("warning", "add_tool method not found in __dict__, skipping tool registration")
            else:
                print("[WARNING] add_tool method not found in __dict__, skipping tool registration")
            return
        
        # Document batch storage tool
        add_tool_method(FinflowTool(
            name="store_documents_batch_tool",
            description="Store multiple documents in BigQuery",
            function=self._tool_store_documents_batch
        ))
        
        # Document retrieval tool
        add_tool_method(FinflowTool(
            name="retrieve_document_tool",
            description="Retrieve a document from BigQuery",
            function=self._tool_retrieve_document
        ))
        
        # Document query tool
        add_tool_method(FinflowTool(
            name="query_documents_tool",
            description="Query documents from BigQuery",
            function=self._tool_query_documents
        ))
        
        # Entity storage tool
        add_tool_method(FinflowTool(
            name="store_entity_tool",
            description="Store an entity in BigQuery",
            function=self._tool_store_entity
        ))
        
        # Entity retrieval tool
        add_tool_method(FinflowTool(
            name="retrieve_entity_tool",
            description="Retrieve an entity from BigQuery",
            function=self._tool_retrieve_entity
        ))
        
        # Document relationship tool
        add_tool_method(FinflowTool(
            name="create_document_relationship_tool",
            description="Create a relationship between documents",
            function=self._tool_create_document_relationship
        ))
        
        # Financial analysis tool
        add_tool_method(FinflowTool(
            name="run_financial_analysis_tool",
            description="Run financial analysis on stored data",
            function=self._tool_run_financial_analysis
        ))
        
        # Custom query tool
        add_tool_method(FinflowTool(
            name="run_custom_query_tool",
            description="Run a custom SQL query against the BigQuery database",
            function=self._tool_run_custom_query
        ))
        
        # Invalidate cache tool
        add_tool_method(FinflowTool(
            name="invalidate_cache_tool",
            description="Invalidate the cache for a specific query",
            function=self._tool_invalidate_cache
        ))
    
    #### Tool implementation methods ####
    
    def _tool_store_document(self, document: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for storing a document.
        
        Args:
            document: Document data to store
            tool_context: Tool context from ADK
        
        Returns:
            Result of the storage operation
        """
        try:
            # Transform the document to the BigQuery schema format if needed
            transformed_doc = self._transform_document_for_storage(document)
            
            # Ensure document has an ID
            if "id" not in transformed_doc and "document_id" in document:
                transformed_doc["id"] = document["document_id"]
            elif "id" not in transformed_doc:
                transformed_doc["id"] = str(uuid.uuid4())
            
            # Store the document
            result = bigquery.store_document(
                document_data=transformed_doc,
                project_id=self.bigquery_config.project_id,
                dataset_id=self.bigquery_config.dataset_id,
                table_id="documents"
            )
            
            # Extract and store line items if present
            if "line_items" in document:
                self._store_line_items(document.get("line_items", []), transformed_doc["id"])
            
            # Create a document version
            version_result = self.create_document_version(
                document_id=transformed_doc["id"],
                content=document,
                change_summary="Initial version" if not self._check_document_exists(transformed_doc["id"]) else "Document update"
            )
            
            # Log audit event
            self.log_audit_event(
                action="store_document",
                resource_type="document",
                resource_id=transformed_doc["id"],
                details={
                    "document_type": transformed_doc.get("document_type", "unknown"),
                    "status": transformed_doc.get("status", "unknown"),
                    "version_id": version_result.get("version", {}).get("id") if version_result["status"] == "success" else None
                }
            )
            
            # Log the operation
            self._log("info", f"Document stored with ID: {transformed_doc['id']}, status: {result['status']}")
            
            # Return result
            return {
                "document_id": transformed_doc["id"],
                "status": result["status"],
                "timestamp": datetime.utcnow().isoformat(),
                "details": result,
                "version": version_result.get("version") if version_result["status"] == "success" else None
            }
        
        except Exception as e:
            self._log("error", f"Error storing document: {str(e)}")
            return {
                "document_id": document.get("document_id", document.get("id", "unknown")),
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {"error": str(e)}
            }
    
    def _tool_store_documents_batch(self, documents: List[Dict[str, Any]], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for storing multiple documents in batch.
        
        Args:
            documents: List of documents to store
            tool_context: Tool context from ADK
        
        Returns:
            Result of the batch storage operation
        """
        try:
            # Transform each document
            transformed_docs = []
            for doc in documents:
                transformed_doc = self._transform_document_for_storage(doc)
                
                # Ensure document has an ID
                if "id" not in transformed_doc and "document_id" in doc:
                    transformed_doc["id"] = doc["document_id"]
                elif "id" not in transformed_doc:
                    transformed_doc["id"] = str(uuid.uuid4())
                
                transformed_docs.append(transformed_doc)
            
            # Store all documents in a batch
            batch_result = bigquery.store_batch(
                rows=transformed_docs,
                project_id=self.bigquery_config.project_id,
                dataset_id=self.bigquery_config.dataset_id,
                table_id="documents"
            )
            
            # Process line items for each document
            for i, doc in enumerate(documents):
                if "line_items" in doc and doc["line_items"]:
                    doc_id = transformed_docs[i]["id"]
                    self._store_line_items(doc["line_items"], doc_id)
            
            return {
                "status": batch_result["status"],
                "count": batch_result.get("count", 0),
                "message": batch_result.get("message", ""),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            self._log("error", f"Error storing documents batch: {str(e)}")
            return {
                "status": "error",
                "count": 0,
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _tool_retrieve_document(self, document_id: str, include_content: bool = True, 
                              tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for retrieving a document.
        
        Args:
            document_id: ID of the document to retrieve
            include_content: Whether to include the full content
            tool_context: Tool context from ADK
        
        Returns:
            The retrieved document or error information
        """
        try:
            # Build and execute query
            content_field = "content," if include_content else ""
            query = f"""
                SELECT
                    id as document_id,
                    document_type,
                    document_number,
                    status,
                    issue_date,
                    due_date,
                    currency,
                    total_amount,
                    subtotal,
                    issuer_id,
                    recipient_id,
                    confidence_score,
                    created_at,
                    updated_at,
                    {content_field}
                    metadata
                FROM
                    `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.documents`
                WHERE
                    id = '{document_id}'
            """
            
            # Execute query
            result = bigquery.query_financial_data(
                query=query,
                project_id=self.bigquery_config.project_id
            )
            
            # Check for errors or no results
            if result["status"] == "error" or result["row_count"] == 0:
                return {
                    "status": "not_found" if result["status"] == "success" else "error",
                    "document_id": document_id,
                    "message": "Document not found" if result["status"] == "success" else result.get("message", "Unknown error"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Get the document data from the first row
            document_data = result["data"][0]
            
            # If requested, get related line items
            if include_content:
                line_items = self._retrieve_line_items(document_id)
                if line_items and "data" in line_items and line_items["data"]:
                    document_data["line_items"] = line_items["data"]
            
            return {
                "status": "success",
                "document": document_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._log("error", f"Error retrieving document {document_id}: {str(e)}")
            return {
                "status": "error",
                "document_id": document_id,
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _tool_query_documents(self, query_params: Dict[str, Any], 
                            tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for querying documents.
        
        Args:
            query_params: Query parameters
                - filters: Dict of field:value filters
                - limit: Max number of results
                - offset: Result offset for pagination
                - order_by: Field to order results by
                - include_content: Whether to include full document content
            tool_context: Tool context from ADK
        
        Returns:
            Query results
        """
        try:
            # Extract query parameters
            filters = query_params.get("filters", {})
            limit = query_params.get("limit", 100)
            offset = query_params.get("offset", 0)
            order_by = query_params.get("order_by", "created_at DESC")
            include_content = query_params.get("include_content", False)
            
            # Build the WHERE clause from filters
            where_clauses = []
            for field, value in filters.items():
                # Handle different types of values
                if isinstance(value, str):
                    where_clauses.append(f"{field} = '{value}'")
                elif isinstance(value, (int, float)):
                    where_clauses.append(f"{field} = {value}")
                elif isinstance(value, bool):
                    where_clauses.append(f"{field} = {str(value).lower()}")
                elif isinstance(value, dict) and "operator" in value and "value" in value:
                    # Support for custom operators like >, <, LIKE, etc.
                    op = value["operator"]
                    val = value["value"]
                    
                    # Format the value based on type
                    if isinstance(val, str):
                        val_formatted = f"'{val}'"
                    elif val is None:
                        val_formatted = "NULL"
                    else:
                        val_formatted = str(val)
                    
                    where_clauses.append(f"{field} {op} {val_formatted}")
            
            # Combine WHERE clauses
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Select fields based on include_content
            content_field = "content," if include_content else ""
            
            # Build the query
            query = f"""
                SELECT
                    id as document_id,
                    document_type,
                    document_number,
                    status,
                    issue_date,
                    due_date,
                    currency,
                    total_amount,
                    subtotal,
                    issuer_id,
                    recipient_id,
                    confidence_score,
                    created_at,
                    updated_at,
                    {content_field}
                    metadata
                FROM
                    `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.documents`
                WHERE
                    {where_clause}
                ORDER BY
                    {order_by}
                LIMIT
                    {limit}
                OFFSET
                    {offset}
            """
            
            # Count query for pagination
            count_query = f"""
                SELECT
                    COUNT(*) as total_count
                FROM
                    `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.documents`
                WHERE
                    {where_clause}
            """
            
            # Execute both queries (data and count)
            result = bigquery.query_financial_data(
                query=query,
                project_id=self.bigquery_config.project_id
            )
            
            count_result = bigquery.query_financial_data(
                query=count_query,
                project_id=self.bigquery_config.project_id
            )
            
            # Get total count
            total_count = 0
            if count_result["status"] == "success" and count_result["row_count"] > 0:
                total_count = count_result["data"][0].get("total_count", 0)
            
            # Return results with pagination info
            return {
                "status": result["status"],
                "documents": result["data"],
                "count": {
                    "returned": result["row_count"],
                    "total": total_count,
                    "limit": limit,
                    "offset": offset
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._log("error", f"Error querying documents: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "documents": [],
                "count": {
                    "returned": 0,
                    "total": 0
                },
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _tool_store_entity(self, entity: Dict[str, Any], 
                         tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for storing an entity.
        
        Args:
            entity: Entity data to store
            tool_context: Tool context from ADK
        
        Returns:
            Result of the entity storage operation
        """
        try:
            # Transform the entity for storage
            transformed_entity = self._transform_entity_for_storage(entity)
            
            # Ensure entity has an ID
            if "id" not in transformed_entity and "entity_id" in entity:
                transformed_entity["id"] = entity["entity_id"]
            elif "id" not in transformed_entity:
                transformed_entity["id"] = str(uuid.uuid4())
            
            # Store the entity
            result = bigquery.store_document(
                document_data=transformed_entity,
                project_id=self.bigquery_config.project_id,
                dataset_id=self.bigquery_config.dataset_id,
                table_id="entities"
            )
            
            return {
                "entity_id": transformed_entity["id"],
                "status": result["status"],
                "timestamp": datetime.utcnow().isoformat(),
                "details": result
            }
            
        except Exception as e:
            self._log("error", f"Error storing entity: {str(e)}")
            return {
                "entity_id": entity.get("entity_id", entity.get("id", "unknown")),
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {"error": str(e)}
            }
    
    def _tool_retrieve_entity(self, entity_id: str, 
                           tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for retrieving an entity.
        
        Args:
            entity_id: ID of the entity to retrieve
            tool_context: Tool context from ADK
        
        Returns:
            The retrieved entity or error information
        """
        try:
            # Build and execute query
            query = f"""
                SELECT
                    id as entity_id,
                    entity_type,
                    name,
                    tax_id,
                    email,
                    phone,
                    website,
                    address,
                    payment_terms,
                    industry,
                    created_at,
                    updated_at,
                    metadata
                FROM
                    `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.entities`
                WHERE
                    id = '{entity_id}'
            """
            
            # Execute query
            result = bigquery.query_financial_data(
                query=query,
                project_id=self.bigquery_config.project_id
            )
            
            # Check for errors or no results
            if result["status"] == "error" or result["row_count"] == 0:
                return {
                    "status": "not_found" if result["status"] == "success" else "error",
                    "entity_id": entity_id,
                    "message": "Entity not found" if result["status"] == "success" else result.get("message", "Unknown error"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Return the entity data
            return {
                "status": "success",
                "entity": result["data"][0],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._log("error", f"Error retrieving entity {entity_id}: {str(e)}")
            return {
                "status": "error",
                "entity_id": entity_id,
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _tool_create_document_relationship(self, source_document_id: str,
                                        target_document_id: str,
                                        relationship_type: str,
                                        metadata: Dict[str, Any] = None,
                                        tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for creating a relationship between documents.
        
        Args:
            source_document_id: ID of the source document
            target_document_id: ID of the target document
            relationship_type: Type of relationship
            metadata: Additional metadata
            tool_context: Tool context from ADK
        
        Returns:
            Result of the relationship creation
        """
        try:
            # Create relationship record
            relationship = {
                "id": str(uuid.uuid4()),
                "source_document_id": source_document_id,
                "target_document_id": target_document_id,
                "relationship_type": relationship_type,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": json.dumps(metadata) if metadata else json.dumps({})
            }
            
            # Store the relationship
            result = bigquery.store_document(
                document_data=relationship,
                project_id=self.bigquery_config.project_id,
                dataset_id=self.bigquery_config.dataset_id,
                table_id="document_relationships"
            )
            
            return {
                "relationship_id": relationship["id"],
                "status": result["status"],
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "source_document_id": source_document_id,
                    "target_document_id": target_document_id,
                    "relationship_type": relationship_type
                }
            }
            
        except Exception as e:
            self._log("error", f"Error creating document relationship: {str(e)}")
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "message": str(e)
            }
    
    def _tool_run_financial_analysis(self, analysis_type: str,
                                  parameters: Dict[str, Any],
                                  tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for running financial analysis.
        
        Args:
            analysis_type: Type of analysis to run
            parameters: Parameters for the analysis
            tool_context: Tool context from ADK
        
        Returns:
            Analysis results
        """
        try:
            # Run the analysis
            result = bigquery.run_financial_analysis(
                analysis_type=analysis_type,
                parameters=parameters,
                project_id=self.bigquery_config.project_id,
                dataset_id=self.bigquery_config.dataset_id
            )
            
            # If successful, add some metadata about the analysis
            if result["status"] == "success":
                result["analysis_info"] = {
                    "type": analysis_type,
                    "parameters": parameters,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return result
            
        except Exception as e:
            self._log("error", f"Error running financial analysis: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "analysis_type": analysis_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _tool_run_custom_query(self, query: str,
                            tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for running a custom query.
        
        Args:
            query: SQL query to execute
            tool_context: Tool context from ADK
        
        Returns:
            Query results
        """
        try:
            # Execute the query
            result = bigquery.query_financial_data(
                query=query,
                project_id=self.bigquery_config.project_id
            )
            
            # Add execution timestamp
            result["timestamp"] = datetime.utcnow().isoformat()
            
            return result
            
        except Exception as e:
            self._log("error", f"Error running custom query: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _tool_invalidate_cache(self, cache_key: str,
                            tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Tool implementation for invalidating cache.
        
        Args:
            cache_key: Cache key to invalidate
            tool_context: Tool context from ADK
        
        Returns:
            Result of cache invalidation
        """
        try:
            # Invalidate the cache by clearing the LRU cache
            # This is a simplistic approach; in production we'd use a more
            # sophisticated cache manager
            bigquery.cached_query_financial_data.cache_clear()
            
            return {
                "status": "success",
                "message": f"Cache invalidated for key: {cache_key}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._log("error", f"Error invalidating cache: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    #### Helper methods ####
    
    def _transform_document_for_storage(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a document for storage in BigQuery.
        
        Args:
            document: The document to transform
        
        Returns:
            Transformed document ready for BigQuery storage
        """
        # Create a new dict to avoid modifying the original
        transformed = {}
        
        # Handle ID field conversion
        if "document_id" in document:
            transformed["id"] = document["document_id"]
        elif "id" in document:
            transformed["id"] = document["id"]
        else:
            transformed["id"] = str(uuid.uuid4())
        
        # Map common fields
        field_mappings = {
            "document_type": "document_type",
            "document_number": "document_number",
            "status": "status",
            "issue_date": "issue_date",
            "due_date": "due_date",
            "currency": "currency",
            "total_amount": "total_amount",
            "subtotal": "subtotal",
            "tax_amount": "tax_amount",
            "confidence_score": "confidence_score",
            "source_file": "source_file"
        }
        
        for target, source in field_mappings.items():
            if source in document:
                transformed[target] = document[source]
        
        # Handle entity references
        if "issuer" in document:
            issuer = document["issuer"]
            if isinstance(issuer, dict) and "entity_id" in issuer:
                transformed["issuer_id"] = issuer["entity_id"]
            elif isinstance(issuer, str):
                transformed["issuer_id"] = issuer
        
        if "recipient" in document:
            recipient = document["recipient"]
            if isinstance(recipient, dict) and "entity_id" in recipient:
                transformed["recipient_id"] = recipient["entity_id"]
            elif isinstance(recipient, str):
                transformed["recipient_id"] = recipient
        
        # Handle date fields that might be strings
        date_fields = ["issue_date", "due_date"]
        for field in date_fields:
            if field in transformed and isinstance(transformed[field], str):
                # If it's an ISO format string, keep it as is
                if "T" in transformed[field]:
                    pass  # Keep ISO format
                else:
                    # Add time component if it's just a date
                    transformed[field] = f"{transformed[field]}T00:00:00Z"
        
        # Store the full content as JSON
        if "content" not in document:
            # If content not provided, use the document itself as content
            content_dict = {k: v for k, v in document.items() 
                          if k not in ["line_items"]}  # Exclude line items
            transformed["content"] = json.dumps(content_dict)
        else:
            # Use provided content
            content_val = document["content"]
            if isinstance(content_val, dict):
                transformed["content"] = json.dumps(content_val)
            else:
                transformed["content"] = content_val
        
        # Handle metadata
        if "metadata" in document:
            metadata_val = document["metadata"]
            if isinstance(metadata_val, dict):
                transformed["metadata"] = json.dumps(metadata_val)
            else:
                transformed["metadata"] = metadata_val
        else:
            transformed["metadata"] = "{}"
        
        # Add timestamps
        now = datetime.utcnow().isoformat()
        if "created_at" not in transformed:
            transformed["created_at"] = document.get("created_at", now)
        if "updated_at" not in transformed:
            transformed["updated_at"] = document.get("updated_at", now)
        
        return transformed
    
    def _transform_entity_for_storage(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform an entity for storage in BigQuery.
        
        Args:
            entity: The entity to transform
        
        Returns:
            Transformed entity ready for BigQuery storage
        """
        # Create a new dict to avoid modifying the original
        transformed = {}
        
        # Handle ID field conversion
        if "entity_id" in entity:
            transformed["id"] = entity["entity_id"]
        elif "id" in entity:
            transformed["id"] = entity["id"]
        else:
            transformed["id"] = str(uuid.uuid4())
        
        # Map common fields
        field_mappings = {
            "entity_type": "entity_type",
            "name": "name",
            "tax_id": "tax_id",
            "email": "email",
            "phone": "phone",
            "website": "website",
            "industry": "industry",
            "payment_terms": "payment_terms"
        }
        
        for target, source in field_mappings.items():
            if source in entity:
                transformed[target] = entity[source]
        
        # Handle address as JSON
        if "address" in entity:
            address_val = entity["address"]
            if isinstance(address_val, dict):
                transformed["address"] = json.dumps(address_val)
            else:
                transformed["address"] = address_val
        
        # Handle metadata
        if "metadata" in entity:
            metadata_val = entity["metadata"]
            if isinstance(metadata_val, dict):
                transformed["metadata"] = json.dumps(metadata_val)
            else:
                transformed["metadata"] = metadata_val
        else:
            transformed["metadata"] = "{}"
        
        # Add timestamps
        now = datetime.utcnow().isoformat()
        if "created_at" not in transformed:
            transformed["created_at"] = entity.get("created_at", now)
        if "updated_at" not in transformed:
            transformed["updated_at"] = entity.get("updated_at", now)
        
        return transformed
    
    def _store_line_items(self, line_items: List[Dict[str, Any]], document_id: str) -> Dict[str, Any]:
        """
        Store line items for a document.
        
        Args:
            line_items: List of line items
            document_id: ID of the parent document
        
        Returns:
            Result of the storage operation
        """
        if not line_items:
            return {"status": "success", "count": 0, "message": "No line items to store"}
        
        # Transform line items for storage
        transformed_items = []
        for item in line_items:
            transformed = {}
            
            # Generate ID if not present
            if "item_id" in item:
                transformed["id"] = item["item_id"]
            elif "id" in item:
                transformed["id"] = item["id"]
            else:
                transformed["id"] = str(uuid.uuid4())
            
            # Set parent document ID
            transformed["document_id"] = document_id
            
            # Map fields
            field_mappings = {
                "description": "description",
                "quantity": "quantity",
                "unit_price": "unit_price",
                "total_amount": "total_amount",
                "tax_amount": "tax_amount",
                "tax_rate": "tax_rate",
                "account_code": "account_code"
            }
            
            for target, source in field_mappings.items():
                if source in item:
                    transformed[target] = item[source]
            
            # Add metadata as JSON
            if "metadata" in item:
                metadata_val = item["metadata"]
                if isinstance(metadata_val, dict):
                    transformed["metadata"] = json.dumps(metadata_val)
                else:
                    transformed["metadata"] = metadata_val
            else:
                transformed["metadata"] = "{}"
            
            # Add timestamp
            transformed["created_at"] = datetime.utcnow().isoformat()
            
            transformed_items.append(transformed)
        
        # Store the batch of line items
        return bigquery.store_batch(
            rows=transformed_items,
            project_id=self.bigquery_config.project_id,
            dataset_id=self.bigquery_config.dataset_id,
            table_id="line_items"
        )
    
    def _retrieve_line_items(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve line items for a document.
        
        Args:
            document_id: ID of the parent document
        
        Returns:
            Query results with line items
        """
        query = f"""
            SELECT
                id as item_id,
                document_id,
                description,
                quantity,
                unit_price,
                total_amount,
                tax_amount,
                tax_rate,
                account_code,
                created_at,
                metadata
            FROM
                `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.line_items`
            WHERE
                document_id = '{document_id}'
            ORDER BY
                id
        """
        
        return bigquery.query_financial_data(
            query=query,
            project_id=self.bigquery_config.project_id
        )
    
    def _check_document_exists(self, document_id: str) -> bool:
        """
        Check if a document already exists in the database.
        
        Args:
            document_id: ID of the document to check
            
        Returns:
            True if the document exists, False otherwise
        """
        try:
            query = f"""
                SELECT COUNT(*) as doc_count
                FROM `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.documents`
                WHERE id = '{document_id}'
            """
            
            result = bigquery.run_query(
                query=query,
                project_id=self.bigquery_config.project_id
            )
            
            if result["status"] == "success" and result["row_count"] > 0:
                return result["data"][0].get("doc_count", 0) > 0
            return False
            
        except Exception as e:
            self._log("error", f"Error checking if document exists: {str(e)}")
            return False
    
    #### Public API methods - for direct use by other agents ####
    
    async def store_document(self, document: Dict[str, Any]) -> StorageResult:
        """
        Store a document in BigQuery.
        
        Args:
            document: The document to store.
            
        Returns:
            Result of the storage operation.
        """
        self._log("info", f"Storing document: {document.get('document_id', document.get('id', 'unknown'))}")
        
        result = self._tool_store_document(document)
        
        self._log("info", f"Document stored: {result['document_id']}, status: {result['status']}")
        
        return cast(StorageResult, {
            "document_id": result["document_id"],
            "status": result["status"],
            "timestamp": result["timestamp"],
            "details": result.get("details")
        })
    
    async def retrieve_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document from storage.
        
        Args:
            document_id: ID of the document to retrieve.
            
        Returns:
            The retrieved document, or None if not found.
        """
        self._log("info", f"Retrieving document: {document_id}")
        
        result = self._tool_retrieve_document(document_id)
        
        if result["status"] != "success":
            self._log("warning", f"Document not found: {document_id}")
            return None
        
        doc_data = result["document"]
        document: Document = {
            "document_id": doc_data["document_id"],
            "document_type": doc_data["document_type"],
            "status": doc_data["status"],
            "content": doc_data
        }
        
        self._log("info", f"Document retrieved: {document_id}")
        return document
    
    async def query_documents(self, query: StorageQuery) -> QueryResult:
        """
        Query documents based on criteria.
        
        Args:
            query: Query parameters.
            
        Returns:
            Query results.
        """
        self._log("info", f"Querying documents with filters: {query.get('filters', {})}")
        
        result = self._tool_query_documents(query)
        
        query_result: QueryResult = {
            "count": result["count"]["returned"],
            "results": result["documents"],
            "status": result["status"],
            "execution_time_ms": 0  # Not tracking execution time in this version
        }
        
        self._log("info", f"Query returned {query_result['count']} documents")
        return query_result
    
    @lru_cache(maxsize=100)
    async def cached_query(self, query_hash: str, query: StorageQuery) -> QueryResult:
        """
        Cached version of query_documents.
        
        Args:
            query_hash: Hash of the query for cache key
            query: Query parameters
            
        Returns:
            Cached query results
        """
        return await self.query_documents(query)
    
    def create_document_version(self, document_id: str, content: Dict[str, Any], 
                               user_id: Optional[str] = None, 
                               change_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new version of a document.
        
        Args:
            document_id: ID of the document to version
            content: Current content of the document
            user_id: ID of the user creating this version
            change_summary: Summary of changes from previous version
            
        Returns:
            Result of the version creation operation
        """
        try:
            # Get current version number
            query = f"""
                SELECT MAX(version_number) as current_version
                FROM `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.document_versions`
                WHERE document_id = '{document_id}'
            """
            
            result = bigquery.run_query(
                query=query,
                project_id=self.bigquery_config.project_id
            )
            
            current_version = 0
            if result["status"] == "success" and result["row_count"] > 0:
                current_version = result["data"][0].get("current_version") or 0
            
            # Increment version number
            new_version = current_version + 1
            
            # Create a version ID
            version_id = f"ver-{document_id}-{new_version}"
            
            # Insert the new version
            version_data = {
                "id": version_id,
                "document_id": document_id,
                "version_number": new_version,
                "content": json.dumps(content),
                "created_at": datetime.utcnow().isoformat(),
                "created_by": user_id,
                "change_summary": change_summary,
                "metadata": json.dumps({
                    "timestamp": datetime.utcnow().isoformat()
                })
            }
            
            insert_result = bigquery.insert_rows(
                project_id=self.bigquery_config.project_id,
                dataset_id=self.bigquery_config.dataset_id,
                table_id="document_versions",
                rows=[version_data]
            )
            
            # Log the version creation event in audit trail
            self.log_audit_event(
                action="create_version",
                resource_type="document",
                resource_id=document_id,
                user_id=user_id,
                details={
                    "version_id": version_id,
                    "version_number": new_version,
                    "change_summary": change_summary,
                }
            )
            
            return {
                "status": "success" if insert_result["success"] else "error",
                "message": f"Document version {new_version} created for {document_id}",
                "version": {
                    "id": version_id,
                    "document_id": document_id,
                    "version_number": new_version,
                    "created_at": version_data["created_at"],
                }
            }
            
        except Exception as e:
            self._log("error", f"Error creating document version: {str(e)}")
            return {
                "status": "error",
                "message": f"Error creating document version: {str(e)}"
            }
    
    def get_document_versions(self, document_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get the version history for a document.
        
        Args:
            document_id: ID of the document
            limit: Maximum number of versions to return
            
        Returns:
            Dict containing version history
        """
        try:
            query = f"""
                SELECT id, document_id, version_number, created_at, created_by, change_summary
                FROM `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.document_versions`
                WHERE document_id = '{document_id}'
                ORDER BY version_number DESC
                LIMIT {limit}
            """
            
            result = bigquery.run_query(
                query=query,
                project_id=self.bigquery_config.project_id
            )
            
            return {
                "status": result["status"],
                "document_id": document_id,
                "versions": result["data"],
                "count": result["row_count"]
            }
            
        except Exception as e:
            self._log("error", f"Error retrieving document versions: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving document versions: {str(e)}",
                "versions": []
            }
    
    def get_document_version(self, version_id: str) -> Dict[str, Any]:
        """
        Get a specific document version.
        
        Args:
            version_id: ID of the version to retrieve
            
        Returns:
            Dict containing the version data
        """
        try:
            query = f"""
                SELECT *
                FROM `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.document_versions`
                WHERE id = '{version_id}'
            """
            
            result = bigquery.run_query(
                query=query,
                project_id=self.bigquery_config.project_id
            )
            
            if result["status"] == "success" and result["row_count"] > 0:
                version_data = result["data"][0]
                
                # Parse the JSON content
                if "content" in version_data:
                    try:
                        version_data["content"] = json.loads(version_data["content"])
                    except:
                        pass
                
                return {
                    "status": "success",
                    "version": version_data
                }
            else:
                return {
                    "status": "error",
                    "message": f"Version {version_id} not found"
                }
            
        except Exception as e:
            self._log("error", f"Error retrieving document version: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving document version: {str(e)}"
            }
    
    def log_audit_event(self, action: str, resource_type: str, resource_id: str,
                       user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
                       ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Dict[str, Any]:
        """
        Log an event to the audit trail.
        
        Args:
            action: Type of action performed (create, update, delete, etc.)
            resource_type: Type of resource affected (document, entity, etc.)
            resource_id: ID of the resource affected
            user_id: ID of the user who performed the action
            details: Additional details about the action
            ip_address: IP address where the action originated
            user_agent: User agent information
            
        Returns:
            Result of the audit logging operation
        """
        try:
            # Create a unique ID for the audit record
            audit_id = f"audit-{uuid.uuid4()}"
            
            # Prepare the audit record
            audit_record = {
                "id": audit_id,
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "user_id": user_id,
                "details": json.dumps(details) if details else None,
                "ip_address": ip_address,
                "user_agent": user_agent
            }
            
            # Insert the audit record
            insert_result = bigquery.insert_rows(
                project_id=self.bigquery_config.project_id,
                dataset_id=self.bigquery_config.dataset_id,
                table_id="audit_trail",
                rows=[audit_record]
            )
            
            return {
                "status": "success" if insert_result["success"] else "error",
                "message": f"Audit event logged: {action} on {resource_type} {resource_id}",
                "audit_id": audit_id
            }
            
        except Exception as e:
            self._log("error", f"Error logging audit event: {str(e)}")
            return {
                "status": "error",
                "message": f"Error logging audit event: {str(e)}"
            }
    
    def get_audit_trail(self, resource_type: Optional[str] = None, 
                       resource_id: Optional[str] = None,
                       action: Optional[str] = None,
                       user_id: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = 100) -> Dict[str, Any]:
        """
        Get audit trail events with optional filtering.
        
        Args:
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            action: Filter by action type
            user_id: Filter by user ID
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            limit: Maximum number of events to return
            
        Returns:
            Dict containing audit events
        """
        try:
            # Build the query with filters
            where_clauses = []
            
            if resource_type:
                where_clauses.append(f"resource_type = '{resource_type}'")
            
            if resource_id:
                where_clauses.append(f"resource_id = '{resource_id}'")
            
            if action:
                where_clauses.append(f"action = '{action}'")
            
            if user_id:
                where_clauses.append(f"user_id = '{user_id}'")
            
            if start_date:
                where_clauses.append(f"timestamp >= '{start_date}'")
            
            if end_date:
                where_clauses.append(f"timestamp <= '{end_date}'")
            
            # Construct the full query
            where_clause = " AND ".join(where_clauses)
            query = f"""
                SELECT *
                FROM `{self.bigquery_config.project_id}.{self.bigquery_config.dataset_id}.audit_trail`
                {f"WHERE {where_clause}" if where_clauses else ""}
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            
            result = bigquery.run_query(
                query=query,
                project_id=self.bigquery_config.project_id
            )
            
            # Parse JSON in details field
            if result["status"] == "success":
                for row in result["data"]:
                    if "details" in row and row["details"]:
                        try:
                            row["details"] = json.loads(row["details"])
                        except:
                            pass
            
            return {
                "status": result["status"],
                "events": result["data"],
                "count": result["row_count"],
                "filters": {
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "action": action,
                    "user_id": user_id,
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
            
        except Exception as e:
            self._log("error", f"Error retrieving audit trail: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving audit trail: {str(e)}",
                "events": []
            }
    
    def _log(self, level: str, message: str):
        """
        Safe logging helper that checks if logger exists before using it.
        
        Args:
            level: The log level ('debug', 'info', 'warning', 'error', 'critical')
            message: The message to log
        """
        if hasattr(self, 'logger') and self.logger:
            log_method = getattr(self.logger, level.lower(), None)
            if log_method:
                log_method(message)
            else:
                # Fallback to print if log level not found
                print(f"[{level.upper()}] {message}")
        else:
            # Fallback to print if logger is not available
            print(f"[{level.upper()}] {message}")
