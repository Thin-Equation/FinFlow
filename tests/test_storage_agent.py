"""
Unit tests for the StorageAgent.
"""

import json
import pytest
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from agents.storage_agent import StorageAgent
from models.documents import DocumentType, DocumentStatus
from tools import bigquery


class TestStorageAgent(unittest.TestCase):
    """Test suite for StorageAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration
        self.mock_config = {
            "bigquery": {
                "project_id": "mock-project",
                "dataset_id": "mock_dataset",
                "location": "US"
            },
            "storage": {
                "enable_cache": True,
                "cache_ttl_seconds": 300,
                "max_cache_size": 100
            }
        }
        
        # Create a sample document for testing
        self.sample_document = {
            "document_id": "doc-001",
            "document_type": "invoice",
            "document_number": "INV-123",
            "status": "submitted",
            "issue_date": "2025-01-01",
            "due_date": "2025-01-31",
            "currency": "USD",
            "total_amount": 1000.00,
            "subtotal": 900.00,
            "issuer": {
                "entity_id": "vendor-001",
                "name": "Test Vendor"
            },
            "recipient": {
                "entity_id": "customer-001",
                "name": "Test Customer"
            },
            "line_items": [
                {
                    "item_id": "item-001",
                    "description": "Test Item 1",
                    "quantity": 2,
                    "unit_price": 250.00,
                    "total_amount": 500.00
                },
                {
                    "item_id": "item-002",
                    "description": "Test Item 2",
                    "quantity": 1,
                    "unit_price": 400.00,
                    "total_amount": 400.00
                }
            ]
        }
        
        # Sample entity for testing
        self.sample_entity = {
            "entity_id": "entity-001",
            "entity_type": "vendor",
            "name": "Test Vendor Corp",
            "tax_id": "123-45-6789",
            "email": "contact@testvendor.com",
            "phone": "555-1234",
            "website": "https://testvendor.com",
            "address": {
                "street_address": "123 Main St",
                "city": "Test City",
                "state": "TS",
                "postal_code": "12345",
                "country": "USA"
            }
        }
    
    @patch('tools.bigquery.create_dataset')
    @patch('tools.bigquery.create_financial_tables')
    def test_init(self, mock_create_tables, mock_create_dataset):
        """Test StorageAgent initialization."""
        # Setup mocks
        mock_create_dataset.return_value = {
            "status": "success",
            "message": "Dataset mock_dataset already exists"
        }
        
        mock_create_tables.return_value = {
            "status": "success",
            "tables": {
                "documents": "created",
                "entities": "created"
            }
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Verify dataset creation was called
        mock_create_dataset.assert_called_once_with(
            project_id="mock-project",
            dataset_id="mock_dataset",
            location="US"
        )
        
        # Verify tables creation was called
        mock_create_tables.assert_called_once_with(
            project_id="mock-project",
            dataset_id="mock_dataset"
        )
        
        # Verify configuration was loaded correctly
        self.assertEqual(agent.bigquery_config.project_id, "mock-project")
        self.assertEqual(agent.bigquery_config.dataset_id, "mock_dataset")
    
    @patch('tools.bigquery.store_document')
    @patch('tools.bigquery.store_batch')
    def test_store_document(self, mock_store_batch, mock_store_document):
        """Test document storage."""
        # Setup mocks
        mock_store_document.return_value = {
            "status": "success",
            "message": "Document stored successfully",
            "document_id": "doc-001"
        }
        
        mock_store_batch.return_value = {
            "status": "success",
            "message": "Successfully inserted 2 rows",
            "count": 2
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test the tool method
        result = agent._tool_store_document(self.sample_document)
        
        # Verify store_document was called
        mock_store_document.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["document_id"], "doc-001")
        
        # Verify line items were stored
        mock_store_batch.assert_called_once()
        args, _ = mock_store_batch.call_args
        self.assertEqual(len(args[0]), 2)  # Two line items
        self.assertEqual(args[1], "mock-project")
        self.assertEqual(args[2], "mock_dataset")
        self.assertEqual(args[3], "line_items")
    
    @patch('tools.bigquery.query_financial_data')
    def test_retrieve_document(self, mock_query_financial_data):
        """Test document retrieval."""
        # Setup mock
        mock_query_financial_data.return_value = {
            "status": "success",
            "row_count": 1,
            "data": [{
                "document_id": "doc-001",
                "document_type": "invoice",
                "document_number": "INV-123",
                "status": "submitted",
                "issue_date": "2025-01-01T00:00:00",
                "due_date": "2025-01-31T00:00:00",
                "currency": "USD",
                "total_amount": 1000.0,
                "subtotal": 900.0,
                "content": json.dumps(self.sample_document)
            }]
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test the tool method
        result = agent._tool_retrieve_document("doc-001")
        
        # Verify query was called
        mock_query_financial_data.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["document"]["document_id"], "doc-001")
        self.assertEqual(result["document"]["document_type"], "invoice")
    
    @patch('tools.bigquery.query_financial_data')
    def test_query_documents(self, mock_query_financial_data):
        """Test document querying."""
        # Setup mock for main query
        mock_query_financial_data.side_effect = [
            # First call - main query
            {
                "status": "success",
                "row_count": 2,
                "data": [
                    {
                        "document_id": "doc-001",
                        "document_type": "invoice",
                        "status": "submitted"
                    },
                    {
                        "document_id": "doc-002",
                        "document_type": "invoice",
                        "status": "approved"
                    }
                ]
            },
            # Second call - count query
            {
                "status": "success",
                "row_count": 1,
                "data": [{"total_count": 2}]
            }
        ]
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test the tool method
        result = agent._tool_query_documents({
            "filters": {"document_type": "invoice"},
            "limit": 10,
            "offset": 0
        })
        
        # Verify queries were called
        self.assertEqual(mock_query_financial_data.call_count, 2)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["documents"]), 2)
        self.assertEqual(result["count"]["returned"], 2)
        self.assertEqual(result["count"]["total"], 2)
    
    @patch('tools.bigquery.store_document')
    def test_store_entity(self, mock_store_document):
        """Test entity storage."""
        # Setup mock
        mock_store_document.return_value = {
            "status": "success",
            "message": "Entity stored successfully",
            "document_id": "entity-001"
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test the tool method
        result = agent._tool_store_entity(self.sample_entity)
        
        # Verify store_document was called
        mock_store_document.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["entity_id"], "entity-001")
    
    @patch('tools.bigquery.query_financial_data')
    def test_retrieve_entity(self, mock_query_financial_data):
        """Test entity retrieval."""
        # Setup mock
        mock_query_financial_data.return_value = {
            "status": "success",
            "row_count": 1,
            "data": [{
                "entity_id": "entity-001",
                "entity_type": "vendor",
                "name": "Test Vendor Corp",
                "tax_id": "123-45-6789",
                "email": "contact@testvendor.com"
            }]
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test the tool method
        result = agent._tool_retrieve_entity("entity-001")
        
        # Verify query was called
        mock_query_financial_data.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["entity"]["entity_id"], "entity-001")
        self.assertEqual(result["entity"]["entity_type"], "vendor")
    
    @patch('tools.bigquery.store_document')
    def test_create_document_relationship(self, mock_store_document):
        """Test creating document relationships."""
        # Setup mock
        mock_store_document.return_value = {
            "status": "success",
            "message": "Relationship stored successfully",
            "document_id": "rel-001"
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test the tool method
        result = agent._tool_create_document_relationship(
            source_document_id="doc-001",
            target_document_id="doc-002",
            relationship_type="invoice_to_receipt"
        )
        
        # Verify store_document was called
        mock_store_document.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["details"]["source_document_id"], "doc-001")
        self.assertEqual(result["details"]["target_document_id"], "doc-002")
    
    @patch('tools.bigquery.run_financial_analysis')
    def test_run_financial_analysis(self, mock_run_financial_analysis):
        """Test running financial analysis."""
        # Setup mock
        mock_run_financial_analysis.return_value = {
            "status": "success",
            "row_count": 3,
            "data": [
                {"month": "2025-01", "total_expenses": 5000.0, "currency": "USD"},
                {"month": "2025-02", "total_expenses": 4500.0, "currency": "USD"},
                {"month": "2025-03", "total_expenses": 6000.0, "currency": "USD"}
            ]
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test the tool method
        result = agent._tool_run_financial_analysis(
            analysis_type="monthly_expenses",
            parameters={
                "start_date": "2025-01-01",
                "end_date": "2025-03-31"
            }
        )
        
        # Verify analysis was called
        mock_run_financial_analysis.assert_called_once_with(
            analysis_type="monthly_expenses",
            parameters={
                "start_date": "2025-01-01",
                "end_date": "2025-03-31"
            },
            project_id="mock-project",
            dataset_id="mock_dataset"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["data"]), 3)
    
    @patch('tools.bigquery.query_financial_data')
    def test_run_custom_query(self, mock_query_financial_data):
        """Test running a custom query."""
        # Setup mock
        mock_query_financial_data.return_value = {
            "status": "success",
            "row_count": 2,
            "data": [
                {"document_id": "doc-001", "total_amount": 1000.0},
                {"document_id": "doc-002", "total_amount": 1500.0}
            ]
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test the tool method
        query = "SELECT document_id, total_amount FROM `mock-project.mock_dataset.documents` LIMIT 2"
        result = agent._tool_run_custom_query(query)
        
        # Verify query was called
        mock_query_financial_data.assert_called_once_with(
            query=query,
            project_id="mock-project"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["row_count"], 2)
    
    def test_transform_document_for_storage(self):
        """Test document transformation for storage."""
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Transform the document
        transformed = agent._transform_document_for_storage(self.sample_document)
        
        # Verify transformation
        self.assertEqual(transformed["id"], "doc-001")
        self.assertEqual(transformed["document_type"], "invoice")
        self.assertEqual(transformed["document_number"], "INV-123")
        self.assertEqual(transformed["total_amount"], 1000.00)
        self.assertEqual(transformed["issuer_id"], "vendor-001")
        self.assertEqual(transformed["recipient_id"], "customer-001")
        
        # Verify content is JSON string
        self.assertIsInstance(transformed["content"], str)
        content = json.loads(transformed["content"])
        self.assertEqual(content["document_type"], "invoice")
    
    def test_transform_entity_for_storage(self):
        """Test entity transformation for storage."""
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Transform the entity
        transformed = agent._transform_entity_for_storage(self.sample_entity)
        
        # Verify transformation
        self.assertEqual(transformed["id"], "entity-001")
        self.assertEqual(transformed["entity_type"], "vendor")
        self.assertEqual(transformed["name"], "Test Vendor Corp")
        self.assertEqual(transformed["tax_id"], "123-45-6789")
        
        # Verify address is JSON string
        self.assertIsInstance(transformed["address"], str)
        address = json.loads(transformed["address"])
        self.assertEqual(address["city"], "Test City")


class TestStorageAgentAsync(unittest.IsolatedAsyncioTestCase):
    """Test suite for StorageAgent async methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration
        self.mock_config = {
            "bigquery": {
                "project_id": "mock-project",
                "dataset_id": "mock_dataset",
                "location": "US"
            },
            "storage": {
                "enable_cache": True,
                "cache_ttl_seconds": 300,
                "max_cache_size": 100
            }
        }
        
        # Create a sample document for testing
        self.sample_document = {
            "document_id": "doc-001",
            "document_type": "invoice",
            "document_number": "INV-123",
            "status": "submitted",
            "total_amount": 1000.00,
        }
    
    @patch.object(StorageAgent, '_tool_store_document')
    async def test_store_document_async(self, mock_tool_store_document):
        """Test async document storage."""
        # Setup mock
        mock_tool_store_document.return_value = {
            "document_id": "doc-001",
            "status": "success",
            "timestamp": "2025-05-19T12:00:00Z",
            "details": {"message": "Document stored successfully"}
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test async method
        result = await agent.store_document(self.sample_document)
        
        # Verify result
        self.assertEqual(result["document_id"], "doc-001")
        self.assertEqual(result["status"], "success")
        mock_tool_store_document.assert_called_once_with(self.sample_document)
    
    @patch.object(StorageAgent, '_tool_retrieve_document')
    async def test_retrieve_document_async(self, mock_tool_retrieve_document):
        """Test async document retrieval."""
        # Setup mock
        mock_tool_retrieve_document.return_value = {
            "status": "success",
            "document": {
                "document_id": "doc-001",
                "document_type": "invoice",
                "status": "submitted"
            },
            "timestamp": "2025-05-19T12:00:00Z"
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test async method
        result = await agent.retrieve_document("doc-001")
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result["document_id"], "doc-001")
        self.assertEqual(result["document_type"], "invoice")
        mock_tool_retrieve_document.assert_called_once_with("doc-001")
    
    @patch.object(StorageAgent, '_tool_query_documents')
    async def test_query_documents_async(self, mock_tool_query_documents):
        """Test async document querying."""
        # Setup mock
        mock_tool_query_documents.return_value = {
            "status": "success",
            "documents": [
                {"document_id": "doc-001", "document_type": "invoice"},
                {"document_id": "doc-002", "document_type": "receipt"}
            ],
            "count": {"returned": 2, "total": 2},
            "timestamp": "2025-05-19T12:00:00Z"
        }
        
        # Create agent
        agent = StorageAgent(config=self.mock_config)
        
        # Test async method
        query = {
            "filters": {"document_type": "invoice"},
            "limit": 10,
            "offset": 0
        }
        result = await agent.query_documents(query)
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["results"]), 2)
        mock_tool_query_documents.assert_called_once_with(query)


if __name__ == "__main__":
    unittest.main()
