"""
Integration tests for the StorageAgent.
These tests interact with actual BigQuery services.
To run these tests, you need to set up BigQuery credentials.

Note: These tests should be run in a test environment, not production.
"""

import os
import json
import uuid
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from google.cloud import bigquery

from agents.storage_agent import StorageAgent
from models.documents import DocumentType, DocumentStatus
from models.entities import EntityType


# Fixture for test configuration
@pytest.fixture
def test_config():
    """Provide test configuration for BigQuery."""
    # Get project ID from environment or use a test project ID
    project_id = os.environ.get('FINFLOW_TEST_PROJECT_ID', 'finflow-test')
    
    # Use a unique dataset for testing to avoid conflicts
    test_id = uuid.uuid4().hex[:8]
    dataset_id = f"finflow_test_{test_id}"
    
    config = {
        "bigquery": {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "location": "US"
        },
        "storage": {
            "enable_cache": True,
            "cache_ttl_seconds": 300,
            "max_cache_size": 100
        }
    }
    
    return config


# Fixture for StorageAgent
@pytest.fixture
async def storage_agent(test_config):
    """Create a StorageAgent instance for testing."""
    agent = StorageAgent(config=test_config)
    
    try:
        yield agent
    finally:
        # Clean up: Delete the test dataset
        client = bigquery.Client(project=test_config["bigquery"]["project_id"])
        dataset_ref = client.dataset(test_config["bigquery"]["dataset_id"])
        try:
            client.delete_dataset(
                dataset_ref, delete_contents=True, not_found_ok=True
            )
        except Exception as e:
            print(f"Error cleaning up test dataset: {e}")


# Test document fixture
@pytest.fixture
def test_document():
    """Create a test document."""
    doc_id = f"test-doc-{uuid.uuid4().hex[:8]}"
    
    return {
        "document_id": doc_id,
        "document_type": "invoice",
        "document_number": f"INV-{uuid.uuid4().hex[:6].upper()}",
        "status": "submitted",
        "issue_date": datetime.now().strftime("%Y-%m-%d"),
        "due_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
        "currency": "USD",
        "total_amount": 1250.75,
        "subtotal": 1150.00,
        "tax_amount": 100.75,
        "issuer": {
            "entity_id": f"vendor-{uuid.uuid4().hex[:8]}",
            "name": "Integration Test Vendor"
        },
        "recipient": {
            "entity_id": f"customer-{uuid.uuid4().hex[:8]}",
            "name": "Integration Test Customer"
        },
        "line_items": [
            {
                "item_id": f"item-{uuid.uuid4().hex[:8]}",
                "description": "Integration Test Item 1",
                "quantity": 2,
                "unit_price": 250.00,
                "total_amount": 500.00,
                "tax_amount": 40.00,
                "tax_rate": 0.08
            },
            {
                "item_id": f"item-{uuid.uuid4().hex[:8]}",
                "description": "Integration Test Item 2",
                "quantity": 1,
                "unit_price": 650.00,
                "total_amount": 650.00,
                "tax_amount": 60.75,
                "tax_rate": 0.0935
            }
        ],
        "metadata": {
            "test_run_id": uuid.uuid4().hex,
            "integration_test": True
        }
    }


# Test entity fixture
@pytest.fixture
def test_entity():
    """Create a test entity."""
    entity_id = f"test-entity-{uuid.uuid4().hex[:8]}"
    
    return {
        "entity_id": entity_id,
        "entity_type": "vendor",
        "name": f"Integration Test Vendor {uuid.uuid4().hex[:4].upper()}",
        "tax_id": "123-45-6789",
        "email": "test@example.com",
        "phone": "555-1234",
        "website": "https://example.com",
        "address": {
            "street_address": "123 Test St",
            "city": "Test City",
            "state": "TS",
            "postal_code": "12345",
            "country": "USA"
        },
        "metadata": {
            "test_run_id": uuid.uuid4().hex,
            "integration_test": True
        }
    }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_lifecycle(storage_agent, test_document):
    """Test the document lifecycle (store, retrieve, query)."""
    # Store document
    store_result = await storage_agent.store_document(test_document)
    
    # Verify storage result
    assert store_result["status"] == "success"
    assert store_result["document_id"] == test_document["document_id"]
    
    # Retrieve document
    doc = await storage_agent.retrieve_document(test_document["document_id"])
    
    # Verify retrieval
    assert doc is not None
    assert doc["document_id"] == test_document["document_id"]
    assert doc["document_type"] == test_document["document_type"]
    assert doc["status"] == test_document["status"]
    
    # Query document
    query_result = await storage_agent.query_documents({
        "filters": {
            "document_type": test_document["document_type"],
            "document_id": test_document["document_id"]
        }
    })
    
    # Verify query result
    assert query_result["status"] == "success"
    assert query_result["count"] > 0
    assert len(query_result["results"]) > 0
    assert query_result["results"][0]["document_id"] == test_document["document_id"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_entity_storage_and_retrieval(storage_agent, test_entity):
    """Test entity storage and retrieval."""
    # Use the tool method directly as we don't have an async entity storage method
    store_result = storage_agent._tool_store_entity(test_entity)
    
    # Verify storage result
    assert store_result["status"] == "success"
    assert store_result["entity_id"] == test_entity["entity_id"]
    
    # Retrieve entity using tool method
    retrieve_result = storage_agent._tool_retrieve_entity(test_entity["entity_id"])
    
    # Verify retrieval
    assert retrieve_result["status"] == "success"
    assert retrieve_result["entity"]["entity_id"] == test_entity["entity_id"]
    assert retrieve_result["entity"]["entity_type"] == test_entity["entity_type"]
    assert retrieve_result["entity"]["name"] == test_entity["name"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_relationships(storage_agent):
    """Test creating and querying document relationships."""
    # Create two test documents
    doc1_id = f"doc-{uuid.uuid4().hex[:8]}"
    doc2_id = f"doc-{uuid.uuid4().hex[:8]}"
    
    # Create sample documents
    doc1 = {
        "document_id": doc1_id,
        "document_type": "purchase_order",
        "document_number": f"PO-{uuid.uuid4().hex[:6].upper()}",
        "status": "approved"
    }
    
    doc2 = {
        "document_id": doc2_id,
        "document_type": "invoice",
        "document_number": f"INV-{uuid.uuid4().hex[:6].upper()}",
        "status": "submitted"
    }
    
    # Store both documents
    await storage_agent.store_document(doc1)
    await storage_agent.store_document(doc2)
    
    # Create relationship
    relationship_result = storage_agent._tool_create_document_relationship(
        source_document_id=doc1_id,
        target_document_id=doc2_id,
        relationship_type="purchase_order_to_invoice",
        metadata={
            "created_by": "integration_test",
            "notes": "Test relationship creation"
        }
    )
    
    # Verify relationship creation
    assert relationship_result["status"] == "success"
    assert "relationship_id" in relationship_result
    assert relationship_result["details"]["source_document_id"] == doc1_id
    assert relationship_result["details"]["target_document_id"] == doc2_id
    
    # Query relationships
    query = f"""
        SELECT *
        FROM `{storage_agent.bigquery_config.project_id}.{storage_agent.bigquery_config.dataset_id}.document_relationships`
        WHERE source_document_id = '{doc1_id}' AND target_document_id = '{doc2_id}'
    """
    
    result = storage_agent._tool_run_custom_query(query)
    
    # Verify query result
    assert result["status"] == "success"
    assert result["row_count"] > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_financial_analysis(storage_agent, test_document):
    """Test running financial analysis."""
    # Store a few test documents with different dates
    doc1 = test_document.copy()
    doc1["issue_date"] = "2025-01-15"
    doc1["total_amount"] = 1000.00
    await storage_agent.store_document(doc1)
    
    doc2 = test_document.copy()
    doc2["document_id"] = f"test-doc-{uuid.uuid4().hex[:8]}"
    doc2["issue_date"] = "2025-02-10"
    doc2["total_amount"] = 1500.00
    await storage_agent.store_document(doc2)
    
    doc3 = test_document.copy()
    doc3["document_id"] = f"test-doc-{uuid.uuid4().hex[:8]}"
    doc3["issue_date"] = "2025-03-05"
    doc3["total_amount"] = 2000.00
    await storage_agent.store_document(doc3)
    
    # Run monthly expenses analysis
    analysis_result = storage_agent._tool_run_financial_analysis(
        analysis_type="monthly_expenses",
        parameters={
            "start_date": "2025-01-01",
            "end_date": "2025-03-31"
        }
    )
    
    # Verify analysis result
    assert analysis_result["status"] == "success"
    assert "data" in analysis_result
    assert len(analysis_result["data"]) > 0
    
    # The exact data structure will depend on the implementation
    # but we should have at least one row for each month
    months_found = set()
    for row in analysis_result["data"]:
        if "month" in row:
            months_found.add(row["month"])
    
    # We should have data for at least January, February, and March
    assert len(months_found) >= 3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_custom_query(storage_agent, test_document):
    """Test running custom queries."""
    # Store a test document
    await storage_agent.store_document(test_document)
    
    # Run a custom query
    query = f"""
        SELECT
            document_id,
            document_type,
            document_number,
            status,
            total_amount
        FROM
            `{storage_agent.bigquery_config.project_id}.{storage_agent.bigquery_config.dataset_id}.documents`
        WHERE
            document_id = '{test_document["document_id"]}'
    """
    
    result = storage_agent._tool_run_custom_query(query)
    
    # Verify query result
    assert result["status"] == "success"
    assert result["row_count"] > 0
    assert result["data"][0]["document_id"] == test_document["document_id"]
