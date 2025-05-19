"""
Demo script for the StorageAgent.

This script demonstrates the core functionality of the StorageAgent
by creating sample financial data and performing various operations.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

from agents.storage_agent import StorageAgent

def create_sample_invoice(invoice_number: str, amount: float, 
                        vendor_id: str, customer_id: str) -> Dict[str, Any]:
    """Create a sample invoice document."""
    doc_id = str(uuid.uuid4())
    issue_date = datetime.now()
    due_date = issue_date + timedelta(days=30)
    
    return {
        "document_id": doc_id,
        "document_type": "invoice",
        "document_number": invoice_number,
        "status": "submitted",
        "issue_date": issue_date.strftime("%Y-%m-%d"),
        "due_date": due_date.strftime("%Y-%m-%d"),
        "currency": "USD",
        "total_amount": amount,
        "subtotal": amount * 0.9,  # Simplified calculation
        "tax_amount": amount * 0.1,  # Simplified tax calculation
        "issuer": {
            "entity_id": vendor_id,
            "name": "Sample Vendor"  # In a real scenario, this would be looked up
        },
        "recipient": {
            "entity_id": customer_id,
            "name": "Sample Customer"  # In a real scenario, this would be looked up
        },
        "line_items": [
            {
                "item_id": str(uuid.uuid4()),
                "description": "Item 1",
                "quantity": 2,
                "unit_price": amount * 0.4,
                "total_amount": amount * 0.8
            },
            {
                "item_id": str(uuid.uuid4()),
                "description": "Item 2",
                "quantity": 1,
                "unit_price": amount * 0.1,
                "total_amount": amount * 0.1
            }
        ],
        "metadata": {
            "created_by": "demo_script",
            "demo": True
        }
    }


def create_sample_vendor() -> Dict[str, Any]:
    """Create a sample vendor entity."""
    entity_id = str(uuid.uuid4())
    
    return {
        "entity_id": entity_id,
        "entity_type": "vendor",
        "name": f"Sample Vendor {uuid.uuid4().hex[:4].upper()}",
        "tax_id": "123-45-6789",
        "email": "vendor@example.com",
        "phone": "555-1234",
        "website": "https://vendor-example.com",
        "address": {
            "street_address": "123 Vendor St",
            "city": "Vendor City",
            "state": "VS",
            "postal_code": "12345",
            "country": "USA"
        },
        "metadata": {
            "created_by": "demo_script",
            "demo": True
        }
    }


def create_sample_customer() -> Dict[str, Any]:
    """Create a sample customer entity."""
    entity_id = str(uuid.uuid4())
    
    return {
        "entity_id": entity_id,
        "entity_type": "customer",
        "name": f"Sample Customer {uuid.uuid4().hex[:4].upper()}",
        "tax_id": "987-65-4321",
        "email": "customer@example.com",
        "phone": "555-5678",
        "website": "https://customer-example.com",
        "address": {
            "street_address": "456 Customer Ave",
            "city": "Customer City",
            "state": "CS",
            "postal_code": "67890",
            "country": "USA"
        },
        "metadata": {
            "created_by": "demo_script",
            "demo": True
        }
    }


async def demo_storage_agent():
    """Demonstrate StorageAgent functionality."""
    print("Initializing StorageAgent...")
    
    # Create explicit configuration for demo
    demo_config = {
        "bigquery": {
            "project_id": "finflow-demo-project",  # Replace with an actual project ID if testing with real BigQuery
            "dataset_id": "finflow_demo",
            "location": "US"
        },
        "storage": {
            "enable_cache": True,
            "cache_ttl_seconds": 300,
            "max_cache_size": 100
        }
    }
    
    # Initialize storage agent with demo configuration
    agent = StorageAgent(config=demo_config)
    
    # Create sample entities
    print("\nCreating sample vendor...")
    vendor = create_sample_vendor()
    vendor_result = agent._tool_store_entity(vendor)
    print(f"Stored vendor: {vendor_result['entity_id']} - Status: {vendor_result['status']}")
    
    print("\nCreating sample customer...")
    customer = create_sample_customer()
    customer_result = agent._tool_store_entity(customer)
    print(f"Stored customer: {customer_result['entity_id']} - Status: {customer_result['status']}")
    
    # Create and store multiple invoices
    invoices: List[Dict[str, Any]] = []
    print("\nCreating sample invoices...")
    
    # Create invoices for different months
    for month in range(1, 4):  # January to March
        for i in range(1, 4):  # Three invoices per month
            invoice_number = f"INV-2025{month:02d}-{i:03d}"
            amount = 1000 * i  # Different amounts
            
            # Create invoice with specific date in the month
            invoice = create_sample_invoice(
                invoice_number=invoice_number,
                amount=amount,
                vendor_id=vendor["entity_id"],
                customer_id=customer["entity_id"]
            )
            
            # Override issue date to be in the specific month
            date = datetime(2025, month, i*5)  # Spread throughout the month
            invoice["issue_date"] = date.strftime("%Y-%m-%d")
            invoice["due_date"] = (date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            invoices.append(invoice)
    
    # Store each invoice
    for invoice in invoices:
        result = await agent.store_document(invoice)
        print(f"Stored invoice: {invoice['document_number']} - Status: {result['status']}")
    
    # Create relationships between some invoices
    print("\nCreating document relationships...")
    for i in range(len(invoices) - 1):
        source_id = invoices[i]["document_id"]
        target_id = invoices[i+1]["document_id"]
        relationship_type = "invoice_sequence"
        
        result = agent._tool_create_document_relationship(
            source_document_id=source_id,
            target_document_id=target_id,
            relationship_type=relationship_type,
            metadata={"demo": True, "sequence": i+1}
        )
        print(f"Created relationship {i+1}: {result['status']}")
    
    # Query documents by type
    print("\nQuerying invoices...")
    query_result = await agent.query_documents({
        "filters": {"document_type": "invoice"},
        "limit": 5,
        "order_by": "total_amount DESC"
    })
    
    print(f"Found {query_result['count']} invoices")
    for i, doc in enumerate(query_result["results"]):
        print(f"  {i+1}. {doc['document_number']} - Amount: {doc['total_amount']}")
    
    # Run financial analysis
    print("\nRunning financial analysis...")
    analysis = agent._tool_run_financial_analysis(
        analysis_type="monthly_expenses",
        parameters={
            "start_date": "2025-01-01",
            "end_date": "2025-03-31"
        }
    )
    
    print("Monthly Expenses:")
    if analysis["status"] == "success" and "data" in analysis:
        for item in analysis["data"]:
            if "month" in item and "total_expenses" in item:
                print(f"  {item['month']}: ${item['total_expenses']:.2f} {item.get('currency', 'USD')}")
    
    # Retrieve a specific document
    demo_doc_id = invoices[0]["document_id"]
    print(f"\nRetrieving document {demo_doc_id}...")
    doc = await agent.retrieve_document(demo_doc_id)
    
    if doc:
        print(f"Retrieved document: {doc['document_id']}")
        print(f"  Type: {doc['document_type']}")
        print(f"  Number: {doc['content']['document_number']}")
        print(f"  Amount: ${doc['content']['total_amount']:.2f}")
    else:
        print("Document not found")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_storage_agent())
