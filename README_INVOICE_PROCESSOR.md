# Document AI Invoice Processor Setup

This directory contains the setup and test scripts for the FinFlow Document AI invoice processor.

## Overview

The Document AI invoice processor is a specialized component that extracts structured information from invoice documents. It uses Google Cloud's Document AI technology to analyze and extract key data fields from invoices such as invoice number, amount, date, vendor information, line items, etc.

## Setup Steps

### 1. Requirements

First, ensure you have the required Python packages:

```bash
pip install -r requirements.txt
```

Additionally, you'll need:
- Google Cloud project with Document AI API enabled
- Service account with Document AI permissions
- Google Cloud SDK installed and configured

### 2. Generate Sample Invoices

Generate sample invoices for training and testing:

```bash
# Generate 20 sample invoices for training
python tools/generate_sample_invoices.py -n 20 -o ./sample_data/invoices/training

# Generate 5 sample invoices for validation
python tools/generate_sample_invoices.py -n 5 -o ./sample_data/invoices/validation
```

### 3. Set Up the Processor

Create and configure the Document AI invoice processor:

```bash
# Make the setup script executable
chmod +x setup_invoice_processor.sh

# Update the PROJECT_ID variable in the script
vim setup_invoice_processor.sh

# Run the setup script
./setup_invoice_processor.sh
```

### 4. Test the Processor

Test the invoice processor with sample invoices:

```bash
# Run the invoice processor test script
./scripts/run_invoice_processor_test.sh

# Test with a specific invoice
python scripts/test_invoice_processing.py --invoice ./sample_data/invoices/sample_invoice_1.pdf

# Debug a specific invoice processing
python scripts/debug_document_processor.py ./sample_data/invoices/sample_invoice_1.pdf
```

## Document Processing Agent Implementation

The DocumentProcessorAgent is implemented using the Google Agent Development Kit (ADK) to process financial documents and extract structured data. It's integrated with Document AI for advanced document analysis.

### Key Components

#### 1. DocumentProcessorAgent Class

Located in `agents/document_processor.py`, this agent:
- Processes documents using Document AI
- Extracts structured information from invoices and other documents
- Normalizes data to standard formats
- Integrates with document ingestion and validation tools

#### 2. Document AI Tools

Located in `tools/document_ai.py`, these tools include:
- `process_document`: Processes general documents using Document AI
- `analyze_financial_document`: Specialized for invoice and receipt analysis
- `test_invoice_processor`: Tests processor performance on sample invoices
- `evaluate_processor_performance`: Evaluates accuracy metrics

#### 3. Agent Tools

Located in `utils/agent_tools.py`, the specialized tools include:
- `DocumentProcessingTool`: General document processing
- `InvoiceProcessingTool`: Specialized for invoice processing

### Using the Document Processor Agent

The agent can be used in two main ways:

1. **Stand-alone processing**:
```python
from agents.document_processor import DocumentProcessorAgent

# Initialize agent
agent = DocumentProcessorAgent()
agent.register_tools()

# Process a document
result = agent.process_document({
    "document_path": "/path/to/invoice.pdf", 
    "document_type": "invoice"
})

# Access structured data
structured_data = result.get("extracted_data", {})
print(f"Invoice Number: {structured_data.get('invoice_number')}")
```

2. **Within the FinFlow agent system**:
```python
# The MasterOrchestratorAgent handles the workflow
result = master_orchestrator.process_document({
    "document_path": "/path/to/invoice.pdf"
})
```

## Data Extraction Capabilities

The Document Processing Agent extracts the following structured data from invoices:

| Field | Description | Example |
|-------|-------------|---------|
| `invoice_number` | Unique invoice identifier | "INV-12345" |
| `date` | Invoice issue date | "2023-05-15" |
| `due_date` | Payment due date | "2023-06-15" |
| `vendor.name` | Name of the vendor | "Acme Corporation" |
| `vendor.tax_id` | Tax ID of the vendor | "12-3456789" |
| `customer.name` | Name of the customer | "TechGlobal Solutions" |
| `customer.tax_id` | Tax ID of the customer | "98-7654321" |
| `total_amount` | Total amount due | 1250.00 |
| `subtotal` | Amount before taxes | 1000.00 |
| `tax_amount` | Amount of tax | 250.00 |
| `currency` | Currency used | "USD" |
| `payment_terms` | Payment terms | "Net 30" |
| `line_items` | Array of items on the invoice | [{"description": "Server Hardware", "quantity": 1, "unit_price": 1000.00, "amount": 1000.00}] |
| `document_type` | Type of document | "invoice" |

### Output Data Structure

The document processor returns a structured data object that looks like this:

```json
{
  "status": "success",
  "document_type": "invoice",
  "extracted_data": {
    "document_type": "invoice",
    "metadata": {
      "confidence": 0.92,
      "pages": 1,
      "processing_timestamp": "2023-05-17T14:23:45.123456"
    },
    "invoice_number": "INV-12345",
    "date": "2023-05-15",
    "due_date": "2023-06-15",
    "vendor": {
      "name": "Acme Corporation",
      "tax_id": "12-3456789"
    },
    "customer": {
      "name": "TechGlobal Solutions",
      "tax_id": "98-7654321"
    },
    "total_amount": 1250.00,
    "subtotal": 1000.00,
    "tax_amount": 250.00,
    "currency": "USD",
    "payment_terms": "Net 30",
    "line_items": [
      {
        "description": "Server Hardware",
        "quantity": 1,
        "unit_price": 1000.00,
        "amount": 1000.00
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

1. **Document AI API not available**:
   - Check Google Cloud project permissions
   - Ensure the Document AI API is enabled
   - Verify service account has proper permissions

2. **Poor extraction results**:
   - Train the processor with more examples of similar invoices
   - Ensure document quality (clear scans, proper orientation)
   - Consider preprocessing documents with the document ingestion tool

3. **Missing fields**:
   - Check if the fields are actually present in the document
   - Train the processor with documents containing those specific fields
   - Some custom fields may require special training

### Debug Mode

For detailed debugging:

```bash
# Enable debug logging
export FINFLOW_LOG_LEVEL=DEBUG

# Debug a specific document
python scripts/debug_document_processor.py /path/to/document.pdf
```

## Future Improvements

1. **Multi-language support**: Enhance extraction for non-English invoices
2. **Custom field detection**: Add support for industry-specific fields
3. **Classification improvement**: Better document type detection
4. **Confidence scoring**: More granular confidence scores per field
5. **Advanced validation**: Cross-field validation logic
6. **Batch processing**: Optimize for processing multiple documents
7. **UI visualization**: Visual interface for extraction results

---

*Last Updated: May 2025*

Test the invoice processor with a sample invoice:

```bash
python test_invoice_processor.py --project-id YOUR_PROJECT_ID --generate-sample
```

## Directory Structure

```
finflow/
├── sample_data/
│   └── invoices/
│       ├── training/      # Training invoice samples
│       ├── validation/    # Validation invoice samples
│       └── test/          # Test invoice samples
├── tools/
│   ├── document_ai.py     # Document AI integration tools
│   └── generate_sample_invoices.py  # Script to generate sample invoices
├── setup_invoice_processor.sh  # Setup script for the invoice processor
├── test_invoice_processor.py   # Test script for the invoice processor
└── docs/
    └── document_ai_invoice_processor.md  # Documentation
```

## Additional Resources

For more information about the invoice processor capabilities and limitations, see:

- [Document AI Invoice Processor Documentation](./docs/document_ai_invoice_processor.md)
- [Google Document AI Documentation](https://cloud.google.com/document-ai)

## Integration with FinFlow

The Document AI invoice processor is integrated with the FinFlow system through the DocumentProcessorAgent. The agent provides tools for processing documents:

- `process_document`: General document processing
- `process_invoice`: Specialized invoice processing

See the [Document Processor Agent](./agents/document_processor.py) for implementation details.
