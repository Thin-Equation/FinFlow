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
