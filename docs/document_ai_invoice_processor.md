# Document AI Invoice Processor

## Overview
The Document AI Invoice Processor is a specialized component of the FinFlow system that extracts structured information from invoice documents using Google's Document AI technology. This document outlines the capabilities, limitations, and usage of the invoice processor.

## Setup and Configuration

To use the Document AI Invoice Processor, you need to:

1. Create a Document AI processor in your Google Cloud project
2. Train the processor with sample invoice documents
3. Test the processor with a validation dataset
4. Configure the FinFlow application to use the processor

The `setup_invoice_processor.sh` script in the project root provides automated setup for these steps.

## Capabilities

The Invoice Processor can extract the following information from invoice documents:

| Field | Description |
|-------|-------------|
| `invoice_number` | Invoice identifier or reference number |
| `invoice_date` | Date when the invoice was issued |
| `due_date` | Payment due date |
| `supplier_name` | Name of the vendor or supplier |
| `supplier_tax_id` | Tax ID or business registration of the supplier |
| `customer_name` | Name of the customer/recipient |
| `customer_tax_id` | Tax ID of the customer |
| `line_items` | Individual items with description, quantity, and price |
| `subtotal` | Subtotal amount before taxes and fees |
| `tax_amount` | Total tax amount |
| `total_amount` | Total invoice amount |
| `currency` | Currency used for the invoice |
| `payment_terms` | Terms of payment |

### Supported Formats
- PDF
- JPEG
- PNG
- TIFF

### Additional Features
- **Confidence Scores**: Provides confidence score for each extracted field
- **Multi-page Support**: Handles multi-page invoice documents
- **Multi-language Support**: Supports multiple languages (varies by processor)

## Limitations

### Document Quality
Low-quality scans or handwritten documents may have lower extraction accuracy. The processor performs best with clearly printed digital documents or high-quality scans.

### Complex Layouts
Non-standard or highly customized invoice layouts may have reduced accuracy. Best results are obtained with common invoice formats.

### Field Types
Custom fields unique to specific vendors may not be recognized unless the processor is trained with examples containing those fields.

### Language Support
Best performance is achieved with English invoices. Other languages may have varied results depending on the training data.

### Processing Time
Complex multi-page documents may take longer to process than simple, single-page invoices.

### Training Data Requirements
- Performance improves with more diverse training examples
- Requires at least 20 sample invoices for basic model training
- For best results, include invoices from a variety of vendors and layouts

## Usage in FinFlow

### Document Processor Agent

The Document Processor Agent in FinFlow provides two tools for working with invoices:

1. **process_document**: General document processing tool
2. **process_invoice**: Specialized tool for invoice documents

```python
# Example usage in agent code
def process_invoice(document_path):
    result = agent.tools.process_invoice(document_path=document_path)
    return result
```

### Structured Output

The processor returns structured data that maps to the FinFlow data model. The `Invoice` class in `models/documents.py` defines the structure for invoice data.

## Performance Metrics

After training with a sample dataset, you can evaluate processor performance using:

```bash
python -c "
from tools.document_ai import evaluate_processor_performance
processor_id = 'projects/YOUR_PROJECT/locations/LOCATION/processors/PROCESSOR_ID'
result = evaluate_processor_performance(processor_id, './sample_data/invoices/validation')
print(result)
"
```

Key metrics include:
- Average confidence score
- Field extraction rate
- Minimum/maximum confidence values
- Standard deviation of confidence

## Best Practices

1. **Training Data**: Include a diverse set of invoice samples, with different layouts, vendors, and amounts.
2. **Regular Updates**: Retrain the processor periodically as you collect more invoice examples.
3. **Validation**: Always validate critical financial information extracted from invoices.
4. **Field Confidence**: Check confidence scores for important fields and implement human review for low-confidence extractions.

## Integration with Financial Workflows

The Invoice Processor is integrated with the broader FinFlow system for:
- Accounts payable processing
- Financial reporting
- Compliance checks
- Data entry automation

For more details on these integrations, see the relevant agent documentation.
