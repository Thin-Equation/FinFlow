#!/bin/bash
# Script to set up and test the Document AI invoice processor

# Set environment variables
export PROJECT_ID="finflow-project"  # Replace with your actual GCP project ID
export LOCATION="us-central1" 
export PROCESSOR_NAME="FinFlow Invoice Processor"
export TRAINING_DATA_DIR="./sample_data/invoices/training"
export VALIDATION_DATA_DIR="./sample_data/invoices/validation"
export TEST_FILE="./sample_data/invoices/test/sample_invoice.pdf"

# Create directories for sample data
mkdir -p $TRAINING_DATA_DIR
mkdir -p $VALIDATION_DATA_DIR
mkdir -p $(dirname $TEST_FILE)

echo "Creating directories for sample data..."
echo "Training data directory: $TRAINING_DATA_DIR"
echo "Validation data directory: $VALIDATION_DATA_DIR"
echo "Test file path: $TEST_FILE"
echo "-------------------"

echo "To complete the invoice processor setup, you need to:"
echo "1. Add sample invoice PDFs to the training and validation directories"
echo "2. Update the PROJECT_ID variable with your actual Google Cloud project ID"
echo "3. Run this script with: bash $(basename $0)"
echo ""
echo "After setup, you can use the processor in your FinFlow application"
echo "by updating the processor_id in your application code."

# When ready to run, uncomment these lines:
# echo "-------------------"
# echo "Setting up the Document AI invoice processor..."
# python -c "
# from tools.document_ai import setup_invoice_processor
# result = setup_invoice_processor('$PROJECT_ID', '$LOCATION', '$PROCESSOR_NAME')
# print(f'Processor setup result: {result}')
# "

# echo "-------------------"
# echo "Training the processor with sample documents..."
# python -c "
# from tools.document_ai import train_invoice_processor
# processor_id = 'projects/$PROJECT_ID/locations/$LOCATION/processors/$PROCESSOR_NAME'
# result = train_invoice_processor(processor_id, '$TRAINING_DATA_DIR')
# print(f'Training result: {result}')
# "

# echo "-------------------"
# echo "Testing the processor with sample document..."
# python -c "
# from tools.document_ai import test_invoice_processor
# processor_id = 'projects/$PROJECT_ID/locations/$LOCATION/processors/$PROCESSOR_NAME'
# result = test_invoice_processor(processor_id, '$TEST_FILE')
# print(f'Test result: {result}')
# "

# echo "-------------------"
# echo "Evaluating processor performance..."
# python -c "
# from tools.document_ai import evaluate_processor_performance
# processor_id = 'projects/$PROJECT_ID/locations/$LOCATION/processors/$PROCESSOR_NAME'
# result = evaluate_processor_performance(processor_id, '$VALIDATION_DATA_DIR')
# print(f'Evaluation result: {result}')
# "
