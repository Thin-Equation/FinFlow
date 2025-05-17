#!/bin/bash
# Run invoice processing tests with the document processor agent

# Set up environment
echo "Setting up invoice processing test environment..."

# Set script directory as base
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SAMPLE_DIR="$ROOT_DIR/sample_data/invoices"
OUTPUT_DIR="$ROOT_DIR/test_results"

# Make sure directories exist
mkdir -p "$SAMPLE_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if there are any sample invoices
INVOICE_COUNT=$(ls -1 "$SAMPLE_DIR"/*.pdf 2>/dev/null | wc -l)
if [ "$INVOICE_COUNT" -lt 3 ]; then
    echo "Generating sample invoices for testing..."
    python "$ROOT_DIR/tools/generate_sample_invoices.py" --output "$SAMPLE_DIR" --count 5
else
    echo "Found $INVOICE_COUNT existing sample invoices"
fi

# Run invoice processing test
echo -e "\n🧪 Running invoice processor tests...\n"
python "$ROOT_DIR/scripts/test_invoice_processing.py" --dir "$SAMPLE_DIR" --output "$OUTPUT_DIR/invoice_results.json"

# Display test results summary
echo -e "\n📊 Invoice Processing Test Summary:"
echo "----------------------------------------"
echo "Test results saved to: $OUTPUT_DIR/invoice_results.json"
echo "Sample invoices directory: $SAMPLE_DIR"

# If jq is installed, show a summary of the results
if command -v jq &>/dev/null; then
    if [ -f "$OUTPUT_DIR/invoice_results.json" ]; then
        echo -e "\nSuccess rate:"
        jq -r '[.[] | select(.result.status == "success")] | length as $success | length as $total | "\($success)/\($total) (\(($success / $total * 100) | floor)%)"' "$OUTPUT_DIR/invoice_results.json"
    fi
fi

echo -e "\n✅ Invoice processor test complete"
echo "============================================"
