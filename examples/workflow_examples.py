"""
Example implementations of financial process workflows.

This module demonstrates how to use the workflow system for various financial processes.
"""

import os
import sys
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.workflow_definitions import (
    Workflow,
    WorkflowTask,
    WorkflowDefinition,
    WorkflowExecutionContext,
)
from workflow.sequential_agent import SequentialAgent
from workflow.parallel_agent import ParallelAgent
from workflow.conditional import (
    ConditionalBranching,
    Branch,
    BranchSelectionStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("workflow_examples")

def create_invoice_processing_workflow() -> WorkflowDefinition:
    """
    Create a workflow for invoice processing.
    
    This workflow demonstrates the following steps:
    1. Document extraction
    2. Validation 
    3. Approval routing based on amount
    4. Payment processing
    5. Recording in accounting system
    
    Returns:
        WorkflowDefinition: A workflow for invoice processing
    """
    workflow = Workflow.create(
        name="Invoice Processing",
        description="Process invoices from receipt to payment",
        process_type="invoice_processing"
    )
    
    # Step 1: Extract information from invoice document
    extract_task = WorkflowTask(
        id="extract_invoice_data",
        name="Extract Invoice Data",
        description="Extract structured data from invoice document",
        execute=lambda ctx: extract_invoice_data(ctx),
    )
    workflow.add_task(extract_task)
    
    # Step 2: Validate invoice data
    validate_task = WorkflowTask(
        id="validate_invoice",
        name="Validate Invoice",
        description="Validate invoice data for completeness and correctness",
        execute=lambda ctx: validate_invoice(ctx),
        dependencies=[extract_task.id]
    )
    workflow.add_task(validate_task)
    
    # Step 3: Calculate invoice metrics
    calculate_metrics_task = WorkflowTask(
        id="calculate_metrics",
        name="Calculate Invoice Metrics",
        description="Calculate financial metrics from invoice data",
        execute=lambda ctx: calculate_invoice_metrics(ctx),
        dependencies=[validate_task.id]
    )
    workflow.add_task(calculate_metrics_task)
    
    # Step 4: Determine approval path (conditional branch)
    
    # Define approval branches
    
    # High value invoices (Manager + Director approval)
    high_value_approval_tasks = [
        WorkflowTask(
            id="manager_approval",
            name="Manager Approval",
            description="Get approval from manager",
            execute=lambda ctx: manager_approval(ctx)
        ),
        WorkflowTask(
            id="director_approval", 
            name="Director Approval",
            description="Get approval from director",
            execute=lambda ctx: director_approval(ctx)
        )
    ]
    
    high_value_branch = Branch(
        name="high_value_approval",
        condition=lambda ctx: ctx.get_result("calculate_metrics")["amount"] >= Decimal("10000"),
        tasks=high_value_approval_tasks,
        priority=3
    )
    
    # Medium value invoices (Manager approval only)
    medium_value_approval_tasks = [
        WorkflowTask(
            id="manager_approval_medium",
            name="Manager Approval",
            description="Get approval from manager",
            execute=lambda ctx: manager_approval(ctx)
        )
    ]
    
    medium_value_branch = Branch(
        name="medium_value_approval",
        condition=lambda ctx: Decimal("1000") <= ctx.get_result("calculate_metrics")["amount"] < Decimal("10000"),
        tasks=medium_value_approval_tasks,
        priority=2
    )
    
    # Low value invoices (Automatic approval)
    low_value_approval_tasks = [
        WorkflowTask(
            id="automatic_approval",
            name="Automatic Approval",
            description="Automatically approve low-value invoice",
            execute=lambda ctx: automatic_approval(ctx)
        )
    ]
    
    low_value_branch = Branch(
        name="low_value_approval",
        condition=lambda ctx: ctx.get_result("calculate_metrics")["amount"] < Decimal("1000"),
        tasks=low_value_approval_tasks, 
        priority=1
    )
    
    # Step 5: Process payment
    payment_task = WorkflowTask(
        id="process_payment",
        name="Process Payment",
        description="Process payment for the approved invoice",
        execute=lambda ctx: process_payment(ctx)
        # Dependencies will be added by the conditional branching
    )
    workflow.add_task(payment_task)
    
    # Step 6: Record in accounting system
    record_task = WorkflowTask(
        id="record_transaction",
        name="Record Transaction",
        description="Record the transaction in the accounting system",
        execute=lambda ctx: record_transaction(ctx),
        dependencies=[payment_task.id]
    )
    workflow.add_task(record_task)
    
    # Step 7: Send notifications
    notification_task = WorkflowTask(
        id="send_notifications",
        name="Send Notifications",
        description="Send notifications to relevant parties",
        execute=lambda ctx: send_notifications(ctx),
        dependencies=[record_task.id]
    )
    workflow.add_task(notification_task)
    
    # Step 8: Archive documents
    archive_task = WorkflowTask(
        id="archive_documents",
        name="Archive Documents",
        description="Archive the invoice and related documents",
        execute=lambda ctx: archive_documents(ctx),
        dependencies=[record_task.id]
    )
    workflow.add_task(archive_task)
    
    # Add conditional branching for approval
    approval_branches = ConditionalBranching.add_switch(
        workflow=workflow,
        switch_name="approval_routing",
        branches=[high_value_branch, medium_value_branch, low_value_branch],
        parent_task_id=calculate_metrics_task.id, 
        exit_task_id=payment_task.id,
        strategy=BranchSelectionStrategy.PRIORITY
    )
    
    return workflow

def create_financial_reporting_workflow() -> WorkflowDefinition:
    """
    Create a workflow for financial reporting.
    
    This workflow demonstrates parallel processing of different report components:
    1. Data extraction from multiple sources
    2. Parallel processing of different report sections
    3. Consolidation of results
    4. Report generation and distribution
    
    Returns:
        WorkflowDefinition: A workflow for financial reporting
    """
    workflow = Workflow.create(
        name="Financial Reporting",
        description="Generate monthly financial reports",
        process_type="financial_reporting"
    )
    
    # Step 1: Extract data from various sources
    extract_gl_task = WorkflowTask(
        id="extract_gl_data",
        name="Extract General Ledger Data",
        description="Extract data from the general ledger",
        execute=lambda ctx: extract_gl_data(ctx)
    )
    workflow.add_task(extract_gl_task)
    
    extract_ap_task = WorkflowTask(
        id="extract_ap_data",
        name="Extract Accounts Payable Data",
        description="Extract data from accounts payable",
        execute=lambda ctx: extract_ap_data(ctx)
    )
    workflow.add_task(extract_ap_task)
    
    extract_ar_task = WorkflowTask(
        id="extract_ar_data",
        name="Extract Accounts Receivable Data",
        description="Extract data from accounts receivable",
        execute=lambda ctx: extract_ar_data(ctx)
    )
    workflow.add_task(extract_ar_task)
    
    # Step 2: Transform and validate data
    transform_data_task = WorkflowTask(
        id="transform_data",
        name="Transform Financial Data",
        description="Transform and normalize financial data",
        execute=lambda ctx: transform_financial_data(ctx),
        dependencies=[extract_gl_task.id, extract_ap_task.id, extract_ar_task.id]
    )
    workflow.add_task(transform_data_task)
    
    # Step 3: Parallel processing of report sections
    income_statement_task = WorkflowTask(
        id="generate_income_statement",
        name="Generate Income Statement",
        description="Generate the income statement section",
        execute=lambda ctx: generate_income_statement(ctx),
        dependencies=[transform_data_task.id]
    )
    workflow.add_task(income_statement_task)
    
    balance_sheet_task = WorkflowTask(
        id="generate_balance_sheet",
        name="Generate Balance Sheet",
        description="Generate the balance sheet section",
        execute=lambda ctx: generate_balance_sheet(ctx),
        dependencies=[transform_data_task.id]
    )
    workflow.add_task(balance_sheet_task)
    
    cash_flow_task = WorkflowTask(
        id="generate_cash_flow",
        name="Generate Cash Flow Statement",
        description="Generate the cash flow statement section",
        execute=lambda ctx: generate_cash_flow(ctx),
        dependencies=[transform_data_task.id]
    )
    workflow.add_task(cash_flow_task)
    
    # Step 4: Consolidate report sections
    consolidate_task = WorkflowTask(
        id="consolidate_reports",
        name="Consolidate Reports",
        description="Consolidate all report sections",
        execute=lambda ctx: consolidate_reports(ctx),
        dependencies=[
            income_statement_task.id,
            balance_sheet_task.id,
            cash_flow_task.id
        ]
    )
    workflow.add_task(consolidate_task)
    
    # Step 5: Generate final report
    generate_report_task = WorkflowTask(
        id="generate_final_report",
        name="Generate Final Report",
        description="Generate the final formatted report",
        execute=lambda ctx: generate_final_report(ctx),
        dependencies=[consolidate_task.id]
    )
    workflow.add_task(generate_report_task)
    
    # Step 6: Distribute report (conditional branching based on report type)
    
    # Define distribution methods
    email_distribution_tasks = [
        WorkflowTask(
            id="email_report",
            name="Email Report",
            description="Email the report to recipients",
            execute=lambda ctx: email_report(ctx)
        )
    ]
    
    email_branch = Branch(
        name="email_distribution",
        condition=lambda ctx: ctx.parameters.get("distribution_method") == "email",
        tasks=email_distribution_tasks,
        priority=2
    )
    
    portal_distribution_tasks = [
        WorkflowTask(
            id="upload_to_portal",
            name="Upload to Portal",
            description="Upload the report to the financial portal",
            execute=lambda ctx: upload_to_portal(ctx)
        )
    ]
    
    portal_branch = Branch(
        name="portal_distribution",
        condition=lambda ctx: ctx.parameters.get("distribution_method") == "portal",
        tasks=portal_distribution_tasks,
        priority=1
    )
    
    # Default distribution for no specified method
    default_distribution_tasks = [
        WorkflowTask(
            id="store_locally",
            name="Store Report Locally",
            description="Store the report in local storage",
            execute=lambda ctx: store_report_locally(ctx)
        )
    ]
    
    default_branch = Branch(
        name="default_distribution",
        condition=lambda ctx: True,  # Always evaluate to true for default case
        tasks=default_distribution_tasks
    )
    
    # Step 7: Archive report
    archive_task = WorkflowTask(
        id="archive_report",
        name="Archive Report",
        description="Archive the generated report",
        execute=lambda ctx: archive_report(ctx)
        # Dependencies will be added by conditional branching
    )
    workflow.add_task(archive_task)
    
    # Add conditional branching for distribution
    distribution_branches = ConditionalBranching.add_switch(
        workflow=workflow,
        switch_name="report_distribution",
        branches=[email_branch, portal_branch],
        parent_task_id=generate_report_task.id,
        exit_task_id=archive_task.id,
        strategy=BranchSelectionStrategy.FIRST,
        default_branch=default_branch
    )
    
    return workflow

#
# Example task implementations
#

# Invoice processing tasks
def extract_invoice_data(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Extracting invoice data...")
    # Simulate document extraction
    time.sleep(0.5)
    
    # Return extracted data
    return {
        "invoice_number": "INV-2025-1234",
        "vendor": "Acme Corp",
        "date": "2025-05-15",
        "due_date": "2025-06-15",
        "amount": 5750.00,
        "currency": "USD",
        "line_items": [
            {"description": "Software License", "quantity": 5, "unit_price": 1000.00, "total": 5000.00},
            {"description": "Support", "quantity": 5, "unit_price": 150.00, "total": 750.00}
        ]
    }

def validate_invoice(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Validating invoice...")
    invoice_data = context.get_result("extract_invoice_data")
    
    # Simulate validation
    time.sleep(0.3)
    
    # Perform validation checks
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required fields
    required_fields = ["invoice_number", "vendor", "amount", "currency"]
    for field in required_fields:
        if field not in invoice_data or not invoice_data[field]:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Missing required field: {field}")
    
    # Check line items sum matches total
    if "line_items" in invoice_data:
        total_from_items = sum(item["total"] for item in invoice_data["line_items"])
        if abs(total_from_items - invoice_data["amount"]) > 0.01:  # Allow for small rounding differences
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Line items total ({total_from_items}) does not match invoice amount ({invoice_data['amount']})"
            )
    
    return validation_result

def calculate_invoice_metrics(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Calculating invoice metrics...")
    invoice_data = context.get_result("extract_invoice_data")
    
    # Convert to Decimal for precise financial calculations
    amount = Decimal(str(invoice_data["amount"]))
    
    # Calculate metrics
    return {
        "amount": amount,
        "tax_amount": amount * Decimal("0.1"),  # Example: 10% tax
        "total_with_tax": amount * Decimal("1.1"),
        "currency": invoice_data["currency"],
        "fiscal_year": 2025,
        "fiscal_quarter": 2  # Q2 2025
    }

def manager_approval(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Getting manager approval...")
    # Simulate approval process
    time.sleep(0.5)
    
    return {
        "approved": True,
        "approver": "Jane Manager",
        "approval_date": datetime.now().isoformat(),
        "comments": "Approved after budget verification"
    }

def director_approval(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Getting director approval...")
    # Simulate approval process
    time.sleep(0.7)
    
    return {
        "approved": True,
        "approver": "John Director",
        "approval_date": datetime.now().isoformat(),
        "comments": "Approved - within budget allocation"
    }

def automatic_approval(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Automatic approval for low-value invoice...")
    # Simulate automatic approval
    time.sleep(0.1)
    
    return {
        "approved": True,
        "approver": "System",
        "approval_date": datetime.now().isoformat(),
        "comments": "Automatically approved (low value)"
    }

def process_payment(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Processing payment...")
    # Simulate payment processing
    time.sleep(0.8)
    
    return {
        "payment_id": f"PAY-{int(time.time())}",
        "payment_date": datetime.now().isoformat(),
        "payment_method": "ACH",
        "status": "completed"
    }

def record_transaction(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Recording transaction in accounting system...")
    # Simulate recording in accounting system
    time.sleep(0.4)
    
    invoice_data = context.get_result("extract_invoice_data")
    payment_data = context.get_result("process_payment")
    
    return {
        "transaction_id": f"TXN-{int(time.time())}",
        "gl_entry_date": datetime.now().isoformat(),
        "accounts": [
            {"account": "Accounts Payable", "debit": invoice_data["amount"], "credit": 0},
            {"account": "Cash", "debit": 0, "credit": invoice_data["amount"]}
        ],
        "status": "posted"
    }

def send_notifications(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Sending notifications...")
    # Simulate sending notifications
    time.sleep(0.2)
    
    return {
        "notifications_sent": [
            {"recipient": "accounts@example.com", "type": "payment_confirmation"},
            {"recipient": "vendor@acmecorp.com", "type": "payment_notification"}
        ]
    }

def archive_documents(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Archiving documents...")
    # Simulate archiving
    time.sleep(0.3)
    
    return {
        "archive_id": f"ARC-{int(time.time())}",
        "storage_path": "/archives/invoices/2025/05",
        "archive_date": datetime.now().isoformat()
    }

# Financial reporting tasks
def extract_gl_data(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Extracting GL data...")
    # Simulate GL data extraction
    time.sleep(0.6)
    
    return {"gl_data": "GL data extracted successfully", "status": "success"}

def extract_ap_data(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Extracting AP data...")
    # Simulate AP data extraction
    time.sleep(0.4)
    
    return {"ap_data": "AP data extracted successfully", "status": "success"}

def extract_ar_data(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Extracting AR data...")
    # Simulate AR data extraction
    time.sleep(0.5)
    
    return {"ar_data": "AR data extracted successfully", "status": "success"}

def transform_financial_data(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Transforming financial data...")
    # Simulate data transformation
    time.sleep(0.7)
    
    return {
        "transformed_data": True,
        "time_period": "May 2025",
        "status": "success"
    }

def generate_income_statement(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Generating income statement...")
    # Simulate report generation
    time.sleep(0.8)
    
    return {"section": "income_statement", "status": "generated"}

def generate_balance_sheet(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Generating balance sheet...")
    # Simulate report generation
    time.sleep(0.6)
    
    return {"section": "balance_sheet", "status": "generated"}

def generate_cash_flow(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Generating cash flow statement...")
    # Simulate report generation
    time.sleep(0.7)
    
    return {"section": "cash_flow", "status": "generated"}

def consolidate_reports(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Consolidating report sections...")
    # Simulate consolidation
    time.sleep(0.5)
    
    return {
        "consolidated": True,
        "sections": ["income_statement", "balance_sheet", "cash_flow"],
        "status": "success"
    }

def generate_final_report(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Generating final report...")
    # Simulate final report generation
    time.sleep(0.6)
    
    return {
        "report_id": f"FR-{int(time.time())}",
        "filename": "Financial_Report_May_2025.pdf",
        "size_kb": 2048,
        "status": "generated"
    }

def email_report(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Emailing report...")
    # Simulate email sending
    time.sleep(0.3)
    
    return {
        "email_sent": True,
        "recipients": ["finance@example.com", "cfo@example.com"],
        "sent_at": datetime.now().isoformat()
    }

def upload_to_portal(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Uploading report to portal...")
    # Simulate portal upload
    time.sleep(0.4)
    
    return {
        "upload_success": True,
        "portal_url": "https://finance-portal.example.com/reports/may2025",
        "uploaded_at": datetime.now().isoformat()
    }

def store_report_locally(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Storing report locally...")
    # Simulate local storage
    time.sleep(0.2)
    
    return {
        "stored": True,
        "path": "/reports/2025/05/Financial_Report_May_2025.pdf"
    }

def archive_report(context: WorkflowExecutionContext) -> Dict[str, Any]:
    logger.info("Archiving report...")
    # Simulate archiving
    time.sleep(0.3)
    
    return {
        "archive_id": f"REP-ARC-{int(time.time())}",
        "storage_path": "/archives/reports/2025/05",
        "retention_period": "7 years"
    }

def run_example_workflows():
    """Run the example workflows."""
    # Create agents
    sequential_agent = SequentialAgent()
    parallel_agent = ParallelAgent()
    
    # Create context
    context = {"start_time": datetime.now().isoformat()}
    
    # Create and run the invoice processing workflow
    logger.info("\n\n=== Starting Invoice Processing Workflow (Sequential) ===\n")
    invoice_workflow = create_invoice_processing_workflow()
    invoice_result = sequential_agent.run_workflow(
        workflow_definition=invoice_workflow,
        context=context,
        parameters={"document_id": "DOC-12345"}
    )
    
    logger.info(f"\nInvoice workflow completed with status: {invoice_result.status}")
    if invoice_result.is_successful:
        logger.info("Invoice processing successful!")
    else:
        logger.error(f"Invoice processing failed: {invoice_result.error}")
    
    # Create and run the financial reporting workflow
    logger.info("\n\n=== Starting Financial Reporting Workflow (Parallel) ===\n")
    reporting_workflow = create_financial_reporting_workflow()
    reporting_result = parallel_agent.run_workflow(
        workflow_definition=reporting_workflow,
        context=context,
        parameters={"report_period": "May 2025", "distribution_method": "email"}
    )
    
    logger.info(f"\nFinancial reporting workflow completed with status: {reporting_result.status}")
    if reporting_result.is_successful:
        logger.info("Financial reporting successful!")
    else:
        logger.error(f"Financial reporting failed: {reporting_result.error}")

if __name__ == "__main__":
    run_example_workflows()
