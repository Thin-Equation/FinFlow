"""
Run and test the workflow system.

This script demonstrates the workflow system capabilities by running the example workflows
and generating visualizations and reports.
"""

import os
import sys
import logging
import time
from datetime import datetime
import argparse
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.workflow_definitions import (
    WorkflowDefinition
)
from workflow.sequential_agent import SequentialAgent
from workflow.parallel_agent import ParallelAgent
from workflow.workflow_monitor import (
    WorkflowMonitor,
    WorkflowInspector,
    create_workflow_visualization
)
from examples.workflow_examples import (
    create_invoice_processing_workflow,
    create_financial_reporting_workflow
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("workflow_runner")

def run_and_visualize_workflow(
    workflow: WorkflowDefinition,
    agent,
    parameters: Dict[str, Any],
    output_dir: str,
    inspector: Optional[WorkflowInspector] = None
):
    """
    Run a workflow, monitor its execution, and create visualizations.
    
    Args:
        workflow: The workflow to run
        agent: The agent to execute the workflow (Sequential or Parallel)
        parameters: Parameters for the workflow
        output_dir: Directory to save visualizations and reports
    """
    logger.info(f"Running workflow '{workflow.name}' with {agent.__class__.__name__}")
    
    # Register workflow with inspector if provided
    if inspector:
        inspector.register_workflow(workflow)
    
    # Create execution context
    context = {"start_time": datetime.now().isoformat()}
    
    # Run the workflow
    start_time = time.time()
    result = agent.run_workflow(workflow, context, parameters)
    execution_time = time.time() - start_time
    
    # Log execution results
    logger.info(f"Workflow '{workflow.name}' completed with status: {result.status}")
    logger.info(f"Execution time: {execution_time:.2f} seconds")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create workflow monitor
    monitor = WorkflowMonitor()
    
    # Analyze workflow
    analysis = monitor.analyze_workflow_definition(workflow)
    logger.info(f"Workflow analysis: {workflow.name}")
    logger.info(f"  Task count: {analysis['task_count']}")
    logger.info(f"  Potential parallelism: max={analysis['potential_parallelism']['max_parallelism']}, avg={analysis['potential_parallelism']['avg_parallelism']:.2f}")
    
    # Create visualizations and reports
    graph_path, report_path = create_workflow_visualization(
        workflow=workflow,
        execution_result=result,
        output_dir=output_dir
    )
    
    # Export execution report in different formats
    text_report_path = os.path.join(output_dir, f"{workflow.id}_report.txt")
    json_report_path = os.path.join(output_dir, f"{workflow.id}_report.json")
    
    monitor.export_execution_report(workflow, result, text_report_path, "text")
    monitor.export_execution_report(workflow, result, json_report_path, "json")
    
    logger.info(f"Workflow execution visualized:")
    logger.info(f"  Graph: {graph_path}")
    logger.info(f"  HTML Report: {report_path}")
    logger.info(f"  Text Report: {text_report_path}")
    logger.info(f"  JSON Report: {json_report_path}")
    
    return result
    
def run_workflow_examples():
    """Run the example workflows."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run workflow examples")
    parser.add_argument("--output-dir", default="workflow_output", help="Directory to save output files")
    parser.add_argument("--invoice-amount", type=float, default=5000, help="Invoice amount for testing")
    parser.add_argument("--sequential-only", action="store_true", help="Run only sequential agent")
    parser.add_argument("--parallel-only", action="store_true", help="Run only parallel agent")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create agents
    sequential_agent = SequentialAgent()
    parallel_agent = ParallelAgent(max_workers=4)
    
    # Create workflow inspector
    inspector = WorkflowInspector()
    
    # Run invoice processing workflow with sequential agent
    if not args.parallel_only:
        logger.info("\n\n=== Running Invoice Processing Workflow with Sequential Agent ===\n")
        invoice_workflow = create_invoice_processing_workflow()
        invoice_result = run_and_visualize_workflow(
            workflow=invoice_workflow,
            agent=sequential_agent,
            parameters={
                "document_id": f"DOC-{int(time.time())}",
                "invoice_amount": args.invoice_amount
            },
            output_dir=os.path.join(args.output_dir, "invoice_sequential"),
            inspector=inspector
        )
    
    # Run invoice processing workflow with parallel agent (to compare performance)
    if not args.sequential_only:
        logger.info("\n\n=== Running Invoice Processing Workflow with Parallel Agent ===\n")
        invoice_workflow_parallel = create_invoice_processing_workflow()
        invoice_result_parallel = run_and_visualize_workflow(
            workflow=invoice_workflow_parallel,
            agent=parallel_agent,
            parameters={
                "document_id": f"DOC-{int(time.time())}",
                "invoice_amount": args.invoice_amount
            },
            output_dir=os.path.join(args.output_dir, "invoice_parallel"),
            inspector=inspector
        )
    
    # Run financial reporting workflow with sequential agent
    if not args.parallel_only:
        logger.info("\n\n=== Running Financial Reporting Workflow with Sequential Agent ===\n")
        reporting_workflow = create_financial_reporting_workflow()
        reporting_result = run_and_visualize_workflow(
            workflow=reporting_workflow,
            agent=sequential_agent,
            parameters={
                "report_period": "May 2025",
                "distribution_method": "email"
            },
            output_dir=os.path.join(args.output_dir, "reporting_sequential"),
            inspector=inspector
        )
    
    # Run financial reporting workflow with parallel agent (most suitable for this workflow)
    if not args.sequential_only:
        logger.info("\n\n=== Running Financial Reporting Workflow with Parallel Agent ===\n")
        reporting_workflow_parallel = create_financial_reporting_workflow()
        reporting_result_parallel = run_and_visualize_workflow(
            workflow=reporting_workflow_parallel,
            agent=parallel_agent,
            parameters={
                "report_period": "May 2025",
                "distribution_method": "portal"
            },
            output_dir=os.path.join(args.output_dir, "reporting_parallel"),
            inspector=inspector
        )
    
    # Print summary
    logger.info("\n\n=== Workflow Execution Summary ===\n")
    
    # Create comparison report if both agents were run
    if not args.sequential_only and not args.parallel_only:
        logger.info("Performance Comparison:")
        
        # Compare invoice workflow
        seq_time = invoice_result.execution_time if invoice_result and hasattr(invoice_result, 'execution_time') else None
        parallel_time = invoice_result_parallel.execution_time if invoice_result_parallel and hasattr(invoice_result_parallel, 'execution_time') else None
        
        logger.info(f"Invoice Processing:")
        if seq_time is not None:
            logger.info(f"  Sequential: {seq_time:.2f}s")
        else:
            logger.info(f"  Sequential: N/A")
            
        if parallel_time is not None:
            logger.info(f"  Parallel: {parallel_time:.2f}s")
        else:
            logger.info(f"  Parallel: N/A")
            
        if seq_time is not None and parallel_time is not None and parallel_time > 0:
            speedup = seq_time / parallel_time
            logger.info(f"  Speedup: {speedup:.2f}x")
        else:
            logger.info(f"  Speedup: N/A")
        
        # Compare reporting workflow
        seq_time = reporting_result.execution_time if reporting_result and hasattr(reporting_result, 'execution_time') else None
        parallel_time = reporting_result_parallel.execution_time if reporting_result_parallel and hasattr(reporting_result_parallel, 'execution_time') else None
        
        logger.info(f"Financial Reporting:")
        if seq_time is not None:
            logger.info(f"  Sequential: {seq_time:.2f}s")
        else:
            logger.info(f"  Sequential: N/A")
            
        if parallel_time is not None:
            logger.info(f"  Parallel: {parallel_time:.2f}s")
        else:
            logger.info(f"  Parallel: N/A")
            
        if seq_time is not None and parallel_time is not None and parallel_time > 0:
            speedup = seq_time / parallel_time
            logger.info(f"  Speedup: {speedup:.2f}x")
        else:
            logger.info(f"  Speedup: N/A")
        
    logger.info(f"\nAll workflow results saved to: {args.output_dir}")
    logger.info("View the HTML reports for detailed execution information")

if __name__ == "__main__":
    run_workflow_examples()
