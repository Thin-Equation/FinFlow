"""
Workflow monitoring and visualization utilities.

This module provides tools for monitoring workflow execution, inspecting the state
of workflows, and visualizing the workflow structure and execution paths.
"""

import logging
import json
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import io
import os
import time
from workflow.workflow_definitions import (
    WorkflowDefinition,
    WorkflowResult,
    TaskStatus,
)

# Custom JSON encoder to handle Decimal objects
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

# Try to import visualization libraries (optional dependencies)
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    from IPython.display import display, HTML  # type: ignore
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

class WorkflowMonitor:
    """
    Monitors and provides insights into workflow execution.
    """
    
    def __init__(self, enable_visualization: bool = True):
        """
        Initialize the workflow monitor.
        
        Args:
            enable_visualization: Whether to enable visualization features
        """
        self.logger = logging.getLogger("finflow.workflow.monitor")
        self.enable_visualization = enable_visualization and _HAS_VISUALIZATION
        
        if self.enable_visualization and not _HAS_VISUALIZATION:
            self.logger.warning(
                "Visualization dependencies (matplotlib, networkx) not found. "
                "Install them to enable visualization features."
            )
        
    def analyze_workflow_definition(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """
        Analyze a workflow definition and extract metrics and insights.
        
        Args:
            workflow: The workflow definition to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Basic workflow metrics
        task_count = len(workflow.get_tasks())
        initial_tasks = len(workflow.get_initial_tasks())
        
        # Analyze dependency structure
        dependency_counts = {}
        max_deps = 0
        task_with_max_deps = None
        
        for task_id, task in workflow.tasks.items():
            dep_count = len(task.dependencies)
            dependency_counts[task_id] = dep_count
            
            if dep_count > max_deps:
                max_deps = dep_count
                task_with_max_deps = task_id
        
        # Identify potential execution paths
        terminal_tasks = []
        for task_id in workflow.tasks:
            if not workflow.get_dependent_tasks(task_id):
                terminal_tasks.append(task_id)
        
        # Check for isolated tasks (no dependencies and no dependents)
        isolated_tasks = []
        for task_id, task in workflow.tasks.items():
            if not task.dependencies and not workflow.get_dependent_tasks(task_id):
                isolated_tasks.append(task_id)
        
        # Estimate potential parallelism
        parallelism_estimate = self._estimate_parallelism(workflow)
        
        # Compile analysis
        analysis = {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "workflow_type": getattr(workflow, "process_type", "unknown"),
            "task_count": task_count,
            "initial_tasks": initial_tasks,
            "terminal_tasks": len(terminal_tasks),
            "isolated_tasks": isolated_tasks,
            "max_dependencies": max_deps,
            "task_with_max_dependencies": task_with_max_deps,
            "potential_parallelism": parallelism_estimate,
            "is_valid": workflow.validate()[0],
            "analysis_time": datetime.now().isoformat(),
        }
        
        return analysis
    
    def _estimate_parallelism(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """
        Estimate the potential for parallel execution in a workflow.
        
        Args:
            workflow: The workflow definition
            
        Returns:
            Dict[str, Any]: Parallelism metrics
        """
        tasks = workflow.get_tasks()
        dependency_map = {task.id: set(task.dependencies) for task in tasks}
        
        # Build execution levels (tasks at the same level can be executed in parallel)
        levels = []
        remaining_tasks = set(workflow.tasks.keys())
        
        while remaining_tasks:
            # Find tasks with no unprocessed dependencies
            current_level = set()
            for task_id in remaining_tasks:
                deps = dependency_map[task_id]
                if not deps.intersection(remaining_tasks):  # No remaining dependencies
                    current_level.add(task_id)
            
            if not current_level:  # Handle cycles
                break
                
            levels.append(current_level)
            remaining_tasks -= current_level
        
        # Calculate parallelism metrics
        max_parallelism = max([len(level) for level in levels]) if levels else 0
        avg_parallelism = sum([len(level) for level in levels]) / len(levels) if levels else 0
        execution_stages = len(levels)
        
        return {
            "max_parallelism": max_parallelism,
            "avg_parallelism": avg_parallelism,
            "execution_stages": execution_stages,
            "stage_sizes": [len(level) for level in levels]
        }
    
    def visualize_workflow(
        self, 
        workflow: WorkflowDefinition,
        execution_result: Optional[WorkflowResult] = None,
        filename: Optional[str] = None,
        show_plot: bool = True
    ) -> Optional[str]:
        """
        Visualize a workflow as a directed graph.
        
        Args:
            workflow: The workflow definition to visualize
            execution_result: Optional execution result to color nodes by status
            filename: Optional filename to save the visualization
            show_plot: Whether to display the plot
            
        Returns:
            Optional[str]: Path to saved file if filename is provided
        """
        if not self.enable_visualization:
            self.logger.warning("Visualization is not enabled or dependencies are missing")
            return None
            
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each task
        for task_id, task in workflow.tasks.items():
            # Set node attributes based on task
            node_attrs = {
                "label": f"{task.name}\n({task.id})",
                "description": task.description,
                "status": task.status.value if task.status else "unknown",
            }
            
            # Add execution details if available
            if execution_result and task_id in execution_result.task_results:
                node_attrs["executed"] = True
                if task.is_completed():
                    if task.start_time and task.end_time:
                        duration = (task.end_time - task.start_time).total_seconds()
                        node_attrs["duration"] = f"{duration:.2f}s"
            
            G.add_node(task_id, **node_attrs)
        
        # Add edges for dependencies
        for task_id, task in workflow.tasks.items():
            for dep_id in task.dependencies:
                G.add_edge(dep_id, task_id)
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Determine node colors based on task status
        node_colors = []
        for task_id in G.nodes():
            task = workflow.get_task(task_id)
            if not task:
                node_colors.append("lightgray")
                continue
                
            if task.status == TaskStatus.COMPLETED:
                node_colors.append("green")
            elif task.status == TaskStatus.FAILED:
                node_colors.append("red")
            elif task.status == TaskStatus.RUNNING:
                node_colors.append("yellow")
            elif task.status == TaskStatus.SKIPPED:
                node_colors.append("lightblue")
            else:
                node_colors.append("lightgray")
        
        # Draw the graph
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=2000,
            font_size=8,
            font_weight="bold",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
        )
        
        # Add a title
        plt.title(f"Workflow: {workflow.name} ({workflow.id})")
        
        # Save if filename provided
        if filename:
            plt.savefig(filename)
            self.logger.info(f"Workflow visualization saved to {filename}")
            
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return filename if filename else None
    
    def generate_execution_report(
        self,
        workflow: WorkflowDefinition,
        result: WorkflowResult,
        include_task_details: bool = True,
        format_type: str = "text"
    ) -> str:
        """
        Generate a detailed execution report.
        
        Args:
            workflow: The workflow definition
            result: The execution result
            include_task_details: Whether to include detailed task data
            format_type: Output format ("text", "html", or "json")
            
        Returns:
            str: The formatted execution report
        """
        # Collect basic workflow information
        workflow_info = {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "process_type": getattr(workflow, "process_type", "unknown"),
            "tasks_count": len(workflow.get_tasks())
        }
        
        # Collect execution information
        execution_info = {
            "status": result.status.value,
            "successful": result.is_successful,
            "execution_time": result.execution_time,
            "error": str(result.error) if result.error else None
        }
        
        # Collect task execution details
        task_details = []
        task_summary = {
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "pending": 0,
            "total": len(workflow.get_tasks())
        }
        
        for task_id, task in workflow.tasks.items():
            status = task.status.value
            
            # Update summary counts
            if status == TaskStatus.COMPLETED.value:
                task_summary["completed"] += 1
            elif status == TaskStatus.FAILED.value:
                task_summary["failed"] += 1
            elif status == TaskStatus.SKIPPED.value:
                task_summary["skipped"] += 1
            elif status == TaskStatus.PENDING.value:
                task_summary["pending"] += 1
                
            # Calculate task duration
            duration = None
            if task.start_time and task.end_time:
                duration = (task.end_time - task.start_time).total_seconds()
            
            # Collect task detail
            task_detail = {
                "id": task.id,
                "name": task.name,
                "status": status,
                "duration": f"{duration:.2f}s" if duration else None,
                "started": task.start_time.isoformat() if task.start_time else None,
                "completed": task.end_time.isoformat() if task.end_time else None,
                "error": str(task.error) if task.error else None,
            }
            
            # Add result data if requested
            if include_task_details and task_id in result.task_results:
                try:
                    # Try to make result serializable
                    task_data = result.task_results[task_id]
                    if hasattr(task_data, "to_dict"):
                        task_detail["result"] = task_data.to_dict()
                    else:
                        task_detail["result"] = task_data
                except Exception as e:
                    task_detail["result"] = f"<Error serializing result: {str(e)}>"
                    
            task_details.append(task_detail)
        
        # Generate report based on format
        if format_type == "json":
            report = {
                "workflow": workflow_info,
                "execution": execution_info,
                "task_summary": task_summary,
                "tasks": task_details,
                "generated_at": datetime.now().isoformat()
            }
            return json.dumps(report, indent=2, cls=DecimalEncoder)
            
        elif format_type == "html":
            html_output = io.StringIO()
            html_output.write("<html><head><style>")
            html_output.write("body { font-family: Arial, sans-serif; margin: 20px; }")
            html_output.write("table { border-collapse: collapse; width: 100%; }")
            html_output.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            html_output.write("th { background-color: #f2f2f2; }")
            html_output.write(".completed { color: green; }")
            html_output.write(".failed { color: red; }")
            html_output.write(".skipped { color: blue; }")
            html_output.write(".pending { color: gray; }")
            html_output.write("</style></head><body>")
            
            # Workflow info
            html_output.write("<h1>Workflow Execution Report</h1>")
            html_output.write(f"<h2>{workflow_info['name']}</h2>")
            html_output.write(f"<p>{workflow_info['description']}</p>")
            
            # Execution summary
            html_output.write("<h3>Execution Summary</h3>")
            html_output.write("<table>")
            html_output.write("<tr><th>Status</th><td")
            if execution_info["status"] == "COMPLETED":
                html_output.write(" class='completed'")
            elif execution_info["status"] == "FAILED":
                html_output.write(" class='failed'")
            html_output.write(f">{execution_info['status']}</td></tr>")
            if execution_info["execution_time"] is not None:
                html_output.write(f"<tr><th>Execution Time</th><td>{execution_info['execution_time']:.2f} seconds</td></tr>")
            else:
                html_output.write("<tr><th>Execution Time</th><td>N/A</td></tr>")
            if execution_info["error"]:
                html_output.write(f"<tr><th>Error</th><td class='failed'>{execution_info['error']}</td></tr>")
            html_output.write("</table>")
            
            # Task summary
            html_output.write("<h3>Task Summary</h3>")
            html_output.write("<table>")
            html_output.write(f"<tr><th>Total</th><td>{task_summary['total']}</td></tr>")
            html_output.write(f"<tr><th>Completed</th><td class='completed'>{task_summary['completed']}</td></tr>")
            html_output.write(f"<tr><th>Failed</th><td class='failed'>{task_summary['failed']}</td></tr>")
            html_output.write(f"<tr><th>Skipped</th><td class='skipped'>{task_summary['skipped']}</td></tr>")
            html_output.write(f"<tr><th>Pending</th><td class='pending'>{task_summary['pending']}</td></tr>")
            html_output.write("</table>")
            
            # Task details
            if include_task_details:
                html_output.write("<h3>Task Details</h3>")
                html_output.write("<table>")
                html_output.write("<tr><th>Name</th><th>Status</th><th>Duration</th><th>Error</th></tr>")
                
                for task in task_details:
                    html_output.write("<tr>")
                    html_output.write(f"<td>{task['name']}</td>")
                    html_output.write(f"<td class='{task['status'].lower()}'>{task['status']}</td>")
                    html_output.write(f"<td>{task['duration'] if task['duration'] else ''}</td>")
                    html_output.write(f"<td>{task['error'] if task['error'] else ''}</td>")
                    html_output.write("</tr>")
                    
                html_output.write("</table>")
                
            html_output.write("</body></html>")
            return html_output.getvalue()
            
        else:  # text format
            text_output = io.StringIO()
            text_output.write("WORKFLOW EXECUTION REPORT\n")
            text_output.write("======================\n\n")
            text_output.write(f"Workflow: {workflow_info['name']} ({workflow_info['id']})\n")
            text_output.write(f"Description: {workflow_info['description']}\n")
            text_output.write(f"Type: {workflow_info['process_type']}\n\n")
            
            text_output.write("EXECUTION SUMMARY\n")
            text_output.write("-----------------\n")
            text_output.write(f"Status: {execution_info['status']}\n")
            if execution_info["execution_time"] is not None:
                text_output.write(f"Execution Time: {execution_info['execution_time']:.2f} seconds\n")
            else:
                text_output.write("Execution Time: N/A\n")
            if execution_info["error"]:
                text_output.write(f"Error: {execution_info['error']}\n")
            text_output.write("\n")
            
            text_output.write("TASK SUMMARY\n")
            text_output.write("------------\n")
            text_output.write(f"Total Tasks: {task_summary['total']}\n")
            text_output.write(f"Completed: {task_summary['completed']}\n")
            text_output.write(f"Failed: {task_summary['failed']}\n")
            text_output.write(f"Skipped: {task_summary['skipped']}\n")
            text_output.write(f"Pending: {task_summary['pending']}\n\n")
            
            if include_task_details:
                text_output.write("TASK DETAILS\n")
                text_output.write("------------\n")
                
                for task in task_details:
                    text_output.write(f"Task: {task['name']} ({task['id']})\n")
                    text_output.write(f"  Status: {task['status']}\n")
                    if task['duration']:
                        text_output.write(f"  Duration: {task['duration']}\n")
                    if task['error']:
                        text_output.write(f"  Error: {task['error']}\n")
                    text_output.write("\n")
                    
            return text_output.getvalue()

    def display_workflow_execution(
        self, 
        workflow: WorkflowDefinition, 
        result: WorkflowResult, 
        format_type: str = "text"
    ) -> None:
        """
        Display a workflow execution report.
        
        Args:
            workflow: The workflow definition
            result: The execution result
            format_type: Output format
        """
        report = self.generate_execution_report(
            workflow=workflow,
            result=result,
            include_task_details=True,
            format_type=format_type
        )
        
        if format_type == "html" and _HAS_VISUALIZATION:
            display(HTML(report))
        else:
            print(report)

    def export_execution_report(
        self,
        workflow: WorkflowDefinition,
        result: WorkflowResult,
        filename: str,
        format_type: Optional[str] = None
    ) -> str:
        """
        Export a workflow execution report to a file.
        
        Args:
            workflow: The workflow definition
            result: The execution result
            filename: The filename to save to
            format_type: Optional format override, otherwise inferred from filename
            
        Returns:
            str: Path to the saved file
        """
        # Determine format from filename if not specified
        if not format_type:
            if filename.endswith('.html'):
                format_type = 'html'
            elif filename.endswith('.json'):
                format_type = 'json'
            else:
                format_type = 'text'
        
        # Generate report
        report = self.generate_execution_report(
            workflow=workflow,
            result=result,
            include_task_details=True,
            format_type=format_type
        )
        
        # Write report to file
        with open(filename, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Execution report saved to {filename}")
        return filename

class WorkflowInspector:
    """
    Provides real-time inspection and metrics for running workflows.
    """
    
    def __init__(self):
        """Initialize the workflow inspector."""
        self.logger = logging.getLogger("finflow.workflow.inspector")
        self.active_workflows = {}
    
    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """
        Register a workflow for inspection.
        
        Args:
            workflow: The workflow to register
        """
        if workflow.id in self.active_workflows:
            self.logger.warning(f"Workflow {workflow.id} already registered for inspection")
            return
            
        self.active_workflows[workflow.id] = {
            "workflow": workflow,
            "start_time": time.time(),
            "task_timings": {},
            "metrics": {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "tasks_skipped": 0,
                "tasks_running": 0
            }
        }
        
        self.logger.info(f"Registered workflow {workflow.name} ({workflow.id}) for inspection")
    
    def update_workflow_state(self, workflow_id: str, task_id: Optional[str] = None, status: Optional[str] = None) -> None:
        """
        Update the state of a workflow or task.
        
        Args:
            workflow_id: The ID of the workflow
            task_id: Optional task ID to update
            status: Optional status update
        """
        if workflow_id not in self.active_workflows:
            self.logger.warning(f"Cannot update workflow {workflow_id}: not registered")
            return
            
        workflow_data = self.active_workflows[workflow_id]
        
        # Update task specific information
        if task_id:
            if task_id not in workflow_data["task_timings"]:
                workflow_data["task_timings"][task_id] = {
                    "start_time": None,
                    "end_time": None,
                    "status": None
                }
                
            # Update task status and timing
            if status:
                workflow_data["task_timings"][task_id]["status"] = status
                
                if status == TaskStatus.RUNNING.value:
                    workflow_data["task_timings"][task_id]["start_time"] = time.time()
                    workflow_data["metrics"]["tasks_running"] += 1
                    
                elif status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.SKIPPED.value):
                    workflow_data["task_timings"][task_id]["end_time"] = time.time()
                    workflow_data["metrics"]["tasks_running"] -= 1
                    
                    if status == TaskStatus.COMPLETED.value:
                        workflow_data["metrics"]["tasks_completed"] += 1
                    elif status == TaskStatus.FAILED.value:
                        workflow_data["metrics"]["tasks_failed"] += 1
                    elif status == TaskStatus.SKIPPED.value:
                        workflow_data["metrics"]["tasks_skipped"] += 1
    
    def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific workflow.
        
        Args:
            workflow_id: The ID of the workflow
            
        Returns:
            Dict[str, Any]: Metrics for the workflow
        """
        if workflow_id not in self.active_workflows:
            self.logger.warning(f"Cannot get metrics for workflow {workflow_id}: not registered")
            return {}
            
        workflow_data = self.active_workflows[workflow_id]
        workflow_obj = workflow_data["workflow"]
        
        # Calculate elapsed time
        elapsed = time.time() - workflow_data["start_time"]
        
        # Calculate estimated remaining time based on completion rate
        remaining = None
        completion_rate = workflow_data["metrics"]["tasks_completed"] / len(workflow_obj.tasks) if workflow_obj.tasks else 0
        
        if 0 < completion_rate < 1:
            remaining = (elapsed / completion_rate) - elapsed
            
        # Compile metrics
        metrics = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_obj.name,
            "elapsed_time": elapsed,
            "estimated_remaining_time": remaining,
            "progress_percentage": completion_rate * 100,
            "tasks_total": len(workflow_obj.tasks),
            "tasks_completed": workflow_data["metrics"]["tasks_completed"],
            "tasks_failed": workflow_data["metrics"]["tasks_failed"],
            "tasks_skipped": workflow_data["metrics"]["tasks_skipped"],
            "tasks_running": workflow_data["metrics"]["tasks_running"],
            "tasks_pending": len(workflow_obj.tasks) - (
                workflow_data["metrics"]["tasks_completed"] + 
                workflow_data["metrics"]["tasks_failed"] + 
                workflow_data["metrics"]["tasks_skipped"] +
                workflow_data["metrics"]["tasks_running"]
            )
        }
        
        return metrics
    
    def get_task_status(self, workflow_id: str, task_id: str) -> Dict[str, Any]:
        """
        Get detailed status for a specific task.
        
        Args:
            workflow_id: The ID of the workflow
            task_id: The ID of the task
            
        Returns:
            Dict[str, Any]: Status information for the task
        """
        if workflow_id not in self.active_workflows:
            self.logger.warning(f"Cannot get task status for workflow {workflow_id}: not registered")
            return {}
            
        workflow_data = self.active_workflows[workflow_id]
        
        if task_id not in workflow_data["task_timings"]:
            return {
                "task_id": task_id,
                "status": "unknown",
                "elapsed_time": None,
                "message": "Task not tracked by inspector"
            }
            
        task_data = workflow_data["task_timings"][task_id]
        workflow_obj = workflow_data["workflow"]
        task = workflow_obj.get_task(task_id)
        
        if not task:
            return {
                "task_id": task_id,
                "status": task_data.get("status", "unknown"),
                "elapsed_time": None,
                "message": "Task not found in workflow"
            }
            
        # Calculate elapsed time
        elapsed = None
        if task_data["start_time"]:
            if task_data["end_time"]:
                elapsed = task_data["end_time"] - task_data["start_time"]
            else:
                elapsed = time.time() - task_data["start_time"]
                
        return {
            "task_id": task_id,
            "task_name": task.name,
            "status": task_data.get("status", "unknown"),
            "elapsed_time": elapsed,
            "dependencies": task.dependencies,
            "dependents": [t.id for t in workflow_obj.get_dependent_tasks(task_id)]
        }
        
    def print_workflow_status(self, workflow_id: str) -> None:
        """
        Print the current status of a workflow to the console.
        
        Args:
            workflow_id: The ID of the workflow to print
        """
        metrics = self.get_workflow_metrics(workflow_id)
        if not metrics:
            print(f"Workflow {workflow_id} not found or not registered")
            return
            
        print(f"\nWorkflow Status: {metrics['workflow_name']} ({workflow_id})")
        print(f"Progress: {metrics['progress_percentage']:.1f}% complete")
        print(f"Elapsed Time: {metrics['elapsed_time']:.2f} seconds")
        
        if metrics['estimated_remaining_time']:
            print(f"Estimated Remaining: {metrics['estimated_remaining_time']:.2f} seconds")
            
        print("\nTasks:")
        print(f"  Total: {metrics['tasks_total']}")
        print(f"  Completed: {metrics['tasks_completed']}")
        print(f"  Running: {metrics['tasks_running']}")
        print(f"  Failed: {metrics['tasks_failed']}")
        print(f"  Skipped: {metrics['tasks_skipped']}")
        print(f"  Pending: {metrics['tasks_pending']}")
        print()


# Function to create a visual representation of the workflow execution
def create_workflow_visualization(
    workflow: WorkflowDefinition,
    execution_result: Optional[WorkflowResult] = None,
    output_dir: str = "workflow_visualizations"
) -> Tuple[str, str]:
    """
    Create visualizations for a workflow definition and execution.
    
    Args:
        workflow: The workflow definition
        execution_result: Optional execution result
        output_dir: Directory to save visualization files
        
    Returns:
        Tuple[str, str]: Paths to the generated files (graph, report)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a workflow monitor
    monitor = WorkflowMonitor()
    
    # Create a visualization of the workflow graph
    graph_path = os.path.join(output_dir, f"{workflow.id}_graph.png")
    monitor.visualize_workflow(
        workflow=workflow,
        execution_result=execution_result,
        filename=graph_path,
        show_plot=False
    )
    
    # Create an execution report if result is provided
    report_path = None
    if execution_result:
        report_path = os.path.join(output_dir, f"{workflow.id}_report.html")
        monitor.export_execution_report(
            workflow=workflow,
            result=execution_result,
            filename=report_path,
            format_type="html"
        )
    
    return graph_path, report_path
