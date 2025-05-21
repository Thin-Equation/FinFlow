"""
Test cases for the workflow system.

This module contains test cases for workflow definitions, sequential and parallel agents,
and conditional branching in the workflow system.
"""

import os
import sys
import unittest
import time
import logging
from decimal import Decimal
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.workflow_definitions import (
    Workflow,
    WorkflowTask,
    WorkflowDefinition,
    WorkflowExecutionContext,
    WorkflowStatus,
    TaskStatus
)
from workflow.sequential_agent import SequentialAgent
from workflow.parallel_agent import ParallelAgent
from workflow.conditional import (
    ConditionalBranching,
    Branch,
    BranchSelectionStrategy
)
from examples.workflow_examples import (
    create_invoice_processing_workflow,
    create_financial_reporting_workflow
)

class WorkflowTests(unittest.TestCase):
    """Test cases for the workflow system."""

    def setUp(self):
        """Set up test cases."""
        # Configure logging for tests
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("workflow_tests")
        
        # Create agents for testing
        self.sequential_agent = SequentialAgent(name="TestSequentialAgent")
        self.parallel_agent = ParallelAgent(name="TestParallelAgent", max_workers=5)
        
        # Base context for all tests
        self.context = {"test_start_time": time.time()}

    def test_basic_workflow_creation(self):
        """Test creating a basic workflow."""
        workflow = Workflow.create(
            name="Test Workflow",
            description="A test workflow",
            process_type="test_workflow"
        )
        
        # Add a simple task
        task = WorkflowTask(
            id="test_task",
            name="Test Task",
            description="A simple test task",
            execute=lambda ctx: {"status": "success"}
        )
        workflow.add_task(task)
        
        # Verify workflow 
        self.assertEqual(workflow.name, "Test Workflow")
        self.assertEqual(workflow.process_type, "test_workflow")
        self.assertEqual(len(workflow.get_tasks()), 1)
        self.assertEqual(workflow.get_task("test_task").name, "Test Task")

    def test_sequential_workflow_execution(self):
        """Test sequential execution of a workflow."""
        workflow = self._create_test_sequence_workflow()
        
        # Execute workflow
        result = self.sequential_agent.run_workflow(
            workflow_definition=workflow,
            context=self.context,
            parameters={"test_param": 42}
        )
        
        # Verify results
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result.is_successful)
        self.assertIsNone(result.error)
        
        # Check task results
        self.assertIn("task1", result.task_results)
        self.assertIn("task2", result.task_results)
        self.assertIn("task3", result.task_results)
        
        # Verify execution order via timestamps
        task1_time = workflow.get_task("task1").end_time
        task2_time = workflow.get_task("task2").end_time
        task3_time = workflow.get_task("task3").end_time
        
        self.assertLess(task1_time, task2_time)
        self.assertLess(task2_time, task3_time)

    def test_parallel_workflow_execution(self):
        """Test parallel execution of a workflow."""
        workflow = self._create_test_parallel_workflow()
        
        # Execute workflow
        result = self.parallel_agent.run_workflow(
            workflow_definition=workflow,
            context=self.context,
            parameters={"test_param": 42}
        )
        
        # Verify results
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result.is_successful)
        self.assertIsNone(result.error)
        
        # Check task results
        self.assertIn("task1", result.task_results)
        self.assertIn("parallel1", result.task_results)
        self.assertIn("parallel2", result.task_results)
        self.assertIn("task4", result.task_results)
        
        # Verify execution constraints
        task1_time = workflow.get_task("task1").end_time
        parallel1_time = workflow.get_task("parallel1").end_time
        parallel2_time = workflow.get_task("parallel2").end_time
        task4_time = workflow.get_task("task4").end_time
        
        # task1 should complete before both parallel tasks
        self.assertLess(task1_time, parallel1_time)
        self.assertLess(task1_time, parallel2_time)
        
        # task4 should complete after both parallel tasks
        self.assertGreater(task4_time, parallel1_time)
        self.assertGreater(task4_time, parallel2_time)

    def test_conditional_branching(self):
        """Test conditional branching in workflows."""
        # Test with different parameter values to test different branches
        for test_value, expected_branch in [
            (5000, "medium_value_branch"),
            (500, "low_value_branch"),
            (15000, "high_value_branch")
        ]:
            with self.subTest(test_value=test_value, expected_branch=expected_branch):
                workflow = self._create_test_conditional_workflow()
                
                # Execute workflow with specified amount
                result = self.sequential_agent.run_workflow(
                    workflow_definition=workflow,
                    context=self.context,
                    parameters={"invoice_amount": test_value}
                )
                
                # Verify results
                self.assertEqual(result.status, WorkflowStatus.COMPLETED)
                self.assertTrue(result.is_successful)
                
                # Check branch execution - only one branch should be taken
                branch_results = result.task_results.get("branch_router", {}).get("branch_results", {})
                self.logger.info(f"Branch results for amount {test_value}: {branch_results}")
                
                # Check which tasks were executed based on expected branch
                if expected_branch == "high_value_branch":
                    self.assertTrue(workflow.get_task("manager_approval").is_completed())
                    self.assertTrue(workflow.get_task("director_approval").is_completed())
                    self.assertFalse(workflow.get_task("automatic_approval").is_completed())
                elif expected_branch == "medium_value_branch":
                    self.assertTrue(workflow.get_task("manager_approval").is_completed())
                    self.assertFalse(workflow.get_task("director_approval").is_completed())
                    self.assertFalse(workflow.get_task("automatic_approval").is_completed())
                else:  # low value branch
                    self.assertFalse(workflow.get_task("manager_approval").is_completed())
                    self.assertFalse(workflow.get_task("director_approval").is_completed())
                    self.assertTrue(workflow.get_task("automatic_approval").is_completed())

    def test_example_invoice_workflow(self):
        """Test the example invoice processing workflow."""
        workflow = create_invoice_processing_workflow()
        
        # Execute workflow
        result = self.sequential_agent.run_workflow(
            workflow_definition=workflow,
            context=self.context,
            parameters={"document_id": "TEST-DOC-123"}
        )
        
        # Verify workflow execution
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result.is_successful)
        
        # Verify key tasks completed
        self.assertIn("extract_invoice_data", result.task_results)
        self.assertIn("validate_invoice", result.task_results)
        self.assertIn("calculate_metrics", result.task_results)
        self.assertIn("process_payment", result.task_results)
        self.assertIn("record_transaction", result.task_results)
        
        # Check that one of the approval paths was taken
        metrics = result.task_results.get("calculate_metrics", {})
        amount = metrics.get("amount") if isinstance(metrics, dict) else None
        
        if amount and amount >= Decimal("10000"):
            self.assertIn("manager_approval", result.task_results)
            self.assertIn("director_approval", result.task_results)
        elif amount and amount >= Decimal("1000"):
            self.assertIn("manager_approval_medium", result.task_results)
        else:
            self.assertIn("automatic_approval", result.task_results)

    def test_example_reporting_workflow(self):
        """Test the example financial reporting workflow."""
        workflow = create_financial_reporting_workflow()
        
        # Execute workflow with parallel agent
        result = self.parallel_agent.run_workflow(
            workflow_definition=workflow,
            context=self.context,
            parameters={"report_period": "Test Period", "distribution_method": "email"}
        )
        
        # Verify workflow execution
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertTrue(result.is_successful)
        
        # Verify key tasks completed
        self.assertIn("extract_gl_data", result.task_results)
        self.assertIn("extract_ap_data", result.task_results)
        self.assertIn("extract_ar_data", result.task_results)
        self.assertIn("transform_data", result.task_results)
        
        # Verify report sections were generated
        self.assertIn("generate_income_statement", result.task_results)
        self.assertIn("generate_balance_sheet", result.task_results)
        self.assertIn("generate_cash_flow", result.task_results)
        self.assertIn("consolidate_reports", result.task_results)
        
        # Verify distribution was handled
        self.assertIn("email_report", result.task_results)
        self.assertNotIn("upload_to_portal", result.task_results)

    def _create_test_sequence_workflow(self) -> WorkflowDefinition:
        """Create a simple sequential test workflow."""
        workflow = Workflow.create(
            name="Test Sequential Workflow",
            description="A test sequential workflow",
            process_type="test"
        )
        
        # Define tasks
        task1 = WorkflowTask(
            id="task1",
            name="Task 1",
            description="First task",
            execute=lambda ctx: {"step": 1, "result": "Task 1 completed"}
        )
        
        task2 = WorkflowTask(
            id="task2",
            name="Task 2",
            description="Second task",
            execute=lambda ctx: {"step": 2, "result": "Task 2 completed"},
            dependencies=[task1.id]
        )
        
        task3 = WorkflowTask(
            id="task3",
            name="Task 3",
            description="Third task",
            execute=lambda ctx: {"step": 3, "result": "Task 3 completed"},
            dependencies=[task2.id]
        )
        
        # Add tasks to workflow
        workflow.add_task(task1)
        workflow.add_task(task2)
        workflow.add_task(task3)
        
        return workflow
        
    def _create_test_parallel_workflow(self) -> WorkflowDefinition:
        """Create a test workflow with parallel tasks."""
        workflow = Workflow.create(
            name="Test Parallel Workflow",
            description="A test workflow with parallel tasks",
            process_type="test"
        )
        
        # Define tasks
        task1 = WorkflowTask(
            id="task1",
            name="Task 1",
            description="First task",
            execute=lambda ctx: {"step": 1, "result": "Task 1 completed"}
        )
        
        parallel1 = WorkflowTask(
            id="parallel1",
            name="Parallel Task 1",
            description="Parallel task 1",
            execute=lambda ctx: {"step": 2.1, "result": "Parallel 1 completed"},
            dependencies=[task1.id]
        )
        
        parallel2 = WorkflowTask(
            id="parallel2",
            name="Parallel Task 2",
            description="Parallel task 2",
            execute=lambda ctx: {"step": 2.2, "result": "Parallel 2 completed"},
            dependencies=[task1.id]
        )
        
        task4 = WorkflowTask(
            id="task4",
            name="Task 4",
            description="Final task",
            execute=lambda ctx: {"step": 3, "result": "Final task completed"},
            dependencies=[parallel1.id, parallel2.id]
        )
        
        # Add tasks to workflow
        workflow.add_task(task1)
        workflow.add_task(parallel1)
        workflow.add_task(parallel2)
        workflow.add_task(task4)
        
        return workflow

    def _create_test_conditional_workflow(self) -> WorkflowDefinition:
        """Create a test workflow with conditional branching."""
        workflow = Workflow.create(
            name="Test Conditional Workflow",
            description="A test workflow with conditional branching",
            process_type="test"
        )
        
        # Initial task - extract invoice amount
        setup_task = WorkflowTask(
            id="setup",
            name="Setup Task",
            description="Setup test data",
            execute=lambda ctx: {"amount": Decimal(str(ctx.parameters.get("invoice_amount", 5000)))}
        )
        workflow.add_task(setup_task)
        
        # Branch router task
        branch_router = WorkflowTask(
            id="branch_router",
            name="Branch Router",
            description="Route based on amount",
            execute=lambda ctx: {"branch_results": {}, "amount": ctx.get_result("setup")["amount"]},
            dependencies=[setup_task.id]
        )
        workflow.add_task(branch_router)
        
        # High value branch
        high_value_tasks = [
            WorkflowTask(
                id="manager_approval",
                name="Manager Approval",
                description="Get manager approval",
                execute=lambda ctx: {"status": "approved", "approver": "Manager"}
            ),
            WorkflowTask(
                id="director_approval",
                name="Director Approval",
                description="Get director approval",
                execute=lambda ctx: {"status": "approved", "approver": "Director"},
                dependencies=["manager_approval"]
            )
        ]
        
        high_value_branch = Branch(
            name="high_value_branch",
            condition=lambda ctx: ctx.get_result("branch_router")["amount"] >= Decimal("10000"),
            tasks=high_value_tasks,
            priority=3
        )
        
        # Medium value branch
        medium_value_tasks = [
            WorkflowTask(
                id="manager_approval",
                name="Manager Approval",
                description="Get manager approval",
                execute=lambda ctx: {"status": "approved", "approver": "Manager"}
            )
        ]
        
        medium_value_branch = Branch(
            name="medium_value_branch",
            condition=lambda ctx: (Decimal("1000") <= ctx.get_result("branch_router")["amount"] < Decimal("10000")),
            tasks=medium_value_tasks,
            priority=2
        )
        
        # Low value branch
        low_value_tasks = [
            WorkflowTask(
                id="automatic_approval",
                name="Automatic Approval",
                description="Automatic approval",
                execute=lambda ctx: {"status": "approved", "approver": "System"}
            )
        ]
        
        low_value_branch = Branch(
            name="low_value_branch",
            condition=lambda ctx: ctx.get_result("branch_router")["amount"] < Decimal("1000"),
            tasks=low_value_tasks,
            priority=1
        )
        
        # Final task
        final_task = WorkflowTask(
            id="process_payment",
            name="Process Payment",
            description="Process payment",
            execute=lambda ctx: {"status": "payment_processed"}
            # Dependencies will be set by conditional branching
        )
        workflow.add_task(final_task)
        
        # Add conditional branching
        ConditionalBranching.add_switch(
            workflow=workflow,
            switch_name="approval_switch",
            branches=[high_value_branch, medium_value_branch, low_value_branch],
            parent_task_id=branch_router.id,
            exit_task_id=final_task.id,
            strategy=BranchSelectionStrategy.PRIORITY
        )
        
        return workflow

if __name__ == "__main__":
    unittest.main()
