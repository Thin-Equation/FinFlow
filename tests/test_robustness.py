"""
Test suite for FinFlow robustness and optimization features.

This module contains tests for:
1. Optimized batch processing
2. Recovery mechanisms
3. Health monitoring
4. Error handling improvements
5. Performance optimizations
"""

import unittest
import os
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock

# Import FinFlow components to test
from batch.optimized_batch import OptimizedBatchProcessor, DocumentBatch
from utils.recovery_manager import RecoveryManager, RecoveryStrategy, RecoveryState, RecoveryCheckpoint
from utils.health_check import HealthCheckManager, HealthStatus
from workflow.optimized_runner import OptimizedWorkflowRunner
from agents.enhanced_document_processor import EnhancedDocumentProcessorAgent


class TestRecoveryMechanisms(unittest.TestCase):
    """Test the recovery mechanisms in the system."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create test directory
        self.test_dir = tempfile.mkdtemp()
        self.recovery_dir = os.path.join(self.test_dir, "recovery")
        os.makedirs(self.recovery_dir, exist_ok=True)
        
        # Initialize recovery manager
        self.recovery_manager = RecoveryManager(recovery_dir=self.recovery_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory
        shutil.rmtree(self.test_dir)
    
    def test_checkpoint_creation(self):
        """Test creating and saving checkpoints."""
        # Create a test workflow checkpoint
        workflow_id = "test_workflow_123"
        state = {"step": "extract", "document_id": "doc123", "progress": 0.5}
        
        checkpoint = RecoveryCheckpoint(
            workflow_id=workflow_id,
            state=state,
            metadata={"document_type": "invoice"}
        )
        
        # Save the checkpoint
        self.recovery_manager.save_checkpoint(checkpoint)
        
        # Check that the checkpoint file was created
        checkpoint_file = os.path.join(self.recovery_dir, f"checkpoint_{workflow_id}.json")
        self.assertTrue(os.path.exists(checkpoint_file))
        
        # Load the checkpoint file and verify its contents
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data["workflow_id"], workflow_id)
        self.assertEqual(data["state"]["document_id"], "doc123")
        self.assertEqual(data["metadata"]["document_type"], "invoice")
    
    def test_recovery_execution(self):
        """Test executing a recovery plan."""
        # Create a mock recovery function
        mock_recovery_function = Mock(return_value={"status": "recovered"})
        
        # Create a recovery plan
        workflow_id = "test_workflow_456"
        recovery_plan = self.recovery_manager.create_recovery_plan(
            workflow_id=workflow_id,
            strategy=RecoveryStrategy.CHECKPOINT,
            recovery_function=mock_recovery_function,
            max_attempts=3
        )
        
        # Execute the recovery plan
        result = self.recovery_manager.execute_recovery_plan(recovery_plan)
        
        # Verify the recovery function was called
        mock_recovery_function.assert_called_once()
        
        # Check the recovery plan was updated
        self.assertEqual(recovery_plan.state, RecoveryState.SUCCEEDED)
        self.assertEqual(result["status"], "recovered")


class TestOptimizedBatchProcessing(unittest.TestCase):
    """Test the optimized batch processing functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create test directories
        self.test_dir = tempfile.mkdtemp()
        self.batch_dir = os.path.join(self.test_dir, "documents")
        self.output_dir = os.path.join(self.test_dir, "results")
        self.recovery_dir = os.path.join(self.test_dir, "recovery")
        
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.recovery_dir, exist_ok=True)
        
        # Create some test files
        self._create_test_files()
        
        # Initialize batch processor
        self.batch_processor = OptimizedBatchProcessor(
            min_workers=1,
            max_workers=2,
            recovery_dir=self.recovery_dir
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory
        shutil.rmtree(self.test_dir)
    
    def _create_test_files(self):
        """Create test document files."""
        # Create a few test PDF files (empty files for testing)
        for i in range(3):
            file_path = os.path.join(self.batch_dir, f"test_doc_{i}.pdf")
            with open(file_path, 'w') as f:
                f.write(f"Test document {i}")
    
    def test_document_batch(self):
        """Test document batch management."""
        batch = DocumentBatch(batch_id="test_batch")
        
        # Add documents to the batch
        batch.add_document("/path/to/doc1.pdf", priority=1)
        batch.add_document("/path/to/doc2.pdf", priority=2)
        batch.add_document("/path/to/doc3.pdf", priority=1)
        
        # Create chunks
        batch.create_chunks()
        
        # Verify documents were added
        self.assertEqual(len(batch.documents), 3)
        self.assertEqual(len(batch.document_paths), 3)
        
        # Verify chunks were created (max chunk size is 10 by default)
        self.assertEqual(len(batch.chunks), 1)  # All docs in one chunk
        
        # Mark documents with different statuses
        batch.mark_document_in_progress("/path/to/doc1.pdf")
        batch.mark_document_complete("/path/to/doc1.pdf", {"result": "success"})
        batch.mark_document_failed("/path/to/doc3.pdf", "processing error")
        
        # Check document counts and progress
        progress = batch.get_progress()
        self.assertEqual(progress["completed"], 1)
        self.assertEqual(progress["failed"], 1)
        self.assertEqual(progress["queued"], 1)  # Doc2 is still queued
        self.assertAlmostEqual(progress["progress_percent"], 66.67, delta=0.1)
    
    @patch("batch.optimized_batch.os.path.getsize")
    @patch("batch.optimized_batch.concurrent.futures.ThreadPoolExecutor")
    def test_batch_processing(self, mock_executor, mock_getsize):
        """Test the batch processing functionality."""
        # Mock getsize to avoid errors
        mock_getsize.return_value = 1024
        
        # Create a mock executor
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        # Create mock agents
        mock_agents = {
            "master_orchestrator": MagicMock()
        }
        mock_agents["master_orchestrator"].process_document.return_value = {
            "document_id": "test_doc",
            "status": "success",
            "extracted_data": {"key": "value"}
        }
        
        # Process the batch
        result = self.batch_processor.process_batch(
            agents=mock_agents,
            config={},
            batch_dir=self.batch_dir,
            output_dir=self.output_dir,
            workflow_type="test"
        )
        
        # Verify the result
        self.assertIsNotNone(result["batch_id"])
        self.assertEqual(result["total"], 3)  # 3 test documents
        
        # Check if output files were created
        summary_files = [f for f in os.listdir(self.output_dir) if f.startswith("batch_summary")]
        self.assertTrue(len(summary_files) > 0)


class TestHealthMonitoring(unittest.TestCase):
    """Test the health monitoring functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize health check manager
        self.health_manager = HealthCheckManager.get_instance({"check_interval_seconds": 1})
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the health check manager
        self.health_manager.stop()
    
    def test_health_check_registration(self):
        """Test registering and running health checks."""
        # Mock check function
        mock_check = Mock(return_value={
            "status": HealthStatus.HEALTHY,
            "details": "Everything is working"
        })
        
        # Register the check
        self.health_manager.register_check(
            name="test_service",
            check_function=mock_check,
            critical=True,
            interval_seconds=1
        )
        
        # Verify check was registered
        self.assertIn("test_service", self.health_manager.checks)
        
        # Run the check manually
        check_result = self.health_manager.run_check("test_service")
        
        # Verify the check ran and returned expected results
        mock_check.assert_called_once()
        self.assertEqual(check_result["status"], HealthStatus.HEALTHY)
        self.assertEqual(check_result["critical"], True)
        self.assertIn("details", check_result)
    
    def test_health_report(self):
        """Test generating a health report."""
        # Register a healthy and unhealthy service
        self.health_manager.register_check(
            name="healthy_service",
            check_function=lambda: {"status": HealthStatus.HEALTHY, "details": "Working"},
            critical=False
        )
        
        self.health_manager.register_check(
            name="unhealthy_service",
            check_function=lambda: {"status": HealthStatus.UNHEALTHY, "details": "Failed"},
            critical=True
        )
        
        # Run all checks
        self.health_manager.run_all_checks()
        
        # Get health report
        report = self.health_manager.get_full_report()
        
        # Verify report structure
        self.assertIn("status", report)
        self.assertIn("checks", report)
        self.assertEqual(len(report["checks"]), 2)
        
        # Overall status should be UNHEALTHY since a critical service is down
        self.assertEqual(report["status"], HealthStatus.UNHEALTHY)


class TestWorkflowOptimizations(unittest.TestCase):
    """Test the optimized workflow functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test directory
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize workflow runner with mocked components
        self.workflow_runner = OptimizedWorkflowRunner(
            config={
                "parallel_execution": True,
                "max_parallel_tasks": 2,
                "timeout_seconds": 10
            }
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory
        shutil.rmtree(self.test_dir)
    
    @patch("workflow.optimized_runner.concurrent.futures.ThreadPoolExecutor")
    def test_parallel_execution(self, mock_executor):
        """Test parallel workflow execution."""
        # Create mock executor
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        # Define mock task functions
        task1 = Mock(return_value={"result": "task1"})
        task2 = Mock(return_value={"result": "task2"})
        task3 = Mock(return_value={"result": "task3"})
        
        # Define workflow
        workflow = {
            "id": "test_workflow",
            "name": "Test Workflow",
            "tasks": [
                {"id": "task1", "function": task1, "dependencies": []},
                {"id": "task2", "function": task2, "dependencies": ["task1"]},
                {"id": "task3", "function": task3, "dependencies": ["task1"]}
            ]
        }
        
        # Run the workflow
        context = {"document_id": "test123"}
        result = self.workflow_runner.run_workflow(workflow, context)
        
        # Verify that all tasks were executed
        task1.assert_called_once_with(context)
        task2.assert_called_once()
        task3.assert_called_once()
        
        # Verify that the workflow result contains all task results
        self.assertIn("task1", result)
        self.assertIn("task2", result)
        self.assertIn("task3", result)


class TestEnhancedDocumentProcessor(unittest.TestCase):
    """Test the enhanced document processor functionality."""
    
    @patch("agents.enhanced_document_processor.super")
    def test_document_caching(self, mock_super):
        """Test document caching functionality."""
        # Create processor with mocked superclass
        processor = EnhancedDocumentProcessorAgent(config={"document_cache_enabled": True})
        
        # Mock the process_document method to return different results
        processor.process_document_internal = Mock()
        processor.process_document_internal.side_effect = [
            {"status": "success", "result": "first_call"},
            {"status": "success", "result": "second_call"}
        ]
        
        # Process the same document twice
        doc_path = "/path/to/document.pdf"
        context1 = {"document_path": doc_path, "workflow_type": "test"}
        context2 = {"document_path": doc_path, "workflow_type": "test"}
        
        result1 = processor.process_document(context1)
        result2 = processor.process_document(context2)
        
        # Verify that the document was only processed once (cached result used second time)
        self.assertEqual(processor.process_document_internal.call_count, 1)
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
