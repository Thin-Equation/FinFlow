"""
End-to-end test framework for FinFlow.

This module provides a comprehensive end-to-end testing framework
for the FinFlow system, testing the integration of all components.
"""

import os
import sys
import time
import unittest
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import load_config
from utils.logging_config import configure_logging
from initialize_agents import initialize_system
from workflow.workflow_runner import run_workflow
from batch.batch_processor import process_batch


class EndToEndTest(unittest.TestCase):
    """Base class for end-to-end tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Configure logging
        configure_logging(log_level="INFO")
        
        # Set environment for testing
        os.environ.setdefault('FINFLOW_ENV', 'development')
        
        # Load configuration
        cls.config = load_config()
        
        # Initialize agent system
        cls.agents = initialize_system(cls.config)
        
        # Set up temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp(prefix="finflow_test_")
        
        # Set up sample data directory
        cls.sample_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
        
    @classmethod
    def tearDownClass(cls):
        """Tear down the test environment."""
        # Clean up temporary files
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up each test."""
        self.start_time = time.time()
    
    def tearDown(self):
        """Tear down each test."""
        elapsed = time.time() - self.start_time
        print(f"\nTest executed in {elapsed:.2f} seconds")


class WorkflowEndToEndTest(EndToEndTest):
    """End-to-end tests for workflow execution."""
    
    def test_standard_workflow(self):
        """Test the standard workflow."""
        # Skip if no sample data
        sample_path = os.path.join(self.sample_data_dir, "invoices", "sample_invoice_1.pdf")
        if not os.path.exists(sample_path):
            self.skipTest("Sample invoice not found")
        
        # Run workflow
        result = run_workflow(
            workflow_name="standard",
            agents=self.agents,
            config=self.config,
            document_path=sample_path
        )
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertIn("workflow_id", result)
        self.assertIn("status", result)
        self.assertEqual(result.get("workflow_name"), "standard")
        self.assertEqual(result.get("status"), "completed")
        self.assertGreater(result.get("processing_time", 0), 0)
        
        # Verify each step
        self.assertIn("results", result)
        self.assertIn("document_extraction", result["results"])
        self.assertIn("validation", result["results"])
        self.assertIn("storage", result["results"])
    
    def test_invoice_workflow(self):
        """Test the invoice workflow."""
        # Skip if no sample data
        sample_path = os.path.join(self.sample_data_dir, "invoices", "sample_invoice_1.pdf")
        if not os.path.exists(sample_path):
            self.skipTest("Sample invoice not found")
        
        # Run workflow
        result = run_workflow(
            workflow_name="invoice",
            agents=self.agents,
            config=self.config,
            document_path=sample_path
        )
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.get("workflow_name"), "invoice")
        self.assertEqual(result.get("status"), "completed")
        
        # Verify each step
        self.assertIn("results", result)
        self.assertIn("document_extraction", result["results"])
        self.assertIn("rules_check", result["results"])
        self.assertIn("validation", result["results"])
        self.assertIn("storage", result["results"])
        self.assertIn("analytics", result["results"])


class BatchEndToEndTest(EndToEndTest):
    """End-to-end tests for batch processing."""
    
    def test_batch_processing(self):
        """Test batch processing of documents."""
        # Skip if no sample data
        if not os.path.exists(os.path.join(self.sample_data_dir, "invoices")):
            self.skipTest("Sample invoices directory not found")
        
        # Copy sample files to temp directory
        batch_dir = os.path.join(self.temp_dir, "batch_test")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Copy sample invoices for testing
        sample_files = []
        for i in range(1, 4):  # Try to copy up to 3 sample invoices
            sample_path = os.path.join(self.sample_data_dir, "invoices", f"sample_invoice_{i}.pdf")
            if os.path.exists(sample_path):
                dest_path = os.path.join(batch_dir, f"invoice_{i}.pdf")
                shutil.copy(sample_path, dest_path)
                sample_files.append(dest_path)
        
        if not sample_files:
            self.skipTest("No sample files found for batch testing")
        
        # Run batch processing
        result = process_batch(
            agents=self.agents,
            config=self.config,
            batch_dir=batch_dir,
            max_workers=2
        )
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result["total"], len(sample_files))
        self.assertEqual(result["success"] + result["failed"], len(sample_files))
        
        # Check for output files
        output_dir = os.path.join(batch_dir, "results")
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "batch_summary.json")))


class AgentIntegrationTest(EndToEndTest):
    """End-to-end tests for agent integration."""
    
    def test_master_orchestrator_integration(self):
        """Test master orchestrator integration with other agents."""
        # Get master orchestrator
        master_orchestrator = self.agents["master_orchestrator"]
        self.assertIsNotNone(master_orchestrator)
        
        # Verify it has all the required worker agents
        required_agents = ["document_processor", "rule_retrieval", "validation_agent", "storage_agent", "analytics_agent"]
        
        for agent_name in required_agents:
            self.assertTrue(hasattr(master_orchestrator, f"{agent_name}"), f"Master orchestrator missing {agent_name}")
    
    def test_document_processor_integration(self):
        """Test document processor integration."""
        # Skip if no sample data
        sample_path = os.path.join(self.sample_data_dir, "invoices", "sample_invoice_1.pdf")
        if not os.path.exists(sample_path):
            self.skipTest("Sample invoice not found")
        
        # Get document processor
        document_processor = self.agents["document_processor"]
        self.assertIsNotNone(document_processor)
        
        # Process a document
        context = {
            "document_path": sample_path,
            "user_id": "test",
            "session_id": "test_session",
        }
        
        result = document_processor.extract_document(context)
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertIn("document_id", result)
        self.assertIn("content", result)
        self.assertIn("metadata", result)
    
    def test_validation_agent_integration(self):
        """Test validation agent integration."""
        # Skip if no sample data
        sample_path = os.path.join(self.sample_data_dir, "invoices", "sample_invoice_1.pdf")
        if not os.path.exists(sample_path):
            self.skipTest("Sample invoice not found")
        
        # Get document processor and validation agent
        document_processor = self.agents["document_processor"]
        validation_agent = self.agents["validation_agent"]
        self.assertIsNotNone(document_processor)
        self.assertIsNotNone(validation_agent)
        
        # Process a document
        context = {
            "document_path": sample_path,
            "user_id": "test",
            "session_id": "test_session",
        }
        
        extracted = document_processor.extract_document(context)
        
        # Validate the document
        validation_result = validation_agent.validate_document(extracted)
        
        # Verify result
        self.assertIsNotNone(validation_result)
        self.assertIn("valid", validation_result)
        self.assertIn("validation_checks", validation_result)


def run_tests():
    """Run all end-to-end tests."""
    configure_logging()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(WorkflowEndToEndTest))
    suite.addTest(unittest.makeSuite(BatchEndToEndTest))
    suite.addTest(unittest.makeSuite(AgentIntegrationTest))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    result = run_tests()
    
    # Exit with non-zero code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)
