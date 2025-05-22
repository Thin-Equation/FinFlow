# FinFlow Optimization and Robustness Guide

This guide documents the optimization and robustness features added to the FinFlow application. These improvements enhance performance, reliability, and maintainability of the system.

## Table of Contents

1. [Overview](#overview)
2. [Optimization Features](#optimization-features)
   - [Parallel Execution](#parallel-execution)
   - [Document Caching](#document-caching)
   - [Resource-Aware Processing](#resource-aware-processing)
   - [Batch Processing](#batch-processing)
3. [Robustness Features](#robustness-features)
   - [Error Handling](#error-handling)
   - [Recovery Mechanisms](#recovery-mechanisms)
   - [Health Monitoring](#health-monitoring)
   - [Metrics Collection](#metrics-collection)
4. [Command-Line Usage](#command-line-usage)
5. [API Endpoints](#api-endpoints)
6. [Configuration](#configuration)

## Overview

The FinFlow optimization and robustness improvements focus on enhancing several key areas:

1. **Performance**: Improved document processing speed and workflow execution through parallelization and caching
2. **Reliability**: Better error handling and recovery mechanisms to ensure operations continue despite failures
3. **Observability**: Comprehensive metrics collection and health monitoring to provide visibility into system operation
4. **Scalability**: Resource-aware processing that adapts to system load

## Optimization Features

### Parallel Execution

The system now supports parallel execution of workflows and document processing tasks:

- **Workflow Parallelization**: Tasks within workflows can run in parallel when their dependencies allow
- **Document Batch Processing**: Multiple documents can be processed simultaneously
- **Adaptive Worker Pool**: The number of workers adjusts based on system resource availability

```python
# Example: Running a workflow with parallel execution
from workflow.optimized_runner import OptimizedWorkflowRunner

runner = OptimizedWorkflowRunner(
    config={
        "parallel_execution": True,
        "max_parallel_tasks": 4,
        "timeout_seconds": 30
    }
)

result = runner.run_workflow(workflow_definition, context)
```

### Document Caching

To avoid redundant processing, the enhanced document processor implements caching:

- **Result Caching**: Results of document processing are cached based on document path and type
- **Time-to-Live (TTL)**: Cache entries expire after a configurable period
- **Cache Control**: API to manually invalidate or clear the cache

```python
# Example: Processing with caching enabled
from agents.enhanced_document_processor import EnhancedDocumentProcessorAgent

processor = EnhancedDocumentProcessorAgent(
    config={
        "document_cache_enabled": True,
        "cache_ttl_seconds": 3600,
        "cache_size": 100
    }
)

result = processor.process_document(context)
```

### Resource-Aware Processing

The system monitors resource usage and adjusts its behavior accordingly:

- **CPU and Memory Monitoring**: Tracks system resource utilization during processing
- **Adaptive Processing**: Adjusts number of worker threads based on available resources
- **Resource Limits**: Configurable thresholds to prevent system overload

```python
# Example: Resource-aware batch processing
from batch.optimized_batch import OptimizedBatchProcessor

processor = OptimizedBatchProcessor(
    min_workers=2,
    max_workers=8
)

result = processor.process_batch(
    agents=agents,
    config=config,
    batch_dir="/path/to/documents",
    adaptive_workers=True  # Enable resource-based worker adjustment
)
```

### Batch Processing

The optimized batch processor improves document processing at scale:

- **Document Prioritization**: Process high-priority documents first
- **Chunking**: Process documents in optimized chunks for better performance
- **Progress Tracking**: Real-time monitoring of batch progress
- **Checkpoint Creation**: Regular checkpoints for recovery from failures

## Robustness Features

### Error Handling

Enhanced error handling throughout the application:

- **Structured Errors**: Hierarchical error types with severity levels
- **Circuit Breaking**: Prevents cascading failures by temporarily disabling failing components
- **Retry Mechanisms**: Automatic retries with configurable backoff
- **Error Reporting**: Detailed error reporting with context information

```python
# Example: Using the error handling utilities
from utils.error_handling import retry, ErrorBoundary, capture_exceptions

@retry(max_attempts=3, delay=1)
def process_with_retry(document_path):
    # Function will be retried up to 3 times if it fails
    return process_document(document_path)

with ErrorBoundary("document_processing", fallback=default_result):
    # If an error occurs, the fallback value is returned
    result = risky_operation()

@capture_exceptions(document_processing_error_handler)
def process_document(path):
    # Errors will be captured and sent to the handler function
    # but won't crash the application
    return extract_document_data(path)
```

### Recovery Mechanisms

Automated recovery for handling failures at different levels:

- **Checkpoint-Based Recovery**: Resume workflows from checkpoints after failures
- **Recovery Strategies**: Multiple strategies including retry, skip, and fallback
- **Partial Results**: Continue processing with partial results when possible
- **Batch Recovery**: Resume interrupted batch processing jobs

```python
# Example: Setting up and using recovery
from utils.recovery_manager import RecoveryManager, RecoveryStrategy

recovery_manager = RecoveryManager()

# Create a recovery plan
recovery_plan = recovery_manager.create_recovery_plan(
    workflow_id="invoice_processing_12345",
    strategy=RecoveryStrategy.CHECKPOINT,
    max_attempts=3
)

# Execute recovery
result = recovery_manager.execute_recovery_plan(recovery_plan)
```

### Health Monitoring

Comprehensive system health monitoring:

- **Service Checks**: Regular checks of critical system components
- **Resource Monitoring**: Track system resource usage and performance
- **Network Checks**: Verify connectivity to external dependencies
- **Health API**: API endpoints to retrieve current health status

```python
# Example: Setting up health monitoring
from utils.health_check import HealthCheckManager

health_manager = HealthCheckManager.get_instance()

# Register a custom health check
health_manager.register_check(
    name="database_connection",
    check_function=check_database_connection,
    critical=True,
    interval_seconds=30
)

# Start health monitoring
health_manager.start()

# Get health status
health_report = health_manager.get_full_report()
```

### Metrics Collection

Extensive metrics collection throughout the application:

- **Performance Metrics**: Timing of operations and throughput
- **Resource Metrics**: CPU, memory, and disk usage
- **Business Metrics**: Document processing success rates and types
- **Export Formats**: Support for Prometheus and JSON formats

```python
# Example: Recording and retrieving metrics
from utils.metrics import AppMetricsCollector, MetricPoint, MetricType

metrics = AppMetricsCollector.get_instance()

# Record a counter
metrics.record(MetricPoint(
    name="documents_processed",
    value=1,
    labels={"document_type": "invoice"},
    metric_type=MetricType.COUNTER
))

# Record processing time
with metrics.timer("document_processing_time", labels={"type": "invoice"}):
    process_document(doc)

# Export metrics
prometheus_metrics = metrics.export_prometheus()
json_metrics = metrics.export_json()
```

## Command-Line Usage

The main application now supports additional command-line arguments for optimization and robustness:

```
python main.py --mode optimized --parallel --workers 4 --recovery
```

Available options:

- `--mode [standard|optimized|monitor]`: Operation mode
- `--parallel`: Enable parallel execution
- `--workers N`: Number of worker threads
- `--recovery`: Enable automatic recovery
- `--metrics-file PATH`: Path to write metrics
- `--health-check`: Enable health checking
- `--batch-dir PATH`: Directory for batch processing

## API Endpoints

New API endpoints for optimization and robustness features:

### Batch Processing

- `POST /batch/process`: Start a batch processing job
- `GET /batch/{batch_id}/status`: Get status of a batch job
- `GET /batch/list/active`: List active batch jobs
- `GET /batch/list/recoverable`: List recoverable batch jobs
- `POST /batch/{batch_id}/resume`: Resume an interrupted batch job

### Health and Metrics

- `GET /health`: Get system health status
- `GET /metrics?format=json|prometheus`: Get system metrics
- `GET /diagnostics`: Get system diagnostic information

## Configuration

Configuration options for the optimized components:

### Workflow Runner Configuration

```json
{
  "workflow_runner": {
    "parallel_execution": true,
    "max_parallel_tasks": 4,
    "timeout_seconds": 30,
    "retry_attempts": 3,
    "retry_delay_seconds": 1
  }
}
```

### Document Processor Configuration

```json
{
  "document_processor": {
    "document_cache_enabled": true,
    "cache_size": 100,
    "cache_ttl_seconds": 3600,
    "ocr_optimization_level": "high"
  }
}
```

### Batch Processor Configuration

```json
{
  "batch_processor": {
    "min_workers": 2,
    "max_workers": 8,
    "adaptive_workers": true,
    "checkpoint_interval": 10,
    "recovery_dir": "/path/to/recovery"
  }
}
```

### Health Check Configuration

```json
{
  "health_check": {
    "check_interval_seconds": 60,
    "health_output_file": "/path/to/health.json",
    "critical_service_timeout_seconds": 5
  }
}
```
