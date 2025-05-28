"""
Metrics and monitoring system for FinFlow.

This module provides:
1. Metric collection and reporting
2. System health monitoring
3. Performance tracking
4. Resource usage monitoring
"""

import logging
import time
import threading
import os
import psutil
import json
import traceback
from typing import TypeVar, Callable, Dict, Any, Optional, List, Tuple
from enum import Enum
from functools import wraps
from dataclasses import dataclass, field

# Define TypeVar for function decorators
F = TypeVar('F', bound=Callable[..., Any])


# -----------------------------------------------------------------------------
# Metric Types and Base Classes
# -----------------------------------------------------------------------------

class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"       # Increases over time (e.g., request count)
    GAUGE = "gauge"           # Can go up and down (e.g., memory usage)
    HISTOGRAM = "histogram"   # Distribution of values (e.g., response times)
    SUMMARY = "summary"       # Similar to histogram but with percentiles
    TIMER = "timer"           # Duration of operations


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "type": self.metric_type.value
        }


class BaseMetric:
    """Base class for all metrics."""
    
    def __init__(
        self, 
        name: str, 
        description: str = "",
        labels: Dict[str, str] = None
    ):
        """
        Initialize the metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: Default labels for this metric
        """
        self.name = name
        self.description = description
        self.default_labels = labels or {}
        self.logger = logging.getLogger(f"finflow.metrics.{name}")
        
    def create_point(
        self,
        value: float,
        labels: Dict[str, str] = None,
        timestamp: Optional[float] = None
    ) -> MetricPoint:
        """
        Create a metric data point.
        
        Args:
            value: The metric value
            labels: Additional labels for this point
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            A MetricPoint object
        """
        combined_labels = dict(self.default_labels)
        if labels:
            combined_labels.update(labels)
            
        return MetricPoint(
            name=self.name,
            value=value,
            labels=combined_labels,
            timestamp=timestamp or time.time(),
            metric_type=self.get_type()
        )
    
    def get_type(self) -> MetricType:
        """Get the type of this metric."""
        raise NotImplementedError("Subclasses must implement this method")


class Counter(BaseMetric):
    """Counter metric that increases over time."""
    
    def __init__(
        self, 
        name: str, 
        description: str = "",
        labels: Dict[str, str] = None
    ):
        """Initialize the counter."""
        super().__init__(name, description, labels)
        self._value = 0.0
        self._lock = threading.Lock()
    
    def inc(self, amount: float = 1.0, labels: Dict[str, str] = None) -> MetricPoint:
        """
        Increment the counter.
        
        Args:
            amount: Amount to increment by
            labels: Additional labels for this increment
            
        Returns:
            A MetricPoint with the new value
        """
        with self._lock:
            self._value += amount
            return self.create_point(self._value, labels)
    
    def increment(self, value: float = 1.0, labels: Dict[str, str] = None) -> MetricPoint:
        """
        Increment the counter (alias for inc()).
        
        Args:
            value: Amount to increment by
            labels: Additional labels for this increment
            
        Returns:
            A MetricPoint with the new value
        """
        return self.inc(value, labels)
    
    def get_value(self) -> float:
        """Get the current value."""
        with self._lock:
            return self._value
    
    def get_type(self) -> MetricType:
        """Get the type of this metric."""
        return MetricType.COUNTER


class Gauge(BaseMetric):
    """Gauge metric that can go up or down."""
    
    def __init__(
        self, 
        name: str, 
        description: str = "",
        labels: Dict[str, str] = None
    ):
        """Initialize the gauge."""
        super().__init__(name, description, labels)
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: float, labels: Dict[str, str] = None) -> MetricPoint:
        """
        Set the gauge to a value.
        
        Args:
            value: Value to set
            labels: Additional labels
            
        Returns:
            A MetricPoint with the new value
        """
        with self._lock:
            self._value = value
            return self.create_point(self._value, labels)
    
    def inc(self, amount: float = 1.0, labels: Dict[str, str] = None) -> MetricPoint:
        """Increment the gauge."""
        with self._lock:
            self._value += amount
            return self.create_point(self._value, labels)
    
    def dec(self, amount: float = 1.0, labels: Dict[str, str] = None) -> MetricPoint:
        """Decrement the gauge."""
        with self._lock:
            self._value -= amount
            return self.create_point(self._value, labels)
    
    def get_value(self) -> float:
        """Get the current value."""
        with self._lock:
            return self._value
    
    def get_type(self) -> MetricType:
        """Get the type of this metric."""
        return MetricType.GAUGE


class Histogram(BaseMetric):
    """Histogram metric for distribution of values."""
    
    def __init__(
        self, 
        name: str, 
        description: str = "",
        labels: Dict[str, str] = None,
        buckets: List[float] = None
    ):
        """
        Initialize the histogram.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: Default labels
            buckets: Bucket boundaries for the histogram
        """
        super().__init__(name, description, labels)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self.buckets.sort()
        
        # Initialize buckets dictionary with counts of 0
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float('inf')] = 0  # Overflow bucket
        
        self._sum = 0.0
        self._count = 0
        
        self._lock = threading.Lock()
    
    def observe(self, value: float, labels: Dict[str, str] = None) -> List[MetricPoint]:
        """
        Observe a value.
        
        Args:
            value: The value to observe
            labels: Additional labels
            
        Returns:
            List of MetricPoints with updated bucket values
        """
        points = []
        
        with self._lock:
            # Update sum and count
            self._sum += value
            self._count += 1
            
            # Update buckets
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
            
            self._bucket_counts[float('inf')] += 1  # Always update overflow bucket
            
            # Create metric points
            timestamp = time.time()
            base_labels = dict(self.default_labels)
            if labels:
                base_labels.update(labels)
            
            # Add points for each bucket
            for bucket, count in self._bucket_counts.items():
                bucket_labels = dict(base_labels)
                bucket_labels["le"] = str(bucket) if bucket != float('inf') else "+Inf"
                points.append(self.create_point(count, bucket_labels, timestamp))
            
            # Add sum and count metrics
            sum_labels = dict(base_labels)
            sum_labels["metric"] = "sum"
            points.append(self.create_point(self._sum, sum_labels, timestamp))
            
            count_labels = dict(base_labels)
            count_labels["metric"] = "count"
            points.append(self.create_point(self._count, count_labels, timestamp))
            
        return points
    
    def get_type(self) -> MetricType:
        """Get the type of this metric."""
        return MetricType.HISTOGRAM


class Timer(BaseMetric):
    """Timer for measuring operation duration."""
    
    def __init__(
        self, 
        name: str, 
        description: str = "",
        labels: Dict[str, str] = None
    ):
        """Initialize the timer."""
        super().__init__(name, description, labels)
        self.histogram = Histogram(
            name=name,
            description=description,
            labels=labels,
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        )
    
    def time(self) -> 'TimerContext':
        """Start timing an operation."""
        return TimerContext(self)
    
    def get_type(self) -> MetricType:
        """Get the type of this metric."""
        return MetricType.TIMER


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, timer: Timer):
        """Initialize timer context with parent timer."""
        self.timer = timer
        self.start_time: float = 0.0
        self.labels: Dict[str, str] = {}
    
    def __enter__(self) -> 'TimerContext':
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Record the elapsed time."""
        elapsed = time.time() - self.start_time
        self.timer.histogram.observe(elapsed, self.labels)
    
    def with_labels(self, labels: Dict[str, str]) -> 'TimerContext':
        """Set labels for this timer execution."""
        self.labels = labels
        return self


# -----------------------------------------------------------------------------
# Metrics Registry
# -----------------------------------------------------------------------------

class MetricsRegistry:
    """Registry for managing metrics."""
    
    _instance: Optional['MetricsRegistry'] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'MetricsRegistry':
        """Get singleton instance of the registry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        if self.__class__._instance is not None:
            raise RuntimeError("This class is a singleton. Use get_instance() instead.")
            
        self.metrics: Dict[str, BaseMetric] = {}
        self.metric_points: List[MetricPoint] = []
        self.max_points = 10000  # Maximum number of points to store
        
        self.logger = logging.getLogger("finflow.metrics")
        
        # For reporting
        self.last_report_time = time.time()
        self.reporting_interval = 60.0  # Report every minute by default
        self._reporting_thread = None
        
        # Callbacks for metric reports
        self._report_callbacks: List[Callable[[List[Dict[str, Any]]], None]] = []
    
    def register_metric(self, metric: BaseMetric) -> BaseMetric:
        """
        Register a metric with the registry.
        
        Args:
            metric: The metric to register
            
        Returns:
            The registered metric
        """
        if metric.name in self.metrics:
            self.logger.warning(f"Metric {metric.name} already registered, returning existing instance")
            return self.metrics[metric.name]
            
        self.metrics[metric.name] = metric
        return metric
    
    def create_counter(self, name: str, description: str = "", labels: Dict[str, str] = None) -> Counter:
        """Create and register a counter metric."""
        metric = Counter(name, description, labels)
        return self.register_metric(metric)  # type: ignore
    
    def create_gauge(self, name: str, description: str = "", labels: Dict[str, str] = None) -> Gauge:
        """Create and register a gauge metric."""
        metric = Gauge(name, description, labels)
        return self.register_metric(metric)  # type: ignore
    
    def create_histogram(
        self, 
        name: str, 
        description: str = "", 
        labels: Dict[str, str] = None,
        buckets: List[float] = None
    ) -> Histogram:
        """Create and register a histogram metric."""
        metric = Histogram(name, description, labels, buckets)
        return self.register_metric(metric)  # type: ignore
    
    def create_timer(self, name: str, description: str = "", labels: Dict[str, str] = None) -> Timer:
        """Create and register a timer metric."""
        metric = Timer(name, description, labels)
        return self.register_metric(metric)  # type: ignore
    
    def record_metric(self, metric_point: MetricPoint) -> None:
        """
        Record a metric data point.
        
        Args:
            metric_point: The metric point to record
        """
        self.metric_points.append(metric_point)
        
        # Trim if we have too many points
        if len(self.metric_points) > self.max_points:
            # Remove oldest 10% of points
            trim_count = self.max_points // 10
            self.metric_points = self.metric_points[trim_count:]
    
    def record_metrics(self, metric_points: List[MetricPoint]) -> None:
        """Record multiple metric points at once."""
        for point in metric_points:
            self.record_metric(point)
    
    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def get_metrics_data(
        self, 
        since: Optional[float] = None, 
        metric_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get metrics data as dictionaries.
        
        Args:
            since: Only include metrics recorded after this timestamp
            metric_names: Only include metrics with these names
            
        Returns:
            List of metric data dictionaries
        """
        result = []
        
        for point in self.metric_points:
            if since is not None and point.timestamp < since:
                continue
                
            if metric_names is not None and point.name not in metric_names:
                continue
                
            result.append(point.to_dict())
            
        return result
    
    def add_report_callback(self, callback: Callable[[List[Dict[str, Any]]], None]) -> None:
        """
        Add a callback for metric reporting.
        
        Args:
            callback: Function to call with metrics data during reporting
        """
        self._report_callbacks.append(callback)
    
    def report_metrics(self) -> None:
        """Generate a metrics report and call registered callbacks."""
        now = time.time()
        
        # Get metrics since last report
        metrics_data = self.get_metrics_data(since=self.last_report_time)
        self.last_report_time = now
        
        # Call callbacks
        for callback in self._report_callbacks:
            try:
                callback(metrics_data)
            except Exception as e:
                self.logger.error(f"Error in metrics report callback: {e}")
    
    def start_reporting(self, interval: float = 60.0) -> None:
        """
        Start periodic metrics reporting in a background thread.
        
        Args:
            interval: Reporting interval in seconds
        """
        if self._reporting_thread is not None and self._reporting_thread.is_alive():
            self.logger.warning("Reporting thread already running")
            return
            
        self.reporting_interval = interval
        
        def reporting_loop() -> None:
            """Background thread function for periodic reporting."""
            while True:
                time.sleep(self.reporting_interval)
                try:
                    self.report_metrics()
                except Exception as e:
                    self.logger.error(f"Error in metrics reporting: {e}")
        
        self._reporting_thread = threading.Thread(
            target=reporting_loop, 
            name="metrics-reporter",
            daemon=True
        )
        self._reporting_thread.start()
        self.logger.info(f"Started metrics reporting with {interval}s interval")


# -----------------------------------------------------------------------------
# System Metrics Collector
# -----------------------------------------------------------------------------

class SystemMetricsCollector:
    """Collects system-level metrics like CPU and memory usage."""
    
    _instance: Optional['SystemMetricsCollector'] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'SystemMetricsCollector':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the system metrics collector."""
        if self.__class__._instance is not None:
            raise RuntimeError("This class is a singleton. Use get_instance() instead.")
            
        self.registry = MetricsRegistry.get_instance()
        self.logger = logging.getLogger("finflow.metrics.system")
        
        # Create system metrics
        self.cpu_usage = self.registry.create_gauge(
            "system_cpu_usage", 
            "CPU usage percentage"
        )
        
        self.memory_usage = self.registry.create_gauge(
            "system_memory_usage", 
            "Memory usage in bytes"
        )
        
        self.memory_percent = self.registry.create_gauge(
            "system_memory_percent", 
            "Memory usage percentage"
        )
        
        self.disk_usage = self.registry.create_gauge(
            "system_disk_usage", 
            "Disk usage in bytes"
        )
        
        self.disk_percent = self.registry.create_gauge(
            "system_disk_percent", 
            "Disk usage percentage"
        )
        
        self.open_files = self.registry.create_gauge(
            "system_open_files", 
            "Number of open files"
        )
        
        self.thread_count = self.registry.create_gauge(
            "system_thread_count", 
            "Number of threads"
        )
        
        # Collection state
        self.collection_interval = 30.0  # seconds
        self._collection_thread = None
    
    def collect_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.5)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            mem = psutil.virtual_memory()
            self.memory_usage.set(mem.used)
            self.memory_percent.set(mem.percent)
            
            # Disk usage
            try:
                current_dir = os.getcwd()
                disk_usage = psutil.disk_usage(current_dir)
                self.disk_usage.set(disk_usage.used)
                self.disk_percent.set(disk_usage.percent)
            except Exception as e:
                self.logger.warning(f"Failed to get disk usage: {e}")
            
            # Process info
            try:
                process = psutil.Process()
                
                # Open files count
                open_files = process.open_files()
                self.open_files.set(len(open_files))
                
                # Thread count
                threads = process.threads()
                self.thread_count.set(len(threads))
                
            except Exception as e:
                self.logger.warning(f"Failed to get process info: {e}")
                
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def start_collection(self, interval: float = 30.0) -> None:
        """
        Start periodic metrics collection in a background thread.
        
        Args:
            interval: Collection interval in seconds
        """
        if self._collection_thread is not None and self._collection_thread.is_alive():
            self.logger.warning("Collection thread already running")
            return
            
        self.collection_interval = interval
        
        def collection_loop() -> None:
            """Background thread function for periodic collection."""
            while True:
                try:
                    self.collect_metrics()
                except Exception as e:
                    self.logger.error(f"Error in metrics collection: {e}")
                
                time.sleep(self.collection_interval)
        
        self._collection_thread = threading.Thread(
            target=collection_loop, 
            name="system-metrics-collector",
            daemon=True
        )
        self._collection_thread.start()
        self.logger.info(f"Started system metrics collection with {interval}s interval")


# -----------------------------------------------------------------------------
# Application Metrics Collector
# -----------------------------------------------------------------------------

class AppMetricsCollector:
    """Collects application-level metrics."""
    
    _instance: Optional['AppMetricsCollector'] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'AppMetricsCollector':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the application metrics collector."""
        if self.__class__._instance is not None:
            raise RuntimeError("This class is a singleton. Use get_instance() instead.")
            
        self.registry = MetricsRegistry.get_instance()
        self.logger = logging.getLogger("finflow.metrics.app")
        
        # Create application metrics
        self.request_count = self.registry.create_counter(
            "app_request_count", 
            "Number of requests"
        )
        
        self.request_duration = self.registry.create_timer(
            "app_request_duration", 
            "Request duration in seconds"
        )
        
        self.error_count = self.registry.create_counter(
            "app_error_count", 
            "Number of errors"
        )
        
        self.active_requests = self.registry.create_gauge(
            "app_active_requests", 
            "Number of active requests"
        )
        
        self.document_count = self.registry.create_counter(
            "app_document_count", 
            "Number of documents processed"
        )
        
        self.document_processing_time = self.registry.create_timer(
            "app_document_processing_time", 
            "Document processing time in seconds"
        )
        
        # Workflow metrics
        self.workflow_start_count = self.registry.create_counter(
            "app_workflow_start_count", 
            "Number of workflows started"
        )
        
        self.workflow_complete_count = self.registry.create_counter(
            "app_workflow_complete_count", 
            "Number of workflows completed"
        )
        
        self.workflow_error_count = self.registry.create_counter(
            "app_workflow_error_count", 
            "Number of workflows that errored"
        )
        
        self.workflow_duration = self.registry.create_timer(
            "app_workflow_duration", 
            "Workflow duration in seconds"
        )
        
        # Agent metrics
        self.agent_call_count = self.registry.create_counter(
            "app_agent_call_count", 
            "Number of agent calls"
        )
        
        self.agent_call_duration = self.registry.create_timer(
            "app_agent_call_duration", 
            "Agent call duration in seconds"
        )
        
        self.agent_error_count = self.registry.create_counter(
            "app_agent_error_count", 
            "Number of agent errors"
        )
        
    def track_request(self, endpoint: str) -> TimerContext:
        """
        Track a request.
        
        Args:
            endpoint: API endpoint being accessed
            
        Returns:
            Timer context for tracking request duration
        """
        self.request_count.inc(labels={"endpoint": endpoint})
        self.active_requests.inc(labels={"endpoint": endpoint})
        
        return self.request_duration.time().with_labels({"endpoint": endpoint})
    
    def end_request(self, endpoint: str) -> None:
        """
        End request tracking.
        
        Args:
            endpoint: API endpoint being accessed
        """
        self.active_requests.dec(labels={"endpoint": endpoint})
    
    def track_error(self, error_type: str, error_code: str) -> None:
        """
        Track an error.
        
        Args:
            error_type: Type of error
            error_code: Error code
        """
        self.error_count.inc(labels={"type": error_type, "code": error_code})
    
    def track_document(self, doc_type: str) -> TimerContext:
        """
        Track document processing.
        
        Args:
            doc_type: Type of document
            
        Returns:
            Timer context for tracking document processing time
        """
        self.document_count.inc(labels={"type": doc_type})
        return self.document_processing_time.time().with_labels({"type": doc_type})
    
    def track_workflow(self, workflow_type: str) -> TimerContext:
        """
        Track workflow execution.
        
        Args:
            workflow_type: Type of workflow
            
        Returns:
            Timer context for tracking workflow duration
        """
        self.workflow_start_count.inc(labels={"type": workflow_type})
        return self.workflow_duration.time().with_labels({"type": workflow_type})
    
    def end_workflow(self, workflow_type: str, success: bool) -> None:
        """
        End workflow tracking.
        
        Args:
            workflow_type: Type of workflow
            success: Whether the workflow completed successfully
        """
        if success:
            self.workflow_complete_count.inc(labels={"type": workflow_type})
        else:
            self.workflow_error_count.inc(labels={"type": workflow_type})
    
    def track_agent_call(self, agent_name: str, operation: str) -> TimerContext:
        """
        Track an agent call.
        
        Args:
            agent_name: Name of the agent
            operation: Operation being performed
            
        Returns:
            Timer context for tracking call duration
        """
        self.agent_call_count.inc(labels={"agent": agent_name, "operation": operation})
        return self.agent_call_duration.time().with_labels({"agent": agent_name, "operation": operation})
    
    def track_agent_error(self, agent_name: str, error_type: str) -> None:
        """
        Track an agent error.
        
        Args:
            agent_name: Name of the agent
            error_type: Type of error
        """
        self.agent_error_count.inc(labels={"agent": agent_name, "type": error_type})
        
    def record(self, metric_point: MetricPoint) -> None:
        """
        Record a metric point.
        
        Args:
            metric_point: The metric point to record
        """
        # Delegate to the registry for recording
        self.registry.record_metric(metric_point)
        
        # Also update the appropriate internal metric if it exists
        metric_name = metric_point.name
        labels = metric_point.labels or {}
        
        # Map common metric names to internal metrics
        if metric_name == "http_requests_total":
            if metric_point.metric_type == MetricType.COUNTER:
                endpoint = labels.get("path", "unknown")
                self.request_count.inc(labels={"endpoint": endpoint})
        elif metric_name == "http_response_time_ms":
            if metric_point.metric_type == MetricType.HISTOGRAM:
                # Convert milliseconds to seconds for our timer
                duration_seconds = metric_point.value / 1000.0
                endpoint = labels.get("path", "unknown")
                # Record the timing directly to the histogram
                self.request_duration.histogram.observe(duration_seconds, {"endpoint": endpoint})
        elif metric_name == "http_errors_total":
            if metric_point.metric_type == MetricType.COUNTER:
                error_type = labels.get("error", "unknown")
                self.error_count.inc(labels={"type": error_type, "code": "500"})
    
    def histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """
        Record a histogram value.
        
        Args:
            name: Metric name
            value: Value to record
            labels: Additional labels
        """
        # Try to find an existing histogram metric
        metric = self.registry.get_metric(name)
        if metric and hasattr(metric, 'observe'):
            metric.observe(value, labels)
        else:
            # Create a new histogram if it doesn't exist
            histogram = self.registry.create_histogram(name, f"Histogram for {name}", labels)
            histogram.observe(value, labels)
    
    def gauge(self, name: str):
        """
        Get or create a gauge metric.
        
        Args:
            name: Metric name
            
        Returns:
            Gauge metric object
        """
        # Try to find an existing gauge metric
        metric = self.registry.get_metric(name)
        if metric and hasattr(metric, 'set'):
            return metric
        else:
            # Create a new gauge if it doesn't exist
            return self.registry.create_gauge(name, f"Gauge for {name}")
    
    def counter(self, name: str):
        """
        Get or create a counter metric.
        
        Args:
            name: Metric name
            
        Returns:
            Counter metric object
        """
        # Try to find an existing counter metric
        metric = self.registry.get_metric(name)
        if metric and hasattr(metric, 'inc'):
            return metric
        else:
            # Create a new counter if it doesn't exist
            return self.registry.create_counter(name, f"Counter for {name}")
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus formatted metrics string
        """
        try:
            lines = []
            
            # Get all metrics from registry
            for metric_name, metric in self.registry.metrics.items():
                # Add metric help
                lines.append(f"# HELP {metric_name} {metric.description}")
                lines.append(f"# TYPE {metric_name} {metric.get_type().value}")
                
                # Get current value based on metric type
                if hasattr(metric, 'get_value'):
                    value = metric.get_value()
                    labels_str = ",".join(f'{k}="{v}"' for k, v in metric.default_labels.items())
                    labels_part = f"{{{labels_str}}}" if labels_str else ""
                    lines.append(f"{metric_name}{labels_part} {value}")
                
            return "\n".join(lines)
        except Exception as e:
            self.logger.error(f"Error exporting Prometheus metrics: {e}")
            return "# Error exporting metrics\n"
    
    def export_json(self) -> str:
        """
        Export metrics in JSON format.
        
        Returns:
            JSON formatted metrics string
        """
        try:
            metrics_data = []
            
            # Get all metrics from registry  
            for metric_name, metric in self.registry.metrics.items():
                metric_info = {
                    "name": metric_name,
                    "description": metric.description,
                    "type": metric.get_type().value,
                    "labels": metric.default_labels
                }
                
                # Get current value based on metric type
                if hasattr(metric, 'get_value'):
                    metric_info["value"] = metric.get_value()
                
                metrics_data.append(metric_info)
            
            return json.dumps(metrics_data, indent=2)
        except Exception as e:
            self.logger.error(f"Error exporting JSON metrics: {e}")
            return json.dumps({"error": "Error exporting metrics"})


# -----------------------------------------------------------------------------
# Metric Decorators
# -----------------------------------------------------------------------------

def time_function(
    name: Optional[str] = None,
    metric_type: MetricType = MetricType.TIMER,
    labels: Optional[Dict[str, str]] = None,
    include_args: bool = False
) -> Callable[[F], F]:
    """
    Decorator to time function execution and record metrics.
    
    Args:
        name: Name of the metric (defaults to function name)
        metric_type: Type of metric to record (defaults to TIMER)
        labels: Default labels to apply to metrics
        include_args: Whether to include function arguments in labels
        
    Returns:
        Decorated function that records timing metrics
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get metrics collector
            app_metrics = AppMetricsCollector.get_instance()
            
            # Generate metric name
            metric_name = name or f"{func.__module__}.{func.__name__}"
            
            # Prepare labels
            metric_labels = dict(labels or {})
            metric_labels["function"] = func.__name__
            metric_labels["module"] = func.__module__
            
            if include_args and args:
                # Add first argument as context if it's a string
                if args and isinstance(args[0], str):
                    metric_labels["context"] = str(args[0])[:50]  # Truncate to avoid long labels
            
            # Start timing
            start_time = time.time()
            error_occurred = False
            error_type = "none"
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                error_type = type(e).__name__
                raise
            finally:
                # Record timing metric
                elapsed_time = time.time() - start_time
                
                # Add status labels
                metric_labels["status"] = "error" if error_occurred else "success"
                metric_labels["error_type"] = error_type
                
                # Record the metric
                if metric_type == MetricType.TIMER:
                    # Record as timer metric
                    metric_point = MetricPoint(
                        name=f"{metric_name}_duration",
                        value=elapsed_time,
                        labels=metric_labels,
                        metric_type=MetricType.HISTOGRAM
                    )
                    app_metrics.record(metric_point)
                    
                    # Also record count
                    count_point = MetricPoint(
                        name=f"{metric_name}_count",
                        value=1.0,
                        labels=metric_labels,
                        metric_type=MetricType.COUNTER
                    )
                    app_metrics.record(count_point)
                
                elif metric_type == MetricType.COUNTER:
                    # Record as counter
                    metric_point = MetricPoint(
                        name=metric_name,
                        value=1.0,
                        labels=metric_labels,
                        metric_type=MetricType.COUNTER
                    )
                    app_metrics.record(metric_point)
        
        return wrapper  # type: ignore
    return decorator


def count_invocations(
    name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None
) -> Callable[[F], F]:
    """
    Decorator to count function invocations.
    
    Args:
        name: Name of the metric (defaults to function name)
        labels: Default labels to apply to metrics
        
    Returns:
        Decorated function that records invocation counts
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get metrics collector
            app_metrics = AppMetricsCollector.get_instance()
            
            # Generate metric name
            metric_name = name or f"{func.__module__}.{func.__name__}_invocations"
            
            # Prepare labels
            metric_labels = dict(labels or {})
            metric_labels["function"] = func.__name__
            metric_labels["module"] = func.__module__
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                metric_labels["status"] = "success"
                return result
            except Exception as e:
                metric_labels["status"] = "error"
                metric_labels["error_type"] = type(e).__name__
                raise
            finally:
                # Record invocation count
                metric_point = MetricPoint(
                    name=metric_name,
                    value=1.0,
                    labels=metric_labels,
                    metric_type=MetricType.COUNTER
                )
                app_metrics.record(metric_point)
        
        return wrapper  # type: ignore
    return decorator


def track_errors(
    name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    reraise: bool = True
) -> Callable[[F], F]:
    """
    Decorator to track function errors.
    
    Args:
        name: Name of the metric (defaults to function name)
        labels: Default labels to apply to metrics
        reraise: Whether to reraise caught exceptions
        
    Returns:
        Decorated function that records error metrics
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get metrics collector
            app_metrics = AppMetricsCollector.get_instance()
            
            # Generate metric name
            metric_name = name or f"{func.__module__}.{func.__name__}_errors"
            
            # Prepare labels
            metric_labels = dict(labels or {})
            metric_labels["function"] = func.__name__
            metric_labels["module"] = func.__module__
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Record error metric
                metric_labels["error_type"] = type(e).__name__
                metric_labels["error_message"] = str(e)[:100]  # Truncate error message
                
                metric_point = MetricPoint(
                    name=metric_name,
                    value=1.0,
                    labels=metric_labels,
                    metric_type=MetricType.COUNTER
                )
                app_metrics.record(metric_point)
                
                if reraise:
                    raise
                else:
                    # Log error instead of reraising
                    logger = logging.getLogger(f"finflow.metrics.{func.__module__}")
                    logger.error(f"Error in {func.__name__}: {e}")
                    return None
        
        return wrapper  # type: ignore
    return decorator


# -----------------------------------------------------------------------------
# Global Metrics Instance
# -----------------------------------------------------------------------------

# Create global metrics instances for easy access
metrics_registry = MetricsRegistry.get_instance()
system_metrics = SystemMetricsCollector.get_instance()
app_metrics = AppMetricsCollector.get_instance()
