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


# -----------------------------------------------------------------------------
# Health Check System
# -----------------------------------------------------------------------------

class HealthStatus(Enum):
    """Health status of a component."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "status": self.status.value,
            "details": self.details,
            "timestamp": self.timestamp
        }


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, component: str, description: str = ""):
        """
        Initialize a health check.
        
        Args:
            component: Component name
            description: Health check description
        """
        self.component = component
        self.description = description
        self.logger = logging.getLogger(f"finflow.health.{component}")
    
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        raise NotImplementedError("Subclasses must implement this method")


class HealthCheckSystem:
    """System for managing health checks."""
    
    _instance: Optional['HealthCheckSystem'] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'HealthCheckSystem':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the health check system."""
        if self.__class__._instance is not None:
            raise RuntimeError("This class is a singleton. Use get_instance() instead.")
            
        self.health_checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        
        self.logger = logging.getLogger("finflow.health")
        
        # For periodic health checking
        self.check_interval = 60.0  # seconds
        self._check_thread = None
    
    def register_check(self, check: HealthCheck) -> None:
        """
        Register a health check.
        
        Args:
            check: The health check to register
        """
        self.health_checks[check.component] = check
        self.logger.info(f"Registered health check for {check.component}")
    
    def unregister_check(self, component: str) -> None:
        """
        Unregister a health check.
        
        Args:
            component: Component name of the check to unregister
        """
        if component in self.health_checks:
            del self.health_checks[component]
            if component in self.results:
                del self.results[component]
            self.logger.info(f"Unregistered health check for {component}")
    
    def run_check(self, component: str) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            component: Component to check
            
        Returns:
            Health check result
            
        Raises:
            KeyError: If the component is not registered
        """
        check = self.health_checks[component]
        try:
            result = check.check()
            self.results[component] = result
            return result
        except Exception as e:
            self.logger.error(f"Health check for {component} failed: {e}")
            result = HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
            self.results[component] = result
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        for component in self.health_checks:
            self.run_check(component)
            
        return self.results
    
    def get_system_status(self) -> Tuple[HealthStatus, Dict[str, HealthCheckResult]]:
        """
        Get overall system health status.
        
        Returns:
            Tuple of (overall status, individual check results)
        """
        # Run all checks to make sure we have fresh results
        self.run_all_checks()
        
        # Determine overall status
        if not self.results:
            return HealthStatus.UNKNOWN, {}
            
        has_unhealthy = any(r.status == HealthStatus.UNHEALTHY for r in self.results.values())
        has_degraded = any(r.status == HealthStatus.DEGRADED for r in self.results.values())
        
        if has_unhealthy:
            overall = HealthStatus.UNHEALTHY
        elif has_degraded:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
            
        return overall, self.results
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get a health report.
        
        Returns:
            Dictionary with health status information
        """
        overall, results = self.get_system_status()
        
        return {
            "status": overall.value,
            "timestamp": time.time(),
            "components": {
                component: result.to_dict()
                for component, result in results.items()
            }
        }
    
    def start_checking(self, interval: float = 60.0) -> None:
        """
        Start periodic health checking in a background thread.
        
        Args:
            interval: Check interval in seconds
        """
        if self._check_thread is not None and self._check_thread.is_alive():
            self.logger.warning("Health check thread already running")
            return
            
        self.check_interval = interval
        
        def check_loop() -> None:
            """Background thread function for periodic health checking."""
            while True:
                try:
                    self.run_all_checks()
                except Exception as e:
                    self.logger.error(f"Error in health checking: {e}")
                
                time.sleep(self.check_interval)
        
        self._check_thread = threading.Thread(
            target=check_loop, 
            name="health-checker",
            daemon=True
        )
        self._check_thread.start()
        self.logger.info(f"Started health checking with {interval}s interval")


# -----------------------------------------------------------------------------
# Common Health Checks
# -----------------------------------------------------------------------------

class SystemResourcesCheck(HealthCheck):
    """Check system resources like CPU and memory."""
    
    def __init__(self, 
                 cpu_threshold: float = 90.0, 
                 memory_threshold: float = 90.0,
                 disk_threshold: float = 90.0):
        """
        Initialize the system resources check.
        
        Args:
            cpu_threshold: CPU usage percentage threshold for degraded status
            memory_threshold: Memory usage percentage threshold for degraded status
            disk_threshold: Disk usage percentage threshold for degraded status
        """
        super().__init__("system_resources", "System resource usage check")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": disk.percent
            }
            
            # Determine status
            status = HealthStatus.HEALTHY
            
            if (cpu_percent > self.cpu_threshold or 
                memory.percent > self.memory_threshold or 
                disk.percent > self.disk_threshold):
                status = HealthStatus.DEGRADED
                
                # Add specific warnings
                warnings = []
                if cpu_percent > self.cpu_threshold:
                    warnings.append(f"CPU usage ({cpu_percent}%) exceeds threshold ({self.cpu_threshold}%)")
                if memory.percent > self.memory_threshold:
                    warnings.append(f"Memory usage ({memory.percent}%) exceeds threshold ({self.memory_threshold}%)")
                if disk.percent > self.disk_threshold:
                    warnings.append(f"Disk usage ({disk.percent}%) exceeds threshold ({self.disk_threshold}%)")
                    
                details["warnings"] = warnings
            
            return HealthCheckResult(
                component=self.component,
                status=status,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"System resources check failed: {e}")
            return HealthCheckResult(
                component=self.component,
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e)}
            )


class DatabaseHealthCheck(HealthCheck):
    """Check database connectivity and performance."""
    
    def __init__(self, db_connection_func: Callable[[], Any]):
        """
        Initialize the database health check.
        
        Args:
            db_connection_func: Function that returns a database connection
        """
        super().__init__("database", "Database connectivity check")
        self.db_connection_func = db_connection_func
    
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        try:
            # Measure connection time
            start_time = time.time()
            conn = self.db_connection_func()
            connection_time = time.time() - start_time
            
            # Perform a simple query to ensure the connection works
            cursor = conn.cursor()
            
            query_start_time = time.time()
            cursor.execute("SELECT 1")
            query_time = time.time() - query_start_time
            
            cursor.close()
            
            # Check for slow responses
            status = HealthStatus.HEALTHY
            details = {
                "connection_time": connection_time,
                "query_time": query_time
            }
            
            # If connection or query is slow, mark as degraded
            if connection_time > 1.0 or query_time > 0.5:
                status = HealthStatus.DEGRADED
                details["warnings"] = []
                
                if connection_time > 1.0:
                    details["warnings"].append(f"Slow database connection: {connection_time:.2f}s")
                    
                if query_time > 0.5:
                    details["warnings"].append(f"Slow database query: {query_time:.2f}s")
            
            return HealthCheckResult(
                component=self.component,
                status=status,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return HealthCheckResult(
                component=self.component,
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e)}
            )


class ExternalServiceCheck(HealthCheck):
    """Check connectivity to an external service."""
    
    def __init__(self, service_name: str, check_func: Callable[[], bool]):
        """
        Initialize the external service check.
        
        Args:
            service_name: Name of the service
            check_func: Function that returns True if service is available
        """
        super().__init__(f"external_service_{service_name}", f"External service: {service_name}")
        self.service_name = service_name
        self.check_func = check_func
    
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        try:
            # Measure response time
            start_time = time.time()
            available = self.check_func()
            response_time = time.time() - start_time
            
            if available:
                status = HealthStatus.HEALTHY
                details = {
                    "response_time": response_time
                }
                
                # If service is slow, mark as degraded
                if response_time > 2.0:
                    status = HealthStatus.DEGRADED
                    details["warnings"] = [f"Slow service response: {response_time:.2f}s"]
            else:
                status = HealthStatus.UNHEALTHY
                details = {
                    "error": "Service is unavailable"
                }
            
            return HealthCheckResult(
                component=self.component,
                status=status,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"External service check for {self.service_name} failed: {e}")
            return HealthCheckResult(
                component=self.component,
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e)}
            )


# -----------------------------------------------------------------------------
# API and Decorators
# -----------------------------------------------------------------------------

def time_function(metric_name: str, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    """
    Decorator to time a function execution.
    
    Args:
        metric_name: Name of the timer metric
        labels: Additional labels for the metric
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = MetricsRegistry.get_instance()
            timer = registry.get_metric(metric_name)
            
            # Create timer if it doesn't exist
            if timer is None or not isinstance(timer, Timer):
                timer = registry.create_timer(metric_name)
            
            # Time the function
            with timer.time().with_labels(labels or {}):
                return func(*args, **kwargs)
                
        return wrapper  # type: ignore
    
    return decorator


def count_invocations(metric_name: str, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    """
    Decorator to count function invocations.
    
    Args:
        metric_name: Name of the counter metric
        labels: Additional labels for the metric
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = MetricsRegistry.get_instance()
            counter = registry.get_metric(metric_name)
            
            # Create counter if it doesn't exist
            if counter is None or not isinstance(counter, Counter):
                counter = registry.create_counter(metric_name)
            
            # Increment counter
            counter.inc(labels=labels)
            
            return func(*args, **kwargs)
                
        return wrapper  # type: ignore
    
    return decorator


def track_errors(metric_name: str, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    """
    Decorator to track errors in a function.
    
    Args:
        metric_name: Name of the counter metric
        labels: Additional labels for the metric
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = MetricsRegistry.get_instance()
            counter = registry.get_metric(metric_name)
            
            # Create counter if it doesn't exist
            if counter is None or not isinstance(counter, Counter):
                counter = registry.create_counter(metric_name)
                
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Increment error counter
                error_labels = dict(labels or {})
                error_labels["error_type"] = e.__class__.__name__
                counter.inc(labels=error_labels)
                raise
                
        return wrapper  # type: ignore
    
    return decorator


def initialize_metrics() -> None:
    """Initialize the metrics system."""
    # Get instances to ensure they're initialized
    registry = MetricsRegistry.get_instance()
    system_collector = SystemMetricsCollector.get_instance()
    AppMetricsCollector.get_instance()  # Initialize but don't store reference
    health_system = HealthCheckSystem.get_instance()
    
    # Register a basic system health check
    system_check = SystemResourcesCheck()
    health_system.register_check(system_check)
    
    # Start collectors
    system_collector.start_collection()
    registry.start_reporting()
    health_system.start_checking()
    
    logging.getLogger("finflow.metrics").info("Metrics system initialized")


def write_metrics_to_log(metrics_data: List[Dict[str, Any]]) -> None:
    """
    Write metrics data to log file.
    
    Args:
        metrics_data: List of metric data dictionaries
    """
    logger = logging.getLogger("finflow.metrics.report")
    
    # Group metrics by name
    metrics_by_name: Dict[str, List[Dict[str, Any]]] = {}
    
    for metric in metrics_data:
        name = metric["name"]
        if name not in metrics_by_name:
            metrics_by_name[name] = []
        metrics_by_name[name].append(metric)
    
    # Log a summary of each metric
    for name, points in metrics_by_name.items():
        # For each unique set of labels, find the latest point
        latest_by_labels: Dict[str, Dict[str, Any]] = {}
        
        for point in points:
            labels_key = json.dumps(point["labels"], sort_keys=True)
            
            if labels_key not in latest_by_labels or point["timestamp"] > latest_by_labels[labels_key]["timestamp"]:
                latest_by_labels[labels_key] = point
        
        # Log each unique point
        for point in latest_by_labels.values():
            labels_str = ", ".join(f"{k}={v}" for k, v in point["labels"].items())
            logger.info(f"Metric: {name} = {point['value']} [{labels_str}]")
