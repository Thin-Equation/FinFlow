"""
Health check system for the FinFlow application.

This module provides:
1. Service health checks
2. Component status monitoring
3. System diagnostics
4. Dependency health checks
"""

import logging
import time
import threading
import json
import os
import sys
import psutil
import socket
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import concurrent.futures

from utils.metrics import AppMetricsCollector, Gauge

# Create module logger
logger = logging.getLogger(__name__)

# Health status enum
class HealthStatus(str, Enum):
    """Health status values."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceCheck:
    """Defines a service check with health check function."""
    
    def __init__(
        self,
        name: str,
        check_function: Callable[[], Dict[str, Any]],
        dependencies: List[str] = None,
        critical: bool = False,
        interval_seconds: int = 60
    ):
        """Initialize a service check.
        
        Args:
            name: Name of the service check
            check_function: Function to perform the check
            dependencies: Optional list of dependency names
            critical: Whether this service is critical for system operation
            interval_seconds: How often to run this check (in seconds)
        """
        self.name = name
        self.check_function = check_function
        self.dependencies = dependencies or []
        self.critical = critical
        self.interval_seconds = interval_seconds
        self.last_check_time = 0
        self.last_status = HealthStatus.UNKNOWN
        self.last_result: Optional[Dict[str, Any]] = None
    
    def should_check(self) -> bool:
        """Check if it's time to run this check again."""
        return time.time() - self.last_check_time >= self.interval_seconds
    
    def run_check(self) -> Dict[str, Any]:
        """Run the health check and record results."""
        try:
            start_time = time.time()
            
            # Run the check function
            result = self.check_function()
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update status based on result
            status = result.get("status", HealthStatus.UNKNOWN)
            if isinstance(status, str):
                status = HealthStatus(status)
                
            self.last_status = status
            self.last_check_time = time.time()
            
            # Create full result
            full_result = {
                "name": self.name,
                "status": status,
                "response_time_ms": round(response_time * 1000, 2),
                "timestamp": self.last_check_time,
                "critical": self.critical,
                **result
            }
            
            self.last_result = full_result
            return full_result
            
        except Exception as e:
            # Handle check failure
            self.last_status = HealthStatus.UNHEALTHY
            self.last_check_time = time.time()
            
            result = {
                "name": self.name,
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": self.last_check_time,
                "critical": self.critical
            }
            
            self.last_result = result
            return result


class HealthCheckManager:
    """
    Manager for health checks of FinFlow system.
    
    Provides:
    1. Service status monitoring
    2. System diagnostics
    3. Dependency health checks
    """
    
    # Singleton instance
    _instance: Optional['HealthCheckManager'] = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls, config: Dict[str, Any] = None) -> 'HealthCheckManager':
        """Get the singleton instance of HealthCheckManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config or {})
            return cls._instance
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the health check manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("finflow.health")
        
        # Initialize metrics collector
        self.metrics = AppMetricsCollector.get_instance()
        
        # Define health status gauge
        self.health_gauge = Gauge("service_health")
        
        # Service checks
        self.checks: Dict[str, ServiceCheck] = {}
        
        # Background thread for checks
        self.check_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Running status
        self.is_running = False
        self.check_interval = config.get("check_interval_seconds", 60)
        
        # Last overall health status
        self.last_health_status = HealthStatus.UNKNOWN
        
        # Set up system checks
        self._setup_system_checks()
        
        # Configure health output file
        self.health_output_file = config.get(
            "health_output_file", 
            os.path.join(os.getcwd(), "health_status.json")
        )
    
    def _setup_system_checks(self) -> None:
        """Set up standard system health checks."""
        # CPU check
        self.register_check(
            name="system.cpu",
            check_function=self._check_cpu,
            critical=False,
            interval_seconds=30
        )
        
        # Memory check
        self.register_check(
            name="system.memory",
            check_function=self._check_memory,
            critical=False,
            interval_seconds=30
        )
        
        # Disk check
        self.register_check(
            name="system.disk",
            check_function=self._check_disk,
            critical=True,
            interval_seconds=60
        )
        
        # Python runtime check
        self.register_check(
            name="system.python_runtime",
            check_function=self._check_python_runtime,
            critical=False,
            interval_seconds=60
        )
        
        # Network connectivity check
        self.register_check(
            name="system.network",
            check_function=self._check_network,
            critical=True,
            interval_seconds=30
        )
    
    def register_check(
        self, 
        name: str,
        check_function: Callable[[], Dict[str, Any]],
        dependencies: List[str] = None,
        critical: bool = False,
        interval_seconds: int = 60
    ) -> None:
        """Register a service check.
        
        Args:
            name: Name of the check
            check_function: Function to perform the check
            dependencies: List of dependency names
            critical: Whether this check is critical
            interval_seconds: How often to run this check
        """
        self.checks[name] = ServiceCheck(
            name=name,
            check_function=check_function,
            dependencies=dependencies,
            critical=critical,
            interval_seconds=interval_seconds
        )
        self.logger.debug(f"Registered health check: {name}")
    
    def start(self) -> None:
        """Start the health check system."""
        if self.is_running:
            return
            
        self.is_running = True
        self.stop_event.clear()
        
        # Start background thread
        self.check_thread = threading.Thread(
            target=self._check_loop,
            daemon=True,
            name="HealthCheckManager"
        )
        self.check_thread.start()
        
        self.logger.info("Health check system started")
    
    def stop(self) -> None:
        """Stop the health check system."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        if self.check_thread:
            self.check_thread.join(timeout=5.0)
            self.check_thread = None
            
        self.logger.info("Health check system stopped")
    
    def _check_loop(self) -> None:
        """Main health check loop."""
        while not self.stop_event.is_set():
            try:
                self._run_checks()
                
                # Sleep until next check interval
                self.stop_event.wait(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                
                # Sleep briefly before retrying
                time.sleep(5.0)
    
    def _run_checks(self) -> None:
        """Run all due health checks."""
        # Determine which checks to run
        checks_to_run = []
        for name, check in self.checks.items():
            if check.should_check():
                checks_to_run.append(check)
        
        if not checks_to_run:
            return
            
        self.logger.debug(f"Running {len(checks_to_run)} health checks")
        
        # Run checks in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_check = {executor.submit(check.run_check): check for check in checks_to_run}
            
            for future in concurrent.futures.as_completed(future_to_check):
                check = future_to_check[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update metrics
                    self._update_check_metrics(result)
                    
                except Exception as e:
                    self.logger.error(f"Error running health check {check.name}: {e}")
        
        # Calculate overall system health
        self._calculate_overall_health()
        
        # Write health status to file
        self._write_health_status()
    
    def _update_check_metrics(self, result: Dict[str, Any]) -> None:
        """Update metrics for a check result."""
        status = result["status"]
        name = result["name"]
        response_time = result.get("response_time_ms", 0)
        
        # Convert status to numeric value for gauge
        status_value = 1.0  # Healthy
        if status == HealthStatus.DEGRADED:
            status_value = 0.5
        elif status == HealthStatus.UNHEALTHY:
            status_value = 0.0
        elif status == HealthStatus.UNKNOWN:
            status_value = -1.0
        
        # Update gauge
        self.health_gauge.set(status_value, labels={"check": name})
        
        # Record response time
        self.metrics.histogram(
            "health_check_response_time_ms",
            response_time,
            labels={"check": name}
        )
    
    def _calculate_overall_health(self) -> HealthStatus:
        """Calculate overall system health based on check results."""
        # Default to unknown
        overall_status = HealthStatus.UNKNOWN
        
        # Check if we have any results
        if not self.checks:
            return overall_status
            
        # Count checks by status
        critical_unhealthy = False
        critical_degraded = False
        has_degraded = False
        has_unhealthy = False
        total_checks = len(self.checks)
        unknown_checks = 0
        
        for name, check in self.checks.items():
            status = check.last_status
            
            if status == HealthStatus.UNKNOWN:
                unknown_checks += 1
            elif status == HealthStatus.UNHEALTHY:
                has_unhealthy = True
                if check.critical:
                    critical_unhealthy = True
            elif status == HealthStatus.DEGRADED:
                has_degraded = True
                if check.critical:
                    critical_degraded = True
        
        # Determine overall status
        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif critical_degraded:
            overall_status = HealthStatus.DEGRADED
        elif has_unhealthy and has_degraded:
            overall_status = HealthStatus.DEGRADED
        elif has_unhealthy:
            # Non-critical unhealthy checks
            overall_status = HealthStatus.DEGRADED
        elif has_degraded:
            overall_status = HealthStatus.DEGRADED
        elif unknown_checks == total_checks:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Update metrics if status changed
        if overall_status != self.last_health_status:
            self.logger.info(f"System health changed from {self.last_health_status} to {overall_status}")
            
            self.metrics.gauge("overall_system_health").set(
                1.0 if overall_status == HealthStatus.HEALTHY else
                0.5 if overall_status == HealthStatus.DEGRADED else
                0.0 if overall_status == HealthStatus.UNHEALTHY else 
                -1.0,  # UNKNOWN
                labels={}
            )
            
            # Record status change event
            self.metrics.counter("health_status_changes").increment()
            
        # Store current status
        self.last_health_status = overall_status
        
        return overall_status
    
    def _write_health_status(self) -> None:
        """Write health status to a JSON file."""
        try:
            # Build complete status
            status = {
                "timestamp": datetime.now().isoformat(),
                "status": self.last_health_status,
                "checks": {}
            }
            
            # Add individual checks
            for name, check in self.checks.items():
                if check.last_result:
                    status["checks"][name] = check.last_result
                else:
                    status["checks"][name] = {
                        "name": name,
                        "status": check.last_status,
                        "timestamp": check.last_check_time
                    }
            
            # Write to file
            with open(self.health_output_file, "w") as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error writing health status: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status.
        
        Returns:
            Dict[str, Any]: Current health status
        """
        # Calculate current health
        overall_status = self._calculate_overall_health()
        
        # Build status report
        status = {
            "timestamp": datetime.now().isoformat(),
            "status": overall_status,
            "checks": {}
        }
        
        # Add individual checks
        for name, check in self.checks.items():
            if check.last_result:
                status["checks"][name] = check.last_result
            else:
                status["checks"][name] = {
                    "name": name,
                    "status": check.last_status,
                    "timestamp": check.last_check_time or 0
                }
        
        return status
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check on demand.
        
        Args:
            name: Name of the check to run
            
        Returns:
            Dict[str, Any]: Check result
        """
        if name not in self.checks:
            raise ValueError(f"Check not found: {name}")
            
        check = self.checks[name]
        return check.run_check()
    
    # System check implementations
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            # Get CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
            
            # Determine status based on CPU usage
            status = HealthStatus.HEALTHY
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
            
            result = {
                "status": status,
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
            }
            
            if load_avg:
                result["load_avg"] = load_avg
                
            return result
            
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "error": str(e)
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            # Get memory metrics
            memory = psutil.virtual_memory()
            
            # Determine status based on memory usage
            status = HealthStatus.HEALTHY
            if memory.percent > 95:
                status = HealthStatus.UNHEALTHY
            elif memory.percent > 85:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "memory_total_mb": memory.total / (1024 * 1024)
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "error": str(e)
            }
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            # Get current working directory
            cwd = os.getcwd()
            
            # Get disk usage for the current directory
            disk_usage = psutil.disk_usage(cwd)
            
            # Determine status based on disk usage
            status = HealthStatus.HEALTHY
            if disk_usage.percent > 95:
                status = HealthStatus.UNHEALTHY
            elif disk_usage.percent > 85:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status,
                "disk_percent": disk_usage.percent,
                "disk_free_gb": disk_usage.free / (1024 * 1024 * 1024),
                "disk_total_gb": disk_usage.total / (1024 * 1024 * 1024),
                "path": cwd
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "error": str(e)
            }
    
    def _check_python_runtime(self) -> Dict[str, Any]:
        """Check Python runtime health."""
        try:
            # Basic Python runtime info
            python_version = sys.version
            python_implementation = sys.implementation.name
            
            return {
                "status": HealthStatus.HEALTHY,
                "python_version": python_version,
                "python_implementation": python_implementation,
                "executable": sys.executable
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "error": str(e)
            }
    
    def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            # Try to connect to google.com
            host = "google.com"
            port = 80
            
            # Measure connection time
            start_time = time.time()
            
            # Create socket and connect
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            
            try:
                sock.connect((host, port))
                connected = True
            except Exception:
                connected = False
            finally:
                sock.close()
            
            connect_time = time.time() - start_time
            
            status = HealthStatus.HEALTHY if connected else HealthStatus.UNHEALTHY
            
            return {
                "status": status,
                "connected": connected,
                "connect_time_ms": round(connect_time * 1000, 2),
                "host": host
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "error": str(e)
            }


def create_agent_check(agent_name: str) -> Callable[[], Dict[str, Any]]:
    """Create a health check function for an agent.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Callable: Health check function
    """
    def check_agent() -> Dict[str, Any]:
        """Check if agent is available and responsive."""
        try:
            # This would normally check agent health through some API
            # For now, just return healthy
            return {
                "status": HealthStatus.HEALTHY,
                "agent": agent_name
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "agent": agent_name,
                "error": str(e)
            }
    
    return check_agent


def register_agent_checks(health_manager: HealthCheckManager, agent_names: List[str]) -> None:
    """Register health checks for all specified agents.
    
    Args:
        health_manager: The health check manager
        agent_names: List of agent names
    """
    for agent_name in agent_names:
        check_name = f"agent.{agent_name}"
        check_function = create_agent_check(agent_name)
        
        health_manager.register_check(
            name=check_name,
            check_function=check_function,
            critical=True,
            interval_seconds=30
        )
