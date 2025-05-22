"""
System initialization for performance and robustness features.

This module initializes:
1. Error handling system
2. Metrics collection
3. Health monitoring
4. Recovery mechanisms
"""

import logging
import psutil
from typing import Any, Dict, Optional

from utils.error_handling import (
    ErrorManager, FeatureFlag, FinFlowError, ErrorSeverity
)
from utils.metrics import (
    MetricsRegistry, SystemMetricsCollector, AppMetricsCollector,
    HealthCheckSystem, SystemResourcesCheck, write_metrics_to_log
)
from config.config_loader import load_config


def initialize_error_handling() -> ErrorManager:
    """
    Initialize the error handling system.
    
    Returns:
        The error manager instance
    """
    logger = logging.getLogger("finflow.system")
    logger.info("Initializing error handling system")
    
    # Get error manager singleton
    error_manager = ErrorManager.get_instance()
    
    # Register error handlers for different types
    def log_critical_error(error: FinFlowError) -> None:
        """Log critical errors and potentially send alerts."""
        if error.severity in (ErrorSeverity.HIGH, ErrorSeverity.FATAL):
            logger.critical(
                f"CRITICAL ERROR: {error.error_code} - {error.message}"
            )
            # In a production system, this would trigger alerts
    
    # Register global error handlers
    error_manager.subscribe(log_critical_error)
    
    # Register specific handlers for errors with retry logic
    def handle_document_error(error: FinFlowError) -> None:
        """Special handling for document processing errors."""
        logger.warning(f"Document processing error detected: {error.message}")
        # In production, this could queue the document for reprocessing
    
    error_manager.register_handler("ERR_DOC_PROC", handle_document_error)
    
    logger.info("Error handling system initialized")
    return error_manager


def initialize_metrics(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize the metrics and monitoring system.
    
    Args:
        config: Application configuration
        
    Returns:
        Dictionary containing metrics components
    """
    logger = logging.getLogger("finflow.system")
    logger.info("Initializing metrics and monitoring system")
    
    # Get registry and collector singletons
    registry = MetricsRegistry.get_instance()
    system_collector = SystemMetricsCollector.get_instance()
    app_collector = AppMetricsCollector.get_instance()
    health_system = HealthCheckSystem.get_instance()
    
    # Configure from config
    metrics_config = config.get("metrics", {})
    collection_interval = metrics_config.get("collection_interval", 30.0)
    reporting_interval = metrics_config.get("reporting_interval", 60.0)
    
    # Register logging reporter
    registry.add_report_callback(write_metrics_to_log)
    
    # Register health checks
    system_check = SystemResourcesCheck(
        cpu_threshold=metrics_config.get("cpu_threshold", 80.0),
        memory_threshold=metrics_config.get("memory_threshold", 80.0),
        disk_threshold=metrics_config.get("disk_threshold", 80.0)
    )
    health_system.register_check(system_check)
    
    # Start collectors and reporters
    system_collector.start_collection(interval=collection_interval)
    registry.start_reporting(interval=reporting_interval)
    health_system.start_checking(interval=collection_interval)
    
    logger.info(
        f"Metrics system initialized with collection interval {collection_interval}s "
        f"and reporting interval {reporting_interval}s"
    )
    
    return {
        "registry": registry,
        "system_collector": system_collector,
        "app_collector": app_collector,
        "health_system": health_system
    }


def initialize_feature_flags(config: Dict[str, Any]) -> FeatureFlag:
    """
    Initialize feature flags for graceful degradation.
    
    Args:
        config: Application configuration
        
    Returns:
        Feature flag manager instance
    """
    logger = logging.getLogger("finflow.system")
    logger.info("Initializing feature flags")
    
    feature_flags = FeatureFlag.get_instance()
    
    # Register feature flags with defaults
    feature_flags.register_flag("advanced_analytics", True)
    feature_flags.register_flag("document_ai_enhanced", True)
    feature_flags.register_flag("batch_processing", True)
    feature_flags.register_flag("auto_validation", True)
    feature_flags.register_flag("auto_recovery", True)
    
    # Apply any config overrides
    ff_config = config.get("feature_flags", {})
    for flag_name, enabled in ff_config.items():
        if enabled:
            feature_flags.enable(flag_name)
        else:
            feature_flags.disable(flag_name)
    
    logger.info(f"Feature flags initialized: {feature_flags.__dict__['flags']}")
    return feature_flags


def register_health_checks(health_system: HealthCheckSystem) -> None:
    """
    Register additional health checks.
    
    Args:
        health_system: Health check system
    """
    # This would be extended in a real system to check connectivity to
    # external services, database health, etc.
    pass


def setup_process_monitoring() -> None:
    """Set up monitoring of the current process."""
    try:
        # Get current process
        process = psutil.Process()
        
        # Set up CPU and memory monitoring
        cpu_percent = process.cpu_percent(interval=0.5)
        memory_info = process.memory_info()
        
        logger = logging.getLogger("finflow.system")
        logger.info(f"Process monitoring initialized. Initial CPU: {cpu_percent}%, Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        logger = logging.getLogger("finflow.system")
        logger.warning(f"Failed to set up process monitoring: {e}")


def initialize_robustness_systems(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize all robustness and performance systems.
    
    Args:
        config: Optional application configuration (will load if not provided)
        
    Returns:
        Dictionary containing initialized components
    """
    if config is None:
        config = load_config()
        
    # Get environment
    env = config.get("environment", "development")
    
    # Initialize systems
    error_manager = initialize_error_handling()
    feature_flags = initialize_feature_flags(config)
    metrics = initialize_metrics(config)
    
    # Register additional health checks
    register_health_checks(metrics["health_system"])
    
    # Set up process monitoring
    setup_process_monitoring()
    
    # Log initialization
    logger = logging.getLogger("finflow.system")
    logger.info(f"Robustness systems initialized in {env} environment")
    
    return {
        "error_manager": error_manager,
        "feature_flags": feature_flags,
        "metrics": metrics
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize systems
    initialize_robustness_systems()
