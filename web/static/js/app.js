/**
 * FinFlow Web Application
 * Main JavaScript file for handling UI interactions and API communication
 */

class FinFlowApp {
    constructor() {
        this.baseURL = window.location.origin;
        this.currentSection = 'dashboard';
        this.charts = {};
        this.chartsModule = null;
        this.intervalIds = {};
        this.uploading = false;
        this.dragCounter = 0;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupFileUpload();
        this.loadDashboard();
        this.startAutoRefresh();
        this.loadWorkflows();
        
        // Initialize charts module
        if (typeof FinFlowCharts !== 'undefined') {
            this.chartsModule = new FinFlowCharts();
        }
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.target.closest('.nav-link').dataset.section;
                this.navigateToSection(section);
            });
        });

        // Form submissions
        const batchForm = document.getElementById('batch-form');
        if (batchForm) {
            batchForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitBatchProcessing();
            });
        }

        // File upload form
        const uploadForm = document.getElementById('upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitDocumentProcessing();
            });
        }

        // Refresh buttons
        document.querySelectorAll('.refresh-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.refreshCurrentSection();
            });
        });

        // Notification close buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('.notification .close-btn')) {
                e.target.closest('.notification').remove();
            }
        });
    }

    setupFileUpload() {
        const dropZone = document.querySelector('.upload-area');
        const fileInput = document.getElementById('document-file');
        
        if (!dropZone || !fileInput) return;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                this.dragCounter++;
                dropZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                this.dragCounter--;
                if (this.dragCounter === 0) {
                    dropZone.classList.remove('drag-over');
                }
            });
        });

        // Handle dropped files
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                this.updateFileDisplay(files[0]);
            }
        });

        // Handle file input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.updateFileDisplay(e.target.files[0]);
            }
        });

        // Handle click on drop zone
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
    }

    updateFileDisplay(file) {
        const fileName = document.querySelector('.file-name');
        const fileSize = document.querySelector('.file-size');
        const fileInfo = document.getElementById('file-info');
        
        if (fileName) {
            fileName.textContent = file.name;
        }
        
        if (fileSize) {
            fileSize.textContent = this.formatFileSize(file.size);
        }

        if (fileInfo) {
            fileInfo.style.display = 'block';
        }

        document.querySelector('.upload-area').classList.add('has-file');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    navigateToSection(section) {
        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-section="${section}"]`).classList.add('active');

        // Show/hide sections
        document.querySelectorAll('.section').forEach(sec => {
            sec.classList.remove('active');
        });
        document.getElementById(`${section}-section`).classList.add('active');

        this.currentSection = section;

        // Load section-specific data
        switch (section) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'analytics':
                this.loadAnalytics();
                break;
            case 'health':
                this.loadSystemHealth();
                break;
            case 'batch':
                this.loadActiveBatches();
                break;
            case 'workflows':
                this.loadWorkflows();
                break;
        }
    }

    async loadDashboard() {
        try {
            // Load system status
            const statusResponse = await fetch(`${this.baseURL}/status`);
            const status = await statusResponse.json();
            this.updateSystemStatus(status);

            // Load health check
            const healthResponse = await fetch(`${this.baseURL}/health`);
            const health = await healthResponse.json();
            this.updateHealthStatus(health);

            // Load metrics
            const metricsResponse = await fetch(`${this.baseURL}/metrics`);
            const metrics = await metricsResponse.json();
            this.updateDashboardMetrics(metrics);

            // Load active batches
            const batchesResponse = await fetch(`${this.baseURL}/batch/list/active`);
            const batches = await batchesResponse.json();
            this.updateActiveBatches(batches);

        } catch (error) {
            console.error('Error loading dashboard:', error);
            this.showNotification('Error loading dashboard data', 'error');
        }
    }

    updateSystemStatus(status) {
        const statusElement = document.querySelector('.system-status .status-value');
        const timestampElement = document.querySelector('.system-status .timestamp');
        
        if (statusElement) {
            statusElement.textContent = status.status === 'ok' ? 'Online' : 'Offline';
            statusElement.className = `status-value ${status.status === 'ok' ? 'status-healthy' : 'status-error'}`;
        }
        
        if (timestampElement) {
            timestampElement.textContent = `Last updated: ${new Date(status.timestamp).toLocaleString()}`;
        }
    }

    updateHealthStatus(health) {
        const healthElement = document.querySelector('.health-status .status-value');
        
        if (healthElement) {
            const status = health.status || 'unknown';
            healthElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            healthElement.className = `status-value status-${status.toLowerCase()}`;
        }
    }

    updateDashboardMetrics(metrics) {
        // Update processing statistics
        const processedDocs = metrics.metrics?.find(m => m.name === 'documents_submitted')?.value || 0;
        const avgProcessingTime = metrics.metrics?.find(m => m.name === 'document_processing_time')?.value || 0;
        const errorRate = metrics.metrics?.find(m => m.name === 'document_processing_errors')?.value || 0;

        this.updateStatCard('processed-docs', processedDocs);
        this.updateStatCard('avg-processing-time', `${avgProcessingTime.toFixed(2)}s`);
        this.updateStatCard('error-rate', `${errorRate}%`);
    }

    updateStatCard(cardId, value) {
        const element = document.getElementById(cardId);
        if (element) {
            element.textContent = value;
        }
    }

    updateActiveBatches(batches) {
        const container = document.querySelector('.active-workflows .workflow-list');
        if (!container) return;

        container.innerHTML = '';
        
        if (batches.active_batches && batches.active_batches.length > 0) {
            batches.active_batches.forEach(batch => {
                const batchElement = document.createElement('div');
                batchElement.className = 'workflow-item';
                batchElement.innerHTML = `
                    <div class="workflow-info">
                        <div class="workflow-name">Batch ${batch}</div>
                        <div class="workflow-status">Processing</div>
                    </div>
                    <div class="workflow-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 50%"></div>
                        </div>
                    </div>
                `;
                container.appendChild(batchElement);
            });
        } else {
            container.innerHTML = '<div class="no-data">No active workflows</div>';
        }
    }

    async loadWorkflows() {
        try {
            const response = await fetch(`${this.baseURL}/workflows`);
            const data = await response.json();
            this.updateWorkflowSelect(data.workflows);
            this.updateWorkflowsList(data.workflows);
        } catch (error) {
            console.error('Error loading workflows:', error);
            this.showNotification('Error loading workflows', 'error');
        }
    }

    updateWorkflowSelect(workflows) {
        const select = document.getElementById('workflow-type');
        if (!select) return;

        select.innerHTML = '';
        workflows.forEach(workflow => {
            const option = document.createElement('option');
            option.value = workflow.id;
            option.textContent = workflow.name;
            select.appendChild(option);
        });
    }

    updateWorkflowsList(workflows) {
        const container = document.querySelector('.workflows-grid');
        if (!container) return;

        container.innerHTML = '';
        workflows.forEach(workflow => {
            const workflowCard = document.createElement('div');
            workflowCard.className = 'workflow-card';
            workflowCard.innerHTML = `
                <h4>${workflow.name}</h4>
                <p>${workflow.description}</p>
                <div class="workflow-actions">
                    <button class="btn btn-primary" onclick="app.selectWorkflow('${workflow.id}')">
                        Select
                    </button>
                </div>
            `;
            container.appendChild(workflowCard);
        });
    }

    selectWorkflow(workflowId) {
        const select = document.getElementById('workflow-type');
        if (select) {
            select.value = workflowId;
        }
        this.navigateToSection('process');
        this.showNotification(`Selected ${workflowId} workflow`, 'success');
    }

    async submitDocumentProcessing() {
        if (this.uploading) return;

        const fileInput = document.getElementById('document-file');
        const workflowSelect = document.getElementById('workflow-type');
        
        if (!fileInput.files[0]) {
            this.showNotification('Please select a file to upload', 'error');
            return;
        }

        this.uploading = true;
        this.showLoading(true);

        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const requestData = {
                workflow_type: workflowSelect.value || 'standard',
                options: {}
            };
            formData.append('request', JSON.stringify(requestData));

            const response = await fetch(`${this.baseURL}/process`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showProcessingResult(result);
            this.showNotification('Document processed successfully!', 'success');
            
            // Reset form
            fileInput.value = '';
            document.querySelector('.upload-area').classList.remove('has-file');
            const fileInfo = document.getElementById('file-info');
            if (fileInfo) {
                fileInfo.style.display = 'none';
            }
            document.querySelector('.file-name').textContent = '';
            document.querySelector('.file-size').textContent = '';

        } catch (error) {
            console.error('Error processing document:', error);
            this.showNotification('Error processing document: ' + error.message, 'error');
        } finally {
            this.uploading = false;
            this.showLoading(false);
        }
    }

    showProcessingResult(result) {
        const resultsContainer = document.querySelector('.processing-results');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = `
            <div class="result-card">
                <h4>Processing Complete</h4>
                <div class="result-details">
                    <p><strong>Document ID:</strong> ${result.document_id}</p>
                    <p><strong>Status:</strong> ${result.status}</p>
                    <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s</p>
                    <p><strong>Timestamp:</strong> ${new Date(result.timestamp).toLocaleString()}</p>
                </div>
                <div class="result-data">
                    <h5>Extracted Data:</h5>
                    <pre>${JSON.stringify(result.result, null, 2)}</pre>
                </div>
            </div>
        `;
        resultsContainer.style.display = 'block';
    }

    async submitBatchProcessing() {
        const form = document.getElementById('batch-form');
        const formData = new FormData(form);
        
        const batchRequest = {
            directory_path: formData.get('directory_path'),
            workflow_type: formData.get('workflow_type') || 'optimized',
            parallel: formData.get('parallel') === 'on',
            adaptive_workers: formData.get('adaptive_workers') === 'on',
            max_workers: parseInt(formData.get('max_workers')) || 4,
            output_directory: formData.get('output_directory') || null
        };

        try {
            const response = await fetch(`${this.baseURL}/batch/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(batchRequest)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showNotification(`Batch processing started: ${result.batch_id}`, 'success');
            this.addBatchToMonitoring(result.batch_id);
            
        } catch (error) {
            console.error('Error starting batch processing:', error);
            this.showNotification('Error starting batch processing: ' + error.message, 'error');
        }
    }

    addBatchToMonitoring(batchId) {
        const container = document.querySelector('.batch-status');
        if (!container) return;

        const batchElement = document.createElement('div');
        batchElement.className = 'batch-item';
        batchElement.id = `batch-${batchId}`;
        batchElement.innerHTML = `
            <div class="batch-header">
                <h4>Batch ${batchId}</h4>
                <span class="batch-status-badge">Starting...</span>
            </div>
            <div class="batch-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="progress-text">0%</div>
            </div>
            <div class="batch-details">
                <span>Processed: <span class="processed-count">0</span></span>
                <span>Failed: <span class="failed-count">0</span></span>
                <span>Total: <span class="total-count">-</span></span>
            </div>
        `;
        container.appendChild(batchElement);

        // Start monitoring this batch
        this.monitorBatch(batchId);
    }

    async monitorBatch(batchId) {
        const intervalId = setInterval(async () => {
            try {
                const response = await fetch(`${this.baseURL}/batch/${batchId}/status`);
                
                if (response.status === 404) {
                    // Batch completed or not found
                    clearInterval(intervalId);
                    this.updateBatchStatus(batchId, { status: 'completed' });
                    return;
                }

                const status = await response.json();
                this.updateBatchStatus(batchId, status);

                if (status.status === 'completed' || status.status === 'failed') {
                    clearInterval(intervalId);
                }

            } catch (error) {
                console.error(`Error monitoring batch ${batchId}:`, error);
            }
        }, 2000); // Update every 2 seconds

        this.intervalIds[batchId] = intervalId;
    }

    updateBatchStatus(batchId, status) {
        const batchElement = document.getElementById(`batch-${batchId}`);
        if (!batchElement) return;

        const statusBadge = batchElement.querySelector('.batch-status-badge');
        const progressFill = batchElement.querySelector('.progress-fill');
        const progressText = batchElement.querySelector('.progress-text');
        const processedCount = batchElement.querySelector('.processed-count');
        const failedCount = batchElement.querySelector('.failed-count');
        const totalCount = batchElement.querySelector('.total-count');

        if (statusBadge) {
            statusBadge.textContent = status.status;
            statusBadge.className = `batch-status-badge status-${status.status}`;
        }

        if (status.progress_percent !== undefined) {
            if (progressFill) {
                progressFill.style.width = `${status.progress_percent}%`;
            }
            if (progressText) {
                progressText.textContent = `${Math.round(status.progress_percent)}%`;
            }
        }

        if (processedCount && status.processed !== undefined) {
            processedCount.textContent = status.processed;
        }
        if (failedCount && status.failed !== undefined) {
            failedCount.textContent = status.failed;
        }
        if (totalCount && status.total !== undefined) {
            totalCount.textContent = status.total;
        }
    }

    async loadActiveBatches() {
        try {
            const response = await fetch(`${this.baseURL}/batch/list/active`);
            const batches = await response.json();
            
            batches.active_batches.forEach(batchId => {
                if (!document.getElementById(`batch-${batchId}`)) {
                    this.addBatchToMonitoring(batchId);
                }
            });
        } catch (error) {
            console.error('Error loading active batches:', error);
        }
    }

    async loadSystemHealth() {
        try {
            const [healthResponse, diagnosticsResponse] = await Promise.all([
                fetch(`${this.baseURL}/health`),
                fetch(`${this.baseURL}/diagnostics`)
            ]);

            const health = await healthResponse.json();
            const diagnostics = await diagnosticsResponse.json();

            this.updateSystemHealthDisplay(health, diagnostics);
        } catch (error) {
            console.error('Error loading system health:', error);
            this.showNotification('Error loading system health data', 'error');
        }
    }

    updateSystemHealthDisplay(health, diagnostics) {
        // Update health overview
        const healthStatus = document.querySelector('.health-overview .status-value');
        if (healthStatus) {
            healthStatus.textContent = health.status || 'Unknown';
            healthStatus.className = `status-value status-${health.status?.toLowerCase() || 'unknown'}`;
        }

        // Update system metrics
        if (diagnostics.system) {
            this.updateMetricValue('cpu-usage', `${diagnostics.system.cpu_percent.toFixed(1)}%`);
            this.updateMetricValue('memory-usage', `${diagnostics.system.memory_percent.toFixed(1)}%`);
            this.updateMetricValue('disk-usage', `${diagnostics.system.disk_percent.toFixed(1)}%`);
        }

        // Update process info
        if (diagnostics.process) {
            this.updateMetricValue('process-memory', `${diagnostics.process.memory_percent.toFixed(2)}%`);
            this.updateMetricValue('process-threads', diagnostics.process.threads);
            this.updateMetricValue('uptime', this.formatUptime(diagnostics.process.uptime_seconds));
        }
    }

    updateMetricValue(metricId, value) {
        const element = document.getElementById(metricId);
        if (element) {
            element.textContent = value;
        }
    }

    formatUptime(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        
        if (days > 0) {
            return `${days}d ${hours}h ${minutes}m`;
        } else if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    }

    async loadAnalytics() {
        try {
            // Initialize charts if not already done
            if (this.chartsModule && Object.keys(this.chartsModule.charts).length === 0) {
                this.chartsModule.initializeCharts();
            }
            
            const response = await fetch(`${this.baseURL}/metrics`);
            const metrics = await response.json();
            
            // Update charts with real data
            if (this.chartsModule) {
                this.chartsModule.updateChartsWithRealData(metrics);
            }
            
            this.updateAnalyticsDisplay(metrics);
        } catch (error) {
            console.error('Error loading analytics:', error);
            this.showNotification('Error loading analytics data', 'error');
        }
    }

    updateAnalyticsDisplay(metrics) {
        // Update any additional analytics displays beyond charts
        console.log('Analytics metrics loaded:', metrics);
    }

    showNotification(message, type = 'info') {
        const notificationContainer = document.querySelector('.notifications') || this.createNotificationContainer();
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
            <button class="close-btn">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        notificationContainer.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    createNotificationContainer() {
        const container = document.createElement('div');
        container.className = 'notifications';
        document.body.appendChild(container);
        return container;
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    showLoading(show) {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    }

    startAutoRefresh() {
        // Refresh dashboard every 30 seconds
        this.intervalIds.dashboard = setInterval(() => {
            if (this.currentSection === 'dashboard') {
                this.loadDashboard();
            }
        }, 30000);

        // Refresh health every 10 seconds
        this.intervalIds.health = setInterval(() => {
            if (this.currentSection === 'health') {
                this.loadSystemHealth();
            }
        }, 10000);
    }

    refreshCurrentSection() {
        switch (this.currentSection) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'health':
                this.loadSystemHealth();
                break;
            case 'analytics':
                this.loadAnalytics();
                break;
            case 'batch':
                this.loadActiveBatches();
                break;
            case 'workflows':
                this.loadWorkflows();
                break;
        }
        this.showNotification('Data refreshed', 'success');
    }

    // Clean up intervals when page unloads
    cleanup() {
        Object.values(this.intervalIds).forEach(intervalId => {
            clearInterval(intervalId);
        });
        
        // Clean up charts
        if (this.chartsModule) {
            this.chartsModule.destroyCharts();
        }
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FinFlowApp();
    
    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        window.app.cleanup();
    });
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FinFlowApp;
}
