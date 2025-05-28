/**
 * FinFlow Charts Module
 * Handles Chart.js integration for analytics visualization
 */

class FinFlowCharts {
    constructor() {
        this.charts = {};
        this.chartColors = {
            primary: '#6366f1',
            secondary: '#8b5cf6',
            success: '#10b981',
            warning: '#f59e0b',
            error: '#ef4444',
            info: '#3b82f6'
        };
    }

    initializeCharts() {
        // Initialize all charts when the analytics section is loaded
        this.createProcessingTimeChart();
        this.createDocumentTypeChart();
        this.createErrorRateChart();
        this.createThroughputChart();
    }

    createProcessingTimeChart() {
        const ctx = document.getElementById('processing-time-chart');
        if (!ctx) return;

        this.charts.processingTime = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.generateTimeLabels(24), // Last 24 hours
                datasets: [{
                    label: 'Average Processing Time (seconds)',
                    data: this.generateSampleData(24, 1, 10),
                    borderColor: this.chartColors.primary,
                    backgroundColor: this.chartColors.primary + '20',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Document Processing Time Trends'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
    }

    createDocumentTypeChart() {
        const ctx = document.getElementById('document-type-chart');
        if (!ctx) return;

        this.charts.documentType = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Invoices', 'Receipts', 'Bank Statements', 'Reports', 'Other'],
                datasets: [{
                    data: [35, 25, 20, 15, 5],
                    backgroundColor: [
                        this.chartColors.primary,
                        this.chartColors.secondary,
                        this.chartColors.success,
                        this.chartColors.warning,
                        this.chartColors.info
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Document Types Distribution'
                    },
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    createErrorRateChart() {
        const ctx = document.getElementById('error-rate-chart');
        if (!ctx) return;

        this.charts.errorRate = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.generateDateLabels(7), // Last 7 days
                datasets: [
                    {
                        label: 'Successful',
                        data: this.generateSampleData(7, 80, 150),
                        backgroundColor: this.chartColors.success,
                        stack: 'stack1'
                    },
                    {
                        label: 'Failed',
                        data: this.generateSampleData(7, 0, 10),
                        backgroundColor: this.chartColors.error,
                        stack: 'stack1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Processing Success vs Error Rate'
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Documents'
                        }
                    }
                }
            }
        });
    }

    createThroughputChart() {
        const ctx = document.getElementById('throughput-chart');
        if (!ctx) return;

        this.charts.throughput = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.generateTimeLabels(12), // Last 12 hours
                datasets: [{
                    label: 'Documents per Hour',
                    data: this.generateSampleData(12, 10, 50),
                    borderColor: this.chartColors.secondary,
                    backgroundColor: this.chartColors.secondary + '20',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Processing Throughput'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Documents/Hour'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
    }

    updateChartsWithRealData(metrics) {
        // Update charts with real metrics data from the API
        if (metrics && metrics.metrics) {
            this.updateProcessingTimeChart(metrics.metrics);
            this.updateThroughputChart(metrics.metrics);
            // Add more update methods as needed
        }
    }

    updateProcessingTimeChart(metrics) {
        const processingTimeMetric = metrics.find(m => m.name === 'document_processing_time');
        if (processingTimeMetric && this.charts.processingTime) {
            // In a real implementation, you would process historical data
            // For now, we'll add the current value to the chart
            const chart = this.charts.processingTime;
            const currentTime = new Date().toLocaleTimeString();
            
            // Add new data point
            chart.data.labels.push(currentTime);
            chart.data.datasets[0].data.push(processingTimeMetric.value);
            
            // Keep only last 24 data points
            if (chart.data.labels.length > 24) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update('none'); // Update without animation for real-time feel
        }
    }

    updateThroughputChart(metrics) {
        const throughputMetric = metrics.find(m => m.name === 'documents_submitted');
        if (throughputMetric && this.charts.throughput) {
            // Similar update logic for throughput
            const chart = this.charts.throughput;
            const currentHour = new Date().getHours();
            
            // Update current hour's data
            const lastIndex = chart.data.datasets[0].data.length - 1;
            chart.data.datasets[0].data[lastIndex] = throughputMetric.value;
            
            chart.update('none');
        }
    }

    generateTimeLabels(hours) {
        const labels = [];
        const now = new Date();
        
        for (let i = hours - 1; i >= 0; i--) {
            const time = new Date(now.getTime() - (i * 60 * 60 * 1000));
            labels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
        }
        
        return labels;
    }

    generateDateLabels(days) {
        const labels = [];
        const now = new Date();
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date(now.getTime() - (i * 24 * 60 * 60 * 1000));
            labels.push(date.toLocaleDateString([], { month: 'short', day: 'numeric' }));
        }
        
        return labels;
    }

    generateSampleData(count, min, max) {
        const data = [];
        for (let i = 0; i < count; i++) {
            data.push(Math.floor(Math.random() * (max - min + 1)) + min);
        }
        return data;
    }

    destroyCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};
    }

    resizeCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
}

// Export for use in main app
if (typeof window !== 'undefined') {
    window.FinFlowCharts = FinFlowCharts;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = FinFlowCharts;
}
