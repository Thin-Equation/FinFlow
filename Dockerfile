FROM python:3.13-alpine

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create non-root user
RUN useradd -m finflow
RUN mkdir -p /var/log/finflow /var/run/finflow /etc/finflow
RUN chown -R finflow:finflow /var/log/finflow /var/run/finflow /etc/finflow

# Set environment variables
ENV PYTHONPATH=/app
ENV FINFLOW_ENV=production

# Set up a data volume for configuration and credentials
VOLUME ["/etc/finflow", "/var/log/finflow"]

# Expose port for API server
EXPOSE 8000

# Switch to non-root user
USER finflow

# Default command
CMD ["python", "main.py", "--mode", "server", "--port", "8000", "--host", "0.0.0.0"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
