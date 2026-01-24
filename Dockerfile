# Multi-stage build for GraphWiz Trader
# Stage 1: Builder with build dependencies
FROM python:3.11-slim AS builder

# Build arguments for metadata
ARG BUILD_DATE
ARG VERSION=0.1.0
LABEL maintainer="GraphWiz Team <info@graphwiz.ai>"
LABEL build-date=$BUILD_DATE
LABEL version=$VERSION
LABEL description="GraphWiz Trader - AI-powered trading system with knowledge graphs"

# Set working directory
WORKDIR /build

# Install build dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        libc-dev \
        git \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# Copy requirements files
COPY requirements.txt requirements-backtesting.txt ./

# Install Python dependencies to /usr/local
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ -f requirements-backtesting.txt ]; then pip install --no-cache-dir -r requirements-backtesting.txt; fi

# Stage 2: Runtime image with minimal footprint and security hardening
FROM python:3.11-slim AS runtime

# Build arguments
ARG BUILD_DATE
ARG VERSION=0.1.0
ARG APP_USER=graphwiz
LABEL maintainer="GraphWiz Team <info@graphwiz.ai>"
LABEL build-date=$BUILD_DATE
LABEL version=$VERSION

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Security hardening
    PYTHONHASHSEED=random \
    # Force stdin, stdout and stderr to be totally unbuffered
    PYTHONUNBUFFERED=1

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r ${APP_USER} && \
    useradd -r -g ${APP_USER} -d /app -s /sbin/nologin -c "GraphWiz Trader user" ${APP_USER}

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Make sure scripts in .local are usable
ENV PATH=/usr/local/bin:$PATH

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/backtests /app/config && \
    chown -R ${APP_USER}:${APP_USER} /app

# Switch to non-root user
USER ${APP_USER}

# Expose ports
EXPOSE 8080 9090 8050

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set up signal handling for graceful shutdown
STOPSIGNAL SIGTERM

# Default command with proper signal handling
CMD ["python", "-u", "-m", "graphwiz_trader.main", "--config", "config/production.yaml"]
