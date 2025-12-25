#!/bin/bash
# Test runner script for graphwiz-trader

set -e

echo "Building Docker image for testing..."
docker build -t graphwiz-trader-test .

echo ""
echo "Running tests..."
docker run --rm graphwiz-trader-test pytest tests/ -v --tb=short --cov=src/graphwiz_trader --cov-report=term-missing --cov-report=xml

echo ""
echo "Tests completed!"
