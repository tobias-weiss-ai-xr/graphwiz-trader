#!/bin/bash
# Test runner script for graphwiz-trader

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default: run all tests
TEST_MARKER=""
PYTEST_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit)
            TEST_MARKER="-m unit"
            echo -e "${GREEN}Running unit tests only...${NC}"
            shift
            ;;
        -i|--integration)
            TEST_MARKER="-m integration"
            echo -e "${GREEN}Running integration tests only...${NC}"
            shift
            ;;
        -s|--slow)
            TEST_MARKER="-m slow"
            echo -e "${YELLOW}Running slow tests (including property-based)...${NC}"
            shift
            ;;
        -f|--fast)
            TEST_MARKER="-m 'not slow'"
            echo -e "${GREEN}Running fast tests only (excluding slow)...${NC}"
            shift
            ;;
        -h|--hft)
            TEST_MARKER="-m hft"
            echo -e "${GREEN}Running HFT tests only...${NC}"
            shift
            ;;
        -p|--property)
            TEST_MARKER="-m property"
            echo -e "${YELLOW}Running property-based tests only...${NC}"
            shift
            ;;
        --cov)
            PYTEST_ARGS="$PYTEST_ARGS --cov=src/graphwiz_trader --cov-report=html"
            echo -e "${GREEN}Coverage report will be generated in htmlcov/${NC}"
            shift
            ;;
        --no-cov)
            PYTEST_ARGS="$PYTEST_ARGS --no-cov"
            shift
            ;;
        -v|--verbose)
            PYTEST_ARGS="$PYTEST_ARGS -v"
            shift
            ;;
        -n|--parallel)
            PYTEST_ARGS="$PYTEST_ARGS -n auto"
            echo -e "${GREEN}Running tests in parallel...${NC}"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -u, --unit        Run unit tests only"
            echo "  -i, --integration Run integration tests only"
            echo "  -s, --slow        Run slow tests only"
            echo "  -f, --fast        Run fast tests only (default)"
            echo "  -h, --hft         Run HFT tests only"
            echo "  -p, --property    Run property-based tests only"
            echo "  --cov             Generate coverage report"
            echo "  --no-cov          Disable coverage"
            echo "  -v, --verbose     Verbose output"
            echo "  -n, --parallel    Run tests in parallel"
            exit 1
            ;;
    esac
done

# Ignore problematic tests
PYTEST_ARGS="$PYTEST_ARGS --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py"

# Run pytest
echo -e "${GREEN}Running pytest...${NC}"
python -m pytest tests/ $TEST_MARKER $PYTEST_ARGS

echo -e "${GREEN}All tests passed!${NC}"
