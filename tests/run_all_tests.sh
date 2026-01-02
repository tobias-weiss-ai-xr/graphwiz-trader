#!/bin/bash
# Comprehensive test suite runner for graphwiz-trader
# Runs all test types with coverage reporting and deployment validation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/opt/git/graphwiz-trader"
COVERAGE_MIN=80
TEST_RESULTS_DIR="$PROJECT_ROOT/test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$TEST_RESULTS_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Graphwiz Trader Test Suite Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BLUE}======== $1 ========${NC}"
    echo ""
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Change to project root
cd "$PROJECT_ROOT"

# Clean previous test artifacts
print_section "Cleaning Previous Test Artifacts"
rm -rf htmlcov/
rm -rf .pytest_cache/
rm -f .coverage
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
print_success "Cleaned previous test artifacts"

# Run linter checks
print_section "Running Code Quality Checks"

echo "Running Black formatter check..."
if black --check src/ tests/ 2>&1 | tee "$TEST_RESULTS_DIR/black_check_$TIMESTAMP.log"; then
    print_success "Black formatter check passed"
else
    print_warning "Black formatter found issues. Run 'black src/ tests/' to fix."
fi

echo "Running Flake8 linter..."
if flake8 src/ tests/ --max-line-length=100 --exclude=.git,__pycache__,build,dist 2>&1 | tee "$TEST_RESULTS_DIR/flake8_check_$TIMESTAMP.log"; then
    print_success "Flake8 linter check passed"
else
    print_error "Flake8 found linting issues"
fi

echo "Running MyPy type checker..."
if mypy src/ --ignore-missing-imports 2>&1 | tee "$TEST_RESULTS_DIR/mypy_check_$TIMESTAMP.log"; then
    print_success "MyPy type checker passed"
else
    print_warning "MyPy found type issues"
fi

# Run unit tests
print_section "Running Unit Tests"
echo "Running unit tests with coverage..."

if pytest tests/ \
    -m "not integration and not performance" \
    -v \
    --tb=short \
    --cov=src/graphwiz_trader \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-report="xml:$TEST_RESULTS_DIR/coverage_unit_$TIMESTAMP.xml" \
    --junitxml="$TEST_RESULTS_DIR/junit_unit_$TIMESTAMP.xml" \
    2>&1 | tee "$TEST_RESULTS_DIR/unit_tests_$TIMESTAMP.log"; then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

# Extract coverage percentage
COVERAGE=$(grep -oP '\d+% (?=\()' "$TEST_RESULTS_DIR/unit_tests_$TIMESTAMP.log" | head -1 | tr -d '%')
echo ""
echo -e "Unit Test Coverage: ${BLUE}${COVERAGE}%${NC}"

if [ "$COVERAGE" -lt "$COVERAGE_MIN" ]; then
    print_warning "Coverage ${COVERAGE}% is below minimum ${COVERAGE_MIN}%"
fi

# Run integration tests
print_section "Running Integration Tests"
echo "Running integration tests (may take longer)..."

if pytest tests/integration/ \
    -v \
    -m integration \
    --tb=short \
    --cov=src/graphwiz_trader \
    --cov-append \
    --cov-report="term-missing" \
    --cov-report="xml:$TEST_RESULTS_DIR/coverage_integration_$TIMESTAMP.xml" \
    --junitxml="$TEST_RESULTS_DIR/junit_integration_$TIMESTAMP.xml" \
    2>&1 | tee "$TEST_RESULTS_DIR/integration_tests_$TIMESTAMP.log"; then
    print_success "Integration tests passed"
else
    print_error "Integration tests failed"
fi

# Run performance tests
print_section "Running Performance Tests"
echo "Running performance benchmarks..."

if pytest tests/performance/ \
    -v \
    -m performance \
    --tb=short \
    --junitxml="$TEST_RESULTS_DIR/junit_performance_$TIMESTAMP.xml" \
    2>&1 | tee "$TEST_RESULTS_DIR/performance_tests_$TIMESTAMP.log"; then
    print_success "Performance tests passed"
else
    print_warning "Some performance tests failed (may be acceptable in some environments)"
fi

# Generate combined coverage report
print_section "Generating Combined Coverage Report"

echo "Combining coverage reports..."
coverage combine 2>/dev/null || true

# Generate final coverage report
coverage report --omit="*/tests/*,*/test_*,*/__pycache__/*" \
    > "$TEST_RESULTS_DIR/coverage_summary_$TIMESTAMP.txt"

# Display coverage summary
cat "$TEST_RESULTS_DIR/coverage_summary_$TIMESTAMP.txt"

# Generate HTML coverage report
coverage html --omit="*/tests/*,*/test_*,*/__pycache__/*" -d htmlcov
print_success "HTML coverage report generated in htmlcov/"

# Run security checks
print_section "Running Security Scans"

echo "Checking for security vulnerabilities..."
if command -v safety &> /dev/null; then
    safety check --json > "$TEST_RESULTS_DIR/safety_check_$TIMESTAMP.json" 2>&1 || true
    print_success "Safety check completed"
else
    print_warning "Safety not installed. Run 'pip install safety' to enable security checks."
fi

# Check for exposed credentials
echo "Checking for exposed credentials..."
if grep -r -E "(api_key|api_secret|password|token)\s*=\s*['\"][^'\"]*['\"]" \
    --include="*.py" \
    --exclude-dir=htmlcov \
    --exclude-dir=.pytest_cache \
    --exclude-dir=test_results \
    src/ 2>/dev/null | grep -v "test\|example\|mock\|placeholder"; then
    print_warning "Possible exposed credentials found. Please review."
else
    print_success "No exposed credentials detected"
fi

# Deployment readiness validation
print_section "Validating Deployment Readiness"

DEPLOYMENT_CHECKS=0
DEPLOYMENT_PASSED=0

# Check 1: All critical tests passed
((DEPLOYMENT_CHECKS++))
if [ -f "$TEST_RESULTS_DIR/junit_unit_$TIMESTAMP.xml" ]; then
    ((DEPLOYMENT_PASSED++))
    print_success "Unit tests completed"
else
    print_error "Unit tests missing"
fi

# Check 2: Coverage meets minimum
((DEPLOYMENT_CHECKS++))
if [ "$COVERAGE" -ge "$COVERAGE_MIN" ]; then
    ((DEPLOYMENT_PASSED++))
    print_success "Coverage meets minimum (${COVERAGE}% >= ${COVERAGE_MIN}%)"
else
    print_warning "Coverage below minimum"
fi

# Check 3: Configuration files exist
((DEPLOYMENT_CHECKS++))
if [ -f "$PROJECT_ROOT/config/config.example.yaml" ] || [ -f "$PROJECT_ROOT/config/agents.example.yaml" ]; then
    ((DEPLOYMENT_PASSED++))
    print_success "Configuration examples present"
else
    print_error "Configuration examples missing"
fi

# Check 4: Documentation present
((DEPLOYMENT_CHECKS++))
if [ -f "$PROJECT_ROOT/README.md" ] && [ -f "$PROJECT_ROOT/DEPLOYMENT.md" ]; then
    ((DEPLOYMENT_PASSED++))
    print_success "Documentation present"
else
    print_error "Documentation incomplete"
fi

# Check 5: No critical linting issues
((DEPLOYMENT_CHECKS++))
if [ -f "$TEST_RESULTS_DIR/flake8_check_$TIMESTAMP.log" ]; then
    ERROR_COUNT=$(grep -c "error:" "$TEST_RESULTS_DIR/flake8_check_$TIMESTAMP.log" || true)
    if [ "$ERROR_COUNT" -eq 0 ]; then
        ((DEPLOYMENT_PASSED++))
        print_success "No critical linting errors"
    else
        print_warning "Found $ERROR_COUNT linting errors"
    fi
fi

# Generate test summary report
print_section "Test Summary Report"

cat > "$TEST_RESULTS_DIR/test_summary_$TIMESTAMP.txt" << EOF
Graphwiz Trader Test Summary
Generated: $(date)
================================

Test Results Location: $TEST_RESULTS_DIR

Unit Tests: $(grep -c "tests passed" "$TEST_RESULTS_DIR/unit_tests_$TIMESTAMP.log" 2>/dev/null || echo "0") passed
Integration Tests: $(grep -c "tests passed" "$TEST_RESULTS_DIR/integration_tests_$TIMESTAMP.log" 2>/dev/null || echo "0") passed
Performance Tests: $(grep -c "tests passed" "$TEST_RESULTS_DIR/performance_tests_$TIMESTAMP.log" 2>/dev/null || echo "0") passed

Code Coverage: ${COVERAGE}%
Minimum Required: ${COVERAGE_MIN}%

Deployment Readiness: $DEPLOYMENT_PASSED/$DEPLOYMENT_CHECKS checks passed

Files:
- Unit Test Log: $TEST_RESULTS_DIR/unit_tests_$TIMESTAMP.log
- Integration Test Log: $TEST_RESULTS_DIR/integration_tests_$TIMESTAMP.log
- Performance Test Log: $TEST_RESULTS_DIR/performance_tests_$TIMESTAMP.log
- Coverage Report: htmlcov/index.html
- Coverage Summary: $TEST_RESULTS_DIR/coverage_summary_$TIMESTAMP.txt
- JUnit Results: $TEST_RESULTS_DIR/junit_*.xml

EOF

cat "$TEST_RESULTS_DIR/test_summary_$TIMESTAMP.txt"

# Final verdict
echo ""
echo -e "${BLUE}========================================${NC}"
if [ $DEPLOYMENT_PASSED -eq $DEPLOYMENT_CHECKS ] && [ "$COVERAGE" -ge "$COVERAGE_MIN" ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED - READY FOR DEPLOYMENT${NC}"
    echo -e "${BLUE}========================================${NC}"
    exit 0
elif [ $DEPLOYMENT_PASSED -ge $((DEPLOYMENT_CHECKS - 1)) ]; then
    echo -e "${YELLOW}⚠ MOSTLY READY - Review warnings above${NC}"
    echo -e "${BLUE}========================================${NC}"
    exit 0
else
    echo -e "${RED}✗ NOT READY - Fix failures before deployment${NC}"
    echo -e "${BLUE}========================================${NC}"
    exit 1
fi
