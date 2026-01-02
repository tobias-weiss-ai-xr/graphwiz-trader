#!/bin/bash
##############################################################################
# GraphWiz Trader Deployment Script
# Purpose: Automated deployment with validation, backup, and rollback support
##############################################################################

set -euo pipefail  # Exit on error, undefined variables, and pipe failures
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_USER="${DEPLOY_USER:-graphwiz}"
BACKUP_DIR="${PROJECT_ROOT}/backups/deployments"
LOG_FILE="${PROJECT_ROOT}/logs/deploy-$(date +%Y%m%d-%H%M%S).log"
ROLLBACK_MARKER="${BACKUP_DIR}/.last_deployment"

# Create necessary directories
mkdir -p "${BACKUP_DIR}" "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/data"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "${BLUE}$@${NC}"; }
log_success() { log "SUCCESS" "${GREEN}$@${NC}"; }
log_warning() { log "WARNING" "${YELLOW}$@${NC}"; }
log_error() { log "ERROR" "${RED}$@${NC}"; }

# Error handler
error_exit() {
    log_error "$1"
    log_error "Deployment failed! Check ${LOG_FILE} for details."
    exit 1
}

# Cleanup handler
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/graphwiz_deploy_* 2>/dev/null || true
}

trap cleanup EXIT
trap 'error_exit "Deployment interrupted by signal"' INT TERM

##############################################################################
# Validation Functions
##############################################################################

check_environment() {
    log_info "Validating deployment environment..."

    # Check if running as root (should not be)
    if [[ $EUID -eq 0 ]]; then
        error_exit "This script should not be run as root. Use user ${DEPLOY_USER}"
    fi

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error_exit "Python 3 is not installed"
    fi

    local python_version=$(python3 --version | awk '{print $2}')
    log_info "Python version: ${python_version}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed"
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error_exit "Docker Compose is not installed"
    fi

    # Check git
    if ! command -v git &> /dev/null; then
        error_exit "Git is not installed"
    fi

    log_success "Environment validation passed"
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check if required files exist
    local required_files=(
        "${PROJECT_ROOT}/requirements.txt"
        "${PROJECT_ROOT}/setup.py"
        "${PROJECT_ROOT}/Dockerfile"
        "${PROJECT_ROOT}/docker-compose.yml"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error_exit "Required file not found: $file"
        fi
    done

    # Check if virtual environment exists
    if [[ ! -d "${PROJECT_ROOT}/venv" ]]; then
        log_warning "Virtual environment not found. Creating..."
        python3 -m venv "${PROJECT_ROOT}/venv" || error_exit "Failed to create virtual environment"
    fi

    # Activate virtual environment
    source "${PROJECT_ROOT}/venv/bin/activate"

    # Install/update dependencies
    log_info "Installing Python dependencies..."
    pip install --quiet --upgrade pip || error_exit "Failed to upgrade pip"
    pip install --quiet -r "${PROJECT_ROOT}/requirements.txt" || error_exit "Failed to install dependencies"

    log_success "Dependencies check passed"
}

validate_configuration() {
    log_info "Validating configuration..."

    # Check if production config exists
    if [[ ! -f "${PROJECT_ROOT}/config/production.yaml" ]]; then
        error_exit "Production configuration not found: config/production.yaml"
    fi

    # Check if .env file exists
    if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
        log_warning ".env file not found. Copying from .env.example..."
        if [[ -f "${PROJECT_ROOT}/.env.example" ]]; then
            cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
            log_warning "Please configure ${PROJECT_ROOT}/.env with your API keys and secrets"
        else
            error_exit "Neither .env nor .env.example found"
        fi
    fi

    # Validate .env file (basic checks)
    source "${PROJECT_ROOT}/.env"

    if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
        log_warning "NEO4J_PASSWORD not set in .env"
    fi

    if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        log_warning "No AI API keys configured. Agent features may not work."
    fi

    log_success "Configuration validation passed"
}

check_system_resources() {
    log_info "Checking system resources..."

    # Check available memory
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_mem -lt 8 ]]; then
        log_warning "System has less than 8GB RAM. Performance may be degraded."
    fi

    # Check disk space
    local available_disk=$(df -BG "${PROJECT_ROOT}" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_disk -lt 20 ]]; then
        log_warning "Less than 20GB disk space available. This may cause issues."
    fi

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running"
    fi

    log_success "System resources check passed"
}

##############################################################################
# Backup Functions
##############################################################################

create_backup() {
    log_info "Creating backup..."

    local backup_name="backup-$(date +%Y%m%d-%H%M%S)"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    mkdir -p "${backup_path}"

    # Backup configuration files
    cp -r "${PROJECT_ROOT}/config" "${backup_path}/" 2>/dev/null || true

    # Backup data directory (if exists)
    if [[ -d "${PROJECT_ROOT}/data" ]]; then
        cp -r "${PROJECT_ROOT}/data" "${backup_path}/data" 2>/dev/null || true
    fi

    # Backup current git state
    cd "${PROJECT_ROOT}"
    git rev-parse HEAD > "${backup_path}/git-commit.txt" 2>/dev/null || true
    git diff > "${backup_path}/git-diff.patch" 2>/dev/null || true

    # Save deployment metadata
    cat > "${backup_path}/deployment-info.txt" <<EOF
Deployment Date: $(date)
Deployment User: $(whoami)
Hostname: $(hostname)
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
Git Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")
EOF

    # Create symlink to latest backup
    ln -sf "${backup_path}" "${BACKUP_DIR}/latest"

    # Save rollback marker
    echo "${backup_path}" > "${ROLLBACK_MARKER}"

    log_success "Backup created: ${backup_path}"
}

perform_rollback() {
    log_warning "Initiating rollback..."

    if [[ ! -f "${ROLLBACK_MARKER}" ]]; then
        error_exit "No backup found for rollback"
    fi

    local backup_path=$(cat "${ROLLBACK_MARKER}")

    if [[ ! -d "${backup_path}" ]]; then
        error_exit "Backup directory not found: ${backup_path}"
    fi

    log_info "Rolling back to: ${backup_path}"

    # Restore configuration
    if [[ -d "${backup_path}/config" ]]; then
        cp -rf "${backup_path}/config" "${PROJECT_ROOT}/"
        log_info "Configuration restored"
    fi

    # Restore data (optional - confirm with user)
    if [[ -d "${backup_path}/data" ]]; then
        read -p "Restore data directory? This will overwrite current data. (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp -rf "${backup_path}/data" "${PROJECT_ROOT}/"
            log_info "Data restored"
        fi
    fi

    log_success "Rollback completed successfully"
}

##############################################################################
# Deployment Functions
##############################################################################

build_docker_images() {
    log_info "Building Docker images..."

    cd "${PROJECT_ROOT}"

    # Build with build args
    local build_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local version=$(git describe --tags --always --dirty 2>/dev/null || echo "0.1.0")

    docker build \
        --build-arg BUILD_DATE="${build_date}" \
        --build-arg VERSION="${version}" \
        --tag "graphwiz-trader:${version}" \
        --tag "graphwiz-trader:latest" \
        --file Dockerfile \
        . || error_exit "Docker build failed"

    log_success "Docker images built successfully"
}

run_tests() {
    log_info "Running tests..."

    cd "${PROJECT_ROOT}"
    source "${PROJECT_ROOT}/venv/bin/activate"

    # Run tests with coverage
    if pytest tests/ -v --cov=graphwiz_trader --cov-report=term-missing --tb=short 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "All tests passed"
    else
        log_warning "Some tests failed. Check ${LOG_FILE} for details."
        read -p "Continue deployment anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error_exit "Deployment cancelled due to test failures"
        fi
    fi
}

start_services() {
    log_info "Starting services..."

    cd "${PROJECT_ROOT}"

    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose down 2>/dev/null || true

    # Start services
    log_info "Starting new services..."
    docker-compose up -d || error_exit "Failed to start services"

    log_success "Services started"
}

perform_health_checks() {
    log_info "Performing health checks..."

    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt ${attempt}/${max_attempts}"

        # Check Neo4j
        if docker-compose ps neo4j | grep -q "Up (healthy)"; then
            log_success "Neo4j is healthy"
        else
            log_warning "Neo4j not ready yet..."
        fi

        # Check main application
        if docker-compose ps graphwiz-trader | grep -q "Up (healthy)"; then
            log_success "GraphWiz Trader is healthy"
        else
            log_warning "GraphWiz Trader not ready yet..."
        fi

        # Check if all services are up
        if docker-compose ps | grep -q "Up (healthy)"; then
            log_success "All services are healthy"
            return 0
        fi

        sleep 10
        ((attempt++))
    done

    log_warning "Health checks timed out. Some services may not be ready."
    log_info "Check logs with: docker-compose logs -f"
}

##############################################################################
# Main Deployment Flow
##############################################################################

main() {
    local skip_tests=${SKIP_TESTS:-false}
    local skip_build=${SKIP_BUILD:-false}

    log_info "=========================================="
    log_info "GraphWiz Trader Deployment Script"
    log_info "=========================================="
    log_info "Started at: $(date)"
    log_info "Deployment user: $(whoami)"
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "=========================================="

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --rollback)
                perform_rollback
                exit 0
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --rollback    Rollback to previous deployment"
                echo "  --skip-tests  Skip running tests"
                echo "  --skip-build  Skip Docker image build"
                echo "  --help        Show this help message"
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done

    # Validation phase
    check_environment
    check_dependencies
    validate_configuration
    check_system_resources

    # Backup phase
    create_backup

    # Build phase
    if [[ "$skip_build" == "false" ]]; then
        build_docker_images
    else
        log_info "Skipping Docker build (--skip-build flag)"
    fi

    # Test phase
    if [[ "$skip_tests" == "false" ]]; then
        run_tests
    else
        log_info "Skipping tests (--skip-tests flag)"
    fi

    # Deployment phase
    start_services

    # Health check phase
    perform_health_checks

    # Success
    log_success "=========================================="
    log_success "Deployment completed successfully!"
    log_success "=========================================="
    log_info "Services status:"
    docker-compose ps
    log_info "=========================================="
    log_info "View logs: docker-compose logs -f"
    log_info "Stop services: docker-compose down"
    log_info "=========================================="
}

# Run main function
main "$@"
