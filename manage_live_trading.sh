#!/bin/bash
# ==============================================================================
# GraphWiz Trader - Live Trading Management Script
# ==============================================================================
# This script provides easy management of the live trading container
#
# Usage: ./manage_live_trading.sh [command]
#
# Commands:
#   build    - Build Docker image
#   start    - Start live trading container
#   stop     - Stop live trading container
#   restart  - Restart live trading container
#   status   - Show container status and recent logs
#   logs     - Tail logs in real-time
#   shell    - Open shell in container
#   validate - Run pre-deployment validation
#   clean    - Remove container and images
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="graphwiz-live-trading"
CONTAINER_NAME="graphwiz-live-trading"
COMPOSE_FILE="docker-compose.live-trading.yml"
CONFIG_FILE="config/germany_live_custom.yaml"

# Helper functions
print_header() {
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        echo ""
        echo "Please create .env file with your Kraken credentials:"
        echo "  KRAKEN_API_KEY=your_api_key_here"
        echo "  KRAKEN_API_SECRET=your_api_secret_here"
        echo ""
        echo "You can copy the example:"
        echo "  cp .env.live.example .env"
        echo "  nano .env"
        exit 1
    fi

    # Check if credentials are set
    if ! grep -q "KRAKEN_API_KEY=" .env || ! grep -q "KRAKEN_API_SECRET=" .env; then
        print_error "Kraken credentials not found in .env!"
        exit 1
    fi

    print_success "Environment file found"
}

# Build Docker image
build_image() {
    print_header "Building Docker Image"

    check_env_file

    if ! docker build -f Dockerfile.live-trading -t $IMAGE_NAME:latest . ; then
        print_error "Failed to build Docker image"
        exit 1
    fi

    print_success "Docker image built successfully: $IMAGE_NAME:latest"
}

# Start container
start_container() {
    print_header "Starting Live Trading Container"

    check_env_file

    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Container already exists"
        read -p "Do you want to restart it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker restart $CONTAINER_NAME
            print_success "Container restarted"
        else
            echo "Aborted"
            exit 0
        fi
    else
        # Start new container
        if ! docker-compose -f $COMPOSE_FILE up -d ; then
            print_error "Failed to start container"
            exit 1
        fi

        print_success "Container started: $CONTAINER_NAME"
    fi

    # Show initial logs
    echo ""
    echo "Waiting for container to initialize..."
    sleep 3

    echo ""
    echo "Container status:"
    docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    echo ""
    echo "View logs with: ./manage_live_trading.sh logs"
    echo "Stop with: ./manage_live_trading.sh stop"
}

# Stop container
stop_container() {
    print_header "Stopping Live Trading Container"

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Container is not running"
        exit 0
    fi

    echo "Stopping container gracefully..."
    docker stop $CONTAINER_NAME

    print_success "Container stopped"
}

# Restart container
restart_container() {
    print_header "Restarting Live Trading Container"

    stop_container
    sleep 2
    start_container
}

# Show status
show_status() {
    print_header "Live Trading Container Status"

    # Check if container exists
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Container does not exist"
        echo ""
        echo "Build and start with:"
        echo "  ./manage_live_trading.sh build"
        echo "  ./manage_live_trading.sh start"
        exit 0
    fi

    # Container info
    echo "Container Information:"
    docker inspect $CONTAINER_NAME --format='  Name: {{.Name}}
  State: {{.State.Status}}
  Started: {{.State.StartedAt}}
  Image: {{.Config.Image}}' 2>/dev/null || echo "  Error getting container info"

    echo ""

    # Resource usage
    echo "Resource Usage:"
    docker stats $CONTAINER_NAME --no-stream --format "  CPU: {{.CPUPerc}}
  Memory: {{.MemUsage}}
  Network I/O: {{.NetIO}}" 2>/dev/null || echo "  Container not running"

    echo ""

    # Recent logs
    echo "Recent Logs (last 20 lines):"
    echo "----------------------------------------"
    docker logs --tail 20 $CONTAINER_NAME 2>&1 || echo "No logs available"
}

# Show logs
show_logs() {
    print_header "Live Trading Logs"

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "Container is not running"
        exit 1
    fi

    echo "Showing logs (Ctrl+C to exit)..."
    echo "----------------------------------------"
    docker logs -f $CONTAINER_NAME 2>&1
}

# Open shell in container
open_shell() {
    print_header "Opening Shell in Container"

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "Container is not running"
        exit 1
    fi

    echo "Opening shell in container..."
    echo "Exit with: exit or Ctrl+D"
    echo ""
    docker exec -it $CONTAINER_NAME /bin/bash
}

# Run validation
run_validation() {
    print_header "Running Pre-Deployment Validation"

    echo ""
    echo "1. Checking environment file..."
    check_env_file

    echo ""
    echo "2. Checking configuration file..."
    if [ ! -f $CONFIG_FILE ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    print_success "Configuration file found"

    echo ""
    echo "3. Checking Docker..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker is installed"

    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed"
        exit 1
    fi
    print_success "docker-compose is installed"

    echo ""
    echo "4. Checking log directories..."
    mkdir -p logs/live_trading
    print_success "Log directories created"

    echo ""
    echo "5. Testing Kraken connection..."
    source venv/bin/activate
    if python3 scripts/live_trade.py --exchange kraken --symbol BTC/EUR --test --no-confirm 2>&1 | grep -q "Price:"; then
        print_success "Kraken connection successful"
    else
        print_warning "Kraken connection test failed (check credentials)"
    fi

    echo ""
    print_success "Validation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Build image: ./manage_live_trading.sh build"
    echo "  2. Start container: ./manage_live_trading.sh start"
    echo "  3. Monitor logs: ./manage_live_trading.sh logs"
}

# Clean up containers and images
clean_all() {
    print_header "Cleaning Up"

    print_warning "This will remove the container and images"
    read -p "Are you sure? (yes/no): " -r
    echo
    if [[ ! $REPLY == "yes" ]]; then
        echo "Aborted"
        exit 0
    fi

    echo ""
    echo "Stopping and removing container..."
    docker rm -f $CONTAINER_NAME 2>/dev/null || true

    echo ""
    echo "Removing Docker image..."
    docker rmi $IMAGE_NAME:latest 2>/dev/null || true

    print_success "Cleanup complete"
}

# Show usage
show_usage() {
    cat << EOF
${BLUE}GraphWiz Trader - Live Trading Management${NC}

Usage: ./manage_live_trading.sh [command]

Commands:
  ${GREEN}build${NC}     Build Docker image
  ${GREEN}start${NC}     Start live trading container
  ${GREEN}stop${NC}      Stop live trading container
  ${GREEN}restart${NC}   Restart live trading container
  ${GREEN}status${NC}    Show container status and recent logs
  ${GREEN}logs${NC}      Tail logs in real-time
  ${GREEN}shell${NC}     Open shell in container
  ${GREEN}validate${NC}  Run pre-deployment validation
  ${GREEN}clean${NC}     Remove container and images
  ${GREEN}help${NC}      Show this help message

Examples:
  ./manage_live_trading.sh validate
  ./manage_live_trading.sh build
  ./manage_live_trading.sh start
  ./manage_live_trading.sh logs

Configuration:
  Config file: $CONFIG_FILE
  Container:   $CONTAINER_NAME
  Image:       $IMAGE_NAME

For more information, see: docs/LIVE_TRADING_GERMANY.md

EOF
}

# Main script logic
case "${1:-}" in
    build)
        build_image
        ;;
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    validate)
        run_validation
        ;;
    clean)
        clean_all
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: ${1:-}"
        echo ""
        show_usage
        exit 1
        ;;
esac
