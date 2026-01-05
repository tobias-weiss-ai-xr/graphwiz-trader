#!/bin/bash
# Paper Trading Container Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="graphwiz-paper-trading"
CONTAINER_NAME="graphwiz-paper-trading"
COMPOSE_FILE="docker-compose.paper-trading.yml"
LOG_DIR="logs/paper_trading"

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Show usage
show_usage() {
    cat << EOF
${GREEN}GoEmotions Paper Trading Container Management${NC}

${BLUE}Usage:${NC}
    $0 [command] [options]

${BLUE}Commands:${NC}
    build       Build the GoEmotions paper trading Docker image
    start       Start the paper trading service
    stop        Stop the paper trading service
    restart     Restart the paper trading service
    status      Show service status and recent trades
    logs        Show service logs (follow mode)
    stats       Show resource usage statistics
    shell       Open shell in container
    clean       Remove container and images
    help        Show this help message

${BLUE}Options:${NC}
    --duration  Trading duration in hours (default: 72)
    --symbols   Trading pairs, e.g., BTC/EUR (default: BTC/EUR)
    --capital   Initial capital in EUR (default: 10000)
    --interval  Analysis interval in minutes (default: 30)

${BLUE}Examples:${NC}
    $0 build                                    # Build image
    $0 start                                    # Start 72-hour validation
    $0 start --duration 24 --interval 10        # Quick 24-hour test
    $0 start --duration 168 --symbols BTC/EUR ETH/EUR  # Full week, 2 symbols
    $0 logs                                     # View live logs
    $0 status                                   # Check status

${BLUE}Features:${NC}
    • Real-time market data from Kraken (German approved)
    • GoEmotions sentiment analysis (27 emotions)
    • Multi-factor signals (Technical + Emotion)
    • Contrarian trading (buy at fear, sell at euphoria)
    • Performance tracking and equity curves

EOF
}

# Build Docker image
build_image() {
    print_msg $BLUE "Building Docker image..."
    docker build -f Dockerfile.paper-trading -t $IMAGE_NAME .
    print_msg $GREEN "✓ Image built successfully"
}

# Start service
start_service() {
    # Parse arguments
    DURATION="72"
    SYMBOLS="BTC/EUR"
    CAPITAL="10000"
    INTERVAL="30"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --duration)
                DURATION="$2"
                shift 2
                ;;
            --symbols)
                SYMBOLS="$2"
                shift 2
                ;;
            --capital)
                CAPITAL="$2"
                shift 2
                ;;
            --interval)
                INTERVAL="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    # Create log directory
    mkdir -p $LOG_DIR

    # Check if container is already running
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_msg $YELLOW "⚠ Container is already running"
        print_msg $BLUE "Use '$0 restart' to restart the service"
        return 1
    fi

    print_msg $BLUE "Starting GoEmotions paper trading service..."
    print_msg $BLUE "  Duration: ${DURATION} hours"
    print_msg $BLUE "  Symbols: ${SYMBOLS}"
    print_msg $BLUE "  Capital: €${CAPITAL}"
    print_msg $BLUE "  Interval: ${INTERVAL} minutes"

    # Start with docker-compose
    DURATION=$DURATION SYMBOLS=$SYMBOLS CAPITAL=$CAPITAL INTERVAL=$INTERVAL \
        docker-compose -f $COMPOSE_FILE up -d

    print_msg $GREEN "✓ Service started successfully"
    print_msg $BLUE "\nView logs: $0 logs"
    print_msg $BLUE "Check status: $0 status"
    print_msg $BLUE "Stop service: $0 stop"
}

# Stop service
stop_service() {
    print_msg $BLUE "Stopping paper trading service..."

    docker-compose -f $COMPOSE_FILE down

    print_msg $GREEN "✓ Service stopped"
}

# Restart service
restart_service() {
    print_msg $BLUE "Restarting paper trading service..."
    stop_service
    sleep 2
    start_service "$@"
}

# Show status
show_status() {
    print_msg $BLUE "Paper Trading Service Status\n"

    # Check if container is running
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_msg $GREEN "✓ Container is RUNNING\n"

        # Show container details
        docker ps -f name=$CONTAINER_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

        echo -e "\n${BLUE}Resource Usage:${NC}"
        docker stats $CONTAINER_NAME --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

        echo -e "\n${BLUE}Recent Logs (last 10 lines):${NC}"
        docker logs --tail 10 $CONTAINER_NAME 2>&1 | grep -E "(INFO|SUCCESS|ERROR|WARNING|Trade)" || true

    else
        print_msg $RED "✗ Container is NOT running"
        echo ""
        echo "Start it with: $0 start"
    fi
}

# Show logs
show_logs() {
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_msg $BLUE "Showing paper trading logs (Ctrl+C to exit)...\n"
        docker logs -f $CONTAINER_NAME 2>&1 | grep -v "_internal"
    else
        print_msg $RED "✗ Container is not running"
        return 1
    fi
}

# Show statistics
show_stats() {
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_msg $BLUE "Resource Usage Statistics\n"
        docker stats $CONTAINER_NAME --no-stream
    else
        print_msg $RED "✗ Container is not running"
        return 1
    fi
}

# Open shell in container
open_shell() {
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_msg $BLUE "Opening shell in container...\n"
        docker exec -it $CONTAINER_NAME /bin/bash
    else
        print_msg $RED "✗ Container is not running"
        return 1
    fi
}

# Clean up
clean_up() {
    print_msg $YELLOW "Cleaning up..."
    docker-compose -f $COMPOSE_FILE down -v --rmi all

    print_msg $GREEN "✓ Cleanup complete"
}

# Main command dispatcher
case "${1:-help}" in
    build)
        build_image
        ;;
    start)
        shift
        start_service "$@"
        ;;
    stop)
        stop_service
        ;;
    restart)
        shift
        restart_service "$@"
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    stats)
        show_stats
        ;;
    shell)
        open_shell
        ;;
    clean)
        clean_up
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_msg $RED "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
