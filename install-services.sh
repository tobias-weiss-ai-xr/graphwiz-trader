#!/bin/bash

# GraphWiz Trader - Systemd Service Installation Script
# This script installs and configures systemd services for GraphWiz Trader

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Services to install
SERVICES=(
    "graphwiz-live-trading.service"
    "graphwiz-paper-trading.service"
)

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if user exists
    if ! id "weiss" &>/dev/null; then
        print_warning "User 'weiss' does not exist. Creating..."
        useradd -r -s /bin/bash -d /opt/git/graphwiz-trader weiss || true
    fi
    
    # Check if .env file exists
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        print_warning ".env file not found. Creating from template..."
        cp "$SCRIPT_DIR/.env.docker" "$SCRIPT_DIR/.env" || {
            print_error "Failed to create .env file. Please copy .env.docker to .env and configure it manually."
            exit 1
        }
        print_warning "Please edit $SCRIPT_DIR/.env with your configuration before starting services."
    fi
    
    # Create necessary directories
    mkdir -p "$SCRIPT_DIR/logs"/{live_trading,paper_trading,system}
    mkdir -p "$SCRIPT_DIR/data"
    mkdir -p "$SCRIPT_DIR/backtests"
    
    # Set ownership
    chown -R weiss:weiss "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data" "$SCRIPT_DIR/backtests" 2>/dev/null || true
    
    print_status "Prerequisites check completed."
}

# Install systemd services
install_services() {
    print_status "Installing systemd services..."
    
    for service in "${SERVICES[@]}"; do
        service_file="$SCRIPT_DIR/$service"
        service_name=$(basename "$service")
        
        if [[ -f "$service_file" ]]; then
            print_status "Installing $service_name..."
            
            # Copy service file
            cp "$service_file" "/etc/systemd/system/"
            
            # Set proper permissions
            chmod 644 "/etc/systemd/system/$service_name"
            
            # Reload systemd
            systemctl daemon-reload
            
            print_status "Service $service_name installed successfully."
        else
            print_warning "Service file $service_file not found. Skipping..."
        fi
    done
}

# Configure services
configure_services() {
    print_status "Configuring services..."
    
    # Enable services (but don't start them)
    for service in "${SERVICES[@]}"; do
        service_name=$(basename "$service")
        if [[ -f "/etc/systemd/system/$service_name" ]]; then
            print_status "Enabling $service_name..."
            systemctl enable "$service_name"
        fi
    done
    
    print_status "Services configured. Use 'systemctl start <service>' to start individual services."
}

# Show usage information
show_usage() {
    print_status "Installation completed successfully!"
    echo
    echo "Available services:"
    echo "  - graphwiz-live-trading.service : Live trading with real money"
    echo "  - graphwiz-paper-trading.service: Paper trading (testing)"
    echo
    echo "Commands:"
    echo "  sudo systemctl start graphwiz-live-trading.service  # Start live trading"
    echo "  sudo systemctl start graphwiz-paper-trading.service # Start paper trading"
    echo
    echo "  sudo systemctl status graphwiz-live-trading.service # Check status"
    echo "  sudo journalctl -u graphwiz-live-trading.service -f # View logs"
    echo
    echo "Docker Compose commands (alternative):"
    echo "  make docker-prod    # Start production stack"
    echo "  make docker-dev     # Start development stack"
    echo "  make docker-down    # Stop all services"
    echo "  make docker-logs    # View logs"
}

# Main installation
main() {
    print_status "GraphWiz Trader - Systemd Service Installation"
    echo "=================================================="
    
    check_root
    check_prerequisites
    install_services
    configure_services
    show_usage
    
    print_status "Installation completed successfully!"
}

# Run main function
main "$@"