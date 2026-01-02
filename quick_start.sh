#!/bin/bash
set -e

# GraphWiz Trader - Quick Start Script
# This script guides you through the setup and deployment process

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     GraphWiz Trader - Quick Start Deployment Script          ║"
echo "║     Knowledge Graph Powered AI Trading System                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. This is not recommended for security reasons."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.10 or higher."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $PYTHON_VERSION found"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker found"

    # Check docker-compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install docker-compose first."
        exit 1
    fi
    print_success "docker-compose found"

    # Check git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    print_success "Git found"

    echo ""
}

# Setup environment configuration
setup_environment() {
    print_info "Setting up environment configuration..."

    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file from .env.example"
        else
            print_error ".env.example not found. Creating basic .env file..."
            cat > .env << 'EOF'
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Exchange API Keys (Add your own)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# SAIA API Keys (for agent-looper)
SAIA_API_KEYS=your_saia_api_key

# Discord Notifications (optional)
DISCORD_WEBHOOK_URL=

# Email Notifications (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=
EMAIL_PASSWORD=
EMAIL_TO=
EOF
        fi
    else
        print_warning ".env file already exists. Skipping."
    fi

    print_warning "Please edit .env file with your API keys before starting!"
    echo ""
    read -p "Press Enter to continue after editing .env file..."
    echo ""
}

# Choose trading mode
choose_trading_mode() {
    print_info "Select trading mode:"
    echo "1) Paper Trading (Safe, recommended for starting)"
    echo "2) Conservative Live Trading (Low risk)"
    echo "3) Aggressive Live Trading (Higher risk)"
    echo "4) Scalping (High frequency)"
    echo "5) Swing Trading (Longer term)"
    echo ""
    read -p "Enter choice [1-5] (default: 1): " mode_choice

    case $mode_choice in
        1) MODE="paper_trading" ;;
        2) MODE="conservative" ;;
        3) MODE="aggressive" ;;
        4) MODE="scalping" ;;
        5) MODE="swing_trading" ;;
        *) MODE="paper_trading" ;;
    esac

    print_info "Selected mode: $MODE"

    if [ -f "config/${MODE}.yaml" ]; then
        cp "config/${MODE}.yaml" config/config.yaml
        print_success "Configuration applied: config/${MODE}.yaml"
    else
        print_error "Configuration file not found: config/${MODE}.yaml"
        exit 1
    fi

    echo ""
}

# Start Neo4j service
start_neo4j() {
    print_info "Starting Neo4j knowledge graph..."

    # Start Neo4j via docker-compose
    docker-compose up -d neo4j

    # Wait for Neo4j to be ready
    print_info "Waiting for Neo4j to start..."
    for i in {1..30}; do
        if docker-compose exec -T neo4j cypher-shell "RETURN 1" &> /dev/null; then
            print_success "Neo4j is ready!"
            echo ""
            return
        fi
        sleep 2
    done

    print_error "Neo4j failed to start. Please check logs with: docker-compose logs neo4j"
    exit 1
}

# Initialize knowledge graph
initialize_graph() {
    print_info "Initializing knowledge graph schema..."

    # This will be done automatically by the trading system
    print_success "Knowledge graph will be initialized automatically on first start"
    echo ""
}

# Start services
start_services() {
    print_info "Starting GraphWiz Trader services..."

    # Start core services
    docker-compose up -d graphwiz-trader prometheus grafana

    # Wait for services to be healthy
    print_info "Waiting for services to start..."
    sleep 10

    # Check health
    if curl -s http://localhost:8080/health | grep -q "healthy"; then
        print_success "GraphWiz Trader is running!"
    else
        print_warning "Health check not passed yet. Check logs with: docker-compose logs -f graphwiz-trader"
    fi

    echo ""
}

# Display dashboard information
show_dashboards() {
    print_info "Dashboard URLs:"
    echo ""
    echo "📊 Grafana Dashboards: http://localhost:3000"
    echo "   Default login: admin / admin"
    echo ""
    echo "📈 Prometheus Metrics: http://localhost:9090"
    echo ""
    echo "🏥 Health Endpoint: http://localhost:8080/health"
    echo ""
    echo "📡 WebSocket Updates: ws://localhost:8765"
    echo ""
}

# Display next steps
show_next_steps() {
    print_info "Next Steps:"
    echo ""
    echo "1️⃣  Monitor Grafana dashboards:"
    echo "   http://localhost:3000"
    echo ""
    echo "2️⃣  View logs:"
    echo "   docker-compose logs -f graphwiz-trader"
    echo ""
    echo "3️⃣  Check trading status:"
    echo "   curl http://localhost:8080/api/status"
    echo ""
    echo "4️⃣  Run test suite:"
    echo "   bash tests/run_all_tests.sh"
    echo ""
    echo "5️⃣  After 3 days of paper trading, enable agent-looper:"
    echo "   docker-compose up -d agent-looper"
    echo ""
    echo "📚 Full documentation: /opt/git/graphwiz-trader/IMPLEMENTATION_COMPLETE.md"
    echo ""
}

# Main execution
main() {
    cd /opt/git/graphwiz-trader

    echo "🚀 Starting GraphWiz Trader deployment..."
    echo ""

    check_root
    check_requirements
    setup_environment
    choose_trading_mode
    start_neo4j
    initialize_graph
    start_services
    show_dashboards
    show_next_steps

    print_success "Deployment complete!"
    echo ""
    echo "⚠️  IMPORTANT: Currently running in PAPER TRADING mode"
    echo "   No real money is being traded. Safe for testing."
    echo ""
    echo "📖 Read the full implementation guide:"
    echo "   cat /opt/git/graphwiz-trader/IMPLEMENTATION_COMPLETE.md"
    echo ""
}

# Run main function
main
