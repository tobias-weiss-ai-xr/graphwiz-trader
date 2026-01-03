#!/bin/bash
# ============================================================================
# GraphWiz Trader - Live Trading Deployment for Germany
# ============================================================================
# This script deploys live trading with BaFin-licensed exchanges
#
# ⚠️  WARNING: This will execute REAL trades with REAL money!
# ============================================================================
#
# Prerequisites:
# 1. Kraken account with API credentials
# 2. Tested thoroughly with paper trading (minimum 72 hours)
# 3. Understand the risks
# 4. Start with small amounts (€500 or less)
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/germany_live.yaml"
ENV_FILE="${PROJECT_ROOT}/.env"
LOG_DIR="${PROJECT_ROOT}/logs/live_trading"
VENV_DIR="${PROJECT_ROOT}/venv"

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

print_warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

show_disclaimer() {
    clear
    print_header "⚠️  LIVE TRADING DISCLAIMER ⚠️ "

    cat << "EOF"
    This script will deploy LIVE TRADING with REAL MONEY.

    You are about to:
    - Execute REAL trades on Kraken (BaFin-licensed exchange)
    - Risk REAL money
    - Deploy an automated trading system

    IMPORTANT:
    ✓ Kraken is fully licensed under MiCA for German users (August 2025)
    ✗ Binance is NOT licensed in Germany (application denied by BaFin in 2023)

    By continuing, you acknowledge that:
    1. You have tested thoroughly with paper trading
    2. You understand the risks involved
    3. You are starting with small amounts (€500 or less recommended)
    4. You are responsible for all trading decisions
    5. Past performance does not guarantee future results

EOF

    print_warning "Do NOT proceed if you don't understand these risks!"
    echo ""
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check if running from project root
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        print_error "Must run from project root directory"
        exit 1
    fi

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $PYTHON_VERSION found"

    # Check virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        print_warning "Virtual environment not found. Creating..."
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"

    # Check required packages
    print_info "Installing required packages..."
    pip install --quiet -r "$PROJECT_ROOT/requirements.txt"
    print_success "Required packages installed"

    # Check configuration file
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    print_success "Configuration file found"

    # Check .env file
    if [ ! -f "$ENV_FILE" ]; then
        print_error ".env file not found: $ENV_FILE"
        print_info "Please create .env file with your API credentials"
        exit 1
    fi
    print_success ".env file found"

    # Create log directory
    mkdir -p "$LOG_DIR"
    print_success "Log directory created: $LOG_DIR"

    echo ""
}

verify_credentials() {
    print_header "Verifying Exchange Credentials"

    source "$ENV_FILE"

    # Check Kraken credentials
    if [ -z "$KRAKEN_API_KEY" ] || [ -z "$KRAKEN_API_SECRET" ]; then
        print_error "Kraken API credentials not found in .env file"
        print_info "Please set KRAKEN_API_KEY and KRAKEN_API_SECRET in your .env file"
        exit 1
    fi

    print_success "Kraken API credentials found"

    # Verify credentials with a test call
    print_info "Testing Kraken connection..."

    python3 << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

try:
    from src.graphwiz_trader.trading.exchanges import create_german_exchange
    import os

    api_key = os.getenv('KRAKEN_API_KEY')
    api_secret = os.getenv('KRAKEN_API_SECRET')

    exchange = create_german_exchange('kraken', api_key, api_secret)

    # Test connection
    balance = exchange.get_balance()
    print("✅ Kraken connection successful")

    exchange.close()
except Exception as e:
    print(f"❌ Failed to connect to Kraken: {e}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        print_success "Kraken connection verified"
    else
        print_error "Failed to connect to Kraken"
        print_info "Please check your API credentials"
        exit 1
    fi

    echo ""
}

confirm_deployment() {
    print_header "Deployment Configuration"

    source "$ENV_FILE"

    echo "Exchange: Kraken"
    echo "License: MiCA (August 2025)"
    echo "Regulator: BaFin"
    echo ""
    echo "Configuration file: $CONFIG_FILE"
    echo "Log directory: $LOG_DIR"
    echo ""

    # Show current balance
    print_info "Fetching account balance..."

    python3 << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

try:
    from src.graphwiz_trader.trading.exchanges import create_german_exchange
    import os

    api_key = os.getenv('KRAKEN_API_KEY')
    api_secret = os.getenv('KRAKEN_API_SECRET')

    exchange = create_german_exchange('kraken', api_key, api_secret)
    balance = exchange.get_balance()

    # Show EUR balance
    if 'ZEUR' in balance:
        eur_balance = balance['ZEUR']['free']
        print(f"EUR Balance: €{float(eur_balance):,.2f}")
    else:
        print("EUR Balance: Not available")

    exchange.close()
except Exception as e:
    print(f"Could not fetch balance: {e}")
EOF

    echo ""
    print_warning "Please verify the information above"
    echo ""

    read -p "Type 'CONFIRM' to proceed with live trading deployment: " confirmation

    if [ "$confirmation" != "CONFIRM" ]; then
        print_error "Deployment cancelled"
        exit 0
    fi

    echo ""
}

start_live_trading() {
    print_header "Starting Live Trading"

    print_warning "Starting live trading with REAL money!"
    echo ""

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Set environment variables
    export GRAPHWIZ_CONFIG="$CONFIG_FILE"
    export TRADING_MODE="live"

    # Start live trading in background
    print_info "Starting live trading engine..."

    nohup python3 "$PROJECT_ROOT/scripts/live_trade.py" \
        --exchange kraken \
        --symbol BTC/EUR \
        --max-position 500 \
        --max-daily-loss 150 \
        --max-daily-trades 3 \
        --interval 3600 \
        > "$LOG_DIR/live_trading_output.log" 2>&1 &

    PID=$!
    echo $PID > "$LOG_DIR/live_trading.pid"

    print_success "Live trading started (PID: $PID)"
    echo ""

    # Show log tail
    print_info "Monitoring initial output (Ctrl+C to exit monitoring, trading continues)..."
    echo ""
    sleep 3

    tail -f "$LOG_DIR/live_trading_output.log"
}

show_status() {
    print_header "Live Trading Status"

    if [ ! -f "$LOG_DIR/live_trading.pid" ]; then
        print_info "No live trading process found"
        return
    fi

    PID=$(cat "$LOG_DIR/live_trading.pid")

    if ps -p $PID > /dev/null; then
        print_success "Live trading is running (PID: $PID)"
        echo ""
        print_info "Recent log entries:"
        echo ""
        tail -n 20 "$LOG_DIR/live_trading_output.log"
    else
        print_warning "Live trading process died"
        print_info "Check logs for details: $LOG_DIR/live_trading_output.log"
    fi
}

stop_live_trading() {
    print_header "Stopping Live Trading"

    if [ ! -f "$LOG_DIR/live_trading.pid" ]; then
        print_info "No live trading process found"
        return
    fi

    PID=$(cat "$LOG_DIR/live_trading.pid")

    if ps -p $PID > /dev/null; then
        print_warning "Stopping live trading (PID: $PID)..."
        kill $PID
        sleep 2

        if ps -p $PID > /dev/null; then
            print_warning "Force stopping..."
            kill -9 $PID
        fi

        print_success "Live trading stopped"
        rm "$LOG_DIR/live_trading.pid"
    else
        print_info "Live trading process not running"
    fi
}

show_menu() {
    echo ""
    echo "Select an option:"
    echo ""
    echo "1) Start Live Trading"
    echo "2) Show Status"
    echo "3) Stop Live Trading"
    echo "4) View Logs"
    echo "5) Exit"
    echo ""
    read -p "Choice [1-5]: " choice

    case $choice in
        1)
            show_disclaimer
            check_prerequisites
            verify_credentials
            confirm_deployment
            start_live_trading
            ;;
        2)
            show_status
            show_menu
            ;;
        3)
            stop_live_trading
            show_menu
            ;;
        4)
            if [ -f "$LOG_DIR/live_trading_output.log" ]; then
                less +F "$LOG_DIR/live_trading_output.log"
            else
                print_info "No log file found"
            fi
            show_menu
            ;;
        5)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            show_menu
            ;;
    esac
}

# ============================================================================
# Main
# ============================================================================

main() {
    cd "$PROJECT_ROOT"

    if [ "$1" == "start" ]; then
        show_disclaimer
        check_prerequisites
        verify_credentials
        confirm_deployment
        start_live_trading
    elif [ "$1" == "stop" ]; then
        stop_live_trading
    elif [ "$1" == "status" ]; then
        show_status
    elif [ "$1" == "logs" ]; then
        if [ -f "$LOG_DIR/live_trading_output.log" ]; then
            tail -f "$LOG_DIR/live_trading_output.log"
        else
            print_info "No log file found"
        fi
    else
        print_header "GraphWiz Trader - Live Trading Deployment (Germany)"
        show_menu
    fi
}

main "$@"
