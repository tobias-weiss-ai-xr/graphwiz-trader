#!/bin/bash
# GraphWiz Trader - Paper Trading Validation Launcher
# Easy launcher for extended paper trading validation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Main menu
show_menu() {
    echo ""
    print_header "GraphWiz Trader - Paper Trading Validation"
    echo ""
    echo "Choose an option:"
    echo ""
    echo "  1) Quick 24-hour test"
    echo "  2) Standard 72-hour validation (RECOMMENDED)"
    echo "  3) Extended 7-day validation"
    echo "  4) Custom configuration"
    echo "  5) Monitor running validation"
    echo "  6) View latest report"
    echo "  7) Stop running validation"
    echo "  8) Help & Documentation"
    echo "  0) Exit"
    echo ""
    read -p "Enter choice [0-8]: " choice
    echo ""

    case $choice in
        1) start_validation 24 ;;
        2) start_validation 72 ;;
        3) start_validation 168 ;;
        4) custom_validation ;;
        5) monitor_validation ;;
        6) view_report ;;
        7) stop_validation ;;
        8) show_help ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) print_error "Invalid choice"; sleep 2; show_menu ;;
    esac
}

start_validation() {
    local duration=$1

    print_header "Starting Paper Trading Validation"

    # Check dependencies
    echo -e "${BLUE}Checking dependencies...${NC}"

    if ! python3 -c "import loguru, ccxt, pandas, numpy" 2>/dev/null; then
        print_error "Missing required Python packages"
        echo ""
        echo "Install with:"
        echo "  pip install loguru ccxt pandas numpy"
        echo ""
        read -p "Press Enter to continue..."
        show_menu
        return
    fi
    print_success "Python packages installed"

    # Check Neo4j
    if docker ps | grep -q neo4j; then
        print_success "Neo4j is running"
    else
        print_warning "Neo4j is not running"
        echo "Neo4j is recommended but not required for paper trading"
        echo ""
        read -p "Continue anyway? (y/n): " continue_anyway
        if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
            show_menu
            return
        fi
    fi

    echo ""
    echo -e "${GREEN}Configuration:${NC}"
    echo "  Duration: ${duration} hours"
    echo "  Symbols:  BTC/USDT, ETH/USDT, SOL/USDT"
    echo "  Capital:  $100,000"
    echo "  Updates:  Every 30 minutes"
    echo ""

    read -p "Start validation in background? (y/n): " confirm

    if [[ $confirm =~ ^[Yy]$ ]]; then
        # Create log directory
        mkdir -p logs/paper_trading

        # Start in background
        echo ""
        echo -e "${BLUE}Starting validation...${NC}"
        nohup python3 run_extended_paper_trading.py --duration $duration > logs/paper_trading/validation_stdout.log 2>&1 &
        local pid=$!
        echo $pid > paper_trading.pid

        sleep 2

        if ps -p $pid > /dev/null; then
            print_success "Validation started successfully!"
            echo ""
            echo "  PID: $pid"
            echo "  Log: logs/paper_trading/validation_$(date +%Y%m%d_%H%M%S).log"
            echo ""
            echo "Monitor with:"
            echo "  tail -f logs/paper_trading/validation_*.log"
            echo ""
            echo "Or use option 5 in this menu"
        else
            print_error "Failed to start validation"
            echo "Check logs/paper_trading/validation_stdout.log for errors"
        fi

        echo ""
        read -p "Press Enter to continue..."
    fi

    show_menu
}

custom_validation() {
    print_header "Custom Validation Configuration"

    read -p "Duration in hours [72]: " duration
    duration=${duration:-72}

    read -p "Update interval in minutes [30]: " interval
    interval=${interval:-30}

    read -p "Initial capital [100000]: " capital
    capital=${capital:-100000}

    echo ""
    echo "  Duration: ${duration} hours"
    echo "  Interval: ${interval} minutes"
    echo "  Capital:  $${capital}"
    echo ""

    read -p "Start with these settings? (y/n): " confirm

    if [[ $confirm =~ ^[Yy]$ ]]; then
        mkdir -p logs/paper_trading
        nohup python3 run_extended_paper_trading.py \
            --duration $duration \
            --interval $interval \
            --capital $capital \
            > logs/paper_trading/validation_stdout.log 2>&1 &
        local pid=$!
        echo $pid > paper_trading.pid

        print_success "Validation started with PID: $pid"
        echo ""
        read -p "Press Enter to continue..."
    fi

    show_menu
}

monitor_validation() {
    print_header "Monitor Running Validation"

    if [ -f paper_trading.pid ]; then
        local pid=$(cat paper_trading.pid)

        if ps -p $pid > /dev/null; then
            print_success "Validation is running (PID: $pid)"
            echo ""

            # Show latest log lines
            echo -e "${BLUE}Recent log entries:${NC}"
            echo ""

            # Find latest log file
            local latest_log=$(ls -t logs/paper_trading/validation_*.log 2>/dev/null | head -1)

            if [ -n "$latest_log" ]; then
                tail -20 "$latest_log"
            else
                print_warning "No log files found yet"
            fi

            echo ""
            echo "Options:"
            echo "  1) Follow log live"
            echo "  2) View detailed report"
            echo "  3) Back to menu"
            echo ""
            read -p "Choose: " monitor_choice

            case $monitor_choice in
                1)
                    echo "Press Ctrl+C to stop following"
                    sleep 2
                    tail -f logs/paper_trading/validation_*.log
                    ;;
                2)
                    python3 monitor_paper_trading.py --report
                    ;;
                3)
                    ;;
            esac
        else
            print_error "Validation is not running"
            rm -f paper_trading.pid
        fi
    else
        print_warning "No validation PID file found"
        echo "No validation appears to be running"
    fi

    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

view_report() {
    print_header "Latest Validation Report"

    python3 monitor_paper_trading.py --report

    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

stop_validation() {
    print_header "Stop Validation"

    if [ -f paper_trading.pid ]; then
        local pid=$(cat paper_trading.pid)

        if ps -p $pid > /dev/null; then
            echo "Stopping validation (PID: $pid)..."
            kill $pid
            sleep 2

            if ps -p $pid > /dev/null; then
                print_warning "Validation did not stop gracefully"
                read -p "Force kill? (y/n): " force_kill
                if [[ $force_kill =~ ^[Yy]$ ]]; then
                    kill -9 $pid
                    print_success "Validation force stopped"
                fi
            else
                print_success "Validation stopped successfully"
            fi

            rm -f paper_trading.pid
        else
            print_error "Validation is not running"
            rm -f paper_trading.pid
        fi
    else
        print_warning "No validation PID file found"
    fi

    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

show_help() {
    print_header "Help & Documentation"

    echo -e "${GREEN}Paper Trading Validation Guide${NC}"
    echo ""
    echo "This tool helps you validate GraphWiz Trader before live trading."
    echo ""
    echo -e "${BLUE}Quick Start:${NC}"
    echo "  1. Choose option 2 (72-hour validation)"
    echo "  2. Let it run for 3 days"
    echo "  3. Check performance with option 6"
    echo "  4. If metrics are good, proceed to live trading"
    echo ""
    echo -e "${BLUE}Target Metrics for Live Trading:${NC}"
    echo "  • Total Return:  > 5%"
    echo "  • Win Rate:     > 50%"
    echo "  • Max Drawdown: < 20%"
    echo "  • Sharpe Ratio: > 1.0"
    echo ""
    echo -e "${BLUE}Files Generated:${NC}"
    echo "  • logs/paper_trading/validation_*.log      - Main log"
    echo "  • logs/paper_trading/trades_*.csv          - Trade history"
    echo "  • logs/paper_trading/equity_*.csv          - Portfolio value"
    echo "  • data/paper_trading/validation_report_*.json - Final report"
    echo ""
    echo -e "${BLUE}Documentation:${NC}"
    echo "  • PAPER_TRADING_QUICKSTART.md - Detailed guide"
    echo "  • docs/PAPER_TRADING.md - Comprehensive docs"
    echo ""
    echo -e "${BLUE}Troubleshooting:${NC}"
    echo "  Problem: No trades executed"
    echo "  Solution: Market conditions may be calm. Run longer or adjust RSI"
    echo ""
    echo "  Problem: Import errors"
    echo "  Solution: pip install loguru ccxt pandas numpy"
    echo ""
    echo "  Problem: Neo4j connection issues"
    echo "  Solution: docker-compose up -d neo4j"
    echo ""

    read -p "Press Enter to continue..."
    show_menu
}

# Check if we're in the right directory
if [ ! -f "run_extended_paper_trading.py" ]; then
    print_error "Must run from GraphWiz Trader root directory"
    echo "cd to the project root first"
    exit 1
fi

# Start the menu
show_menu
