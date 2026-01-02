#!/bin/bash
# Start Agent Looper for GraphWiz Trader optimization

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Starting Agent Looper for GraphWiz Trader               ║"
echo "║     Autonomous Trading Optimization System                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Change to agent-looper directory
cd /opt/git/agent-looper

# Activate virtual environment
source venv/bin/activate

# Check SAIA keys
if [ ! -f .saia-keys ]; then
    echo "❌ ERROR: .saia-keys file not found!"
    echo "   Please create .saia-keys with your SAIA API keys"
    echo "   Format: One key per line"
    exit 1
fi

KEY_COUNT=$(grep -v "^#" .saia-keys | grep -v "^$" | wc -l)
echo "✓ Found $KEY_COUNT SAIA API key(s)"
echo ""

# Check configuration
if [ ! -f src/projects/graphwiz-trader/config.yaml ]; then
    echo "❌ ERROR: Configuration file not found!"
    exit 1
fi

echo "✓ Configuration loaded"
echo ""

# Display optimization info
echo "Optimization Configuration:"
echo "  Mode: PAPER TRADING (Safe)"
echo "  Dry Run: YES (No actual changes)"
echo "  Approval: Required for critical changes"
echo ""

echo "Optimization Goals:"
echo "  • Maximize Sharpe Ratio (Target: 2.5)"
echo "  • Minimize Max Drawdown (Target: < 8%)"
echo "  • Maximize Win Rate (Target: > 65%)"
echo "  • Maximize Profit Factor (Target: 2.5)"
echo "  • Improve Agent Accuracy (Target: 70%)"
echo ""

echo "Optimization Schedule:"
echo "  • Strategy Parameters: Daily"
echo "  • Risk Limits: Weekly"
echo "  • Agent Weights: Daily (auto-approve)"
echo "  • Trading Pairs: Weekly"
echo "  • Indicators: Monthly"
echo ""

echo "═════════════════════════════════════════════════════════════"
echo ""

# Start the optimizer
echo "🚀 Starting Agent Looper..."
echo ""

# Run in background with logging
nohup python3 run_optimizer.py > logs/optimizer_output.log 2>&1 &
OPTIMIZER_PID=$!

echo "✓ Agent Looper started (PID: $OPTIMIZER_PID)"
echo ""
echo "Logs:"
echo "  • Main log: tail -f /opt/git/graphwiz-trader/logs/optimizer_*.log"
echo "  • Output log: tail -f /opt/git/agent-looper/logs/optimizer_output.log"
echo ""
echo "Status Check:"
echo "  • Process: ps aux | grep $OPTIMIZER_PID"
echo "  • Stop: kill $OPTIMIZER_PID"
echo ""
echo "═════════════════════════════════════════════════════════════"
echo ""
echo "✨ Agent Looper is now running autonomously!"
echo "   It will optimize trading parameters safely in paper trading mode."
echo ""
