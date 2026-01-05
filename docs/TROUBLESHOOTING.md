# Troubleshooting Guide

Common issues and solutions for GraphWiz Trader.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Connection Issues](#connection-issues)
3. [Performance Issues](#performance-issues)
4. [Trading Issues](#trading-issues)
5. [Agent Issues](#agent-issues)
6. [Database Issues](#database-issues)
7. [Testing Issues](#testing-issues)
8. [Deployment Issues](#deployment-issues)

---

## Installation Issues

### Python Version Error

**Problem:** `Python 3.10+ required` error

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.10+
sudo apt update
sudo apt install python3.10 python3.10-venv

# Create venv with correct version
python3.10 -m venv venv
```

### Dependency Installation Failed

**Problem:** `pip install` fails with dependency errors

**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install with specific versions
pip install -r requirements.txt --no-cache-dir

# If still failing, install individually
pip install neo4j==5.15.0
pip install ccxt==4.4.99
pip install langchain==0.1.0
```

### Neo4j Driver Not Found

**Problem:** `ImportError: No module named 'neo4j'`

**Solution:**
```bash
pip install neo4j==5.15.0

# Verify installation
python -c "import neo4j; print(neo4j.__version__)"
```

---

## Connection Issues

### Neo4j Connection Failed

**Problem:** `Failed to connect to Neo4j at bolt://localhost:7687`

**Solutions:**

1. Check Neo4j is running:
```bash
sudo systemctl status neo4j
sudo systemctl start neo4j
```

2. Check port is open:
```bash
netstat -tlnp | grep 7687
telnet localhost 7687
```

3. Verify credentials in config:
```yaml
neo4j:
  uri: bolt://localhost:7687
  username: neo4j
  password: your_password  # Verify this is correct
```

4. Check Neo4j logs:
```bash
sudo tail -f /var/log/neo4j/neo4j.log
```

### Exchange API Connection Failed

**Problem:** `Authentication failed for exchange`

**Solutions:**

1. Verify API keys:
```bash
# Check .env file
cat .env | grep API
```

2. Test API connection manually:
```python
import ccxt
exchange = ccxt.binance({
    'apiKey': 'your_key',
    'secret': 'your_secret',
    'enableRateLimit': True
})
balance = exchange.fetch_balance()
print(balance)
```

3. Check if IP is whitelisted (exchange requirement)

4. Verify sandbox mode for testnet:
```yaml
exchanges:
  binance:
    sandbox: true  # Must be true for testnet
```

### WebSocket Connection Drops

**Problem:** WebSocket keeps disconnecting

**Solution:**
```yaml
# config.yaml
websocket:
  ping_interval: 30
  ping_timeout: 10
  auto_reconnect: true
  max_reconnect_attempts: 5
```

---

## Performance Issues

### High Memory Usage

**Problem:** System using too much memory

**Solutions:**

1. Reduce knowledge graph cache:
```yaml
neo4j:
  cache_ttl: 300  # Reduce from 600
  connection_pool_size: 20  # Reduce from 50
```

2. Limit historical data loaded:
```python
# Load only recent data
data = data.tail(10000)  # Last 10k candles
```

3. Disable unused agents:
```yaml
agents:
  sentiment:
    enabled: false  # Disable if not needed
```

4. Clear cache regularly:
```python
kg.clear_cache()
```

### Slow Agent Response

**Problem:** Agents taking too long to respond

**Solutions:**

1. Use faster model:
```yaml
agents:
  technical:
    model: "gpt-3.5-turbo"  # Instead of gpt-4
    timeout: 10  # Reduce timeout
```

2. Reduce max_tokens:
```yaml
agents:
  technical:
    max_tokens: 200  # Reduce from 500
```

3. Enable caching:
```yaml
agents:
  technical:
    use_cache: true
    cache_ttl: 300
```

4. Run agents in parallel:
```python
# In orchestrator
decisions = await asyncio.gather(*[
    agent.analyze(market_data)
    for agent in agents.values()
])
```

### High CPU Usage

**Problem:** CPU usage at 100%

**Solutions:**

1. Reduce frequency of operations:
```python
# Increase interval between checks
TRADING_INTERVAL = 60  # seconds (instead of 30)
```

2. Limit concurrent operations:
```python
# In configuration
max_concurrent_requests = 5  # Reduce from 10
```

3. Profile code to find bottlenecks:
```bash
python -m cProfile -o profile.stats your_script.py
python -m pstats profile.stats
# Type: stats 10  # Top 10 functions
```

---

## Trading Issues

### Order Rejected

**Problem:** Orders being rejected by exchange

**Solutions:**

1. Check account balance:
```python
balance = exchange.fetch_balance()
print(balance['USDT']['free'])
```

2. Verify order size limits:
```python
markets = exchange.load_markets()
min_order = markets['BTC/USDT']['limits']['amount']['min']
print(f"Minimum order: {min_order}")
```

3. Check if market is open:
```python
market = exchange.fetch_market('BTC/USDT')
print(market['active'])
```

### Slippage Too High

**Problem:** Orders executing at poor prices

**Solutions:**

1. Use limit orders instead of market:
```yaml
trading:
  order_type: "limit"  # Instead of "market"
```

2. Reduce position size:
```python
# Smaller orders = less slippage
quantity /= 2
```

3. Check liquidity before trading:
```python
orderbook = exchange.fetch_order_book('BTC/USDT')
bid_volume = sum([bid[1] for bid in orderbook['bids'][:5]])
if bid_volume < min_liquidity:
    print("Insufficient liquidity")
```

### Positions Not Closing

**Problem:** Exit signals not triggering position closes

**Solutions:**

1. Check exit conditions:
```python
# Verify stop loss logic
if current_price <= position['stop_loss']:
    await engine.close_position(symbol, "Stop loss hit")
```

2. Check if positions are tracked:
```python
positions = engine.get_positions()
print(f"Open positions: {len(positions)}")
```

3. Enable logging:
```yaml
logging:
  level: "DEBUG"
  file: "logs/trading_debug.log"
```

---

## Agent Issues

### Agent Timeout

**Problem:** Agent analysis timing out

**Solutions:**

1. Increase timeout:
```yaml
agents:
  technical:
    timeout: 30  # Increase from 10
```

2. Add retry logic:
```python
max_retries = 3
for i in range(max_retries):
    try:
        decision = await agent.analyze(market_data)
        break
    except TimeoutError:
        if i == max_retries - 1:
            raise
```

3. Use fallback agent:
```python
try:
    decision = await primary_agent.analyze(market_data)
except TimeoutError:
    decision = await fallback_agent.analyze(market_data)
```

### Low Confidence Decisions

**Problem:** Agents returning low confidence scores

**Solutions:**

1. Adjust confidence threshold:
```yaml
trading:
  min_confidence: 0.65  # Lower from 0.75
```

2. Improve input data quality:
```python
# Ensure all indicators are calculated
market_data['indicators'] = {
    'rsi': calculate_rsi(prices),
    'macd': calculate_macd(prices),
    # Add more indicators
}
```

3. Tune agent parameters:
```yaml
agents:
  technical:
    temperature: 0.5  # Lower = more confident
```

### Agent Disagreement

**Problem:** Agents giving conflicting signals

**Solutions:**

1. Adjust agent weights:
```yaml
agents:
  technical:
    weight: 0.5  # Increase weight
  sentiment:
    weight: 0.3  # Decrease weight
```

2. Use consensus threshold:
```python
# Require 75% agreement
consensus_threshold = 0.75
if agreement_score < consensus_threshold:
    action = "hold"  # Default to hold
```

3. Add tie-breaker agent:
```yaml
agents:
  tie_breaker:
    enabled: true
    model: "gpt-4"
```

---

## Database Issues

### Neo4j Slow Queries

**Problem:** Knowledge graph queries are slow

**Solutions:**

1. Add indexes:
```cypher
CREATE INDEX asset_symbol_index FOR (a:Asset) ON (a.symbol);
CREATE INDEX market_timestamp_index FOR (m:Market) ON (m.timestamp);
```

2. Optimize queries:
```cypher
// Instead of:
MATCH (n) WHERE n.symbol = 'BTC' RETURN n

// Use parameterized:
MATCH (n:Asset {symbol: $symbol}) RETURN n
```

3. Use query caching:
```yaml
neo4j:
  enable_query_cache: true
  cache_ttl: 600
```

4. Monitor slow queries:
```cypher
CALL dbms.listQueries()
YIELD query, elapsedTime
WHERE elapsedTime > 1000
RETURN query, elapsedTime
```

### Database Connection Pool Exhausted

**Problem:** Too many connections to Neo4j

**Solution:**
```yaml
neo4j:
  connection_pool_size: 50  # Increase pool size
  max_connection_lifetime: 3600
  connection_acquisition_timeout: 60
```

---

## Testing Issues

### Tests Failing with Import Errors

**Problem:** `ModuleNotFoundError: No module named 'graphwiz_trader'`

**Solution:**
```bash
# Install package in editable mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/graphwiz-trader/src"
```

### Coverage Below 80%

**Problem:** Test coverage target not met

**Solutions:**

1. Run tests with coverage report:
```bash
pytest tests/ --cov=src/graphwiz_trader --cov-report=html
open htmlcov/index.html
```

2. Identify untested code:
```bash
# View missing lines
pytest tests/ --cov=src/graphwiz_trader --cov-report=term-missing
```

3. Add tests for missing coverage:
```python
# Test edge cases
def test_edge_case():
    result = engine.execute_trade_signal({
        "price": -1  # Invalid price
    })
    assert result["status"] == "error"
```

### Integration Tests Failing

**Problem:** Integration tests fail due to external dependencies

**Solutions:**

1. Mock external dependencies:
```python
@pytest.fixture
def mock_exchange():
    exchange = MagicMock()
    exchange.create_order = AsyncMock(return_value={"id": "12345"})
    return exchange
```

2. Use test configuration:
```bash
cp config/paper_trading.yaml config/test_config.yaml
# Update with test values
pytest tests/ --config=config/test_config.yaml
```

3. Run tests in isolation:
```bash
pytest tests/integration/ -k "test_specific_test" -v
```

---

## Deployment Issues

### Docker Container Crashing

**Problem:** Docker container exits immediately

**Solutions:**

1. Check logs:
```bash
docker logs graphwiz-trader
```

2. Run in foreground to see errors:
```bash
docker run -it graphwiz-trader:latest
```

3. Check health status:
```bash
docker inspect graphwiz-trader --format='{{.State.Health.Status}}'
```

4. Verify environment variables:
```bash
docker exec graphwiz-trader env | grep NEO4J
```

### High Latency in Production

**Problem:** Production system slow compared to development

**Solutions:**

1. Use faster hosting (closer to exchanges):
- Consider cloud servers in exchange's region
- Use low-latency VPS services

2. Enable WebSocket connections:
```yaml
exchanges:
  binance:
    enable_websocket: true
```

3. Optimize database:
```bash
# Use Neo4j enterprise for better performance
# Or use graph data science library
```

4. Profile and optimize:
```python
import cProfile
cProfile.run('engine.execute_trade_signal(data)', 'profile_output')
```

### Memory Leaks

**Problem:** Memory usage grows over time

**Solutions:**

1. Monitor memory:
```bash
watch -n 5 'ps aux | grep graphwiz'
```

2. Clear caches regularly:
```python
# Add to main loop
if iteration % 100 == 0:
    kg.clear_cache()
    gc.collect()
```

3. Use memory profiler:
```bash
pip install memory_profiler
python -m memory_profiler your_script.py
```

4. Restart periodically:
```bash
# Add to crontab
0 3 * * * /path/to/restart_script.sh
```

---

## Getting Help

If you can't resolve your issue:

1. **Check logs first:**
   ```bash
   tail -f logs/graphwiz-trader.log
   ```

2. **Search existing issues:**
   [GitHub Issues](https://github.com/tobias-weiss-ai-xr/graphwiz-trader/issues)

3. **Create new issue with:**
   - GraphWiz Trader version
   - Python version
   - Error message
   - Steps to reproduce
   - Relevant logs

4. **Consult documentation:**
   - [README.md](README.md)
   - [DEPLOYMENT.md](DEPLOYMENT.md)
   - [API.md](API.md)

---

## Debug Mode

Enable debug mode for detailed logging:

```yaml
# config/debug.yaml
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/debug.log"

monitoring:
  debug_mode: true
  profile_performance: true
```

Run with debug config:
```bash
python -m graphwiz_trader.main --config config/debug.yaml
```
