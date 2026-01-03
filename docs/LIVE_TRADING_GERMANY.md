# Live Trading Setup for German Users

## 🇩🇪 Germany-Specific Live Trading Guide

This guide covers setting up live trading for German users with **BaFin-licensed exchanges** under **MiCA regulation**.

---

## ⚠️ CRITICAL WARNING

**LIVE TRADING INVOLVES REAL MONEY AND REAL RISKS**

- You will execute **REAL trades** with **REAL money**
- Past performance does **NOT** guarantee future results
- Start with small amounts (€500 or less)
- Monitor trades closely
- **Only use BaFin-licensed exchanges**

---

## 📜 Regulatory Status (2026)

### ✅ APPROVED Exchanges for German Users

| Exchange | License Status | Regulator | Available |
|----------|---------------|-----------|-----------|
| **Kraken** | ✅ MiCA License | BaFin | August 2025 |
| **Bitpanda** | ✅ MiCA License | BaFin | January 2025 |

### ❌ NOT APPROVED for German Users

| Exchange | Status | Issue |
|----------|--------|-------|
| **Binance** | ❌ Not Licensed | BaFin **denied** custody license application in 2023 |

**IMPORTANT**: Binance is **NOT** compliant for German users. Do not use Binance for live trading in Germany.

---

## 🚀 Quick Start

### Prerequisites

1. **Kraken Account** (or Bitpanda)
   - Fully verified account
   - 2FA enabled
   - API keys generated

2. **Paper Trading Completion**
   - Minimum 72 hours of paper trading
   - Consistent performance
   - Understanding of system behavior

3. **Technical Requirements**
   - Python 3.11+
   - 4GB RAM minimum
   - Stable internet connection
   - Linux/macOS/WSL2 environment

### Step 1: Obtain Kraken API Credentials

1. Log in to [Kraken](https://www.kraken.com)
2. Go to **Settings** → **API**
3. Create a new API key with the following permissions:
   - ✅ Query funds/balances
   - ✅ Query open orders/closed orders
   - ✅ Place/cancel orders
   - ❌ Withdraw funds (DO NOT enable for trading bot)

4. Set IP whitelist (recommended):
   - Add your server's IP address
   - Restricts access to specific IPs only

5. Save your API Key and API Secret securely

### Step 2: Configure Environment

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file and add your Kraken credentials:**
   ```bash
   # Kraken API Credentials
   KRAKEN_API_KEY=your_kraken_api_key_here
   KRAKEN_API_SECRET=your_kraken_api_secret_here
   ```

3. **Add Neo4j credentials (if using knowledge graph):**
   ```bash
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_neo4j_password
   ```

4. **Save and close the file**

### Step 3: Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 4: Test Connection

Test your Kraken connection before starting live trading:

```bash
python scripts/live_trade.py \
  --exchange kraken \
  --symbol BTC/EUR \
  --test
```

This will:
- Connect to Kraken
- Fetch current price
- Generate trading signal
- **NOT execute any trades** (test mode)

### Step 5: Start Live Trading

**Option A: Interactive Menu**

```bash
./deploy_live_trading_germany.sh
```

Follow the prompts to start live trading.

**Option B: Direct Start**

```bash
./deploy_live_trading_germany.sh start
```

### Step 6: Monitor Trading

**View real-time logs:**
```bash
./deploy_live_trading_germany.sh logs
```

**Check status:**
```bash
./deploy_live_trading_germany.sh status
```

**Stop trading:**
```bash
./deploy_live_trading_germany.sh stop
```

---

## 📊 Configuration Options

### Default Safety Limits (config/germany_live.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_position_eur` | €500 | Maximum position size |
| `max_daily_loss_eur` | €150 | Maximum daily loss |
| `max_daily_trades` | 3 | Maximum trades per day |
| `stop_loss_pct` | 2% | Stop loss percentage |
| `take_profit_pct` | 4% | Take profit percentage |
| `cooldown_seconds` | 3600 | Cooldown between trades |

### Customizing Limits

Edit `config/germany_live.yaml`:

```yaml
live_trading:
  max_position_eur: 1000  # Increase to €1000
  max_daily_loss_eur: 300  # Increase to €300
  max_daily_trades: 5  # Increase to 5 trades per day
```

**WARNING**: Only increase limits if you have experience and can afford larger losses.

---

## 🔒 Security Best Practices

### 1. API Key Security

- ✅ **DO**:
  - Use IP whitelisting
  - Set withdrawal permissions to OFF
  - Rotate keys periodically
  - Never commit `.env` to git

- ❌ **DON'T**:
  - Share API keys
  - Enable withdrawal permissions
  - Use production keys for testing

### 2. System Security

```bash
# Set restrictive file permissions
chmod 600 .env
chmod 700 logs/

# Use firewall to restrict access
sudo ufw allow from YOUR_IP to any port 22
sudo ufw enable
```

### 3. Financial Security

- Start with minimum amounts (€500 or less)
- Use Kraken's withdrawal whitelist
- Enable 2FA on Kraken account
- Monitor account activity daily

---

## 📈 Trading Strategy

### Default Strategy: RSI Mean Reversion

**Entry Signals:**
- **BUY**: RSI < 25 (oversold)
- **SELL**: RSI > 75 (overbought)

**Risk Management:**
- Stop loss: 2% below entry
- Take profit: 4% above entry
- Trailing stop: 1%

**Confirmation:**
- Volume confirmation required
- 2-period confirmation
- Minimum 24h volume: €1M

### Customizing Strategy

Edit `config/germany_live.yaml`:

```yaml
strategy:
  name: "rsi_mean_reversion"
  parameters:
    oversold: 20  # More aggressive buy
    overbought: 80  # More aggressive sell
    rsi_period: 14
```

---

## 🛠️ Troubleshooting

### Issue: "API key not found"

**Solution:**
1. Check `.env` file exists
2. Verify `KRAKEN_API_KEY` is set
3. Ensure no typos in credentials

### Issue: "Connection failed"

**Solution:**
1. Check internet connection
2. Verify Kraken API status: https://status.kraken.com
3. Check IP whitelist settings

### Issue: "Insufficient funds"

**Solution:**
1. Verify EUR balance in Kraken
2. Check `max_position_eur` setting
3. Ensure sufficient funds for trade + fees

### Issue: "Order rejected"

**Solution:**
1. Check trading pair availability
2. Verify minimum order size (Kraken: ~€10-50)
3. Ensure account is verified

---

## 📚 Additional Resources

### Official Documentation

- **[GraphWiz Trader Main Docs](../README.md)**
- **[Kraken API Documentation](https://docs.kraken.com/rest/)**
- **[BaFin Crypto Supervision](https://www.bafin.de/DE/Aufgaben/Aufsichtsmaessnahmen/Kryptowaehrungen/kryptowaehrungen_node.html)**

### Regulatory Information

- **[MiCA Regulation](https://www.europarl.europa.eu/topics/article/20230601STO93812/markets-in-crypto-assets-mica)**
- **[Kraken Germany Updates](https://support.kraken.com/articles/updates-for-clients-in-germany)**

### Community

- **[GitHub Issues](https://github.com/your-repo/issues)**
- **[Discord Server](https://discord.gg/your-server)**

---

## ⚖️ Legal Disclaimer

```
This software is provided for educational purposes only.

LIVE TRADING INVOLVES SUBSTANTIAL RISK OF LOSS AND IS NOT SUITABLE
FOR ALL INVESTORS. YOU SHOULD CAREFULLY CONSIDER WHETHER TRADING
IS APPROPRIATE FOR YOU IN LIGHT OF YOUR CIRCUMSTANCES, KNOWLEDGE,
AND FINANCIAL RESOURCES.

THE AUTHORS ARE NOT REGISTERED INVESTMENT ADVISORS AND DO NOT
PROVIDE INVESTMENT ADVICE. PAST PERFORMANCE DOES NOT GUARANTEE
FUTURE RESULTS.

YOU ARE SOLELY RESPONSIBLE FOR YOUR TRADING DECISIONS AND ALL
TRADING RISKS. USE AT YOUR OWN RISK.

This software uses Kraken exchange, which is licensed under MiCA
for German users. However, regulations may change. Always verify
current regulatory status before trading.
```

---

## 📞 Support

If you encounter issues:

1. **Check logs**: `logs/live_trading/live_trading_output.log`
2. **Review configuration**: `config/germany_live.yaml`
3. **Test connection**: Run `--test` flag first
4. **Open GitHub issue**: Include logs and configuration

---

## 🔄 Updates & Maintenance

### Recommended Daily Checks

- [ ] Review trading logs
- [ ] Verify account balance
- [ ] Check for system updates
- [ ] Monitor open positions

### Recommended Weekly Tasks

- [ ] Review weekly performance
- [ ] Analyze trade history
- [ ] Adjust strategy parameters if needed
- [ ] Backup configuration and logs

### Recommended Monthly Tasks

- [ ] Rotate API keys
- [ ] Review and update security settings
- [ ] Verify regulatory compliance
- [ ] Update system dependencies

---

**Last Updated**: January 2026
**License**: MIT
**Jurisdiction**: Germany (BaFin-regulated)

Sources:
- [10 Best Crypto Exchanges in Germany (January 2026)](https://koinly.io/blog/best-crypto-exchanges-germany/)
- [Bitpanda secures MiCAR licence](https://blog.bitpanda.com/en/bitpanda-secures-micar-licence)
- [Where is Kraken licensed or regulated?](https://support.kraken.com/articles/where-is-kraken-licensed-or-regulated)
- [Is Binance Legal in Germany?](https://www.binance.com/en/square/post/28212157431689)
