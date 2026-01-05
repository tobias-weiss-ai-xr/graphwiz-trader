"""
Live trading engine for real money trading.

⚠️ WARNING: This executes REAL trades with REAL money.
Use extreme caution and test thoroughly with paper trading first.
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
from loguru import logger
from ccxt import Exchange

from ..trading.exchange import create_exchange
from ..backtesting import RSIMeanReversionStrategy
from .safety_limits import SafetyLimits
from .risk_manager import RiskManager


class LiveTradingEngine:
    """Live trading engine for real money trading."""

    def __init__(
        self,
        exchange_name: str = "binance",
        symbol: str = "BTC/USDT",
        api_key: str = "",
        api_secret: str = "",
        strategy_config: Optional[Dict[str, Any]] = None,
        safety_limits: Optional[SafetyLimits] = None,
    ):
        """Initialize live trading engine.

        ⚠️ WARNING: This will execute REAL trades!

        Args:
            exchange_name: Exchange to use
            symbol: Trading pair symbol
            api_key: Exchange API key
            api_secret: Exchange API secret
            strategy_config: Strategy configuration
            safety_limits: Safety limits configuration
        """
        if not api_key or not api_secret:
            raise ValueError(
                "API credentials required for live trading. "
                "Set API_KEY and API_SECRET environment variables."
            )

        self.exchange_name = exchange_name
        self.symbol = symbol

        # Create exchange with API credentials
        self.exchange: Exchange = create_exchange(
            exchange_name, api_key=api_key, api_secret=api_secret
        )

        # Safety limits
        self.safety_limits = safety_limits or SafetyLimits()

        # Risk manager
        self.risk_manager = RiskManager(self.safety_limits)

        # Strategy
        strategy_config = strategy_config or {"oversold": 25, "overbought": 65}
        self.strategy = RSIMeanReversionStrategy(**strategy_config)

        # State
        self.is_running = False
        self.last_signal = None
        self.trade_history = []

        logger.warning("=" * 80)
        logger.warning("🚨 LIVE TRADING MODE - REAL MONEY WILL BE USED")
        logger.warning("=" * 80)
        logger.warning(f"Exchange: {exchange_name}")
        logger.warning(f"Symbol: {symbol}")
        logger.warning(
            f"Strategy: RSI({strategy_config.get('oversold')}/{strategy_config.get('overbought')})"
        )
        logger.warning("=" * 80)

    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance.

        Returns:
            Dictionary with balance info
        """
        try:
            balance = self.exchange.fetch_balance()

            # Get base and quote currencies
            base, quote = self.symbol.split("/")

            quote_balance = balance.get(quote, {}).get("free", 0.0)
            base_balance = balance.get(base, {}).get("free", 0.0)

            return {
                "quote_currency": quote,
                "quote_balance": float(quote_balance),
                "base_currency": base,
                "base_balance": float(base_balance),
            }

        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise

    def fetch_current_price(self) -> float:
        """Fetch current market price.

        Returns:
            Current price
        """
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return float(ticker["last"])

        except Exception as e:
            logger.error(f"Failed to fetch price: {e}")
            raise

    def fetch_latest_data(self, limit: int = 100) -> pd.DataFrame:
        """Fetch latest market data.

        Args:
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, "1d", limit=limit)

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """Generate trading signal.

        Args:
            data: Market data DataFrame

        Returns:
            Signal: "buy", "sell", or None
        """
        return self.strategy.generate_signal(data)

    def execute_buy_order(self, quantity: float) -> Dict[str, Any]:
        """Execute a buy order.

        ⚠️ WARNING: This spends REAL money!

        Args:
            quantity: Quantity to buy

        Returns:
            Order result
        """
        try:
            logger.warning(f"⚠️  EXECUTING BUY ORDER: {quantity:.4f} {self.symbol}")

            # Execute market buy order
            order = self.exchange.create_market_buy_order(self.symbol, quantity)

            logger.success(f"✅ BUY ORDER EXECUTED: {order}")

            self.trade_history.append(
                {
                    "timestamp": datetime.now(),
                    "action": "buy",
                    "symbol": self.symbol,
                    "quantity": quantity,
                    "order": order,
                }
            )

            return order

        except Exception as e:
            logger.error(f"❌ BUY ORDER FAILED: {e}")
            raise

    def execute_sell_order(self, quantity: float) -> Dict[str, Any]:
        """Execute a sell order.

        ⚠️ WARNING: This sells REAL assets!

        Args:
            quantity: Quantity to sell

        Returns:
            Order result
        """
        try:
            logger.warning(f"⚠️  EXECUTING SELL ORDER: {quantity:.4f} {self.symbol}")

            # Execute market sell order
            order = self.exchange.create_market_sell_order(self.symbol, quantity)

            logger.success(f"✅ SELL ORDER EXECUTED: {order}")

            self.trade_history.append(
                {
                    "timestamp": datetime.now(),
                    "action": "sell",
                    "symbol": self.symbol,
                    "quantity": quantity,
                    "order": order,
                }
            )

            return order

        except Exception as e:
            logger.error(f"❌ SELL ORDER FAILED: {e}")
            raise

    def run_once(self) -> Dict[str, Any]:
        """Run one iteration of live trading.

        Returns:
            Status dict
        """
        try:
            # Get account balance
            balance = self.get_account_balance()
            portfolio_value = balance["quote_balance"]

            # Fetch market data
            data = self.fetch_latest_data(limit=100)
            current_price = self.fetch_current_price()

            # Generate signal
            signal = self.generate_signal(data)

            action_taken = None
            order_result = None

            if signal:
                # Check if trade is allowed
                can_trade, reason, quantity = self.risk_manager.can_trade(
                    signal=signal,
                    symbol=self.symbol,
                    portfolio_value=portfolio_value,
                    current_price=current_price,
                    current_time=datetime.now(),
                )

                if can_trade:
                    # Check for existing positions
                    base, quote = self.symbol.split("/")
                    has_position = balance["base_balance"] > 0

                    if signal == "buy" and not has_position:
                        # Execute buy order
                        order_result = self.execute_buy_order(quantity)

                        # Add to risk manager
                        self.risk_manager.add_position(
                            symbol=self.symbol,
                            side="long",
                            quantity=quantity,
                            entry_price=current_price,
                            stop_loss=current_price * 0.98,  # 2% stop loss
                            take_profit=current_price * 1.05,  # 5% take profit
                        )

                        action_taken = "buy"

                    elif signal == "sell" and has_position:
                        # Execute sell order
                        sell_quantity = balance["base_balance"]
                        order_result = self.execute_sell_order(sell_quantity)

                        # Update risk manager
                        pnl = self.risk_manager.close_position(self.symbol, current_price)

                        action_taken = "sell"

                    else:
                        logger.info(f"Signal {signal} but no action taken (already positioned)")

                else:
                    logger.warning(f"Trade blocked by safety limits: {reason}")

            # Check existing positions for risk management
            current_prices = {self.symbol: current_price}
            risk_actions = self.risk_manager.check_positions(current_prices)

            for symbol_pos, should_close, reason in risk_actions:
                if should_close:
                    logger.warning(f"Risk management: {reason}")
                    # Force close position
                    balance = self.get_account_balance()
                    if balance["base_balance"] > 0:
                        self.execute_sell_order(balance["base_balance"])
                        self.risk_manager.close_position(symbol_pos, current_price)

            return {
                "status": "success",
                "timestamp": datetime.now(),
                "price": current_price,
                "signal": signal,
                "action_taken": action_taken,
                "order_result": order_result,
                "portfolio_value": portfolio_value,
                "daily_pnl": self.risk_manager.daily_pnl,
            }

        except Exception as e:
            logger.error(f"Error in live trading iteration: {e}")
            return {"status": "error", "message": str(e)}

    def start(self, interval_seconds: int = 3600):
        """Start continuous live trading.

        ⚠️ WARNING: This will execute REAL trades continuously!

        Args:
            interval_seconds: Seconds between iterations
        """
        self.is_running = True

        logger.warning("")
        logger.warning("=" * 80)
        logger.warning("🚨 STARTING LIVE TRADING")
        logger.warning("=" * 80)
        logger.warning(f"Interval: {interval_seconds}s")
        logger.warning(f"Max position: ${self.safety_limits.max_position_size:,.2f}")
        logger.warning(f"Max daily loss: ${self.safety_limits.max_daily_loss:,.2f}")
        logger.warning(f"Max daily trades: {self.safety_limits.max_daily_trades}")
        logger.warning("=" * 80)
        logger.warning("")

        # Require confirmation
        if self.safety_limits.require_confirmation:
            response = input("Type 'CONFIRM' to start live trading: ")
            if response != "CONFIRM":
                logger.error("Live trading cancelled")
                return

        try:
            while self.is_running:
                logger.info(f"\n{'='*60}")
                logger.info(f"Live Trading Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")

                # Run one iteration
                result = self.run_once()

                if result["status"] == "success":
                    logger.info(
                        f"Price: ${result['price']:,.2f}, "
                        f"Signal: {result['signal']}, "
                        f"Action: {result['action_taken'] or 'None'}, "
                        f"Daily P&L: ${result['daily_pnl']:+,.2f}"
                    )
                else:
                    logger.error(f"Iteration failed: {result.get('message')}")

                # Save trade history
                self.save_results()

                # Wait for next iteration
                logger.info(f"Waiting {interval_seconds}s until next check...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\n\nReceived interrupt signal")

        finally:
            self.is_running = False
            self.save_results()
            logger.warning("Live trading stopped")

    def stop(self):
        """Stop live trading."""
        self.is_running = False
        logger.warning("Live trading stop requested")

    def save_results(self, output_dir: str = "data/live_trading"):
        """Save trading results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol_clean = self.symbol.replace("/", "_")

        # Save trade history
        if self.trade_history:
            history_file = output_path / f"{symbol_clean}_history_{timestamp_str}.json"
            with open(history_file, "w") as f:
                json.dump(self.trade_history, f, indent=2, default=str)

        # Save portfolio summary
        summary = self.risk_manager.get_portfolio_summary()
        summary_file = output_path / f"{symbol_clean}_summary_{timestamp_str}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def print_summary(self):
        """Print trading summary."""
        summary = self.risk_manager.get_portfolio_summary()

        print("\n" + "=" * 80)
        print("LIVE TRADING SUMMARY")
        print("=" * 80)
        print(f"Symbol: {self.symbol}")
        print(f"Open Positions: {summary['open_positions']}")
        print(f"Daily P&L: ${summary['daily_pnl']:+,.2f}")
        print(f"Daily Trades: {summary['daily_trade_count']}")
        print("-" * 80)

        for symbol, pos in summary["positions"].items():
            print(f"{symbol}: {pos['side']} {pos['quantity']:.4f} @ ${pos['entry_price']:,.2f}")

        print("=" * 80 + "\n")
