"""
Germany-compliant exchange integrations for live trading.

This module provides enhanced exchange integrations specifically for
German users, focusing on BaFin-licensed exchanges under MiCA regulation.

Supported exchanges:
- Kraken (MiCA-licensed from August 2025)
- One Trading (formerly Bitpanda Pro, MiCA-licensed from January 2025)
"""

import ccxt
from typing import Optional, Dict, Any, List
from loguru import logger
from decimal import Decimal


class GermanExchange:
    """Base class for German exchange integrations."""

    def __init__(self, exchange_id: str, config: Dict[str, Any]):
        """Initialize German exchange.

        Args:
            exchange_id: Exchange identifier (kraken, onetrading)
            config: Exchange configuration
        """
        self.exchange_id = exchange_id
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self._initialize()

    def _initialize(self):
        """Initialize exchange connection."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)

            # Base configuration
            exchange_config = {
                'enableRateLimit': True,
                'timeout': self.config.get('timeout', 30000),
                'options': {
                    'defaultType': 'spot',
                },
            }

            # Add API credentials
            if self.config.get('api_key'):
                exchange_config['apiKey'] = self.config['api_key']
            if self.config.get('api_secret'):
                exchange_config['secret'] = self.config['api_secret']

            # Exchange-specific options
            if self.exchange_id == 'kraken':
                exchange_config['options'].update({
                    'adjustForTimeDifference': True,
                })
            elif self.exchange_id == 'onetrading':
                # One Trading (formerly Bitpanda Pro)
                exchange_config['options'].update({
                    'defaultType': 'spot',
                })

            # Create exchange instance
            self.exchange = exchange_class(exchange_config)

            # Test connection
            self.exchange.load_markets()

            logger.success(f"✅ Connected to {self.exchange_name} (BaFin-licensed)")

        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.exchange_id}: {e}")
            raise

    @property
    def exchange_name(self) -> str:
        """Get human-readable exchange name."""
        names = {
            'kraken': 'Kraken',
            'onetrading': 'One Trading (Bitpanda Pro)'
        }
        return names.get(self.exchange_id, self.exchange_id)

    def get_balance(self) -> Dict[str, Any]:
        """Get account balance.

        Returns:
            Dictionary with balance information
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker for symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker information
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """Get OHLCV data.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles

        Returns:
            List of OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise

    def create_market_buy_order(self, symbol: str, amount: float) -> Dict[str, Any]:
        """Create market buy order.

        ⚠️ WARNING: This spends REAL money!

        Args:
            symbol: Trading pair symbol
            amount: Amount to buy

        Returns:
            Order information
        """
        try:
            logger.warning(f"⚠️  EXECUTING MARKET BUY: {amount} {symbol}")

            # Validate symbol for exchange
            symbol = self._validate_symbol(symbol)

            # Execute order
            order = self.exchange.create_market_buy_order(symbol, amount)

            logger.success(f"✅ BUY ORDER EXECUTED: {order['id']}")
            return order

        except Exception as e:
            logger.error(f"❌ BUY ORDER FAILED: {e}")
            raise

    def create_market_sell_order(self, symbol: str, amount: float) -> Dict[str, Any]:
        """Create market sell order.

        ⚠️ WARNING: This sells REAL assets!

        Args:
            symbol: Trading pair symbol
            amount: Amount to sell

        Returns:
            Order information
        """
        try:
            logger.warning(f"⚠️  EXECUTING MARKET SELL: {amount} {symbol}")

            # Validate symbol for exchange
            symbol = self._validate_symbol(symbol)

            # Execute order
            order = self.exchange.create_market_sell_order(symbol, amount)

            logger.success(f"✅ SELL ORDER EXECUTED: {order['id']}")
            return order

        except Exception as e:
            logger.error(f"❌ SELL ORDER FAILED: {e}")
            raise

    def create_limit_buy_order(self, symbol: str, amount: float, price: float) -> Dict[str, Any]:
        """Create limit buy order.

        ⚠️ WARNING: This spends REAL money!

        Args:
            symbol: Trading pair symbol
            amount: Amount to buy
            price: Limit price

        Returns:
            Order information
        """
        try:
            logger.warning(f"⚠️  EXECUTING LIMIT BUY: {amount} {symbol} @ {price}")

            symbol = self._validate_symbol(symbol)
            order = self.exchange.create_limit_buy_order(symbol, amount, price)

            logger.success(f"✅ LIMIT BUY ORDER PLACED: {order['id']}")
            return order

        except Exception as e:
            logger.error(f"❌ LIMIT BUY ORDER FAILED: {e}")
            raise

    def create_limit_sell_order(self, symbol: str, amount: float, price: float) -> Dict[str, Any]:
        """Create limit sell order.

        ⚠️ WARNING: This sells REAL assets!

        Args:
            symbol: Trading pair symbol
            amount: Amount to sell
            price: Limit price

        Returns:
            Order information
        """
        try:
            logger.warning(f"⚠️  EXECUTING LIMIT SELL: {amount} {symbol} @ {price}")

            symbol = self._validate_symbol(symbol)
            order = self.exchange.create_limit_sell_order(symbol, amount, price)

            logger.success(f"✅ LIMIT SELL ORDER PLACED: {order['id']}")
            return order

        except Exception as e:
            logger.error(f"❌ LIMIT SELL ORDER FAILED: {e}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel order.

        Args:
            order_id: Order ID
            symbol: Trading pair symbol

        Returns:
            Cancellation result
        """
        try:
            symbol = self._validate_symbol(symbol)
            result = self.exchange.cancel_order(order_id, symbol)
            logger.info(f"✅ Order cancelled: {order_id}")
            return result
        except Exception as e:
            logger.error(f"❌ Failed to cancel order {order_id}: {e}")
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            raise

    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order details.

        Args:
            order_id: Order ID
            symbol: Trading pair symbol

        Returns:
            Order details
        """
        try:
            symbol = self._validate_symbol(symbol)
            order = self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            raise

    def _validate_symbol(self, symbol: str) -> str:
        """Validate and convert symbol for exchange.

        Args:
            symbol: Input symbol

        Returns:
            Validated symbol in exchange format
        """
        if self.exchange_id == 'kraken':
            # Kraken uses specific symbol formats
            # Convert BTC/EUR to XXBTZEUR
            symbol_map = {
                'BTC/EUR': 'XXBTZEUR',
                'ETH/EUR': 'XETHZEUR',
                'SOL/EUR': 'SOLEUR',
                'ADA/EUR': 'ADAEUR',
                'DOT/EUR': 'DOT/EUR',
            }

            if symbol in symbol_map:
                return symbol_map[symbol]

        elif self.exchange_id == 'onetrading':
            # One Trading (Bitpanda Pro) uses standard format
            # BTC/EUR, ETH/EUR, etc.
            # No conversion needed
            pass

        return symbol

    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Fee information
        """
        try:
            symbol = self._validate_symbol(symbol)
            market = self.exchange.market(symbol)

            # Default fees if not in market
            if self.exchange_id == 'kraken':
                maker_fee = market.get('maker', 0.0016)  # 0.16%
                taker_fee = market.get('taker', 0.0026)  # 0.26%
            elif self.exchange_id == 'onetrading':
                maker_fee = market.get('maker', 0.0010)  # 0.10%
                taker_fee = market.get('taker', 0.0015)  # 0.15%
            else:
                maker_fee = 0.0010
                taker_fee = 0.0020

            return {
                'maker': maker_fee,
                'taker': taker_fee,
            }
        except Exception as e:
            logger.error(f"Failed to fetch fees for {symbol}: {e}")
            return {'maker': 0.0010, 'taker': 0.0020}

    def withdraw(self, currency: str, amount: float, address: str, **kwargs) -> Dict[str, Any]:
        """Withdraw funds.

        ⚠️ WARNING: This withdraws REAL funds!

        Args:
            currency: Currency code
            amount: Amount to withdraw
            address: Withdrawal address
            **kwargs: Additional parameters

        Returns:
            Withdrawal information
        """
        try:
            logger.warning(f"⚠️  WITHDRAWAL: {amount} {currency} to {address}")

            result = self.exchange.withdraw(currency, amount, address, **kwargs)

            logger.success(f"✅ Withdrawal initiated: {result.get('id')}")
            return result

        except Exception as e:
            logger.error(f"❌ Withdrawal failed: {e}")
            raise

    def close(self):
        """Close exchange connection."""
        if self.exchange:
            self.exchange.close()
            logger.info(f"Closed connection to {self.exchange_name}")


class KrakenExchange(GermanExchange):
    """Kraken exchange integration for German users.

    Kraken is fully licensed under MiCA for German users (August 2025).
    """

    def __init__(self, api_key: str, api_secret: str):
        """Initialize Kraken exchange.

        Args:
            api_key: Kraken API key
            api_secret: Kraken API secret
        """
        config = {
            'api_key': api_key,
            'api_secret': api_secret,
            'timeout': 30000,
        }

        super().__init__('kraken', config)

        logger.info("=" * 80)
        logger.info("🇩🇪 KRAKEN EXCHANGE - Germany Compliant")
        logger.info("=" * 80)
        logger.info("License: MiCA (August 2025)")
        logger.info("Regulator: BaFin")
        logger.info("Status: ✅ Fully Licensed")
        logger.info("=" * 80)

    def _validate_symbol(self, symbol: str) -> str:
        """Validate symbol for Kraken.

        Kraken uses specific symbol formats:
        - BTC/EUR -> XXBTZEUR
        - ETH/EUR -> XETHZEUR
        - etc.
        """
        # Comprehensive Kraken symbol map
        kraken_symbols = {
            # EUR pairs
            'BTC/EUR': 'XXBTZEUR',
            'ETH/EUR': 'XETHZEUR',
            'XRP/EUR': 'XXRPZEUR',
            'LTC/EUR': 'XLTCZEUR',
            'BCH/EUR': 'BCH/EUR',
            'XLM/EUR': 'XXLMZEUR',
            'ADA/EUR': 'ADAEUR',
            'DOT/EUR': 'DOT/EUR',
            'SOL/EUR': 'SOLEUR',
            'LINK/EUR': 'LINK/EUR',
            'USDT/EUR': 'USDT/ZEUR',

            # USD pairs (for reference)
            'BTC/USD': 'XXBTZUSD',
            'ETH/USD': 'XETHZUSD',
        }

        return kraken_symbols.get(symbol, symbol)


class OneTradingExchange(GermanExchange):
    """One Trading (formerly Bitpanda Pro) exchange integration.

    One Trading is fully licensed under MiCA for German users (January 2025).
    This was previously known as Bitpanda Pro.
    """

    def __init__(self, api_key: str, api_secret: str):
        """Initialize One Trading exchange.

        Args:
            api_key: One Trading API key
            api_secret: One Trading API secret
        """
        config = {
            'api_key': api_key,
            'api_secret': api_secret,
            'timeout': 30000,
        }

        super().__init__('onetrading', config)

        logger.info("=" * 80)
        logger.info("🇩🇪 ONE TRADING EXCHANGE - Germany Compliant")
        logger.info("=" * 80)
        logger.info("Formerly: Bitpanda Pro")
        logger.info("License: MiCA (January 2025)")
        logger.info("Regulator: BaFin")
        logger.info("Status: ✅ Fully Licensed")
        logger.info("=" * 80)

    def _validate_symbol(self, symbol: str) -> str:
        """Validate symbol for One Trading.

        One Trading uses standard symbol formats:
        - BTC/EUR
        - ETH/EUR
        - BEST/EUR (native token)
        """
        # One Trading uses standard format, no conversion needed
        # Just ensure it's in the correct format
        return symbol


def create_german_exchange(
    exchange_id: str,
    api_key: str,
    api_secret: str
) -> GermanExchange:
    """Create German exchange instance.

    Args:
        exchange_id: Exchange identifier ('kraken' or 'onetrading')
        api_key: API key
        api_secret: API secret

    Returns:
        German exchange instance

    Raises:
        ValueError: If exchange is not supported for German users

    Examples:
        >>> # Kraken
        >>> exchange = create_german_exchange('kraken', 'key', 'secret')

        >>> # One Trading (Bitpanda Pro)
        >>> exchange = create_german_exchange('onetrading', 'key', 'secret')
    """
    supported_exchanges = {
        'kraken': KrakenExchange,
        'onetrading': OneTradingExchange,
        'bitpanda': OneTradingExchange,  # Alias for backwards compatibility
    }

    if exchange_id not in supported_exchanges:
        raise ValueError(
            f"Exchange '{exchange_id}' is not supported for German users.\n"
            f"Supported exchanges: {', '.join(supported_exchanges.keys())}\n\n"
            f"Note: Binance is NOT licensed in Germany (BaFin denied license in 2023).\n"
            f"Use Kraken or One Trading (Bitpanda Pro) for compliant German trading."
        )

    exchange_class = supported_exchanges[exchange_id]
    return exchange_class(api_key, api_secret)


def get_available_exchanges() -> List[Dict[str, str]]:
    """Get list of available German exchanges.

    Returns:
        List of exchange information dictionaries
    """
    return [
        {
            'id': 'kraken',
            'name': 'Kraken',
            'license': 'MiCA',
            'license_date': 'August 2025',
            'regulator': 'BaFin',
            'status': 'Active',
            'url': 'https://www.kraken.com',
            'notes': 'Well-established exchange with EUR markets'
        },
        {
            'id': 'onetrading',
            'name': 'One Trading (Bitpanda Pro)',
            'license': 'MiCA',
            'license_date': 'January 2025',
            'regulator': 'BaFin',
            'status': 'Active',
            'url': 'https://www.onetrading.com',
            'notes': 'Formerly Bitpanda Pro. Austrian-based EU exchange.'
        },
    ]
