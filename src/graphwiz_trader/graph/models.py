"""Neo4j graph data models for market entities and relationships."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class AssetType(str, Enum):
    """Asset type enumeration."""
    CRYPTOCURRENCY = "CRYPTOCURRENCY"
    STOCK = "STOCK"
    FOREX = "FOREX"
    COMMODITY = "COMMODITY"
    INDEX = "INDEX"
    ETF = "ETF"


class RelationshipType(str, Enum):
    """Relationship type enumeration."""
    CORRELATED_WITH = "CORRELATED_WITH"
    ARBITRAGE_WITH = "ARBITRAGE_WITH"
    TRADED_ON = "TRADED_ON"
    IMPACTS = "IMPACTS"
    DERIVED_FROM = "DERIVED_FROM"
    HEDGE_AGAINST = "HEDGE_AGAINST"
    SIMILAR_TO = "SIMILAR_TO"


class SignalType(str, Enum):
    """Signal type enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class IndicatorType(str, Enum):
    """Indicator type enumeration."""
    RSI = "RSI"
    MACD = "MACD"
    SMA = "SMA"
    EMA = "EMA"
    BOLLINGER_BANDS = "BOLLINGER_BANDS"
    FIBONACCI = "FIBONACCI"
    VOLUME_PROFILE = "VOLUME_PROFILE"
    ORDER_IMBALANCE = "ORDER_IMBALANCE"


@dataclass
class AssetNode:
    """Represents a tradeable asset in the knowledge graph.

    Attributes:
        symbol: Asset symbol (e.g., "BTC/USD", "AAPL")
        name: Full asset name
        asset_type: Type of asset (crypto, stock, forex, etc.)
        base_currency: Base currency for trading pairs
        quote_currency: Quote currency for trading pairs
        decimals: Number of decimal places for precision
        min_quantity: Minimum tradeable quantity
        max_quantity: Maximum tradeable quantity
        metadata: Additional asset metadata
    """
    symbol: str
    name: str
    asset_type: AssetType
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    decimals: int = 8
    min_quantity: float = 0.00000001
    max_quantity: float = 1000000000
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "asset_type": self.asset_type.value,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "decimals": self.decimals,
            "min_quantity": self.min_quantity,
            "max_quantity": self.max_quantity,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ExchangeNode:
    """Represents an exchange in the knowledge graph.

    Attributes:
        name: Exchange name (e.g., "binance", "kraken")
        display_name: Human-readable exchange name
        maker_fee: Maker fee rate (as decimal, e.g., 0.001 for 0.1%)
        taker_fee: Taker fee rate
        supports_margin: Whether margin trading is supported
        supports_futures: Whether futures trading is supported
        supports_spot: Whether spot trading is supported
        min_order_size: Minimum order size
        max_order_size: Maximum order size
        has_websocket: Whether websocket API is available
        api_rate_limit: API rate limit per minute
        countries: Countries where exchange is available
        metadata: Additional exchange metadata
    """
    name: str
    display_name: str
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    supports_margin: bool = False
    supports_futures: bool = False
    supports_spot: bool = True
    min_order_size: float = 0.001
    max_order_size: float = 1000000
    has_websocket: bool = True
    api_rate_limit: int = 1200
    countries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
            "supports_margin": self.supports_margin,
            "supports_futures": self.supports_futures,
            "supports_spot": self.supports_spot,
            "min_order_size": self.min_order_size,
            "max_order_size": self.max_order_size,
            "has_websocket": self.has_websocket,
            "api_rate_limit": self.api_rate_limit,
            "countries": self.countries,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class OHLCVNode:
    """Represents OHLCV (candlestick) data in the knowledge graph.

    Attributes:
        symbol: Asset symbol
        exchange: Exchange name
        timestamp: Candle timestamp
        timeframe: Timeframe (e.g., "1m", "5m", "1h", "1d")
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        trades_count: Number of trades in this candle
    """
    symbol: str
    exchange: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades_count: int = 0

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "trades_count": self.trades_count
        }


@dataclass
class TradeNode:
    """Represents a trade execution in the knowledge graph.

    Attributes:
        trade_id: Unique trade identifier
        symbol: Asset symbol
        exchange: Exchange name
        timestamp: Trade timestamp
        side: Trade side ("BUY" or "SELL")
        price: Execution price
        quantity: Executed quantity
        amount: Total amount (price * quantity)
        fee: Trading fee
        fee_currency: Currency of the fee
        order_id: Associated order ID
        is_maker: Whether trade was maker or taker
        metadata: Additional trade metadata
    """
    trade_id: str
    symbol: str
    exchange: str
    timestamp: datetime
    side: str
    price: float
    quantity: float
    amount: float
    fee: float = 0.0
    fee_currency: str = "USD"
    order_id: Optional[str] = None
    is_maker: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat(),
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "amount": self.amount,
            "fee": self.fee,
            "fee_currency": self.fee_currency,
            "order_id": self.order_id,
            "is_maker": self.is_maker,
            "metadata": self.metadata
        }


@dataclass
class OrderBookNode:
    """Represents an order book snapshot in the knowledge graph.

    Attributes:
        symbol: Asset symbol
        exchange: Exchange name
        timestamp: Snapshot timestamp
        bids: List of bid prices and quantities [[price, quantity], ...]
        asks: List of ask prices and quantities [[price, quantity], ...]
        bid_depth: Total bid volume
        ask_depth: Total ask volume
        spread: Bid-ask spread
        spread_percentage: Spread as percentage
    """
    symbol: str
    exchange: str
    timestamp: datetime
    bids: List[List[float]]
    asks: List[List[float]]
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    spread: float = 0.0
    spread_percentage: float = 0.0

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat(),
            "bids": self.bids,
            "asks": self.asks,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "spread": self.spread,
            "spread_percentage": self.spread_percentage
        }


@dataclass
class IndicatorNode:
    """Represents a technical indicator value in the knowledge graph.

    Attributes:
        symbol: Asset symbol
        exchange: Exchange name
        timestamp: Indicator timestamp
        timeframe: Timeframe
        indicator_type: Type of indicator
        value: Indicator value(s) - can be single value or dict
        parameters: Indicator parameters (e.g., period for RSI)
        metadata: Additional indicator metadata
    """
    symbol: str
    exchange: str
    timestamp: datetime
    timeframe: str
    indicator_type: IndicatorType
    value: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe,
            "indicator_type": self.indicator_type.value,
            "value": self.value if not isinstance(self.value, dict) else str(self.value),
            "parameters": self.parameters,
            "metadata": self.metadata
        }


@dataclass
class SignalNode:
    """Represents a trading signal in the knowledge graph.

    Attributes:
        signal_id: Unique signal identifier
        symbol: Asset symbol
        exchange: Exchange name
        timestamp: Signal generation timestamp
        signal_type: Type of signal (BUY, SELL, HOLD, etc.)
        agent_name: Name of the agent that generated the signal
        confidence: Signal confidence score (0-1)
        reason: Reasoning behind the signal
        target_price: Target price for the signal
        stop_loss: Stop loss price
        take_profit: Take profit price
        indicators: List of indicators that contributed to the signal
        metadata: Additional signal metadata
    """
    signal_id: str
    symbol: str
    exchange: str
    timestamp: datetime
    signal_type: SignalType
    agent_name: str
    confidence: float
    reason: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.value,
            "agent_name": self.agent_name,
            "confidence": self.confidence,
            "reason": self.reason,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "indicators": self.indicators,
            "metadata": self.metadata
        }


@dataclass
class CorrelationRelationship:
    """Represents a correlation relationship between assets.

    Attributes:
        symbol1: First asset symbol
        symbol2: Second asset symbol
        correlation_coefficient: Pearson correlation coefficient (-1 to 1)
        p_value: Statistical significance
        window: Time window used for calculation
        timestamp: Calculation timestamp
    """
    symbol1: str
    symbol2: str
    correlation_coefficient: float
    p_value: float
    window: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "correlation_coefficient": self.correlation_coefficient,
            "p_value": self.p_value,
            "window": self.window,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ArbitrageRelationship:
    """Represents an arbitrage opportunity between exchanges.

    Attributes:
        symbol: Asset symbol
        exchange1: First exchange
        exchange2: Second exchange
        price1: Price on exchange1
        price2: Price on exchange2
        spread_percentage: Price difference as percentage
        profit_potential: Estimated profit after fees
        timestamp: Detection timestamp
    """
    symbol: str
    exchange1: str
    exchange2: str
    price1: float
    price2: float
    spread_percentage: float
    profit_potential: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "symbol": self.symbol,
            "exchange1": self.exchange1,
            "exchange2": self.exchange2,
            "price1": self.price1,
            "price2": self.price2,
            "spread_percentage": self.spread_percentage,
            "profit_potential": self.profit_potential,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SentimentNode:
    """Represents market sentiment data in the knowledge graph.

    Attributes:
        symbol: Asset symbol
        timestamp: Sentiment timestamp
        source: Sentiment source (twitter, reddit, news, etc.)
        sentiment_score: Sentiment score (-1 to 1, negative to positive)
        confidence: Confidence in sentiment analysis
        volume: Volume of mentions
        keywords: Key terms extracted
        metadata: Additional sentiment metadata
    """
    symbol: str
    timestamp: datetime
    source: str
    sentiment_score: float
    confidence: float
    volume: int = 0
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher query parameters."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "volume": self.volume,
            "keywords": self.keywords,
            "metadata": self.metadata
        }
