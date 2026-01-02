"""
Comprehensive example demonstrating the knowledge graph integration for graphwiz-trader.

This example shows how to:
1. Initialize and connect to the knowledge graph
2. Store and query market data (OHLCV, trades, orderbook)
3. Create market relationships (correlations, arbitrage)
4. Perform graph analytics
5. Use real-time data streaming
6. Implement data retention policies
"""

from datetime import datetime, timedelta
from loguru import logger

from graphwiz_trader.graph import (
    KnowledgeGraph,
    GraphAnalytics,
    GraphDataManager,
    AssetNode,
    ExchangeNode,
    OHLCVNode,
    TradeNode,
    SignalNode,
    AssetType,
    IndicatorType,
    SignalType,
)


def example_initialization():
    """Example: Initialize the knowledge graph."""
    config = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password",
        "database": "neo4j"
    }

    # Initialize graph connection
    graph = KnowledgeGraph(config)
    graph.connect()

    # Initialize analytics engine
    analytics = GraphAnalytics(graph)

    # Initialize data manager
    data_manager = GraphDataManager(graph)

    return graph, analytics, data_manager


def example_asset_management(graph: KnowledgeGraph):
    """Example: Create assets and exchanges."""
    # Create exchange
    binance = ExchangeNode(
        name="binance",
        display_name="Binance",
        maker_fee=0.001,
        taker_fee=0.001,
        supports_spot=True,
        supports_futures=True,
        supports_margin=True,
        has_websocket=True,
        api_rate_limit=1200
    )
    graph.create_exchange(binance)

    # Create assets
    btc = AssetNode(
        symbol="BTC/USD",
        name="Bitcoin",
        asset_type=AssetType.CRYPTOCURRENCY,
        base_currency="BTC",
        quote_currency="USD",
        decimals=8,
        min_quantity=0.00000001,
        metadata={"market_cap": 500000000000}
    )
    graph.create_asset(btc)

    eth = AssetNode(
        symbol="ETH/USD",
        name="Ethereum",
        asset_type=AssetType.CRYPTOCURRENCY,
        base_currency="ETH",
        quote_currency="USD",
        decimals=8,
        metadata={"market_cap": 200000000000}
    )
    graph.create_asset(eth)

    # Link assets to exchange
    graph.link_asset_to_exchange("BTC/USD", "binance")
    graph.link_asset_to_exchange("ETH/USD", "binance")


def example_ohlcv_storage(graph: KnowledgeGraph):
    """Example: Store OHLCV data."""
    # Single OHLCV record
    ohlcv = OHLCVNode(
        symbol="BTC/USD",
        exchange="binance",
        timestamp=datetime.utcnow(),
        timeframe="1h",
        open=45000.0,
        high=45500.0,
        low=44800.0,
        close=45200.0,
        volume=1234.56,
        trades_count=5000
    )
    graph.store_ohlcv(ohlcv)

    # Batch storage
    ohlcv_batch = []
    for i in range(100):
        ohlcv = OHLCVNode(
            symbol="BTC/USD",
            exchange="binance",
            timestamp=datetime.utcnow() - timedelta(hours=i),
            timeframe="1h",
            open=45000.0 + i * 10,
            high=45500.0 + i * 10,
            low=44800.0 + i * 10,
            close=45200.0 + i * 10,
            volume=1234.56,
            trades_count=5000
        )
        ohlcv_batch.append(ohlcv)

    graph.store_ohlcv_batch(ohlcv_batch)

    # Query OHLCV data
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)

    ohlcv_data = graph.get_ohlcv(
        symbol="BTC/USD",
        exchange="binance",
        start_time=start_time,
        end_time=end_time,
        timeframe="1h",
        limit=100
    )

    logger.info(f"Retrieved {len(ohlcv_data)} OHLCV records")


def example_trade_storage(graph: KnowledgeGraph, data_manager: GraphDataManager):
    """Example: Store trade data."""
    # Using data manager for batch ingestion
    trade_data = []
    for i in range(1000):
        trade = {
            "trade_id": f"trade_{i}",
            "symbol": "BTC/USD",
            "exchange": "binance",
            "timestamp": datetime.utcnow() - timedelta(seconds=i),
            "side": "BUY" if i % 2 == 0 else "SELL",
            "price": 45000.0 + (i % 100) * 10,
            "quantity": 0.1 + (i % 10) * 0.01,
            "fee": 0.001,
            "fee_currency": "USD",
            "is_maker": i % 3 == 0
        }
        trade_data.append(trade)

    count = data_manager.ingest_trade_data(trade_data, batch_size=500)
    logger.info(f"Ingested {count} trades")

    # Query trade volume
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    volumes = graph.get_trade_volume(
        symbol="BTC/USD",
        exchange="binance",
        start_time=start_time,
        end_time=end_time
    )

    logger.info(f"Trade volumes: {volumes}")


def example_signals(graph: KnowledgeGraph):
    """Example: Store and query trading signals."""
    # Create signal
    signal = SignalNode(
        signal_id="signal_001",
        symbol="BTC/USD",
        exchange="binance",
        timestamp=datetime.utcnow(),
        signal_type=SignalType.BUY,
        agent_name="TechnicalAnalysisAgent",
        confidence=0.85,
        reason="RSI oversold, MACD bullish crossover",
        target_price=47000.0,
        stop_loss=44500.0,
        take_profit=48500.0,
        indicators=["RSI", "MACD", "BB"],
        metadata={"backtest_win_rate": 0.72}
    )
    graph.store_signal(signal)

    # Query high-confidence buy signals
    buy_signals = graph.get_signals(
        symbol="BTC/USD",
        signal_type=SignalType.BUY,
        min_confidence=0.7,
        limit=10
    )

    logger.info(f"Found {len(buy_signals)} high-confidence buy signals")


def example_correlations(analytics: GraphAnalytics):
    """Example: Calculate and store correlations."""
    symbols = ["BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD"]

    # Calculate correlation matrix
    result = analytics.calculate_correlation_matrix(
        symbols=symbols,
        exchange="binance",
        window="24h",
        min_correlation=0.5
    )

    logger.info(f"Calculated {len(result['correlations'])} correlations")

    # Find correlated clusters
    clusters = analytics.find_correlation_clusters(
        symbols=symbols,
        exchange="binance",
        window="24h",
        eps=0.3
    )

    logger.info(f"Found {len(clusters)} correlation clusters")
    for i, cluster in enumerate(clusters):
        logger.info(f"Cluster {i}: {cluster}")


def example_arbitrage_detection(analytics: GraphAnalytics, graph: KnowledgeGraph):
    """Example: Detect arbitrage opportunities."""
    exchanges = ["binance", "kraken", "coinbase"]

    # Detect simple arbitrage
    opportunities = analytics.detect_arbitrage_opportunities(
        exchanges=exchanges,
        symbols=["BTC/USD", "ETH/USD"],
        min_profit_percentage=0.5,
        include_fees=True
    )

    logger.info(f"Found {len(opportunities)} arbitrage opportunities")
    for opp in opportunities[:5]:
        logger.info(
            f"{opp['symbol']}: Buy {opp['buy_exchange']} @ {opp['buy_price']}, "
            f"Sell {opp['sell_exchange']} @ {opp['sell_price']}, "
            f"Profit: {opp['net_profit_pct']:.2f}%"
        )

    # Query stored arbitrage opportunities
    arb_opps = graph.get_arbitrage_opportunities(
        min_profit_percentage=0.5,
        limit=10
    )

    logger.info(f"Retrieved {len(arb_opps)} stored arbitrage opportunities")


def example_pattern_detection(analytics: GraphAnalytics):
    """Example: Detect market patterns."""
    # Detect pump and dump
    pump_dump = analytics.detect_pump_and_dump(
        symbol="BTC/USD",
        exchange="binance",
        lookback_hours=24,
        volume_spike_threshold=3.0,
        price_change_threshold=20.0
    )

    logger.info(f"Pump & dump detection: {pump_dump}")

    # Detect accumulation/distribution
    acc_dist = analytics.detect_accumulation_distribution(
        symbol="BTC/USD",
        exchange="binance",
        lookback_hours=48
    )

    logger.info(f"Accumulation/Distribution: {acc_dist['pattern']} "
                f"(confidence: {acc_dist['confidence']:.2f})")


def example_graph_analytics(graph: KnowledgeGraph, analytics: GraphAnalytics):
    """Example: Perform graph analytics."""
    # Find shortest path between assets
    path = graph.find_shortest_path(
        from_symbol="BTC/USD",
        to_symbol="ETH/USD",
        max_depth=3
    )

    logger.info(f"Path from BTC/USD to ETH/USD: {path}")

    # Detect triangular arbitrage
    tri_arb = graph.detect_triangular_arbitrage(
        base_currency="USD",
        min_profit_percentage=0.1
    )

    logger.info(f"Found {len(tri_arb)} triangular arbitrage opportunities")

    # Get price history statistics
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)

    price_stats = graph.get_price_history_stats(
        symbol="BTC/USD",
        exchange="binance",
        start_time=start_time,
        end_time=end_time,
        timeframe="1h"
    )

    logger.info(f"Price statistics: {price_stats}")


def example_realtime_streaming(data_manager: GraphDataManager):
    """Example: Set up real-time data streaming."""
    # Start streaming with custom settings
    data_manager.start_streaming(
        queue_size=10000,
        worker_count=2,
        batch_size=100,
        batch_timeout_ms=1000
    )

    # Stream data (non-blocking)
    for i in range(10):
        # Simulate incoming OHLCV data
        ohlcv = {
            "symbol": "BTC/USD",
            "exchange": "binance",
            "timestamp": datetime.utcnow(),
            "timeframe": "1m",
            "open": 45000.0 + i,
            "high": 45100.0 + i,
            "low": 44900.0 + i,
            "close": 45050.0 + i,
            "volume": 100.0,
            "trades_count": 500
        }
        data_manager.stream_ohlcv(ohlcv)

        # Simulate incoming trade
        trade = {
            "trade_id": f"stream_trade_{i}",
            "symbol": "BTC/USD",
            "exchange": "binance",
            "timestamp": datetime.utcnow(),
            "side": "BUY" if i % 2 == 0 else "SELL",
            "price": 45000.0 + i,
            "quantity": 0.1,
            "fee": 0.001,
            "fee_currency": "USD"
        }
        data_manager.stream_trade(trade)

    # Stop streaming (processes remaining items)
    data_manager.stop_streaming()

    logger.info("Real-time streaming completed")


def example_data_retention(data_manager: GraphDataManager):
    """Example: Implement data retention policies."""
    # Define retention policies (in days)
    policies = {
        "OHLCV": 30,        # Keep OHLCV for 30 days
        "Trade": 7,         # Keep trades for 7 days
        "OrderBook": 1,     # Keep orderbook snapshots for 1 day
        "Indicator": 30,    # Keep indicators for 30 days
        "Signal": 90        # Keep signals for 90 days
    }

    # Dry run to see what would be deleted
    results = data_manager.cleanup_old_data(policies, dry_run=True)
    logger.info(f"Dry run results: {results}")

    # Actually clean up (commented out for safety)
    # results = data_manager.cleanup_old_data(policies, dry_run=False)
    # logger.info(f"Cleanup results: {results}")


def example_storage_stats(data_manager: GraphDataManager):
    """Example: Get storage statistics."""
    stats = data_manager.get_storage_stats()

    logger.info("Knowledge Graph Storage Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:,}")
        else:
            logger.info(f"  {key}: {value}")


def example_market_report(analytics: GraphAnalytics):
    """Example: Generate comprehensive market report."""
    symbols = ["BTC/USD", "ETH/USD", "BNB/USD"]

    report = analytics.generate_market_report(
        symbols=symbols,
        exchange="binance",
        lookback_hours=24
    )

    logger.info(f"Generated market report with {len(report['assets'])} assets")

    # Save report to file (optional)
    import json
    with open("market_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Market report saved to market_report.json")


def main():
    """Run all examples."""
    logger.info("Starting knowledge graph examples")

    # Initialize
    graph, analytics, data_manager = example_initialization()

    # Asset management
    example_asset_management(graph)

    # OHLCV data
    example_ohlcv_storage(graph)

    # Trade data
    example_trade_storage(graph, data_manager)

    # Signals
    example_signals(graph)

    # Correlations
    example_correlations(analytics)

    # Arbitrage detection
    example_arbitrage_detection(analytics, graph)

    # Pattern detection
    example_pattern_detection(analytics)

    # Graph analytics
    example_graph_analytics(graph, analytics)

    # Real-time streaming
    example_realtime_streaming(data_manager)

    # Data retention
    example_data_retention(data_manager)

    # Storage statistics
    example_storage_stats(data_manager)

    # Market report
    example_market_report(analytics)

    # Disconnect
    graph.disconnect()
    logger.info("Knowledge graph examples completed")


if __name__ == "__main__":
    main()
