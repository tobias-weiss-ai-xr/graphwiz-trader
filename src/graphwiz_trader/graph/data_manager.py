"""Graph data manager for batch ingestion and real-time streaming."""

from loguru import logger
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import asyncio
from queue import Queue
import threading

from .neo4j_graph import KnowledgeGraph
from .models import (
    AssetNode, ExchangeNode, OHLCVNode, TradeNode,
    OrderBookNode, IndicatorNode, SignalNode, SentimentNode
)


class GraphDataManager:
    """Manages data ingestion and streaming to the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph):
        """Initialize data manager.

        Args:
            graph: KnowledgeGraph instance
        """
        self.graph = graph
        self.write_queue: Optional[Queue] = None
        self.is_streaming = False
        self.stream_worker: Optional[threading.Thread] = None
        logger.info("Initialized GraphDataManager")

    # ========== BATCH INGESTION ==========

    def ingest_exchange_data(
        self,
        exchange_name: str,
        assets_data: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """Batch ingest exchange and asset data.

        Args:
            exchange_name: Name of the exchange
            assets_data: List of asset dictionaries
            batch_size: Batch size for processing

        Returns:
            Dictionary with counts of created nodes
        """
        logger.info("Starting batch ingestion for exchange: {}", exchange_name)

        # Create exchange node
        exchange = ExchangeNode(
            name=exchange_name,
            display_name=assets_data[0].get("exchange_display_name", exchange_name),
            maker_fee=assets_data[0].get("maker_fee", 0.001),
            taker_fee=assets_data[0].get("taker_fee", 0.001),
            supports_margin=assets_data[0].get("supports_margin", False),
            supports_futures=assets_data[0].get("supports_futures", False),
            supports_spot=assets_data[0].get("supports_spot", True)
        )
        self.graph.create_exchange(exchange)

        # Create asset nodes in batches
        created_assets = 0
        for i in range(0, len(assets_data), batch_size):
            batch = assets_data[i:i+batch_size]
            for asset_data in batch:
                asset = AssetNode(
                    symbol=asset_data["symbol"],
                    name=asset_data.get("name", asset_data["symbol"]),
                    asset_type=asset_data.get("asset_type", "CRYPTOCURRENCY"),
                    base_currency=asset_data.get("base_currency"),
                    quote_currency=asset_data.get("quote_currency"),
                    decimals=asset_data.get("decimals", 8),
                    min_quantity=asset_data.get("min_quantity", 0.00000001),
                    max_quantity=asset_data.get("max_quantity", 1000000000),
                    metadata=asset_data.get("metadata", {})
                )
                self.graph.create_asset(asset)
                self.graph.link_asset_to_exchange(asset.symbol, exchange_name)
                created_assets += 1

        logger.info("Ingested {} assets for exchange {}", created_assets, exchange_name)

        return {
            "exchange": 1,
            "assets": created_assets
        }

    def ingest_ohlcv_data(
        self,
        ohlcv_data: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> int:
        """Batch ingest OHLCV data.

        Args:
            ohlcv_data: List of OHLCV dictionaries
            batch_size: Batch size for UNWIND operations

        Returns:
            Number of OHLCV nodes created/updated
        """
        if not ohlcv_data:
            return 0

        logger.info("Batch ingesting {} OHLCV records", len(ohlcv_data))

        # Convert to OHLCVNode objects
        ohlcv_nodes = []
        for data in ohlcv_data:
            node = OHLCVNode(
                symbol=data["symbol"],
                exchange=data["exchange"],
                timestamp=data.get("timestamp", datetime.utcnow()),
                timeframe=data.get("timeframe", "1m"),
                open=float(data["open"]),
                high=float(data["high"]),
                low=float(data["low"]),
                close=float(data["close"]),
                volume=float(data["volume"]),
                trades_count=data.get("trades_count", 0)
            )
            ohlcv_nodes.append(node)

        # Process in batches
        total_ingested = 0
        for i in range(0, len(ohlcv_nodes), batch_size):
            batch = ohlcv_nodes[i:i+batch_size]
            self.graph.store_ohlcv_batch(batch)
            total_ingested += len(batch)

        logger.info("Ingested {} OHLCV records", total_ingested)
        return total_ingested

    def ingest_trade_data(
        self,
        trade_data: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """Batch ingest trade data.

        Args:
            trade_data: List of trade dictionaries
            batch_size: Batch size for UNWIND operations

        Returns:
            Number of trade nodes created/updated
        """
        if not trade_data:
            return 0

        logger.info("Batch ingesting {} trades", len(trade_data))

        # Convert to TradeNode objects
        trade_nodes = []
        for data in trade_data:
            node = TradeNode(
                trade_id=data["trade_id"],
                symbol=data["symbol"],
                exchange=data["exchange"],
                timestamp=data.get("timestamp", datetime.utcnow()),
                side=data["side"],
                price=float(data["price"]),
                quantity=float(data["quantity"]),
                amount=float(data["price"]) * float(data["quantity"]),
                fee=float(data.get("fee", 0)),
                fee_currency=data.get("fee_currency", "USD"),
                order_id=data.get("order_id"),
                is_maker=data.get("is_maker", False),
                metadata=data.get("metadata", {})
            )
            trade_nodes.append(node)

        # Process in batches
        total_ingested = 0
        for i in range(0, len(trade_nodes), batch_size):
            batch = trade_nodes[i:i+batch_size]
            self.graph.store_trade_batch(batch)
            total_ingested += len(batch)

        logger.info("Ingested {} trades", total_ingested)
        return total_ingested

    def ingest_indicator_data(
        self,
        indicator_data: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> int:
        """Batch ingest indicator data.

        Args:
            indicator_data: List of indicator dictionaries
            batch_size: Batch size for UNWIND operations

        Returns:
            Number of indicator nodes created/updated
        """
        if not indicator_data:
            return 0

        logger.info("Batch ingesting {} indicators", len(indicator_data))

        # Convert to IndicatorNode objects
        indicator_nodes = []
        for data in indicator_data:
            from .models import IndicatorType
            node = IndicatorNode(
                symbol=data["symbol"],
                exchange=data["exchange"],
                timestamp=data.get("timestamp", datetime.utcnow()),
                timeframe=data.get("timeframe", "1h"),
                indicator_type=IndicatorType(data["indicator_type"]),
                value=data["value"],
                parameters=data.get("parameters", {}),
                metadata=data.get("metadata", {})
            )
            indicator_nodes.append(node)

        # Process in batches
        total_ingested = 0
        for i in range(0, len(indicator_nodes), batch_size):
            batch = indicator_nodes[i:i+batch_size]
            self.graph.store_indicator_batch(batch)
            total_ingested += len(batch)

        logger.info("Ingested {} indicators", total_ingested)
        return total_ingested

    def ingest_signal_data(
        self,
        signal_data: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """Batch ingest signal data.

        Args:
            signal_data: List of signal dictionaries
            batch_size: Batch size for UNWIND operations

        Returns:
            Number of signal nodes created/updated
        """
        if not signal_data:
            return 0

        logger.info("Batch ingesting {} signals", len(signal_data))

        # Convert to SignalNode objects
        signal_nodes = []
        for data in signal_data:
            from .models import SignalType
            node = SignalNode(
                signal_id=data["signal_id"],
                symbol=data["symbol"],
                exchange=data["exchange"],
                timestamp=data.get("timestamp", datetime.utcnow()),
                signal_type=SignalType(data["signal_type"]),
                agent_name=data["agent_name"],
                confidence=float(data["confidence"]),
                reason=data.get("reason", ""),
                target_price=data.get("target_price"),
                stop_loss=data.get("stop_loss"),
                take_profit=data.get("take_profit"),
                indicators=data.get("indicators", []),
                metadata=data.get("metadata", {})
            )
            signal_nodes.append(node)

        # Process in batches
        total_ingested = 0
        for i in range(0, len(signal_nodes), batch_size):
            batch = signal_nodes[i:i+batch_size]
            self.graph.store_signal_batch(batch)
            total_ingested += len(batch)

        logger.info("Ingested {} signals", total_ingested)
        return total_ingested

    # ========== REAL-TIME STREAMING ==========

    def start_streaming(
        self,
        queue_size: int = 10000,
        worker_count: int = 2,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000
    ) -> None:
        """Start real-time data streaming to graph.

        Args:
            queue_size: Maximum queue size
            worker_count: Number of worker threads
            batch_size: Batch size for writes
            batch_timeout_ms: Maximum time to wait before flushing batch
        """
        if self.is_streaming:
            logger.warning("Streaming already active")
            return

        self.write_queue = Queue(maxsize=queue_size)
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.is_streaming = True

        # Start worker threads
        self.stream_workers = []
        for i in range(worker_count):
            worker = threading.Thread(
                target=self._stream_worker,
                name=f"GraphStreamWorker-{i}",
                daemon=True
            )
            worker.start()
            self.stream_workers.append(worker)

        logger.info("Started {} streaming workers", worker_count)

    def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        if not self.is_streaming:
            return

        self.is_streaming = False

        # Wait for workers to finish
        for worker in self.stream_workers:
            worker.join(timeout=5)

        # Process remaining items in queue
        if self.write_queue:
            remaining = self.write_queue.qsize()
            if remaining > 0:
                logger.info("Processing {} remaining items in queue", remaining)
                self._flush_queue()

        logger.info("Stopped streaming")

    def stream_ohlcv(self, ohlcv: Dict[str, Any]) -> None:
        """Stream OHLCV data to graph (non-blocking).

        Args:
            ohlcv: OHLCV data dictionary
        """
        if self.write_queue:
            self.write_queue.put(("ohlcv", ohlcv))

    def stream_trade(self, trade: Dict[str, Any]) -> None:
        """Stream trade data to graph (non-blocking).

        Args:
            trade: Trade data dictionary
        """
        if self.write_queue:
            self.write_queue.put(("trade", trade))

    def stream_orderbook(self, orderbook: Dict[str, Any]) -> None:
        """Stream orderbook snapshot to graph (non-blocking).

        Args:
            orderbook: Orderbook data dictionary
        """
        if self.write_queue:
            self.write_queue.put(("orderbook", orderbook))

    def stream_indicator(self, indicator: Dict[str, Any]) -> None:
        """Stream indicator data to graph (non-blocking).

        Args:
            indicator: Indicator data dictionary
        """
        if self.write_queue:
            self.write_queue.put(("indicator", indicator))

    def stream_signal(self, signal: Dict[str, Any]) -> None:
        """Stream signal to graph (non-blocking).

        Args:
            signal: Signal data dictionary
        """
        if self.write_queue:
            self.write_queue.put(("signal", signal))

    def stream_sentiment(self, sentiment: Dict[str, Any]) -> None:
        """Stream sentiment data to graph (non-blocking).

        Args:
            sentiment: Sentiment data dictionary
        """
        if self.write_queue:
            self.write_queue.put(("sentiment", sentiment))

    def _stream_worker(self) -> None:
        """Worker thread for processing streaming writes."""
        batch_buffer = defaultdict(list)
        last_flush = datetime.utcnow()

        while self.is_streaming or not self.write_queue.empty():
            try:
                # Get item with timeout
                item = self.write_queue.get(timeout=0.1)

                data_type, data = item
                batch_buffer[data_type].append(data)

                # Check if we should flush
                time_since_flush = (datetime.utcnow() - last_flush).total_seconds() * 1000
                should_flush = (
                    time_since_flush >= self.batch_timeout_ms or
                    sum(len(items) for items in batch_buffer.values()) >= self.batch_size
                )

                if should_flush:
                    self._flush_batch(batch_buffer)
                    batch_buffer.clear()
                    last_flush = datetime.utcnow()

            except Exception as e:
                if self.is_streaming:  # Only log if we're supposed to be running
                    logger.debug("Stream worker error: {}", e)
                continue

        # Final flush
        if batch_buffer:
            self._flush_batch(batch_buffer)

    def _flush_batch(self, batch_buffer: Dict[str, List]) -> None:
        """Flush a batch of data to the graph.

        Args:
            batch_buffer: Dictionary of data type to data list
        """
        for data_type, items in batch_buffer.items():
            if not items:
                continue

            try:
                if data_type == "ohlcv":
                    nodes = [OHLCVNode(**item) for item in items]
                    self.graph.store_ohlcv_batch(nodes)
                elif data_type == "trade":
                    nodes = [TradeNode(**item) for item in items]
                    self.graph.store_trade_batch(nodes)
                elif data_type == "indicator":
                    from .models import IndicatorType
                    nodes = []
                    for item in items:
                        item["indicator_type"] = IndicatorType(item["indicator_type"])
                        nodes.append(IndicatorNode(**item))
                    self.graph.store_indicator_batch(nodes)
                elif data_type == "signal":
                    from .models import SignalType
                    nodes = []
                    for item in items:
                        item["signal_type"] = SignalType(item["signal_type"])
                        nodes.append(SignalNode(**item))
                    self.graph.store_signal_batch(nodes)
                elif data_type == "orderbook":
                    for item in items:
                        node = OrderBookNode(**item)
                        self.graph.store_orderbook(node)
                elif data_type == "sentiment":
                    for item in items:
                        node = SentimentNode(**item)
                        self.graph.store_sentiment(node)

                logger.debug("Flushed {} {} items", len(items), data_type)

            except Exception as e:
                logger.error("Error flushing batch for {}: {}", data_type, e)

    def _flush_queue(self) -> None:
        """Flush all remaining items in the queue."""
        batch_buffer = defaultdict(list)

        while not self.write_queue.empty():
            try:
                item = self.write_queue.get_nowait()
                data_type, data = item
                batch_buffer[data_type].append(data)
            except Exception:
                break

        if batch_buffer:
            self._flush_batch(batch_buffer)

    # ========== DATA RETENTION ==========

    def cleanup_old_data(
        self,
        retention_policies: Dict[str, int],
        dry_run: bool = False
    ) -> Dict[str, int]:
        """Clean up old data based on retention policies.

        Args:
            retention_policies: Dictionary of node type to retention days
                Example: {"OHLCV": 30, "Trade": 7, "OrderBook": 1}
            dry_run: If True, only report what would be deleted

        Returns:
            Dictionary of node type to deletion count
        """
        results = {}

        for node_type, retention_days in retention_policies.items():
            if dry_run:
                # Count only
                cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
                cypher = f"""
                MATCH (n:{node_type})
                WHERE n.timestamp < datetime($cutoff_time)
                RETURN count(n) AS count
                """
                count_result = self.graph.query(cypher, cutoff_time=cutoff_time.isoformat())
                count = count_result[0]["count"] if count_result else 0
                results[node_type] = count
                logger.info("Would delete {} {} nodes (dry run)", count, node_type)
            else:
                # Actually delete
                deleted = self.graph.cleanup_old_data(
                    node_type=node_type,
                    older_than_days=retention_days,
                    batch_size=1000
                )
                results[node_type] = deleted
                logger.info("Deleted {} {} nodes", deleted, node_type)

        return results

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for the graph.

        Returns:
            Dictionary with storage statistics
        """
        stats = {}

        # Count nodes by type
        node_types = ["Asset", "Exchange", "OHLCV", "Trade", "OrderBook", "Indicator", "Signal", "Sentiment"]

        for node_type in node_types:
            cypher = f"MATCH (n:{node_type}) RETURN count(n) AS count"
            result = self.graph.query(cypher)
            stats[f"{node_type.lower()}_count"] = result[0]["count"] if result else 0

        # Count relationships by type
        rel_types = ["TRADED_ON", "CORRELATED_WITH", "ARBITRAGE_WITH"]

        for rel_type in rel_types:
            cypher = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
            result = self.graph.query(cypher)
            stats[f"{rel_type.lower()}_count"] = result[0]["count"] if result else 0

        # Time range data
        for symbol_node in ["OHLCV", "Trade", "Indicator"]:
            cypher = f"""
            MATCH (n:{symbol_node})
            RETURN min(n.timestamp) AS earliest, max(n.timestamp) AS latest
            """
            result = self.graph.query(cypher)
            if result and result[0]:
                stats[f"{symbol_node.lower()}_earliest"] = result[0].get("earliest")
                stats[f"{symbol_node.lower()}_latest"] = result[0].get("latest")

        return stats

    # ========== BACKFILL OPERATIONS ==========

    def backfill_historical_data(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: str,
        data_provider: Callable,
        batch_size: int = 500
    ) -> int:
        """Backfill historical data from an external provider.

        Args:
            symbol: Asset symbol
            exchange: Exchange name
            start_time: Start of backfill period
            end_time: End of backfill period
            timeframe: Candle timeframe
            data_provider: Function that fetches OHLCV data
            batch_size: Batch size for writes

        Returns:
            Number of records backfilled
        """
        logger.info("Starting backfill for {} on {} from {} to {}",
                   symbol, exchange, start_time, end_time)

        total_backfilled = 0
        current_time = start_time

        while current_time < end_time:
            # Fetch batch of data
            try:
                ohlcv_data = data_provider(
                    symbol=symbol,
                    exchange=exchange,
                    start_time=current_time,
                    end_time=min(current_time + timedelta(days=1), end_time),
                    timeframe=timeframe
                )

                if ohlcv_data:
                    # Convert to OHLCV nodes
                    ohlcv_nodes = []
                    for data in ohlcv_data:
                        node = OHLCVNode(
                            symbol=symbol,
                            exchange=exchange,
                            timestamp=data["timestamp"],
                            timeframe=timeframe,
                            open=float(data["open"]),
                            high=float(data["high"]),
                            low=float(data["low"]),
                            close=float(data["close"]),
                            volume=float(data["volume"]),
                            trades_count=data.get("trades_count", 0)
                        )
                        ohlcv_nodes.append(node)

                    # Store batch
                    self.graph.store_ohlcv_batch(ohlcv_nodes)
                    total_backfilled += len(ohlcv_nodes)

                # Move to next batch
                current_time += timedelta(days=1)

            except Exception as e:
                logger.error("Error during backfill: {}", e)
                break

        logger.info("Backfilled {} records for {} on {}", total_backfilled, symbol, exchange)
        return total_backfilled
