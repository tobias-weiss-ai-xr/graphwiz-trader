"""Neo4j knowledge graph implementation."""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
from loguru import logger
from typing import Any, Dict, List, Optional, Callable
from functools import lru_cache
from datetime import datetime, timedelta
import threading
import time


class KnowledgeGraph:
    """Neo4j knowledge graph connector with connection pooling and caching."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize knowledge graph connection.

        Args:
            config: Neo4j configuration dict with uri, username, password
        """
        self.uri = config.get("uri", "bolt://localhost:7687")
        self.username = config.get("username", "neo4j")
        self.password = config.get("password", "")
        self.database = config.get("database", "neo4j")
        self.driver: Optional[GraphDatabase.driver] = None

        # Connection pooling configuration
        self.max_connection_lifetime = config.get("max_connection_lifetime", 3600)  # 1 hour
        self.max_connection_pool_size = config.get("max_connection_pool_size", 50)
        self.connection_acquisition_timeout = config.get("connection_acquisition_timeout", 60)

        # Query cache configuration
        self._query_cache_enabled = config.get("query_cache_enabled", True)
        self._query_cache_ttl = config.get("query_cache_ttl", 300)  # 5 minutes
        self._query_cache: Dict[str, tuple[List[Dict[str, Any]], datetime]] = {}
        self._cache_lock = threading.Lock()

        # Batch operation buffer
        self._batch_buffer: List[tuple[str, Dict[str, Any]]] = []
        self._batch_size = config.get("batch_size", 100)
        self._batch_lock = threading.Lock()

        # Performance metrics
        self._query_count = 0
        self._query_time_total = 0.0
        self._metrics_lock = threading.Lock()

    def connect(self) -> None:
        """Connect to Neo4j database with optimized connection pooling."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size,
                connection_acquisition_timeout=self.connection_acquisition_timeout,
                resolver=lambda _: [
                    (self.uri.replace("bolt://", "").replace("neo4j://", "").split(":")[0], 7687)
                ],
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(
                "✅ Connected to Neo4j at {} (pool size: {}, lifetime: {}s)",
                self.uri,
                self.max_connection_pool_size,
                self.max_connection_lifetime,
            )
        except Exception as e:
            logger.error("Failed to connect to Neo4j: {}", e)
            raise

    def disconnect(self) -> None:
        """Disconnect from Neo4j database and cleanup resources."""
        if self.driver:
            # Flush any pending batch operations
            if self._batch_buffer:
                logger.warning(
                    "Flushing {} pending batch operations before disconnect",
                    len(self._batch_buffer),
                )
                self.flush_batch()

            self.driver.close()
            logger.info("Disconnected from Neo4j")

            # Log performance metrics
            self._log_metrics()

    def query(self, cypher: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute a Cypher query with caching and retry logic.

        Args:
            cypher: Cypher query string
            **kwargs: Query parameters

        Returns:
            List of result dictionaries
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")

        # Check cache for read-only queries (SELECT-like)
        cache_key = None
        if self._query_cache_enabled and self._is_read_query(cypher):
            cache_key = self._generate_cache_key(cypher, kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.debug("Query cache hit for key: {}", cache_key[:50])
                return cached_result

        # Execute with retry logic for transient errors
        result = self._execute_with_retry(lambda: self._execute_query(cypher, kwargs))

        # Cache read-only query results
        if cache_key:
            self._add_to_cache(cache_key, result)

        return result

    def _execute_query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute query and record metrics."""
        start_time = time.time()

        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, params)
            records = [record.data() for record in result]

        # Update metrics
        query_time = time.time() - start_time
        with self._metrics_lock:
            self._query_count += 1
            self._query_time_total += query_time

        logger.debug("Query executed in {:.3f}s, returned {} records", query_time, len(records))
        return records

    def write(self, cypher: str, **kwargs) -> Any:
        """Execute a write Cypher query.

        Args:
            cypher: Cypher query string
            **kwargs: Query parameters

        Returns:
            Result summary
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")

        return self._execute_with_retry(lambda: self._execute_write(cypher, kwargs))

    def _execute_write(self, cypher: str, params: Dict[str, Any]) -> Any:
        """Execute write query and record metrics."""
        start_time = time.time()

        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, params)
            summary = result.consume()

        query_time = time.time() - start_time
        with self._metrics_lock:
            self._query_count += 1
            self._query_time_total += query_time

        logger.debug("Write executed in {:.3f}s", query_time)
        return summary

    def _execute_with_retry(self, func: Callable, max_retries: int = 3) -> Any:
        """Execute function with retry logic for transient errors.

        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts

        Returns:
            Function result

        Raises:
            Exception: If all retries are exhausted
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                return func()
            except (ServiceUnavailable, TransientError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        "Transient error (attempt {}/{}): {}. Retrying in {}s...",
                        attempt + 1,
                        max_retries,
                        e,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries ({}) exhausted for query: {}", max_retries, e)
                    raise

        raise last_error if last_error else RuntimeError("Unknown error in retry logic")

    def add_to_batch(self, cypher: str, **kwargs) -> None:
        """Add query to batch buffer for later execution.

        Args:
            cypher: Cypher query string
            **kwargs: Query parameters
        """
        with self._batch_lock:
            self._batch_buffer.append((cypher, kwargs))

            # Auto-flush if buffer is full
            if len(self._batch_buffer) >= self._batch_size:
                self.flush_batch()

    def flush_batch(self) -> None:
        """Execute all batched queries in a single transaction.

        This significantly improves performance for bulk writes by using
        a single transaction instead of individual queries.
        """
        with self._batch_lock:
            if not self._batch_buffer:
                return

            queries = self._batch_buffer.copy()
            self._batch_buffer.clear()

        try:
            start_time = time.time()

            with self.driver.session(database=self.database) as session:
                with session.begin_transaction() as tx:
                    for cypher, params in queries:
                        tx.run(cypher, params)

            query_time = time.time() - start_time
            logger.info(
                "Flushed batch of {} queries in {:.3f}s ({:.1f} queries/s)",
                len(queries),
                query_time,
                len(queries) / query_time if query_time > 0 else 0,
            )

        except Exception as e:
            logger.error("Failed to execute batch: {}. Re-queueing {} queries.", e, len(queries))
            # Re-queue failed queries
            with self._batch_lock:
                self._batch_buffer.extend(queries)
            raise

    def _is_read_query(self, cypher: str) -> bool:
        """Check if query is a read-only query.

        Args:
            cypher: Cypher query string

        Returns:
            True if query is read-only (SELECT-like)
        """
        cypher_upper = cypher.strip().upper()
        read_keywords = ["MATCH", "RETURN", "WHERE", "WITH", "ORDER BY", "LIMIT", "SKIP"]
        write_keywords = ["CREATE", "SET", "DELETE", "DETACH", "MERGE", "DROP", "CALL"]

        # Check for write keywords first (they override read keywords)
        for keyword in write_keywords:
            if keyword in cypher_upper:
                return False

        # Check for read keywords
        for keyword in read_keywords:
            if keyword in cypher_upper:
                return True

        return False

    def _generate_cache_key(self, cypher: str, params: Dict[str, Any]) -> str:
        """Generate cache key from query and parameters.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            Cache key string
        """
        # Normalize query (remove extra whitespace)
        normalized_query = " ".join(cypher.split())

        # Create hash of query and params
        import hashlib

        param_str = str(sorted(params.items()))
        key_string = f"{normalized_query}:{param_str}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get result from cache if not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None if not found/expired
        """
        with self._cache_lock:
            if cache_key in self._query_cache:
                result, timestamp = self._query_cache[cache_key]

                # Check if expired
                if datetime.now() - timestamp < timedelta(seconds=self._query_cache_ttl):
                    return result
                else:
                    # Remove expired entry
                    del self._query_cache[cache_key]

        return None

    def _add_to_cache(self, cache_key: str, result: List[Dict[str, Any]]) -> None:
        """Add result to cache.

        Args:
            cache_key: Cache key
            result: Query result to cache
        """
        with self._cache_lock:
            self._query_cache[cache_key] = (result, datetime.now())

            # Optional: Prune cache if too large
            max_cache_size = 1000
            if len(self._query_cache) > max_cache_size:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self._query_cache.keys())[
                    : len(self._query_cache) - max_cache_size
                ]
                for key in keys_to_remove:
                    del self._query_cache[key]
                logger.debug("Pruned query cache (removed {} entries)", len(keys_to_remove))

    def clear_cache(self) -> None:
        """Clear the query cache."""
        with self._cache_lock:
            cache_size = len(self._query_cache)
            self._query_cache.clear()
            logger.info("Cleared query cache (removed {} entries)", cache_size)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        with self._metrics_lock:
            avg_query_time = (
                self._query_time_total / self._query_count if self._query_count > 0 else 0
            )

            return {
                "query_count": self._query_count,
                "total_query_time": self._query_time_total,
                "average_query_time": avg_query_time,
                "cache_size": len(self._query_cache),
                "batch_buffer_size": len(self._batch_buffer),
            }

    def _log_metrics(self) -> None:
        """Log performance metrics."""
        metrics = self.get_metrics()
        logger.info(
            "Neo4j Performance Metrics: {} queries, {:.3f}s total, {:.3f}s avg, "
            "{} cached entries, {} pending batch ops",
            metrics["query_count"],
            metrics["total_query_time"],
            metrics["average_query_time"],
            metrics["cache_size"],
            metrics["batch_buffer_size"],
        )

    # ===== Sentiment Analysis Methods =====

    async def create_sentiment_node(
        self,
        symbol: str,
        timestamp: datetime,
        source: str,
        sentiment_score: float,
        confidence: float,
        volume: int,
        keywords: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        """Create a sentiment node in the knowledge graph.

        Args:
            symbol: Trading symbol (e.g., 'BTC')
            timestamp: Sentiment timestamp
            source: Data source (news, twitter, reddit, etc.)
            sentiment_score: Sentiment score (-1 to 1)
            confidence: Confidence level (0 to 1)
            volume: Volume of mentions
            keywords: Extracted keywords
            metadata: Additional metadata
        """
        cypher = """
        MERGE (s:Sentiment {
            symbol: $symbol,
            timestamp: datetime($timestamp),
            source: $source
        })
        SET s.sentiment_score = $sentiment_score,
            s.confidence = $confidence,
            s.volume = $volume,
            s.keywords = $keywords,
            s.metadata = $metadata,
            s.updated_at = datetime()
        RETURN s
        """

        self.query(
            cypher,
            symbol=symbol,
            timestamp=timestamp.isoformat(),
            source=source,
            sentiment_score=sentiment_score,
            confidence=confidence,
            volume=volume,
            keywords=keywords,
            metadata=metadata,
        )

    def get_recent_sentiment(
        self, symbol: str, hours_back: int = 24, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent sentiment data for a symbol.

        Args:
            symbol: Trading symbol
            hours_back: Hours to look back
            source: Optional source filter

        Returns:
            List of sentiment data points
        """
        cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()

        if source:
            cypher = """
            MATCH (s:Sentiment)
            WHERE s.symbol = $symbol
                AND s.source = $source
                AND s.timestamp >= datetime($cutoff)
            RETURN s
            ORDER BY s.timestamp DESC
            """
            return self.query(cypher, symbol=symbol, source=source, cutoff=cutoff)
        else:
            cypher = """
            MATCH (s:Sentiment)
            WHERE s.symbol = $symbol
                AND s.timestamp >= datetime($cutoff)
            RETURN s
            ORDER BY s.timestamp DESC
            """
            return self.query(cypher, symbol=symbol, cutoff=cutoff)

    def get_aggregate_sentiment(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get aggregate sentiment metrics for a symbol.

        Args:
            symbol: Trading symbol
            hours_back: Hours to look back

        Returns:
            Dictionary with aggregate metrics
        """
        cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()

        cypher = """
        MATCH (s:Sentiment)
        WHERE s.symbol = $symbol
            AND s.timestamp >= datetime($cutoff)
        WITH s
        ORDER BY s.timestamp DESC
        RETURN
            avg(s.sentiment_score) as avg_sentiment,
            avg(s.confidence) as avg_confidence,
            sum(s.volume) as total_volume,
            count(s) as data_points,
            collect(DISTINCT s.source) as sources
        """

        results = self.query(cypher, symbol=symbol, cutoff=cutoff)

        if results:
            return results[0]
        else:
            return {
                "avg_sentiment": 0.0,
                "avg_confidence": 0.0,
                "total_volume": 0,
                "data_points": 0,
                "sources": [],
            }

    def get_sentiment_trend(
        self, symbol: str, hours_back: int = 24, interval_hours: int = 1
    ) -> List[Dict[str, Any]]:
        """Get sentiment trend over time.

        Args:
            symbol: Trading symbol
            hours_back: Hours to look back
            interval_hours: Interval for grouping data

        Returns:
            List of time-bucketed sentiment data
        """
        cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()

        cypher = """
        MATCH (s:Sentiment)
        WHERE s.symbol = $symbol
            AND s.timestamp >= datetime($cutoff)
        WITH s, datetime({
            year: s.timestamp.year,
            month: s.timestamp.month,
            day: s.timestamp.day,
            hour: (s.timestamp.hour - $interval_hours)
        }) as time_bucket
        WITH time_bucket,
            avg(s.sentiment_score) as avg_sentiment,
            sum(s.volume) as volume,
            count(s) as count
        RETURN time_bucket, avg_sentiment, volume, count
        ORDER BY time_bucket ASC
        """

        return self.query(cypher, symbol=symbol, cutoff=cutoff, interval_hours=interval_hours)
