"""
HFT Analytics and Pattern Storage.

Provides knowledge graph analytics for HFT strategies and pattern recognition.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger


class HFTAnalytics:
    """Analytics and pattern storage for HFT strategies."""

    def __init__(self, knowledge_graph: Any) -> None:
        """
        Initialize HFT analytics.

        Args:
            knowledge_graph: Neo4j knowledge graph instance
        """
        self.kg = knowledge_graph

    async def store_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Store market pattern in knowledge graph.

        Args:
            pattern: Pattern information including type, symbol, success_rate, etc.

        Returns:
            True if successful, False otherwise
        """
        if not self.kg:
            logger.warning("Knowledge graph not available for pattern storage")
            return False

        try:
            await self.kg.write(
                """
                CREATE (p:Pattern {
                    type: $type,
                    symbol: $symbol,
                    exchange: $exchange,
                    success_rate: $success_rate,
                    avg_profit_bps: $avg_profit_bps,
                    occurrence_count: $occurrence_count,
                    last_seen: datetime(),
                    indicators: $indicators
                })
                """,
                type=pattern.get("type"),
                symbol=pattern.get("symbol"),
                exchange=pattern.get("exchange", "unknown"),
                success_rate=pattern.get("success_rate", 0.0),
                avg_profit_bps=pattern.get("avg_profit_bps", 0.0),
                occurrence_count=pattern.get("occurrence_count", 1),
                indicators=str(pattern.get("indicators", {})),
            )

            # Link to related patterns
            await self.link_related_patterns(
                pattern.get("symbol"), pattern.get("type")
            )

            logger.info(
                f"Stored pattern: {pattern.get('type')} for {pattern.get('symbol')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store pattern: {e}")
            return False

    async def link_related_patterns(
        self, symbol: str, pattern_type: str, days: int = 7
    ) -> None:
        """
        Link related patterns in the knowledge graph.

        Args:
            symbol: Trading symbol
            pattern_type: Type of pattern
            days: Look back period in days
        """
        if not self.kg:
            return

        try:
            await self.kg.write(
                """
                MATCH (p:Pattern {symbol: $symbol, type: $pattern_type})
                MATCH (related:Pattern {symbol: $symbol})
                WHERE related.type <> p.type
                AND related.last_seen > datetime() - duration({days: $days})
                MERGE (p)-[:CORRELATED_WITH]->(related)
                """,
                symbol=symbol,
                pattern_type=pattern_type,
                days=days,
            )
        except Exception as e:
            logger.error(f"Failed to link related patterns: {e}")

    async def get_strategy_performance(
        self, strategy_name: str, days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Get strategy performance from knowledge graph.

        Args:
            strategy_name: Name of the strategy
            days: Look back period in days

        Returns:
            Performance metrics or None
        """
        if not self.kg:
            logger.warning("Knowledge graph not available for performance query")
            return None

        try:
            results = await self.kg.query(
                """
                MATCH (t:Trade {strategy: $strategy})
                WHERE t.timestamp > datetime() - duration({days: $days})
                WITH
                    count(t) as total_trades,
                    sum(t.pnl) as total_pnl,
                    sum(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    sum(CASE WHEN t.pnl > 0 THEN t.pnl ELSE 0 END) as total_profit,
                    sum(CASE WHEN t.pnl < 0 THEN t.pnl ELSE 0 END) as total_loss
                RETURN
                    total_trades,
                    total_pnl,
                    winning_trades,
                    total_profit,
                    total_loss,
                    CASE
                        WHEN total_trades > 0
                        THEN round(100.0 * winning_trades / total_trades, 2)
                        ELSE 0
                    END as win_rate
                """,
                strategy=strategy_name,
                days=days,
            )

            if results and len(results) > 0:
                return results[0]
            return None

        except Exception as e:
            logger.error(f"Failed to query strategy performance: {e}")
            return None

    async def get_best_patterns(
        self, symbol: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the most profitable patterns for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of patterns to return

        Returns:
            List of pattern dictionaries
        """
        if not self.kg:
            return []

        try:
            results = await self.kg.query(
                """
                MATCH (p:Pattern {symbol: $symbol})
                WHERE p.success_rate > 0.5
                RETURN p
                ORDER BY p.avg_profit_bps DESC
                LIMIT $limit
                """,
                symbol=symbol,
                limit=limit,
            )

            return results if results else []

        except Exception as e:
            logger.error(f"Failed to query best patterns: {e}")
            return []

    async def get_correlated_patterns(
        self, symbol: str, pattern_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get patterns correlated with a specific pattern type.

        Args:
            symbol: Trading symbol
            pattern_type: Type of pattern

        Returns:
            List of correlated patterns
        """
        if not self.kg:
            return []

        try:
            results = await self.kg.query(
                """
                MATCH (p:Pattern {symbol: $symbol, type: $pattern_type})
                      -[:CORRELATED_WITH]->(related:Pattern)
                RETURN related
                ORDER BY related.success_rate DESC
                """,
                symbol=symbol,
                pattern_type=pattern_type,
            )

            return results if results else []

        except Exception as e:
            logger.error(f"Failed to query correlated patterns: {e}")
            return []

    async def get_trade_history(
        self, strategy: Optional[str] = None, symbol: Optional[str] = None, days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get trade history from knowledge graph.

        Args:
            strategy: Filter by strategy name (optional)
            symbol: Filter by symbol (optional)
            days: Look back period in days

        Returns:
            List of trades
        """
        if not self.kg:
            return []

        try:
            query_parts = ["MATCH (t:Trade)"]
            where_clauses = ["t.timestamp > datetime() - duration({days: $days})"]

            params: Dict[str, Any] = {"days": days}

            if strategy:
                where_clauses.append("t.strategy = $strategy")
                params["strategy"] = strategy

            if symbol:
                where_clauses.append("t.symbol = $symbol")
                params["symbol"] = symbol

            if where_clauses:
                query_parts.append(f"WHERE {' AND '.join(where_clauses)}")

            query_parts.append(
                """
                RETURN t
                ORDER BY t.timestamp DESC
                LIMIT 1000
                """
            )

            query = "\n".join(query_parts)
            results = await self.kg.query(query, **params)

            return results if results else []

        except Exception as e:
            logger.error(f"Failed to query trade history: {e}")
            return []

    async def get_top_performers(
        self, metric: str = "total_pnl", limit: int = 5, days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get top performing strategies.

        Args:
            metric: Metric to rank by (total_pnl, win_rate, total_trades)
            limit: Maximum number of strategies to return
            days: Look back period in days

        Returns:
            List of top performing strategies
        """
        if not self.kg:
            return []

        try:
            # Determine order by clause based on metric
            order_by = {
                "total_pnl": "total_pnl DESC",
                "win_rate": "win_rate DESC",
                "total_trades": "total_trades DESC",
            }.get(metric, "total_pnl DESC")

            results = await self.kg.query(
                f"""
                MATCH (t:Trade)
                WHERE t.timestamp > datetime() - duration({{days: $days}})
                WITH
                    t.strategy as strategy,
                    count(t) as total_trades,
                    sum(t.pnl) as total_pnl,
                    sum(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                WITH
                    strategy,
                    total_trades,
                    total_pnl,
                    CASE
                        WHEN total_trades > 0
                        THEN round(100.0 * winning_trades / total_trades, 2)
                        ELSE 0
                    END as win_rate
                RETURN strategy, total_trades, total_pnl, win_rate
                ORDER BY {order_by}
                LIMIT $limit
                """,
                days=days,
                limit=limit,
            )

            return results if results else []

        except Exception as e:
            logger.error(f"Failed to query top performers: {e}")
            return []

    async def store_arbitrage_opportunity(
        self, opportunity: Dict[str, Any]
    ) -> bool:
        """
        Store arbitrage opportunity for analysis.

        Args:
            opportunity: Arbitrage opportunity details

        Returns:
            True if successful, False otherwise
        """
        if not self.kg:
            return False

        try:
            await self.kg.write(
                """
                CREATE (a:ArbitrageOpportunity {
                    symbol: $symbol,
                    buy_exchange: $buy_exchange,
                    sell_exchange: $sell_exchange,
                    buy_price: $buy_price,
                    sell_price: $sell_price,
                    profit_bps: $profit_bps,
                    quantity: $quantity,
                    timestamp: datetime()
                })
                """,
                symbol=opportunity.get("symbol"),
                buy_exchange=opportunity.get("buy_exchange"),
                sell_exchange=opportunity.get("sell_exchange"),
                buy_price=opportunity.get("buy_price", 0.0),
                sell_price=opportunity.get("sell_price", 0.0),
                profit_bps=opportunity.get("profit_bps", 0.0),
                quantity=opportunity.get("quantity", 0.0),
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store arbitrage opportunity: {e}")
            return False

    async def get_arbitrage_statistics(
        self, symbol: str, days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Get arbitrage opportunity statistics.

        Args:
            symbol: Trading symbol
            days: Look back period in days

        Returns:
            Statistics dictionary or None
        """
        if not self.kg:
            return None

        try:
            results = await self.kg.query(
                """
                MATCH (a:ArbitrageOpportunity {symbol: $symbol})
                WHERE a.timestamp > datetime() - duration({days: $days})
                WITH
                    count(a) as total_opportunities,
                    avg(a.profit_bps) as avg_profit_bps,
                    max(a.profit_bps) as max_profit_bps,
                    min(a.profit_bps) as min_profit_bps
                RETURN
                    total_opportunities,
                    avg_profit_bps,
                    max_profit_bps,
                    min_profit_bps
                """,
                symbol=symbol,
                days=days,
            )

            if results and len(results) > 0:
                return results[0]
            return None

        except Exception as e:
            logger.error(f"Failed to query arbitrage statistics: {e}")
            return None
