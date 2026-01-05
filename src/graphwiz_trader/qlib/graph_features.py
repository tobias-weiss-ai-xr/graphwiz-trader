"""Graph feature extraction from Neo4j knowledge graph.

This module extracts graph-based features to complement Qlib's Alpha158 features,
creating a unique hybrid approach that no other trading system has.

Graph features include:
- Correlation network metrics
- Trading pattern clusters
- Market regime indicators
- Symbol relationships
- Historical trading patterns
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("Neo4j driver not available. Graph features will be limited.")

from .config import QlibConfig


class GraphFeatureExtractor:
    """
    Extract graph-based features from Neo4j knowledge graph.

    This class provides unique features that capture market relationships,
    trading patterns, and symbol correlations that traditional time-series
    analysis misses.
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        config: Optional[QlibConfig] = None,
    ):
        """
        Initialize graph feature extractor.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            config: Qlib configuration
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.config = config or QlibConfig()

        self.driver = None

        if NEO4J_AVAILABLE:
            self._connect()

        logger.info("Graph feature extractor initialized")

    def _connect(self):
        """Establish Neo4j connection."""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j not available")
            return

        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def extract_all_features(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Extract all graph features for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timestamp: Current timestamp

        Returns:
            Dictionary of graph features
        """
        if timestamp is None:
            timestamp = datetime.now()

        features = {}

        # Extract different feature types
        features.update(self._extract_network_features(symbol, timestamp))
        features.update(self._extract_correlation_features(symbol, timestamp))
        features.update(self._extract_trading_pattern_features(symbol, timestamp))
        features.update(self._extract_market_regime_features(symbol, timestamp))

        logger.debug(f"Extracted {len(features)} graph features for {symbol}")

        return features

    def _extract_network_features(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """Extract network centrality features."""
        features = {}

        if not self.driver:
            return features

        try:
            with self.driver.session() as session:
                # Degree centrality (number of correlations)
                query = """
                MATCH (s:Symbol {name: $symbol})
                OPTIONAL MATCH (s)-[r:CORRELATES_WITH]-(other:Symbol)
                WHERE ABS(r.correlation) > 0.5
                WITH s, COUNT(DISTINCT other) as degree
                RETURN degree
                """

                result = session.run(query, symbol=symbol)
                single_result = result.single()
                if single_result:
                    features["graph_degree_centrality"] = single_result["degree"]
                else:
                    features["graph_degree_centrality"] = 0.0

                # Betweenness centrality (how often on shortest paths)
                # Simplified version - count bridges between high-correlation pairs
                query = """
                MATCH (s:Symbol {name: $symbol})
                MATCH (s)-[r1:CORRELATES_WITH]-(a:Symbol)
                MATCH (s)-[r2:CORRELATES_WITH]-(b:Symbol)
                WHERE a.name < b.name
                  AND ABS(r1.correlation) > 0.7
                  AND ABS(r2.correlation) > 0.7
                  AND NOT (a)-[:CORRELATES_WITH]-(b)
                WITH COUNT(*) as bridge_count
                RETURN bridge_count
                """

                result = session.run(query, symbol=symbol)
                single_result = result.single()
                if single_result:
                    features["graph_betweenness"] = single_result["bridge_count"]
                else:
                    features["graph_betweenness"] = 0.0

                # Clustering coefficient (how interconnected are neighbors)
                query = """
                MATCH (s:Symbol {name: $symbol})
                MATCH (s)-[r1:CORRELATES_WITH]-(neighbor:Symbol)
                WHERE ABS(r1.correlation) > 0.5
                WITH s, COLLECT(DISTINCT neighbor) as neighbors
                UNWIND neighbors as n1
                UNWIND neighbors as n2
                WHERE n1.name < n2.name
                MATCH (n1)-[r:CORRELATES_WITH]-(n2)
                WHERE ABS(r.correlation) > 0.5
                WITH COUNT(*) as actual_edges, SIZE(neighbors) as n
                RETURN CASE WHEN n > 1 THEN (2.0 * actual_edges) / (n * (n - 1)) ELSE 0 END as clustering
                """

                result = session.run(query, symbol=symbol)
                single_result = result.single()
                if single_result:
                    features["graph_clustering_coefficient"] = single_result["clustering"]
                else:
                    features["graph_clustering_coefficient"] = 0.0

        except Exception as e:
            logger.error(f"Error extracting network features: {e}")

        return features

    def _extract_correlation_features(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """Extract correlation-based features."""
        features = {}

        if not self.driver:
            return features

        try:
            with self.driver.session() as session:
                # Average correlation with other symbols
                query = """
                MATCH (s:Symbol {name: $symbol})-[r:CORRELATES_WITH]-(other:Symbol)
                WHERE ABS(r.correlation) > 0.3
                WITH AVG(r.correlation) as avg_corr,
                     MAX(r.correlation) as max_corr,
                     MIN(r.correlation) as min_corr,
                     STDDEV(r.correlation) as std_corr
                RETURN avg_corr, max_corr, min_corr, std_corr
                """

                result = session.run(query, symbol=symbol)
                single_result = result.single()
                if single_result:
                    features["graph_avg_correlation"] = single_result["avg_corr"] or 0.0
                    features["graph_max_correlation"] = single_result["max_corr"] or 0.0
                    features["graph_min_correlation"] = single_result["min_corr"] or 0.0
                    features["graph_std_correlation"] = single_result["std_corr"] or 0.0
                else:
                    features["graph_avg_correlation"] = 0.0
                    features["graph_max_correlation"] = 0.0
                    features["graph_min_correlation"] = 0.0
                    features["graph_std_correlation"] = 0.0

                # Number of highly correlated symbols
                query = """
                MATCH (s:Symbol {name: $symbol})-[r:CORRELATES_WITH]-(other:Symbol)
                WHERE ABS(r.correlation) > 0.7
                RETURN COUNT(DISTINCT other) as highly_correlated_count
                """

                result = session.run(query, symbol=symbol)
                single_result = result.single()
                if single_result:
                    features["graph_highly_correlated_count"] = single_result[
                        "highly_correlated_count"
                    ]
                else:
                    features["graph_highly_correlated_count"] = 0

        except Exception as e:
            logger.error(f"Error extracting correlation features: {e}")

        return features

    def _extract_trading_pattern_features(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """Extract features from historical trading patterns."""
        features = {}

        if not self.driver:
            return features

        try:
            with self.driver.session() as session:
                # Recent trading activity (last 7 days)
                query = """
                MATCH (s:Symbol {name: $symbol})<-[:TRADED]-(t:Trade)
                WHERE t.timestamp > datetime($since)
                WITH COUNT(t) as recent_trades,
                     AVG(t.profit_loss) as avg_profit_loss,
                     SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(t) as win_rate
                RETURN recent_trades,
                       COALESCE(avg_profit_loss, 0) as avg_profit_loss,
                       COALESCE(win_rate, 0.5) as win_rate
                """

                since = (timestamp - timedelta(days=7)).isoformat()
                result = session.run(query, symbol=symbol, since=since)
                single_result = result.single()
                if single_result:
                    features["graph_recent_trades_7d"] = single_result["recent_trades"]
                    features["graph_avg_profit_loss_7d"] = single_result["avg_profit_loss"]
                    features["graph_win_rate_7d"] = single_result["win_rate"]
                else:
                    features["graph_recent_trades_7d"] = 0
                    features["graph_avg_profit_loss_7d"] = 0.0
                    features["graph_win_rate_7d"] = 0.5

                # Trading pattern cluster
                query = """
                MATCH (s:Symbol {name: $symbol})
                MATCH (s)-[:IN_PATTERN]->(p:Pattern)
                RETURN p.name as pattern_name, p.frequency as frequency
                ORDER BY frequency DESC
                LIMIT 1
                """

                result = session.run(query, symbol=symbol)
                single_result = result.single()
                if single_result:
                    # Convert pattern name to numeric (hash of first char for simplicity)
                    pattern_code = hash(single_result["pattern_name"]) % 1000 / 1000.0
                    features["graph_dominant_pattern"] = pattern_code
                    features["graph_pattern_frequency"] = single_result["frequency"]
                else:
                    features["graph_dominant_pattern"] = 0.0
                    features["graph_pattern_frequency"] = 0.0

        except Exception as e:
            logger.error(f"Error extracting trading pattern features: {e}")

        return features

    def _extract_market_regime_features(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """Extract market regime indicators."""
        features = {}

        if not self.driver:
            return features

        try:
            with self.driver.session() as session:
                # Current market regime
                query = """
                MATCH (s:Symbol {name: $symbol})-[:IN_REGIME]->(r:Regime)
                WHERE r.active = true
                RETURN r.name as regime_name, r.volatility as volatility, r.trend as trend
                """

                result = session.run(query, symbol=symbol)
                single_result = result.single()
                if single_result:
                    # Encode regime as numeric
                    regime_encoding = {
                        "BULL": 1.0,
                        "BEAR": -1.0,
                        "SIDEWAYS": 0.0,
                        "HIGH_VOLATILITY": 0.5,
                        "LOW_VOLATILITY": -0.5,
                    }
                    features["graph_market_regime"] = regime_encoding.get(
                        single_result["regime_name"], 0.0
                    )
                    features["graph_regime_volatility"] = single_result["volatility"] or 0.0
                    features["graph_regime_trend"] = single_result["trend"] or 0.0
                else:
                    features["graph_market_regime"] = 0.0
                    features["graph_regime_volatility"] = 0.0
                    features["graph_regime_trend"] = 0.0

        except Exception as e:
            logger.error(f"Error extracting market regime features: {e}")

        return features

    def extract_features_for_multiple_symbols(
        self,
        symbols: List[str],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract graph features for multiple symbols.

        Args:
            symbols: List of symbols
            timestamp: Current timestamp

        Returns:
            Dictionary mapping symbols to their features
        """
        if timestamp is None:
            timestamp = datetime.now()

        all_features = {}

        for symbol in symbols:
            try:
                features = self.extract_all_features(symbol, timestamp)
                all_features[symbol] = features
            except Exception as e:
                logger.error(f"Error extracting features for {symbol}: {e}")
                all_features[symbol] = {}

        return all_features

    def get_graph_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            "total_symbols": 0,
            "total_correlations": 0,
            "total_trades": 0,
            "total_patterns": 0,
            "avg_correlation": 0.0,
        }

        if not self.driver:
            return stats

        try:
            with self.driver.session() as session:
                # Count symbols
                result = session.run("MATCH (s:Symbol) RETURN COUNT(s) as count")
                stats["total_symbols"] = result.single()["count"]

                # Count correlations
                result = session.run(
                    """
                    MATCH (:Symbol)-[r:CORRELATES_WITH]-(:Symbol)
                    RETURN COUNT(r) as count
                """
                )
                stats["total_correlations"] = result.single()["count"]

                # Count trades
                result = session.run("MATCH (t:Trade) RETURN COUNT(t) as count")
                stats["total_trades"] = result.single()["count"]

                # Count patterns
                result = session.run("MATCH (p:Pattern) RETURN COUNT(p) as count")
                stats["total_patterns"] = result.single()["count"]

                # Average correlation
                result = session.run(
                    """
                    MATCH (:Symbol)-[r:CORRELATES_WITH]-(:Symbol)
                    RETURN AVG(r.correlation) as avg_corr
                """
                )
                single_result = result.single()
                stats["avg_correlation"] = single_result["avg_corr"] if single_result else 0.0

        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")

        return stats


async def populate_sample_graph_data(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    symbols: Optional[List[str]] = None,
):
    """
    Populate Neo4j with sample graph data for testing.

    This creates:
    - Symbol nodes
    - Correlation relationships
    - Trade nodes
    - Pattern nodes
    - Regime nodes

    Args:
        neo4j_uri: Neo4j URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        symbols: List of symbols to create
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]

    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j not available. Cannot populate sample data.")
        return

    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password),
    )

    try:
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing graph data")

            # Create symbol nodes
            for symbol in symbols:
                session.run(
                    """
                    MERGE (s:Symbol {name: $symbol})
                    SET s.created = datetime()
                """,
                    symbol=symbol,
                )
            logger.info(f"Created {len(symbols)} symbol nodes")

            # Create correlation relationships (random for demo)
            np.random.seed(42)
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i + 1 :]:
                    correlation = np.random.uniform(-0.8, 0.9)
                    session.run(
                        """
                        MATCH (s1:Symbol {name: $symbol1})
                        MATCH (s2:Symbol {name: $symbol2})
                        MERGE (s1)-[r:CORRELATES_WITH]-(s2)
                        SET r.correlation = $correlation,
                            r.updated = datetime()
                    """,
                        symbol1=symbol1,
                        symbol2=symbol2,
                        correlation=correlation,
                    )
            logger.info("Created correlation relationships")

            # Create regime nodes
            regimes = [
                ("BULL", 0.15, 1.0),
                ("BEAR", 0.25, -1.0),
                ("SIDEWAYS", 0.10, 0.0),
            ]

            for regime_name, volatility, trend in regimes:
                session.run(
                    """
                    CREATE (r:Regime {
                        name: $name,
                        volatility: $volatility,
                        trend: $trend,
                        active: false
                    })
                """,
                    name=regime_name,
                    volatility=volatility,
                    trend=trend,
                )

            # Set one regime as active
            session.run(
                """
                MATCH (r:Regime {name: 'BULL'})
                SET r.active = true
            """
            )

            # Link symbols to regime
            for symbol in symbols:
                session.run(
                    """
                    MATCH (s:Symbol {name: $symbol})
                    MATCH (r:Regime {active: true})
                    MERGE (s)-[:IN_REGIME]->(r)
                """,
                    symbol=symbol,
                )

            logger.info("Created regime nodes")

            # Create some sample trades
            for symbol in symbols[:3]:  # Create trades for first 3 symbols
                for _ in range(10):
                    profit_loss = np.random.uniform(-100, 200)
                    session.run(
                        """
                        MATCH (s:Symbol {name: $symbol})
                        CREATE (t:Trade {
                            symbol: $symbol,
                            profit_loss: $profit_loss,
                            timestamp: datetime() - duration('P' + toString(rand() * 30) + 'D')
                        })
                        CREATE (s)<-[:TRADED]-(t)
                    """,
                        symbol=symbol,
                        profit_loss=profit_loss,
                    )

            logger.info("Created sample trades")

            # Create pattern nodes
            patterns = ["MOMENTUM", "MEAN_REVERSION", "BREAKOUT", "REVERSAL"]
            for pattern_name in patterns:
                frequency = np.random.uniform(0.1, 0.5)
                session.run(
                    """
                    CREATE (p:Pattern {
                        name: $name,
                        frequency: $frequency
                    })
                """,
                    name=pattern_name,
                    frequency=frequency,
                )

                # Link some symbols to patterns
                for symbol in symbols[:2]:
                    if np.random.random() > 0.5:
                        session.run(
                            """
                            MATCH (s:Symbol {name: $symbol})
                            MATCH (p:Pattern {name: $pattern_name})
                            MERGE (s)-[:IN_PATTERN]->(p)
                        """,
                            symbol=symbol,
                            pattern_name=pattern_name,
                        )

            logger.info("Created pattern nodes")

        logger.info("Sample graph data populated successfully")

    except Exception as e:
        logger.error(f"Error populating sample data: {e}")
    finally:
        driver.close()


def create_graph_feature_extractor(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
) -> GraphFeatureExtractor:
    """
    Convenience function to create graph feature extractor.

    Args:
        neo4j_uri: Neo4j URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        GraphFeatureExtractor instance
    """
    return GraphFeatureExtractor(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )
