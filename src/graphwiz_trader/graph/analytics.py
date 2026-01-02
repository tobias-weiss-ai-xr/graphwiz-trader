"""Graph analytics module for market insights and pattern detection."""

from loguru import logger
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN

from .neo4j_graph import KnowledgeGraph
from .models import IndicatorType, SignalType


class GraphAnalytics:
    """Advanced graph analytics for market intelligence."""

    def __init__(self, graph: KnowledgeGraph):
        """Initialize analytics engine.

        Args:
            graph: KnowledgeGraph instance
        """
        self.graph = graph
        logger.info("Initialized GraphAnalytics engine")

    # ========== CORRELATION ANALYSIS ==========

    def calculate_correlation_matrix(
        self,
        symbols: List[str],
        exchange: str,
        window: str = "24h",
        min_correlation: float = 0.5
    ) -> Dict[str, Any]:
        """Calculate correlation matrix for multiple assets.

        Args:
            symbols: List of asset symbols
            exchange: Exchange name
            window: Time window for calculation
            min_correlation: Minimum correlation to store

        Returns:
            Dictionary with correlation matrix and metadata
        """
        logger.info("Calculating correlation matrix for {} symbols", len(symbols))

        end_time = datetime.utcnow()
        start_time = end_time - self._parse_window(window)

        # Fetch price data for all symbols
        price_data = {}
        for symbol in symbols:
            ohlcv_data = self.graph.get_ohlcv(
                symbol, exchange, start_time, end_time, "1h", limit=1000
            )
            if ohlcv_data:
                # Extract close prices in chronological order
                closes = [float(o["close"]) for o in reversed(ohlcv_data)]
                price_data[symbol] = closes

        # Calculate correlations
        correlations = []
        correlation_matrix = defaultdict(dict)

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if sym1 in price_data and sym2 in price_data:
                    # Align data lengths
                    min_len = min(len(price_data[sym1]), len(price_data[sym2]))
                    data1 = price_data[sym1][-min_len:]
                    data2 = price_data[sym2][-min_len:]

                    if len(data1) > 10:  # Need sufficient data points
                        corr, p_value = stats.pearsonr(data1, data2)

                        if abs(corr) >= min_correlation:
                            correlation_matrix[sym1][sym2] = corr
                            correlation_matrix[sym2][sym1] = corr

                            correlations.append({
                                "symbol1": sym1,
                                "symbol2": sym2,
                                "correlation": corr,
                                "p_value": p_value
                            })

                            # Store in graph
                            from .models import CorrelationRelationship
                            corr_rel = CorrelationRelationship(
                                symbol1=sym1,
                                symbol2=sym2,
                                correlation_coefficient=corr,
                                p_value=p_value,
                                window=window
                            )
                            self.graph.create_correlation(corr_rel)

        return {
            "correlation_matrix": dict(correlation_matrix),
            "correlations": correlations,
            "window": window,
            "exchange": exchange,
            "timestamp": datetime.utcnow().isoformat()
        }

    def find_correlation_clusters(
        self,
        symbols: List[str],
        exchange: str,
        window: str = "24h",
        eps: float = 0.3
    ) -> List[List[str]]:
        """Find clusters of highly correlated assets using DBSCAN.

        Args:
            symbols: List of asset symbols
            exchange: Exchange name
            window: Time window
            eps: DBSCAN epsilon parameter for clustering

        Returns:
            List of clusters (each cluster is a list of symbols)
        """
        result = self.calculate_correlation_matrix(symbols, exchange, window)
        corr_matrix = result["correlation_matrix"]

        if not corr_matrix:
            return []

        # Build feature matrix from correlations
        symbol_list = list(corr_matrix.keys())
        n = len(symbol_list)

        if n < 2:
            return []

        # Create correlation-based feature matrix
        features = np.zeros((n, n))
        for i, sym1 in enumerate(symbol_list):
            for j, sym2 in enumerate(symbol_list):
                if sym1 == sym2:
                    features[i][j] = 1.0
                elif sym2 in corr_matrix.get(sym1, {}):
                    features[i][j] = abs(corr_matrix[sym1][sym2])
                else:
                    features[i][j] = 0.0

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
        # Convert to distance matrix (1 - correlation)
        distances = 1 - np.abs(features)
        labels = clustering.fit_predict(distances)

        # Group by cluster
        clusters = defaultdict(list)
        for symbol, label in zip(symbol_list, labels):
            if label >= 0:  # Ignore noise points
                clusters[label].append(symbol)

        return list(clusters.values())

    def update_correlations(
        self,
        exchange: str,
        asset_type: Optional[str] = None,
        window: str = "24h"
    ) -> int:
        """Update correlations for all assets on an exchange.

        Args:
            exchange: Exchange name
            asset_type: Optional asset type filter
            window: Time window

        Returns:
            Number of correlations updated
        """
        from .models import AssetType

        if asset_type:
            assets = self.graph.get_assets_by_type(AssetType(asset_type))
        else:
            assets = self.graph.query("MATCH (a:Asset)-[:TRADED_ON]->(e:Exchange {name: $name}) RETURN a", name=exchange)
            assets = [r["a"] for r in assets]

        symbols = [a["symbol"] for a in assets]

        if len(symbols) < 2:
            logger.warning("Not enough assets for correlation analysis")
            return 0

        result = self.calculate_correlation_matrix(symbols, exchange, window)
        logger.info("Updated {} correlations for {}", len(result["correlations"]), exchange)

        return len(result["correlations"])

    # ========== ARBITRAGE DETECTION ==========

    def detect_arbitrage_opportunities(
        self,
        exchanges: List[str],
        symbols: Optional[List[str]] = None,
        min_profit_percentage: float = 0.5,
        include_fees: bool = True
    ) -> List[Dict[str, Any]]:
        """Detect arbitrage opportunities across exchanges.

        Args:
            exchanges: List of exchange names to check
            symbols: Optional list of symbols to check (default: all)
            min_profit_percentage: Minimum profit threshold
            include_fees: Whether to account for trading fees

        Returns:
            List of arbitrage opportunities
        """
        logger.info("Scanning for arbitrage opportunities across {} exchanges", len(exchanges))

        opportunities = []

        # Get latest prices from all exchanges
        prices_by_symbol = defaultdict(dict)

        for exchange in exchanges:
            exchange_data = self.graph.get_exchange(exchange)
            if not exchange_data:
                continue

            taker_fee = exchange_data.get("taker_fee", 0.001)

            if symbols:
                check_symbols = symbols
            else:
                # Get all symbols traded on this exchange
                assets = self.graph.query(
                    "MATCH (a:Asset)-[:TRADED_ON]->(e:Exchange {name: $name}) RETURN a",
                    name=exchange
                )
                check_symbols = [r["a"]["symbol"] for r in assets]

            for symbol in check_symbols:
                latest = self.graph.get_latest_ohlcv(symbol, exchange, "1m", count=1)
                if latest:
                    price = float(latest[0]["close"])
                    prices_by_symbol[symbol][exchange] = {
                        "price": price,
                        "fee": taker_fee
                    }

        # Find arbitrage opportunities
        for symbol, exchange_prices in prices_by_symbol.items():
            if len(exchange_prices) < 2:
                continue

            exchanges_list = list(exchange_prices.keys())

            for i in range(len(exchanges_list)):
                for j in range(i+1, len(exchanges_list)):
                    ex1 = exchanges_list[i]
                    ex2 = exchanges_list[j]

                    price1 = exchange_prices[ex1]["price"]
                    price2 = exchange_prices[ex2]["price"]
                    fee1 = exchange_prices[ex1]["fee"]
                    fee2 = exchange_prices[ex2]["fee"]

                    # Calculate profit for buy on ex1, sell on ex2
                    if price1 < price2:  # Buy low, sell high
                        buy_exchange, sell_exchange = ex1, ex2
                        buy_price, sell_price = price1, price2
                    else:
                        buy_exchange, sell_exchange = ex2, ex1
                        buy_price, sell_price = price2, price1

                    # Calculate gross profit percentage
                    gross_profit_pct = ((sell_price - buy_price) / buy_price) * 100

                    # Subtract fees
                    if include_fees:
                        total_fee_pct = (fee1 + fee2) * 100
                        net_profit_pct = gross_profit_pct - total_fee_pct
                    else:
                        net_profit_pct = gross_profit_pct

                    if net_profit_pct >= min_profit_percentage:
                        opportunities.append({
                            "symbol": symbol,
                            "buy_exchange": buy_exchange,
                            "sell_exchange": sell_exchange,
                            "buy_price": buy_price,
                            "sell_price": sell_price,
                            "gross_profit_pct": gross_profit_pct,
                            "net_profit_pct": net_profit_pct,
                            "spread_pct": ((sell_price - buy_price) / buy_price) * 100,
                            "timestamp": datetime.utcnow().isoformat()
                        })

                        # Store in graph
                        from .models import ArbitrageRelationship
                        arb = ArbitrageRelationship(
                            symbol=symbol,
                            exchange1=buy_exchange,
                            exchange2=sell_exchange,
                            price1=buy_price,
                            price2=sell_price,
                            spread_percentage=net_profit_pct,
                            profit_potential=net_profit_pct
                        )
                        self.graph.create_arbitrage_opportunity(arb)

        # Sort by profit potential
        opportunities.sort(key=lambda x: x["net_profit_pct"], reverse=True)

        logger.info("Found {} arbitrage opportunities", len(opportunities))
        return opportunities

    def detect_triangular_arbitrage(
        self,
        base_currency: str = "USD",
        exchanges: Optional[List[str]] = None,
        min_profit_percentage: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Detect triangular arbitrage opportunities.

        Args:
            base_currency: Base currency (e.g., "USD")
            exchanges: Optional list of exchanges (default: all)
            min_profit_percentage: Minimum profit threshold

        Returns:
            List of triangular arbitrage opportunities
        """
        if not exchanges:
            exchanges = self.graph.query("MATCH (e:Exchange) RETURN e.name AS name")
            exchanges = [r["name"] for r in exchanges]

        opportunities = []

        for exchange in exchanges:
            # Find all assets traded on this exchange
            assets = self.graph.query(
                "MATCH (a:Asset)-[:TRADED_ON]->(e:Exchange {name: $name}) RETURN a",
                name=exchange
            )

            # Build currency pairs graph
            pairs = defaultdict(list)
            for asset_record in assets:
                asset = asset_record["a"]
                if asset.get("quote_currency") == base_currency:
                    pairs[asset["base_currency"]].append(asset["symbol"])

            # Look for triangular paths: BASE -> CUR1 -> CUR2 -> BASE
            for cur1 in pairs:
                for cur2 in pairs:
                    if cur1 != cur2:
                        # Check if CUR1/CUR2 pair exists
                        pair_symbol = f"{cur1}/{cur2}"
                        reverse_pair = f"{cur2}/{cur1}"

                        # Get current prices
                        base_cur1 = self.graph.get_latest_ohlcv(f"{cur1}/{base_currency}", exchange, "1m", count=1)
                        base_cur2 = self.graph.get_latest_ohlcv(f"{cur2}/{base_currency}", exchange, "1m", count=1)
                        cur1_cur2 = self.graph.get_latest_ohlcv(pair_symbol, exchange, "1m", count=1)

                        if not cur1_cur2:
                            cur1_cur2 = self.graph.get_latest_ohlcv(reverse_pair, exchange, "1m", count=1)

                        if base_cur1 and base_cur2 and cur1_cur2:
                            price_base_cur1 = float(base_cur1[0]["close"])
                            price_base_cur2 = float(base_cur2[0]["close"])
                            price_cur1_cur2 = float(cur1_cur2[0]["close"])

                            # Calculate triangular arbitrage profit
                            # Path: BASE -> CUR1 -> CUR2 -> BASE
                            amount = 1.0
                            amount_after_step1 = amount / price_base_cur1  # Buy CUR1 with BASE
                            amount_after_step2 = amount_after_step1 * price_cur1_cur2  # Sell CUR1 for CUR2
                            amount_after_step3 = amount_after_step2 * price_base_cur2  # Sell CUR2 for BASE

                            profit_pct = ((amount_after_step3 - amount) / amount) * 100

                            if profit_pct >= min_profit_percentage:
                                opportunities.append({
                                    "exchange": exchange,
                                    "path": f"{base_currency} -> {cur1} -> {cur2} -> {base_currency}",
                                    "profit_percentage": profit_pct,
                                    "prices": {
                                        f"{cur1}/{base_currency}": price_base_cur1,
                                        f"{cur2}/{base_currency}": price_base_cur2,
                                        f"{cur1}/{cur2}": price_cur1_cur2
                                    },
                                    "timestamp": datetime.utcnow().isoformat()
                                })

        opportunities.sort(key=lambda x: x["profit_percentage"], reverse=True)
        logger.info("Found {} triangular arbitrage opportunities", len(opportunities))

        return opportunities

    # ========== MARKET IMPACT ANALYSIS ==========

    def analyze_market_impact(
        self,
        symbol: str,
        exchange: str,
        volume_thresholds: List[float] = [1000, 10000, 100000],
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze market impact of trades at different volume levels.

        Args:
            symbol: Asset symbol
            exchange: Exchange name
            volume_thresholds: List of volume thresholds to analyze
            lookback_hours: Hours to look back for analysis

        Returns:
            Market impact analysis results
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)

        results = {}

        for threshold in volume_thresholds:
            impact = self.graph.analyze_market_impact(symbol, exchange, threshold)
            results[f"threshold_{threshold}"] = impact

        # Calculate order book depth impact
        orderbook = self.graph.get_latest_orderbook(symbol, exchange, limit=1)

        depth_analysis = {}
        if orderbook:
            ob = orderbook[0]
            depth_analysis = {
                "bid_depth": ob.get("bid_depth", 0),
                "ask_depth": ob.get("ask_depth", 0),
                "spread": ob.get("spread", 0),
                "spread_percentage": ob.get("spread_percentage", 0)
            }

        return {
            "symbol": symbol,
            "exchange": exchange,
            "volume_impacts": results,
            "depth_analysis": depth_analysis,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    # ========== PATTERN DETECTION ==========

    def detect_pump_and_dump(
        self,
        symbol: str,
        exchange: str,
        lookback_hours: int = 24,
        volume_spike_threshold: float = 3.0,
        price_change_threshold: float = 20.0
    ) -> Dict[str, Any]:
        """Detect potential pump and dump patterns.

        Args:
            symbol: Asset symbol
            exchange: Exchange name
            lookback_hours: Hours to analyze
            volume_spike_threshold: Volume spike multiplier threshold
            price_change_threshold: Price change percentage threshold

        Returns:
            Pump and dump detection results
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)

        # Get price and volume data
        ohlcv_data = self.graph.get_ohlcv(
            symbol, exchange, start_time, end_time, "1h", limit=lookback_hours
        )

        if not ohlcv_data or len(ohlcv_data) < 12:
            return {
                "detected": False,
                "reason": "Insufficient data"
            }

        # Sort chronologically
        ohlcv_data = list(reversed(ohlcv_data))

        volumes = [float(o["volume"]) for o in ohlcv_data]
        closes = [float(o["close"]) for o in ohlcv_data]

        # Calculate average volume for first half
        mid_point = len(volumes) // 2
        baseline_volume = np.mean(volumes[:mid_point])
        recent_volume = np.mean(volumes[mid_point:])

        # Calculate price changes
        price_start = closes[0]
        price_max = max(closes)
        price_current = closes[-1]

        price_increase_pct = ((price_max - price_start) / price_start) * 100
        price_decrease_pct = ((price_max - price_current) / price_max) * 100 if price_max > 0 else 0
        volume_spike = recent_volume / baseline_volume if baseline_volume > 0 else 0

        detected = (
            volume_spike >= volume_spike_threshold and
            price_increase_pct >= price_change_threshold and
            price_decrease_pct >= price_change_threshold * 0.5  # Significant drop after peak
        )

        return {
            "detected": detected,
            "symbol": symbol,
            "exchange": exchange,
            "volume_spike": volume_spike,
            "price_increase_pct": price_increase_pct,
            "price_decrease_pct": price_decrease_pct,
            "peak_price": price_max,
            "current_price": price_current,
            "baseline_volume": baseline_volume,
            "recent_volume": recent_volume,
            "timestamp": datetime.utcnow().isoformat()
        }

    def detect_accumulation_distribution(
        self,
        symbol: str,
        exchange: str,
        lookback_hours: int = 48
    ) -> Dict[str, Any]:
        """Detect accumulation or distribution patterns.

        Args:
            symbol: Asset symbol
            exchange: Exchange name
            lookback_hours: Hours to analyze

        Returns:
            Accumulation/distribution analysis
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)

        # Get trade data
        trades = self.graph.get_trades(symbol, exchange, start_time, end_time, limit=10000)

        if not trades:
            return {
                "detected": False,
                "pattern": "insufficient_data"
            }

        # Separate buy and sell trades
        buy_trades = [t for t in trades if t["side"] == "BUY"]
        sell_trades = [t for t in trades if t["side"] == "SELL"]

        buy_volume = sum(float(t["amount"]) for t in buy_trades)
        sell_volume = sum(float(t["amount"]) for t in sell_trades)
        total_volume = buy_volume + sell_volume

        # Calculate buy/sell ratio
        buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else float("inf")

        # Detect pattern
        if buy_sell_ratio > 1.5:
            pattern = "accumulation"
            confidence = min((buy_sell_ratio - 1.0) / 2.0, 1.0)
        elif buy_sell_ratio < 0.67:
            pattern = "distribution"
            confidence = min((1.0 - buy_sell_ratio) / 2.0, 1.0)
        else:
            pattern = "neutral"
            confidence = 0.0

        # Get price trend
        ohlcv = self.graph.get_ohlcv(symbol, exchange, start_time, end_time, "1h", limit=lookback_hours)
        price_trend = 0.0

        if ohlcv and len(ohlcv) > 1:
            ohlcv = list(reversed(ohlcv))
            start_price = float(ohlcv[0]["close"])
            end_price = float(ohlcv[-1]["close"])
            price_trend = ((end_price - start_price) / start_price) * 100

        return {
            "detected": confidence > 0.3,
            "pattern": pattern,
            "confidence": confidence,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "buy_sell_ratio": buy_sell_ratio,
            "price_trend_pct": price_trend,
            "trade_count": len(trades),
            "timestamp": datetime.utcnow().isoformat()
        }

    # ========== SENTIMENT PROPAGATION ==========

    def track_sentiment_propagation(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Track how sentiment propagates through correlated assets.

        Args:
            symbol: Asset symbol
            lookback_hours: Hours to analyze

        Returns:
            Sentiment propagation analysis
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)

        # Get correlated assets
        correlated = self.graph.get_correlated_assets(symbol, min_correlation=0.7, limit=20)

        results = {
            "source_symbol": symbol,
            "correlated_assets": [],
            "sentiment_lag_analysis": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Get sentiment for source symbol
        source_sentiment = self.graph.get_average_sentiment(symbol, start_time, end_time)

        for corr in correlated:
            corr_symbol = corr["symbol"]
            correlation = corr["correlation"]

            # Get sentiment for correlated asset
            corr_sentiment = self.graph.get_average_sentiment(corr_symbol, start_time, end_time)

            if corr_sentiment and source_sentiment:
                results["correlated_assets"].append({
                    "symbol": corr_symbol,
                    "correlation": correlation,
                    "sentiment_score": corr_sentiment.get("avg_score", 0),
                    "sentiment_volume": corr_sentiment.get("total_volume", 0)
                })

        # Calculate sentiment correlation
        if len(results["correlated_assets"]) > 0:
            sentiment_scores = [a["sentiment_score"] for a in results["correlated_assets"]]
            results["avg_sentiment_score"] = np.mean(sentiment_scores) if sentiment_scores else 0
            results["sentiment_std"] = np.std(sentiment_scores) if sentiment_scores else 0

        return results

    # ========== UTILITY METHODS ==========

    def _parse_window(self, window: str) -> timedelta:
        """Parse time window string to timedelta.

        Args:
            window: Window string (e.g., "24h", "7d", "1w")

        Returns:
            timedelta object
        """
        window = window.lower()

        if "h" in window:
            hours = int(window.replace("h", "").strip())
            return timedelta(hours=hours)
        elif "d" in window:
            days = int(window.replace("d", "").strip())
            return timedelta(days=days)
        elif "w" in window:
            weeks = int(window.replace("w", "").strip())
            return timedelta(weeks=weeks)
        else:
            # Default to 24 hours
            return timedelta(hours=24)

    def generate_market_report(
        self,
        symbols: List[str],
        exchange: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate comprehensive market report.

        Args:
            symbols: List of symbols to analyze
            exchange: Exchange name
            lookback_hours: Hours to analyze

        Returns:
            Comprehensive market report
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)

        report = {
            "exchange": exchange,
            "analysis_period": f"{lookback_hours}h",
            "timestamp": datetime.utcnow().isoformat(),
            "assets": []
        }

        # Analyze correlations
        correlations = self.calculate_correlation_matrix(symbols, exchange, f"{lookback_hours}h")
        report["correlation_summary"] = {
            "total_correlations": len(correlations["correlations"]),
            "high_correlations": [
                c for c in correlations["correlations"]
                if abs(c["correlation"]) > 0.8
            ]
        }

        # Analyze each asset
        for symbol in symbols:
            asset_report = {
                "symbol": symbol
            }

            # Price stats
            price_stats = self.graph.get_price_history_stats(
                symbol, exchange, start_time, end_time
            )
            asset_report["price_stats"] = price_stats

            # Market impact
            impact = self.analyze_market_impact(symbol, exchange)
            asset_report["market_impact"] = impact

            # Pattern detection
            pump_dump = self.detect_pump_and_dump(symbol, exchange)
            asset_report["pump_dump_detection"] = pump_dump

            acc_dist = self.detect_accumulation_distribution(symbol, exchange)
            asset_report["accumulation_distribution"] = acc_dist

            # Sentiment
            sentiment = self.graph.get_average_sentiment(symbol, start_time, end_time)
            asset_report["sentiment"] = sentiment

            report["assets"].append(asset_report)

        # Arbitrage opportunities
        arbitrage = self.detect_arbitrage_opportunities([exchange], symbols)
        report["arbitrage_opportunities"] = arbitrage[:10]  # Top 10

        return report
