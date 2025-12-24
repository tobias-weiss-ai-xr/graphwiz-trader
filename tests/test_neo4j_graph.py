"""Tests for Neo4j knowledge graph module."""

import pytest
from unittest.mock import MagicMock, patch
from neo4j import GraphDatabase

from graphwiz_trader.graph import KnowledgeGraph


class TestKnowledgeGraph:
    """Test suite for KnowledgeGraph class."""

    def test_initialization(self, neo4j_config):
        """Test KnowledgeGraph initialization."""
        kg = KnowledgeGraph(neo4j_config)

        assert kg.uri == "bolt://localhost:7687"
        assert kg.username == "neo4j"
        assert kg.password == "test_password"
        assert kg.database == "neo4j"
        assert kg.driver is None

    def test_initialization_default_values(self):
        """Test initialization with default config values."""
        config = {}
        kg = KnowledgeGraph(config)

        assert kg.uri == "bolt://localhost:7687"
        assert kg.username == "neo4j"
        assert kg.password == ""
        assert kg.database == "neo4j"

    @patch('graphwiz_trader.graph.neo4j_graph.GraphDatabase.driver')
    def test_connect_success(self, mock_driver_class, neo4j_config, mock_neo4j_driver):
        """Test successful connection to Neo4j."""
        mock_driver_class.return_value = mock_neo4j_driver

        kg = KnowledgeGraph(neo4j_config)
        kg.connect()

        assert kg.driver is not None
        mock_driver_class.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "test_password")
        )
        mock_neo4j_driver.verify_connectivity.assert_called_once()

    @patch('graphwiz_trader.graph.neo4j_graph.GraphDatabase.driver')
    def test_connect_failure(self, mock_driver_class, neo4j_config):
        """Test connection failure handling."""
        mock_driver_class.side_effect = Exception("Connection failed")

        kg = KnowledgeGraph(neo4j_config)

        with pytest.raises(Exception) as exc_info:
            kg.connect()

        assert "Connection failed" in str(exc_info.value)

    def test_disconnect(self, neo4j_config, mock_neo4j_driver):
        """Test disconnecting from Neo4j."""
        kg = KnowledgeGraph(neo4j_config)
        kg.driver = mock_neo4j_driver

        kg.disconnect()

        mock_neo4j_driver.close.assert_called_once()

    def test_disconnect_no_driver(self, neo4j_config):
        """Test disconnect when no driver exists."""
        kg = KnowledgeGraph(neo4j_config)
        # Should not raise an exception
        kg.disconnect()

    def test_query_success(self, neo4j_config, mock_neo4j_driver):
        """Test successful query execution."""
        kg = KnowledgeGraph(neo4j_config)
        kg.driver = mock_neo4j_driver

        cypher = "MATCH (n) RETURN n LIMIT 1"
        results = kg.query(cypher)

        assert results == [{"key": "value"}]
        mock_neo4j_driver.session.assert_called_once_with(database="neo4j")
        session_mock = mock_neo4j_driver.session.return_value.__enter__.return_value
        session_mock.run.assert_called_once_with(cypher, {})

    def test_query_with_parameters(self, neo4j_config, mock_neo4j_driver):
        """Test query execution with parameters."""
        kg = KnowledgeGraph(neo4j_config)
        kg.driver = mock_neo4j_driver

        cypher = "MATCH (n {name: $name}) RETURN n"
        params = {"name": "BTC"}
        results = kg.query(cypher, name="BTC")

        assert results == [{"key": "value"}]
        session_mock = mock_neo4j_driver.session.return_value.__enter__.return_value
        session_mock.run.assert_called_once_with(cypher, params)

    def test_query_no_connection(self, neo4j_config):
        """Test query when not connected."""
        kg = KnowledgeGraph(neo4j_config)

        with pytest.raises(RuntimeError) as exc_info:
            kg.query("MATCH (n) RETURN n")

        assert "Not connected to Neo4j" in str(exc_info.value)

    def test_write_success(self, neo4j_config, mock_neo4j_driver):
        """Test successful write operation."""
        kg = KnowledgeGraph(neo4j_config)
        kg.driver = mock_neo4j_driver

        cypher = "CREATE (n:Asset {symbol: 'BTC'})"
        result = kg.write(cypher)

        assert result is not None
        mock_neo4j_driver.session.assert_called_once_with(database="neo4j")
        session_mock = mock_neo4j_driver.session.return_value.__enter__.return_value
        session_mock.run.assert_called_once_with(cypher, {})

    def test_write_with_parameters(self, neo4j_config, mock_neo4j_driver):
        """Test write operation with parameters."""
        kg = KnowledgeGraph(neo4j_config)
        kg.driver = mock_neo4j_driver

        cypher = "CREATE (n:Asset {symbol: $symbol})"
        params = {"symbol": "ETH"}
        result = kg.write(cypher, symbol="ETH")

        assert result is not None
        session_mock = mock_neo4j_driver.session.return_value.__enter__.return_value
        session_mock.run.assert_called_once_with(cypher, params)

    def test_write_no_connection(self, neo4j_config):
        """Test write when not connected."""
        kg = KnowledgeGraph(neo4j_config)

        with pytest.raises(RuntimeError) as exc_info:
            kg.write("CREATE (n:Asset {symbol: 'BTC'})")

        assert "Not connected to Neo4j" in str(exc_info.value)
