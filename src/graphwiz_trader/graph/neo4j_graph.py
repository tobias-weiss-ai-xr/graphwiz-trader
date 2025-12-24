"""Neo4j knowledge graph implementation."""

from neo4j import GraphDatabase
from loguru import logger
from typing import Any, Dict, List, Optional


class KnowledgeGraph:
    """Neo4j knowledge graph connector."""

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

    def connect(self) -> None:
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j at {}", self.uri)
        except Exception as e:
            logger.error("Failed to connect to Neo4j: {}", e)
            raise

    def disconnect(self) -> None:
        """Disconnect from Neo4j database."""
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Neo4j")

    def query(self, cypher: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string
            **kwargs: Query parameters

        Returns:
            List of result dictionaries
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")

        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, kwargs)
            return [record.data() for record in result]

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

        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, kwargs)
            return result.consume()
