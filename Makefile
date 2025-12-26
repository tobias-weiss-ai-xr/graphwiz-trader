.PHONY: help test test-unit test-fast test-slow test-integration test-hft test-property test-parallel coverage clean

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

test:  ## Run all tests
	python -m pytest tests/ --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

test-unit:  ## Run unit tests only
	python -m pytest tests/ -m unit --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

test-fast:  ## Run fast tests only (exclude slow)
	python -m pytest tests/ -m "not slow" --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

test-slow:  ## Run slow tests only
	python -m pytest tests/ -m slow --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

test-integration:  ## Run integration tests only
	python -m pytest tests/ -m integration --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

test-hft:  ## Run HFT tests only
	python -m pytest tests/ -m hft --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

test-property:  ## Run property-based tests only
	python -m pytest tests/ -m property --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

test-parallel:  ## Run tests in parallel
	python -m pytest tests/ -n auto --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

coverage:  ## Run tests with coverage report
	python -m pytest tests/ --cov=src/graphwiz_trader --cov-report=html --cov-report=term --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

coverage-fast:  ## Run fast tests with coverage
	python -m pytest tests/ -m "not slow" --cov=src/graphwiz_trader --cov-report=html --cov-report=term --ignore=tests/test_neo4j_graph.py --ignore=tests/test_main.py

clean:  ## Clean up test artifacts
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .tox *.egg-info dist build 2>/dev/null || true
	@echo "Cleaned up test artifacts"
