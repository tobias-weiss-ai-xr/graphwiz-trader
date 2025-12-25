# GraphWiz Trader Tests

This directory contains the comprehensive test suite for the GraphWiz Trader project.

## Test Structure

```
tests/
├── __init__.py              # Test package marker
├── conftest.py              # Pytest fixtures and mocks
├── test_basic.py            # Basic import and version tests
├── test_neo4j_graph.py      # Knowledge graph module tests
├── test_config.py           # Configuration system tests
├── test_trading_engine.py   # Trading engine tests
├── test_agents.py           # Agent orchestrator tests
├── test_main.py             # Integration tests for main system
├── test_edge_cases.py       # Edge case and error handling tests
└── TEST_SUMMARY.md          # Detailed test coverage summary
```

## Quick Start

### Install Dependencies

```bash
# Install package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov
```

### Run All Tests

```bash
# Basic test run
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src/graphwiz_trader --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src/graphwiz_trader --cov-report=html
# Open htmlcov/index.html in your browser
```

### Run Specific Test Files

```bash
# Test only knowledge graph
pytest tests/test_neo4j_graph.py -v

# Test only trading engine
pytest tests/test_trading_engine.py -v

# Test only specific test class
pytest tests/test_agents.py::TestAgentOrchestrator -v

# Test specific test
pytest tests/test_main.py::TestGraphWizTrader::test_start_success -v
```

## Test Categories

### Unit Tests
- **test_neo4j_graph.py**: Neo4j knowledge graph operations
- **test_config.py**: Configuration loading and parsing
- **test_trading_engine.py**: Trading engine functionality
- **test_agents.py**: AI agent orchestrator

### Integration Tests
- **test_main.py**: System orchestration and end-to-end workflows

### Edge Case Tests
- **test_edge_cases.py**: Boundary conditions and error scenarios

### Basic Tests
- **test_basic.py**: Import checks and basic functionality

## Fixtures

The `conftest.py` file provides reusable fixtures:

- `mock_neo4j_driver`: Mocked Neo4j database connection
- `neo4j_config`: Standard Neo4j configuration
- `trading_config`: Trading engine configuration
- `exchanges_config`: Exchange configurations
- `agents_config`: AI agent configurations
- `mock_kg`: Mock knowledge graph instance
- `mock_agent_orchestrator`: Mock agent orchestrator
- `temp_config_file`: Temporary YAML configuration file
- `mock_ccxt_exchange`: Mock CCXT exchange instance

## Coverage Goals

- **Unit Tests**: 90%+ coverage for core modules
- **Integration Tests**: Key user workflows covered
- **Edge Cases**: Error paths and boundary conditions tested

## CI/CD

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Multiple Python versions (3.10, 3.11, 3.12)

See `.github/workflows/ci.yml` for details.

## Contributing Tests

When adding new functionality:

1. Write tests first (TDD approach)
2. Ensure all existing tests still pass
3. Add fixtures to `conftest.py` if needed
4. Update this README if adding new test files
5. Maintain or improve test coverage

## Debugging Failed Tests

```bash
# Show detailed output
pytest tests/ -vv

# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Show local variables on failure
pytest tests/ -l

# Run only failed tests from last run
pytest tests/ --lf
```

## Test Best Practices

1. **One test per assert**: Keep tests focused and simple
2. **Descriptive names**: Use `test_<function>_<condition>` format
3. **Mock external dependencies**: Use fixtures for isolation
4. **Test both success and failure**: Cover error paths
5. **Use type hints**: Helps with test readability
6. **Keep tests fast**: Unit tests should run in milliseconds

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
