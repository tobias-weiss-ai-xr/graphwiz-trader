# Contributing to GraphWiz Trader

Thank you for your interest in contributing to GraphWiz Trader! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Reporting Issues](#reporting-issues)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

- Use inclusive language
- Respect differing viewpoints
- Accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or derogatory comments
- Personal or political attacks
- Public or private harassment
- Publishing private information
- Any other unethical or unprofessional conduct

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- GitHub account
- Familiarity with Python, asyncio, and trading concepts

### Setup Development Environment

```bash
# Fork and clone repository
git clone https://github.com/your-username/graphwiz-trader.git
cd graphwiz-trader

# Add upstream remote
git remote add upstream https://github.com/tobias-weiss-ai-xr/graphwiz-trader.git

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
./tests/run_all_tests.sh
```

---

## Development Workflow

### 1. Branch Strategy

```bash
# Main branches
main          # Production code
develop       # Integration branch

# Feature branches
feature/feature-name
bugfix/bug-description
hotfix/critical-fix
docs/documentation-update
```

### 2. Creating a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout develop
git merge upstream/develop

# Create feature branch
git checkout -b feature/your-feature-name
```

### 3. Making Changes

```bash
# Make your changes
# ...

# Run tests
pytest tests/

# Format code
black src/ tests/

# Type check
mypy src/

# Run linting
flake8 src/ tests/
```

### 4. Committing Changes

```bash
# Stage changes
git add .

# Commit with clear message
git commit -m "feat: add support for trailing stop loss

- Implement trailing stop logic in RiskManager
- Add configuration for trailing distance
- Update tests for trailing stop functionality
- Document trailing stop in API.md

Closes #123"
```

**Commit Message Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** feat, fix, docs, style, refactor, test, chore

**Example:**
```
feat(trading): add multi-exchange support

Implement ability to trade across multiple exchanges simultaneously
with automatic order routing and consolidation.

Closes #45
```

---

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

```python
# Good
class TradingEngine:
    """Main trading engine for executing orders."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trading engine."""
        self.config = config
        self.positions: List[Dict] = []

    async def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute trade with validation.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Optional limit price

        Returns:
            Order result dictionary

        Raises:
            OrderExecutionError: If order fails
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        # Implementation...
        return {"status": "success"}
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import Dict, List, Optional, Union, Any

def process_market_data(
    data: Dict[str, Any],
    indicators: Optional[List[str]] = None
) -> Dict[str, float]:
    """Process market data and return indicators."""
    if indicators is None:
        indicators = ["rsi", "macd"]

    # Implementation...
    return {"rsi": 65.0, "macd": 0.5}
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float
) -> float:
    """Calculate optimal position size based on risk parameters.

    Uses fixed risk formula to determine position size that risks
    exactly risk_per_trade percent of account balance.

    Args:
        account_balance: Total account value in USD
        risk_per_trade: Risk percentage (e.g., 0.02 for 2%)
        entry_price: Entry price per unit
        stop_loss: Stop loss price per unit

    Returns:
        Position size in base currency

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> size = calculate_position_size(10000, 0.02, 50000, 49000)
        >>> print(size)
        0.2
    """
    risk_amount = account_balance * risk_per_trade
    price_risk = abs(entry_price - stop_loss)
    return risk_amount / price_risk
```

### Error Handling

```python
# Good: Specific exceptions
try:
    order = await exchange.create_order(symbol, type, side, amount)
except ccxt.NetworkError as e:
    logger.error(f"Network error: {e}")
    raise OrderExecutionError("Network connection failed") from e
except ccxt.ExchangeError as e:
    logger.error(f"Exchange error: {e}")
    raise
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise

# Bad: Generic exception handling
try:
    order = await exchange.create_order(symbol, type, side, amount)
except:
    pass  # Never do this
```

### Async/Await Best Practices

```python
# Good: Concurrent operations
async def fetch_multiple_symbols(symbols: List[str]) -> Dict[str, Dict]:
    """Fetch data for multiple symbols concurrently."""
    tasks = [
        fetch_symbol_data(symbol)
        for symbol in symbols
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip(symbols, results))

# Bad: Sequential operations
async def fetch_multiple_symbols(symbols: List[str]) -> Dict[str, Dict]:
    results = {}
    for symbol in symbols:
        results[symbol] = await fetch_symbol_data(symbol)  # Slow
    return results
```

---

## Testing Guidelines

### Test Structure

```python
# tests/test_trading_engine.py
import pytest
from unittest.mock import Mock, AsyncMock, patch

@pytest.mark.asyncio
class TestTradingEngine:
    """Test suite for TradingEngine."""

    @pytest.fixture
    async def engine(self):
        """Create test engine instance."""
        mock_kg = Mock()
        mock_exchange = Mock()
        mock_exchange.create_order = AsyncMock(
            return_value={"id": "12345"}
        )
        return TradingEngine(
            exchanges={"binance": mock_exchange},
            knowledge_graph=mock_kg,
            risk_params={"max_position_size": 1000}
        )

    async def test_execute_buy_order(self, engine):
        """Test buy order execution."""
        result = await engine.execute_trade_signal({
            "symbol": "BTC/USDT",
            "price": 50000,
            "action": "buy",
            "quantity": 0.1
        })

        assert result["status"] == "success"
        assert result["action"] == "buy"
        assert "order_id" in result

    async def test_insufficient_balance(self, engine):
        """Test order rejection with insufficient balance."""
        # Test implementation...
        pass

    async def test_risk_limit_exceeded(self, engine):
        """Test order rejection when risk limit exceeded."""
        # Test implementation...
        pass
```

### Test Coverage Goals

- **Overall Coverage**: 80%+ required
- **Critical Paths**: 90%+ coverage (trading, risk, agents)
- **Edge Cases**: Must be tested

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific category
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m performance

# Run with coverage
pytest tests/ --cov=src/graphwiz_trader --cov-report=html

# Run specific file
pytest tests/test_trading_engine.py -v

# Run specific test
pytest tests/test_trading_engine.py::TestTradingEngine::test_execute_buy_order -v
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_result = MagicMock()

    mock_result.data.return_value = [{"key": "value"}]
    mock_session.run.return_value = mock_result
    mock_driver.session.return_value = mock_session

    return mock_driver

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "symbol": "BTC/USDT",
        "price": 50000,
        "volume": 1000.0,
        "timestamp": "2024-01-01T00:00:00Z"
    }
```

---

## Documentation

### Code Documentation

- All classes must have docstrings
- All public methods must have docstrings
- Use examples in docstrings for complex functions
- Document parameters, returns, and exceptions

### README Updates

When adding features:
1. Update feature list in README
2. Add usage example
3. Update configuration examples if needed
4. Add to API documentation

### API Documentation

For new APIs or changes:
1. Update API.md
2. Include function signature
3. Provide usage example
4. Document parameters and returns

### Changelog

Maintain CHANGELOG.md:

```markdown
## [Unreleased]

### Added
- Trailing stop loss support
- Multi-exchange order routing

### Changed
- Improved agent response time by 40%
- Updated Neo4j dependency to 5.15

### Fixed
- Memory leak in knowledge graph queries
- Order execution error handling

### Deprecated
- Old position sizing method (will be removed in v0.2.0)
```

---

## Pull Request Process

### Before Submitting

1. **Code Quality**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

2. **Tests**
   ```bash
   ./tests/run_all_tests.sh
   ```

3. **Documentation**
   - Update README.md if needed
   - Update API.md if adding/changing API
   - Add/update docstrings

### Creating Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create PR on GitHub
# Title: feat: add trailing stop loss support
# Description:
# ## Summary
# - Implements trailing stop loss functionality
# - Adds configuration options
#
# ## Changes
# - Added trailing_stop_enabled to risk config
# - Implemented trailing stop adjustment logic
# - Added tests for trailing stop
#
# ## Testing
# - All tests passing (80%+ coverage)
# - Manual testing in paper trading mode
#
# ## Checklist
# - [x] Code follows style guidelines
# - [x] Tests added/updated
# - [x] Documentation updated
# - [x] No merge conflicts with develop
#
# Closes #123
```

### PR Review Process

1. **Automated Checks**
   - All tests must pass
   - Code coverage >= 80%
   - No linting errors
   - Type checking passes

2. **Code Review**
   - At least one maintainer approval required
   - Address all review comments
   - Update tests if requested

3. **Integration**
   - Ensure no conflicts with develop branch
   - Test in integration environment
   - Verify documentation is complete

### PR Title Format

Use Conventional Commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Build process or auxiliary tool changes

---

## Reporting Issues

### Bug Reports

Include:
1. GraphWiz Trader version
2. Python version
3. Operating system
4. Steps to reproduce
5. Expected behavior
6. Actual behavior
7. Error messages/traceback
8. Configuration used

**Template:**
```markdown
### Description
[Clear description of the bug]

### Environment
- Version: 0.1.0
- Python: 3.10
- OS: Ubuntu 22.04

### Steps to Reproduce
1. Configure with paper_trading.yaml
2. Run: python -m graphwiz_trader.main
3. Wait for 10 minutes
4. System crashes with memory error

### Expected Behavior
System should run indefinitely without memory issues

### Actual Behavior
System crashes after 10 minutes

### Logs
[Error traceback and relevant logs]
```

### Feature Requests

Include:
1. Use case description
2. Why this feature would be useful
3. Proposed implementation (if you have ideas)
4. Alternatives considered
5. Mockups or examples (if applicable)

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes for significant contributions
- Credited in commit history

Thank you for contributing to GraphWiz Trader!
