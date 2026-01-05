#!/usr/bin/env python3
"""
Configuration validation tests.

Tests that configuration files are valid and complete.
"""

import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    print("Install: pip install pyyaml python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()


class TestEnvironmentConfiguration(unittest.TestCase):
    """Test .env configuration."""

    def test_env_file_exists(self):
        """Test .env file exists."""
        env_path = Path(__file__).parent.parent / '.env'
        print(f"\n  Checking: {env_path}")
        self.assertTrue(env_path.exists(), ".env file must exist")

    def test_kraken_credentials_set(self):
        """Test Kraken API credentials are configured."""
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_API_SECRET')

        print(f"\n  KRAKEN_API_KEY: {'Set' if api_key else 'Not set'}")
        print(f"  KRAKEN_API_SECRET: {'Set' if api_secret else 'Not set'}")

        # Only warn, don't fail - might be testing without credentials
        if not api_key or not api_secret:
            print("  ⚠️  Warning: Kraken credentials not set")
        else:
            self.assertIsNotNone(api_key)
            self.assertIsNotNone(api_secret)
            self.assertGreater(len(api_key), 10)
            self.assertGreater(len(api_secret), 10)

    def test_trading_mode_set(self):
        """Test TRADING_MODE is configured."""
        mode = os.getenv('TRADING_MODE', 'test')
        print(f"\n  TRADING_MODE: {mode}")
        self.assertIn(mode, ['test', 'live', 'paper'])

    def test_neo4j_config(self):
        """Test Neo4j configuration."""
        uri = os.getenv('NEO4J_URI')
        username = os.getenv('NEO4J_USERNAME')

        print(f"\n  NEO4J_URI: {uri or 'Not set'}")
        print(f"  NEO4J_USERNAME: {username or 'Not set'}")

        # Optional - Neo4j not required for basic trading
        if uri:
            self.assertTrue(uri.startswith('bolt://'))


class TestLiveTradingConfig(unittest.TestCase):
    """Test germany_live_custom.yaml configuration."""

    def setUp(self):
        """Load configuration file."""
        config_path = Path(__file__).parent.parent / 'config/germany_live_custom.yaml'
        self.config_path = config_path

        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = None

    def test_config_file_exists(self):
        """Test config file exists."""
        print(f"\n  Checking: {self.config_path}")
        self.assertTrue(self.config_path.exists(), "germany_live_custom.yaml must exist")

    def test_live_trading_section(self):
        """Test live_trading configuration section."""
        if not self.config:
            self.skipTest("Config file not loaded")

        self.assertIn('live_trading', self.config)
        live = self.config['live_trading']

        print(f"\n  Live trading config:")
        print(f"    Max position: €{live.get('max_position_eur', 'N/A')}")
        print(f"    Max daily loss: €{live.get('max_daily_loss_eur', 'N/A')}")
        print(f"    Max daily trades: {live.get('max_daily_trades', 'N/A')}")

        # Validate values
        self.assertGreater(live.get('max_position_eur', 0), 0)
        self.assertGreater(live.get('max_daily_loss_eur', 0), 0)
        self.assertGreater(live.get('max_daily_trades', 0), 0)
        self.assertTrue(live.get('require_confirmation', False))

    def test_risk_section(self):
        """Test risk management configuration."""
        if not self.config:
            self.skipTest("Config file not loaded")

        self.assertIn('risk', self.config)
        risk = self.config['risk']

        print(f"\n  Risk config:")
        print(f"    Max position size: €{risk.get('max_position_size', 'N/A')}")
        print(f"    Stop loss: {risk.get('stop_loss', {}).get('default_percent', 'N/A')}")
        print(f"    Take profit: {risk.get('take_profit', {}).get('default_percent', 'N/A')}")

        # Validate stop loss is reasonable
        stop_loss = risk.get('stop_loss', {}).get('default_percent', 0)
        self.assertGreater(stop_loss, 0)
        self.assertLess(stop_loss, 0.1)  # Should be < 10%

        # Validate take profit
        take_profit = risk.get('take_profit', {}).get('default_percent', 0)
        self.assertGreater(take_profit, 0)
        self.assertLess(take_profit, 0.2)  # Should be < 20%

        # Risk/reward should be at least 1:1
        self.assertGreater(take_profit, stop_loss)

    def test_exchanges_section(self):
        """Test exchanges configuration."""
        if not self.config:
            self.skipTest("Config file not loaded")

        self.assertIn('exchanges', self.config)
        exchanges = self.config['exchanges']

        print(f"\n  Configured exchanges:")
        for name, config in exchanges.items():
            enabled = config.get('enabled', False)
            print(f"    {name}: {'Enabled' if enabled else 'Disabled'}")

        # Should have at least Kraken enabled
        self.assertIn('kraken', exchanges)
        self.assertTrue(exchanges['kraken'].get('enabled', False))

    def test_monitoring_section(self):
        """Test monitoring configuration."""
        if not self.config:
            self.skipTest("Config file not loaded")

        self.assertIn('monitoring', self.config)
        monitoring = self.config['monitoring']

        print(f"\n  Monitoring config:")
        print(f"    Enabled: {monitoring.get('enabled', False)}")
        print(f"    Health port: {monitoring.get('health', {}).get('port', 'N/A')}")
        print(f"    Metrics port: {monitoring.get('metrics', {}).get('port', 'N/A')}")

        # Validate ports
        health_port = monitoring.get('health', {}).get('port')
        metrics_port = monitoring.get('metrics', {}).get('port')

        if health_port:
            self.assertGreater(health_port, 1024)
            self.assertLess(health_port, 65536)

        if metrics_port:
            self.assertGreater(metrics_port, 1024)
            self.assertLess(metrics_port, 65536)


class TestDockerComposeConfig(unittest.TestCase):
    """Test docker-compose.live-trading.yml configuration."""

    def setUp(self):
        """Load docker-compose file."""
        compose_path = Path(__file__).parent.parent / 'docker-compose.live-trading.yml'

        if compose_path.exists():
            with open(compose_path, 'r') as f:
                self.compose_config = yaml.safe_load(f)
        else:
            self.compose_config = None

    def test_compose_file_exists(self):
        """Test docker-compose file exists."""
        compose_path = Path(__file__).parent.parent / 'docker-compose.live-trading.yml'
        print(f"\n  Checking: {compose_path}")
        self.assertTrue(compose_path.exists())

    def test_service_configuration(self):
        """Test live-trading service configuration."""
        if not self.compose_config:
            self.skipTest("Docker compose file not loaded")

        self.assertIn('services', self.compose_config)
        services = self.compose_config['services']

        print(f"\n  Services: {list(services.keys())}")
        self.assertIn('live-trading', services)

        service = services['live-trading']
        print(f"    Image: {service.get('image', 'N/A')}")
        print(f"    Restart: {service.get('restart', 'N/A')}")

        self.assertEqual(service.get('image'), 'graphwiz-live-trading:latest')
        self.assertIn('restart', service)

    def test_environment_variables(self):
        """Test required environment variables are defined."""
        if not self.compose_config:
            self.skipTest("Docker compose file not loaded")

        service = self.compose_config['services']['live-trading']
        env_vars = service.get('environment', [])

        print(f"\n  Environment variables:")
        required_vars = ['TRADING_MODE', 'KRAKEN_API_KEY', 'KRAKEN_API_SECRET']

        for var in required_vars:
            found = any(var in str(env) for env in env_vars)
            print(f"    {var}: {'✓' if found else '✗'}")
            self.assertTrue(found, f"Required env var {var} not found")

    def test_volume_mounts(self):
        """Test volume mounts are configured."""
        if not self.compose_config:
            self.skipTest("Docker compose file not loaded")

        service = self.compose_config['services']['live-trading']
        volumes = service.get('volumes', [])

        print(f"\n  Volume mounts:")
        for vol in volumes:
            print(f"    {vol}")

        # Should have logs, config, and data volumes
        volume_str = str(volumes)
        self.assertTrue('logs' in volume_str or '/app/logs' in volume_str)
        self.assertTrue('config' in volume_str or '/app/config' in volume_str)

    def test_health_check(self):
        """Test health check is configured."""
        if not self.compose_config:
            self.skipTest("Docker compose file not loaded")

        service = self.compose_config['services']['live-trading']
        healthcheck = service.get('healthcheck')

        self.assertIsNotNone(healthcheck, "Health check must be configured")

        print(f"\n  Health check:")
        print(f"    Test: {healthcheck.get('test', 'N/A')}")
        print(f"    Interval: {healthcheck.get('interval', 'N/A')}")

        self.assertIn('test', healthcheck)
        self.assertIn('interval', healthcheck)


class TestDockerfileValidation(unittest.TestCase):
    """Test Dockerfile.live-trading configuration."""

    def test_dockerfile_exists(self):
        """Test Dockerfile exists."""
        dockerfile_path = Path(__file__).parent.parent / 'Dockerfile.live-trading'
        print(f"\n  Checking: {dockerfile_path}")
        self.assertTrue(dockerfile_path.exists())

    def test_dockerfile_content(self):
        """Test Dockerfile has required content."""
        dockerfile_path = Path(__file__).parent.parent / 'Dockerfile.live-trading'

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        required_elements = [
            'FROM python:',
            'WORKDIR /app',
            'RUN mkdir -p logs/live_trading',
            'ENV PYTHONPATH=/app/src',
            'ENV TRADING_MODE=live',
            'HEALTHCHECK'
        ]

        print(f"\n  Dockerfile validation:")
        for element in required_elements:
            found = element in content
            print(f"    {element}: {'✓' if found else '✗'}")
            self.assertTrue(found, f"Required element not found: {element}")


def run_tests():
    """Run all configuration validation tests."""
    print("=" * 80)
    print("Configuration Validation Tests")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEnvironmentConfiguration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLiveTradingConfig))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDockerComposeConfig))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDockerfileValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
