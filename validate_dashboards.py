#!/usr/bin/env python3
"""Validate dashboard implementations without running them."""

import sys
from pathlib import Path


def validate_terminal_dashboard():
    """Validate terminal dashboard code."""
    print("Validating terminal dashboard...")

    try:
        dashboard_file = Path("monitor_performance.py")
        assert dashboard_file.exists(), "Terminal dashboard file not found"

        content = dashboard_file.read_text()

        # Check for key features
        assert "class MetricsCollector" in content, "MetricsCollector class missing"
        assert "def create_neo4j_panel" in content, "Neo4j panel function missing"
        assert "def create_trading_panel" in content, "Trading panel function missing"
        assert "def create_system_panel" in content, "System panel function missing"
        assert "from rich.console import Console" in content, "Rich import missing"

        print("✅ Terminal dashboard structure validated")
        print("   - Metrics collector: ✓")
        print("   - Neo4j panel: ✓")
        print("   - Trading panel: ✓")
        print("   - System panel: ✓")
        print("   - Rich UI: ✓")
        return True

    except Exception as e:
        print(f"❌ Terminal dashboard validation failed: {e}")
        return False


def validate_web_dashboard():
    """Validate web dashboard code."""
    print("\nValidating web dashboard...")

    try:
        dashboard_file = Path("dashboard_performance.py")
        assert dashboard_file.exists(), "Web dashboard file not found"

        content = dashboard_file.read_text()

        # Check for key features
        assert "import streamlit as st" in content, "Streamlit import missing"
        assert "def render_overview" in content, "Overview page missing"
        assert "def render_neo4j" in content, "Neo4j page missing"
        assert "def render_trading" in content, "Trading page missing"
        assert "def render_system" in content, "System page missing"
        assert "def render_historical" in content, "Historical page missing"
        assert "import plotly" in content, "Plotly import missing"

        print("✅ Web dashboard structure validated")
        print("   - Streamlit framework: ✓")
        print("   - Overview page: ✓")
        print("   - Neo4j performance page: ✓")
        print("   - Trading engine page: ✓")
        print("   - System metrics page: ✓")
        print("   - Historical data page: ✓")
        print("   - Plotly charts: ✓")
        return True

    except Exception as e:
        print(f"❌ Web dashboard validation failed: {e}")
        return False


def validate_documentation():
    """Validate dashboard documentation."""
    print("\nValidating documentation...")

    try:
        guide_file = Path("DASHBOARD_GUIDE.md")
        readme_file = Path("DASHBOARD_README.md")
        requirements_file = Path("requirements-dashboard.txt")

        assert guide_file.exists(), "Dashboard guide missing"
        assert readme_file.exists(), "Dashboard README missing"
        assert requirements_file.exists(), "Dashboard requirements missing"

        guide_content = guide_file.read_text()
        readme_content = readme_file.read_text()

        # Check guide content
        assert "Installation" in guide_content, "Installation section missing"
        assert "Configuration" in guide_content, "Configuration section missing"
        assert "Troubleshooting" in guide_content, "Troubleshooting section missing"

        # Check README content
        assert "Quick Start" in readme_content, "Quick start section missing"
        assert "Requirements" in readme_content, "Requirements section missing"

        print("✅ Documentation validated")
        print(f"   - Guide: {len(guide_content)} characters")
        print(f"   - README: {len(readme_content)} characters")
        print(f"   - Requirements: ✓")
        return True

    except Exception as e:
        print(f"❌ Documentation validation failed: {e}")
        return False


def check_dependencies():
    """Check which dashboard dependencies are installed."""
    print("\nChecking dependencies...")

    dependencies = {
        "streamlit": "Web dashboard (optional)",
        "plotly": "Web dashboard charts (optional)",
        "pandas": "Web dashboard data (optional)",
        "rich": "Terminal dashboard (recommended)"
    }

    installed = {}
    for package, description in dependencies.items():
        try:
            __import__(package)
            installed[package] = True
            print(f"✅ {package:15} - Installed ({description})")
        except ImportError:
            installed[package] = False
            print(f"❌ {package:15} - Not installed ({description})")

    return installed


def print_dashboard_summary():
    """Print summary of dashboard features."""
    print("\n" + "=" * 70)
    print("DASHBOARD FEATURES SUMMARY")
    print("=" * 70)

    print("\n📊 Terminal Dashboard (monitor_performance.py)")
    print("   • Real-time performance monitoring")
    print("   • Neo4j metrics: queries, cache, batch ops")
    print("   • Trading metrics: trades, ticker cache, API reduction")
    print("   • System metrics: CPU, memory, threads")
    print("   • Performance insights and recommendations")
    print("   • Low resource usage (~50MB RAM)")
    print("   • Auto-refresh every second")
    print("   • Requires: rich>=13.6.0")

    print("\n🖥️  Web Dashboard (dashboard_performance.py)")
    print("   • Interactive web interface")
    print("   • 5 pages: Overview, Neo4j, Trading, System, Historical")
    print("   • Real-time charts with Plotly")
    print("   • Configurable auto-refresh")
    print("   • Historical trend analysis")
    print("   • Detailed performance insights")
    print("   • Requires: streamlit, plotly, pandas")

    print("\n📚 Documentation")
    print("   • DASHBOARD_GUIDE.md - Comprehensive guide")
    print("   • DASHBOARD_README.md - Quick reference")
    print("   • requirements-dashboard.txt - Dependencies")

    print("\n" + "=" * 70)


def main():
    """Run all validations."""
    print("=" * 70)
    print("DASHBOARD VALIDATION")
    print("=" * 70)
    print()

    results = {
        "terminal": validate_terminal_dashboard(),
        "web": validate_web_dashboard(),
        "docs": validate_documentation()
    }

    dependencies = check_dependencies()

    print_dashboard_summary()

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print("\n✅ Dashboard Implementations:")
    if results["terminal"]:
        print("   ✓ Terminal dashboard validated")
    else:
        print("   ✗ Terminal dashboard has issues")

    if results["web"]:
        print("   ✓ Web dashboard validated")
    else:
        print("   ✗ Web dashboard has issues")

    if results["docs"]:
        print("   ✓ Documentation validated")

    print("\n📦 Dependency Status:")
    if not any(dependencies.values()):
        print("   ⚠️  No dashboard dependencies installed")
        print("\n   To install:")
        print("   • Terminal dashboard: pip install rich")
        print("   • Web dashboard: pip install -r requirements-dashboard.txt")
    elif dependencies.get("rich"):
        print("   ✅ Ready to run terminal dashboard")
        print("     Run: python3 monitor_performance.py")
    elif dependencies.get("streamlit"):
        print("   ✅ Ready to run web dashboard")
        print("     Run: streamlit run dashboard_performance.py")
    else:
        print("   ⚠️  Some dependencies missing")

    print("\n" + "=" * 70)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
