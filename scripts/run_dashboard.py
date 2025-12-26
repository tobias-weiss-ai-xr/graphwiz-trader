#!/usr/bin/env python3
"""
Paper trading dashboard launcher.

Starts the Streamlit dashboard for visualizing paper trading results.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    # Add src to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    # Dashboard module path
    dashboard_app = project_root / "src" / "graphwiz_trader" / "paper_trading" / "dashboard.py"

    if not dashboard_app.exists():
        print(f"Error: Dashboard app not found at {dashboard_app}")
        sys.exit(1)

    # Launch streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_app),
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "light",
    ]

    print("📈 Starting GraphWiz Paper Trading Dashboard...")
    print(f"   Dashboard: {dashboard_app}")
    print(f"   URL: http://localhost:8501")
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
