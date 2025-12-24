"""Setup configuration for GraphWiz Trader."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="graphwiz-trader",
    version="0.1.0",
    author="Tobias Weiss",
    author_email="tobias.weiss@example.com",
    description="Automated trading system powered by knowledge graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tobias-weiss-ai-xr/graphwiz-trader",
    project_urls={
        "Bug Tracker": "https://github.com/tobias-weiss-ai-xr/graphwiz-trader/issues",
        "Documentation": "https://github.com/tobias-weiss-ai-xr/graphwiz-trader/blob/main/README.md",
        "Source Code": "https://github.com/tobias-weiss-ai-xr/graphwiz-trader",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "neo4j>=5.15.0",
        "langchain>=0.1.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "ccxt>=4.1.0",
        "aiohttp>=3.9.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "ai": [
            "langchain-openai>=0.0.2",
            "openai>=1.10.0",
            "anthropic>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "graphwiz-trader=graphwiz_trader.main:main",
        ],
    },
)
