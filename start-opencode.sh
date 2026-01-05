#!/bin/bash
# Start opencode from graphwiz-trader src directory to avoid venv scanning issues

cd "$(dirname "$0")/src" && opencode "$@"