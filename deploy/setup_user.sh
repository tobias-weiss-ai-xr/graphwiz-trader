#!/bin/bash
##############################################################################
# GraphWiz Trader System Setup Script
# Creates the system user and sets up directories with proper permissions
##############################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="/opt/git/graphwiz-trader"
APP_USER="graphwiz"
APP_GROUP="graphwiz"

log_info() { echo -e "${GREEN}[INFO]${NC} $@"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $@"; }
log_error() { echo -e "${RED}[ERROR]${NC} $@"; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root"
   exit 1
fi

log_info "Setting up GraphWiz Trader system..."

# Create group if it doesn't exist
if ! getent group "${APP_GROUP}" >/dev/null 2>&1; then
    log_info "Creating group: ${APP_GROUP}"
    groupadd -r "${APP_GROUP}"
else
    log_info "Group ${APP_GROUP} already exists"
fi

# Create user if it doesn't exist
if ! id -u "${APP_USER}" >/dev/null 2>&1; then
    log_info "Creating user: ${APP_USER}"
    useradd -r -g "${APP_GROUP}" -d "${PROJECT_ROOT}" -s /bin/bash \
        -c "GraphWiz Trader Application" "${APP_USER}"
else
    log_info "User ${APP_USER} already exists"
fi

# Create necessary directories
log_info "Creating directories..."
mkdir -p "${PROJECT_ROOT}"/{data,logs,backtests,config}
mkdir -p "${PROJECT_ROOT}/data"/{neo4j,prometheus,grafana}
mkdir -p "${PROJECT_ROOT}/logs"/{neo4j,nginx}
mkdir -p "${PROJECT_ROOT}/backtests"

# Set ownership
log_info "Setting ownership and permissions..."
chown -R "${APP_USER}:${APP_GROUP}" "${PROJECT_ROOT}"
chmod 755 "${PROJECT_ROOT}"
chmod 750 "${PROJECT_ROOT}"/{data,logs,backtests}
chmod 750 "${PROJECT_ROOT}/data"/*

# Set permissions for config (more restrictive)
chmod 750 "${PROJECT_ROOT}/config"

# Add user to docker group (for docker-compose management)
if getent group docker >/dev/null 2>&1; then
    log_info "Adding ${APP_USER} to docker group..."
    usermod -aG docker "${APP_USER}"
fi

# Setup sudoers file for systemctl access (optional)
log_info "Setting up sudoers..."
cat > /etc/sudoers.d/graphwiz-trader <<EOF
# Allow graphwiz user to manage the service
${APP_USER} ALL=(ALL) NOPASSWD: /bin/systemctl start graphwiz-trader.service
${APP_USER} ALL=(ALL) NOPASSWD: /bin/systemctl stop graphwiz-trader.service
${APP_USER} ALL=(ALL) NOPASSWD: /bin/systemctl restart graphwiz-trader.service
${APP_USER} ALL=(ALL) NOPASSWD: /bin/systemctl status graphwiz-trader.service
EOF

chmod 440 /etc/sudoers.d/graphwiz-trader

# Create systemd service symlink
log_info "Installing systemd service..."
cp "${PROJECT_ROOT}/deploy/graphwiz-trader.service" /etc/systemd/system/
systemctl daemon-reload

log_info "System setup completed successfully!"
log_info "User: ${APP_USER}"
log_info "Group: ${APP_GROUP}"
log_info "Project root: ${PROJECT_ROOT}"
