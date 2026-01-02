# GraphWiz Trader Deployment Guide

This guide provides comprehensive instructions for deploying GraphWiz Trader in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Deployment Methods](#deployment-methods)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Maintenance](#maintenance)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 50GB, Recommended 100GB+
- **CPU**: 4+ cores recommended

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Nginx 1.25+ (for reverse proxy)
- Git

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/graphwiz-trader.git /opt/git/graphwiz-trader
cd /opt/git/graphwiz-trader
```

### 2. Run System Setup

```bash
sudo bash deploy/setup_user.sh
```

### 3. Configure Environment

```bash
cp .env.example .env
nano .env  # Edit with your configuration
```

### 4. Deploy

```bash
bash deploy/deploy.sh
```

## Deployment Methods

### Method 1: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Start with looper agent
docker-compose --profile looper up -d

# Start with nginx proxy
docker-compose --profile proxy up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Method 2: Systemd Service

```bash
# Install service
sudo systemctl enable graphwiz-trader.service

# Start service
sudo systemctl start graphwiz-trader.service

# Check status
sudo systemctl status graphwiz-trader.service

# View logs
sudo journalctl -u graphwiz-trader.service -f
```

### Method 3: Automated Deployment Script

```bash
# Full deployment with validation
bash deploy/deploy.sh

# Skip tests
bash deploy/deploy.sh --skip-tests

# Skip Docker build
bash deploy/deploy.sh --skip-build

# Rollback to previous deployment
bash deploy/deploy.sh --rollback
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Neo4j Configuration
NEO4J_PASSWORD=your_secure_password_here

# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Exchange API Keys
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret

# Security
JWT_SECRET=your_jwt_secret_here

# Application
LOG_LEVEL=INFO
TRADING_MODE=paper  # Options: paper, live, simulation
```

### Production Configuration

The main configuration is in `config/production.yaml`. Key settings:

- **Trading Mode**: Start with `paper` for testing
- **Risk Limits**: Adjust max daily loss and drawdown limits
- **API Keys**: Configure exchange API credentials
- **Neo4j**: Ensure password is set correctly

## Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8080/health

# Docker health status
docker-compose ps

# Individual service health
docker inspect graphwiz-trader --format='{{.State.Health.Status}}'
```

### Metrics

Access Prometheus metrics:

```bash
curl http://localhost:9090/metrics
```

### Grafana Dashboards

1. Access Grafana: `http://localhost:3000`
2. Default credentials (change these!)
   - User: `admin`
   - Password: `graphwiz_grafana`
3. Import dashboards from `dashboards/` directory

### Log Locations

- Application: `/opt/git/graphwiz-trader/logs/graphwiz-trader.log`
- Errors: `/opt/git/graphwiz-trader/logs/errors.log`
- Neo4j: `/opt/git/graphwiz-trader/logs/neo4j/`
- Nginx: `/opt/git/graphwiz-trader/logs/nginx/`

## Maintenance

### Backup

```bash
# Manual backup
bash deploy/deploy.sh --backup

# Automated daily backups (add to crontab)
0 2 * * * /opt/git/graphwiz-trader/deploy/deploy.sh --backup
```

### Updates

```bash
# Pull latest code
git pull origin main

# Redeploy
bash deploy/deploy.sh
```

### Resource Management

```bash
# Check resource usage
docker stats

# Clean up old images
docker image prune -a

# Clean up old volumes (careful!)
docker volume prune
```

### Database Maintenance

```bash
# Access Neo4j shell
docker exec -it graphwiz-neo4j cypher-shell -u neo4j -p your_password

# Backup Neo4j database
docker exec graphwiz-neo4j neo4j-admin database backup \
    --from-path=/data \
    --to-path=/backups/neo4j
```

## Troubleshooting

### Common Issues

#### Services Won't Start

```bash
# Check logs
docker-compose logs -f

# Check port conflicts
netstat -tulpn | grep LISTEN

# Check disk space
df -h
```

#### Neo4j Connection Issues

```bash
# Verify Neo4j is running
docker-compose ps neo4j

# Check Neo4j logs
docker-compose logs neo4j

# Test connection
cypher-shell -u neo4j -p your_password "RETURN 1"
```

#### Memory Issues

```bash
# Check memory usage
free -h
docker stats --no-stream

# Adjust limits in docker-compose.yml
# Change: memory: 8G
```

#### Permission Issues

```bash
# Fix permissions
sudo chown -R graphwiz:graphwiz /opt/git/graphwiz-trader
sudo chmod -R 750 /opt/git/graphwiz-trader/data
sudo chmod -R 750 /opt/git/graphwiz-trader/logs
```

### Recovery Procedures

#### Rollback Deployment

```bash
bash deploy/deploy.sh --rollback
```

#### Restore from Backup

```bash
# Stop services
docker-compose down

# Restore data
cp -r backups/deployments/backup-YYYYMMDD-HHMMSS/data/* /opt/git/graphwiz-trader/data/

# Restart services
docker-compose up -d
```

## Security Best Practices

1. **Change Default Passwords**
   - Neo4j: Update `NEO4J_PASSWORD` in `.env`
   - Grafana: Update admin password on first login

2. **SSL/TLS**
   - Use reverse proxy (Nginx) with SSL
   - Generate strong SSL certificates
   - Enable HTTPS only

3. **Firewall**
   ```bash
   # Allow only necessary ports
   ufw allow 80/tcp    # HTTP
   ufw allow 443/tcp   # HTTPS
   ufw allow 22/tcp    # SSH
   ufw enable
   ```

4. **API Security**
   - Never commit API keys to git
   - Use environment variables for secrets
   - Rotate credentials regularly

5. **Network Isolation**
   - Use Docker networks
   - Don't expose services publicly
   - Use internal networks for inter-service communication

## Performance Tuning

### Neo4j Optimization

```yaml
# In docker-compose.yml
NEO4J_dbms_memory_heap_initial__size=1g
NEO4J_dbms_memory_heap_max__size=4g
NEO4J_dbms_memory_pagecache_size=2g
```

### Application Performance

```yaml
# In config/production.yaml
performance:
  thread_pool_size: 16  # Adjust based on CPU cores
  async_io: true
  enable_profiling: false
```

### Docker Optimization

```bash
# Build with optimizations
docker build --build-arg BUILDKIT_INLINE_CACHE=1 .

# Use multi-stage builds (already in Dockerfile)
# Minimize layers (already optimized)
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/graphwiz-trader/issues
- Documentation: https://docs.graphwiz.ai
- Email: support@graphwiz.ai
