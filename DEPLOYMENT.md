# Deployment Guide for GraphWiz Trader

This guide provides step-by-step instructions for deploying GraphWiz Trader in various environments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Configuration](#production-configuration)
6. [Security Best Practices](#security-best-practices)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Backup and Recovery](#backup-and-recovery)
9. [Production Checklist](#production-checklist)

---

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **OS**: Linux (Ubuntu 22.04 LTS recommended)
- **Python**: 3.10 or higher
- **Neo4j**: 5.x
- **Docker**: 20.10+ (for containerized deployment)

### Recommended Requirements (Production)

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 200GB SSD
- **Network**: Low-latency connection (<50ms to major exchanges)

---

## Local Development Setup

### 1. Install Dependencies

```bash
# Install Python 3.10+
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Install Neo4j
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j

# Start Neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j
```

### 2. Setup Project

```bash
# Clone repository
git clone https://github.com/tobias-weiss-ai-xr/graphwiz-trader.git
cd graphwiz-trader

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Set Neo4j password
neo4j-admin set-initial-password your_secure_password
```

### 3. Configure Environment

```bash
# Copy example configuration
cp config/paper_trading.yaml config/config.yaml

# Create .env file
cat > .env << EOF
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
OPENAI_API_KEY=your_openai_api_key
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret
EOF

# Load environment variables
source .env
```

### 4. Initialize Knowledge Graph

```bash
# Run initialization script
python -m graphwiz_trader.knowledge_graph.init_graph
```

---

## Docker Deployment

### 1. Using Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f graphwiz-trader

# Stop services
docker-compose down
```

### 2. Custom Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ /app/src/
COPY config/ /app/config/

# Set environment variables
ENV PYTHONPATH=/app/src

# Run application
CMD ["python", "-m", "graphwiz_trader.main"]
```

Build and run:

```bash
# Build image
docker build -t graphwiz-trader:latest .

# Run container
docker run -d \
  --name graphwiz-trader \
  --env-file .env \
  -p 8050:8050 \
  -v $(pwd)/logs:/app/logs \
  graphwiz-trader:latest
```

---

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup

```bash
# Launch EC2 instance (Ubuntu 22.04)
# Instance type: t3.large (2 vCPU, 8GB RAM minimum)
# Storage: 100GB GP3 SSD

# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone https://github.com/tobias-weiss-ai-xr/graphwiz-trader.git
cd graphwiz-trader

# Configure environment
cp .env.example .env
nano .env  # Edit with your credentials

# Start services
docker-compose up -d
```

#### 2. RDS for Neo4j (Optional)

```bash
# Use AWS Neptune or deploy Neo4j on EC2
# For Neo4j on EC2:
# - Use m5.large instance
# - Attach 200GB EBS volume
# - Configure security group for port 7687
```

### Google Cloud Platform Deployment

```bash
# Create Compute Engine instance
gcloud compute instances create graphwiz-trader \
  --zone=us-central1-a \
  --machine-type=e2-highmem-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd

# SSH into instance
gcloud compute ssh graphwiz-trader --zone=us-central1-a

# Follow Docker deployment steps above
```

### Azure Deployment

```bash
# Create Azure VM
az vm create \
  --resource-group GraphWizTrader \
  --name graphwiz-trader-vm \
  --image Ubuntu2204 \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/id_rsa.pub

# Open ports
az vm open-port \
  --resource-group GraphWizTrader \
  --name graphwiz-trader-vm \
  --port 8050

# SSH and deploy
ssh azureuser@<public-ip>
# Follow Docker deployment steps
```

---

## Production Configuration

### 1. Security Hardening

```yaml
# config/production.yaml
trading:
  execution_type: "real"
  require_confirmation: false

  # Production rate limits
  rate_limit:
    requests_per_second: 10
    orders_per_second: 5

# Security settings
security:
  enable_encryption: true
  api_key_rotation_days: 90
  audit_logging: true
  ip_whitelist:
    - your_ip_address

# Logging
logging:
  level: "INFO"
  file: "/var/log/graphwiz-trader/production.log"
  retention: "365 days"
  audit_trail: true
```

### 2. Performance Tuning

```yaml
# Neo4j configuration
neo4j:
  connection_pool_size: 100
  max_connection_lifetime: 3600
  acquire_timeout: 60

  # Caching
  cache_ttl: 600
  enable_query_cache: true

# Agent configuration
agents:
  timeout: 30
  max_concurrent: 10
  retry_attempts: 3
```

---

## Security Best Practices

### 1. API Key Management

```bash
# Use environment variables, never hardcode keys
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

# Use secrets management in production
# AWS Secrets Manager, AWS Parameter Store, GCP Secret Manager
```

### 2. Firewall Configuration

```bash
# UFW rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8050/tcp  # Dashboard
sudo ufw allow from your_ip to any port 7687  # Neo4j
sudo ufw enable
```

### 3. SSL/TLS Configuration

```bash
# Use HTTPS for dashboard
# Generate SSL certificate
certbot --nginx -d your-domain.com

# Configure Neo4j with SSL
# Edit neo4j.conf
dbms.connector.bolt.tls_level=REQUIRED
```

---

## Monitoring and Logging

### 1. Application Monitoring

```yaml
# monitoring_config.yaml
monitoring:
  enabled: true

  # Prometheus metrics
  prometheus:
    enabled: true
    port: 9090

  # Grafana dashboard
  grafana:
    enabled: true
    dashboards:
      - trading_performance
      - risk_metrics
      - system_health
```

### 2. Log Aggregation

```bash
# Setup centralized logging (ELK Stack)
# Filebeat → Logstash → Elasticsearch → Kibana

# Or use cloud services
# AWS CloudWatch
# GCP Cloud Logging
# Azure Monitor
```

---

## Backup and Recovery

### 1. Neo4j Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/neo4j"
DATE=$(date +%Y%m%d_%H%M%S)

neo4j-admin backup \
  --from=bolt://localhost:7687 \
  --backup-dir=$BACKUP_DIR \
  --backup-name=backup_$DATE

# Keep last 7 days
find $BACKUP_DIR -mtime +7 -delete
```

### 2. Configuration Backup

```bash
# Backup configurations
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/

# Backup to cloud storage
aws s3 cp config_backup_*.tar.gz s3://your-backup-bucket/
```

---

## Production Checklist

### Pre-Deployment

- [ ] System requirements met
- [ ] All dependencies installed
- [ ] Neo4j configured and tested
- [ ] API keys obtained and tested
- [ ] Configuration files reviewed
- [ ] Security settings configured
- [ ] Firewall rules applied
- [ ] SSL/TLS certificates installed
- [ ] Monitoring configured
- [ ] Backup strategy in place
- [ ] Recovery procedures tested

### Testing

- [ ] Paper trading mode tested
- [ ] Unit tests passing (80%+ coverage)
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Risk management verified
- [ ] Failover tested
- [ ] Load testing completed

### Go-Live

- [ ] Start with conservative configuration
- [ ] Use paper trading first (1-2 weeks)
- [ ] Gradually increase position sizes
- [ ] Monitor closely for first month
- [ ] Review and adjust risk parameters
- [ ] Document all incidents
- [ ] Regular performance reviews

### Ongoing Operations

- [ ] Daily log review
- [ ] Weekly performance reports
- [ ] Monthly strategy reviews
- [ ] Quarterly security audits
- [ ] Regular backup verification
- [ ] Update dependencies regularly
- [ ] Monitor exchange API changes
- [ ] Track market conditions

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common deployment issues.

**Quick Checks:**

```bash
# Check Neo4j status
systemctl status neo4j
curl http://localhost:7474

# Check application logs
tail -f logs/production.log

# Check Docker containers
docker-compose ps
docker-compose logs

# Test exchange connectivity
python -m graphwiz_trader.utils.test_exchange_connection
```

---

## Support

For deployment issues:
- GitHub Issues: [graphwiz-trader/issues](https://github.com/tobias-weiss-ai-xr/graphwiz-trader/issues)
- Documentation: [README.md](README.md)

---

**Remember: Always test thoroughly in paper trading mode before live trading with real funds.**
