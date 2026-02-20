# Production Deployment Guide

Deploy SmarterRouter in production with proper security, monitoring, and reliability.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Docker Compose Production Setup](#docker-compose-production-setup)
- [Security Hardening](#security-hardening)
- [Monitoring & Alerting](#monitoring--alerting)
- [SSL/TLS Termination](#ssltls-termination)
- [Backup Strategy](#backup-strategy)
- [Scaling Considerations](#scaling-considerations)

## Prerequisites

- Docker and Docker Compose
- SSL certificate (for production HTTPS)
- Reverse proxy (nginx, Traefik, Caddy) - optional but recommended
- Monitoring infrastructure (Prometheus, Grafana) - optional but recommended

---

## Docker Compose Production Setup

### Basic Production Configuration

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  smarterrouter:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: smarterrouter
    ports:
      - "11436:11436"
    env_file:
      - .env
    volumes:
      - ./router.db:/app/router.db:ro  # read-only mount
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    read_only: true  # immutable filesystem
    security_opt:
      - no-new-privileges:true
    networks:
      - smarterrouter-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11436/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 2G

networks:
  smarterrouter-network:
    driver: bridge
```

### Start in Production

```bash
# Build production image
docker-compose -f docker-compose.prod.yml build

# Start service
docker-compose -f docker-compose.prod.yml up -d

# Check health
docker-compose -f docker-compose.prod.yml ps
docker logs smarterrouter
```

---

## Security Hardening

### 1. Set Admin API Key (REQUIRED)

```bash
# Generate secure random key
openssl rand -hex 32

# Add to .env
ROUTER_ADMIN_API_KEY=sk-smarterrouter-<your-random-key>
```

**Never leave admin endpoints unprotected in production!**

### 2. Enable Rate Limiting

```env
ROUTER_RATE_LIMIT_ENABLED=true
ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE=120
ROUTER_RATE_LIMIT_ADMIN_REQUESTS_PER_MINUTE=10
```

### 3. Restrict CORS

```env
# Only allow your frontend origins
ROUTER_CORS_ALLOWED_ORIGINS=https://your-app.com,https://admin.your-app.com
```

### 4. Use Non-Root User

The Dockerfile already creates a non-root user `smarterrouter`. Ensure it's being used:

```yaml
# In docker-compose.prod.yml
services:
  smarterrouter:
    user: "1000:1000"  # smarterrouter user
```

### 5. Network Isolation

Use internal Docker network; expose port only to trusted network:

```yaml
networks:
  smarterrouter-network:
    internal: true  # no external internet access
```

Or use firewall rules to restrict port 11436 to specific IPs.

### 6. Regular Security Updates

```bash
# Pull latest security patches
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

Monitor security advisories:
- Subscribe to GitHub Security Advisories for this repo
- Watch for Python/Docker/NVIDIA security updates

---

## Monitoring & Alerting

### 1. Prometheus Metrics

SmarterRouter exposes metrics at `GET /metrics`.

Add to Prometheus config:

```yaml
scrape_configs:
  - job_name: 'smarterrouter'
    static_configs:
      - targets: ['localhost:11436']
    scrape_interval: 30s
```

### 2. Key Alerts

Create alerts for:

- **High error rate:** `rate(smarterrouter_errors_total[5m]) > 0.1`
- **High latency:** `histogram_quantile(0.95, rate(smarterrouter_request_duration_seconds_bucket[5m])) > 10`
- **VRAM pressure:** `smarterrouter_vram_utilization_pct > 90`
- **Low cache hit rate:** `rate(smarterrouter_cache_hits_total[5m]) / (rate(smarterrouter_cache_hits_total[5m]) + rate(smarterrouter_cache_misses_total[5m])) < 0.5`
- **Service down:** `up{job="smarterrouter"} == 0`

### 3. Grafana Dashboard

Import dashboard (example JSON to be provided). Key panels:
- Request rate and latency
- Error rate by endpoint
- Model selection distribution
- VRAM usage over time
- Cache hit rates
- Backend connectivity status

---

## SSL/TLS Termination

### Option A: Reverse Proxy (Recommended)

Use nginx/Traefik/Caddy to handle HTTPS:

```nginx
# nginx config
server {
    listen 443 ssl http2;
    server_name router.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:11436;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Option B: Let's Encrypt with Caddy

Caddy auto-configures SSL:

```caddyfile
router.your-domain.com {
    reverse_proxy localhost:11436
}
```

### Option C: TLS in Docker

Mount certificates in container:

```yaml
services:
  smarterrouter:
    volumes:
      - ./certs:/app/certs:ro
    environment:
      - SSL_CERT_PATH=/app/certs/fullchain.pem
      - SSL_KEY_PATH=/app/certs/privkey.pem
```

*(Note: SmarterRouter doesn't have built-in TLS; use reverse proxy approach)*

---

## Backup Strategy

### Critical Data to Back Up

1. **Database (`router.db` or PostgreSQL):** Contains all model profiles and routing history
2. **Configuration (`.env`):** Your settings (redact secrets before storing)
3. **Logs:** For debugging and audit trail (optional, can be large)

### Automated Backups (Daily)

Create `backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR=/backups/smarterrouter
DATE=$(date +%Y%m%d-%H%M%S)

# Backup database
cp /path/to/router.db $BACKUP_DIR/router-$DATE.db

# Backup .env (redact API keys first!)
sed 's/ROUTER_ADMIN_API_KEY=.*/ROUTER_ADMIN_API_KEY=REDACTED/' .env > $BACKUP_DIR/env-$DATE.txt

# Optional: compress old backups
find $BACKUP_DIR -name "*.db" -mtime +30 -exec gzip {} \;
find $BACKUP_DIR -name "*.txt" -mtime +7 -delete

# Optional: upload to S3
# aws s3 cp $BACKUP_DIR/router-$DATE.db s3://your-bucket/backups/
```

Add to cron:

```bash
0 2 * * * /path/to/backup.sh
```

### Restore from Backup

```bash
# Stop SmarterRouter
docker-compose down

# Restore database
cp /backups/router-20240220.db router.db

# Restore config (manual edit)
cp /backups/env-20240220.txt .env
# Edit .env to add back your actual secrets

# Restart
docker-compose up -d
```

---

## Scaling Considerations

### Multiple Router Instances

For high availability and load distribution:

```yaml
# docker-compose.yml
services:
  smarterrouter-1:
    # ... same config
    ports:
      - "11436:11436"
  
  smarterrouter-2:
    # ... same config
    ports:
      - "11437:11436"
```

Use a load balancer (nginx, HAProxy) in front:

```nginx
upstream smarterrouter {
    server localhost:11436;
    server localhost:11437;
}

server {
    listen 80;
    location / {
        proxy_pass http://smarterrouter;
    }
}
```

### Shared Database

All instances must share the same database:

```yaml
services:
  smarterrouter-1:
    volumes:
      - postgres_data:/app/data  # Use PostgreSQL
  
  smarterrouter-2:
    volumes:
      - postgres_data:/app/data

volumes:
  postgres_data:
```

Or use external PostgreSQL:

```env
ROUTER_DATABASE_URL=postgresql://user:pass@postgres-host:5432/smarterrouter
```

**Warning:** SQLite doesn't work well with multiple writers. Use PostgreSQL for multi-instance deployments.

---

## Disaster Recovery

### High Availability Setup

1. **Database:** Use managed PostgreSQL (RDS, CloudSQL) with replication
2. **Multiple router instances:** At least 2 in different availability zones
3. **Load balancer health checks:** Route traffic only to healthy instances
4. **Regular backups:** Automated, tested restore process
5. **Monitoring alerts:** Immediate notification of failures

### Recovery Procedures

Document step-by-step:
1. How to manually failover to backup instance
2. How to restore database from backup
3. How to rebuild router instance from scratch
4. Contact information for critical incidents

---

## Performance Tuning for Production

See [Performance Tuning](../performance.md) for detailed guidance.

**Production recommendations:**
- Set `ROUTER_CACHE_ENABLED=true` with appropriate size (1000-2000)
- Pin a small model: `ROUTER_PINNED_MODEL=phi3:mini`
- Use PostgreSQL instead of SQLite for concurrent access
- Enable `ROUTER_VRAM_AUTO_UNLOAD_ENABLED=true`
- Set appropriate `ROUTER_VRAM_MAX_TOTAL_GB` (leave 10-15% headroom)
- Use `ROUTER_LOG_FORMAT=json` for log aggregation
- Set `ROUTER_LOG_LEVEL=WARNING` (avoid INFO logs in production)

---

## Maintenance

### Health Checks

Monitor these endpoints:
- `GET /health` - Overall health (profiling complete, backend connected)
- `GET /metrics` - Prometheus metrics
- `GET /admin/vram` - VRAM status

### Log Rotation

Configure log rotation to prevent disk fill:

```bash
# /etc/logrotate.d/smarterrouter
/path/to/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 smarterrouter smarterrouter
    postrotate
        docker exec smarterrouter kill -USR1 1
    endscript
}
```

### Database Maintenance

For PostgreSQL:
- Regular `VACUUM ANALYZE`
- Monitor table size
- Set up point-in-time recovery

For SQLite:
- Periodically run `VACUUM` during maintenance windows
- Backup before large schema changes
- Monitor file size growth

---

## Next Steps

- [Configuration Reference](../configuration.md) - All available settings
- [Troubleshooting](../troubleshooting.md) - Production issues and solutions
- [Performance Tuning](../performance.md) - Optimize for your workload
- [API Documentation](../api.md) - Complete API reference
