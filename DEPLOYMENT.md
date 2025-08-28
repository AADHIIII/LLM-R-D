# LLM Optimization Platform - Deployment Guide

This guide provides comprehensive instructions for deploying the LLM Optimization Platform using Docker containers in both development and production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Development Deployment](#development-deployment)
4. [Production Deployment](#production-deployment)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Backup and Recovery](#backup-and-recovery)
8. [Troubleshooting](#troubleshooting)
9. [Scaling](#scaling)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Memory**: Minimum 8GB RAM (16GB+ recommended for production)
- **Storage**: Minimum 50GB free space (100GB+ recommended for production)
- **CPU**: 4+ cores recommended

### Software Requirements

- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later
- **Git**: For cloning the repository
- **curl**: For health checks and testing

### Installation

#### Docker Installation

**Ubuntu/Debian:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS:**
```bash
# Install Docker Desktop from https://docker.com/products/docker-desktop
# Or using Homebrew:
brew install --cask docker
```

**Windows:**
Install Docker Desktop from https://docker.com/products/docker-desktop

#### Docker Compose Installation

Docker Compose is included with Docker Desktop. For Linux:
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd llm-optimization-platform
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Deploy development environment:**
   ```bash
   ./scripts/deploy.sh --environment development
   ```

4. **Access the application:**
   - Backend API: http://localhost:5000
   - Frontend: http://localhost:3000
   - Streamlit: http://localhost:8501

## Development Deployment

### Configuration

1. **Environment Setup:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file:**
   ```bash
   # Required settings
   POSTGRES_PASSWORD=your_secure_password
   REDIS_PASSWORD=your_redis_password
   SECRET_KEY=your_secret_key
   
   # API Keys (optional for development)
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```

### Deployment

1. **Using deployment script (recommended):**
   ```bash
   ./scripts/deploy.sh --environment development
   ```

2. **Manual deployment:**
   ```bash
   # Build and start services
   docker-compose up -d --build
   
   # Check service status
   docker-compose ps
   
   # View logs
   docker-compose logs -f
   ```

### Verification

1. **Health checks:**
   ```bash
   curl http://localhost:5000/api/v1/health
   curl http://localhost:3000
   ```

2. **Database connection:**
   ```bash
   docker-compose exec database psql -U llm_user -d llm_platform -c "SELECT version();"
   ```

## Production Deployment

### Security Considerations

1. **Environment Variables:**
   - Use strong, unique passwords
   - Generate secure secret keys
   - Restrict API key access
   - Configure proper CORS origins

2. **Network Security:**
   - Use HTTPS with valid SSL certificates
   - Configure firewall rules
   - Implement rate limiting
   - Use private networks for internal communication

### SSL/TLS Configuration

1. **Obtain SSL certificates:**
   ```bash
   # Using Let's Encrypt (recommended)
   sudo apt install certbot
   sudo certbot certonly --standalone -d your-domain.com
   ```

2. **Configure SSL in nginx:**
   ```bash
   # Copy certificates to nginx/ssl/
   sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
   sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
   ```

### Production Deployment

1. **Environment setup:**
   ```bash
   cp .env.example .env
   # Configure production values
   ```

2. **Deploy production environment:**
   ```bash
   ./scripts/deploy.sh --environment production
   ```

3. **Alternative manual deployment:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d --build
   ```

### Load Balancing

For high availability, configure multiple backend instances:

```yaml
# In docker-compose.prod.yml
services:
  backend:
    deploy:
      replicas: 3
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FLASK_ENV` | Application environment | `production` | Yes |
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `REDIS_URL` | Redis connection string | - | Yes |
| `SECRET_KEY` | Flask secret key | - | Yes |
| `OPENAI_API_KEY` | OpenAI API key | - | No |
| `ANTHROPIC_API_KEY` | Anthropic API key | - | No |
| `CORS_ORIGINS` | Allowed CORS origins | - | Yes |

### Database Configuration

The platform uses PostgreSQL with the following default settings:
- Database: `llm_platform`
- User: `llm_user`
- Port: `5432`

### Redis Configuration

Redis is used for caching and session storage:
- Port: `6379`
- Password protected
- Persistence enabled

## Monitoring

### Prometheus Metrics

The platform exposes metrics at `/api/v1/metrics` including:
- Request count and duration
- Database connection pool status
- Model inference latency
- Cost tracking metrics

### Grafana Dashboards

Access Grafana at http://localhost:3001 (production) with:
- Username: `admin`
- Password: Set via `GRAFANA_PASSWORD` environment variable

### Health Checks

All services include health checks:
- Backend: `/api/v1/health`
- Frontend: `/health`
- Database: `pg_isready`
- Redis: `redis-cli ping`

### Log Aggregation

Logs are collected in the `./logs` directory:
- `api.log`: Backend API logs
- `app.log`: Application logs
- `nginx/`: Nginx access and error logs

## Backup and Recovery

### Automated Backups

Use the backup script:
```bash
./scripts/backup.sh
```

Options:
- `--skip-models`: Skip model files backup
- `--skip-logs`: Skip log files backup
- `--no-cleanup`: Don't remove old backups

### Manual Database Backup

```bash
docker-compose exec database pg_dump -U llm_user llm_platform > backup.sql
```

### Restore from Backup

```bash
docker-compose exec -T database psql -U llm_user llm_platform < backup.sql
```

### Model and Data Backup

Models and datasets are stored in persistent volumes:
- `./models/`: Fine-tuned models
- `./datasets/`: Training datasets
- `./logs/`: Application logs

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check port usage
   sudo netstat -tlnp | grep :5000
   
   # Change ports in .env file
   API_PORT=5001
   ```

2. **Memory issues:**
   ```bash
   # Increase Docker memory limit
   # Or reduce batch sizes in configuration
   ```

3. **Database connection errors:**
   ```bash
   # Check database logs
   docker-compose logs database
   
   # Verify connection
   docker-compose exec database psql -U llm_user -d llm_platform
   ```

4. **SSL certificate issues:**
   ```bash
   # Verify certificate files
   openssl x509 -in nginx/ssl/cert.pem -text -noout
   
   # Check nginx configuration
   docker-compose exec nginx nginx -t
   ```

### Log Analysis

```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f backend

# Check system resources
docker stats

# Inspect container
docker-compose exec backend bash
```

### Performance Tuning

1. **Database optimization:**
   - Adjust `shared_buffers` and `work_mem`
   - Monitor query performance
   - Add indexes for frequent queries

2. **Redis optimization:**
   - Configure memory limits
   - Enable persistence if needed
   - Monitor memory usage

3. **Application optimization:**
   - Adjust worker processes
   - Configure connection pooling
   - Monitor memory usage

## Scaling

### Horizontal Scaling

1. **Backend scaling:**
   ```yaml
   services:
     backend:
       deploy:
         replicas: 3
   ```

2. **Database scaling:**
   - Use read replicas for read-heavy workloads
   - Consider database sharding for large datasets

3. **Load balancing:**
   - Configure nginx upstream servers
   - Use external load balancers (AWS ALB, etc.)

### Vertical Scaling

1. **Resource limits:**
   ```yaml
   services:
     backend:
       deploy:
         resources:
           limits:
             memory: 8G
             cpus: '4.0'
   ```

2. **Database resources:**
   - Increase memory allocation
   - Use faster storage (SSD)
   - Optimize configuration

### Container Orchestration

For production at scale, consider:
- **Kubernetes**: For advanced orchestration
- **Docker Swarm**: For simpler clustering
- **Cloud services**: AWS ECS, Google Cloud Run, etc.

## Security Best Practices

1. **Container security:**
   - Use non-root users
   - Scan images for vulnerabilities
   - Keep base images updated

2. **Network security:**
   - Use private networks
   - Implement proper firewall rules
   - Enable SSL/TLS everywhere

3. **Data security:**
   - Encrypt data at rest
   - Use secure API keys
   - Implement proper authentication

4. **Monitoring:**
   - Monitor for suspicious activity
   - Set up alerting
   - Regular security audits

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Check Docker and system resources
4. Consult the project documentation

## Updates and Maintenance

1. **Regular updates:**
   ```bash
   # Pull latest changes
   git pull origin main
   
   # Rebuild and deploy
   ./scripts/deploy.sh --environment production
   ```

2. **Database migrations:**
   ```bash
   # Run migrations (if applicable)
   docker-compose exec backend python manage.py migrate
   ```

3. **Backup before updates:**
   ```bash
   ./scripts/backup.sh
   ```