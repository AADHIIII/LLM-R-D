# 🚀 Complete Setup Guide

This comprehensive guide will help you set up the LLM R&D Platform from scratch.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Quick Setup (Docker)](#quick-setup-docker)
4. [Development Setup](#development-setup)
5. [Production Deployment](#production-deployment)
6. [Configuration](#configuration)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

## 💻 System Requirements

### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4GB available
- **Storage**: 5GB free space
- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+

### Recommended Requirements
- **CPU**: 4+ cores, 2.5+ GHz
- **RAM**: 8GB+ available
- **Storage**: 20GB+ free space (for models and data)
- **Network**: Stable internet connection for API access

## 🛠️ Prerequisites Installation

### 1. Install Docker & Docker Compose

#### macOS
```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# Verify installation
docker --version
docker-compose --version
```

#### Ubuntu/Linux
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker-compose --version
```

#### Windows
```powershell
# Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop

# Verify installation (in PowerShell)
docker --version
docker-compose --version
```

### 2. Install Git
```bash
# macOS (with Homebrew)
brew install git

# Ubuntu/Linux
sudo apt-get update
sudo apt-get install git

# Windows
# Download from: https://git-scm.com/download/win

# Verify installation
git --version
```

### 3. Install Node.js (for frontend development)
```bash
# macOS (with Homebrew)
brew install node@18

# Ubuntu/Linux
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Windows
# Download from: https://nodejs.org/

# Verify installation
node --version
npm --version
```

### 4. Install Python 3.11+ (for local development)
```bash
# macOS (with Homebrew)
brew install python@3.11

# Ubuntu/Linux
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv python3.11-pip

# Windows
# Download from: https://www.python.org/downloads/

# Verify installation
python3 --version
pip3 --version
```

## 🐳 Quick Setup (Docker)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/LLM-RnD.git
cd LLM-RnD
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment file
nano .env  # or use your preferred editor
```

**Required Environment Variables:**
```bash
# Add your API keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GEMINI_API_KEY=your-gemini-api-key-here

# Database configuration (defaults work for Docker)
DATABASE_URL=postgresql://llm_user:llm_password@llm-platform-db:5432/llm_platform
REDIS_URL=redis://llm-platform-redis:6379/0

# Security keys (generate secure keys for production)
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this
ENCRYPTION_KEY=your-32-character-encryption-key
```

### 3. Start Services
```bash
# Start all services in background
docker-compose up -d

# Check service status
docker-compose ps

# View logs (optional)
docker-compose logs -f
```

### 4. Initialize Database
```bash
# Run database migrations
docker-compose exec llm-platform-backend python scripts/init_db.py

# Create admin user (optional)
docker-compose exec llm-platform-backend python scripts/create_admin.py
```

### 5. Access Platform
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:9000/api/v1/docs
- **Health Check**: http://localhost:9000/api/v1/health

## 🔧 Development Setup

### 1. Backend Development Setup
```bash
# Navigate to project directory
cd LLM-RnD

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python scripts/init_db.py

# Start development server
python run_api.py
```

### 2. Frontend Development Setup
```bash
# Navigate to frontend directory
cd web_interface/frontend

# Install Node.js dependencies
npm install

# Create environment file
echo "REACT_APP_API_URL=http://localhost:5000/api/v1" > .env.local

# Start development server
npm start
```

### 3. Database Setup (Local Development)
```bash
# Install PostgreSQL
# macOS:
brew install postgresql
brew services start postgresql

# Ubuntu/Linux:
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql
CREATE DATABASE llm_platform;
CREATE USER llm_user WITH PASSWORD 'llm_password';
GRANT ALL PRIVILEGES ON DATABASE llm_platform TO llm_user;
\q

# Install Redis
# macOS:
brew install redis
brew services start redis

# Ubuntu/Linux:
sudo apt-get install redis-server
sudo systemctl start redis-server
```

## 🚀 Production Deployment

### 1. Production Environment Setup
```bash
# Clone repository on production server
git clone https://github.com/yourusername/LLM-RnD.git
cd LLM-RnD

# Copy and configure production environment
cp .env.example .env.production

# Edit production configuration
nano .env.production
```

**Production Environment Variables:**
```bash
# Set production mode
FLASK_ENV=production
FLASK_DEBUG=false

# Use strong secrets (generate with: openssl rand -hex 32)
JWT_SECRET_KEY=your-production-jwt-secret
ENCRYPTION_KEY=your-production-encryption-key

# Production database (use managed service recommended)
DATABASE_URL=postgresql://user:pass@prod-db-host:5432/llm_platform

# Security settings
SSL_REQUIRED=true
SECURE_COOKIES=true
SESSION_COOKIE_SECURE=true

# Monitoring
LOG_LEVEL=WARNING
ENABLE_MONITORING=true
```

### 2. Deploy with Docker Compose
```bash
# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d

# Check deployment
docker-compose -f docker-compose.prod.yml ps
```

### 3. Set Up Reverse Proxy (Nginx)
```bash
# Install Nginx
sudo apt-get install nginx

# Copy Nginx configuration
sudo cp nginx/nginx.conf /etc/nginx/sites-available/llm-platform
sudo ln -s /etc/nginx/sites-available/llm-platform /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 4. SSL Certificate Setup
```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal setup
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## ⚙️ Configuration

### 1. API Keys Configuration

#### OpenAI API Key
1. Visit https://platform.openai.com/api-keys
2. Create new API key
3. Add to `.env`: `OPENAI_API_KEY=sk-your-key-here`

#### Anthropic API Key
1. Visit https://console.anthropic.com/
2. Generate API key
3. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-your-key-here`

#### Google Gemini API Key
1. Visit https://makersuite.google.com/app/apikey
2. Create API key
3. Add to `.env`: `GEMINI_API_KEY=your-key-here`

### 2. Database Configuration

#### PostgreSQL Settings
```bash
# Connection settings
DATABASE_URL=postgresql://username:password@host:port/database

# Connection pool settings
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
```

#### Redis Configuration
```bash
# Redis connection
REDIS_URL=redis://host:port/db

# Connection pool
REDIS_CONNECTION_POOL_SIZE=10
```

### 3. Security Configuration

#### JWT Settings
```bash
# JWT configuration
JWT_SECRET_KEY=your-secret-key
JWT_ACCESS_TOKEN_EXPIRES=3600  # 1 hour
JWT_REFRESH_TOKEN_EXPIRES=2592000  # 30 days
```

#### Rate Limiting
```bash
# Rate limiting settings
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_PER_DAY=10000
```

### 4. Model Configuration

#### Default Model Settings
```bash
# Default model configuration
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS_DEFAULT=1000
TEMPERATURE_DEFAULT=0.7
```

#### Cost Tracking
```bash
# Cost management
COST_ALERT_THRESHOLD_USD=100
DAILY_BUDGET_LIMIT_USD=500
ENABLE_COST_ALERTS=true
```

## ✅ Verification

### 1. Health Checks
```bash
# API health check
curl http://localhost:9000/api/v1/health

# Database connectivity
curl http://localhost:9000/api/v1/status

# Redis connectivity
docker-compose exec llm-platform-redis redis-cli ping
```

### 2. Service Status
```bash
# Check all services
docker-compose ps

# Check logs
docker-compose logs llm-platform-backend
docker-compose logs llm-platform-db
docker-compose logs llm-platform-redis
```

### 3. Frontend Verification
```bash
# Check frontend accessibility
curl -I http://localhost:3000

# Check API connectivity from frontend
# Open browser developer tools and check network requests
```

### 4. Authentication Test
```bash
# Register test user
curl -X POST http://localhost:9000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPassword123!",
    "role": "developer"
  }'

# Login test
curl -X POST http://localhost:9000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "TestPassword123!"
  }'
```

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: Docker Services Won't Start
```bash
# Check Docker daemon
docker info

# Check port conflicts
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :9000  # Backend API
lsof -i :3000  # Frontend

# Solution: Stop conflicting services or change ports
```

#### Issue 2: Database Connection Failed
```bash
# Check PostgreSQL container
docker-compose logs llm-platform-db

# Check database credentials in .env
grep DATABASE_URL .env

# Reset database
docker-compose down -v
docker-compose up -d
```

#### Issue 3: API Authentication Errors
```bash
# Check JWT secret configuration
grep JWT_SECRET_KEY .env

# Verify user registration
docker-compose exec llm-platform-backend python scripts/list_users.py

# Reset admin user
docker-compose exec llm-platform-backend python scripts/create_admin.py --reset
```

#### Issue 4: Frontend Build Errors
```bash
# Clear npm cache
cd web_interface/frontend
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node.js version
node --version  # Should be 16+
```

#### Issue 5: API Key Issues
```bash
# Test API keys
curl -H "Authorization: Bearer sk-your-openai-key" \
  https://api.openai.com/v1/models

# Check environment variables
docker-compose exec llm-platform-backend env | grep API_KEY
```

### Performance Issues

#### High Memory Usage
```bash
# Monitor resource usage
docker stats

# Adjust memory limits in docker-compose.yml
# Add under services:
#   mem_limit: 512m
#   memswap_limit: 512m
```

#### Slow API Responses
```bash
# Check database performance
docker-compose exec llm-platform-db psql -U llm_user -d llm_platform -c "
  SELECT query, mean_time, calls 
  FROM pg_stat_statements 
  ORDER BY mean_time DESC 
  LIMIT 10;"

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:9000/api/v1/health
```

### Getting Help

#### Log Collection
```bash
# Collect all logs
docker-compose logs > platform-logs.txt

# Collect system information
docker info > docker-info.txt
docker-compose config > compose-config.txt
```

#### Debug Mode
```bash
# Enable debug mode
echo "FLASK_DEBUG=true" >> .env
echo "LOG_LEVEL=DEBUG" >> .env

# Restart services
docker-compose restart llm-platform-backend
```

## 📞 Support Resources

### Documentation
- [API Documentation](docs/api_documentation.html)
- [User Guide](docs/getting_started.md)
- [Troubleshooting Guide](docs/troubleshooting_guide.md)

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share experiences
- Discord: Real-time community support

### Professional Support
- Email: support@yourcompany.com
- Enterprise Support: Available for production deployments
- Consulting: Custom implementation and optimization services

---

**Setup Complete! 🎉**

Your LLM R&D Platform should now be running and accessible. Continue with the [Testing Guide](TESTING_GUIDE.md) to verify all functionality works correctly.