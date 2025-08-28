#!/bin/bash

# LLM Optimization Platform Deployment Script
# This script handles the deployment of the platform using Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROD_COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"
BACKUP_DIR="./backups"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

check_environment() {
    log_info "Checking environment configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f ".env.example" ]; then
            log_warning "No .env file found. Copying from .env.example"
            cp .env.example .env
            log_warning "Please edit .env file with your configuration before proceeding"
            exit 1
        else
            log_error "No .env file found and no .env.example available"
            exit 1
        fi
    fi
    
    # Check for required environment variables
    source .env
    required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "SECRET_KEY")
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log_success "Environment configuration check passed"
}

create_directories() {
    log_info "Creating necessary directories..."
    
    directories=("$BACKUP_DIR" "./logs" "./models" "./datasets")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Directory creation completed"
}

backup_existing_data() {
    log_info "Creating backup of existing data..."
    
    if docker-compose ps | grep -q "llm-platform-db"; then
        timestamp=$(date +"%Y%m%d_%H%M%S")
        backup_file="$BACKUP_DIR/pre_deploy_backup_$timestamp.sql"
        
        log_info "Backing up database to $backup_file"
        docker-compose exec -T database pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" > "$backup_file" 2>/dev/null || {
            log_warning "Could not create database backup (database might not be running)"
        }
    fi
}

deploy_development() {
    log_info "Deploying development environment..."
    
    # Stop existing containers
    docker-compose down
    
    # Build and start services
    docker-compose up -d --build
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log_success "Development deployment completed"
}

deploy_production() {
    log_info "Deploying production environment..."
    
    if [ ! -f "$PROD_COMPOSE_FILE" ]; then
        log_error "Production compose file not found: $PROD_COMPOSE_FILE"
        exit 1
    fi
    
    # Stop existing containers
    docker-compose -f "$PROD_COMPOSE_FILE" down
    
    # Build and start services
    docker-compose -f "$PROD_COMPOSE_FILE" up -d --build
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 60
    
    # Check service health
    check_service_health_prod
    
    log_success "Production deployment completed"
}

check_service_health() {
    log_info "Checking service health..."
    
    # Check backend health
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:5000/api/v1/health &> /dev/null; then
            log_success "Backend service is healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Backend service health check failed after $max_attempts attempts"
            docker-compose logs backend
            exit 1
        fi
        
        log_info "Attempt $attempt/$max_attempts: Waiting for backend service..."
        sleep 10
        ((attempt++))
    done
    
    # Check frontend (if available)
    if curl -f http://localhost:3000 &> /dev/null; then
        log_success "Frontend service is healthy"
    else
        log_warning "Frontend service health check failed (might not be deployed)"
    fi
    
    # Check database
    if docker-compose exec database pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" &> /dev/null; then
        log_success "Database service is healthy"
    else
        log_error "Database service is not ready"
        exit 1
    fi
}

check_service_health_prod() {
    log_info "Checking production service health..."
    
    # Similar to development but with production endpoints
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost/api/v1/health &> /dev/null; then
            log_success "Production backend service is healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Production backend service health check failed"
            docker-compose -f "$PROD_COMPOSE_FILE" logs backend
            exit 1
        fi
        
        log_info "Attempt $attempt/$max_attempts: Waiting for production backend service..."
        sleep 10
        ((attempt++))
    done
}

show_status() {
    log_info "Service Status:"
    docker-compose ps
    
    log_info "Available endpoints:"
    echo "  - Backend API: http://localhost:5000"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Streamlit: http://localhost:8501"
    echo "  - Health Check: http://localhost:5000/api/v1/health"
}

show_logs() {
    log_info "Recent logs:"
    docker-compose logs --tail=50
}

cleanup() {
    log_info "Cleaning up unused Docker resources..."
    docker system prune -f
    log_success "Cleanup completed"
}

# Main script
main() {
    echo "=========================================="
    echo "LLM Optimization Platform Deployment"
    echo "=========================================="
    
    # Parse command line arguments
    ENVIRONMENT="development"
    SKIP_BACKUP=false
    SHOW_LOGS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --logs)
                SHOW_LOGS=true
                shift
                ;;
            --cleanup)
                cleanup
                exit 0
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  -e, --environment ENV    Deployment environment (development|production)"
                echo "  --skip-backup           Skip database backup"
                echo "  --logs                  Show logs after deployment"
                echo "  --cleanup               Clean up unused Docker resources"
                echo "  -h, --help              Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    check_prerequisites
    check_environment
    create_directories
    
    if [ "$SKIP_BACKUP" = false ]; then
        backup_existing_data
    fi
    
    case $ENVIRONMENT in
        development|dev)
            deploy_development
            ;;
        production|prod)
            deploy_production
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, production"
            exit 1
            ;;
    esac
    
    show_status
    
    if [ "$SHOW_LOGS" = true ]; then
        show_logs
    fi
    
    log_success "Deployment completed successfully!"
    log_info "Run '$0 --logs' to view recent logs"
    log_info "Run 'docker-compose logs -f [service]' to follow logs for a specific service"
}

# Run main function
main "$@"