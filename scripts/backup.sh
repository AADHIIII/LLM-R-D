#!/bin/bash

# LLM Optimization Platform Backup Script
# This script creates backups of the database and important files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COMPOSE_FILE="docker-compose.yml"
PROD_COMPOSE_FILE="docker-compose.prod.yml"

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

create_backup_directory() {
    log_info "Creating backup directory..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR/database"
    mkdir -p "$BACKUP_DIR/models"
    mkdir -p "$BACKUP_DIR/logs"
    mkdir -p "$BACKUP_DIR/config"
    
    log_success "Backup directory structure created"
}

backup_database() {
    log_info "Backing up database..."
    
    # Check if database container is running
    if ! docker-compose ps | grep -q "llm-platform-db.*Up"; then
        log_error "Database container is not running"
        return 1
    fi
    
    # Load environment variables
    if [ -f ".env" ]; then
        source .env
    else
        log_error "No .env file found"
        return 1
    fi
    
    # Create database backup
    backup_file="$BACKUP_DIR/database/db_backup_$TIMESTAMP.sql"
    
    log_info "Creating database backup: $backup_file"
    
    if docker-compose exec -T database pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" > "$backup_file"; then
        log_success "Database backup created successfully"
        
        # Compress the backup
        gzip "$backup_file"
        log_success "Database backup compressed: ${backup_file}.gz"
        
        # Get backup size
        backup_size=$(du -h "${backup_file}.gz" | cut -f1)
        log_info "Backup size: $backup_size"
        
        return 0
    else
        log_error "Failed to create database backup"
        return 1
    fi
}

backup_models() {
    log_info "Backing up models..."
    
    if [ -d "./models" ] && [ "$(ls -A ./models)" ]; then
        model_backup="$BACKUP_DIR/models/models_backup_$TIMESTAMP.tar.gz"
        
        log_info "Creating models backup: $model_backup"
        
        if tar -czf "$model_backup" -C . models/; then
            log_success "Models backup created successfully"
            
            # Get backup size
            backup_size=$(du -h "$model_backup" | cut -f1)
            log_info "Models backup size: $backup_size"
        else
            log_error "Failed to create models backup"
            return 1
        fi
    else
        log_warning "No models directory found or directory is empty"
    fi
}

backup_logs() {
    log_info "Backing up logs..."
    
    if [ -d "./logs" ] && [ "$(ls -A ./logs)" ]; then
        logs_backup="$BACKUP_DIR/logs/logs_backup_$TIMESTAMP.tar.gz"
        
        log_info "Creating logs backup: $logs_backup"
        
        if tar -czf "$logs_backup" -C . logs/; then
            log_success "Logs backup created successfully"
            
            # Get backup size
            backup_size=$(du -h "$logs_backup" | cut -f1)
            log_info "Logs backup size: $backup_size"
        else
            log_error "Failed to create logs backup"
            return 1
        fi
    else
        log_warning "No logs directory found or directory is empty"
    fi
}

backup_configuration() {
    log_info "Backing up configuration files..."
    
    config_backup="$BACKUP_DIR/config/config_backup_$TIMESTAMP.tar.gz"
    
    # Files to backup
    config_files=()
    
    [ -f ".env" ] && config_files+=(".env")
    [ -f "docker-compose.yml" ] && config_files+=("docker-compose.yml")
    [ -f "docker-compose.prod.yml" ] && config_files+=("docker-compose.prod.yml")
    [ -f "requirements.txt" ] && config_files+=("requirements.txt")
    [ -d "config/" ] && config_files+=("config/")
    
    if [ ${#config_files[@]} -gt 0 ]; then
        log_info "Creating configuration backup: $config_backup"
        
        if tar -czf "$config_backup" "${config_files[@]}"; then
            log_success "Configuration backup created successfully"
        else
            log_error "Failed to create configuration backup"
            return 1
        fi
    else
        log_warning "No configuration files found to backup"
    fi
}

backup_docker_volumes() {
    log_info "Backing up Docker volumes..."
    
    # Get list of volumes
    volumes=$(docker volume ls --filter name=llm-optimization-platform --format "{{.Name}}")
    
    if [ -n "$volumes" ]; then
        volumes_backup="$BACKUP_DIR/volumes_backup_$TIMESTAMP.tar.gz"
        
        log_info "Creating volumes backup: $volumes_backup"
        
        # Create temporary container to backup volumes
        temp_container="backup-temp-$TIMESTAMP"
        
        if docker run --rm -v "$(pwd)/$BACKUP_DIR:/backup" \
            $(echo "$volumes" | sed 's/^/-v /; s/$/:\/data\/&/') \
            alpine tar -czf "/backup/volumes_backup_$TIMESTAMP.tar.gz" -C /data .; then
            log_success "Docker volumes backup created successfully"
        else
            log_error "Failed to create Docker volumes backup"
            return 1
        fi
    else
        log_warning "No Docker volumes found to backup"
    fi
}

cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    # Keep only last 7 days of backups
    find "$BACKUP_DIR" -name "*.gz" -mtime +7 -delete 2>/dev/null || true
    find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete 2>/dev/null || true
    
    log_success "Old backups cleaned up"
}

create_backup_manifest() {
    log_info "Creating backup manifest..."
    
    manifest_file="$BACKUP_DIR/backup_manifest_$TIMESTAMP.txt"
    
    {
        echo "LLM Optimization Platform Backup Manifest"
        echo "========================================"
        echo "Backup Date: $(date)"
        echo "Timestamp: $TIMESTAMP"
        echo ""
        echo "Files included in this backup:"
        echo ""
        
        find "$BACKUP_DIR" -name "*$TIMESTAMP*" -type f | while read -r file; do
            size=$(du -h "$file" | cut -f1)
            echo "  - $(basename "$file") ($size)"
        done
        
        echo ""
        echo "Docker containers status at backup time:"
        docker-compose ps 2>/dev/null || echo "  No containers running"
        
        echo ""
        echo "System information:"
        echo "  - Docker version: $(docker --version)"
        echo "  - Docker Compose version: $(docker-compose --version)"
        echo "  - Host: $(hostname)"
        echo "  - User: $(whoami)"
        
    } > "$manifest_file"
    
    log_success "Backup manifest created: $manifest_file"
}

verify_backup() {
    log_info "Verifying backup integrity..."
    
    # Check if backup files exist and are not empty
    backup_files=$(find "$BACKUP_DIR" -name "*$TIMESTAMP*" -type f)
    
    if [ -z "$backup_files" ]; then
        log_error "No backup files found for timestamp $TIMESTAMP"
        return 1
    fi
    
    for file in $backup_files; do
        if [ ! -s "$file" ]; then
            log_error "Backup file is empty: $file"
            return 1
        fi
        
        # Test compressed files
        if [[ "$file" == *.gz ]]; then
            if ! gzip -t "$file" 2>/dev/null; then
                log_error "Corrupted compressed file: $file"
                return 1
            fi
        fi
        
        if [[ "$file" == *.tar.gz ]]; then
            if ! tar -tzf "$file" >/dev/null 2>&1; then
                log_error "Corrupted tar file: $file"
                return 1
            fi
        fi
    done
    
    log_success "Backup verification completed successfully"
}

show_backup_summary() {
    log_info "Backup Summary:"
    echo "==============="
    
    total_size=0
    file_count=0
    
    find "$BACKUP_DIR" -name "*$TIMESTAMP*" -type f | while read -r file; do
        size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
        size_human=$(du -h "$file" | cut -f1)
        
        echo "  $(basename "$file"): $size_human"
        
        total_size=$((total_size + size_bytes))
        file_count=$((file_count + 1))
    done
    
    echo ""
    echo "Total files: $file_count"
    echo "Backup location: $BACKUP_DIR"
    echo "Timestamp: $TIMESTAMP"
}

# Main script
main() {
    echo "=========================================="
    echo "LLM Optimization Platform Backup"
    echo "=========================================="
    
    # Parse command line arguments
    SKIP_MODELS=false
    SKIP_LOGS=false
    SKIP_VOLUMES=false
    CLEANUP=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-models)
                SKIP_MODELS=true
                shift
                ;;
            --skip-logs)
                SKIP_LOGS=true
                shift
                ;;
            --skip-volumes)
                SKIP_VOLUMES=true
                shift
                ;;
            --no-cleanup)
                CLEANUP=false
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-models     Skip models backup"
                echo "  --skip-logs       Skip logs backup"
                echo "  --skip-volumes    Skip Docker volumes backup"
                echo "  --no-cleanup      Don't clean up old backups"
                echo "  -h, --help        Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Start backup process
    log_info "Starting backup process..."
    
    create_backup_directory
    
    # Always backup database and configuration
    backup_database
    backup_configuration
    
    # Optional backups
    if [ "$SKIP_MODELS" = false ]; then
        backup_models
    fi
    
    if [ "$SKIP_LOGS" = false ]; then
        backup_logs
    fi
    
    if [ "$SKIP_VOLUMES" = false ]; then
        backup_docker_volumes
    fi
    
    # Create manifest and verify
    create_backup_manifest
    verify_backup
    
    # Cleanup old backups
    if [ "$CLEANUP" = true ]; then
        cleanup_old_backups
    fi
    
    # Show summary
    show_backup_summary
    
    log_success "Backup completed successfully!"
    log_info "Backup timestamp: $TIMESTAMP"
    log_info "Backup location: $BACKUP_DIR"
}

# Run main function
main "$@"