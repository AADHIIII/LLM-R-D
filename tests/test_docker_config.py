"""
Tests for Docker configuration files.
"""

import pytest
import os
import yaml
import json


class TestDockerConfiguration:
    """Test Docker configuration files exist and are valid."""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and is valid."""
        dockerfile_path = "Dockerfile"
        assert os.path.exists(dockerfile_path), "Dockerfile not found"
        
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            
        # Check for essential Dockerfile instructions
        assert "FROM python:" in content, "Missing Python base image"
        assert "WORKDIR /app" in content, "Missing working directory"
        assert "COPY requirements.txt" in content, "Missing requirements copy"
        assert "RUN pip install" in content, "Missing pip install"
        assert "EXPOSE" in content, "Missing port exposure"
        assert "HEALTHCHECK" in content, "Missing health check"
        assert "CMD" in content, "Missing CMD instruction"
    
    def test_production_dockerfile_exists(self):
        """Test that production Dockerfile exists with security hardening."""
        dockerfile_path = "Dockerfile.prod"
        assert os.path.exists(dockerfile_path), "Production Dockerfile not found"
        
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            
        # Check for security features
        assert "adduser" in content or "useradd" in content, "Missing non-root user creation"
        assert "USER" in content, "Missing user switch"
        assert "FLASK_ENV=production" in content, "Missing production environment"
        assert "gunicorn" in content, "Missing production server"
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists and is valid."""
        compose_path = "docker-compose.yml"
        assert os.path.exists(compose_path), "docker-compose.yml not found"
        
        with open(compose_path, 'r') as f:
            try:
                compose_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in docker-compose.yml: {e}")
        
        # Check for essential services
        assert 'services' in compose_config, "Missing services section"
        services = compose_config['services']
        
        required_services = ['database', 'redis', 'backend']
        for service in required_services:
            assert service in services, f"Missing service: {service}"
        
        # Check for essential configurations
        assert 'networks' in compose_config, "Missing network configuration"
        assert 'volumes' in compose_config, "Missing volume configuration"
        
        # Check health checks exist
        for service_name, service_config in services.items():
            if service_name in ['database', 'redis', 'backend']:
                assert 'healthcheck' in service_config, f"Missing health check for {service_name}"
    
    def test_production_compose_exists(self):
        """Test that production docker-compose file exists."""
        prod_compose_path = "docker-compose.prod.yml"
        assert os.path.exists(prod_compose_path), "Production compose file not found"
        
        with open(prod_compose_path, 'r') as f:
            try:
                compose_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in docker-compose.prod.yml: {e}")
        
        # Check for production-specific features
        services = compose_config.get('services', {})
        
        # Check for resource limits
        backend_service = services.get('backend', {})
        if 'deploy' in backend_service:
            deploy_config = backend_service['deploy']
            assert 'resources' in deploy_config, "Missing resource configuration"
            assert 'limits' in deploy_config['resources'], "Missing resource limits"
    
    def test_environment_configuration(self):
        """Test environment variable configuration."""
        env_example_path = ".env.example"
        assert os.path.exists(env_example_path), ".env.example not found"
        
        with open(env_example_path, 'r') as f:
            content = f.read()
            
        # Check for essential environment variables
        required_vars = [
            'FLASK_ENV', 'DATABASE_URL', 'REDIS_URL',
            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'SECRET_KEY'
        ]
        
        for var in required_vars:
            assert var in content, f"Missing environment variable: {var}"
    
    def test_dockerignore_exists(self):
        """Test that .dockerignore exists and excludes appropriate files."""
        dockerignore_path = ".dockerignore"
        assert os.path.exists(dockerignore_path), ".dockerignore not found"
        
        with open(dockerignore_path, 'r') as f:
            content = f.read()
            
        # Check for essential exclusions
        exclusions = ['.git', '__pycache__', '*.py[cod]', '.env', 'node_modules', 'logs']
        for exclusion in exclusions:
            assert exclusion in content, f"Missing exclusion: {exclusion}"
    
    def test_database_initialization_script(self):
        """Test database initialization script exists."""
        init_script_path = "database/init.sql"
        assert os.path.exists(init_script_path), "Database init script not found"
        
        with open(init_script_path, 'r') as f:
            content = f.read()
        
        # Check for essential database objects
        db_objects = [
            'CREATE TABLE IF NOT EXISTS experiments',
            'CREATE TABLE IF NOT EXISTS models',
            'CREATE TABLE IF NOT EXISTS evaluations',
            'CREATE INDEX',
            'uuid_generate_v4()'
        ]
        
        for obj in db_objects:
            assert obj in content, f"Missing database object: {obj}"
    
    def test_frontend_dockerfile_exists(self):
        """Test frontend Dockerfile exists."""
        frontend_dockerfile = "web_interface/frontend/Dockerfile"
        if os.path.exists(frontend_dockerfile):
            with open(frontend_dockerfile, 'r') as f:
                content = f.read()
            
            # Check for Node.js setup
            assert "FROM node:" in content, "Missing Node.js base image"
            assert "npm" in content, "Missing npm commands"
            assert "nginx" in content, "Missing nginx for serving"
    
    def test_nginx_configuration_exists(self):
        """Test nginx configuration exists for frontend."""
        nginx_config = "web_interface/frontend/nginx.conf"
        if os.path.exists(nginx_config):
            with open(nginx_config, 'r') as f:
                content = f.read()
            
            # Check for essential nginx configuration
            assert "server {" in content, "Missing server block"
            assert "location /api/" in content, "Missing API proxy configuration"
            assert "proxy_pass" in content, "Missing proxy configuration"
    
    def test_deployment_scripts_exist(self):
        """Test deployment scripts exist and are executable."""
        scripts = [
            "scripts/deploy.sh",
            "scripts/backup.sh"
        ]
        
        for script in scripts:
            assert os.path.exists(script), f"Script not found: {script}"
            assert os.access(script, os.X_OK), f"Script not executable: {script}"
    
    def test_deployment_script_content(self):
        """Test deployment script has essential functions."""
        deploy_script = "scripts/deploy.sh"
        if os.path.exists(deploy_script):
            with open(deploy_script, 'r') as f:
                content = f.read()
            
            # Check for essential functions
            functions = [
                "check_prerequisites",
                "check_environment",
                "deploy_development",
                "deploy_production",
                "check_service_health"
            ]
            
            for func in functions:
                assert func in content, f"Missing function: {func}"
    
    def test_backup_script_content(self):
        """Test backup script has essential functions."""
        backup_script = "scripts/backup.sh"
        if os.path.exists(backup_script):
            with open(backup_script, 'r') as f:
                content = f.read()
            
            # Check for essential functions
            functions = [
                "backup_database",
                "backup_models",
                "backup_configuration",
                "verify_backup"
            ]
            
            for func in functions:
                assert func in content, f"Missing function: {func}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])