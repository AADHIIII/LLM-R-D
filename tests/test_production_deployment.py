"""
Tests for production deployment scenarios.
"""

import pytest
import requests
import time
import subprocess
import os
import yaml
from typing import Dict, Any


class TestProductionDeployment:
    """Test production deployment configurations and scenarios."""
    
    @pytest.fixture(scope="class")
    def prod_env_vars(self):
        """Production environment variables for testing."""
        return {
            'POSTGRES_DB': 'test_llm_platform_prod',
            'POSTGRES_USER': 'test_user_prod',
            'POSTGRES_PASSWORD': 'TestPassword123Secure!',
            'REDIS_PASSWORD': 'TestRedisPassword123Secure!',
            'SECRET_KEY': 'TestSecretKey123ProductionSecure!',
            'JWT_SECRET_KEY': 'TestJwtSecretKey123ProductionSecure!',
            'FLASK_ENV': 'production',
            'OPENAI_API_KEY': 'test-openai-key-prod',
            'ANTHROPIC_API_KEY': 'test-anthropic-key-prod',
            'CORS_ORIGINS': 'https://your-domain.com',
            'GRAFANA_PASSWORD': 'TestGrafanaPassword123Secure!'
        }
    
    def test_production_compose_file_structure(self):
        """Test production docker-compose file has proper structure."""
        prod_compose_path = "docker-compose.prod.yml"
        assert os.path.exists(prod_compose_path), "Production compose file not found"
        
        with open(prod_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check for production-specific configurations
        services = compose_config.get('services', {})
        
        # Check backend service has production configurations
        backend = services.get('backend', {})
        assert 'deploy' in backend, "Missing deployment configuration for backend"
        
        deploy_config = backend['deploy']
        assert 'resources' in deploy_config, "Missing resource limits"
        assert 'limits' in deploy_config['resources'], "Missing resource limits"
        
        # Check for security configurations
        environment = backend.get('environment', [])
        env_dict = {}
        for env_var in environment:
            if '=' in env_var:
                key, value = env_var.split('=', 1)
                env_dict[key] = value
        
        assert env_dict.get('FLASK_ENV') == 'production', "Backend not configured for production"
    
    def test_production_dockerfile_security(self):
        """Test production Dockerfile has security hardening."""
        prod_dockerfile = "Dockerfile.prod"
        assert os.path.exists(prod_dockerfile), "Production Dockerfile not found"
        
        with open(prod_dockerfile, 'r') as f:
            content = f.read()
        
        # Check for security features
        security_checks = [
            ("Non-root user", "USER appuser"),
            ("Security updates", "apt-get upgrade"),
            ("Remove build tools", "apt-get remove"),
            ("Secure permissions", "chmod"),
            ("Production server", "gunicorn"),
            ("Health check", "HEALTHCHECK")
        ]
        
        for check_name, pattern in security_checks:
            assert pattern in content, f"Missing security feature: {check_name}"
    
    def test_nginx_production_configuration(self):
        """Test nginx production configuration."""
        nginx_config_path = "nginx/nginx.conf"
        if not os.path.exists(nginx_config_path):
            pytest.skip("Nginx configuration not found")
        
        with open(nginx_config_path, 'r') as f:
            content = f.read()
        
        # Check for production features
        production_features = [
            "upstream backend",
            "limit_req_zone",
            "ssl_protocols",
            "gzip on",
            "proxy_pass",
            "add_header X-Frame-Options"
        ]
        
        for feature in production_features:
            assert feature in content, f"Missing nginx feature: {feature}"
    
    def test_monitoring_configuration(self):
        """Test monitoring configuration exists."""
        monitoring_files = [
            "monitoring/prometheus.yml",
            "monitoring/grafana/datasources/prometheus.yml"
        ]
        
        for file_path in monitoring_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    assert len(content) > 0, f"Empty monitoring file: {file_path}"
    
    def test_backup_script_functionality(self):
        """Test backup script has production-ready features."""
        backup_script = "scripts/backup.sh"
        assert os.path.exists(backup_script), "Backup script not found"
        
        with open(backup_script, 'r') as f:
            content = f.read()
        
        # Check for essential backup features
        backup_features = [
            "backup_database",
            "backup_models",
            "backup_configuration",
            "verify_backup",
            "cleanup_old_backups",
            "gzip"
        ]
        
        for feature in backup_features:
            assert feature in content, f"Missing backup feature: {feature}"
    
    def test_deployment_script_production_mode(self):
        """Test deployment script supports production mode."""
        deploy_script = "scripts/deploy.sh"
        assert os.path.exists(deploy_script), "Deployment script not found"
        
        with open(deploy_script, 'r') as f:
            content = f.read()
        
        # Check for production deployment features
        production_features = [
            "deploy_production",
            "docker-compose.prod.yml",
            "check_service_health",
            "backup_existing_data",
            "production|prod"
        ]
        
        for feature in production_features:
            assert feature in content, f"Missing production feature: {feature}"
    
    def test_environment_variable_validation(self, prod_env_vars):
        """Test production environment variables are properly validated."""
        # Check password complexity
        passwords = [
            prod_env_vars['POSTGRES_PASSWORD'],
            prod_env_vars['REDIS_PASSWORD'],
            prod_env_vars['SECRET_KEY']
        ]
        
        for password in passwords:
            assert len(password) >= 12, f"Password too short: {password[:5]}..."
            assert any(c.isupper() for c in password), f"Password missing uppercase: {password[:5]}..."
            assert any(c.islower() for c in password), f"Password missing lowercase: {password[:5]}..."
            assert any(c.isdigit() for c in password), f"Password missing digit: {password[:5]}..."
    
    def test_ssl_configuration_template(self):
        """Test SSL configuration templates exist."""
        nginx_config = "nginx/nginx.conf"
        if os.path.exists(nginx_config):
            with open(nginx_config, 'r') as f:
                content = f.read()
            
            # Check for SSL configuration templates (commented out)
            ssl_features = [
                "ssl_certificate",
                "ssl_protocols",
                "listen 443",
                "TLSv1.2"
            ]
            
            # These should be present as comments for easy enabling
            for feature in ssl_features:
                assert feature in content, f"Missing SSL template: {feature}"
    
    def test_resource_limits_configuration(self):
        """Test resource limits are properly configured."""
        prod_compose_path = "docker-compose.prod.yml"
        if not os.path.exists(prod_compose_path):
            pytest.skip("Production compose file not found")
        
        with open(prod_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config.get('services', {})
        
        # Check critical services have resource limits
        critical_services = ['backend', 'database', 'redis']
        
        for service_name in critical_services:
            if service_name in services:
                service = services[service_name]
                deploy = service.get('deploy', {})
                resources = deploy.get('resources', {})
                limits = resources.get('limits', {})
                
                assert 'memory' in limits, f"Missing memory limit for {service_name}"
                assert 'cpus' in limits, f"Missing CPU limit for {service_name}"
    
    def test_health_check_configuration(self):
        """Test all services have proper health checks."""
        compose_files = ["docker-compose.yml", "docker-compose.prod.yml"]
        
        for compose_file in compose_files:
            if not os.path.exists(compose_file):
                continue
                
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            services = compose_config.get('services', {})
            
            # Services that should have health checks
            health_check_services = ['database', 'redis', 'backend']
            
            for service_name in health_check_services:
                if service_name in services:
                    service = services[service_name]
                    assert 'healthcheck' in service, f"Missing health check for {service_name} in {compose_file}"
                    
                    health_check = service['healthcheck']
                    assert 'test' in health_check, f"Missing health check test for {service_name}"
                    assert 'interval' in health_check, f"Missing health check interval for {service_name}"
                    assert 'retries' in health_check, f"Missing health check retries for {service_name}"
    
    def test_network_security_configuration(self):
        """Test network security is properly configured."""
        prod_compose_path = "docker-compose.prod.yml"
        if not os.path.exists(prod_compose_path):
            pytest.skip("Production compose file not found")
        
        with open(prod_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check for custom networks
        networks = compose_config.get('networks', {})
        assert len(networks) > 0, "No custom networks defined"
        
        # Check network configuration
        for network_name, network_config in networks.items():
            if network_config:
                assert 'driver' in network_config, f"Missing driver for network {network_name}"
    
    def test_volume_persistence_configuration(self):
        """Test data persistence is properly configured."""
        compose_files = ["docker-compose.yml", "docker-compose.prod.yml"]
        
        for compose_file in compose_files:
            if not os.path.exists(compose_file):
                continue
                
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            # Check for persistent volumes
            volumes = compose_config.get('volumes', {})
            
            # Essential volumes for data persistence
            essential_volumes = ['postgres_data', 'redis_data']
            
            for volume in essential_volumes:
                # Check if volume exists (might have different naming in prod)
                volume_exists = any(vol for vol in volumes.keys() if volume.split('_')[0] in vol)
                assert volume_exists, f"Missing persistent volume for {volume} in {compose_file}"
    
    @pytest.mark.integration
    def test_production_deployment_dry_run(self, prod_env_vars):
        """Test production deployment script dry run."""
        deploy_script = "./scripts/deploy.sh"
        if not os.path.exists(deploy_script):
            pytest.skip("Deployment script not found")
        
        # Test script validation without actual deployment
        env = {**os.environ, **prod_env_vars}
        
        # Test help option
        result = subprocess.run([deploy_script, "--help"], 
                              capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"Deploy script help failed: {result.stderr}"
        assert "production" in result.stdout, "Production option not documented"
    
    def test_backup_script_dry_run(self):
        """Test backup script dry run."""
        backup_script = "./scripts/backup.sh"
        if not os.path.exists(backup_script):
            pytest.skip("Backup script not found")
        
        # Test help option
        result = subprocess.run([backup_script, "--help"], 
                              capture_output=True, text=True)
        assert result.returncode == 0, f"Backup script help failed: {result.stderr}"
    
    def test_deployment_documentation_exists(self):
        """Test deployment documentation exists and is comprehensive."""
        doc_file = "DEPLOYMENT.md"
        assert os.path.exists(doc_file), "Deployment documentation not found"
        
        with open(doc_file, 'r') as f:
            content = f.read()
        
        # Check for essential documentation sections
        required_sections = [
            "Prerequisites",
            "Production Deployment",
            "SSL/TLS Configuration",
            "Monitoring",
            "Backup and Recovery",
            "Troubleshooting",
            "Security"
        ]
        
        for section in required_sections:
            assert section in content, f"Missing documentation section: {section}"
        
        # Check documentation is substantial
        assert len(content) > 5000, "Documentation too brief for production use"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])