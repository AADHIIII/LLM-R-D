"""
Main entry point for LLM optimization platform.
Initializes the application and starts the appropriate service.
"""

import argparse
import sys
from typing import Optional

from config.settings import get_settings, create_directories
from utils.logging import setup_logging, get_logger


def setup_application():
    """Initialize the application with proper configuration."""
    # Create necessary directories
    create_directories()
    
    # Setup logging
    setup_logging()
    
    logger = get_logger(__name__)
    logger.info("LLM Optimization Platform starting up...")
    
    # Validate configuration
    settings = get_settings()
    logger.info(f"Configuration loaded - Environment: {settings.flask_env}")
    
    return settings


def run_api_server():
    """Run the Flask API server."""
    logger = get_logger(__name__)
    logger.info("Starting API server...")
    
    try:
        # Import here to avoid circular imports
        from api.app import create_app
        
        settings = get_settings()
        app = create_app()
        
        app.run(
            host=settings.api_host,
            port=settings.api_port,
            debug=settings.flask_debug,
        )
    except ImportError:
        logger.error("API module not yet implemented")
        sys.exit(1)


def run_web_interface():
    """Run the Streamlit web interface."""
    logger = get_logger(__name__)
    logger.info("Starting web interface...")
    
    try:
        import subprocess
        import os
        
        settings = get_settings()
        
        # Set Streamlit configuration
        env = os.environ.copy()
        env['STREAMLIT_SERVER_PORT'] = str(settings.streamlit_port)
        env['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        
        # Run Streamlit app
        subprocess.run([
            'streamlit', 'run', 'web_interface/app.py',
            '--server.port', str(settings.streamlit_port),
            '--server.address', '0.0.0.0'
        ], env=env)
        
    except FileNotFoundError:
        logger.error("Streamlit not installed or web interface not implemented")
        sys.exit(1)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="LLM Optimization Platform")
    parser.add_argument(
        'service',
        choices=['api', 'web', 'all'],
        help='Service to run (api, web, or all)'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration file',
        default=None
    )
    
    args = parser.parse_args()
    
    # Setup application
    settings = setup_application()
    logger = get_logger(__name__)
    
    try:
        if args.service == 'api':
            run_api_server()
        elif args.service == 'web':
            run_web_interface()
        elif args.service == 'all':
            logger.info("Starting all services...")
            # In a production environment, you'd use a process manager
            # For now, we'll just start the API server
            run_api_server()
        
    except KeyboardInterrupt:
        logger.info("Application shutdown requested")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()