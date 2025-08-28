#!/usr/bin/env python3
"""
Script to run the API gateway service.
"""
import os
from api.app import create_app

if __name__ == '__main__':
    # Get environment from environment variable or default to development
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Create the Flask app
    app = create_app(env)
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = env == 'development'
    
    print(f"Starting API Gateway in {env} mode on port {port}")
    print(f"Available endpoints:")
    print(f"  - GET  /api/v1/health")
    print(f"  - GET  /api/v1/status")
    print(f"  - GET  /api/v1/ready")
    print(f"  - GET  /api/v1/models")
    print(f"  - POST /api/v1/generate")
    print(f"  - POST /api/v1/evaluate")
    
    app.run(host='0.0.0.0', port=port, debug=debug)