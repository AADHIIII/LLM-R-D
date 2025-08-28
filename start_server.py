#!/usr/bin/env python3
"""
Start the LLM Optimization Platform locally.
"""
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print('🚀 LLM OPTIMIZATION PLATFORM - LOCAL DEPLOYMENT')
    print('=' * 60)
    
    # Create necessary directories
    directories = ['logs', 'models', 'datasets']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f'✅ Created {directory}/ directory')
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = 'false'  # Disable auto-reload to avoid issues
    
    # Use port 8000 to avoid AirPlay conflict
    port = int(os.environ.get('PORT', 8000))
    
    print(f'\\n🌐 Starting API Gateway on port {port}...')
    print(f'Server available at: http://localhost:{port}')
    print('\\nAPI Endpoints:')
    print(f'  - GET  http://localhost:{port}/api/v1/health')
    print(f'  - GET  http://localhost:{port}/api/v1/models')
    print(f'  - POST http://localhost:{port}/api/v1/generate')
    print(f'  - GET  http://localhost:{port}/api/v1/monitoring/metrics')
    print('\\nPress Ctrl+C to stop the server')
    print('=' * 60)
    
    try:
        # Import and create the Flask app
        from api.app import create_app
        
        app = create_app('development')
        
        # Run the app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,  # Disable debug mode to avoid restart issues
            use_reloader=False  # Disable auto-reloader
        )
        
    except KeyboardInterrupt:
        print('\\n\\n👋 Server stopped by user')
    except Exception as e:
        print(f'\\n❌ Error starting server: {e}')
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())