#!/usr/bin/env python3
"""
Simple startup script that bypasses complex configuration.
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, '.')

# Set basic environment variables
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = 'true'

def start_server():
    """Start the Flask server directly."""
    print("🚀 LLM OPTIMIZATION PLATFORM")
    print("=" * 50)
    print("Starting API server...")
    print("Available at: http://localhost:5000")
    print("Health check: http://localhost:5000/api/v1/health")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        from api.app import create_app
        
        # Create app with minimal config
        app = create_app('testing')  # Use testing config to avoid validation issues
        
        # Override some settings
        app.config['SECRET_KEY'] = 'dev-secret-key'
        app.config['TESTING'] = False
        
        # Start server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Disable reloader to avoid issues
        )
        
    except KeyboardInterrupt:
        print("\\n\\n👋 Server stopped")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("\\nTrying alternative startup method...")
        
        # Alternative: use run_api.py
        try:
            exec(open('run_api.py').read())
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")

if __name__ == "__main__":
    start_server()