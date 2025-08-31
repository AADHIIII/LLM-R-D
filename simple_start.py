#!/usr/bin/env python3
"""
Simple startup script for development testing
"""
import os
import sys
from flask import Flask, jsonify
from flask_cors import CORS

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_simple_app():
    """Create a simple Flask app for testing"""
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def home():
        return jsonify({
            "message": "LLM Optimization Platform API",
            "status": "running",
            "version": "1.0.0"
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            "status": "healthy",
            "timestamp": "2025-08-29T02:48:05.302851"
        })
    
    @app.route('/api/v1/health')
    def api_health():
        return jsonify({
            "status": "healthy",
            "service": "LLM Optimization Platform",
            "timestamp": "2025-08-29T02:48:05.302851"
        })
    
    @app.route('/api/v1/models')
    def models():
        return jsonify({
            "models": [
                {
                    "id": "gemini-1.5-flash",
                    "name": "Gemini 1.5 Flash",
                    "provider": "Google",
                    "status": "available"
                }
            ]
        })
    
    @app.route('/api/v1/generate', methods=['POST'])
    def generate():
        try:
            from api.services.gemini_client import GeminiClient
            
            # Test Gemini integration
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return jsonify({
                    "error": "API key not configured",
                    "message": "Please set GEMINI_API_KEY in .env file"
                }), 400
            
            client = GeminiClient(api_key)
            result = client.generate_text(
                prompt="Hello! This is a test from your LLM optimization platform.",
                model="gemini-1.5-flash"
            )
            
            return jsonify({
                "success": True,
                "result": result,
                "model": "gemini-1.5-flash"
            })
            
        except Exception as e:
            return jsonify({
                "error": "generation_failed",
                "message": str(e)
            }), 500
    
    return app

if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    app = create_simple_app()
    
    # Find an available port
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    
    print(f"🚀 Starting LLM Optimization Platform on http://localhost:{port}")
    print(f"📋 Available endpoints:")
    print(f"   GET  http://localhost:{port}/")
    print(f"   GET  http://localhost:{port}/health")
    print(f"   GET  http://localhost:{port}/api/v1/health")
    print(f"   GET  http://localhost:{port}/api/v1/models")
    print(f"   POST http://localhost:{port}/api/v1/generate")
    print(f"")
    print(f"🧪 Test your Gemini integration:")
    print(f"   curl -X POST http://localhost:{port}/api/v1/generate")
    print(f"")
    print(f"🛑 Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=port, debug=True)