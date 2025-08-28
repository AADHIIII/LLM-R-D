#!/usr/bin/env python3
"""
Quick start script that bypasses configuration issues.
"""
import os
import sys

# Set up environment
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = 'true'

# Add to path
sys.path.insert(0, '.')

def create_minimal_app():
    """Create a minimal Flask app for testing."""
    from flask import Flask, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    # Basic configuration
    app.config['SECRET_KEY'] = 'dev-secret-key'
    app.config['DEBUG'] = True
    
    @app.route('/api/v1/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'service': 'LLM Optimization Platform',
            'version': '1.0.0'
        })
    
    @app.route('/api/v1/models')
    def models():
        return jsonify({
            'models': [
                {
                    'id': 'gemini-pro',
                    'name': 'Gemini Pro',
                    'provider': 'google',
                    'type': 'commercial',
                    'status': 'available' if os.getenv('GEMINI_API_KEY') else 'needs_api_key'
                },
                {
                    'id': 'gpt2-local',
                    'name': 'GPT-2 (Local Fine-tuning)',
                    'provider': 'local',
                    'type': 'fine-tuned',
                    'status': 'available'
                }
            ]
        })
    
    @app.route('/api/v1/generate', methods=['POST'])
    def generate():
        return jsonify({
            'text': 'This is a demo response. Add API keys to enable real generation.',
            'metadata': {
                'model_type': 'demo',
                'provider': 'local',
                'tokens': 10
            }
        })
    
    @app.route('/')
    def index():
        return '''
        <h1>🤖 LLM Optimization Platform</h1>
        <p>Platform is running successfully!</p>
        <h3>Available Endpoints:</h3>
        <ul>
            <li><a href="/api/v1/health">Health Check</a></li>
            <li><a href="/api/v1/models">Available Models</a></li>
            <li>POST /api/v1/generate - Text Generation</li>
        </ul>
        <h3>Add API Keys:</h3>
        <p>To enable commercial models, add these to your environment:</p>
        <ul>
            <li><strong>Gemini:</strong> GEMINI_API_KEY (Get free key at <a href="https://makersuite.google.com/app/apikey">Google AI Studio</a>)</li>
            <li><strong>OpenAI:</strong> OPENAI_API_KEY</li>
            <li><strong>Anthropic:</strong> ANTHROPIC_API_KEY</li>
        </ul>
        '''
    
    return app

def main():
    """Main function."""
    print("🚀 LLM OPTIMIZATION PLATFORM - QUICK START")
    print("=" * 60)
    print("🔧 Starting minimal server...")
    print("📍 Available at: http://localhost:8080")
    print("🏥 Health check: http://localhost:8080/api/v1/health")
    print("🤖 Models: http://localhost:8080/api/v1/models")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        app = create_minimal_app()
        app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\\n\\n👋 Server stopped by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()