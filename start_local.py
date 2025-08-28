#!/usr/bin/env python3
"""
Quick start script for local development without Docker.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Run from project root.")
        return False
    
    print("✅ Project structure verified")
    return True

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        
        # Install additional dependencies we found missing
        additional_deps = ["python-magic", "PyJWT"]
        for dep in additional_deps:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
                print(f"✅ Installed {dep}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Could not install {dep} (may already be installed)")
        
        print("✅ Dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment variables."""
    print("\n🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# LLM Optimization Platform - Local Development
FLASK_ENV=development
FLASK_DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production
API_HOST=0.0.0.0
API_PORT=5000
DATABASE_URL=sqlite:///llm_optimization.db
LOG_LEVEL=INFO

# Add your API keys here for full functionality
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
"""
        env_file.write_text(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file exists")
    
    # Create necessary directories
    directories = ["logs", "models", "datasets"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/ directory")

def start_application():
    """Start the application."""
    print("\n🚀 Starting LLM Optimization Platform...")
    print("=" * 60)
    
    # Set environment variables
    os.environ["PYTHONPATH"] = "."
    
    try:
        # Start the API server
        print("Starting API Gateway on http://localhost:5000")
        print("\nAvailable endpoints:")
        print("  - GET  http://localhost:5000/api/v1/health")
        print("  - GET  http://localhost:5000/api/v1/models")
        print("  - POST http://localhost:5000/api/v1/generate")
        print("  - GET  http://localhost:5000/api/v1/monitoring/metrics")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the API server
        subprocess.run([sys.executable, "run_api.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Application failed to start: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🚀 LLM Optimization Platform - Local Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Start application
    if not start_application():
        sys.exit(1)

if __name__ == "__main__":
    main()