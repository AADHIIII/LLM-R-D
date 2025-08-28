"""
Setup script for LLM optimization platform.
Handles virtual environment creation and dependency installation.
"""

import os
import subprocess
import sys
import venv
from pathlib import Path


def create_virtual_environment(venv_path: str = "venv"):
    """Create a Python virtual environment."""
    print(f"Creating virtual environment at {venv_path}...")
    
    try:
        venv.create(venv_path, with_pip=True)
        print(f"✓ Virtual environment created at {venv_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False


def get_pip_command(venv_path: str = "venv"):
    """Get the pip command for the virtual environment."""
    if os.name == 'nt':  # Windows
        return os.path.join(venv_path, "Scripts", "pip")
    else:  # Unix/Linux/macOS
        return os.path.join(venv_path, "bin", "pip")


def install_dependencies(venv_path: str = "venv"):
    """Install dependencies from requirements.txt."""
    pip_cmd = get_pip_command(venv_path)
    
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt not found")
        return False
    
    print("Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    if os.path.exists(".env"):
        print("✓ .env file already exists")
        return True
    
    if not os.path.exists(".env.example"):
        print("✗ .env.example not found")
        return False
    
    try:
        with open(".env.example", "r") as src:
            content = src.read()
        
        with open(".env", "w") as dst:
            dst.write(content)
        
        print("✓ Created .env file from .env.example")
        print("  Please update .env with your actual API keys and configuration")
        return True
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "datasets", 
        "logs",
        "data",
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def main():
    """Main setup function."""
    print("🚀 Setting up LLM Optimization Platform...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("2. Update .env file with your API keys")
    print("3. Run the application:")
    print("   python main.py api")
    print("   or")
    print("   python main.py web")


if __name__ == "__main__":
    main()