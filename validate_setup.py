#!/usr/bin/env python3
"""
Simple validation script to check project setup without external dependencies.
"""

import os
import sys
from pathlib import Path


def check_directory_structure():
    """Check if all required directories exist."""
    required_dirs = [
        "fine_tuning",
        "api", 
        "evaluator",
        "web_interface",
        "tests",
        "config",
        "utils",
    ]
    
    print("Checking directory structure...")
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ (missing)")
            missing_dirs.append(directory)
    
    return len(missing_dirs) == 0


def check_required_files():
    """Check if all required files exist."""
    required_files = [
        "requirements.txt",
        ".env.example", 
        "main.py",
        "setup.py",
        "README.md",
        "config/settings.py",
        "utils/logging.py",
        "utils/exceptions.py",
        "utils/error_handler.py",
    ]
    
    print("\nChecking required files...")
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0


def check_python_version():
    """Check Python version compatibility."""
    print(f"\nChecking Python version...")
    version = sys.version_info
    
    if version >= (3, 8):
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_requirements_file():
    """Check if requirements.txt has expected dependencies."""
    print("\nChecking requirements.txt...")
    
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt not found")
        return False
    
    expected_deps = [
        "transformers",
        "torch", 
        "flask",
        "langchain",
        "pydantic",
        "python-dotenv",
    ]
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read().lower()
        
        missing_deps = []
        for dep in expected_deps:
            if dep.lower() in content:
                print(f"✓ {dep}")
            else:
                print(f"✗ {dep} (missing)")
                missing_deps.append(dep)
        
        return len(missing_deps) == 0
        
    except Exception as e:
        print(f"✗ Error reading requirements.txt: {e}")
        return False


def main():
    """Main validation function."""
    print("🔍 Validating LLM Optimization Platform Setup")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
        ("Dependencies", check_requirements_file),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"✗ Error in {check_name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Run: python setup.py")
        print("2. Activate virtual environment")
        print("3. Update .env file with your API keys")
        print("4. Start the application: python main.py api")
    else:
        print("❌ Some checks failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()