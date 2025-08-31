#!/usr/bin/env python3
"""
Quick test script to verify Gemini API integration
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_connection():
    """Test Gemini API connection"""
    print("🔍 Testing Gemini API connection...")
    
    # Check if API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    print(f"🔍 API key found: {api_key[:10]}..." if api_key else "❌ No API key found")
    
    if not api_key or api_key.startswith('paste_your'):
        print("❌ GEMINI_API_KEY not set or still using placeholder")
        return False
    
    print("✅ Gemini API key found")
    
    try:
        # Import and test Gemini client
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Test with a simple prompt
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'Hello from Gemini!' if you can read this.")
        
        print(f"✅ Gemini API test successful!")
        print(f"📝 Response: {response.text}")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

def test_platform_services():
    """Test if platform services can be imported"""
    print("\n🔍 Testing platform service imports...")
    
    try:
        # Test Gemini client import
        from api.services.gemini_client import GeminiClient
        print("✅ GeminiClient import successful")
        
        # Test other key services
        from api.services.text_generator import TextGenerator
        print("✅ TextGenerator import successful")
        
        from database.connection import DatabaseManager
        print("✅ DatabaseManager import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 LLM Optimization Platform - Gemini Integration Test")
    print("=" * 60)
    
    # Test Gemini connection
    gemini_ok = test_gemini_connection()
    
    # Test platform services
    services_ok = test_platform_services()
    
    print("\n" + "=" * 60)
    if gemini_ok and services_ok:
        print("✅ All tests passed! Your platform is ready to run.")
        print("\n🚀 Start the platform with:")
        print("   ./quick-start.sh  (for production)")
        print("   ./start-dev.sh    (for development)")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if not gemini_ok:
            print("💡 Make sure your GEMINI_API_KEY is correctly set in .env")
        if not services_ok:
            print("💡 Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()