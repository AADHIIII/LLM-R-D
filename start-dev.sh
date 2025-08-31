#!/bin/bash

# LLM Optimization Platform - Development Mode
# Starts frontend and backend separately for development

set -e

echo "🛠️  LLM Optimization Platform - Development Mode"
echo "==============================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Setup backend
echo "🔧 Setting up backend..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file - please update with your API keys"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Setup frontend
echo "🔧 Setting up frontend..."
cd web_interface/frontend

if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Create frontend .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "REACT_APP_API_URL=http://localhost:5000/api/v1" > .env
    echo "📝 Created frontend .env file"
fi

cd ../..

# Start services
echo "🚀 Starting services..."

# Start backend in background
echo "Starting backend on http://localhost:5000..."
python3 start_local.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend in background
echo "Starting frontend on http://localhost:3000..."
cd web_interface/frontend
npm start &
FRONTEND_PID=$!

cd ../..

echo ""
echo "✅ Development environment is running!"
echo ""
echo "🌐 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:5000/api/v1"
echo "   API Health: http://localhost:5000/api/v1/health"
echo ""
echo "🔍 Logs are displayed in the terminal"
echo "🛑 Press Ctrl+C to stop both services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ Services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait