#!/bin/bash

# LLM Optimization Platform - Quick Start Script
# This script helps you get the platform running quickly

set -e

echo "🚀 LLM Optimization Platform - Quick Start"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys and passwords before continuing"
    echo "   Required: OPENAI_API_KEY, POSTGRES_PASSWORD, REDIS_PASSWORD, SECRET_KEY"
    read -p "Press Enter after updating .env file..."
fi

echo "✅ Environment file ready"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs models datasets backups
echo "✅ Directories created"

# Build and start services
echo "🔨 Building and starting services..."
echo "This may take a few minutes on first run..."

# Start core services first
docker-compose -f docker-compose.prod.yml up -d database redis

echo "⏳ Waiting for database to be ready..."
sleep 10

# Start backend
docker-compose -f docker-compose.prod.yml up -d backend

echo "⏳ Waiting for backend to be ready..."
sleep 15

# Start frontend and nginx
docker-compose -f docker-compose.prod.yml up -d frontend nginx

echo "✅ All services started!"
echo ""
echo "🌐 Your LLM Optimization Platform is now running:"
echo "   Frontend: http://localhost"
echo "   API: http://localhost/api/v1"
echo "   Health Check: http://localhost/health"
echo ""
echo "📊 Optional monitoring (run with --profile monitoring):"
echo "   Prometheus: http://localhost:9090"
echo "   Grafana: http://localhost:3001"
echo ""
echo "🔍 To view logs:"
echo "   docker-compose -f docker-compose.prod.yml logs -f"
echo ""
echo "🛑 To stop:"
echo "   docker-compose -f docker-compose.prod.yml down"

# Check if services are healthy
echo "🔍 Checking service health..."
sleep 5

if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Platform is healthy and ready to use!"
else
    echo "⚠️  Platform may still be starting up. Check logs if issues persist."
    echo "   Run: docker-compose -f docker-compose.prod.yml logs"
fi