#!/bin/bash

# Production startup script for containerized deployment

set -e

echo "🚀 Starting LLM Optimization Platform in Production Mode"

# Initialize database if needed
python3 -c "
from database.connection import db_manager
try:
    db_manager.initialize()
    print('✅ Database initialized')
except Exception as e:
    print(f'⚠️  Database initialization: {e}')
"

# Start the application
echo "🌐 Starting API server on port ${PORT:-8080}"

# Use gunicorn for production
if command -v gunicorn &> /dev/null; then
    exec gunicorn \
        --bind 0.0.0.0:${PORT:-8080} \
        --workers 4 \
        --worker-class gevent \
        --worker-connections 1000 \
        --timeout 120 \
        --keepalive 5 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --access-logfile - \
        --error-logfile - \
        --log-level info \
        "api.app:create_app()"
else
    # Fallback to Flask development server
    exec python3 run_api.py
fi