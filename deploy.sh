#!/bin/bash

# Production deployment script for Bowling RAG API

set -e

echo "🚀 Starting Bowling RAG API deployment..."

# Run configuration check
echo "🔍 Checking configuration..."
python3 check_config.py
if [ $? -ne 0 ]; then
    echo "❌ Configuration check failed!"
    exit 1
fi

echo "✅ Configuration check passed!"

# Check if .env file exists
if [ ! -f "app/.env" ]; then
    echo "❌ Error: app/.env file not found!"
    echo "📝 Please copy .env.example to app/.env and configure your settings"
    exit 1
fi

# Check if service account file exists
if [ ! -f "app/service_account.json" ]; then
    echo "⚠️  Warning: app/service_account.json not found"
    echo "   This is required for Google Sheets integration"
fi

# Build and start services
echo "🏗️  Building Docker images..."
docker-compose build

echo "🐳 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Service is healthy!"
    echo "📊 API Documentation: http://localhost:8000/docs"
    echo "❤️  Health Check: http://localhost:8000/health"
    echo "🔧 Readiness Check: http://localhost:8000/ready"
else
    echo "❌ Service health check failed!"
    echo "📋 Checking logs..."
    docker-compose logs bowling-rag-api
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo ""
echo "📝 Useful commands:"
echo "  • View logs: docker-compose logs -f bowling-rag-api"
echo "  • Stop services: docker-compose down"
echo "  • Restart: docker-compose restart"
echo "  • Update: docker-compose pull && docker-compose up -d"
