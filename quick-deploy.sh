#!/bin/bash

# Quick Local Deployment Script
# For testing and development

set -e

echo "🚀 Quick deployment for Bowling RAG API..."

# Check if .env exists
if [ ! -f "app/.env" ]; then
    echo "⚠️  app/.env not found. Copying from template..."
    cp .env.example app/.env
    echo "📝 Please edit app/.env with your configuration"
    echo "   nano app/.env"
    exit 1
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start
echo "🏗️  Building and starting services..."
docker-compose up -d --build

# Wait for startup
echo "⏳ Waiting for services to start..."
sleep 20

# Health check
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Service is healthy!"
    echo ""
    echo "📊 API Documentation: http://localhost:8000/docs"
    echo "❤️  Health Check: http://localhost:8000/health"
    echo "🔧 Readiness Check: http://localhost:8000/ready"
    echo ""
    echo "📝 Useful commands:"
    echo "  • View logs: docker-compose logs -f"
    echo "  • Stop: docker-compose down"
    echo "  • Restart: docker-compose restart"
else
    echo "❌ Service health check failed!"
    echo "📋 Checking logs..."
    docker-compose logs bowling-rag-api
    exit 1
fi

echo "🎉 Quick deployment completed successfully!"
