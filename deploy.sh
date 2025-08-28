#!/bin/bash

# Bowling RAG API - Local Deployment Script

set -e

echo "🚀 Bowling RAG API - Local Deployment"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_status "Docker is running"
}

# Check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose >/dev/null 2>&1; then
        print_error "docker-compose is not installed"
        exit 1
    fi
    print_status "docker-compose is available"
}

# Check configuration
check_config() {
    if [ ! -f "app/.env" ]; then
        print_warning "app/.env not found. Copying from .env.example..."
        cp .env.example app/.env
        print_warning "Please edit app/.env with your actual configuration"
    fi

    if [ ! -f "app/service_account.json" ]; then
        print_warning "app/service_account.json not found"
        print_warning "This is required for Google Sheets integration"
    fi
}

# Build and start services
deploy() {
    print_status "Building Docker images..."
    docker-compose build

    print_status "Starting services..."
    docker-compose up -d

    print_status "Waiting for services to be ready..."
    sleep 10

    print_status "Checking service health..."
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_status "✅ Deployment successful!"
        echo ""
        echo "🌐 API available at: http://localhost:8000"
        echo "💚 Health check: http://localhost:8000/health"
        echo "📖 API docs: http://localhost:8000/docs"
        echo ""
        echo "To view logs: docker-compose logs -f"
        echo "To stop: docker-compose down"
    else
        print_error "Health check failed. Check logs:"
        docker-compose logs bowling-rag-api
        exit 1
    fi
}

# Main deployment process
main() {
    print_status "Starting deployment process..."

    check_docker
    check_docker_compose
    check_config
    deploy

    print_status "🎉 Deployment completed successfully!"
}

# Handle script interruption
trap 'print_error "Deployment interrupted by user"; exit 1' INT TERM

# Run main function
main
