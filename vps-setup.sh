#!/bin/bash

# VPS Setup Script for Bowling RAG API
# This script prepares a fresh Ubuntu server for deployment

set -e

echo "🚀 Starting VPS setup for Bowling RAG API..."

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
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

print_status "Installing required packages..."
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    htop \
    ufw \
    nginx \
    certbot \
    python3-certbot-nginx

# Install Docker
print_status "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    print_status "Docker installed successfully"
else
    print_status "Docker already installed"
fi

# Install Docker Compose
print_status "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    print_status "Docker Compose installed successfully"
else
    print_status "Docker Compose already installed"
fi

# Configure firewall
print_status "Configuring firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw status

# Create application directory
print_status "Creating application directory..."
mkdir -p ~/bowling-rag
cd ~/bowling-rag

# Create .env template
print_status "Creating environment template..."
cat > .env.example << 'EOF'
# GigaChat Configuration
GIGACHAT_AUTH_KEY=your_gigachat_auth_key_here
GIGACHAT_BASE=https://api.gigachat.sber.ru
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat

# Qdrant Configuration (Cloud)
QDRANT_URL=https://your-qdrant-url.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=bowling_knowledge

# Application Settings
TOP_K=5
MAX_QUERY_LEN=512
MAX_CONTEXT_LEN=2000
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2

# Graph API Secret
GRAPH_SECRET=your_graph_secret_here

# Google Sheets Integration (optional)
SERVICE_ACCOUNT_FILE=service_account.json
SPREADSHEET_ID=your_google_sheets_id_here
EOF

print_status "VPS setup completed successfully!"
print_warning "Next steps:"
echo "1. Configure your .env file: cp .env.example app/.env && nano app/.env"
echo "2. Clone your repository: git clone https://github.com/your-repo/bowling-rag.git ."
echo "3. Run deployment: ./deploy.sh"
echo "4. Configure Nginx: Follow DEPLOYMENT_GUIDE.md"
echo "5. Setup SSL: sudo certbot --nginx -d your-domain.com"

print_status "Setup complete! 🎉"