#!/bin/bash

# Bowling RAG API - VPS Setup Script
# This script sets up a Ubuntu/Debian VPS for production deployment

set -e

echo "🚀 Bowling RAG API - VPS Setup"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Update system
update_system() {
    print_step "Updating system packages..."
    apt-get update
    apt-get upgrade -y
    apt-get autoremove -y
    print_status "System updated"
}

# Install required packages
install_packages() {
    print_step "Installing required packages..."
    apt-get install -y \
        curl \
        wget \
        git \
        ufw \
        fail2ban \
        nginx \
        certbot \
        python3-certbot-nginx \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release

    print_status "Packages installed"
}

# Install Docker
install_docker() {
    print_step "Installing Docker..."

    # Remove old versions
    apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

    # Start and enable Docker
    systemctl start docker
    systemctl enable docker

    print_status "Docker installed and started"
}

# Configure firewall
configure_firewall() {
    print_step "Configuring firewall..."

    # Reset UFW
    ufw --force reset

    # Allow SSH
    ufw allow ssh
    ufw allow 22

    # Allow HTTP and HTTPS
    ufw allow 80
    ufw allow 443

    # Enable firewall
    echo "y" | ufw enable

    print_status "Firewall configured"
}

# Create application directory
create_app_directory() {
    print_step "Creating application directory..."

    mkdir -p /opt/bowling-rag
    mkdir -p /opt/bowling-rag/ssl
    mkdir -p /opt/bowling-rag/logs

    print_status "Application directory created"
}

# Configure nginx
configure_nginx() {
    print_step "Configuring nginx..."

    # Backup default config
    cp /etc/nginx/sites-available/default /etc/nginx/sites-available/default.backup

    # Create bowling-rag site config
    cat > /etc/nginx/sites-available/bowling-rag << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Static files
    location /static/ {
        alias /opt/bowling-rag/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Logs
    access_log /opt/bowling-rag/logs/nginx.access.log;
    error_log /opt/bowling-rag/logs/nginx.error.log;
}
EOF

    # Enable site
    ln -sf /etc/nginx/sites-available/bowling-rag /etc/nginx/sites-enabled/
    rm /etc/nginx/sites-enabled/default

    # Test nginx configuration
    nginx -t

    # Restart nginx
    systemctl restart nginx
    systemctl enable nginx

    print_status "Nginx configured"
}

# Setup SSL (optional)
setup_ssl() {
    print_step "Setting up SSL certificate..."

    read -p "Enter your domain name (or press Enter to skip SSL setup): " domain

    if [ -n "$domain" ]; then
        print_status "Setting up SSL for $domain"

        # Stop nginx temporarily
        systemctl stop nginx

        # Get SSL certificate
        certbot certonly --standalone -d $domain

        # Configure nginx with SSL
        cat > /etc/nginx/sites-available/bowling-rag << EOF
server {
    listen 80;
    server_name $domain;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name $domain;

    ssl_certificate /etc/letsencrypt/live/$domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$domain/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Static files
    location /static/ {
        alias /opt/bowling-rag/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Logs
    access_log /opt/bowling-rag/logs/nginx.access.log;
    error_log /opt/bowling-rag/logs/nginx.error.log;
}
EOF

        # Start nginx
        systemctl start nginx

        # Setup certbot renewal
        (crontab -l ; echo "0 12 * * * /usr/bin/certbot renew --quiet") | crontab -

        print_status "SSL configured for $domain"
    else
        print_warning "SSL setup skipped"
    fi
}

# Create deployment user
create_deploy_user() {
    print_step "Creating deployment user..."

    useradd -m -s /bin/bash deploy || true
    usermod -aG docker deploy

    # Set password
    echo "deploy:deploy_password_change_me" | chpasswd

    print_status "Deployment user created (password: deploy_password_change_me)"
    print_warning "Remember to change the default password!"
}

# Print completion message
print_completion() {
    echo ""
    print_status "🎉 VPS setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Change the deploy user password:"
    echo "   sudo passwd deploy"
    echo ""
    echo "2. Upload your application:"
    echo "   scp -r /path/to/bowling-rag deploy@your-vps:/opt/"
    echo ""
    echo "3. Configure environment variables:"
    echo "   nano /opt/bowling-rag/app/.env"
    echo ""
    echo "4. Deploy the application:"
    echo "   cd /opt/bowling-rag && ./deploy.sh"
    echo ""
    echo "🔒 Security reminders:"
    echo "- Change default passwords"
    echo "- Configure SSH key authentication"
    echo "- Keep system updated"
    echo ""
    echo "🌐 Your API will be available at:"
    echo "   http://your-vps-ip"
    if [ -n "$domain" ]; then
        echo "   https://$domain"
    fi
}

# Main setup process
main() {
    check_root
    update_system
    install_packages
    install_docker
    configure_firewall
    create_app_directory
    configure_nginx
    setup_ssl
    create_deploy_user
    print_completion
}

# Run main function
main
