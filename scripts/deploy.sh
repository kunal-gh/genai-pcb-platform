#!/bin/bash

# Production Deployment Script for Stuff-made-easy

set -e

echo "========================================="
echo "Stuff-made-easy Production Deployment"
echo "========================================="
echo ""

# Check if .env.production exists
if [ ! -f .env.production ]; then
    echo "Error: .env.production file not found!"
    echo "Please create it from .env.production.example"
    exit 1
fi

# Load environment variables
export $(cat .env.production | grep -v '^#' | xargs)

echo "Step 1: Pulling latest code..."
git pull origin main

echo ""
echo "Step 2: Building Docker images..."
docker-compose -f docker-compose.prod.yml build --no-cache

echo ""
echo "Step 3: Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

echo ""
echo "Step 4: Starting services..."
docker-compose -f docker-compose.prod.yml up -d

echo ""
echo "Step 5: Waiting for services to be healthy..."
sleep 10

echo ""
echo "Step 6: Running database migrations..."
docker-compose -f docker-compose.prod.yml exec -T app \
    python -c "from src.models.database import init_db; init_db()"

echo ""
echo "Step 7: Checking service health..."
docker-compose -f docker-compose.prod.yml ps

echo ""
echo "Step 8: Testing health endpoint..."
sleep 5
curl -f http://localhost:8000/health || echo "Warning: Health check failed"

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Services running:"
echo "  - Application: http://localhost:8000"
echo "  - Nginx: http://localhost:80"
echo "  - Database: localhost:5432"
echo "  - Redis: localhost:6379"
echo ""
echo "View logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "Stop services: docker-compose -f docker-compose.prod.yml down"
echo ""
