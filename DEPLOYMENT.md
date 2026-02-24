# Production Deployment Guide

## Prerequisites

- Docker and Docker Compose installed
- Domain name configured (optional but recommended)
- SSL certificates (Let's Encrypt recommended)
- Minimum 4GB RAM, 2 CPU cores
- 20GB disk space

## Quick Deployment

### 1. Clone Repository

```bash
git clone https://github.com/kunal-gh/genai-pcb-platform.git
cd genai-pcb-platform
```

### 2. Configure Environment

```bash
cp .env.production.example .env.production
nano .env.production  # Edit with your values
```

### 3. Deploy

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

## Manual Deployment Steps

### 1. Environment Setup

Create `.env.production` with these required variables:

```env
SECRET_KEY=your-secret-key
POSTGRES_PASSWORD=strong-password
REDIS_PASSWORD=strong-password
OPENAI_API_KEY=your-api-key
```

### 2. SSL Certificates

For production, obtain SSL certificates:

```bash
# Using Let's Encrypt
sudo certbot certonly --standalone -d yourdomain.com
```

Copy certificates to `docker/nginx/ssl/`:

```bash
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem docker/nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem docker/nginx/ssl/key.pem
```

### 3. Build and Start

```bash
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Initialize Database

```bash
docker-compose -f docker-compose.prod.yml exec app \
  python -c "from src.models.database import init_db; init_db()"
```

## Verification

Check all services are running:

```bash
docker-compose -f docker-compose.prod.yml ps
```

Test health endpoint:

```bash
curl http://localhost:8000/health
```

## Monitoring

View logs:

```bash
docker-compose -f docker-compose.prod.yml logs -f
```

## Backup

### Database Backup

```bash
docker-compose -f docker-compose.prod.yml exec postgres \
  pg_dump -U stuffmadeeasy stuffmadeeasy_prod > backup.sql
```

### Restore Database

```bash
cat backup.sql | docker-compose -f docker-compose.prod.yml exec -T postgres \
  psql -U stuffmadeeasy stuffmadeeasy_prod
```

## Scaling

Scale application instances:

```bash
docker-compose -f docker-compose.prod.yml up -d --scale app=3
```

## Troubleshooting

### Check logs
```bash
docker-compose -f docker-compose.prod.yml logs app
```

### Restart services
```bash
docker-compose -f docker-compose.prod.yml restart
```

### Clean restart
```bash
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d
```

## Security Checklist

- [ ] Change all default passwords
- [ ] Configure SSL certificates
- [ ] Set up firewall rules
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Set up monitoring alerts
- [ ] Regular backups scheduled
- [ ] Update dependencies regularly

## Cloud Deployment

### AWS

Use AWS ECS or EC2 with the provided Docker setup.

### Azure

Deploy using Azure Container Instances or App Service.

### Google Cloud

Use Google Cloud Run or GKE with the Docker configuration.

## Support

For issues, check:
- GitHub Issues
- Documentation
- Logs: `docker-compose logs`
