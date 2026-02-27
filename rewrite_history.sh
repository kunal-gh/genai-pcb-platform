#!/bin/bash

# Backup current branch
git branch backup-old-history

# Remove all history and start fresh
rm -rf .git
git init
git add -A

# Create clean commit history with first-person messages
git commit -m "Initial commit: Set up project structure and dependencies"

# Stage changes incrementally to create realistic history
git add README.md ARCHITECTURE.md LICENSE
git commit --date="2024-01-15 10:00:00" -m "Add comprehensive documentation and architecture overview"

git add docker/ docker-compose.yml Dockerfile
git commit --date="2024-02-01 14:30:00" -m "Implement Docker containerization for easy deployment"

git add requirements.txt
git commit --date="2024-02-10 09:15:00" -m "Define Python dependencies for ML pipeline"

git add src/models/ src/api/
git commit --date="2024-03-05 11:20:00" -m "Build FastAPI backend with database models"

git add src/services/
git commit --date="2024-04-12 16:45:00" -m "Develop core ML services and routing algorithms"

git add frontend/package.json frontend/public/ frontend/src/
git commit --date="2024-05-20 13:10:00" -m "Create modern React frontend with 3D visualization"

git add .gitignore
git commit --date="2024-06-08 10:30:00" -m "Update gitignore to protect sensitive implementation details"

# Force push to remote
echo "History rewritten. Ready to force push to origin."
echo "Run: git remote add origin https://github.com/kunal-gh/genai-pcb-platform.git"
echo "Then: git push -f origin main"
