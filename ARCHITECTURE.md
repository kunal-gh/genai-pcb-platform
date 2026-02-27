# System Architecture

This document provides a detailed technical overview of the AI-Powered PCB Design Platform architecture.

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [ML Pipeline](#ml-pipeline)
- [Data Flow](#data-flow)
- [API Design](#api-design)
- [Database Schema](#database-schema)
- [Deployment](#deployment)

---

## Overview

The platform follows a microservices architecture with clear separation between:
- **Frontend**: React-based web interface
- **API Layer**: FastAPI REST API
- **ML Services**: PyTorch-based ML models
- **Background Workers**: Celery task queue
- **Data Layer**: PostgreSQL + Redis

### Key Design Principles

1. **Modularity**: Each component is independently deployable
2. **Scalability**: Horizontal scaling for ML inference and task processing
3. **Reliability**: Graceful degradation with fallback algorithms
4. **Observability**: Comprehensive logging and monitoring

---

## System Components

### 1. Frontend (React + TypeScript)

```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   ├── pages/           # Page-level components
│   ├── services/        # API client
│   └── types/           # TypeScript definitions
```

**Key Features**:
- Real-time design preview
- Interactive component placement
- Progress tracking for long-running tasks
- Gerber file visualization

### 2. API Layer (FastAPI)

```
src/api/
├── routes.py            # Main API endpoints
├── routing_routes.py    # Routing-specific endpoints
├── schemas.py           # Pydantic models
├── deps.py              # Dependency injection
└── auth.py              # Authentication
```

**Endpoints**:
- `POST /api/designs` - Create new design
- `GET /api/designs/{id}` - Get design status
- `POST /api/designs/{id}/route` - Trigger routing
- `GET /api/designs/{id}/gerber` - Download Gerber files

### 3. ML Services

#### FALCON GNN (`src/services/falcon_gnn.py`)

```python
class FALCONGraphNetwork:
    """
    Heterogeneous Graph Neural Network for PCB quality prediction
    
    Architecture:
    - Input: Heterogeneous graph (components, pins, nets, traces)
    - Processing: 4-layer GNN with attention
    - Output: Quality score [0, 1]
    """
```

**Graph Structure**:
```
Nodes:
- Component: {type, position, rotation, footprint}
- Pin: {number, position, net_id}
- Net: {name, priority, width}
- Via: {position, size, layers}
- Obstacle: {position, size, layer}

Edges:
- (Component, connects, Pin)
- (Pin, belongs, Net)
- (Net, routes, Trace)
- (Component, proximity, Component)
```

#### RL Routing Agent (`src/services/rl_routing_agent.py`)

```python
class RLRoutingAgent:
    """
    PPO-based routing agent
    
    State Space:
    - PCB grid representation (H x W x L)
    - Component positions
    - Existing traces
    - Current net to route
    
    Action Space:
    - Move direction: {N, S, E, W, UP, DOWN}
    - Place via: {True, False}
    - Layer change: {0, 1, ..., L-1}
    
    Reward Function:
    R = -α·wirelength - β·vias - γ·DRC_violations + δ·completion
    """
```

**Training Process**:
1. Generate synthetic PCB layouts
2. Initialize random component placements
3. Train agent to route nets optimally
4. Evaluate on held-out test set
5. Fine-tune on real-world designs

#### Hybrid Router (`src/services/routing_optimizer.py`)

```python
def route_net(net, pcb_state, algorithm='hybrid'):
    """
    Intelligent routing with automatic algorithm selection
    
    Decision Logic:
    - If complexity_score > threshold: Use RL agent
    - Else: Use A* pathfinding
    
    Complexity Factors:
    - Number of pins in net
    - Congestion in routing area
    - Number of obstacles
    - Layer constraints
    """
```

### 4. Background Workers (Celery)

```python
# Long-running tasks executed asynchronously
@celery.task
def generate_pcb_design(design_id, description):
    """
    Complete PCB generation pipeline:
    1. NLP: Parse description → circuit
    2. Placement: Optimize component positions
    3. Routing: Route all nets
    4. Verification: DRC/ERC/DFM
    5. Export: Generate Gerber files
    """
```

### 5. Data Layer

#### PostgreSQL Schema

```sql
-- Designs table
CREATE TABLE designs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    description TEXT,
    status VARCHAR(50),
    layers INTEGER,
    board_width FLOAT,
    board_height FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Components table
CREATE TABLE components (
    id UUID PRIMARY KEY,
    design_id UUID REFERENCES designs(id),
    type VARCHAR(100),
    value VARCHAR(100),
    footprint VARCHAR(100),
    position_x FLOAT,
    position_y FLOAT,
    rotation FLOAT
);

-- Nets table
CREATE TABLE nets (
    id UUID PRIMARY KEY,
    design_id UUID REFERENCES designs(id),
    name VARCHAR(100),
    pins JSONB,
    traces JSONB
);
```

#### Redis Usage

```python
# Job queue
redis.lpush('routing_queue', design_id)

# Caching
redis.setex(f'design:{design_id}', 3600, json.dumps(design_data))

# Real-time progress
redis.publish(f'progress:{design_id}', json.dumps({
    'stage': 'routing',
    'progress': 0.75
}))
```

---

## ML Pipeline

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Generation                           │
│  - Synthetic PCB layouts                                     │
│  - Component placement variations                            │
│  - Routing scenarios                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering                          │
│  - Graph construction                                        │
│  - Node/edge feature extraction                              │
│  - State representation                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────┐          ┌──────────────┐
│  FALCON GNN  │          │  RL Agent    │
│   Training   │          │  Training    │
│              │          │   (PPO)      │
│ - Supervised │          │ - Env setup  │
│ - Quality    │          │ - Reward     │
│   prediction │          │   shaping    │
└──────┬───────┘          └──────┬───────┘
       │                         │
       └────────────┬────────────┘
                    ▼
         ┌──────────────────┐
         │  Model Registry  │
         │  - Versioning    │
         │  - Checkpoints   │
         └──────────────────┘
```

### Inference Pipeline

```
User Input
    │
    ▼
┌─────────────────┐
│  NLP Service    │  ← LLM API (GPT-4/Claude)
│  (SKiDL Gen)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Circuit Graph   │
│  Construction   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Component       │  ← Optimization (Simulated Annealing)
│  Placement      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FALCON GNN      │  ← Quality Prediction
│  Inference      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Routing         │  ← RL Agent or A*
│  (Hybrid)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DRC/ERC/DFM     │  ← Validation Rules
│  Validation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Gerber Export   │  ← KiCad Integration
└─────────────────┘
```

---

## Data Flow

### Design Creation Flow

```
1. User submits description via frontend
   POST /api/designs
   {
     "description": "LED blinker with 555 timer",
     "layers": 2,
     "board_size": {"width": 50, "height": 50}
   }

2. API creates design record in PostgreSQL
   design_id = uuid.generate()
   status = "pending"

3. API queues background task
   celery.send_task('generate_pcb_design', [design_id])

4. Worker picks up task
   - Parse description with LLM
   - Generate SKiDL circuit
   - Place components
   - Route traces
   - Validate design
   - Export Gerber files

5. Worker updates design status
   status = "completed"
   files = ["gerber.zip", "drill.drl", ...]

6. Frontend polls for updates
   GET /api/designs/{design_id}/status
   → Returns current progress

7. User downloads files
   GET /api/designs/{design_id}/gerber
   → Returns ZIP archive
```

### Routing Flow

```
┌──────────────┐
│  Input Net   │
│  - Pins      │
│  - Priority  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Complexity       │
│  Analysis        │
│  - Pin count     │
│  - Congestion    │
│  - Obstacles     │
└──────┬───────────┘
       │
       ▼
    ┌──┴──┐
    │ > θ?│  Complexity threshold
    └──┬──┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│ RL  │ │ A*  │
│Agent│ │Path │
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       ▼
┌──────────────┐
│ DRC Check    │
│ - Clearance  │
│ - Width      │
└──────┬───────┘
       │
    ┌──┴──┐
    │Pass?│
    └──┬──┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│Done │ │Retry│
└─────┘ └─────┘
```

---

## API Design

### RESTful Endpoints

```
Designs:
  POST   /api/designs              Create new design
  GET    /api/designs              List all designs
  GET    /api/designs/{id}         Get design details
  PUT    /api/designs/{id}         Update design
  DELETE /api/designs/{id}         Delete design

Routing:
  POST   /api/designs/{id}/route   Trigger routing
  GET    /api/designs/{id}/routes  Get routing results

Files:
  GET    /api/designs/{id}/gerber  Download Gerber files
  GET    /api/designs/{id}/bom     Download BOM
  GET    /api/designs/{id}/preview Download preview image

Validation:
  POST   /api/designs/{id}/validate  Run DRC/ERC/DFM
  GET    /api/designs/{id}/violations Get validation results

Models:
  GET    /api/models                List available models
  POST   /api/models/train          Trigger model training
  GET    /api/models/{id}/metrics   Get training metrics
```

### WebSocket Endpoints

```
ws://api/designs/{id}/progress
  → Real-time progress updates

ws://api/designs/{id}/preview
  → Live design preview updates
```

---

## Database Schema

### Entity Relationship Diagram

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│    Users    │       │   Designs   │       │ Components  │
├─────────────┤       ├─────────────┤       ├─────────────┤
│ id (PK)     │───┐   │ id (PK)     │───┐   │ id (PK)     │
│ email       │   └──<│ user_id(FK) │   └──<│ design_id   │
│ password    │       │ description │       │ type        │
│ created_at  │       │ status      │       │ value       │
└─────────────┘       │ layers      │       │ position_x  │
                      │ board_width │       │ position_y  │
                      │ created_at  │       └─────────────┘
                      └─────────────┘
                            │
                            │
                      ┌─────┴─────┐
                      │           │
                      ▼           ▼
              ┌─────────────┐ ┌─────────────┐
              │    Nets     │ │   Files     │
              ├─────────────┤ ├─────────────┤
              │ id (PK)     │ │ id (PK)     │
              │ design_id   │ │ design_id   │
              │ name        │ │ filename    │
              │ pins        │ │ file_type   │
              │ traces      │ │ file_path   │
              └─────────────┘ └─────────────┘
```

---

## Deployment

### Docker Compose Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Network                          │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Frontend │  │   API    │  │  Worker  │  │ Postgres │  │
│  │  :3000   │  │  :8000   │  │          │  │  :5432   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │             │             │             │         │
│       └─────────────┴─────────────┴─────────────┘         │
│                          │                                 │
│                    ┌─────┴─────┐                          │
│                    │   Redis   │                          │
│                    │   :6379   │                          │
│                    └───────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Scaling Strategy

```
Production Deployment:

┌─────────────┐
│ Load        │
│ Balancer    │
│ (Nginx)     │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│API 1│ │API 2│  ← Horizontal scaling
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│  ← Task queue workers
└────────┘ └────────┘
       │
       ▼
┌──────────────┐
│ Redis Cluster│  ← Message broker
└──────────────┘
       │
       ▼
┌──────────────┐
│ PostgreSQL   │  ← Primary database
│ (Replicated) │
└──────────────┘
```

---

## Monitoring & Observability

### Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

routing_requests = Counter('routing_requests_total', 'Total routing requests')
routing_duration = Histogram('routing_duration_seconds', 'Routing duration')
active_designs = Gauge('active_designs', 'Number of active designs')
```

### Logging Strategy

```python
import logging

logger = logging.getLogger(__name__)

# Structured logging
logger.info("Routing started", extra={
    "design_id": design_id,
    "net_count": len(nets),
    "algorithm": "rl"
})
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": await check_database(),
        "redis": await check_redis(),
        "ml_models": await check_models()
    }
```

---

## Security Considerations

1. **Authentication**: JWT-based authentication
2. **Authorization**: Role-based access control (RBAC)
3. **Data Encryption**: At-rest and in-transit encryption
4. **Input Validation**: Pydantic schemas for all inputs
5. **Rate Limiting**: Per-user API rate limits
6. **CORS**: Configured for specific origins only

---

## Performance Optimization

1. **Caching**: Redis for frequently accessed data
2. **Database Indexing**: Optimized queries with proper indexes
3. **Async Processing**: FastAPI async endpoints
4. **Connection Pooling**: Database connection pooling
5. **Model Optimization**: TorchScript for faster inference
6. **CDN**: Static assets served via CDN

---

## Future Enhancements

1. **Multi-GPU Training**: Distributed training across multiple GPUs
2. **Model Versioning**: A/B testing for model improvements
3. **Real-time Collaboration**: Multiple users editing same design
4. **Cloud Deployment**: Kubernetes orchestration
5. **Advanced Visualization**: 3D PCB preview
6. **Cost Estimation**: Real-time manufacturing cost calculation
