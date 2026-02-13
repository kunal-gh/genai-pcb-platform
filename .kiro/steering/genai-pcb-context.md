---
inclusion: auto
description: Core project context and overview for the GenAI PCB Design Platform
---

# GenAI PCB Design Platform Context

## Project Overview

**stuff-made-easy** is a cloud-first GenAI platform that democratizes PCB design by converting natural language descriptions into verified, manufacturable PCB designs. The platform generates complete design artifacts including schematics, netlists, PCB layouts, Gerber files, and 3D models with integrated simulation support.

## Core Vision

Democratize PCB design — let humans describe an electronic idea in plain language and receive a manufacturable, tested PCB design including schematic, netlist, PCB layout and optional 3D model; enable fast prototyping, cheap iteration, and scale to industrial-grade designs.

## Key Differentiators

1. **Natural Language → SKiDL/EDA Code Pipeline**: LLM outputs executable code rather than just descriptions
2. **Tight Verification Loop**: LLM → rule engine → SPICE/DFM/EM feedback → LLM revision
3. **Modular Architecture**: AI layer, EDA engine layer, simulation microservices, web UI, manufacturer APIs
4. **Open-by-Default**: KiCad, SKiDL, PySpice, OpenEMS with optional commercial plugins

## Technology Stack

- **Frontend**: React + TypeScript, Material-UI, Three.js for 3D preview
- **Backend**: FastAPI with Python, PostgreSQL, Redis
- **AI/ML**: LLM integration (OpenAI/Anthropic), SKiDL for schematic generation
- **EDA Tools**: KiCad Python API, SKiDL library
- **Simulation**: PySpice (electrical), OpenEMS (electromagnetic), thermal analysis
- **Infrastructure**: Docker + Kubernetes, cloud storage (S3-compatible)

## Phase 1 MVP Goals

- Natural language prompt → structured JSON → SKiDL code → KiCad netlist → Gerber files
- ≥80% DFM pass rate for generated designs
- Basic ERC/DRC verification
- Simple web UI for prompt input and file download
- Target: 10-60 second processing time for simple designs

## Target Users

- **Phase 1**: DIY makers, hobbyists, students, educators (Arduino/Raspberry Pi community)
- **Phase 2**: Hardware startups & small teams for IoT prototypes  
- **Phase 3**: Enterprise engineering teams (automotive, medical, telecom)

## Success Metrics

- **Quality**: ≥90% of simulated devices pass basic functional tests
- **Performance**: Average time from prompt → downloadable Gerber ≤ 10 minutes
- **Adoption**: 100 beta users within first 3 months; NPS ≥ 7
- **Manufacturability**: ≥80% of generated Gerbers pass automated DFM checks

## Development Standards

- **Testing**: Dual approach with unit tests + property-based tests using Hypothesis
- **Code Quality**: Type hints, comprehensive docstrings, automated linting
- **Security**: JWT authentication, data encryption, audit logging
- **Performance**: Async processing, caching, horizontal scaling
- **Documentation**: API docs, user guides, technical specifications

## File References

- Spec: `.kiro/specs/genai-pcb-platform/`
- Requirements: `#[[file:.kiro/specs/genai-pcb-platform/requirements.md]]`
- Design: `#[[file:.kiro/specs/genai-pcb-platform/design.md]]`
- Tasks: `#[[file:.kiro/specs/genai-pcb-platform/tasks.md]]`