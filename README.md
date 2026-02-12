# stuff-made-easy: GenAI PCB Design Platform

> Democratizing PCB design through natural language → manufacturable PCB pipeline

## 🚀 Vision

Transform natural language descriptions into verified, manufacturable PCB designs including schematics, netlists, PCB layouts, Gerber files, and 3D models. Enable fast prototyping, cheap iteration, and scale from hobbyist to industrial-grade designs.

## ✨ Key Features

- **Natural Language Input**: Describe your circuit in plain English
- **AI-Powered Generation**: LLM converts descriptions to executable SKiDL code
- **Complete Pipeline**: Schematic → Netlist → PCB Layout → Manufacturing files
- **Verification Loop**: Automated ERC/DRC checks and DFM validation
- **Simulation Support**: Electrical, thermal, and electromagnetic analysis
- **Manufacturing Ready**: Direct integration with PCB manufacturers

## 🏗️ Architecture

```
Natural Language → LLM Service → SKiDL Engine → KiCad Integration → Verification → Gerber Export
```

### Core Components
- **Frontend**: React + TypeScript web interface
- **API Gateway**: FastAPI backend with authentication
- **LLM Service**: OpenAI/Anthropic integration for code generation
- **SKiDL Engine**: Python-based schematic capture
- **KiCad Integration**: Automated PCB layout and file export
- **Verification Engine**: ERC/DRC/DFM validation
- **Simulation Suite**: PySpice + OpenEMS integration

## 🎯 MVP Goals (Phase 1)

- ✅ Natural language → structured JSON parsing
- ✅ SKiDL code generation via LLM
- ✅ KiCad netlist creation and Gerber export
- ✅ Basic ERC/DRC verification
- ✅ Web UI for prompt input and file download
- ✅ **Target**: ≥80% DFM pass rate for generated designs

## 🚦 Getting Started

### Prerequisites
- Python 3.10+
- KiCad 7.0+
- Docker & Docker Compose
- Node.js 18+ (for frontend)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/[username]/stuff-made-easy.git
cd stuff-made-easy

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install KiCad and SKiDL
pip install skidl

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration

# Run the development server
docker-compose up -d
python -m uvicorn src.main:app --reload
```

### Example Usage
```python
# Natural language input
prompt = "Design a 40x20mm PCB with a 9V battery, LED, and 220-ohm resistor"

# Generated output
- Schematic file (.sch)
- Netlist file (.net) 
- PCB layout (.kicad_pcb)
- Gerber files (manufacturing)
- Bill of Materials (BOM)
- 3D model (STEP file)
```

## 📁 Project Structure

```
stuff-made-easy/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── services/          # Core business logic
│   ├── models/            # Data models
│   └── utils/             # Utility functions
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── property/          # Property-based tests
├── frontend/              # React web application
├── docs/                  # Documentation
├── .kiro/                 # Kiro configuration
│   ├── specs/             # Feature specifications
│   └── steering/          # Development guidelines
└── docker/                # Docker configurations
```

## 🧪 Testing

We use a dual testing approach:

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Property-Based Tests
```bash
pytest tests/property/ -v --hypothesis-show-statistics
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

## 📊 Success Metrics

- **Quality**: ≥90% of simulated devices pass functional tests
- **Performance**: Prompt → Gerber files in ≤60 seconds
- **Manufacturability**: ≥80% DFM pass rate
- **User Experience**: NPS ≥7 from beta users

## 🛣️ Roadmap

### Phase 1 - MVP (0-3 months)
- [x] Natural language processing
- [x] SKiDL code generation
- [x] Basic KiCad integration
- [x] Web UI and file downloads
- [ ] Beta testing with 10 users

### Phase 2 - Scale (3-12 months)
- [ ] Multi-layer board support
- [ ] ML-based placement/routing
- [ ] Supplier integration (BOM pricing)
- [ ] One-click prototype ordering

### Phase 3 - Enterprise (12-36 months)
- [ ] High-speed/RF design support
- [ ] Altium plugin integration
- [ ] Certification workflows
- [ ] Private model fine-tuning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SKiDL](https://github.com/devbisme/skidl) - Schematic capture in Python
- [KiCad](https://kicad.org/) - Open-source EDA suite
- [PySpice](https://github.com/FabriceSalvaire/PySpice) - Circuit simulation
- [OpenEMS](https://openems.de/) - Electromagnetic simulation

## 📞 Support

- 📧 Email: support@stuff-made-easy.com
- 💬 Discord: [Join our community](https://discord.gg/stuff-made-easy)
- 📖 Documentation: [docs.stuff-made-easy.com](https://docs.stuff-made-easy.com)
- 🐛 Issues: [GitHub Issues](https://github.com/[username]/stuff-made-easy/issues)

---

**Made with ❤️ by the stuff-made-easy team**