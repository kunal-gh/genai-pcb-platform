"""
Design project data models.

SQLAlchemy models for PCB design projects, files, and metadata.
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Boolean, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import uuid
import enum

from .database import Base


class DesignStatus(str, enum.Enum):
    """Design project status enumeration."""
    DRAFT = "draft"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class FileType(str, enum.Enum):
    """Design file type enumeration."""
    SCHEMATIC = "schematic"
    NETLIST = "netlist"
    PCB_LAYOUT = "pcb_layout"
    GERBER = "gerber"
    DRILL = "drill"
    STEP = "step"
    BOM = "bom"
    SIMULATION = "simulation"
    DOCUMENTATION = "documentation"


class DesignProject(Base):
    """
    Main design project model.
    
    Represents a complete PCB design from natural language prompt
    through to manufacturable files.
    """
    __tablename__ = "design_projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Natural language input
    natural_language_prompt = Column(Text, nullable=False)
    
    # Structured requirements (JSON)
    structured_requirements = Column(JSON)
    
    # RAG context (retrieved documents)
    rag_context = Column(JSON)
    
    # Generated code
    skidl_code = Column(Text)
    
    # Analog topologies (AnalogGenie results)
    analog_topologies = Column(JSON)
    
    # Optimized circuit (CircuitVAE results)
    optimized_circuit = Column(JSON)
    
    # Status and metadata
    status = Column(Enum(DesignStatus), default=DesignStatus.DRAFT, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    branch = Column(String(100), default="main")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    files = relationship("DesignFile", back_populates="project", cascade="all, delete-orphan")
    verification_results = relationship("VerificationResult", back_populates="project", uselist=False)
    simulation_results = relationship("SimulationResult", back_populates="project", uselist=False)
    
    def __repr__(self):
        return f"<DesignProject(id={self.id}, name='{self.name}', status={self.status})>"


class DesignFile(Base):
    """
    Design artifact file model.
    
    Stores metadata for generated files (schematics, Gerbers, etc.).
    """
    __tablename__ = "design_files"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("design_projects.id"), nullable=False)
    
    file_type = Column(Enum(FileType), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)  # bytes
    checksum = Column(String(64))  # SHA-256
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("DesignProject", back_populates="files")
    
    def __repr__(self):
        return f"<DesignFile(id={self.id}, type={self.file_type}, path='{self.file_path}')>"


class VerificationResult(Base):
    """
    Design verification results model.
    
    Stores ERC, DRC, DFM, and security analysis results.
    """
    __tablename__ = "verification_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("design_projects.id"), nullable=False, unique=True)
    
    # Verification results (JSON arrays)
    erc_results = Column(JSON)  # List of ERC violations
    drc_results = Column(JSON)  # List of DRC violations
    dfm_results = Column(JSON)  # List of DFM violations
    
    # Scores
    manufacturability_score = Column(Float)
    dft_coverage = Column(Float)  # Design-for-test coverage
    signal_integrity_score = Column(Float)
    thermal_score = Column(Float)
    sustainability_score = Column(Float)
    
    # Overall status
    overall_status = Column(String(50))
    
    # Recommendations
    recommendations = Column(JSON)  # List of strings
    
    # Security analysis
    security_analysis = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("DesignProject", back_populates="verification_results")
    
    def __repr__(self):
        return f"<VerificationResult(project_id={self.project_id}, score={self.manufacturability_score})>"


class SimulationResult(Base):
    """
    Simulation results model.
    
    Stores SPICE, EM, and thermal simulation results.
    """
    __tablename__ = "simulation_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("design_projects.id"), nullable=False, unique=True)
    
    # ML surrogate results
    insight_results = Column(JSON)  # INSIGHT neural SPICE
    ml_surrogate_results = Column(JSON)
    
    # Full SPICE validation
    full_spice_results = Column(JSON)
    
    # EM and thermal
    em_simulation_results = Column(JSON)  # OpenEMS
    thermal_simulation_results = Column(JSON)  # ElmerFEM
    
    # Confidence scores
    confidence_scores = Column(JSON)
    
    # Validation status
    validation_status = Column(String(50))
    
    # Waveforms and S-parameters (stored as file references)
    waveform_files = Column(ARRAY(String))
    s_parameter_files = Column(ARRAY(String))
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("DesignProject", back_populates="simulation_results")
    
    def __repr__(self):
        return f"<SimulationResult(project_id={self.project_id}, status={self.validation_status})>"