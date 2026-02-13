"""
Component database models.

SQLAlchemy models for electronic components, manufacturers, and categories.
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Boolean, JSON, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import uuid
import enum

from .database import Base


class ComponentCategory(str, enum.Enum):
    """Component category enumeration."""
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    DIODE = "diode"
    LED = "led"
    TRANSISTOR = "transistor"
    IC = "ic"
    CONNECTOR = "connector"
    SWITCH = "switch"
    RELAY = "relay"
    CRYSTAL = "crystal"
    FUSE = "fuse"
    TRANSFORMER = "transformer"
    SENSOR = "sensor"
    OTHER = "other"


class PackageType(str, enum.Enum):
    """Component package type enumeration."""
    SMD = "smd"
    THROUGH_HOLE = "through_hole"
    BGA = "bga"
    QFP = "qfp"
    SOIC = "soic"
    TSSOP = "tssop"
    DIP = "dip"
    TO220 = "to220"
    SOT23 = "sot23"
    OTHER = "other"


class Manufacturer(Base):
    """
    Component manufacturer model.
    
    Stores information about component manufacturers.
    """
    __tablename__ = "manufacturers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    website = Column(String(500))
    description = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    components = relationship("Component", back_populates="manufacturer")
    
    def __repr__(self):
        return f"<Manufacturer(id={self.id}, name='{self.name}')>"


class Component(Base):
    """
    Electronic component model.
    
    Stores detailed information about electronic components including
    electrical parameters, package types, and availability.
    """
    __tablename__ = "components"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic information
    part_number = Column(String(255), nullable=False, unique=True, index=True)
    manufacturer_id = Column(UUID(as_uuid=True), ForeignKey("manufacturers.id"), nullable=False)
    category = Column(Enum(ComponentCategory), nullable=False, index=True)
    
    description = Column(Text)
    datasheet_url = Column(String(500))
    
    # Package information
    package_type = Column(Enum(PackageType), nullable=False)
    package_name = Column(String(100))  # e.g., "0805", "SOT-23-3"
    
    # Electrical parameters (JSON for flexibility)
    electrical_parameters = Column(JSON, nullable=False)
    # Example structure:
    # {
    #   "resistance": {"value": 10000, "unit": "ohm", "tolerance": 0.01},
    #   "power_rating": {"value": 0.125, "unit": "W"},
    #   "voltage_rating": {"value": 50, "unit": "V"},
    #   "capacitance": {"value": 100e-9, "unit": "F"},
    #   "temperature_coefficient": {"value": 100, "unit": "ppm/C"}
    # }
    
    # Footprint and symbol references
    footprint_id = Column(String(255))  # KiCad footprint library reference
    symbol_id = Column(String(255))  # KiCad symbol library reference
    
    # Availability and pricing
    in_stock = Column(Boolean, default=True)
    lifecycle_status = Column(String(50))  # "active", "obsolete", "nrnd"
    min_order_quantity = Column(Integer, default=1)
    
    # Pricing tiers (JSON array)
    pricing = Column(JSON)
    # Example: [{"quantity": 1, "price": 0.10}, {"quantity": 100, "price": 0.05}]
    
    # Supplier information
    suppliers = Column(JSON)
    # Example: [{"name": "DigiKey", "sku": "123-456", "url": "..."}]
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_verified_at = Column(DateTime)
    
    # Relationships
    manufacturer = relationship("Manufacturer", back_populates="components")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_component_category_package', 'category', 'package_type'),
        Index('idx_component_lifecycle', 'lifecycle_status', 'in_stock'),
    )
    
    def __repr__(self):
        return f"<Component(id={self.id}, part_number='{self.part_number}', category={self.category})>"
    
    def get_parameter(self, param_name: str):
        """
        Get electrical parameter value.
        
        Args:
            param_name: Parameter name (e.g., "resistance", "capacitance")
            
        Returns:
            Parameter dict or None if not found
        """
        if not self.electrical_parameters:
            return None
        return self.electrical_parameters.get(param_name)
    
    def get_price_for_quantity(self, quantity: int) -> float:
        """
        Get unit price for given quantity.
        
        Args:
            quantity: Order quantity
            
        Returns:
            Unit price or 0.0 if pricing not available
        """
        if not self.pricing:
            return 0.0
        
        # Find applicable price tier
        applicable_price = 0.0
        for tier in sorted(self.pricing, key=lambda x: x.get('quantity', 0)):
            if quantity >= tier.get('quantity', 0):
                applicable_price = tier.get('price', 0.0)
            else:
                break
        
        return applicable_price
