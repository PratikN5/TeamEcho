# models/database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON, Enum
from app.models.db import Base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum

# Enums
class DocumentStatus(enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"

class DocumentSource(enum.Enum):
    UPLOAD = "upload"
    SHAREPOINT = "sharepoint"
    GITHUB = "github"
    CONFLUENCE = "confluence"
    S3 = "s3"

class DiscrepancyType(enum.Enum):
    OUTDATED = "outdated"
    MISSING = "missing"
    INCONSISTENT = "inconsistent"
    UNDOCUMENTED_CODE = "undocumented_code"

class FlagStatus(enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"

class RuleSeverity(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Models
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    source = Column(Enum(DocumentSource), nullable=False)
    s3_key = Column(String(500), nullable=True)  # S3 object key
    file_size = Column(Integer)  # in bytes
    mime_type = Column(String(100))
    upload_date = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.UPLOADED)
    uploaded_by = Column(String(100))  # user identifier
    checksum = Column(String(64))  # SHA-256 hash for integrity
    s3_version_id = Column(String(100), nullable=True)  # Add this line
    text = Column(Text, nullable=True)
    
    # Relationships
    metadata_entries = relationship("DocumentMetadata", back_populates="document", cascade="all, delete-orphan")
    discrepancy_flags = relationship("DiscrepancyFlag", back_populates="document")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text)
    value_type = Column(String(50), default="string")  # string, number, boolean, json
    extracted_date = Column(DateTime, default=datetime.utcnow)
    extraction_method = Column(String(50))  # ocr, nlp, manual, etc.
    confidence_score = Column(Float)  # 0.0 to 1.0
    
    # Relationships
    document = relationship("Document", back_populates="metadata_entries")
    
    def __repr__(self):
        return f"<DocumentMetadata(doc_id={self.doc_id}, key={self.key})>"

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))  # document, query, generation, etc.
    resource_id = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    details = Column(JSON)  # Additional context as JSON
    session_id = Column(String(100))
    
    def __repr__(self):
        return f"<AuditLog(user_id={self.user_id}, action={self.action}, timestamp={self.timestamp})>"

class ComplianceRule(Base):
    __tablename__ = "compliance_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id = Column(String(50), unique=True, nullable=False)  # Human-readable ID
    name = Column(String(200), nullable=False)
    description = Column(Text)
    criteria = Column(JSON, nullable=False)  # Rule logic as JSON
    severity = Column(Enum(RuleSeverity), default=RuleSeverity.MEDIUM)
    is_active = Column(Boolean, default=True)
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    category = Column(String(100))  # documentation, code, security, etc.
    
    def __repr__(self):
        return f"<ComplianceRule(rule_id={self.rule_id}, name={self.name}, severity={self.severity})>"

class DiscrepancyFlag(Base):
    __tablename__ = "discrepancy_flags"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    flag_id = Column(String(50), unique=True, nullable=False)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)  # Nullable for code-only issues
    rule_id = Column(UUID(as_uuid=True), ForeignKey("compliance_rules.id"), nullable=True)
    flag_type = Column(Enum(DiscrepancyType), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(Enum(FlagStatus), default=FlagStatus.OPEN)
    priority = Column(Enum(RuleSeverity), default=RuleSeverity.MEDIUM)
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    assigned_to = Column(String(100))
    resolved_date = Column(DateTime)
    resolution_notes = Column(Text)
    extra_metadata = Column(JSON)  # Additional context
    
    # Relationships
    document = relationship("Document", back_populates="discrepancy_flags")
    rule = relationship("ComplianceRule")
    
    def __repr__(self):
        return f"<DiscrepancyFlag(flag_id={self.flag_id}, type={self.flag_type}, status={self.status})>"

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feedback_id = Column(String(50), unique=True, nullable=False)
    user_id = Column(String(100), nullable=False)
    session_id = Column(String(100))
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    rating = Column(Integer)  # 1-5 scale
    correction = Column(Text)  # User's suggested correction
    feedback_type = Column(String(50))  # helpful, incorrect, incomplete, etc.
    timestamp = Column(DateTime, default=datetime.utcnow)
    context = Column(JSON)  # Query context, sources used, etc.
    
    def __repr__(self):
        return f"<Feedback(feedback_id={self.feedback_id}, rating={self.rating})>"

# Chat Memory Model (for PostgreSQL backup of Redis data)
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(String(100), nullable=False)
    created_date = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), ForeignKey("chat_sessions.session_id"), nullable=False)
    message_id = Column(String(100), nullable=False)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sources = Column(JSON)  # Documents/sources referenced
    extra_metadata = Column(JSON)  # Processing metadata
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
