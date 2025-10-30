# models/schemas.py
from pydantic import BaseModel, validator, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from app.models.database import DocumentStatus, DocumentSource, DiscrepancyType, FlagStatus, RuleSeverity

# Document Schemas
class DocumentBase(BaseModel):
    filename: str
    source: DocumentSource
    uploaded_by: Optional[str] = None

class DocumentCreate(DocumentBase):
    original_filename: str
    s3_key: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    checksum: Optional[str] = None

class DocumentUpdate(BaseModel):
    filename: Optional[str] = None
    status: Optional[DocumentStatus] = None
    version: Optional[int] = None
    s3_key: Optional[str] = None
    s3_version_id: Optional[str] = None
    checksum: Optional[str] = None
    file_size: Optional[int] = None
    text: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

class DocumentResponse(DocumentBase):
    id: UUID
    original_filename: str
    s3_key: Optional[str] = None
    file_size: Optional[int]
    mime_type: Optional[str]
    upload_date: datetime
    last_modified: datetime
    version: int
    status: DocumentStatus
    checksum: Optional[str]
    
    model_config = ConfigDict(from_attributes=True)

# Metadata Schemas
class MetadataBase(BaseModel):
    key: str
    value: Optional[str] = None
    value_type: str = "string"
    extraction_method: Optional[str] = None
    confidence_score: Optional[float] = None

class MetadataCreate(MetadataBase):
    doc_id: UUID

class MetadataResponse(MetadataBase):
    id: UUID
    doc_id: UUID
    extracted_date: datetime
    
    class Config:
        orm_mode = True

# Audit Log Schemas
class AuditLogCreate(BaseModel):
    user_id: str
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class AuditLogResponse(AuditLogCreate):
    id: UUID
    timestamp: datetime
    
    class Config:
        orm_mode = True

# Compliance Rule Schemas
class ComplianceRuleBase(BaseModel):
    rule_id: str
    name: str
    description: Optional[str] = None
    criteria: Dict[str, Any]
    severity: RuleSeverity = RuleSeverity.MEDIUM
    category: Optional[str] = None

class ComplianceRuleCreate(ComplianceRuleBase):
    created_by: Optional[str] = None

class ComplianceRuleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    criteria: Optional[Dict[str, Any]] = None
    severity: Optional[RuleSeverity] = None
    is_active: Optional[bool] = None
    category: Optional[str] = None

class ComplianceRuleResponse(ComplianceRuleBase):
    id: UUID
    is_active: bool
    created_date: datetime
    updated_date: datetime
    created_by: Optional[str]
    
    model_config = ConfigDict(from_attributes=True)

# Discrepancy Flag Schemas
class DiscrepancyFlagBase(BaseModel):
    flag_id: str
    flag_type: DiscrepancyType
    title: str
    description: Optional[str] = None
    priority: RuleSeverity = RuleSeverity.MEDIUM

class DiscrepancyFlagCreate(DiscrepancyFlagBase):
    doc_id: Optional[UUID] = None
    rule_id: Optional[UUID] = None
    assigned_to: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DiscrepancyFlagUpdate(BaseModel):
    status: Optional[FlagStatus] = None
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    priority: Optional[RuleSeverity] = None

class DiscrepancyFlagResponse(DiscrepancyFlagBase):
    id: UUID
    doc_id: Optional[UUID]
    rule_id: Optional[UUID]
    status: FlagStatus
    created_date: datetime
    updated_date: datetime
    assigned_to: Optional[str]
    resolved_date: Optional[datetime]
    resolution_notes: Optional[str]
    metadata: Optional[Dict[str, Any]]
    
    model_config = ConfigDict(from_attributes=True)

# Feedback Schemas
class FeedbackBase(BaseModel):
    query: str
    answer: str
    rating: Optional[int] = None
    correction: Optional[str] = None
    feedback_type: Optional[str] = None

    @validator('rating')
    def validate_rating(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError('Rating must be between 1 and 5')
        return v

class FeedbackCreate(FeedbackBase):
    feedback_id: str
    user_id: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class FeedbackResponse(FeedbackBase):
    id: UUID
    feedback_id: str
    user_id: str
    session_id: Optional[str]
    timestamp: datetime
    context: Optional[Dict[str, Any]]
    
    model_config = ConfigDict(from_attributes=True)

# Chat Schemas
class ChatMessageCreate(BaseModel):
    session_id: str
    message_id: str
    user_message: str
    ai_response: str
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatMessageResponse(ChatMessageCreate):
    id: UUID
    timestamp: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ChatSessionCreate(BaseModel):
    session_id: str
    user_id: str

class ChatSessionResponse(ChatSessionCreate):
    id: UUID
    created_date: datetime
    last_activity: datetime
    is_active: bool
    
    model_config = ConfigDict(from_attributes=True)

# Generic Response Schemas
class StatusResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
