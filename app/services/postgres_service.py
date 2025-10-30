# services/postgres_service.py
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta

from app.models.database import (
    Document, DocumentMetadata, AuditLog, ComplianceRule, 
    DiscrepancyFlag, Feedback, ChatSession, ChatMessage,
    DocumentStatus, DocumentSource, DiscrepancyType, FlagStatus
)
from app.models.schemas import (
    DocumentCreate, DocumentUpdate, MetadataCreate, 
    AuditLogCreate, ComplianceRuleCreate, ComplianceRuleUpdate,
    DiscrepancyFlagCreate, DiscrepancyFlagUpdate, FeedbackCreate,
    ChatSessionCreate, ChatMessageCreate
)

class PostgresService:
    def __init__(self, db: Session):
        self.db = db
    
    # Document Operations
    def create_document(self, document: DocumentCreate) -> Document:
        db_document = Document(**document.dict())
        self.db.add(db_document)
        self.db.commit()
        self.db.refresh(db_document)
        return db_document
    
    def get_document(self, document_id: UUID) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == document_id).first()
    
    def get_document_by_filename(self, filename: str, source: DocumentSource) -> Optional[Document]:
        return self.db.query(Document).filter(
            and_(Document.filename == filename, Document.source == source)
        ).first()
    
    def get_documents(
        self, 
        skip: int = 0, 
        limit: int = 100,
        source: Optional[DocumentSource] = None,
        status: Optional[DocumentStatus] = None,
        uploaded_by: Optional[str] = None
    ) -> List[Document]:
        query = self.db.query(Document)
        
        if source:
            query = query.filter(Document.source == source)
        if status:
            query = query.filter(Document.status == status)
        if uploaded_by:
            query = query.filter(Document.uploaded_by == uploaded_by)
            
        return query.offset(skip).limit(limit).all()
    
    def update_document(self, document_id: UUID, document_update: DocumentUpdate) -> Optional[Document]:
        db_document = self.get_document(document_id)
        if db_document:
            update_data = document_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_document, field, value)
            self.db.commit()
            self.db.refresh(db_document)
        return db_document
    
    def delete_document(self, document_id: UUID) -> bool:
        db_document = self.get_document(document_id)
        if db_document:
            self.db.delete(db_document)
            self.db.commit()
            return True
        return False
    
    # Metadata Operations
    def create_metadata(self, doc_id: UUID, key: str, value: str, extraction_method: str) -> DocumentMetadata:
        db_metadata = DocumentMetadata(
            doc_id=doc_id,
            key=key,
            value=value,
            extraction_method=extraction_method
        )
        self.db.add(db_metadata)
        self.db.commit()
        self.db.refresh(db_metadata)
        return db_metadata

    
    def get_document_metadata(self, document_id: UUID) -> List[DocumentMetadata]:
        return self.db.query(DocumentMetadata).filter(
            DocumentMetadata.doc_id == document_id
        ).all()
    
    def get_metadata_by_key(self, document_id: UUID, key: str) -> Optional[DocumentMetadata]:
        return self.db.query(DocumentMetadata).filter(
            and_(DocumentMetadata.doc_id == document_id, DocumentMetadata.key == key)
        ).first()
    
    # Audit Log Operations
    def create_audit_log(self, audit_log: AuditLogCreate) -> AuditLog:
        db_audit_log = AuditLog(**audit_log.dict())
        self.db.add(db_audit_log)
        self.db.commit()
        self.db.refresh(db_audit_log)
        return db_audit_log
    
    def get_audit_logs(
        self,
        skip: int = 0,
        limit: int = 100,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditLog]:
        query = self.db.query(AuditLog)
        
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if action:
            query = query.filter(AuditLog.action == action)
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
            
        return query.order_by(desc(AuditLog.timestamp)).offset(skip).limit(limit).all()
    
    # Compliance Rule Operations
    def create_compliance_rule(self, rule: ComplianceRuleCreate) -> ComplianceRule:
        db_rule = ComplianceRule(**rule.dict())
        self.db.add(db_rule)
        self.db.commit()
        self.db.refresh(db_rule)
        return db_rule
    
    def get_compliance_rule(self, rule_id: str) -> Optional[ComplianceRule]:
        return self.db.query(ComplianceRule).filter(ComplianceRule.rule_id == rule_id).first()
    
    def get_active_compliance_rules(self) -> List[ComplianceRule]:
        return self.db.query(ComplianceRule).filter(ComplianceRule.is_active == True).all()
    
    def update_compliance_rule(self, rule_id: str, rule_update: ComplianceRuleUpdate) -> Optional[ComplianceRule]:
        db_rule = self.get_compliance_rule(rule_id)
        if db_rule:
            update_data = rule_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_rule, field, value)
            self.db.commit()
            self.db.refresh(db_rule)
        return db_rule
    
    # Discrepancy Flag Operations
    def create_discrepancy_flag(self, flag: DiscrepancyFlagCreate) -> DiscrepancyFlag:
        db_flag = DiscrepancyFlag(**flag.dict())
        self.db.add(db_flag)
        self.db.commit()
        self.db.refresh(db_flag)
        return db_flag
    
    def get_discrepancy_flag(self, flag_id: str) -> Optional[DiscrepancyFlag]:
        return self.db.query(DiscrepancyFlag).filter(DiscrepancyFlag.flag_id == flag_id).first()
    
    def get_open_discrepancy_flags(self, assigned_to: Optional[str] = None) -> List[DiscrepancyFlag]:
        query = self.db.query(DiscrepancyFlag).filter(DiscrepancyFlag.status == FlagStatus.OPEN)
        if assigned_to:
            query = query.filter(DiscrepancyFlag.assigned_to == assigned_to)
        return query.order_by(desc(DiscrepancyFlag.created_date)).all()
    
    def update_discrepancy_flag(self, flag_id: str, flag_update: DiscrepancyFlagUpdate) -> Optional[DiscrepancyFlag]:
        db_flag = self.get_discrepancy_flag(flag_id)
        if db_flag:
            update_data = flag_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_flag, field, value)
            
            # Set resolved_date if status is being changed to resolved
            if flag_update.status == FlagStatus.RESOLVED and db_flag.resolved_date is None:
                db_flag.resolved_date = datetime.utcnow()
                
            self.db.commit()
            self.db.refresh(db_flag)
        return db_flag
    
    # Feedback Operations
    def create_feedback(self, feedback: FeedbackCreate) -> Feedback:
        db_feedback = Feedback(**feedback.dict())
        self.db.add(db_feedback)
        self.db.commit()
        self.db.refresh(db_feedback)
        return db_feedback
    
    def get_feedback(self, skip: int = 0, limit: int = 100) -> List[Feedback]:
        return self.db.query(Feedback).order_by(desc(Feedback.timestamp)).offset(skip).limit(limit).all()
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        total_feedback = self.db.query(func.count(Feedback.id)).scalar()
        avg_rating = self.db.query(func.avg(Feedback.rating)).filter(Feedback.rating.isnot(None)).scalar()
        
        rating_distribution = self.db.query(
            Feedback.rating, func.count(Feedback.rating)
        ).filter(Feedback.rating.isnot(None)).group_by(Feedback.rating).all()
        
        return {
            "total_feedback": total_feedback,
            "average_rating": float(avg_rating) if avg_rating else None,
            "rating_distribution": dict(rating_distribution)
        }
    
    # Chat Operations
    def create_chat_session(self, session: ChatSessionCreate) -> ChatSession:
        db_session = ChatSession(**session.dict())
        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)
        return db_session
    
    def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        return self.db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    
    def create_chat_message(self, message: ChatMessageCreate) -> ChatMessage:
        db_message = ChatMessage(**message.dict())
        self.db.add(db_message)
        self.db.commit()
        self.db.refresh(db_message)
        return db_message
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        return self.db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(desc(ChatMessage.timestamp)).limit(limit).all()
    
    # Analytics and Statistics
    def get_document_stats(self) -> Dict[str, Any]:
        total_docs = self.db.query(func.count(Document.id)).scalar()
        docs_by_source = self.db.query(
            Document.source, func.count(Document.id)
        ).group_by(Document.source).all()
        docs_by_status = self.db.query(
            Document.status, func.count(Document.id)
        ).group_by(Document.status).all()
        
        return {
            "total_documents": total_docs,
            "by_source": dict(docs_by_source),
            "by_status": dict(docs_by_status)
        }
    
    def get_flag_stats(self) -> Dict[str, Any]:
        total_flags = self.db.query(func.count(DiscrepancyFlag.id)).scalar()
        flags_by_type = self.db.query(
            DiscrepancyFlag.flag_type, func.count(DiscrepancyFlag.id)
        ).group_by(DiscrepancyFlag.flag_type).all()
        flags_by_status = self.db.query(
            DiscrepancyFlag.status, func.count(DiscrepancyFlag.id)
        ).group_by(DiscrepancyFlag.status).all()
        
        return {
            "total_flags": total_flags,
            "by_type": dict(flags_by_type),
            "by_status": dict(flags_by_status)
        }

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        source: Optional[DocumentSource] = None,
        status: Optional[DocumentStatus] = None,
        uploaded_by: Optional[str] = None
    ) -> List[Document]:
        query = self.db.query(Document)
        
        if source:
            query = query.filter(Document.source == source)
        if status:
            query = query.filter(Document.status == status)
        if uploaded_by:
            query = query.filter(Document.uploaded_by == uploaded_by)
        
        return query.offset(offset).limit(limit).all()
    
    def get_document_by_s3_key(self, s3_key: str) -> Optional[Document]:
        """
        Fetch a document by its S3 key.
        """
        return self.db.query(Document).filter(Document.s3_key == s3_key).first()

    def get_document_by_checksum(self, checksum: str) -> Optional[Document]:
        """
        Fetch a document by its checksum (used to detect duplicates).
        """
        return self.db.query(Document).filter(Document.checksum == checksum).first()

