# services/chunking_service.py
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

class ChunkingService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Split document into chunks for embedding
        """
        try:
            # Update text splitter settings if different from default
            if chunk_size != 1000 or chunk_overlap != 200:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                # Create chunk metadata
                chunk_metadata = {
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    **metadata
                }
                
                # Add chunk to list
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Split document into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            return []
    
    async def chunk_by_section(
        self,
        text: str,
        metadata: Dict[str, Any],
        section_pattern: str = r"(?:^|\n)#+\s+(.*?)(?=\n#+\s+|\Z)",
        min_section_length: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Split document by sections (e.g., Markdown headers)
        """
        try:
            # Find all sections based on pattern
            sections = []
            current_section = ""
            current_title = "Introduction"
            
            # Use regex to find sections
            matches = re.finditer(section_pattern, text, re.DOTALL)
            last_end = 0
            
            for match in matches:
                # Add content before this header to current section
                section_content = text[last_end:match.start()]
                if section_content.strip() and len(current_section) > min_section_length:
                    sections.append({
                        "title": current_title,
                        "content": current_section.strip()
                    })
                
                # Update current section
                current_title = match.group(1).strip()
                current_section = text[match.start():match.end()]
                last_end = match.end()
            
            # Add final section
            if last_end < len(text):
                final_content = text[last_end:]
                if final_content.strip():
                    sections.append({
                        "title": current_title,
                        "content": final_content.strip()
                    })
            
            # If no sections found, use default chunking
            if not sections:
                return await self.chunk_document(text, metadata)
            
            # Create chunk objects with metadata
            chunks = []
            for i, section in enumerate(sections):
                # Create chunk metadata
                chunk_metadata = {
                    "chunk_index": i,
                    "total_chunks": len(sections),
                    "section_title": section["title"],
                    **metadata
                }
                
                # Add chunk to list
                chunks.append({
                    "text": section["content"],
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Split document into {len(chunks)} sections")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking document by section: {str(e)}")
            # Fall back to regular chunking
            return await self.chunk_document(text, metadata)
    
    async def chunk_code(
        self,
        code: str,
        metadata: Dict[str, Any],
        language: str = "python",
        chunk_by_function: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Split code document by functions/classes or fixed size
        """
        try:
            chunks = []
            
            if chunk_by_function and language == "python":
                # Pattern for Python functions and classes
                pattern = r"((?:def|class)\s+\w+\s*\([^)]*\)\s*:(?:(?!\ndef|\nclass).)*)"
                matches = re.finditer(pattern, code, re.DOTALL)
                
                for i, match in enumerate(matches):
                    function_code = match.group(0)
                    
                    # Extract function/class name
                    name_match = re.search(r"(def|class)\s+(\w+)", function_code)
                    function_name = name_match.group(2) if name_match else f"code_block_{i}"
                    
                    # Create chunk metadata
                    chunk_metadata = {
                        "chunk_index": i,
                        "code_type": "function" if "def " in function_code[:10] else "class",
                        "function_name": function_name,
                        "language": language,
                        **metadata
                    }
                    
                    # Add chunk to list
                    chunks.append({
                        "text": function_code,
                        "metadata": chunk_metadata
                    })
                
                # If no functions found, fall back to regular chunking
                if not chunks:
                    return await self.chunk_document(code, metadata)
            else:
                # Use regular chunking for non-Python code or when not chunking by function
                return await self.chunk_document(code, metadata)
            
            logger.info(f"Split code into {len(chunks)} functions/classes")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking code: {str(e)}")
            # Fall back to regular chunking
            return await self.chunk_document(code, metadata)
