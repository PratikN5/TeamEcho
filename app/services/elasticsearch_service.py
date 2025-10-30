import os
import logging
from typing import Dict, List, Any, Optional, Union
from elasticsearch import AsyncElasticsearch, NotFoundError
import json
import re
from datetime import datetime

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class ElasticsearchService:
    """Service for indexing and searching code in Elasticsearch"""

    def __init__(self):
        self.es = None

    async def init(self):
        try:
            self.es = AsyncElasticsearch(
                hosts=[settings.ELASTICSEARCH_URL],
                basic_auth=(settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD),
                verify_certs=settings.ELASTICSEARCH_VERIFY_CERTS
            )
            logger.info("Connected to Elasticsearch")

            await self._initialize_index()

        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {str(e)}")
            self.es = None


    async def _initialize_index(self):
        """Initialize Elasticsearch index for code"""
        if not self.es:
            return
            
        try:
            # Check if index exists
            if not await self.es.indices.exists(index=settings.ELASTICSEARCH_CODE_INDEX):
                # Create index with mapping
                await self.es.indices.create(
                    index=settings.ELASTICSEARCH_CODE_INDEX,
                    body={
                        "mappings": {
                            "properties": {
                                "file_path": {"type": "keyword"},
                                "filename": {"type": "keyword"},
                                "language": {"type": "keyword"},
                                "content": {"type": "text"},
                                "functions": {
                                    "type": "nested",
                                    "properties": {
                                        "name": {"type": "keyword"},
                                        "signature": {"type": "text"},
                                        "docstring": {"type": "text"},
                                        "body": {"type": "text"},
                                        "start_line": {"type": "integer"},
                                        "end_line": {"type": "integer"},
                                        "has_docstring": {"type": "boolean"}
                                    }
                                },
                                "classes": {
                                    "type": "nested",
                                    "properties": {
                                        "name": {"type": "keyword"},
                                        "docstring": {"type": "text"},
                                        "methods": {
                                            "type": "nested",
                                            "properties": {
                                                "name": {"type": "keyword"},
                                                "signature": {"type": "text"},
                                                "docstring": {"type": "text"},
                                                "body": {"type": "text"},
                                                "start_line": {"type": "integer"},
                                                "end_line": {"type": "integer"},
                                                "has_docstring": {"type": "boolean"}
                                            }
                                        },
                                        "start_line": {"type": "integer"},
                                        "end_line": {"type": "integer"},
                                        "has_docstring": {"type": "boolean"}
                                    }
                                },
                                "imports": {
                                    "type": "nested",
                                    "properties": {
                                        "module": {"type": "keyword"},
                                        "name": {"type": "keyword"},
                                        "alias": {"type": "keyword"}
                                    }
                                },
                                "doc_coverage": {"type": "float"},
                                "indexed_at": {"type": "date"},
                                "last_modified": {"type": "date"},
                                "repo": {"type": "keyword"},
                                "branch": {"type": "keyword"},
                                "loc": {"type": "integer"}
                            }
                        }
                    }
                )
                logger.info(f"Created Elasticsearch index: {settings.ELASTICSEARCH_CODE_INDEX}")
        except Exception as e:
            logger.error(f"Error initializing Elasticsearch index: {str(e)}")

    async def index_code_file(self, file_path: str, content: str, language: str = None, 
                                metadata: Dict[str, Any] = None) -> bool:
        """
        Index a code file in Elasticsearch
        
        Args:
            file_path: Path to the code file
            content: Content of the code file
            language: Programming language (auto-detected if None)
            metadata: Additional metadata
            
        Returns:
            True if indexing was successful, False otherwise
        """
        if not self.es:
            logger.error("Elasticsearch client not initialized")
            return False
            
        try:
            # Auto-detect language if not provided
            if not language:
                language = self._detect_language(file_path)
            
            # Parse code structure
            code_structure = await self._parse_code_structure(content, language)
            
            # Calculate documentation coverage
            doc_coverage = self._calculate_doc_coverage(code_structure)
            
            # Prepare document for indexing
            filename = os.path.basename(file_path)
            doc = {
                "file_path": file_path,
                "filename": filename,
                "language": language,
                "content": content,
                "functions": code_structure.get("functions", []),
                "classes": code_structure.get("classes", []),
                "imports": code_structure.get("imports", []),
                "doc_coverage": doc_coverage,
                "indexed_at": datetime.utcnow().isoformat(),
                "last_modified": metadata.get("last_modified", datetime.utcnow().isoformat()),
                "repo": metadata.get("repo", ""),
                "branch": metadata.get("branch", ""),
                "loc": content.count("\n") + 1
            }
            
            # Add any additional metadata
            if metadata:
                for key, value in metadata.items():
                    if key not in doc and isinstance(value, (str, int, float, bool)):
                        doc[key] = value
            
            # Index document
            await self.es.index(
                index=settings.ELASTICSEARCH_CODE_INDEX,
                id=self._generate_document_id(file_path),
                document=doc
            )
            
            logger.info(f"Indexed code file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error indexing code file {file_path}: {str(e)}")
            return False

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".rs": "rust",
            ".scala": "scala",
            ".sh": "bash",
            ".pl": "perl",
            ".r": "r"
        }
        
        return language_map.get(ext, "unknown")

    async def _parse_code_structure(self, content: str, language: str) -> Dict[str, Any]:
        """
        Parse code structure using language-specific parsers
        
        This is a simplified implementation. In a production system, you'd use
        proper parsers like tree-sitter, AST modules, or language-specific tools.
        """
        if language == "python":
            return self._parse_python_code(content)
        elif language in ["javascript", "typescript"]:
            return self._parse_js_code(content)
        else:
            # Basic regex-based parsing for other languages
            return self._parse_generic_code(content, language)

    def _parse_python_code(self, content: str) -> Dict[str, Any]:
        """Parse Python code structure using regex (simplified)"""
        result = {
            "functions": [],
            "classes": [],
            "imports": []
        }
        
        lines = content.split("\n")
        
        # Extract imports
        import_pattern = re.compile(r'^(?:from\s+(\S+)\s+)?import\s+(.+)$')
        for i, line in enumerate(lines):
            match = import_pattern.match(line.strip())
            if match:
                from_module = match.group(1)
                imports = match.group(2).split(",")
                
                for imp in imports:
                    imp = imp.strip()
                    if " as " in imp:
                        name, alias = imp.split(" as ")
                        result["imports"].append({
                            "module": from_module,
                            "name": name.strip(),
                            "alias": alias.strip()
                        })
                    else:
                        result["imports"].append({
                            "module": from_module,
                            "name": imp,
                            "alias": None
                        })
        
        # Extract functions
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Function definition
            if line.startswith("def ") and "(" in line:
                func_name = line[4:line.index("(")].strip()
                signature = line
                
                # Find end of function signature
                while i < len(lines) and ")" not in lines[i]:
                    i += 1
                    signature += " " + lines[i].strip()
                
                start_line = i
                docstring = ""
                has_docstring = False
                body = []
                
                # Check for docstring
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                
                if j < len(lines) and (lines[j].strip().startswith('"""') or lines[j].strip().startswith("'''")):
                    has_docstring = True
                    docstring_start = j
                    docstring_delimiter = lines[j].strip()[0:3]
                    
                    # Find end of docstring
                    while j < len(lines) and docstring_delimiter not in lines[j][3:]:
                        docstring += lines[j] + "\n"
                        j += 1
                    
                    if j < len(lines):
                        docstring += lines[j]
                        j += 1
                
                # Find function body and end line
                indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
                while j < len(lines):
                    if lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) <= indent_level:
                        break
                    body.append(lines[j])
                    j += 1
                
                end_line = j - 1
                
                result["functions"].append({
                    "name": func_name,
                    "signature": signature,
                    "docstring": docstring,
                    "body": "\n".join(body),
                    "start_line": start_line + 1,
                    "end_line": end_line + 1,
                    "has_docstring": has_docstring
                })
                
                i = j - 1
            
            # Class definition
            elif line.startswith("class "):
                class_name = line[6:].split("(")[0].strip()
                
                start_line = i
                docstring = ""
                has_docstring = False
                methods = []
                
                # Check for docstring
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                
                if j < len(lines) and (lines[j].strip().startswith('"""') or lines[j].strip().startswith("'''")):
                    has_docstring = True
                    docstring_start = j
                    docstring_delimiter = lines[j].strip()[0:3]
                    
                    # Find end of docstring
                    while j < len(lines) and docstring_delimiter not in lines[j][3:]:
                        docstring += lines[j] + "\n"
                        j += 1
                    
                    if j < len(lines):
                        docstring += lines[j]
                        j += 1
                
                # Find class body and methods
                indent_level = len(lines[i + 1]) - len(lines[i + 1].lstrip()) if i + 1 < len(lines) else 0
                
                while j < len(lines):
                    if lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) <= indent_level and not lines[j].strip().startswith("#"):
                        break
                    
                    # Method definition
                    if lines[j].strip().startswith("def ") and "(" in lines[j]:
                        method_indent = len(lines[j]) - len(lines[j].lstrip())
                        if method_indent == indent_level + 4:
                            method_name = lines[j].strip()[4:lines[j].strip().index("(")].strip()
                            method_signature = lines[j].strip()
                            
                            # Find end of method signature
                            k = j
                            while k < len(lines) and ")" not in lines[k]:
                                k += 1
                                method_signature += " " + lines[k].strip()
                            
                            method_start_line = k
                            method_docstring = ""
                            method_has_docstring = False
                            method_body = []
                            
                            # Check for method docstring
                            l = k + 1
                            while l < len(lines) and lines[l].strip() == "":
                                l += 1
                            
                            if l < len(lines) and (lines[l].strip().startswith('"""') or lines[l].strip().startswith("'''")):
                                method_has_docstring = True
                                method_docstring_start = l
                                method_docstring_delimiter = lines[l].strip()[0:3]
                                
                                # Find end of docstring
                                while l < len(lines) and method_docstring_delimiter not in lines[l][3:]:
                                    method_docstring += lines[l] + "\n"
                                    l += 1
                                
                                if l < len(lines):
                                    method_docstring += lines[l]
                                    l += 1
                            
                            # Find method body and end line
                            method_indent_level = method_indent
                            while l < len(lines):
                                if lines[l].strip() and len(lines[l]) - len(lines[l].lstrip()) <= method_indent_level:
                                    break
                                method_body.append(lines[l])
                                l += 1
                            
                            method_end_line = l - 1
                            
                            methods.append({
                                "name": method_name,
                                "signature": method_signature,
                                "docstring": method_docstring,
                                "body": "\n".join(method_body),
                                "start_line": method_start_line + 1,
                                "end_line": method_end_line + 1,
                                "has_docstring": method_has_docstring
                            })
                            
                            j = l - 1
                    
                    j += 1
                
                end_line = j - 1
                
                result["classes"].append({
                    "name": class_name,
                    "docstring": docstring,
                    "methods": methods,
                    "start_line": start_line + 1,
                    "end_line": end_line + 1,
                    "has_docstring": has_docstring
                })
                
                i = j - 1
            
            i += 1
        
        return result

    def _parse_js_code(self, content: str) -> Dict[str, Any]:
        """Parse JavaScript/TypeScript code structure (simplified)"""
        result = {
            "functions": [],
            "classes": [],
            "imports": []
        }
        
        lines = content.split("\n")
        
        # Extract imports
        import_pattern = re.compile(r'^import\s+(.+)\s+from\s+[\'"](.+)[\'"]')
        for i, line in enumerate(lines):
            match = import_pattern.match(line.strip())
            if match:
                imports = match.group(1)
                module = match.group(2)
                
                if imports.startswith("{"):
                    # Named imports
                    imports = imports.strip("{} ").split(",")
                    for imp in imports:
                        imp = imp.strip()
                        if " as " in imp:
                            name, alias = imp.split(" as ")
                            result["imports"].append({
                                "module": module,
                                "name": name.strip(),
                                "alias": alias.strip()
                            })
                        else:
                            result["imports"].append({
                                "module": module,
                                "name": imp,
                                "alias": None
                            })
                else:
                    # Default import
                    result["imports"].append({
                        "module": module,
                        "name": imports.strip(),
                        "alias": None
                    })
        
        # This is a simplified implementation
        # A real implementation would use a proper parser
        
        return result

    def _parse_generic_code(self, content: str, language: str) -> Dict[str, Any]:
        """Generic code structure parsing for unsupported languages"""
        result = {
            "functions": [],
            "classes": [],
            "imports": []
        }
        
        # Basic function detection using regex
        if language in ["java", "cpp", "csharp"]:
            # Match function signatures like:
            # public void myMethod(String arg) { ... }
            func_pattern = re.compile(r'(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *\{?')
            
            for match in func_pattern.finditer(content):
                func_name = match.group(1)
                start_pos = match.start()
                
                # Find function start line
                start_line = content[:start_pos].count('\n') + 1
                
                result["functions"].append({
                    "name": func_name,
                    "signature": match.group(0),
                    "docstring": "",  # Would need more complex parsing
                    "body": "",  # Would need more complex parsing
                    "start_line": start_line,
                    "end_line": start_line,  # Approximation
                    "has_docstring": False
                })
            
            # Match class definitions
            class_pattern = re.compile(r'(?:public|private|protected|static|\s) +class +(\w+)')
            
            for match in class_pattern.finditer(content):
                class_name = match.group(1)
                start_pos = match.start()
                
                # Find class start line
                start_line = content[:start_pos].count('\n') + 1
                
                result["classes"].append({
                    "name": class_name,
                    "docstring": "",
                    "methods": [],  # Would need more complex parsing
                    "start_line": start_line,
                    "end_line": start_line,  # Approximation
                    "has_docstring": False
                })
        
        return result

    def _calculate_doc_coverage(self, code_structure: Dict[str, Any]) -> float:
        """Calculate documentation coverage percentage"""
        total_items = 0
        documented_items = 0
        
        # Count functions
        for func in code_structure.get("functions", []):
            total_items += 1
            if func.get("has_docstring", False):
                documented_items += 1
        
        # Count classes
        for cls in code_structure.get("classes", []):
            total_items += 1
            if cls.get("has_docstring", False):
                documented_items += 1
            
            # Count methods
            for method in cls.get("methods", []):
                total_items += 1
                if method.get("has_docstring", False):
                    documented_items += 1
        
        # Calculate percentage
        if total_items == 0:
            return 0.0
        
        return round(documented_items / total_items * 100, 2)

    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID for a file path"""
        # Use a hash of the file path to ensure uniqueness
        return file_path.replace("/", "_").replace("\\", "_").replace(".", "_")

    async def search_code(self, query: str, language: str = None, 
                            file_path: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code using full-text search
        
        Args:
            query: Search query
            language: Filter by programming language
            file_path: Filter by file path
            limit: Maximum number of results
            
        Returns:
            List of matching code files
        """
        if not self.es:
            logger.error("Elasticsearch client not initialized")
            return []
            
        try:
            # Build search query
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"multi_match": {
                                "query": query,
                                "fields": ["content", "functions.name^2", "functions.docstring^1.5", 
                                            "classes.name^2", "classes.docstring^1.5"]
                            }}
                        ],
                        "filter": []
                    }
                },
                "highlight": {
                    "fields": {
                        "content": {},
                        "functions.name": {},
                        "functions.docstring": {},
                        "classes.name": {},
                        "classes.docstring": {}
                    }
                },
                "size": limit
            }
            
            # Add filters
            if language:
                search_query["query"]["bool"]["filter"].append({"term": {"language": language}})
            
            if file_path:
                search_query["query"]["bool"]["filter"].append({"wildcard": {"file_path": f"*{file_path}*"}})
            
            # Execute search
            response = await self.es.search(
                index=settings.ELASTICSEARCH_CODE_INDEX,
                body=search_query
            )
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                highlights = hit.get("highlight", {})
                
                result = {
                    "id": hit["_id"],
                    "file_path": source["file_path"],
                    "filename": source["filename"],
                    "language": source["language"],
                    "score": hit["_score"],
                    "doc_coverage": source["doc_coverage"],
                    "highlights": highlights,
                    "functions": [f["name"] for f in source.get("functions", [])[:5]],  # First 5 functions
                    "classes": [c["name"] for c in source.get("classes", [])[:5]]  # First 5 classes
                }
                
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching code: {str(e)}")
            return []

    async def analyze_code_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze code structure for a file
        
        Args:
            file_path: Path to code file
            
        Returns:
            Code structure analysis
        """
        if not self.es:
            logger.error("Elasticsearch client not initialized")
            return {}
            
        try:
            # Get document
            doc_id = self._generate_document_id(file_path)
            doc = await self.es.get(
                index=settings.ELASTICSEARCH_CODE_INDEX,
                id=doc_id
            )
            
            source = doc["_source"]
            
            # Calculate metrics
            num_functions = len(source.get("functions", []))
            num_classes = len(source.get("classes", []))
            num_methods = sum(len(cls.get("methods", [])) for cls in source.get("classes", []))
            
            functions_with_docs = sum(1 for f in source.get("functions", []) if f.get("has_docstring", False))
            classes_with_docs = sum(1 for c in source.get("classes", []) if c.get("has_docstring", False))
            methods_with_docs = sum(
                sum(1 for m in cls.get("methods", []) if m.get("has_docstring", False))
                for cls in source.get("classes", [])
            )
            
            # Build analysis
            analysis = {
                "file_path": source["file_path"],
                "language": source["language"],
                "loc": source["loc"],
                "doc_coverage": source["doc_coverage"],
                "metrics": {
                    "num_functions": num_functions,
                    "num_classes": num_classes,
                    "num_methods": num_methods,
                    "total_callable": num_functions + num_methods,
                    "documented_functions": functions_with_docs,
                    "documented_classes": classes_with_docs,
                    "documented_methods": methods_with_docs,
                    "total_documented": functions_with_docs + classes_with_docs + methods_with_docs
                },
                "imports": source.get("imports", []),
                "functions": [{
                    "name": f["name"],
                    "has_docstring": f.get("has_docstring", False),
                    "line_count": f["end_line"] - f["start_line"] + 1
                } for f in source.get("functions", [])],
                "classes": [{
                    "name": c["name"],
                    "has_docstring": c.get("has_docstring", False),
                    "method_count": len(c.get("methods", [])),
                    "documented_methods": sum(1 for m in c.get("methods", []) if m.get("has_docstring", False)),
                    "line_count": c["end_line"] - c["start_line"] + 1
                } for c in source.get("classes", [])]
            }
            
            return analysis
        except NotFoundError:
            logger.error(f"Code file {file_path} not found in index")
            return {"error": "File not found in index"}
        except Exception as e:
            logger.error(f"Error analyzing code structure: {str(e)}")
            return {"error": str(e)}

    async def close(self):
        """Close Elasticsearch connection"""
        if self.es:
            await self.es.close()
