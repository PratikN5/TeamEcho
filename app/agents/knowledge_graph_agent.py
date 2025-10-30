import spacy
import logging
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class KnowledgeGraphAgent:
    """Agent for extracting entities and relationships to build a knowledge graph in Neo4j"""
    
    def __init__(self):
        # Initialize NLP model for entity recognition
        try:
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy NLP model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            self.nlp = None
            
        # Connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            logger.info("Connected to Neo4j database")
            self._init_schema()
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}")
            self.driver = None
    
    def _init_schema(self):
        """Initialize Neo4j schema with constraints and indexes"""
        with self.driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Code) REQUIRE c.path IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Function) REQUIRE (f.name, f.file_path) IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE (c.name, c.file_path) IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
            
            # Create indexes
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Code) ON (c.language)")
    
    async def extract_entities(self, text: str, doc_id: str = None) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using spaCy
        
        Args:
            text: Text to extract entities from
            doc_id: Optional document ID for reference
            
        Returns:
            List of extracted entities with type and metadata
        """
        if not self.nlp:
            logger.error("NLP model not loaded")
            return []
            
        try:
            # Process text with spaCy
            doc = self.nlp(text[:1000000])  # Limit text size to avoid memory issues
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entity = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "doc_id": doc_id
                }
                entities.append(entity)
                
            # Extract technical concepts (custom logic)
            # This is a simplified approach - in production you'd want more sophisticated concept extraction
            technical_terms = self._extract_technical_concepts(doc)
            for term in technical_terms:
                entity = {
                    "text": term["text"],
                    "label": "TECHNICAL_CONCEPT",
                    "start": term["start"],
                    "end": term["end"],
                    "doc_id": doc_id
                }
                entities.append(entity)
                
            logger.info(f"Extracted {len(entities)} entities from text")
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _extract_technical_concepts(self, doc) -> List[Dict[str, Any]]:
        """Extract technical concepts using custom rules"""
        technical_terms = []
        
        # Look for noun phrases that might be technical concepts
        for chunk in doc.noun_chunks:
            # Check if the chunk contains technical indicators
            if any(token.text.lower() in self._get_technical_indicators() for token in chunk):
                technical_terms.append({
                    "text": chunk.text,
                    "start": chunk.start_char,
                    "end": chunk.end_char
                })
                
        # Add code-related terms
        code_patterns = [
            "function", "class", "method", "api", "endpoint", "variable",
            "module", "library", "framework", "algorithm", "data structure"
        ]
        
        for token in doc:
            if token.text.lower() in code_patterns:
                # Get the full noun phrase if possible
                if token.head.pos_ == "NOUN" and token.dep_ in ["compound", "amod"]:
                    phrase = " ".join([t.text for t in token.head.subtree])
                    technical_terms.append({
                        "text": phrase,
                        "start": token.idx,
                        "end": token.idx + len(phrase)
                    })
                else:
                    technical_terms.append({
                        "text": token.text,
                        "start": token.idx,
                        "end": token.idx + len(token.text)
                    })
        
        return technical_terms
    
    def _get_technical_indicators(self) -> List[str]:
        """Return list of words that indicate technical concepts"""
        return [
            "algorithm", "api", "architecture", "code", "component", "data", 
            "database", "function", "interface", "library", "method", "module", 
            "object", "pattern", "protocol", "schema", "service", "structure", 
            "system", "technology", "tool"
        ]
    
    async def extract_relationships(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities
        
        Args:
            entities: List of extracted entities
            text: Original text
            
        Returns:
            List of relationships between entities
        """
        relationships = []
        
        # Simple co-occurrence based relationship extraction
        # In production, you'd want more sophisticated approaches
        entity_map = {e["text"].lower(): e for e in entities}
        
        # Process text with spaCy for dependency parsing
        doc = self.nlp(text[:1000000])
        
        # Extract relationships based on dependency parsing
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ["dobj", "pobj"] and token.head.pos_ == "VERB":
                    # Subject-verb-object relationship
                    subj = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                    if subj:
                        subj_text = subj[0].text.lower()
                        obj_text = token.text.lower()
                        verb = token.head.text
                        
                        if subj_text in entity_map and obj_text in entity_map:
                            rel = {
                                "source": entity_map[subj_text]["text"],
                                "source_type": entity_map[subj_text]["label"],
                                "target": entity_map[obj_text]["text"],
                                "target_type": entity_map[obj_text]["label"],
                                "relation": verb.upper()
                            }
                            relationships.append(rel)
        
        # Add technical relationships based on domain knowledge
        technical_rels = self._extract_technical_relationships(entities)
        relationships.extend(technical_rels)
        
        logger.info(f"Extracted {len(relationships)} relationships from text")
        return relationships
    
    def _extract_technical_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract technical relationships based on entity types"""
        relationships = []
        
        # Group entities by type
        entity_by_type = {}
        for entity in entities:
            entity_type = entity["label"]
            if entity_type not in entity_by_type:
                entity_by_type[entity_type] = []
            entity_by_type[entity_type].append(entity)
        
        # Create relationships based on entity types
        if "PERSON" in entity_by_type and "TECHNICAL_CONCEPT" in entity_by_type:
            for person in entity_by_type["PERSON"]:
                for concept in entity_by_type["TECHNICAL_CONCEPT"]:
                    # Assume people might be experts in technical concepts
                    relationships.append({
                        "source": person["text"],
                        "source_type": "PERSON",
                        "target": concept["text"],
                        "target_type": "TECHNICAL_CONCEPT",
                        "relation": "EXPERT_IN"
                    })
        
        return relationships
    
    async def build_graph(self, doc_id: str, title: str, entities: List[Dict[str, Any]], 
                         relationships: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> bool:
        """
        Build knowledge graph in Neo4j from extracted entities and relationships
        
        Args:
            doc_id: Document ID
            title: Document title
            entities: List of extracted entities
            relationships: List of extracted relationships
            metadata: Additional document metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return False
            
        try:
            with self.driver.session() as session:
                # Create document node
                session.run(
                    """
                    MERGE (d:Document {id: $id})
                    SET d.title = $title,
                        d.updated_at = timestamp()
                    """,
                    id=doc_id,
                    title=title
                )
                
                # Add metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            session.run(
                                f"MATCH (d:Document {{id: $id}}) SET d.{key} = $value",
                                id=doc_id,
                                value=value
                            )
                
                # Create entity nodes and link to document
                for entity in entities:
                    entity_type = entity["label"]
                    entity_text = entity["text"]
                    
                    if entity_type == "PERSON":
                        session.run(
                            """
                            MERGE (p:Person {name: $name})
                            WITH p
                            MATCH (d:Document {id: $doc_id})
                            MERGE (d)-[:MENTIONS]->(p)
                            """,
                            name=entity_text,
                            doc_id=doc_id
                        )
                    elif entity_type == "TECHNICAL_CONCEPT" or entity_type == "PRODUCT":
                        session.run(
                            """
                            MERGE (c:Concept {name: $name})
                            SET c.type = $type
                            WITH c
                            MATCH (d:Document {id: $doc_id})
                            MERGE (d)-[:DESCRIBES]->(c)
                            """,
                            name=entity_text,
                            type=entity_type,
                            doc_id=doc_id
                        )
                    elif entity_type == "ORG":
                        session.run(
                            """
                            MERGE (o:Organization {name: $name})
                            WITH o
                            MATCH (d:Document {id: $doc_id})
                            MERGE (d)-[:MENTIONS]->(o)
                            """,
                            name=entity_text,
                            doc_id=doc_id
                        )
                
                # Create relationships
                for rel in relationships:
                    source = rel["source"]
                    source_type = rel["source_type"]
                    target = rel["target"]
                    target_type = rel["target_type"]
                    relation = rel["relation"]
                    
                    # Map entity types to Neo4j node labels
                    source_label = self._map_entity_type_to_label(source_type)
                    target_label = self._map_entity_type_to_label(target_type)
                    
                    if source_label and target_label:
                        session.run(
                            f"""
                            MATCH (s:{source_label} {{name: $source}})
                            MATCH (t:{target_label} {{name: $target}})
                            MERGE (s)-[r:{relation}]->(t)
                            """,
                            source=source,
                            target=target
                        )
                
                logger.info(f"Built knowledge graph for document {doc_id}")
                return True
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            return False
    
    def _map_entity_type_to_label(self, entity_type: str) -> str:
        """Map entity type to Neo4j node label"""
        mapping = {
            "PERSON": "Person",
            "TECHNICAL_CONCEPT": "Concept",
            "ORG": "Organization",
            "PRODUCT": "Concept",
            "GPE": "Location"
        }
        return mapping.get(entity_type, "Entity")
    
    async def update_graph(self, doc_id: str, entities: List[Dict[str, Any]], 
                          relationships: List[Dict[str, Any]]) -> bool:
        """
        Update existing knowledge graph with new entities and relationships
        
        Args:
            doc_id: Document ID
            entities: List of new entities
            relationships: List of new relationships
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return False
            
        try:
            with self.driver.session() as session:
                # Check if document exists
                result = session.run(
                    "MATCH (d:Document {id: $id}) RETURN d",
                    id=doc_id
                )
                
                if not result.single():
                    logger.error(f"Document {doc_id} not found in graph")
                    return False
                
                # Update timestamp
                session.run(
                    """
                    MATCH (d:Document {id: $id})
                    SET d.updated_at = timestamp()
                    """,
                    id=doc_id
                )
                
                # Add new entities and relationships
                for entity in entities:
                    entity_type = entity["label"]
                    entity_text = entity["text"]
                    
                    if entity_type == "PERSON":
                        session.run(
                            """
                            MERGE (p:Person {name: $name})
                            WITH p
                            MATCH (d:Document {id: $doc_id})
                            MERGE (d)-[:MENTIONS]->(p)
                            """,
                            name=entity_text,
                            doc_id=doc_id
                        )
                    elif entity_type == "TECHNICAL_CONCEPT":
                        session.run(
                            """
                            MERGE (c:Concept {name: $name})
                            WITH c
                            MATCH (d:Document {id: $doc_id})
                            MERGE (d)-[:DESCRIBES]->(c)
                            """,
                            name=entity_text,
                            doc_id=doc_id
                        )
                
                # Add new relationships
                for rel in relationships:
                    source = rel["source"]
                    source_type = rel["source_type"]
                    target = rel["target"]
                    target_type = rel["target_type"]
                    relation = rel["relation"]
                    
                    # Map entity types to Neo4j node labels
                    source_label = self._map_entity_type_to_label(source_type)
                    target_label = self._map_entity_type_to_label(target_type)
                    
                    if source_label and target_label:
                        session.run(
                            f"""
                            MATCH (s:{source_label} {{name: $source}})
                            MATCH (t:{target_label} {{name: $target}})
                            MERGE (s)-[r:{relation}]->(t)
                            """,
                            source=source,
                            target=target
                        )
                
                logger.info(f"Updated knowledge graph for document {doc_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {str(e)}")
            return False
    
    async def link_code_to_document(self, doc_id: str, code_path: str, relationship_type: str = "DOCUMENTS") -> bool:
        """
        Link code file to document in knowledge graph
        
        Args:
            doc_id: Document ID
            code_path: Path to code file
            relationship_type: Type of relationship between document and code
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return False
            
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:Code {path: $code_path})
                    MERGE (d)-[r:DOCUMENTS]->(c)
                    """,
                    doc_id=doc_id,
                    code_path=code_path
                )
                
                logger.info(f"Linked code {code_path} to document {doc_id}")
                return True
        except Exception as e:
            logger.error(f"Error linking code to document: {str(e)}")
            return False
    
    async def get_document_graph(self, doc_id: str) -> Dict[str, Any]:
        """
        Get knowledge graph for a document
        
        Args:
            doc_id: Document ID
            
        Returns:
            Graph data in a format suitable for visualization
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return {"nodes": [], "edges": []}
            
        try:
            with self.driver.session() as session:
                # Get document node
                doc_result = session.run(
                    """
                    MATCH (d:Document {id: $id})
                    RETURN d
                    """,
                    id=doc_id
                )
                
                doc_node = doc_result.single()
                if not doc_node:
                    logger.error(f"Document {doc_id} not found in graph")
                    return {"nodes": [], "edges": []}
                
                # Get all nodes connected to document
                result = session.run(
                    """
                    MATCH (d:Document {id: $id})-[r]->(n)
                    RETURN n, type(r) as relationship
                    UNION
                    MATCH (n)-[r]->(d:Document {id: $id})
                    RETURN n, type(r) as relationship
                    """,
                    id=doc_id
                )
                
                nodes = []
                edges = []
                
                # Add document node
                doc_data = dict(doc_node["d"])
                nodes.append({
                    "id": doc_id,
                    "label": doc_data.get("title", "Document"),
                    "type": "Document",
                    "properties": doc_data
                })
                
                # Add connected nodes and edges
                node_ids = set([doc_id])
                for record in result:
                    node = record["n"]
                    node_id = node.get("name", node.get("id", node.get("path", str(hash(node)))))
                    
                    if node_id not in node_ids:
                        node_data = dict(node)
                        node_type = next(iter(node.labels))
                        
                        nodes.append({
                            "id": node_id,
                            "label": node_data.get("name", node_id),
                            "type": node_type,
                            "properties": node_data
                        })
                        node_ids.add(node_id)
                    
                    # Add edge
                    edges.append({
                        "source": doc_id,
                        "target": node_id,
                        "label": record["relationship"]
                    })
                
                return {
                    "nodes": nodes,
                    "edges": edges
                }
        except Exception as e:
            logger.error(f"Error getting document graph: {str(e)}")
            return {"nodes": [], "edges": []}
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
