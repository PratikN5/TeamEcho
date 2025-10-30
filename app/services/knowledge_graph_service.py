import logging
from typing import List, Dict, Any, Optional
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    """Service for querying the Neo4j knowledge graph using the KnowledgeGraphAgent"""
    
    def __init__(self, knowledge_graph_agent: Optional[KnowledgeGraphAgent] = None):
        """Initialize with existing KnowledgeGraphAgent or create a new one"""
        self.kg_agent = knowledge_graph_agent or KnowledgeGraphAgent()
    
    async def query_entities(self, entity_names: List[str]) -> List[Dict[str, Any]]:
        """
        Query knowledge graph for entities by name
        
        Args:
            entity_names: List of entity names to query
            
        Returns:
            List of entity data with relationships
        """
        if not self.kg_agent.driver or not entity_names:
            return []
            
        try:
            results = []
            for entity_name in entity_names:
                # Query for entity in Neo4j
                with self.kg_agent.driver.session() as session:
                    # Query for entity by name
                    entity_result = session.run(
                        """
                        MATCH (e)
                        WHERE toLower(e.name) = toLower($name)
                        RETURN e, labels(e) as types
                        LIMIT 1
                        """,
                        name=entity_name
                    )
                    
                    record = entity_result.single()
                    if not record:
                        continue
                        
                    entity = dict(record["e"])
                    entity_types = record["types"]
                    
                    # Get entity type
                    entity_type = "Entity"
                    if entity_types:
                        entity_type = entity_types[0]
                    
                    # Get relationships
                    relationships = await self._get_entity_relationships(entity_name)
                    
                    # Get description by querying connected documents
                    description = entity.get("description", "")
                    if not description:
                        description = await self._generate_entity_description(entity_name, entity_type, relationships)
                    
                    results.append({
                        "id": entity.get("id", entity_name),
                        "name": entity_name,
                        "type": entity_type,
                        "description": description,
                        "relationships": relationships
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error querying entities: {str(e)}")
            return []
    
    async def _get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get relationships for an entity"""
        try:
            with self.kg_agent.driver.session() as session:
                # Query for outgoing relationships
                outgoing_result = session.run(
                    """
                    MATCH (e {name: $name})-[r]->(target)
                    RETURN type(r) as relation, target.name as target, labels(target)[0] as target_type
                    LIMIT 10
                    """,
                    name=entity_name
                )
                
                relationships = []
                for record in outgoing_result:
                    relationships.append({
                        "source": entity_name,
                        "relation": record["relation"],
                        "target": record["target"],
                        "target_type": record["target_type"]
                    })
                
                # Query for incoming relationships
                incoming_result = session.run(
                    """
                    MATCH (source)-[r]->(e {name: $name})
                    RETURN type(r) as relation, source.name as source, labels(source)[0] as source_type
                    LIMIT 10
                    """,
                    name=entity_name
                )
                
                for record in incoming_result:
                    relationships.append({
                        "source": record["source"],
                        "relation": record["relation"],
                        "target": entity_name,
                        "source_type": record["source_type"]
                    })
                
                return relationships
        except Exception as e:
            logger.error(f"Error getting entity relationships: {str(e)}")
            return []
    
    async def _generate_entity_description(
        self, 
        entity_name: str, 
        entity_type: str, 
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Generate a description for an entity based on its relationships"""
        try:
            # Find documents that mention this entity
            with self.kg_agent.driver.session() as session:
                doc_result = session.run(
                    """
                    MATCH (d:Document)-[:DESCRIBES|MENTIONS]->(e {name: $name})
                    RETURN d.title as title, d.id as id
                    LIMIT 3
                    """,
                    name=entity_name
                )
                
                docs = [f"{record['title']} (ID: {record['id']})" for record in doc_result]
                
                # Generate description based on type and relationships
                if entity_type == "Concept":
                    description = f"{entity_name} is a technical concept"
                    if docs:
                        description += f" mentioned in {', '.join(docs)}"
                    
                    # Add relationship info
                    if relationships:
                        rel_texts = []
                        for rel in relationships[:5]:
                            if rel["source"] == entity_name:
                                rel_texts.append(f"{rel['relation'].lower()} {rel['target']}")
                            else:
                                rel_texts.append(f"is {rel['relation'].lower()} by {rel['source']}")
                        
                        if rel_texts:
                            description += ". It " + ", ".join(rel_texts)
                    
                    return description
                
                elif entity_type == "Person":
                    description = f"{entity_name} is a person"
                    if docs:
                        description += f" mentioned in {', '.join(docs)}"
                    
                    # Add expertise info
                    expertise = []
                    for rel in relationships:
                        if rel["source"] == entity_name and rel["relation"] == "EXPERT_IN":
                            expertise.append(rel["target"])
                    
                    if expertise:
                        description += f". Expert in: {', '.join(expertise)}"
                    
                    return description
                
                else:
                    return f"{entity_name} is a {entity_type.lower()}"
                    
        except Exception as e:
            logger.error(f"Error generating entity description: {str(e)}")
            return f"{entity_name} ({entity_type})"
    
    async def query_related_documents(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Find documents related to an entity
        
        Args:
            entity_name: Entity name to find related documents
            
        Returns:
            List of related document data
        """
        if not self.kg_agent.driver:
            return []
            
        try:
            with self.kg_agent.driver.session() as session:
                # Query for documents related to entity
                result = session.run(
                    """
                    MATCH (d:Document)-[r]->(e {name: $name})
                    RETURN d.id as doc_id, d.title as title, type(r) as relation
                    UNION
                    MATCH (e {name: $name})-[r]->(d:Document)
                    RETURN d.id as doc_id, d.title as title, type(r) as relation
                    LIMIT 5
                    """,
                    name=entity_name
                )
                
                documents = []
                for record in result:
                    documents.append({
                        "doc_id": record["doc_id"],
                        "title": record["title"],
                        "relation": record["relation"]
                    })
                
                return documents
        except Exception as e:
            logger.error(f"Error querying related documents: {str(e)}")
            return []
    
    async def close(self):
        """Close Neo4j connection"""
        self.kg_agent.close()
