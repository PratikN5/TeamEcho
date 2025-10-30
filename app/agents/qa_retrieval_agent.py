import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from datetime import datetime

from app.core.config import get_settings
from app.services.bedrock_service import BedrockService
from app.services.redis_service import RedisService
from app.services.embedding_service import EmbeddingService
from app.services.elasticsearch_service import ElasticsearchService
from app.services.knowledge_graph_service import KnowledgeGraphService

settings = get_settings()
logger = logging.getLogger(__name__)

class QARetrievalAgent:
    """
    Contextual Q&A and Retrieval Agent that combines multiple knowledge sources
    to provide comprehensive answers to technical questions.
    """
    
    def __init__(
        self,
        bedrock_service: BedrockService,
        redis_service: RedisService,
        elasticsearch_service: ElasticsearchService,
        embedding_service: EmbeddingService,
        knowledge_graph_service: Optional[KnowledgeGraphService] = None
    ):
        self.bedrock_service = bedrock_service
        self.redis_service = redis_service
        self.elasticsearch_service = elasticsearch_service
        self.embedding_service = embedding_service
        self.knowledge_graph_service = knowledge_graph_service
    
    async def process_query(
        self, 
        query: str,
        session_id: str = None,
        user_id: str = None,
        include_sources: bool = True,
        max_sources: int = 5,
        follow_up_suggestions: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query by retrieving relevant context and generating an answer
        
        Args:
            query: User's question
            session_id: Chat session ID for context tracking
            user_id: User ID for personalization
            include_sources: Whether to include sources in the response
            max_sources: Maximum number of sources to include
            follow_up_suggestions: Whether to suggest follow-up questions
            
        Returns:
            Dictionary with answer, sources, and follow-up questions
        """
        try:
            # Generate a session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
                
            # Get chat history for context
            chat_history = await self._get_chat_history(session_id)
            
            # Analyze query to determine retrieval strategy
            query_type, entities = await self._analyze_query(query)
            
            # Retrieve relevant context based on query type
            context, sources = await self._retrieve_context(
                query=query,
                query_type=query_type,
                entities=entities,
                max_sources=max_sources
            )
            
            # Generate answer using LLM with retrieved context
            answer, reasoning, citations = await self._generate_answer(
                query=query,
                context=context,
                chat_history=chat_history,
                include_citations=include_sources
            )
            
            # Generate follow-up questions if requested
            follow_ups = []
            if follow_up_suggestions:
                follow_ups = await self._generate_follow_up_questions(
                    query=query,
                    answer=answer,
                    context=context,
                    chat_history=chat_history
                )
            
            # Store interaction in chat history
            await self._store_chat_interaction(
                session_id=session_id,
                user_id=user_id,
                query=query,
                answer=answer,
                context=context,
                sources=sources
            )
            
            # Prepare response
            response = {
                "answer": answer,
                "reasoning": reasoning,
                "session_id": session_id
            }
            
            if include_sources and sources:
                response["sources"] = self._format_sources(sources, citations)
                
            if follow_up_suggestions and follow_ups:
                response["follow_up_questions"] = follow_ups
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "error": f"Failed to process query: {str(e)}",
                "answer": "I'm sorry, I encountered an error while processing your question. Please try again or rephrase your question."
            }
    
    async def _analyze_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Analyze the query to determine the best retrieval strategy and extract entities
        
        Args:
            query: User's question
            
        Returns:
            Tuple of (query_type, entities)
        """
        # Use LLM to analyze the query
        analysis_prompt = f"""
        Analyze the following user query to determine the query type and extract entities:
        
        Query: {query}
        
        Determine the primary query type from these categories:
        - CODE_EXPLANATION: Asking about code functionality, implementation, or best practices
        - DOCUMENTATION: Asking about documentation, specifications, or requirements
        - TECHNICAL_CONCEPT: Asking about a technical concept, architecture, or design pattern
        - TROUBLESHOOTING: Asking about an error, bug, or how to fix something
        - COMPARISON: Asking to compare different technologies, approaches, or implementations
        - RECOMMENDATION: Asking for recommendations or best practices
        
        Also extract any entities mentioned in the query, such as:
        - Code elements (functions, classes, methods, variables)
        - Technical concepts
        - Technologies or frameworks
        - File names or paths
        - Error messages
        
        Format your response as a JSON object with these fields:
        - query_type: The primary query type
        - entities: Array of objects with "text" and "type" fields
        
        JSON response:
        """
        
        try:
            analysis_response = await self.bedrock_service.get_completion(analysis_prompt)
            
            # Extract JSON from response
            analysis_json = self._extract_json_from_response(analysis_response)
            
            if not analysis_json or "query_type" not in analysis_json:
                # Fallback if parsing fails
                return "TECHNICAL_CONCEPT", []
            
            return analysis_json["query_type"], analysis_json.get("entities", [])
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            # Default fallback
            return "TECHNICAL_CONCEPT", []
    
    async def _retrieve_context(
        self,
        query: str,
        query_type: str,
        entities: List[Dict[str, Any]],
        max_sources: int = 5
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant context from multiple sources based on query type
        
        Args:
            query: User's question
            query_type: Type of query (from _analyze_query)
            entities: Entities extracted from query
            max_sources: Maximum number of sources to retrieve
            
        Returns:
            Tuple of (combined_context, sources)
        """
        sources = []
        combined_context = ""
        
        # Extract query embedding for vector search
        query_embedding = await self._get_query_embedding(query)
        
        # Adjust retrieval strategy based on query type
        if query_type in ["CODE_EXPLANATION", "TROUBLESHOOTING"]:
            # Prioritize code search and vector search
            code_results = await self._search_code_index(query, entities)
            sources.extend(code_results[:max_sources//2])
            
            # Vector search for documentation
            if query_embedding:
                vector_results = await self._vector_search(
                    query_embedding, 
                    filter_str="@source:{documentation}" if query_type == "CODE_EXPLANATION" else "*"
                )
                sources.extend(vector_results[:max_sources - len(sources)])
            
            # Knowledge graph for related concepts
            kg_results = await self._query_knowledge_graph(entities)
            sources.extend(kg_results[:max_sources - len(sources)])
            
        elif query_type in ["DOCUMENTATION", "TECHNICAL_CONCEPT"]:
            # Prioritize vector search and knowledge graph
            if query_embedding:
                vector_results = await self._vector_search(query_embedding)
                sources.extend(vector_results[:max_sources//2])
            
            # Knowledge graph for concepts and relationships
            kg_results = await self._query_knowledge_graph(entities)
            sources.extend(kg_results[:max_sources//2])
            
            # Code search for implementation examples
            code_results = await self._search_code_index(query, entities)
            sources.extend(code_results[:max_sources - len(sources)])
            
        elif query_type in ["COMPARISON", "RECOMMENDATION"]:
            # Balanced approach with all sources
            if query_embedding:
                vector_results = await self._vector_search(query_embedding)
                sources.extend(vector_results[:max_sources//3])
            
            kg_results = await self._query_knowledge_graph(entities)
            sources.extend(kg_results[:max_sources//3])
            
            code_results = await self._search_code_index(query, entities)
            sources.extend(code_results[:max_sources - len(sources)])
        
        else:
            # Default strategy for other query types
            if query_embedding:
                vector_results = await self._vector_search(query_embedding)
                sources.extend(vector_results[:max_sources])
        
        # Deduplicate sources
        unique_sources = []
        seen_ids = set()
        for source in sources:
            if source["id"] not in seen_ids:
                seen_ids.add(source["id"])
                unique_sources.append(source)
        
        # Build combined context from unique sources
        for i, source in enumerate(unique_sources[:max_sources]):
            source_text = source.get("text", "")
            source_type = source.get("type", "document")
            
            # Format based on source type
            if source_type == "code":
                combined_context += f"\nSOURCE {i+1} [CODE - {source.get('file_path', 'Unknown')}]:\n{source_text}\n"
            elif source_type == "knowledge_graph":
                combined_context += f"\nSOURCE {i+1} [CONCEPT - {source.get('name', 'Unknown')}]:\n{source_text}\n"
            else:
                combined_context += f"\nSOURCE {i+1} [DOCUMENT - {source.get('title', 'Unknown')}]:\n{source_text}\n"
        
        return combined_context, unique_sources[:max_sources]
    
    async def _search_code_index(
        self, 
        query: str, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search code index using Elasticsearch"""
        try:
            # Extract code-related entities
            code_entities = [e["text"] for e in entities if e.get("type") in 
                           ["function", "class", "method", "variable", "code", "file"]]
            
            # Combine query with code entities for better search
            search_query = query
            if code_entities:
                search_query = f"{query} {' '.join(code_entities)}"
            
            # Search code index
            results = await self.elasticsearch_service.search_code(search_query)
            
            # Format results
            formatted_results = []
            for result in results:
                # Extract code snippets from highlights or functions
                code_snippet = ""
                if "highlights" in result and "content" in result["highlights"]:
                    code_snippet = "\n".join(result["highlights"]["content"])
                elif "functions" in result and result["functions"]:
                    # Get first function as sample
                    func_name = result["functions"][0]
                    code_snippet = f"Function: {func_name}"
                
                formatted_results.append({
                    "id": result["id"],
                    "title": result["filename"],
                    "text": code_snippet,
                    "file_path": result["file_path"],
                    "language": result["language"],
                    "type": "code",
                    "score": result.get("score", 0)
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching code index: {str(e)}")
            return []
    
    async def _vector_search(
        self,
        query_embedding: List[float],
        filter_str: str = "*",
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search vector database using query embedding"""
        try:
            # Search chunks first for more specific context
            chunk_results = await self.redis_service.vector_search(
                query_embedding=query_embedding,
                index_name="chunk_vector_idx",
                filter_str=filter_str,
                k=k
            )
            
            # Format chunk results
            formatted_results = []
            for result in chunk_results:
                doc_id = result["metadata"].get("doc_id", "unknown")
                formatted_results.append({
                    "id": result["id"],
                    "title": f"Document {doc_id}",
                    "text": result["text"],
                    "type": "document",
                    "chunk_id": result["metadata"].get("chunk_id"),
                    "doc_id": doc_id,
                    "score": result["score"]
                })
            
            # If not enough chunk results, search documents
            if len(formatted_results) < k:
                doc_results = await self.redis_service.vector_search(
                    query_embedding=query_embedding,
                    index_name="doc_vector_idx",
                    filter_str=filter_str,
                    k=k - len(formatted_results)
                )
                
                # Format document results
                for result in doc_results:
                    formatted_results.append({
                        "id": result["id"],
                        "title": result["metadata"].get("filename", "Unknown"),
                        "text": result["text"] if "text" in result else "Document content not available",
                        "type": "document",
                        "doc_id": result["metadata"].get("doc_id"),
                        "score": result["score"]
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    async def _query_knowledge_graph(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Query knowledge graph for relevant concepts and relationships"""
        if not self.knowledge_graph_service or not entities:
            return []
            
        try:
            # Extract entity names for knowledge graph query
            entity_names = [e["text"] for e in entities]
            
            # Query knowledge graph for these entities
            kg_results = await self.knowledge_graph_service.query_entities(entity_names)
            
            # Format results
            formatted_results = []
            for result in kg_results:
                # Get entity description and relationships
                description = result.get("description", "")
                relationships = result.get("relationships", [])
                
                # Format relationships as text
                rel_text = ""
                for rel in relationships[:5]:  # Limit to 5 relationships
                    rel_text += f"- {rel['source']} {rel['relation']} {rel['target']}\n"
                
                # Combine description and relationships
                text = f"{description}\n\nRelationships:\n{rel_text}" if rel_text else description
                
                formatted_results.append({
                    "id": f"kg:{result['id']}",
                    "name": result["name"],
                    "text": text,
                    "type": "knowledge_graph",
                    "entity_type": result.get("type", "concept"),
                    "score": 1.0  # Knowledge graph results are explicitly requested
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {str(e)}")
            return []
    
    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for the query text"""
        try:
            # Use the embedding service through Redis service
            # This assumes your RedisService has a method to get embeddings
            embedding = await self.embedding_service.get_embedding(query)
            return embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {str(e)}")
            return None
    
    async def _generate_answer(
        self,
        query: str,
        context: str,
        chat_history: List[Dict[str, str]],
        include_citations: bool = True
    ) -> Tuple[str, str, List[int]]:
        """
        Generate an answer using the LLM with retrieved context
        
        Args:
            query: User's question
            context: Retrieved context from various sources
            chat_history: Previous conversation history
            include_citations: Whether to include source citations
            
        Returns:
            Tuple of (answer, reasoning, citations)
        """
        # Format chat history for context
        history_text = ""
        if chat_history:
            for i, exchange in enumerate(chat_history[-3:]):  # Last 3 exchanges
                history_text += f"User: {exchange['user_message']}\n"
                history_text += f"Assistant: {exchange['ai_response']}\n\n"
        
        # Build the prompt
        system_prompt = """You are a technical assistant specializing in software development, documentation, and technical concepts. 
        You help developers understand code, documentation, and technical concepts.
        
        When answering questions:
        1. Be precise and technical but explain complex concepts clearly
        2. Include code examples when relevant
        3. Cite your sources using [SOURCE X] notation when referencing information
        4. If you're unsure or the context doesn't contain relevant information, acknowledge the limitations
        5. Provide step-by-step reasoning when explaining technical concepts or code
        6. Focus on practical, actionable information
        
        Your goal is to help the user understand technical concepts and solve problems efficiently."""
        
        # Prepare the optional history section outside the f-string to avoid backslashes in { } expressions
        if history_text:
            history_section = "Here's our recent conversation for context:\n" + history_text
        else:
            history_section = ""

        user_prompt = f"""I need help with the following question:

        {query}

        {history_section}
        
        I've retrieved the following relevant information to help answer this question:
        
        {context}
        
        Please answer the question based on the information provided. Include your step-by-step reasoning and cite sources using [SOURCE X] notation where X is the source number. If the information doesn't fully answer the question, acknowledge the limitations.
        
        {"Include citations to the relevant sources using [SOURCE X] notation." if include_citations else ""}
        """
        
        # Generate the answer
        try:
            # Prepare the messages for the Bedrock service
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get completion from Bedrock
            response = await self.bedrock_service.get_completion_with_messages(messages)
            
            # Extract answer, reasoning, and citations
            answer, reasoning, citations = self._parse_llm_response(response)
            
            return answer, reasoning, citations
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return (
                "I'm sorry, I couldn't generate a complete answer based on the available information.",
                "Error in answer generation process.",
                []
            )
    
    def _parse_llm_response(self, response: str) -> Tuple[str, str, List[int]]:
        """
        Parse the LLM response to extract answer, reasoning, and citations
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (answer, reasoning, citations)
        """
        # Extract citations
        citations = []
        citation_pattern = r'\[SOURCE\s+(\d+)\]'
        import re
        citation_matches = re.findall(citation_pattern, response)
        for match in citation_matches:
            try:
                citation_num = int(match)
                if citation_num not in citations:
                    citations.append(citation_num)
            except ValueError:
                pass
        
        # Check if response contains explicit reasoning section
        reasoning = ""
        answer = response
        
        if "REASONING:" in response and "ANSWER:" in response:
            parts = response.split("REASONING:")
            if len(parts) > 1:
                reasoning_and_answer = parts[1]
                reasoning_parts = reasoning_and_answer.split("ANSWER:")
                if len(reasoning_parts) > 1:
                    reasoning = reasoning_parts[0].strip()
                    answer = reasoning_parts[1].strip()
        elif "Step-by-step reasoning:" in response:
            parts = response.split("Step-by-step reasoning:")
            if len(parts) > 1:
                reasoning = parts[1].split("\n\n")[0].strip()
                # Keep the full response as the answer in this case
        
        return answer, reasoning, citations
    
    async def _generate_follow_up_questions(
        self,
        query: str,
        answer: str,
        context: str,
        chat_history: List[Dict[str, str]]
    ) -> List[str]:
        """Generate follow-up questions based on the query, answer, and context"""
        try:
            # Build the prompt
            prompt = f"""Based on the user's question, my answer, and the available context, suggest 3 relevant follow-up questions the user might want to ask next.
            
            User's question: {query}
            
            My answer: {answer}
            
            Context summary: {context[:500]}...
            
            Generate 3 concise, specific follow-up questions that:
            1. Explore related aspects of the topic
            2. Dive deeper into technical details mentioned
            3. Address potential implementation or practical concerns
            
            Format each question on a new line, numbered 1-3. Keep questions concise and directly related to the topic.
            """
            
            # Get completion
            response = await self.bedrock_service.get_completion(prompt)
            
            # Parse questions
            questions = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and (line.startswith("1.") or line.startswith("2.") or line.startswith("3.")):
                    # Remove the number prefix
                    question = line[2:].strip()
                    questions.append(question)
                elif line and len(questions) < 3 and "?" in line:
                    questions.append(line)
            
            # Ensure we have at most 3 questions
            return questions[:3]
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            return []
    
    async def _get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session"""
        try:
            history = await self.redis_service.get_chat_history(session_id)
            return history
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []
    
    async def _store_chat_interaction(
        self,
        session_id: str,
        user_id: str,
        query: str,
        answer: str,
        context: str,
        sources: List[Dict[str, Any]]
    ) -> bool:
        """Store the chat interaction in Redis"""
        try:
            message_id = str(uuid.uuid4())
            
            # Prepare metadata with sources and context
            metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "sources": [{"id": s["id"], "title": s.get("title", ""), "type": s.get("type", "")} for s in sources],
                "context_length": len(context)
            }
            
            # Store in Redis
            success = await self.redis_service.store_chat_memory(
                session_id=session_id,
                message_id=message_id,
                user_message=query,
                ai_response=answer,
                metadata=metadata
            )
            
            return success
        except Exception as e:
            logger.error(f"Error storing chat interaction: {str(e)}")
            return False
    
    def _format_sources(self, sources: List[Dict[str, Any]], citations: List[int]) -> List[Dict[str, Any]]:
        """Format sources for inclusion in the response"""
        formatted_sources = []
        
        # If we have specific citations, prioritize those sources
        if citations:
            # Adjust citation numbers to 0-based indices
            indices = [i-1 for i in citations if 0 <= i-1 < len(sources)]
            
            # Add cited sources first
            for idx in indices:
                if idx < len(sources):
                    source = sources[idx]
                    formatted_sources.append({
                        "id": source.get("id", ""),
                        "title": source.get("title", "Unknown"),
                        "type": source.get("type", "document"),
                        "cited": True
                    })
            
            # Add remaining sources
            for i, source in enumerate(sources):
                if i not in indices:
                    formatted_sources.append({
                        "id": source.get("id", ""),
                        "title": source.get("title", "Unknown"),
                        "type": source.get("type", "document"),
                        "cited": False
                    })
        else:
            # No specific citations, include all sources
            for source in sources:
                formatted_sources.append({
                    "id": source.get("id", ""),
                    "title": source.get("title", "Unknown"),
                    "type": source.get("type", "document"),
                    "cited": False
                })
        
        return formatted_sources
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON object from LLM response"""
        try:
            import re
            
            # Remove any markdown code block markers
            response = response.strip()
            
            # First try to find JSON in code blocks with json marker
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Try to find JSON in code blocks without json marker
            json_match = re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Try to find JSON without code block markers
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # If response is already a JSON object
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
            
            logger.warning(f"Could not extract JSON from response: {response[:200]}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error extracting JSON from response: {str(e)}")
            logger.error(f"Response was: {response[:500]}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error extracting JSON: {str(e)}")
            return {}
        
    async def test_vector_search(self, query: str):
       """Test vector search functionality"""
       embedding = await self._get_query_embedding(query)
       if embedding:
           results = await self._vector_search(embedding)
           return {
               "query": query,
               "embedding_generated": True,
               "results_count": len(results),
               "results": results
           }
       return {
           "query": query,
           "embedding_generated": False,
           "error": "Failed to generate embedding"
       }
   
    async def test_knowledge_graph(self, query: str):
       """Test knowledge graph functionality"""
       _, entities = await self._analyze_query(query)
       results = await self._query_knowledge_graph(entities)
       return {
           "query": query,
           "entities_extracted": entities,
           "results_count": len(results),
           "results": results
       }
   
    async def test_code_search(self, query: str):
       """Test code search functionality"""
       _, entities = await self._analyze_query(query)
       results = await self._search_code_index(query, entities)
       return {
           "query": query,
           "results_count": len(results),
           "results": results
       }
   