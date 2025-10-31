import logging
import json , re , numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from datetime import datetime
from app.core.config import get_settings
from app.services.bedrock_service import BedrockService
from app.services.redis_service import RedisService
from app.services.embedding_service import EmbeddingService
from app.services.elasticsearch_service import ElasticsearchService
from app.services.knowledge_graph_service import KnowledgeGraphService
from sklearn.metrics.pairwise import cosine_similarity


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
        knowledge_graph_service: Optional[KnowledgeGraphService] = None,
        llm_service: Optional[BedrockService] = None 
    ):
        self.bedrock_service = bedrock_service
        self.redis_service = redis_service
        self.elasticsearch_service = elasticsearch_service
        self.embedding_service = embedding_service
        self.knowledge_graph_service = knowledge_graph_service
        self.llm_service = llm_service or bedrock_service
    
    async def process_query(
    self,
    query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    include_sources: bool = True,
    max_sources: int = 5,
    follow_up_suggestions: bool = True
) -> Dict[str, Any]:
      """Process a query with improved context understanding and natural language support."""
      
      # Import required modules at the beginning
      import json
      import os

      # Add this new detection pattern near the beginning of your process_query function:
      is_generate_flowchart_request = re.search(r'(generate|create|make|build)(\s+a|\s+new|\s+the)?\s+(flow\s*chart|flowchart|diagram)', query.lower()) is not None

      # Then add a new condition block after your other special handlers:
      # Special handling for GENERATE flowchart/diagram requests
      if is_generate_flowchart_request:
          try:
              # Extract the subject/requirement from the query
              requirement_match = re.search(r'for\s+(.+?)(?:\.|$)', query)
              if not requirement_match:
                  requirement_match = re.search(r'(flow\s*chart|flowchart|diagram)\s+for\s+(.+?)(?:\.|$)', query)
                  requirement = requirement_match.group(2) if requirement_match else None
              else:
                  requirement = requirement_match.group(1)
              
              if not requirement:
                  # Try to extract any business requirement
                  requirement_match = re.search(r'(business requirement|requirement)\s+(?:is|to)?\s+(.+?)(?:\.|$)', query)
                  requirement = requirement_match.group(2) if requirement_match else "the requested business process"
              
              # Determine what type of diagram to generate
              diagram_type = "flowchart"
              if "sequence diagram" in query.lower():
                  diagram_type = "sequence diagram"
              elif "entity relationship" in query.lower() or "er diagram" in query.lower():
                  diagram_type = "entity relationship diagram"
              elif "class diagram" in query.lower():
                  diagram_type = "class diagram"
              elif "architecture diagram" in query.lower():
                  diagram_type = "architecture diagram"
              
              # Get relevant context for the requirement
              context_documents, context_texts = await self._retrieve_relevant_documents(requirement, max_sources=3)
              
              context = "\n\n".join([f"Document {i+1}:\n{text}" for i, text in enumerate(context_texts)])
              
              # Create prompt for generating the flowchart
              prompt = f"""
              I need to generate a {diagram_type} for the following business requirement:
              "{requirement}"
              
              Here's some context from our existing documentation that might be relevant:
              {context}
              
              Please create:
              
              1. A textual description of what the {diagram_type} should contain
              2. A mermaid.js diagram code that visualizes this {diagram_type}
              3. An explanation of the key components and flow
              
              For the mermaid.js code, use the appropriate syntax based on the diagram type:
              - For flowcharts, use flowchart TD (top-down) or LR (left-right)
              - For sequence diagrams, use the sequence syntax
              - For entity relationship diagrams, use the erDiagram syntax
              - For class diagrams, use the classDiagram syntax
              
              Make the diagram comprehensive but not overly complex, focusing on the main components and flows.
              """
              
              # Generate the flowchart and explanation
              response = await self.bedrock_service.get_completion(prompt)
              
              # Extract the mermaid code from the response
              mermaid_code = None
              if "mermaid" in response:
                  mermaid_match = re.search(r'```mermaid\n(.*?)\n```', response, re.DOTALL)
                  if mermaid_match:
                      mermaid_code = mermaid_match.group(1)
              # Format the final response
              if mermaid_code:
                  # Add a special marker for the frontend to render the mermaid diagram
                  final_response = response.replace("```mermaid\n" + mermaid_code + "\n```", 
                                                f"<mermaid>\n{mermaid_code}\n</mermaid>")
              else:
                  final_response = response
              # Create sources list for citation
              sources = []
              for doc in context_documents:
                  metadata = doc.get("metadata", {})
                  sources.append({
                      "id": doc.get("id", ""),
                      "title": metadata.get("filename", "Unknown document"),
                      "type": "document",
                      "cited": True
                  })
              return {
                  "answer": final_response,
                  "reasoning": f"Generated a new {diagram_type} for the requirement: {requirement}",
                  "session_id": session_id,
                  "sources": sources,
                  "follow_up_questions": [
                      f"Can you modify this {diagram_type} to include more details?",
                      f"Can you explain a specific part of this {diagram_type}?",
                      f"Can you generate a different type of diagram for this requirement?"
                  ] if follow_up_suggestions else []
              }
          except Exception as e:
              logger.error(f"Error generating flowchart: {str(e)}", exc_info=True)
              return {
                  "answer": f"I encountered an error while trying to generate the {diagram_type}. Please try again with more details about your requirement.",
                  "reasoning": f"Error: {str(e)}",
                  "session_id": session_id,
                  "sources": [],
                  "follow_up_questions": []
              }
      
      # First, analyze the query intent semantically
      query_intent = await self._analyze_query_intent(query)
      
      # More specific detection of visual document types
      is_flowchart_request = re.search(r'(show|display|get|see|give)(\s+the|\s+a|\s+any)?\s+(flow\s*chart|flowchart)', query.lower()) is not None
      is_diagram_request = re.search(r'(show|display|get|see|give)(\s+the|\s+a|\s+any)?\s+(diagram|architecture)', query.lower()) is not None
      is_image_request = re.search(r'(show|display|get|see|give)(\s+the|\s+a|\s+any)?\s+(image|picture|visual)', query.lower()) is not None
      is_any_visual_request = is_flowchart_request or is_diagram_request or is_image_request
      
      # Check for explanation requests specifically for visual documents
      is_explain_flowchart_request = re.search(r'(explain|describe|tell\s+me\s+about|what\s+is)(\s+the|\s+a|\s+this)?\s+(flow\s*chart|flowchart)', query.lower()) is not None
      is_explain_diagram_request = re.search(r'(explain|describe|tell\s+me\s+about|what\s+is)(\s+the|\s+a|\s+this)?\s+(diagram|architecture)', query.lower()) is not None
      is_explain_image_request = re.search(r'(explain|describe|tell\s+me\s+about|what\s+is)(\s+the|\s+a|\s+this)?\s+(image|picture|visual)', query.lower()) is not None
      is_explain_visual_request = is_explain_flowchart_request or is_explain_diagram_request or is_explain_image_request
      
      # Check if this is a request to list all documents
      is_list_all_documents_request = re.search(r'(list|show|get|display)(\s+all|\s+the)?\s+(documents|files|docs)', query.lower()) is not None
      
      # Extract the subject from the query (what the flowchart/diagram is about)
      subject_match = re.search(r'(?:of|about|for|on)\s+(\w+)', query.lower())
      subject = subject_match.group(1) if subject_match else None
      
      # Check for source specification in the query
      source_match = re.search(r'(?:from|in) (sharepoint|onedrive|google drive|local|database)', query.lower())
      source_filter = None
      source_name = None
      
      if source_match:
          source_name = source_match.group(1)
          # Map common source names to your metadata field values
          source_mapping = {
              "sharepoint": "sharepoint",
              "onedrive": "onedrive",
              "google drive": "google_drive",
              "local": "local_upload",
              "database": "database"
          }
          
          if source_name in source_mapping:
              # Create filter for ChromaDB using $eq operator
              source_filter = {"source": {"$eq": source_mapping[source_name]}}
              logger.info(f"Created source filter: {source_filter}")
      
      # Special handling for EXPLAIN visual document requests
      if is_explain_visual_request:
          try:
              # Create a filter for image files
              image_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/svg+xml"]
              
              # Use filename extension as fallback
              image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".svg"]
              
              # Get images based on mime_type or filename
              filter_dict = None
              
              # Try to find images with proper mime type first
              mime_type_filter = {"mime_type": {"$in": image_types}}
              
              # Combine with source filter if it exists
              if source_filter:
                  filter_dict = {
                      "$and": [
                          mime_type_filter,
                          source_filter
                      ]
                  }
              else:
                  filter_dict = mime_type_filter
              
              logger.info(f"Using filter for images: {filter_dict}")
              
              # Get all image documents
              image_documents = await self._get_documents_from_chroma(
                  filter_dict=filter_dict,
                  limit=20  # Get more documents to filter from
              )
              
              # If no results with mime_type, try filename extensions
              if not image_documents:
                  logger.info("No images found with mime_type filter, trying filename extensions")
                  # Since $contains isn't supported, we'll get all documents and filter manually
                  all_docs = await self._get_documents_from_chroma(
                      filter_dict=source_filter,  # Only apply source filter if it exists
                      limit=50
                  )
                  
                  # Filter for image extensions manually
                  image_documents = []
                  for doc in all_docs:
                      filename = doc.get("metadata", {}).get("filename", "").lower()
                      if any(filename.endswith(ext) for ext in image_extensions):
                          image_documents.append(doc)
              
              # Apply more specific filtering based on request type
              filtered_documents = []
              
              if is_explain_flowchart_request:
                  # Only include documents with "flowchart" or "flow chart" in the filename
                  filtered_documents = [doc for doc in image_documents if 
                                      "flowchart" in doc.get("metadata", {}).get("filename", "").lower() or 
                                      "flow chart" in doc.get("metadata", {}).get("filename", "").lower() or
                                      "flow-chart" in doc.get("metadata", {}).get("filename", "").lower()]
                  doc_type_description = "flowchart"
              elif is_explain_diagram_request:
                  # Only include documents with "diagram" or "architecture" in the filename
                  filtered_documents = [doc for doc in image_documents if 
                                      "diagram" in doc.get("metadata", {}).get("filename", "").lower() or 
                                      "architecture" in doc.get("metadata", {}).get("filename", "").lower()]
                  doc_type_description = "diagram"
              else:
                  # For general image requests, use all image documents
                  filtered_documents = image_documents
                  doc_type_description = "image"
              
              # Further filter by subject if provided
              if subject and filtered_documents:
                  subject_filtered = [doc for doc in filtered_documents if 
                                    subject.lower() in doc.get("metadata", {}).get("filename", "").lower()]
                  # Only use subject filtering if it doesn't eliminate all results
                  if subject_filtered:
                      filtered_documents = subject_filtered
              
              if not filtered_documents:
                  source_phrase = f" from {source_name}" if source_name else ""
                  subject_phrase = f" of {subject}" if subject else ""
                  return {
                      "answer": f"I couldn't find any {doc_type_description}{subject_phrase}{source_phrase} to explain. Would you like to see a list of available visual documents instead?",
                      "reasoning": f"No {doc_type_description} found{subject_phrase}{source_phrase}.",
                      "session_id": session_id,
                      "sources": [],
                      "follow_up_questions": [
                          "Would you like to see all available visual documents?",
                          "Would you like to search for other types of documents?"
                      ] if follow_up_suggestions else []
                  }
              
              # Now we have the visual document, create a response that shows it and explains it
              visual_doc = filtered_documents[0]  # Take the first matching document
              metadata = visual_doc.get("metadata", {})
              file_path = metadata.get("file_path", "")
              filename = metadata.get("filename", "Untitled")
              doc_id = visual_doc.get("id", "")
              
              # Create a source entry for the visual document
              visual_source = {
                  "id": doc_id,
                  "title": filename,
                  "type": "image",
                  "file_path": file_path,
                  "cited": True,
                  "display": True
              }
              
              # Now find text documents that might help explain this visual
              # Use the filename (without extension) as search term
              import os
              filename_base = os.path.splitext(filename)[0] if '.' in filename else filename
              
              # If subject is provided, use it to enhance the search
              search_terms = []
              if subject:
                  search_terms.append(subject)
              
              # Add document type and filename base to search terms
              if is_explain_flowchart_request:
                  search_terms.extend(["flowchart", "flow chart", "process flow"])
              elif is_explain_diagram_request:
                  search_terms.extend(["diagram", "architecture", "system design"])
              
              search_terms.append(filename_base)
              
              # Create search query
              explain_query = " ".join(search_terms)
              
              # Get relevant text documents
              text_documents, document_texts = await self._retrieve_relevant_documents(explain_query, max_sources=3)
              
              # Replace the sources creation section with this more flexible solution:
              sources = [visual_source]  # Always include the primary visual document

              # Check if the query specifically asks about this exact document
              filename_lower = visual_source["title"].lower()
              query_lower = query.lower()

              # If the query contains the specific filename or a unique identifier for this document
              is_specific_document_request = (
                  filename_lower in query_lower or
                  # Check for other unique identifiers that might be in the query
                  (doc_id in query_lower) or
                  # If the query contains very specific details about this document
                  (is_explain_flowchart_request and "flowchart" in filename_lower) or
                  (is_explain_diagram_request and "diagram" in filename_lower) or
                  (is_explain_image_request and any(ext in filename_lower for ext in [".png", ".jpg", ".jpeg", ".gif"]))
              )

              # Only add supporting text documents if this isn't a request specifically about the visual
              if not is_specific_document_request:
                  # Add supporting documents for context but mark them as not directly cited
                  for doc in text_documents:
                      metadata = doc.get("metadata", {})
                      sources.append({
                          "id": doc.get("id", ""),
                          "title": metadata.get("filename", "Unknown document"),
                          "type": "document",
                          "cited": True  # You could set this to False if you want them to appear but not as citations
                      })

              
              # Create a prompt for the LLM to explain the visual
              context = "\n\n".join([f"Document {i+1}:\n{text}" for i, text in enumerate(document_texts)])
              
              # Create a prompt for the LLM to explain the visual with better structure and light interactive elements
              prompt = f"""
              I need you to explain a {doc_type_description} named "{filename}".
              Here's some context from related documents that might help with the explanation:
              {context}

              Please provide a detailed explanation of what this {doc_type_description} shows, using a clear, structured format with the following sections:

              1. OVERVIEW: A brief introduction explaining what this {doc_type_description} represents (1-2 sentences).

              2. KEY COMPONENTS: List and describe each major component shown in the {doc_type_description}.
                - Format as bullet points
                - Name each component clearly
                - Provide a brief description of each component's purpose

              3. FLOW/PROCESS: Explain the sequence of interactions or steps shown in the {doc_type_description}.
                - Number the steps in sequence
                - Explain what happens at each step
                - Highlight any conditional paths or decision points

              4. PURPOSE & SIGNIFICANCE: Explain why this {doc_type_description} is important and how it relates to the overall system.

              After providing the explanation, end with a brief, conversational note suggesting 1-2 follow-up questions or next steps, such as:
              These questions are for reference - "Would you like me to explain how you might implement this in your codebase? Or would you like more details about any specific component in this diagram?"

              Format your response with clear section headers, bullet points for lists, and numbered steps for processes.
              Use markdown formatting to enhance readability (bold for component names, etc.).
              """

              # Generate explanation using LLM
              explanation = await self.bedrock_service.get_completion(prompt)

              # Create a response that displays the image and provides the structured explanation
              response_text = f"# {filename}\n\n"
              response_text += f"![{filename}](image:{doc_id})\n\n"
              response_text += explanation  # The explanation now includes its own headers and structure

              
              return {
                  "answer": response_text,
                  "reasoning": f"Retrieved and explained {doc_type_description} '{filename}'.",
                  "session_id": session_id,
                  "sources": sources,
                  "follow_up_questions": [
                      f"What is the purpose of this {doc_type_description}?",
                      f"How does this {doc_type_description} relate to the overall system?",
                      f"Can you explain a specific part of this {doc_type_description} in more detail?"
                  ] if follow_up_suggestions else []
              }
          except Exception as e:
              logger.error(f"Error explaining visual document: {str(e)}", exc_info=True)
              return {
                  "answer": f"I encountered an error while trying to explain the {doc_type_description}. Please try again.",
                  "reasoning": f"Error: {str(e)}",
                  "session_id": session_id,
                  "sources": [],
                  "follow_up_questions": []
              }
      
      # Special handling for SHOW image requests
      if is_any_visual_request:
          try:
              # Create a filter for image files
              image_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/svg+xml"]
              
              # Use filename extension as fallback
              image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".svg"]
              
              # Get images based on mime_type or filename
              filter_dict = None
              
              # Try to find images with proper mime type first
              mime_type_filter = {"mime_type": {"$in": image_types}}
              
              # Combine with source filter if it exists
              if source_filter:
                  filter_dict = {
                      "$and": [
                          mime_type_filter,
                          source_filter
                      ]
                  }
              else:
                  filter_dict = mime_type_filter
              
              logger.info(f"Using filter for images: {filter_dict}")
              
              # Get all image documents
              image_documents = await self._get_documents_from_chroma(
                  filter_dict=filter_dict,
                  limit=20  # Get more documents to filter from
              )
              
              # If no results with mime_type, try filename extensions
              if not image_documents:
                  logger.info("No images found with mime_type filter, trying filename extensions")
                  # Since $contains isn't supported, we'll get all documents and filter manually
                  all_docs = await self._get_documents_from_chroma(
                      filter_dict=source_filter,  # Only apply source filter if it exists
                      limit=50
                  )
                  
                  # Filter for image extensions manually
                  image_documents = []
                  for doc in all_docs:
                      filename = doc.get("metadata", {}).get("filename", "").lower()
                      if any(filename.endswith(ext) for ext in image_extensions):
                          image_documents.append(doc)
              
              # Apply more specific filtering based on request type
              filtered_documents = []
              
              if is_flowchart_request:
                  # Only include documents with "flowchart" or "flow chart" in the filename
                  filtered_documents = [doc for doc in image_documents if 
                                      "flowchart" in doc.get("metadata", {}).get("filename", "").lower() or 
                                      "flow chart" in doc.get("metadata", {}).get("filename", "").lower() or
                                      "flow-chart" in doc.get("metadata", {}).get("filename", "").lower()]
                  doc_type_description = "flowchart"
              elif is_diagram_request:
                  # Only include documents with "diagram" or "architecture" in the filename
                  filtered_documents = [doc for doc in image_documents if 
                                      "diagram" in doc.get("metadata", {}).get("filename", "").lower() or 
                                      "architecture" in doc.get("metadata", {}).get("filename", "").lower()]
                  doc_type_description = "diagram"
              else:
                  # For general image requests, use all image documents
                  filtered_documents = image_documents
                  doc_type_description = "image"
              
              # Further filter by subject if provided
              if subject and filtered_documents:
                  subject_filtered = [doc for doc in filtered_documents if 
                                    subject.lower() in doc.get("metadata", {}).get("filename", "").lower()]
                  # Only use subject filtering if it doesn't eliminate all results
                  if subject_filtered:
                      filtered_documents = subject_filtered
              
              if not filtered_documents:
                  source_phrase = f" from {source_name}" if source_name else ""
                  subject_phrase = f" of {subject}" if subject else ""
                  return {
                      "answer": f"I couldn't find any {doc_type_description}{subject_phrase}{source_phrase} in the system. Would you like to see a list of all available images instead?",
                      "reasoning": f"No {doc_type_description}{subject_phrase} found{source_phrase}.",
                      "session_id": session_id,
                      "sources": [],
                      "follow_up_questions": [
                          "Would you like to see all available images?",
                          "Would you like to see all available documents?",
                          "Would you like to upload some images?"
                      ] if follow_up_suggestions else []
                  }
              
              # Format response with actual image display
              image_info = []
              for doc in filtered_documents:
                  metadata = doc.get("metadata", {})
                  file_path = metadata.get("file_path", "")
                  filename = metadata.get("filename", "Untitled")
                  
                  image_info.append({
                      "title": filename,
                      "type": metadata.get("mime_type", "Unknown"),
                      "id": doc.get("id", ""),
                      "file_path": file_path
                  })
              
              # Create sources list for citation and image display
              sources = []
              for doc in filtered_documents:
                  metadata = doc.get("metadata", {})
                  file_path = metadata.get("file_path", "")
                  
                  sources.append({
                      "id": doc.get("id", ""),
                      "title": metadata.get("filename", "Unknown image"),
                      "type": "image",
                      "file_path": file_path,
                      "cited": True,
                      "display": True  # Flag to indicate this should be displayed
                  })
              
              # Create a response that displays the images
              source_phrase = f" from {source_name}" if source_name else ""
              subject_phrase = f" of {subject}" if subject else ""
              
              if is_flowchart_request:
                  if len(filtered_documents) == 1:
                      response_text = f"Here's the flowchart{subject_phrase}{source_phrase} I found:\n\n"
                  else:
                      response_text = f"Here are the flowcharts{subject_phrase}{source_phrase} I found:\n\n"
              elif is_diagram_request:
                  if len(filtered_documents) == 1:
                      response_text = f"Here's the diagram{subject_phrase}{source_phrase} I found:\n\n"
                  else:
                      response_text = f"Here are the diagrams{subject_phrase}{source_phrase} I found:\n\n"
              else:
                  response_text = f"Here are the images{subject_phrase}{source_phrase} I found:\n\n"
              
              # Add image references - these will be rendered by the frontend
              for i, img in enumerate(image_info):
                  response_text += f"**Image {i+1}**: {img['title']}\n"
                  response_text += f"![{img['title']}](image:{img['id']})\n\n"
              
              return {
                  "answer": response_text,
                  "reasoning": f"Found {len(filtered_documents)} {doc_type_description}s{subject_phrase}{source_phrase}.",
                  "session_id": session_id,
                  "sources": sources,
                  "follow_up_questions": [
                      f"Would you like me to explain this {doc_type_description}?",
                      "Would you like to see other types of documents?",
                      f"Would you like to search for related {doc_type_description}s?"
                  ] if follow_up_suggestions else []
              }
          except Exception as e:
              logger.error(f"Error processing image request: {str(e)}", exc_info=True)
              return {
                  "answer": "I encountered an error while trying to retrieve images. Please try again.",
                  "reasoning": f"Error: {str(e)}",
                  "session_id": session_id,
                  "sources": [],
                  "follow_up_questions": []
              }
      
      # Special handling for document listing requests
      if is_list_all_documents_request:
          try:
              # Get all documents (with optional source filter)
              all_documents = await self._get_documents_from_chroma(
                  filter_dict=source_filter,
                  limit=50
              )
              
              # Format document information
              document_info = []
              for doc in all_documents:
                  metadata = doc.get("metadata", {})
                  doc_info = {
                      "title": metadata.get("filename", "Untitled"),
                      "type": metadata.get("mime_type", "Unknown"),
                      "id": doc.get("id", ""),
                      "source": metadata.get("source", "Unknown")  # Include source in doc info
                  }
                  document_info.append(doc_info)
              
              # Create prompt for listing documents
              source_phrase = f" from {source_name}" if source_name else ""
              prompt = f"""
              The user has requested to see documents{source_phrase} with the query: "{query}"
              
              Here are the documents that match this request:
              {json.dumps(document_info, indent=2)}
              
              Create a response that:
              1. Starts with "Here are the documents{source_phrase}:"
              2. Lists all the documents in a clear table format with columns for Title and Type
              3. Mentions the total count of documents{' from ' + source_name if source_name else ''}
              4. Uses markdown formatting for the table
              5. If no documents were found from the requested source, clearly state that
              
              Keep your response focused and concise.
              """
              
              # Generate response using LLM
              response_text = await self.bedrock_service.get_completion(prompt)
              
              # Create sources list for citation
              sources = []
              for doc in all_documents:
                  metadata = doc.get("metadata", {})
                  sources.append({
                      "id": doc.get("id", ""),
                      "title": metadata.get("filename", "Unknown document"),
                      "type": "document",
                      "cited": True
                  })
              
              # Specific follow-up questions for document listing
              follow_up_questions = [
                  "Would you like to see more details about any specific document?",
                  "Would you like to search for documents on a particular topic?",
                  "Would you like to filter documents by type?"
              ]
              
              # Customize reasoning based on source filter
              if source_name:
                  reasoning = f"Listed {len(all_documents)} documents from {source_name}."
              else:
                  reasoning = f"Listed all {len(all_documents)} documents in the system."
              
              return {
                  "answer": response_text,
                  "reasoning": reasoning,
                  "session_id": session_id,
                  "sources": sources,
                  "follow_up_questions": follow_up_questions
              }
          except Exception as e:
              logger.error(f"Error in document listing: {str(e)}", exc_info=True)
              return {
                  "answer": "I encountered an error while trying to list the documents. Please try again.",
                  "reasoning": f"Error in document listing: {str(e)}",
                  "session_id": session_id,
                  "sources": [],
                  "follow_up_questions": []
              }
      
      # ============================================================
      # MAIN NATURAL LANGUAGE PROCESSING - HANDLES ALL OTHER QUERIES
      # ============================================================
      try:
          # Step 1: Retrieve relevant documents using semantic search
          documents, document_texts = await self._retrieve_relevant_documents(query, max_sources)
          
          # Step 2: Check if we found any relevant documents
          if not documents or len(documents) == 0:
              # No documents found - provide helpful fallback
              return {
                  "answer": "I couldn't find any documents directly related to your question. However, I can help you with:\n\n- Listing all available documents\n- Searching for specific topics\n- Explaining flowcharts or diagrams\n- Generating new flowcharts for business requirements\n\nWhat would you like to do?",
                  "reasoning": "No relevant documents found for the query.",
                  "session_id": session_id,
                  "sources": [],
                  "follow_up_questions": [
                      "Would you like to see all available documents?",
                      "Can you rephrase your question with more context?",
                      "Would you like to search for something specific?"
                  ] if follow_up_suggestions else []
              }
          
          # Step 3: Determine if the query is specifically about visual content (flowcharts, diagrams, images)
          # even if not using specific keywords like "show" or "explain"
          query_mentions_visual = any(keyword in query.lower() for keyword in [
              'flowchart', 'flow chart', 'diagram', 'architecture', 'image', 'visual',
              'picture', 'chart', 'graph', 'illustration', 'drawing'
          ])
          
          # Step 4: If query mentions visual content, check if we have image documents
          if query_mentions_visual:
              # Filter for image documents from the retrieved documents
              image_docs = [doc for doc in documents if 
                          doc.get("metadata", {}).get("mime_type", "").startswith("image/")]
              
              # If we have image documents, prioritize them in the response
              if image_docs:
                  logger.info(f"Found {len(image_docs)} image documents for visual query")
                  
                  # Create sources with image documents
                  sources = []
                  for doc in image_docs[:3]:  # Limit to top 3 images
                      metadata = doc.get("metadata", {})
                      sources.append({
                          "id": doc.get("id", ""),
                          "title": metadata.get("filename", "Unknown image"),
                          "type": "image",
                          "file_path": metadata.get("file_path", ""),
                          "cited": True,
                          "display": True
                      })
                  
                  # Add text documents as supporting sources
                  for doc in documents:
                      if not doc.get("metadata", {}).get("mime_type", "").startswith("image/"):
                          metadata = doc.get("metadata", {})
                          sources.append({
                              "id": doc.get("id", ""),
                              "title": metadata.get("filename", "Unknown document"),
                              "type": "document",
                              "cited": True
                          })
                  
                  # Generate response that includes the images
                  context = "\n\n".join([f"Document {i+1}:\n{text}" for i, text in enumerate(document_texts)])
                  
                  prompt = f"""
                  User's question: "{query}"
                  
                  I found the following visual content and related documentation:
                  
                  Visual Documents:
                  {json.dumps([{"filename": doc.get("metadata", {}).get("filename", ""), "id": doc.get("id", "")} for doc in image_docs[:3]], indent=2)}
                  
                  Related Documentation Context:
                  {context}
                  
                  Please provide a comprehensive answer that:
                  1. Directly addresses the user's question about the visual content
                  2. References the specific flowchart/diagram/image by name
                  3. Uses information from the related documentation to provide detailed context
                  4. Explains any technical details, processes, or components shown
                  5. Is conversational and natural, not overly structured
                  
                  Important: The images will be displayed separately, so focus on explaining what they show and answering the user's specific question.
                  """
                  
                  response_text = await self.bedrock_service.get_completion(prompt)
                  
                  # Prepend image references
                  images_section = "\n\n"
                  for i, img_doc in enumerate(image_docs[:3]):
                      filename = img_doc.get("metadata", {}).get("filename", "Untitled")
                      doc_id = img_doc.get("id", "")
                      images_section += f"**{filename}**\n"
                      images_section += f"![{filename}](image:{doc_id})\n\n"
                  
                  final_response = images_section + response_text
                  
                  return {
                      "answer": final_response,
                      "reasoning": f"Found {len(image_docs)} relevant visual documents and provided contextual answer.",
                      "session_id": session_id,
                      "sources": sources,
                      "follow_up_questions": [
                          "Can you explain a specific part of this in more detail?",
                          "How does this relate to other components in the system?",
                          "Would you like to see related documentation?"
                      ] if follow_up_suggestions else []
                  }
          
          # Step 5: For all other queries (including modifications, questions about content, etc.)
          # Generate a contextual response using the retrieved documents
          
          # Create sources list
          sources = []
          for doc in documents:
              metadata = doc.get("metadata", {})
              doc_type = "image" if metadata.get("mime_type", "").startswith("image/") else "document"
              
              source_entry = {
                  "id": doc.get("id", ""),
                  "title": metadata.get("filename", "Unknown document"),
                  "type": doc_type,
                  "cited": True
              }
              
              # Add file_path and display flag for images
              if doc_type == "image":
                  source_entry["file_path"] = metadata.get("file_path", "")
                  source_entry["display"] = False  # Don't auto-display unless specifically requested
              
              sources.append(source_entry)
          
          # Build comprehensive context from all retrieved documents
          context = "\n\n".join([f"Document {i+1} ({documents[i].get('metadata', {}).get('filename', 'Unknown')}):\n{text}" 
                                for i, text in enumerate(document_texts)])
          
          # Create an intelligent prompt that handles various query types
          prompt = f"""
          User's question: "{query}"
          
          Available context from the knowledge base:
          {context}
          
          Based on the context provided, please answer the user's question comprehensively and naturally.
          
          Guidelines:
          1. If the question is about modifying something (e.g., "can we modify the flowchart"):
            - Confirm what can be modified based on the context
            - Explain the current state from the documents
            - Suggest what changes are possible
            - If specific technical details are needed, reference them from the context
          
          2. If the question is asking for information or explanation:
            - Provide a detailed answer using the context
            - Reference specific documents or sections when relevant
            - Be thorough but conversational
          
          3. If the question is about capabilities or possibilities:
            - Explain what's currently implemented (from the context)
            - Discuss what modifications or additions are possible
            - Provide practical guidance
          
          4. If the context doesn't fully answer the question but is related:
            - Answer what you can from the available context
            - Be honest about any gaps
            - Suggest related information that might be helpful
          
          Important:
          - Write in a natural, conversational tone
          - Don't use overly structured formats unless the question asks for it
          - Reference specific filenames or documents when it adds clarity
          - Be direct and helpful
          - If you mention making changes or modifications, base it on what exists in the context
          
          Provide your complete answer now:
          """
          
          # Generate the response
          response_text = await self.bedrock_service.get_completion(prompt)
          
          # Generate contextual follow-up questions based on the query type
          follow_up_prompt = f"""
          Based on the user's question: "{query}"
          And the answer provided, suggest 3 relevant follow-up questions that:
          1. Dive deeper into specific aspects mentioned
          2. Explore related topics from the context
          3. Help the user take next steps
          
          Format as a simple JSON array of strings. Only return the JSON, nothing else.
          Example: ["Question 1?", "Question 2?", "Question 3?"]
          """
          
          follow_up_response = await self.bedrock_service.get_completion(follow_up_prompt)
          
          # Parse follow-up questions
          try:
              # Extract JSON array from response
              import json
              follow_up_questions = json.loads(follow_up_response)
              if not isinstance(follow_up_questions, list):
                  follow_up_questions = []
          except:
              # Fallback to generic questions
              follow_up_questions = [
                  "Can you provide more details about this?",
                  "How does this relate to other parts of the system?",
                  "What are the next steps I should consider?"
              ]
          
          return {
              "answer": response_text,
              "reasoning": f"Retrieved {len(documents)} relevant documents and generated contextual response using {query_intent}.",
              "session_id": session_id,
              "sources": sources if include_sources else [],
              "follow_up_questions": follow_up_questions[:3] if follow_up_suggestions else []
          }
          
      except Exception as e:
          logger.error(f"Error in natural language processing: {str(e)}", exc_info=True)
          return {
              "answer": "I encountered an error while processing your question. This might be due to a temporary issue. Please try:\n\n1. Rephrasing your question\n2. Being more specific about what you're looking for\n3. Asking about a particular document or topic\n\nI'm here to help with any questions about your documents!",
              "reasoning": f"Error in processing: {str(e)}",
              "session_id": session_id,
              "sources": [],
              "follow_up_questions": [
                  "Would you like to see all available documents?",
                  "Can you tell me more about what you're looking for?"
              ] if follow_up_suggestions else []
          }

      
    async def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
      """Analyze query intent using semantic understanding."""
      
      # Use embeddings to compare with known intents
      query_embedding = await self.embedding_service.get_embedding(query)
      
      # Reshape to 2D array - THIS IS THE FIX
      query_embedding = np.array(query_embedding).reshape(1, -1)
      
      # Define intent prototypes
      intent_examples = {
          "DOCUMENT_LISTING": [
              "list all documents", 
              "show me the documents", 
              "what documents do you have",
              "display all files"
          ],
          "DOCUMENT_CONTENT": [
              "what does this document contain",
              "explain the content of document",
              "summarize this document",
              "tell me about this document"
          ],
          "DOCUMENT_SEARCH": [
              "find documents about",
              "search for information on",
              "look for documents mentioning",
              "are there any documents about"
          ]
      }
      
      # Get embeddings for all intent examples
      intent_embeddings = {}
      for intent, examples in intent_examples.items():
          intent_embeddings[intent] = [
              np.array(await self.embedding_service.get_embedding(example)).reshape(1, -1)  # Reshape here too
              for example in examples
          ]
      
      # Calculate similarity with each intent
      intent_scores = {}
      for intent, embeddings in intent_embeddings.items():
          similarities = [
              cosine_similarity(query_embedding, example_embedding)[0][0]  # Extract scalar from 2D result
              for example_embedding in embeddings
          ]
          intent_scores[intent] = max(similarities) if similarities else 0
      
      # Get the highest scoring intent
      best_intent = max(intent_scores.items(), key=lambda x: x[1])
      
      # Extract potential document references from query
      doc_references = []
      doc_name_match = re.search(r'(?:about|for|in|the|this) [""]?([^""]+)[""]? (?:document|file)', query.lower())
      if doc_name_match:
          doc_references.append(doc_name_match.group(1))
      
      # If the score is too low, default to general search
      if best_intent[1] < 0.7:
          intent = "GENERAL_QUERY"
      else:
          intent = best_intent[0]
      
      return {
          "intent": intent,
          "confidence": best_intent[1],
          "document_references": doc_references
      }

    

    async def _retrieve_relevant_documents(
    self, 
    query: str, 
    max_sources: int = 5
) -> Tuple[List[Dict[str, Any]], List[str]]:
      """Retrieve documents relevant to the query."""
      
      # Special case for document listing
      if any(phrase in query.lower() for phrase in ["list document", "show document", "all document"]):
          # Get all documents
          documents = await self._get_documents_from_chroma(limit=max_sources)
          document_texts = [doc.get("text", "") for doc in documents]
          return documents, document_texts
      
      # Check for document name in query
      doc_name_match = re.search(r'(?:about|for|in|the|this) [""]?([^""]+)[""]? (?:document|file)', query.lower())
      if doc_name_match:
          doc_name = doc_name_match.group(1)
          # Try to find document by name
          documents = await self._get_documents_from_chroma(
              filter_dict={"filename": {"$eq": doc_name}},
              limit=1
          )
          if documents:
              document_texts = [doc.get("text", "") for doc in documents]
              return documents, document_texts
      
      # Default: semantic search
      # FIXED: Use search_similar_documents instead of search_documents
      search_results = await self.embedding_service.search_similar_documents(
          query=query,  # Pass the query string directly
          k=max_sources,  # Use k instead of limit
          min_score=0.3  # Add a reasonable minimum score
      )
      print("search results",search_results)
      
      documents = search_results  # search_similar_documents already returns formatted documents
      document_texts = [doc.get("text", "") for doc in documents]
      print("document_texts",document_texts, documents)
      
      return documents, document_texts


    async def _generate_contextual_response(
    self,
    query: str,
    query_intent: Dict[str, Any],
    documents: List[Dict[str, Any]],
    document_texts: List[str],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
      """Generate a contextual response based on query and documents."""
      # Add this at the beginning of the method, before the document listing check
      # Check if this is a document retrieval request vs explanation request
      
      is_retrieval_request = any(word in query.lower() for word in ["get", "give", "show", "find", "retrieve"])
      is_visual_request = any(term in query.lower() for term in ["flowchart", "flow chart", "diagram", "visual", "image"])
      is_explanation_request = any(word in query.lower() for word in ["explain", "summarize", "describe", "tell me about"])

      # Handle direct document retrieval requests
      # Handle direct document retrieval requests
      if (is_retrieval_request or is_visual_request) and not is_explanation_request:
          # Filter documents based on request type
          filtered_documents = []
          
          # If asking for flowchart/diagram/visual, only include image files
          if is_visual_request or "flowchart" in query.lower() or "diagram" in query.lower():
              filtered_documents = [doc for doc in documents if 
                                doc.get("metadata", {}).get("filename", "").lower().endswith(
                                    (".png", ".jpg", ".jpeg", ".svg", ".gif"))]
              doc_type_description = "visual document"
              
              # For visual documents, create a response that shows the actual images
              if filtered_documents:
                  image_info = []
                  for doc in filtered_documents:
                      metadata = doc.get("metadata", {})
                      file_path = metadata.get("file_path", "")
                      filename = metadata.get("filename", "Untitled")
                      
                      image_info.append({
                          "title": filename,
                          "type": metadata.get("mime_type", "Unknown"),
                          "id": doc.get("id", ""),
                          "file_path": file_path
                      })
                  
                  # Create sources list for citation and image display
                  sources = []
                  for doc in filtered_documents:
                      metadata = doc.get("metadata", {})
                      file_path = metadata.get("file_path", "")
                      
                      sources.append({
                          "id": doc.get("id", ""),
                          "title": metadata.get("filename", "Unknown image"),
                          "type": "image",
                          "file_path": file_path,
                          "cited": True,
                          "display": True  # Flag to indicate this should be displayed
                      })
                  
                  # Create a response that displays the images
                  response_text = f"Here are the visual documents I found:\n\n"
                  
                  # Add image references - these will be rendered by the frontend
                  for i, img in enumerate(image_info):
                      response_text += f"**Image {i+1}**: {img['title']}\n"
                      response_text += f"![{img['title']}](image:{img['id']})\n\n"
                  
                  return {
                      "answer": response_text,
                      "reasoning": f"Found {len(filtered_documents)} visual documents.",
                      "session_id": session_id,
                      "sources": sources,
                      "follow_up_questions": [
                          "Would you like me to explain any of these visuals?",
                          "Would you like to see other types of documents?",
                          "Would you like to search for specific content related to these visuals?"
                      ]
                  }
          
          # If asking for documentation/document, only include document files
          elif "documentation" in query.lower() or "document" in query.lower():
              filtered_documents = [doc for doc in documents if 
                                doc.get("metadata", {}).get("filename", "").lower().endswith(
                                    (".doc", ".docx", ".pdf", ".txt", ".md"))]
              doc_type_description = "documentation"
          else:
              # Default case - use all documents
              filtered_documents = documents
              doc_type_description = "document"
          
          # Format document information - only for filtered documents
          document_info = []
          for doc in filtered_documents:
              metadata = doc.get("metadata", {})
              doc_info = {
                  "title": metadata.get("filename", "Untitled"),
                  "type": metadata.get("mime_type", "Unknown"),
                  "id": doc.get("id", "")
              }
              document_info.append(doc_info)
          
          # Create prompt for document retrieval with strict instructions
          prompt = f"""
          You are an assistant helping retrieve documents based on the user query: "{query}"
          
          I found the following documents that match the request:
          {json.dumps(document_info, indent=2)}
          
          IMPORTANT INSTRUCTIONS:
          1. Start with a friendly greeting like "Here's the {doc_type_description} you requested:" or similar appropriate phrase
          2. Create a concise response that only lists the documents that exactly match what the user asked for
          3. Only mention the title and type of each matching document in a simple table format
          4. Do NOT mention any other documents or provide any additional commentary
          5. Do NOT make assumptions about document contents
          6. Keep your response brief and to the point
          
          Format the response in markdown.
          """
          
          # Generate response using LLM
          response_text = await self.bedrock_service.get_completion(prompt)
          
          # Create sources list for citation - only include filtered documents
          sources = []
          for doc in filtered_documents:
              metadata = doc.get("metadata", {})
              sources.append({
                  "id": doc.get("id", ""),
                  "title": metadata.get("filename", "Unknown document"),
                  "type": "document",
                  "cited": True
              })
          
          # Specific follow-up questions based on document type
          if is_visual_request:
              follow_up_questions = [
                  "Would you like me to explain what this diagram shows?",
                  "Do you need me to describe any specific part of this diagram?",
                  "Would you like to see related documentation?"
              ]
          else:
              follow_up_questions = [
                  "Would you like me to explain the content of this document?",
                  "Do you need any specific information from this document?",
                  "Would you like to see related documents?"
              ]
          
          return {
              "answer": response_text,
              "reasoning": f"Retrieved {len(filtered_documents)} documents matching the user's request.",
              "session_id": session_id,
              "sources": sources,
              "follow_up_questions": follow_up_questions
          }


      # Check if this is a document listing request with source filter
      source_match = re.search(r'(?:from|in) (sharepoint|onedrive|google drive|local|database)', query.lower())
      source_name = source_match.group(1) if source_match else None
      
      # For document listing intent
      if query_intent.get("intent") == "DOCUMENT_LISTING" or "list document" in query.lower():
          # Format document information
          document_info = []
          for doc in documents:
              metadata = doc.get("metadata", {})
              doc_info = {
                  "title": metadata.get("filename", "Untitled"),
                  "type": metadata.get("mime_type", "Unknown"),
                  "id": doc.get("id", ""),
                  "source": metadata.get("source", "Unknown")
              }
              document_info.append(doc_info)
          
          # Create prompt for document listing
          source_phrase = f" from {source_name}" if source_name else ""
          prompt = f"""
          You are an assistant helping list documents{source_phrase}. Here are the documents:
          {json.dumps(document_info, indent=2)}
          
          Create a concise response that:
          1. States how many documents were found{' in ' + source_name if source_name else ''}
          2. Lists each document with its title, type, and ID
          3. If no documents were found from the requested source, clearly state that
          
          Format the response in a clear, readable way using markdown.
          """
          
          # Generate response using LLM
          response_text = await self.bedrock_service.get_completion(prompt)
          
          # Create sources list for citation
          sources = []
          for doc in documents:
              metadata = doc.get("metadata", {})
              sources.append({
                  "id": doc.get("id", ""),
                  "title": metadata.get("filename", "Unknown document"),
                  "type": "document",
                  "cited": True
              })
          
          # Generate follow-up questions
          follow_up_questions = [
              "Would you like to see more details about any specific document?",
              "Would you like to search within these documents?",
              "Would you like to upload more documents?"
          ]
          
          return {
              "answer": response_text,
              "reasoning": f"Retrieved {len(documents)} documents from the database.",
              "session_id": session_id,
              "sources": sources,
              "follow_up_questions": follow_up_questions
          }
      
      # For other query types, continue with your existing implementation...
      # Combine document texts for context
      context = "\n\n".join([f"Document {i+1}:\n{text}" for i, text in enumerate(document_texts)])

      # Create prompt for the LLM
      prompt = f"""
      You are an AI assistant answering questions based on provided documents.

      User query: {query}

      Context from relevant documents:
      {context}

      Instructions:
      1. If the user is asking for an explanation or summary, provide a comprehensive analysis of the document content.
      2. If the user is asking about specific information, focus on answering that specific question.
      3. Always cite your sources by document number.
      4. If the documents include flowcharts, diagrams, or visual elements, describe what they show conceptually.
      5. If the documents don't contain relevant information, say so clearly.

      Provide a well-structured, informative response.
      """

      # Generate response using LLM
      response_text = await self.bedrock_service.get_completion(prompt)
      
      # Create sources list for citation
      sources = []
      for doc in documents:
          metadata = doc.get("metadata", {})
          sources.append({
              "id": doc.get("id", ""),
              "title": metadata.get("filename", "Unknown document"),
              "type": "document",
              "cited": True
          })
      
      # Generate follow-up questions based on context
      follow_up_prompt = f"""
      Based on the user query "{query}" and the context provided, 
      suggest 3 natural follow-up questions the user might ask next.
      Return only the questions as a JSON array of strings.
      """
      
      follow_up_response = await self.bedrock_service.get_completion(follow_up_prompt)
      
      # Try to parse follow-up questions as JSON, fallback to defaults if needed
      try:
          follow_up_questions = json.loads(follow_up_response)
          if not isinstance(follow_up_questions, list):
              raise ValueError("Not a list")
      except:
          follow_up_questions = [
              "Can you explain more about this topic?",
              "How does this relate to other concepts?",
              "Are there any examples you can provide?"
          ]
      
      return {
          "answer": response_text,
          "reasoning": f"Generated response based on {len(documents)} relevant documents.",
          "session_id": session_id,
          "sources": sources,
          "follow_up_questions": follow_up_questions
      }



    async def _generate_follow_up_questions(
        self, 
        query: str, 
        response: str, 
        document_texts: List[str],
    ) -> List[str]:
        """Generate contextually relevant follow-up questions."""
        
        combined_text = "\n".join(document_texts)
        
        prompt = f"""
        Based on this query: "{query}"
        
        And this response: "{response}"
        
        Generate 3 natural follow-up questions that would help explore the document content further.
        Each question should be specific to the content and help the user discover more information.
        Return only the questions, one per line.
        """
        
        try:
            questions_text = await self.llm_service.get_completion_with_messages(prompt)
            print("follow-up questions text",questions_text)
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            print("follow-up questions",questions)
            return questions[:3]  # Limit to 3 questions
        except Exception as e:
            logger.warning(f"Error generating follow-up questions: {str(e)}")
            return [
                "Would you like more details about this topic?",
                "Would you like to see other related documents?",
                "Do you have any other questions about these documents?"
            ]


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
            # Check if this is a document listing query first
            # Check for document listing intent first
            # Normalize the query
            normalized_query = query.lower().strip()
            
            # More comprehensive document listing patterns
            doc_listing_patterns = [
                "list document", "list the document", "show document", "what document", 
                "document do i have", "show all document", "list all document",
                "what file", "show file", "list file", "list the file",
                "document in", "document on", "document about", "document related",
                "get document", "retrieve document", "find document", "document list",
                "display document", "see document", "view document"
            ]
            
            # Check if any pattern is in the normalized query
            for pattern in doc_listing_patterns:
                if pattern in normalized_query:
                    return "DOCUMENT_LISTING", [{"text": "documents", "type": "GENERAL"}]
            
            # Check for document explanation requests
            doc_explanation_patterns = [
                "explain document", "explain the document", 
                "tell me about document", "describe document",
                "what is in document", "what does document contain",
                "summarize document", "content of document"
            ]
            
            # Extract document references from query
            doc_references = re.findall(r'document (?:called|named|titled) [""]?([^""]+)[""]?', normalized_query)
            doc_references.extend(re.findall(r'[""]?([^""]+)[""]? document', normalized_query))
            
            if any(pattern in normalized_query for pattern in doc_explanation_patterns) or doc_references:
                return "DOCUMENT_EXPLANATION", [{"text": doc_references[0] if doc_references else "document", "type": "DOCUMENT_REFERENCE"}]
    
            analysis_response = await self.bedrock_service.get_completion(analysis_prompt)
            print("analysis response", analysis_response)
            
            # Extract JSON from response
            analysis_json = self._extract_json_from_response(analysis_response)
            print("analysis json", analysis_json)
            
            if not analysis_json or "query_type" not in analysis_json:
                # Fallback if parsing fails
                return "TECHNICAL_CONCEPT", []
            print("returning analysis", analysis_json["query_type"], analysis_json.get("entities", []))
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
        max_sources: int = 5,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Balanced retrieval that always attempts vector search (chunks + docs), code search, and KG.
        Returns combined context (string) and list of unique sources (max_sources)
        """
        sources: List[Dict[str, Any]] = []

        # 1) Embedding-based vector search (best-effort)
        query_embedding = await self._get_query_embedding(query)
        print("query embedding",query_embedding)
        if query_embedding:
            try:
                vector_results = await self._vector_search(query_embedding, k=max_sources)
                sources.extend(vector_results)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # 2) Code search (Elasticsearch) — helpful for CODE_EXPLANATION/TROUBLESHOOTING
        try:
            code_results = await self._search_code_index(query, entities)
            sources.extend(code_results)
        except Exception as e:
            logger.warning(f"Code search failed: {e}")

        # 3) Knowledge graph
        try:
            kg_results = await self._query_knowledge_graph(entities)
            print("kg results",kg_results)
            sources.extend(kg_results)
            print("sources after kg",sources)
        except Exception as e:
            logger.warning(f"KG query failed: {e}")

        # Deduplicate preserving order and keep top-k by score if available
        unique_sources: List[Dict[str, Any]] = []
        seen_ids = set()
        for s in sources:
            sid = s.get("id") or s.get("doc_id") or s.get("name")
            if not sid:
                sid = json.dumps(s, sort_keys=True)
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            unique_sources.append(s)
            if len(unique_sources) >= max_sources:
                break

        # Build combined context text with numbered SOURCE tokens — LLM can reference [SOURCE 1], etc.
        combined_context = ""
        for i, src in enumerate(unique_sources, 1):
            src_text = src.get("text") or src.get("description") or ""
            src_type = src.get("type", "document")
            title = src.get("title") or src.get("name") or src.get("file_path") or f"Source {i}"

            header = f"SOURCE {i} [{src_type.upper()} - {title}]:\n"
            combined_context += header + src_text.strip() + "\n\n"

        return combined_context.strip(), unique_sources
    
    async def _retrieve_relevant_documents(
    self, 
    query: str, 
    max_sources: int = 5
) -> Tuple[List[Dict[str, Any]], List[str]]:
      """Retrieve documents relevant to the query."""
      
      # Check for document listing with source specification
      source_match = re.search(r'(?:from|in) (sharepoint|onedrive|google drive|local|database)', query.lower())
      source_filter = None
      
      if source_match:
          source_name = source_match.group(1)
          # Map common source names to your metadata field values
          source_mapping = {
              "sharepoint": "sharepoint",
              "onedrive": "onedrive",
              "google drive": "google_drive",
              "local": "local_upload",
              "database": "database"
          }
          
          if source_name in source_mapping:
              source_filter = {"source": {"$eq": source_mapping[source_name]}}
              logger.info(f"Filtering documents by source: {source_name}")
      
      # Special case for document listing
      if any(phrase in query.lower() for phrase in ["list document", "show document", "all document"]):
          # Get documents with optional source filter
          documents = await self._get_documents_from_chroma(
              filter_dict=source_filter,
              limit=max_sources
          )
          document_texts = [doc.get("text", "") for doc in documents]
          return documents, document_texts
      
      # Rest of the method remains the same...
      # Check for document name in query
      doc_name_match = re.search(r'(?:about|for|in|the|this) [""]?([^""]+)[""]? (?:document|file)', query.lower())
      if doc_name_match:
          doc_name = doc_name_match.group(1)
          # Try to find document by name
          filter_dict = {"filename": {"$eq": doc_name}}
          if source_filter:
              # Combine with source filter if present
              filter_dict = {"$and": [filter_dict, source_filter]}
              
          documents = await self._get_documents_from_chroma(
              filter_dict=filter_dict,
              limit=1
          )
          if documents:
              document_texts = [doc.get("text", "") for doc in documents]
              return documents, document_texts
          
      # Check for direct document requests
      direct_doc_request = re.search(r'(get|give|show|find|retrieve) (?:me |the )?([\w\s]+) (document|file|image)', query.lower())
      if direct_doc_request:
          doc_subject = direct_doc_request.group(2).strip()
          
          # Create filter for finding the document - use $eq instead of $contains
          documents = await self._get_documents_from_chroma(limit=50)
          
          # Filter manually since $contains isn't supported
          filtered_docs = []
          for doc in documents:
              filename = doc.get("metadata", {}).get("filename", "").lower()
              title = doc.get("metadata", {}).get("title", "").lower()
              
              if doc_subject.lower() in filename or doc_subject.lower() in title:
                  # Also check source filter if present
                  if source_filter:
                      source_value = source_filter["source"]["$eq"]
                      doc_source = doc.get("metadata", {}).get("source", "")
                      if doc_source == source_value:
                          filtered_docs.append(doc)
                  else:
                      filtered_docs.append(doc)
          
          if filtered_docs:
              document_texts = [doc.get("text", "") for doc in filtered_docs]
              logger.info(f"Found {len(filtered_docs)} documents matching direct request for '{doc_subject}'")
              return filtered_docs, document_texts

      # Check for requests for visual documents like flowcharts/diagrams
      visual_doc_match = re.search(r'(flow ?chart|diagram|graph|visual) (?:of|for|about) ([\w\s]+)', query.lower())
      if visual_doc_match:
          visual_type = visual_doc_match.group(1)
          subject = visual_doc_match.group(2).strip()
          
          # Get all documents and filter manually
          documents = await self._get_documents_from_chroma(
              filter_dict=source_filter,  # Only apply source filter if present
              limit=50
          )
          
          # Filter for visual documents about the subject
          filtered_docs = []
          for doc in documents:
              filename = doc.get("metadata", {}).get("filename", "").lower()
              title = doc.get("metadata", {}).get("title", "").lower()
              
              # Check if filename contains both the subject and visual type or has image extension
              is_image = any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".svg", ".gif"])
              has_subject = subject.lower() in filename or subject.lower() in title
              has_visual_type = visual_type.lower().replace(" ", "") in filename or visual_type.lower() in title
              
              if (has_subject and (has_visual_type or is_image)):
                  filtered_docs.append(doc)
          
          if filtered_docs:
              document_texts = [doc.get("text", "") for doc in filtered_docs]
              logger.info(f"Found {len(filtered_docs)} visual documents matching '{subject} {visual_type}'")
              return filtered_docs, document_texts
      
      # Default: semantic search with optional source filter
      search_results = await self.embedding_service.search_similar_documents(
          query=query,
          filter_dict=source_filter,  # Apply source filter if present
          k=max_sources,
          min_score=0.3
      )
      
      documents = search_results
      document_texts = [doc.get("text", "") for doc in documents]
      
      return documents, document_texts

    
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
      """Search vector database using query embedding - Chroma DB version"""
      try:
          # Search chunks first for more specific context using Chroma
          formatted_results = []
          
          # Convert filter_str to Chroma filter dict if needed
          filter_dict = None
          if filter_str and filter_str != "*":
              # Convert your filter string to a Chroma-compatible filter dict
              # This is a simplified example - adjust based on your filter string format
              if ":" in filter_str:
                  key, value = filter_str.split(":", 1)
                  filter_dict = {key: value}
          
          # Query Chroma chunk collection
          chunk_results = self.embedding_service.chunk_collection.query(
              query_embeddings=[query_embedding],
              n_results=k,
              where=filter_dict,
              include=["documents", "metadatas", "distances"]
          )
          
          # Process chunk results
          if chunk_results and chunk_results.get('ids') and chunk_results['ids'][0]:
              for i, chunk_id in enumerate(chunk_results['ids'][0]):
                  metadata = chunk_results['metadatas'][0][i] if 'metadatas' in chunk_results and chunk_results['metadatas'][0] else {}
                  doc_id = metadata.get("doc_id", "unknown")
                  
                  formatted_results.append({
                      "id": f"chunk:{chunk_id}",
                      "title": metadata.get("filename", f"Document {doc_id}"),
                      "text": chunk_results['documents'][0][i] if 'documents' in chunk_results and chunk_results['documents'][0] else "",
                      "type": "document",
                      "chunk_id": chunk_id,
                      "doc_id": doc_id,
                      "score": 1.0 - (chunk_results['distances'][0][i] if 'distances' in chunk_results else 0)
                  })
          
          # If not enough chunk results, search documents
          if len(formatted_results) < k:
              # Query Chroma document collection
              doc_results = self.embedding_service.doc_collection.query(
                  query_embeddings=[query_embedding],
                  n_results=k - len(formatted_results),
                  where=filter_dict,
                  include=["documents", "metadatas", "distances"]
              )
              
              # Process document results
              if doc_results and doc_results.get('ids') and doc_results['ids'][0]:
                  for i, doc_id in enumerate(doc_results['ids'][0]):
                      metadata = doc_results['metadatas'][0][i] if 'metadatas' in doc_results and doc_results['metadatas'][0] else {}
                      
                      formatted_results.append({
                          "id": f"doc:{doc_id}",
                          "title": metadata.get("filename", "Unknown"),
                          "text": doc_results['documents'][0][i] if 'documents' in doc_results and doc_results['documents'][0] else "Document content not available",
                          "type": "document",
                          "doc_id": doc_id,
                          "score": 1.0 - (doc_results['distances'][0][i] if 'distances' in doc_results else 0)
                      })
          
          print(f"Chroma search returned {len(formatted_results)} results")
          return formatted_results
      except Exception as e:
          logger.error(f"Error in Chroma vector search: {str(e)}")
          import traceback
          traceback.print_exc()
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
        include_citations: bool = True,
    ) -> Tuple[str, str, List[int]]:
        """
        Generate an answer with a strict system prompt that forces the model to use ONLY provided context.
        If the information is not present in the context, the model must reply with a fixed "not found" sentence.
        """
        # Prepare history text (last few exchanges)
        history_text = ""
        if chat_history:
            for exchange in chat_history[-3:]:
                history_text += f"User: {exchange.get('user_message','')}\nAssistant: {exchange.get('ai_response','')}\n\n"

        system_prompt = (
            "You are a Retrieval-Augmented-Generation (RAG) assistant.\n"
            "IMPORTANT RULES (MUST FOLLOW):\n"
            "1) You MUST answer using ONLY the information contained in the provided context. "
            "Do NOT use or invent any external knowledge or assumptions.\n"
            "2) If the answer cannot be found within the context, reply exactly: "
            "\"The provided data does not contain information about this.\"\n"
            "3) When you reference material from the context, annotate it inline like [SOURCE X] where X is the source number.\n"
            "4) Keep answers concise, technical when needed, and provide step-by-step reasoning only if requested.\n"
        )

        user_prompt = (
            f"Question:\n{query}\n\n"
            "Context (only use what's below):\n"
            f"{context}\n\n"
            "Conversation history (for context, optional):\n"
            f"{history_text}\n"
            "Instructions:\n"
            "Answer the question strictly from the context. If you use facts from the context, mark them with [SOURCE X]."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.bedrock_service.get_completion_with_messages(messages)

            # Parse the LLM text for citations and reasoning
            answer, reasoning, citations = self._parse_llm_response(response)

            # Additional safety: if the LLM didn't follow the "not found" rule but answer is empty, enforce it
            if (not answer or answer.strip() == ""):
                answer = "The provided data does not contain information about this."
                reasoning = "No context-supporting content was present."
                citations = []

            return answer, reasoning, citations
        except Exception as e:
            logger.exception("LLM generation failed")
            return (
                "The provided data does not contain information about this.",
                "LLM generation failed.",
                [],
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
            json_match = re.search(r'json\s*({.*?})\s*```', response, re.DOTALL)
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
    async def _handle_document_listing(
    self,
    query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    include_sources: bool = True,
    follow_up_suggestions: bool = True,
    max_sources: int = 10
) -> Dict[str, Any]:
      """Handle document listing queries specifically"""
      try:
          # Get documents from Chroma DB
          documents = await self._get_documents_from_chroma(limit=max_sources)
          if not documents:
              return {
                  "answer": "I don't see any documents in the system. Would you like to upload some?",
                  "reasoning": "No documents found in the Chroma database.",
                  "session_id": session_id,
                  "sources": [],
                  "follow_up_questions": [
                      "Would you like to upload documents?",
                      "Do you need help with document management?",
                      "Would you like to learn how to add documents to the system?"
                  ] if follow_up_suggestions else []
              }
          # Format document list for display
          doc_list = "\n\n".join([
              f"**{i+1}. {doc.get('metadata', {}).get('filename', doc.get('title', 'Untitled Document'))}**\n" +
              f"   - Type: {doc.get('metadata', {}).get('mime_type', 'Document')}\n" +
              f"   - ID: {doc.get('id', 'Unknown')}"
              for i, doc in enumerate(documents)
          ])
          answer = f"I found {len(documents)} documents in the system:\n\n{doc_list}"
          # Format sources for response
          sources = []
          for doc in documents:
              doc_id = doc.get("id", "")
              title = doc.get("metadata", {}).get("filename", doc.get("title", "Unknown"))
              sources.append({
                  "id": doc_id,
                  "title": title,
                  "type": "document",
                  "cited": True
              })
          return {
              "answer": answer,
              "reasoning": f"Retrieved {len(documents)} documents from Chroma database.",
              "session_id": session_id,
              "sources": sources[:max_sources] if include_sources else [],
              "follow_up_questions": [
                  "Would you like to see more details about any specific document?",
                  "Would you like to search within these documents?",
                  "Would you like to upload more documents?"
              ] if follow_up_suggestions else []
          }
      except Exception as e:
          logger.exception(f"Error handling document listing: {str(e)}")
          return {
              "answer": "I encountered an error while trying to list your documents.",
              "reasoning": f"Error: {str(e)}",
              "session_id": session_id,
              "sources": [],
              "follow_up_questions": ["Would you like to try again?"] if follow_up_suggestions else []
          }
    async def _get_documents_from_chroma(
    self, 
    filter_dict: Optional[Dict[str, Any]] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
      """Get documents directly from ChromaDB with optional filtering"""
      try:
          logger.info(f"Getting documents from ChromaDB with filter: {filter_dict}")
          # Convert simple filter dict to ChromaDB where clause format
          where_clause = None
          if filter_dict:
              # Check if filter_dict already contains operators (nested dict structure)
              # If it does, use it as-is; otherwise, convert it
              first_value = next(iter(filter_dict.values()))
              if isinstance(first_value, dict) and any(k.startswith('$') for k in first_value.keys()):
                  # Already in proper format
                  where_clause = filter_dict
                  logger.info(f"Filter dict already in proper format: {where_clause}")
              else:
                  # Convert simple dict to proper ChromaDB format
                  if len(filter_dict) == 1:
                      key, value = next(iter(filter_dict.items()))
                      where_clause = {key: {"$eq": value}}
                  else:
                      # For multiple filters, use $and operator
                      where_clause = {
                          "$and": [
                              {key: {"$eq": value}} for key, value in filter_dict.items()
                          ]
                      }
                  logger.info(f"Constructed where clause: {where_clause}")
          # Get documents from ChromaDB
          if where_clause:
              results = self.embedding_service.doc_collection.get(
                  where=where_clause,
                  limit=limit
              )
          else:
              results = self.embedding_service.doc_collection.get(
                  limit=limit
              )
          # Format results
          formatted_results = []
          if results and 'ids' in results and results['ids']:
              for i in range(len(results['ids'])):
                  metadata = results.get('metadatas', [{}])[i] if results.get('metadatas') else {}
                  formatted_results.append({
                      "id": results['ids'][i],
                      "text": results['documents'][i] if 'documents' in results else "",
                      "metadata": metadata,
                      "source": metadata.get("source", "unknown"),
                      "filename": metadata.get("filename", "unknown")
                  })
              logger.info(f"Retrieved {len(formatted_results)} documents from ChromaDB")
          else:
              logger.warning("No documents found in ChromaDB or empty result structure")
          return formatted_results
      except ValueError as ve:
          # Specific handling for operator errors
          if "$contains" in str(ve):
              logger.error(f"Invalid operator $contains used. Filter dict: {filter_dict}")
              logger.error("Falling back to retrieving all documents without filter")
              # Retry without filter
              try:
                  results = self.embedding_service.doc_collection.get(limit=limit)
                  formatted_results = []
                  if results and 'ids' in results and results['ids']:
                      for i in range(len(results['ids'])):
                          metadata = results.get('metadatas', [{}])[i] if results.get('metadatas') else {}
                          formatted_results.append({
                              "id": results['ids'][i],
                              "text": results['documents'][i] if 'documents' in results else "",
                              "metadata": metadata,
                              "source": metadata.get("source", "unknown"),
                              "filename": metadata.get("filename", "unknown")
                          })
                      # Apply filter manually if needed
                      if filter_dict:
                          formatted_results = [
                              doc for doc in formatted_results
                              if all(doc.get('metadata', {}).get(k) == v for k, v in filter_dict.items())
                          ]
                  return formatted_results
              except Exception as retry_error:
                  logger.error(f"Retry also failed: {str(retry_error)}", exc_info=True)
                  return []
          else:
              logger.error(f"ValueError getting documents from ChromaDB: {str(ve)}", exc_info=True)
              return []
      except Exception as e:
          logger.error(f"Error getting documents from ChromaDB: {str(e)}", exc_info=True)
          return []
    def _format_document_list(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into a readable list"""
        formatted = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            # Extract relevant metadata
            filename = metadata.get("filename", metadata.get("title", "Unknown"))
            doc_type = metadata.get("mime_type", metadata.get("file_type", "Unknown"))
            upload_date = metadata.get("upload_date", metadata.get("timestamp", "Unknown"))
            file_size = metadata.get("file_size", "Unknown")
            # Format size if it's a number
            if isinstance(file_size, (int, float)):
                if file_size < 1024:
                    size_str = f"{file_size} bytes"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.2f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.2f} MB"
            else:
                size_str = str(file_size)
            formatted.append(
                f"{i}. **{filename}**\n"
                f"   - Type: {doc_type}\n"
                f"   - Size: {size_str}\n"
                f"   - Uploaded: {upload_date}\n"
                f"   - ID: {doc.get('id', 'Unknown')}"
            )
        return "\n\n".join(formatted)