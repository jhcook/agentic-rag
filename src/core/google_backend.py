"""
Google Gemini + Drive Backend Implementation.

This backend bypasses the local FAISS vector store and uses Google's 
native "Long Context" capabilities (Gemini 1.5 Pro) to perform RAG 
directly over documents stored in Google Drive.
"""

import os
import logging
import base64
import io
import re
from typing import Dict, Any, List, Optional

# You will need to install: google-generativeai google-auth google-auth-oauthlib google-auth-httplib2
try:
    import google.generativeai as genai
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    HAS_GOOGLE_DEPS = True
except ImportError:
    HAS_GOOGLE_DEPS = False

# Vertex AI dependencies (optional)
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Tool, grounding
    HAS_VERTEX_DEPS = True
except ImportError:
    HAS_VERTEX_DEPS = False

from src.core.interfaces import RAGBackend
try:
    from src.core.google_auth import GoogleAuthManager
except ImportError:
    GoogleAuthManager = None

logger = logging.getLogger(__name__)

class GoogleGeminiBackend:
    """
    Implements RAGBackend using Google Gemini 1.5 Pro and Google Drive.
    
    Authentication:
    - Uses OAuth2 credentials via GoogleAuthManager.
    - Supports both Drive API and Generative Language API.
    - Supports Vertex AI Grounding (Enterprise).
    """

    def __init__(self):
        if not HAS_GOOGLE_DEPS:
            raise ImportError("Google dependencies missing. Install 'google-generativeai' and 'google-api-python-client'.")
        
        # Load Vertex Config if exists
        if os.path.exists("vertex_config.json"):
            try:
                import json
                with open("vertex_config.json", "r") as f:
                    config = json.load(f)
                    os.environ.update(config)
            except Exception as e:
                logger.error(f"Failed to load vertex_config.json: {e}")

        # Initialize Auth Manager
        self.auth_manager = GoogleAuthManager()
        self.creds = self.auth_manager.get_credentials()
        
        # Mode configuration
        self.mode = os.getenv("GOOGLE_GROUNDING_MODE", "manual") # manual | vertex_ai_search
        
        # Initialize services to None to avoid AttributeError if auth fails
        self.drive_service = None
        self.gmail_service = None
        self.gen_service = None
        self.model = None

        if not self.creds:
            logger.warning("No valid Google credentials found. Please authenticate.")
            return

        # Initialize Drive Service
        self.drive_service = build('drive', 'v3', credentials=self.creds)
        
        # Initialize Gmail Service
        self.gmail_service = build('gmail', 'v1', credentials=self.creds)

        # Initialize Gemini (Generative Language API)
        # Note: google-generativeai SDK primarily uses API keys. 
        # For OAuth, we use the lower-level googleapiclient or configure genai if supported.
        # Here we use googleapiclient for the model to ensure OAuth compliance.
        self.gen_service = build('generativelanguage', 'v1beta', credentials=self.creds, static_discovery=False)
        
        # Initialize Vertex AI if configured
        if self.mode == "vertex_ai_search":
            if not HAS_VERTEX_DEPS:
                logger.error("Vertex AI dependencies missing. Install 'google-cloud-aiplatform'.")
            else:
                self._init_vertex()

    def _init_vertex(self):
        """Initialize Vertex AI client."""
        project_id = os.getenv("VERTEX_PROJECT_ID")
        location = os.getenv("VERTEX_LOCATION", "global")
        
        if not project_id:
            logger.warning("VERTEX_PROJECT_ID not set. Vertex AI mode may fail.")
            return

        try:
            # Vertex AI usually requires ADC or service account. 
            # We can try to use the OAuth credentials if they have cloud-platform scope.
            # However, vertexai.init() expects project/location.
            # The credentials handling is implicit via google.auth.default() usually.
            # To use our user credentials, we might need to pass them explicitly if supported,
            # or rely on the environment being set up correctly.
            # For now, we assume the environment (or gcloud auth application-default login) is set up,
            # OR we try to pass credentials if the SDK allows (it usually does via client options).
            
            # Note: vertexai.init doesn't take credentials directly in all versions.
            # It's better to let google.auth find them or set GOOGLE_APPLICATION_CREDENTIALS.
            # But since we have self.creds, let's see if we can use them.
            # The vertexai SDK uses google.auth.credentials.Credentials.
            
            vertexai.init(project=project_id, location=location, credentials=self.creds)
            logger.info(f"Vertex AI initialized for project {project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")

    def reload_auth(self):
        """Reload credentials and re-initialize services."""
        logger.info("Reloading Google credentials...")
        self.creds = self.auth_manager.authenticate()
        
        if self.creds:
            self.drive_service = build('drive', 'v3', credentials=self.creds)
            self.gmail_service = build('gmail', 'v1', credentials=self.creds)
            self.gen_service = build('generativelanguage', 'v1beta', credentials=self.creds, static_discovery=False)
            logger.info("Google services re-initialized.")
        else:
            logger.warning("Reload failed: No valid credentials found.")

    def _construct_drive_query(self, query: str) -> str:
        """Construct a Drive API query from a natural language string."""
        # Basic keyword extraction to convert natural language to Drive query
        # Remove common conversational filler
        stop_words = {
            "search", "find", "show", "me", "files", "documents", "about", "that", "discuss", 
            "are", "there", "any", "in", "my", "google", "drive", "the", "a", "an", "for", "of"
        }
        
        words = query.lower().replace("?", "").replace(".", "").split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        if not keywords:
            # Fallback: if everything was filtered, use the original query if it's short, else just list recent
            if len(words) < 3:
                return f"fullText contains '{query}' and trashed = false"
            return "trashed = false"

        # Construct query: fullText contains 'keyword'
        # We use 'and' to be specific, or 'or' to be broad. Let's use 'and' for relevance.
        clauses = [f"fullText contains '{k}'" for k in keywords]
        return " and ".join(clauses) + " and trashed = false"

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search Google Drive for files matching the query.
        """
        if not self.drive_service:
            # Try to reload if not initialized
            self.reload_auth()
            if not self.drive_service:
                return {"error": "Not authenticated"}

        # Construct a better query
        q = self._construct_drive_query(query)
        logger.info(f"Searching Google Drive with query: {q} (Original: {query})")
        
        try:
            results = self.drive_service.files().list(
                q=q, pageSize=top_k, fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
            
            files = results.get('files', [])
            sources = [f"drive://{f['id']} ({f['name']})" for f in files]
            
            return {
                "answer": f"Found {len(files)} files in Drive matching '{query}'",
                "sources": sources,
                "files": files
            }
        except Exception as e:
            logger.error(f"Drive search failed: {e}")
            return {"error": str(e)}

    def search_gmail(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search Gmail for emails matching the query.
        """
        if not self.gmail_service:
            return {"error": "Not authenticated"}

        logger.info(f"Searching Gmail for: {query}")
        
        try:
            # Search messages
            results = self.gmail_service.users().messages().list(
                userId='me', q=query, maxResults=top_k
            ).execute()
            
            messages = results.get('messages', [])
            email_data = []
            
            for msg in messages:
                try:
                    # Get full message details
                    full_msg = self.gmail_service.users().messages().get(
                        userId='me', id=msg['id'], format='full'
                    ).execute()
                    
                    payload = full_msg.get('payload', {})
                    headers = payload.get('headers', [])
                    
                    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                    sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown')
                    date = next((h['value'] for h in headers if h['name'].lower() == 'date'), 'Unknown')
                    snippet = full_msg.get('snippet', '')
                    
                    # Extract body
                    body = ""
                    if 'parts' in payload:
                        for part in payload['parts']:
                            if part['mimeType'] == 'text/plain':
                                data = part['body'].get('data')
                                if data:
                                    body += base64.urlsafe_b64decode(data).decode('utf-8')
                    elif 'body' in payload:
                        data = payload['body'].get('data')
                        if data:
                            body = base64.urlsafe_b64decode(data).decode('utf-8')
                            
                    email_data.append({
                        "id": msg['id'],
                        "subject": subject,
                        "sender": sender,
                        "date": date,
                        "snippet": snippet,
                        "body": body or snippet
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch email {msg['id']}: {e}")
            
            sources = [f"gmail://{e['id']} ({e['subject']})" for e in email_data]
            
            return {
                "answer": f"Found {len(email_data)} emails matching '{query}'",
                "sources": sources,
                "emails": email_data
            }
            
        except Exception as e:
            logger.error(f"Gmail search failed: {e}")
            return {"error": str(e)}

    def grounded_answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Generate an answer using Gemini with content from Drive files.
        Dispatches to the configured mode (manual or vertex_ai_search).
        """
        if self.mode == "vertex_ai_search":
            return self.grounded_answer_vertex(question)
            
        # Default: Manual Drive API + Gemini
        return self.grounded_answer_manual(question, k)

    def grounded_answer_vertex(self, question: str) -> Dict[str, Any]:
        """
        Generate an answer using Vertex AI Grounding with Google Drive (Data Store).
        """
        if not HAS_VERTEX_DEPS:
            return {"error": "Vertex AI dependencies missing."}
            
        data_store_id = os.getenv("VERTEX_DATA_STORE_ID")
        if not data_store_id:
            return {"error": "VERTEX_DATA_STORE_ID not set."}
            
        try:
            # Configure the tool
            # Format: projects/{project_id}/locations/{location}/collections/default_collection/dataStores/{data_store_id}
            project_id = os.getenv("VERTEX_PROJECT_ID")
            location = os.getenv("VERTEX_LOCATION", "global")
            
            # The SDK might handle the full path construction if we use the helper, 
            # but usually we need to pass the full resource name or just the ID if using a specific helper.
            # Using the Tool.from_retrieval helper:
            
            vertex_tool = Tool.from_retrieval(
                retrieval=grounding.Retrieval(
                    source=grounding.VertexAISearch(
                        datastore=data_store_id,
                        project=project_id,
                        location=location
                    )
                )
            )
            
            model = GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                question,
                tools=[vertex_tool]
            )
            
            # Extract text and grounding metadata
            answer_text = response.text
            
            # Parse grounding metadata for sources
            sources = []
            if response.candidates and response.candidates[0].grounding_metadata:
                gm = response.candidates[0].grounding_metadata
                if gm.grounding_chunks:
                    for chunk in gm.grounding_chunks:
                        if chunk.web:
                            sources.append(f"{chunk.web.title} ({chunk.web.uri})")
                        elif chunk.retrieved_context:
                            sources.append(f"{chunk.retrieved_context.title} ({chunk.retrieved_context.uri})")
            
            return {
                "answer": answer_text,
                "sources": sources,
                "mode": "vertex_ai_search"
            }
            
        except Exception as e:
            logger.error(f"Vertex AI generation failed: {e}")
            return {"error": str(e)}

    def _fetch_context(self, query: str, k: int = 3) -> tuple[str, list[str], list[str]]:
        """Helper to fetch and format context from Drive and Gmail."""
        context_parts = []
        sources = []
        errors = []
        
        # 1. Find relevant files in Drive
        drive_res = self.search(query, top_k=k)
        if "error" in drive_res:
            errors.append(f"Drive Error: {drive_res['error']}")
        else:
            files = drive_res.get("files", [])
            logger.info(f"Drive search found {len(files)} files.")
            for f in files:
                try:
                    file_id = f['id']
                    mime = f['mimeType']
                    name = f['name']
                    content = ""
                    
                    # Try to extract content based on MIME type
                    if "application/vnd.google-apps.document" in mime:
                        content = self.drive_service.files().export(
                            fileId=file_id, mimeType="text/plain"
                        ).execute().decode('utf-8')
                    elif "text/plain" in mime:
                        content = self.drive_service.files().get_media(
                            fileId=file_id
                        ).execute().decode('utf-8')
                    elif "application/pdf" in mime:
                        # For now, just note it's a PDF. 
                        # TODO: Add PDF text extraction (requires pypdf or similar)
                        content = "[PDF File - Content extraction not yet implemented]"
                    else:
                        content = f"[File type {mime} - Content not extracted]"
                    
                    # Always add metadata, even if content is empty/missing
                    context_parts.append(f"File: {name} (ID: {file_id}, Type: {mime})\nContent:\n{content}\n")
                    sources.append(f"drive://{file_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to read file {f.get('name')}: {e}")
                    # Still add the file existence to context
                    context_parts.append(f"File: {f.get('name')} (ID: {f.get('id')})\n[Error reading content: {str(e)}]\n")

        # 2. Find relevant emails in Gmail
        gmail_res = self.search_gmail(query, top_k=k)
        if "error" in gmail_res:
            errors.append(f"Gmail Error: {gmail_res['error']}")
        else:
            emails = gmail_res.get("emails", [])
            logger.info(f"Gmail search found {len(emails)} emails.")
            for e in emails:
                context_parts.append(f"Email: {e['subject']} (from {e['sender']})\nDate: {e['date']}\nContent:\n{e['body']}\n")
                sources.append(f"gmail://{e['id']}")
            
        full_context = "\n---\n".join(context_parts)
        return full_context, sources, errors

    def grounded_answer_manual(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Generate an answer using Gemini 1.5 Pro with content from Drive files and Gmail (Manual Mode).
        """
        if not self.gen_service or not self.drive_service:
            return {"error": "Not authenticated"}

        logger.info(f"Generating grounded answer with Gemini for: {question}")
        
        full_context, sources, errors = self._fetch_context(question, k=k)
        
        # Check for API enablement errors to report
        api_error_msg = ""
        for err in errors:
            match = re.search(r'(https://console\.developers\.google\.com/apis/api/[^/]+/overview\?project=\d+)', err)
            if match:
                url = match.group(1)
                api_name = "Google API"
                if "gmail" in url: api_name = "Gmail API"
                elif "drive" in url: api_name = "Google Drive API"
                api_error_msg += f"\n\n⚠️ **Action Required**: The {api_name} is not enabled. [Click here to enable it]({url})."

        if not full_context and not api_error_msg:
            return {"answer": "No relevant documents or emails found.", "sources": []}
        
        if not full_context and api_error_msg:
             return {"answer": f"I couldn't search your data because of a missing permission.{api_error_msg}", "sources": []}

        # 5. Call Gemini via REST API (v1beta)
        prompt = f"""
        You are a helpful assistant with access to the user's Google Drive files and Gmail.
        Answer the following question based ONLY on the provided documents and emails.
        
        Context:
        {full_context}
        
        Question: {question}
        """
        
        try:
            body = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2048
                }
            }
            
            response = self.gen_service.models().generateContent(
                model="models/gemini-2.0-flash",
                body=body
            ).execute()
            
            candidates = response.get('candidates', [])
            if candidates:
                answer_text = candidates[0]['content']['parts'][0]['text']
                if api_error_msg:
                    answer_text += api_error_msg
                return {
                    "answer": answer_text,
                    "sources": sources,
                    "mode": "manual"
                }
            else:
                return {"answer": "No answer generated." + api_error_msg, "sources": []}
                
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return {"error": str(e)}

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Conversational chat with Gemini 1.5 Pro, with RAG context from Drive/Gmail.
        """
        if not self.gen_service:
            return {"error": "Not authenticated"}

        logger.info("Chat request received")
        
        # 1. Get the latest user message to use as a search query
        last_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
        query = last_user_msg["content"] if last_user_msg else ""
        
        context_str = ""
        api_error_msg = ""
        
        if query:
            # We use a smaller k for chat to avoid overwhelming context
            full_context, _, errors = self._fetch_context(query, k=3)
            
            if full_context:
                context_str = f"\n\nRelevant Context from Drive/Gmail:\n{full_context}"
            else:
                context_str = f"\n\n[System Note: A search of the user's Google Drive and Gmail for '{query}' returned no results.]"
            
            for err in errors:
                match = re.search(r'(https://console\.developers\.google\.com/apis/api/[^/]+/overview\?project=\d+)', err)
                if match:
                    url = match.group(1)
                    api_name = "Google API"
                    if "gmail" in url: api_name = "Gmail API"
                    elif "drive" in url: api_name = "Google Drive API"
                    api_error_msg += f"\n\n⚠️ **Action Required**: The {api_name} is not enabled. [Click here to enable it]({url})."

        # If we have no context and a critical error, return early
        if not context_str and api_error_msg:
             return {
                 "role": "assistant",
                 "content": f"I couldn't search your data because of a missing permission.{api_error_msg}"
             }

        contents = []
        for i, m in enumerate(messages):
            role = "user" if m["role"] == "user" else "model"
            if m["role"] == "assistant":
                role = "model"
            
            text = m["content"]
            
            # Inject context into the LAST user message
            if i == len(messages) - 1 and role == "user" and context_str:
                text += context_str
                
            contents.append({"role": role, "parts": [{"text": text}]})

        try:
            body = {
                "systemInstruction": {
                    "parts": [{"text": "You are a helpful assistant. You have access to the user's Google Drive and Gmail via a RAG system. Relevant documents and emails will be injected into the conversation context. If you see context provided, use it to answer the user's questions. If the user asks to search for something and the system note says no results were found, inform the user."}]
                },
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2048
                }
            }
            
            response = self.gen_service.models().generateContent(
                model="models/gemini-2.0-flash",
                body=body
            ).execute()
            
            candidates = response.get('candidates', [])
            if candidates:
                answer_text = candidates[0]['content']['parts'][0]['text']
                if api_error_msg:
                    answer_text += api_error_msg
                return {
                    "role": "assistant",
                    "content": answer_text
                }
            else:
                return {"error": "No response generated"}
                
        except Exception as e:
            logger.error(f"Gemini chat failed: {e}")
            return {"error": str(e)}

    def list_drive_files(self, folder_id: str = None) -> List[Dict[str, Any]]:
        """List files in Google Drive with metadata."""
        if not self.drive_service:
            self.reload_auth()
            if not self.drive_service:
                return []

        try:
            q = "trashed = false"
            if folder_id:
                q += f" and '{folder_id}' in parents"
            else:
                # Default to root if no folder specified to avoid listing entire drive
                q += " and 'root' in parents"
            
            results = self.drive_service.files().list(
                q=q, 
                pageSize=100, 
                fields="nextPageToken, files(id, name, mimeType, size, webViewLink, iconLink, createdTime, modifiedTime)"
            ).execute()
            
            return results.get('files', [])
        except Exception as e:
            logger.error(f"Drive list failed: {e}")
            return []

    def upload_file(self, name: str, content: bytes, mime_type: str) -> Dict[str, Any]:
        """Upload a file to Google Drive."""
        if not self.drive_service:
            return {"error": "Not authenticated"}

        try:
            file_metadata = {'name': name}
            media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=True)
            file = self.drive_service.files().create(
                body=file_metadata, 
                media_body=media, 
                fields='id, name, webViewLink'
            ).execute()
            return file
        except Exception as e:
            logger.error(f"Drive upload failed: {e}")
            return {"error": str(e)}

    def logout(self):
        """Clear credentials and services."""
        logger.info("Logging out from Google Backend...")
        if self.auth_manager:
            self.auth_manager.logout()
        self.creds = None
        self.drive_service = None
        self.gmail_service = None
        self.gen_service = None
        self.model = None

    def get_mode(self) -> str:
        """Get the current grounding mode."""
        return self.mode

    def get_available_modes(self) -> List[str]:
        """Get available grounding modes."""
        modes = ["manual"]
        if HAS_VERTEX_DEPS:
            modes.append("vertex_ai_search")
        return modes

    def set_mode(self, mode: str) -> bool:
        """Set the grounding mode."""
        if mode not in self.get_available_modes():
            return False
        
        self.mode = mode
        if mode == "vertex_ai_search":
            self._init_vertex()
        return True

    # --- Stubs for Interface Compliance ---
    
    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        # For Drive, "upsert" might mean uploading a file or creating a GDoc
        return {"status": "skipped", "reason": "Managed by Google Drive"}

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        return {"status": "skipped", "reason": "Managed by Google Drive"}

    def load_store(self) -> bool:
        return True

    def save_store(self) -> bool:
        return True

    def list_documents(self) -> List[str]:
        # Would list files in the specific Drive folder
        return []

    def rebuild_index(self) -> None:
        pass

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Gemini handles relevance internally
        return passages

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        # Gemini has a specific 'grounding' API we can use here
        return {"verified": True, "confidence": 0.9}

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        return {"status": "skipped"}

    def flush_cache(self) -> Dict[str, Any]:
        return {"status": "ok"}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "backend": "google-gemini",
            "model": "gemini-1.5-pro"
        }
