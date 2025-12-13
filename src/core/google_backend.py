"""Google Gemini + Drive Backend Implementation.

This backend bypasses the local pgvector-backed store and uses Google's native
"Long Context" capabilities (Gemini 1.5 Pro) to perform RAG directly over
documents stored in Google Drive.
"""
# pylint: disable=too-many-lines

import os
import logging
import base64
import io
import re
import json
import datetime
from typing import Dict, Any, List, Optional

# You will need to install: google-generativeai google-auth google-auth-oauthlib
# google-auth-httplib2
try:
    # import google.generativeai as genai
    # from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp
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

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from src.core.interfaces import RAGBackend
from src.core.config_paths import (
    CONFIG_DIR,
    VERTEX_CONFIG_PATH,
    get_ca_bundle_path,
)
from src.core.exceptions import ConfigurationError, AuthenticationError, ProviderError
try:
    from src.core.google_auth import GoogleAuthManager
except ImportError:
    GoogleAuthManager = None

logger = logging.getLogger(__name__)

# 5 minutes timeout for large context operations
TIMEOUT_SECONDS = 300

# Gemini API supported image MIME types
# Note: TIFF (image/tiff) is NOT supported by Gemini API
GEMINI_SUPPORTED_IMAGE_TYPES = {
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/gif',
    'image/webp'
}


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class GoogleGeminiBackend(RAGBackend):
    """
    Implements RAGBackend using Google Gemini (default: 2.0 Flash) and Google Drive.

    Authentication:
    - Uses OAuth2 credentials via GoogleAuthManager.
    - Supports both Drive API and Generative Language API.
    - Supports Vertex AI Grounding (Enterprise).
    """

    def __init__(self):
        if not HAS_GOOGLE_DEPS:
            raise ImportError(
                "Google dependencies missing. Install 'google-generativeai' and "
                "'google-api-python-client'."
            )

        # Load Vertex Config if exists
        vertex_config = self._load_vertex_config()
        if vertex_config:
            os.environ.update(vertex_config)

        # Initialize Auth Manager
        self.auth_manager = GoogleAuthManager()
        self.creds = self.auth_manager.get_credentials()

        # Configure httplib2 with CA bundle if needed
        ca_bundle = get_ca_bundle_path()
        self.http_client = httplib2.Http(
            timeout=TIMEOUT_SECONDS,
            ca_certs=ca_bundle
        )

        # Mode configuration
        self.mode = os.getenv("GOOGLE_GROUNDING_MODE", "google_gemini")  # google_gemini | vertex_ai_search
        self.model_name = os.getenv(
            "GOOGLE_MODEL_NAME", "models/gemini-2.0-flash"
        )

        # Initialize services to None to avoid AttributeError if auth fails
        self.drive_service = None
        self.gmail_service = None
        self.gen_service = None
        self.model = None

        if not self.creds:
            logger.warning("No valid Google credentials found. Please authenticate.")
            return

        # Initialize Drive Service
        try:
            authorized_http = AuthorizedHttp(self.creds, http=self.http_client)
            self.drive_service = build('drive', 'v3', http=authorized_http)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to initialize Drive service (offline?): %s", e)
            self.drive_service = None

        # Initialize Gmail Service
        try:
            authorized_http = AuthorizedHttp(self.creds, http=self.http_client)
            self.gmail_service = build('gmail', 'v1', http=authorized_http)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to initialize Gmail service (offline?): %s", e)
            self.gmail_service = None

        # Initialize Gemini (Generative Language API)
        # Note: google-generativeai SDK primarily uses API keys.
        # For OAuth, we use the lower-level googleapiclient or configure genai if supported.
        # Here we use googleapiclient for the model to ensure OAuth compliance.
        try:
            http_client = httplib2.Http(timeout=TIMEOUT_SECONDS)
            authorized_http = AuthorizedHttp(self.creds, http=http_client)
            self.gen_service = build(
                'generativelanguage', 'v1beta', static_discovery=False, http=authorized_http
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to initialize Gemini service (offline?): %s", e)
            self.gen_service = None

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
            # To use our user credentials, we might need to pass them explicitly if
            # supported,
            # or rely on the environment being set up correctly.
            # For now, we assume the environment (or gcloud auth application-default login)
            # is set up,
            # OR we try to pass credentials if the SDK allows (it usually does via client
            # options).

            # Note: vertexai.init doesn't take credentials directly in all versions.
            # It's better to let google.auth find them or set
            # GOOGLE_APPLICATION_CREDENTIALS.
            # But since we have self.creds, let's see if we can use them.
            # The vertexai SDK uses google.auth.credentials.Credentials.

            vertexai.init(project=project_id, location=location, credentials=self.creds)
            logger.info("Vertex AI initialized for project %s", project_id)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize Vertex AI: %s", e)

    def _load_vertex_config(self) -> Optional[Dict[str, str]]:
        """Load Vertex config from config directory, migrating legacy files if present."""
        try:
            if VERTEX_CONFIG_PATH.exists():
                with open(VERTEX_CONFIG_PATH, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            # Migrated: Legacy vertex config path removal logic deleted
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load vertex config: %s", exc)
        return None

    def reload_auth(self):
        """Reload credentials and re-initialize services."""
        logger.info("Reloading Google credentials...")
        self.creds = self.auth_manager.authenticate()

        if self.creds:
            http_client = httplib2.Http(timeout=TIMEOUT_SECONDS)
            self.drive_service = build(
                'drive', 'v3', http=AuthorizedHttp(self.creds, http=http_client)
            )

            http_client = httplib2.Http(timeout=TIMEOUT_SECONDS)
            self.gmail_service = build(
                'gmail', 'v1', http=AuthorizedHttp(self.creds, http=http_client)
            )

            http_client = httplib2.Http(timeout=TIMEOUT_SECONDS)
            self.gen_service = build(
                'generativelanguage', 'v1beta', static_discovery=False,
                http=AuthorizedHttp(self.creds, http=http_client)
            )
            logger.info("Google services re-initialized.")
        else:
            logger.warning("Reload failed: No valid credentials found.")

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from a natural language query."""
        stop_words = {
            "search", "find", "show", "me", "files", "documents", "about", "that", "discuss",
            "are", "there", "any", "in", "my", "google", "drive", "the", "a", "an", "for", "of",
            "you", "need", "to", "look", "through", "all", "and", "their", "content", "because",
            "is", "it", "with", "on", "at", "by", "from", "up", "down", "over", "under", "again",
            "please", "can", "could", "would", "should", "will", "do", "does", "did", "has", "have",
            "file", "folder", "named", "called", "title", "text", "txt", "email", "emails",
            "message", "messages", "attachment", "attachments", "containing",
            "information", "info", "details", "regarding", "related",
            "january", "february", "march", "april", "may", "june", "july", "august", "september",
            "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            "know", "had", "were", "was", "been", "being", "be", "am", "i", "we", "they",
            "he", "she", "this", "these", "those", "here", "where",
            "when", "why", "how", "which", "who", "whom", "whose",
            "as", "year", "ok", "great", "hello", "hi", "earlier", "later", "ago", "recent",
            "recently", "forwarded", "fwd", "attached",
            "someone", "else", "sometime", "around", "female", "male", "person",
            "people",
            "no", "not", "yes", "yep", "nope",
            "his", "her", "him", "hers", "theirs", "mine", "our", "ours"
        }

        # Words to keep even if they are short
        keep_words = {
            "dr", "mr", "ms", "ai", "id", "hr", "vp", "ceo", "cto", "cfo", "ct", "er", "pr",
            "md", "rn"
        }

        # Clean punctuation but keep dots for filenames? No, dots attach to words.
        # We should replace dots with spaces unless they look like a file extension or email.
        # Simple approach: replace all non-alphanumeric characters (except maybe @) with spaces.
        clean_query = query.lower()
        for char in ".,?!-:;\"'()[]{}":
            clean_query = clean_query.replace(char, " ")

        words = clean_query.split()

        # Handle negation: if "no" or "not" precedes a word, ignore it.
        # We need to look at the original query structure roughly.
        # Since we replaced punctuation with spaces, the order is preserved.
        ignored_indices = set()
        for i, w in enumerate(words):
            if w in ["no", "not"] and i + 1 < len(words):
                ignored_indices.add(i + 1)

        keywords = []
        for i, w in enumerate(words):
            if i in ignored_indices:
                continue
            if w not in stop_words and (len(w) > 2 or w in keep_words):
                keywords.append(w)

        return keywords

    def _construct_drive_query(self, query: str) -> str:
        """Construct a Drive API query from a natural language string."""
        keywords = self._extract_keywords(query)

        if not keywords:
            # Fallback: if everything was filtered, use the original query if it's short
            words = query.lower().split()
            if len(words) < 3:
                return f"fullText contains '{query}' and trashed = false"
            return "trashed = false"

        # Construct query: (name contains 'k' or fullText contains 'k')
        clauses = []
        for k in keywords:
            # Escape single quotes in keywords
            k_safe = k.replace("'", "\\'")
            clauses.append(
                f"(name contains '{k_safe}' or fullText contains '{k_safe}')"
            )

        return " and ".join(clauses) + " and trashed = false"

    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """
        Search Google Drive for files matching the query.
        Note: kwargs like model/temperature are not used for Drive search.
        """
        if not self.drive_service:
            # Try to reload if not initialized
            self.reload_auth()
            if not self.drive_service:
                raise AuthenticationError("Google Drive not authenticated. Please login via settings.")

        # Construct a better query
        q = self._construct_drive_query(query)
        logger.info("Searching Google Drive with query: %s (Original: %s)", q, query)

        try:
            # pylint: disable=no-member
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
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Drive search failed: %s", e)
            if "401" in str(e) or "403" in str(e):
                raise AuthenticationError(f"Drive access denied: {e}") from e
            raise ProviderError(f"Drive search failed: {e}") from e

    def _download_attachment(self, message_id: str, attachment_id: str) -> Optional[bytes]:
        """Download an attachment from a Gmail message."""
        try:
            # pylint: disable=no-member
            attachment = self.gmail_service.users().messages().attachments().get(
                userId='me', messageId=message_id, id=attachment_id
            ).execute()
            data = base64.urlsafe_b64decode(attachment['data'])
            return data
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to download attachment %s: %s", attachment_id, e)
            return None

    def _generate_smart_query(self, user_query: str) -> Optional[str]:
        """
        Use Gemini to generate an optimized Gmail search query from natural language.
        """
        if not self.gen_service:
            return None

        current_date = datetime.date.today().strftime("%Y-%m-%d")

        try:
            prompt = f"""
            You are an expert at creating Google Gmail search queries.
            Convert the following natural language request into a concise and effective Gmail search query.

            Context:
            - Current Date: {current_date}
            - "This year" means the current year in the date above.

            Rules:
            1. Remove conversational filler (e.g., "as you know", "please find").
            2. Use OR operators for likely synonyms (e.g., "physician" -> "(physician OR doctor OR dr)").
            3. Use standard Gmail search operators (e.g., "has:attachment", "from:", "to:", "subject:", "after:", "before:").
            4. Do NOT use invalid operators like "forwarded:".
            5. If the user says "forwarded", you MAY use "fwd" or "forwarded" as a keyword.
            6. Do NOT use "filename:" unless the user explicitly says "filename is X". If they say "with a letter attached", just search for "has:attachment" and maybe the word "letter" as a loose term.
            7. If the user mentions a date range (e.g. "June-August"), ALWAYS include the 'after:' and 'before:' operators.
            8. CRITICAL: If the user says "letter from X attached" or "forwarded email from X", do NOT use "from:X" unless X is clearly the sender of the email. Instead, just use "X" as a keyword.
            9. CRITICAL: If the user says "not X" or "no X is someone else", do NOT include X in the query.
            10. Do NOT include descriptive terms like "female", "male", "person", "someone", "else" unless they are likely part of the email content.
            11. If the user describes a person (e.g. "his PA", "the manager"), do NOT include the description as a keyword unless it's a proper noun.
            12. If the user says "around [Month]", expand the date range to include the month before and after.
            13. Output ONLY the raw query string, no markdown or explanations.

            Request: "{user_query}"
            """

            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 100
                }
            }

            # pylint: disable=no-member
            response = self.gen_service.models().generateContent(
                model=self.model_name,
                body=body
            ).execute()

            candidates = response.get('candidates', [])
            if candidates:
                query = candidates[0]['content']['parts'][0]['text'].strip()
                # Remove any markdown code blocks if the model adds them
                query = query.replace("```", "").strip()
                logger.info("Smart Query Generated: '%s' (from '%s')", query, user_query)
                return query
            return None

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Smart query generation failed: %s", e)
            return None

    # pylint: disable=too-many-locals,too-many-statements,too-many-branches,too-many-nested-blocks,cell-var-from-loop
    def search_gmail(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search Gmail for emails matching the query.
        """
        if not self.gmail_service:
            self.reload_auth()
            if not self.gmail_service:
                raise AuthenticationError("Gmail not authenticated.")

        # Try to generate a smart query first
        smart_query = self._generate_smart_query(query)
        used_smart_query = False

        if smart_query:
            gmail_query = smart_query
            used_smart_query = True
        else:
            # Use keyword extraction for better recall
            keywords = self._extract_keywords(query)

            # Check for attachment intent
            # "attach" covers "attachment", "attached", "attaching"
            has_attachment = "attach" in query.lower()

            if keywords:
                # Join with spaces for implicit AND
                gmail_query = " ".join(keywords)
            else:
                # Fallback to original if no keywords found
                gmail_query = query

            if has_attachment:
                gmail_query += " has:attachment"

        logger.info("Searching Gmail for: %s (Original: %s)", gmail_query, query)

        try:
            # Search messages
            # pylint: disable=no-member
            results = self.gmail_service.users().messages().list(
                userId='me', q=gmail_query, maxResults=top_k
            ).execute()

            messages = results.get('messages', [])

            # Fallback: If few results, try relaxing the query
            if len(messages) < 3:
                # If smart query failed, try basic keyword search
                if used_smart_query:
                    logger.info("Smart query returned few results. Falling back to keyword search.")
                    keywords = self._extract_keywords(query)
                    if keywords:
                        relaxed_query = " ".join(keywords)

                        # Preserve date filters from smart query if they exist
                        if "after:" in smart_query:
                            after_match = re.search(r'after:(\S+)', smart_query)
                            if after_match:
                                relaxed_query += f" after:{after_match.group(1)}"
                        if "before:" in smart_query:
                            before_match = re.search(r'before:(\S+)', smart_query)
                            if before_match:
                                relaxed_query += f" before:{before_match.group(1)}"

                        if "attach" in query.lower():
                            relaxed_query += " has:attachment"

                        logger.info("Fallback query: %s", relaxed_query)
                        relaxed_results = self.gmail_service.users().messages().list(
                            userId='me', q=relaxed_query, maxResults=top_k
                        ).execute()

                        new_messages = relaxed_results.get('messages', [])
                        # Merge
                        existing_ids = {m['id'] for m in messages}
                        for m in new_messages:
                            if m['id'] not in existing_ids:
                                messages.append(m)
                                existing_ids.add(m['id'])

                # If still few results, try dropping keywords (Relaxation)
                if len(messages) < 3:
                    keywords = self._extract_keywords(query)
                    if len(keywords) > 2:
                        # Drop the last keyword
                        relaxed_keywords = keywords[:-1]
                        relaxed_query = " ".join(relaxed_keywords)
                        if "attach" in query.lower():
                            relaxed_query += " has:attachment"

                        logger.info("Few results found. Trying relaxed query: %s", relaxed_query)

                        relaxed_results = self.gmail_service.users().messages().list(
                            userId='me', q=relaxed_query, maxResults=top_k
                        ).execute()

                        new_messages = relaxed_results.get('messages', [])

                        # Merge and deduplicate
                        existing_ids = {m['id'] for m in messages}
                        for m in new_messages:
                            if m['id'] not in existing_ids:
                                messages.append(m)
                                existing_ids.add(m['id'])

            email_data = []

            for msg in messages:
                try:
                    # Get full message details
                    full_msg = self.gmail_service.users().messages().get(
                        userId='me', id=msg['id'], format='full'
                    ).execute()

                    payload = full_msg.get('payload', {})
                    headers = payload.get('headers', [])

                    subject = next(
                        (h['value'] for h in headers if h['name'].lower() == 'subject'),
                        'No Subject'
                    )
                    sender = next(
                        (h['value'] for h in headers if h['name'].lower() == 'from'),
                        'Unknown'
                    )
                    date = next(
                        (h['value'] for h in headers if h['name'].lower() == 'date'),
                        'Unknown'
                    )
                    snippet = full_msg.get('snippet', '')

                    # Extract body and attachments
                    body_text = ""
                    attachments = []

                    def walk_parts(parts):
                        nonlocal body_text
                        for part in parts:
                            mime_type = part.get('mimeType')
                            filename = part.get('filename')

                            if part.get('parts'):
                                walk_parts(part['parts'])

                            # Extract text body
                            if (mime_type == 'text/plain' and 'body' in part and
                                    'data' in part['body']):
                                body_text += base64.urlsafe_b64decode(
                                    part['body']['data']
                                ).decode('utf-8')

                            # Extract attachments
                            if filename and 'body' in part and 'attachmentId' in part['body']:
                                attachments.append({
                                    'id': part['body']['attachmentId'],
                                    'filename': filename,
                                    'mimeType': mime_type,
                                    'size': int(part['body'].get('size', 0))
                                })

                    if 'parts' in payload:
                        walk_parts(payload['parts'])
                    elif 'body' in payload and 'data' in payload['body']:
                        # Single part message
                        body_text = base64.urlsafe_b64decode(
                            payload['body']['data']
                        ).decode('utf-8')

                    email_data.append({
                        "id": msg['id'],
                        "subject": subject,
                        "sender": sender,
                        "date": date,
                        "snippet": snippet,
                        "body": body_text or snippet,
                        "attachments": attachments
                    })

                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.warning("Failed to fetch email %s: %s", msg['id'], e)

            sources = [f"gmail://{e['id']} ({e['subject']})" for e in email_data]

            return {
                "answer": f"Found {len(email_data)} emails matching '{query}'",
                "sources": sources,
                "emails": email_data
            }

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Gmail search failed: %s", e)
            if "401" in str(e) or "403" in str(e):
                raise AuthenticationError(f"Gmail access denied: {e}") from e
            raise ProviderError(f"Gmail search failed: {e}") from e

    def grounded_answer(self, question: str, k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate an answer using Gemini with content from Drive files.
        Dispatches to the configured mode (google_gemini or vertex_ai_search).
        """
        if self.mode == "vertex_ai_search":
            return self.grounded_answer_vertex(question)

        # Default: Google Gemini with Drive API + Gemini
        return self.grounded_answer_gemini(question, k, **kwargs)

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
            # Format: projects/{project_id}/locations/{location}/collections/default_collection/
            # dataStores/{data_store_id}
            project_id = os.getenv("VERTEX_PROJECT_ID")
            location = os.getenv("VERTEX_LOCATION", "global")

            # The SDK might handle the full path construction if we use the helper,
            # but usually we need to pass the full resource name or just the ID if using a
            # specific helper.
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

            model = GenerativeModel(self.model_name.replace("models/", ""))
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
                            sources.append(
                                f"{chunk.retrieved_context.title} ({chunk.retrieved_context.uri})"
                            )

            return {
                "answer": answer_text,
                "sources": sources,
                "mode": "vertex_ai_search"
            }

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Vertex AI generation failed: %s", e)
            return {"error": str(e)}

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks
    def _fetch_context(self, query: str, k: int = 3) -> tuple[
            List[Dict[str, Any]], list[str], list[str]]:
        """
        Helper to fetch and format context from Drive and Gmail.
        Returns:
            - context_parts: List of Gemini API parts (text and inlineData)
            - sources: List of source URIs
            - errors: List of error messages
        """
        gemini_parts = []
        sources = []
        errors = []

        query_lower = query.lower()
        search_drive = True
        search_gmail = True

        # Simple intent detection
        if "drive" in query_lower and "gmail" not in query_lower and "email" not in query_lower:
            search_gmail = False
        elif ("gmail" in query_lower or "email" in query_lower) and "drive" not in query_lower:
            search_drive = False

        # 1. Find relevant files in Drive
        if search_drive:
            drive_res = self.search(query, top_k=k)
            if "error" in drive_res:
                errors.append(f"Drive Error: {drive_res['error']}")
            else:
                files = drive_res.get("files", [])
                logger.info("Drive search found %d files.", len(files))
                for f in files:
                    try:
                        file_id = f['id']
                        mime = f['mimeType']
                        name = f['name']
                        content = ""

                        # Try to extract content based on MIME type
                        if "application/vnd.google-apps.document" in mime:
                            # pylint: disable=no-member
                            content = self.drive_service.files().export(
                                fileId=file_id, mimeType="text/plain"
                            ).execute().decode('utf-8')
                            gemini_parts.append({
                                "text": f"File: {name} (ID: {file_id}, Type: {mime})\n"
                                        f"Content:\n{content}\n"
                            })
                        elif "text/plain" in mime:
                            # pylint: disable=no-member
                            content = self.drive_service.files().get_media(
                                fileId=file_id
                            ).execute().decode('utf-8')
                            gemini_parts.append({
                                "text": f"File: {name} (ID: {file_id}, Type: {mime})\n"
                                        f"Content:\n{content}\n"
                            })
                        elif "application/pdf" in mime:
                            # Download PDF content for Gemini
                            # pylint: disable=no-member
                            data = self.drive_service.files().get_media(fileId=file_id).execute()
                            gemini_parts.append({
                                "text": f"File: {name} (ID: {file_id}, Type: {mime})\n"
                            })
                            gemini_parts.append({
                                "inlineData": {
                                    "mimeType": "application/pdf",
                                    "data": base64.b64encode(data).decode('utf-8')
                                }
                            })
                            gemini_parts.append({"text": "\n"})
                        else:
                            content = f"[File type {mime} - Content not extracted]"
                            gemini_parts.append({
                                "text": f"File: {name} (ID: {file_id}, Type: {mime})\n"
                                        f"Content:\n{content}\n"
                            })

                        sources.append(f"drive://{file_id}")

                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.warning("Failed to read file %s: %s", f.get('name'), e)
                        gemini_parts.append({
                            "text": f"File: {f.get('name')} (ID: {f.get('id')})\n"
                                    f"[Error reading content: {str(e)}]\n"
                        })

        # 2. Find relevant emails in Gmail
        if search_gmail:
            gmail_res = self.search_gmail(query, top_k=k)
            if "error" in gmail_res:
                errors.append(f"Gmail Error: {gmail_res['error']}")
            else:
                emails = gmail_res.get("emails", [])
                logger.info("Gmail search found %d emails.", len(emails))
                for e in emails:
                    # Text Part
                    text_content = (f"Email: {e['subject']} (from {e['sender']})\n"
                                    f"Date: {e['date']}\nContent:\n{e['body']}\n")

                    # Handle Attachments
                    attachments_text = ""
                    attachment_parts = []

                    if e.get('attachments'):
                        attachments_text = "\nAttachments:\n"
                        for att in e['attachments']:
                            attachments_text += f"- {att['filename']} ({att['mimeType']})\n"

                            # Check if supported type and reasonable size (< 10MB)
                            if att['size'] < 10 * 1024 * 1024:
                                # PDF and Images (Gemini Native)
                                is_pdf = att['mimeType'] == 'application/pdf'
                                is_supported_image = att['mimeType'] in GEMINI_SUPPORTED_IMAGE_TYPES
                                if is_pdf or is_supported_image:
                                    data = self._download_attachment(e['id'], att['id'])
                                    if data:
                                        # Add as inlineData
                                        attachment_parts.append({
                                            "inlineData": {
                                                "mimeType": att['mimeType'],
                                                "data": base64.b64encode(data).decode('utf-8')
                                            }
                                        })
                                elif att['mimeType'].startswith('image/'):
                                    # Unsupported image type (e.g., TIFF)
                                    logger.warning(
                                        "Skipping unsupported image type %s for attachment %s",
                                        att['mimeType'], att['filename']
                                    )
                                    attachments_text += (
                                        f"  [Note: {att['filename']} is in unsupported format "
                                        f"({att['mimeType']}) and was not included]\n"
                                    )
                                # Text Files
                                elif att['mimeType'] == 'text/plain':
                                    data = self._download_attachment(e['id'], att['id'])
                                    if data:
                                        try:
                                            text = data.decode('utf-8')
                                            attachments_text += (
                                                f"\n[Content of {att['filename']}]\n"
                                                f"{text}\n"
                                            )
                                        except Exception:  # pylint: disable=broad-exception-caught
                                            pass
                                # Word Docs
                                elif (att['mimeType'] == 'application/vnd.openxmlformats-'
                                      'officedocument.wordprocessingml.document' and HAS_DOCX):
                                    data = self._download_attachment(e['id'], att['id'])
                                    if data:
                                        try:
                                            doc = docx.Document(io.BytesIO(data))
                                            text = "\n".join([p.text for p in doc.paragraphs])
                                            attachments_text += (
                                                f"\n[Content of {att['filename']}]\n"
                                                f"{text}\n"
                                            )
                                        except Exception as err:  # pylint: disable=broad-exception-caught
                                            logger.warning(
                                                "Failed to parse docx %s: %s",
                                                att['filename'], err
                                            )

                    gemini_parts.append({"text": text_content + attachments_text})
                    gemini_parts.extend(attachment_parts)
                    gemini_parts.append({"text": "\n---\n"})

                    sources.append(f"gmail://{e['id']}")

        return gemini_parts, sources, errors

    def list_models(self) -> List[str]:
        """List available Gemini models."""
        if not self.gen_service:
            return []

        try:
            # pylint: disable=no-member
            models = self.gen_service.models().list().execute()
            # Filter for generateContent support and gemini models
            gemini_models = [
                m['name'].replace('models/', '')
                for m in models.get('models', [])
                if 'generateContent' in m.get('supportedGenerationMethods', [])
                and 'gemini' in m['name']
            ]

            # Sort to put newer/pro models first
            def sort_key(name):
                # Extract version (e.g. 1.5, 1.0)
                version_match = re.search(r'(\d+)\.(\d+)', name)
                version_score = 0
                if version_match:
                    # 1.5 -> 150, 2.0 -> 200
                    version_score = (
                        int(version_match.group(1)) * 100 +
                        int(version_match.group(2)) * 10
                    )

                # Tier score
                tier_score = 0
                if 'ultra' in name:
                    tier_score = 3
                elif 'pro' in name:
                    tier_score = 2
                elif 'flash' in name:
                    tier_score = 1

                # Prefer "latest" versions
                latest_score = 0.5 if 'latest' in name else 0

                return version_score + tier_score + latest_score

            return sorted(gemini_models, key=sort_key, reverse=True)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to list models: %s", e)
            return []

    # pylint: disable=too-many-locals
    def grounded_answer_gemini(self, question: str, k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate an answer using Gemini API with content from Drive files and Gmail
        (Google Gemini Mode).
        """
        if not self.gen_service or not self.drive_service:
            self.reload_auth()
            if not self.gen_service or not self.drive_service:
                raise AuthenticationError("Google services not authenticated.")

        model = kwargs.get("model")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        target_model = f"models/{model}" if model else self.model_name
        logger.info("Generating grounded answer with %s for: %s", target_model, question)

        context_parts, sources, errors = self._fetch_context(question, k=k)

        # Log the context for debugging hallucinations
        if context_parts:
            logger.info("Context retrieved for generation (%d parts)", len(context_parts))
        else:
            logger.info("No context retrieved for generation.")

        # Check for API enablement errors to report
        api_error_msg = ""
        for err in errors:
            match = re.search(
                r'(https://console\.developers\.google\.com/apis/api/[^/]+/overview\?'
                r'project=\d+)',
                err
            )
            if match:
                url = match.group(1)
                api_name = "Google API"
                if "gmail" in url:
                    api_name = "Gmail API"
                elif "drive" in url:
                    api_name = "Google Drive API"
                api_error_msg += (
                    f" The {api_name} is not enabled. Please enable it at {url}."
                )
        
        if api_error_msg:
             raise ConfigurationError(f"Google API Config Error:{api_error_msg}")

        if not context_parts and not api_error_msg:
             # Just return empty result if truly no docs found
             return {"answer": "No relevant documents or emails found.", "sources": []}

        # 5. Call Gemini via REST API (v1beta)
        system_instruction = """
        You are a helpful assistant with access to the user's Google Drive files and Gmail.
        Answer the following question based ONLY on the provided documents and emails.
        """

        parts = [{"text": system_instruction + "\n\nContext:\n"}]
        parts.extend(context_parts)
        parts.append({"text": f"\nQuestion: {question}"})

        try:
            body = {
                "contents": [{
                    "parts": parts
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }

            # pylint: disable=no-member
            response = self.gen_service.models().generateContent(
                model=target_model,
                body=body
            ).execute()

            candidates = response.get('candidates', [])
            if candidates:
                answer_text = candidates[0]['content']['parts'][0]['text']
                return {
                    "answer": answer_text,
                    "sources": sources,
                    "mode": "google_gemini"
                }
            # pylint: disable=no-else-return
            return {"answer": "No answer generated by model.", "sources": sources}

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Gemini generation failed: %s", e)
            if "401" in str(e) or "403" in str(e):
                raise AuthenticationError(f"Gemini API refused access: {e}") from e
            raise ProviderError(f"Gemini generation failed: {e}") from e

    # pylint: disable=too-many-locals
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """
        Conversational chat with Gemini 1.5 Pro, with RAG context from Drive/Gmail.
        """
        if not self.gen_service:
            self.reload_auth()
            if not self.gen_service:
                return {"error": "Not authenticated"}

        model = kwargs.get("model")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        target_model = f"models/{model}" if model else self.model_name
        logger.info("Chat request received (model=%s)", target_model)

        # 1. Get the latest user message to use as a search query
        last_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
        query = last_user_msg["content"] if last_user_msg else ""

        context_parts = []
        api_error_msg = ""

        if query:
            # We use a larger k for chat since Gemini 1.5 has a large context window
            context_parts, _, errors = self._fetch_context(query, k=10)

            for err in errors:
                match = re.search(
                    r'(https://console\.developers\.google\.com/apis/api/[^/]+/overview\?'
                    r'project=\d+)',
                    err
                )
                if match:
                    url = match.group(1)
                    api_name = "Google API"
                    if "gmail" in url:
                        api_name = "Gmail API"
                    elif "drive" in url:
                        api_name = "Google Drive API"
                    api_error_msg += (
                        f"\n\n⚠️ **Action Required**: The {api_name} is not enabled. "
                        f"[Click here to enable it]({url})."
                    )

        # If we have no context and a critical error, return early
        if not context_parts and api_error_msg:
            return {
                "role": "assistant",
                "content": f"I couldn't search your data because of a missing permission."
                           f"{api_error_msg}"
            }

        contents = []
        for i, m in enumerate(messages):
            role = "user" if m["role"] == "user" else "model"
            if m["role"] == "assistant":
                role = "model"

            # Inject context into the LAST user message
            parts = [{"text": m["content"]}]

            if i == len(messages) - 1 and role == "user":
                if context_parts:
                    new_parts = [{"text": "Relevant Context from Drive/Gmail:\n"}]
                    new_parts.extend(context_parts)
                    new_parts.append({"text": "\n\nUser Query: " + m["content"]})
                    parts = new_parts
                else:
                    # Explicitly tell the model we found nothing
                    parts = [{
                        "text": (
                            f"System Note: The RAG system searched for '{query}' but found no "
                            f"relevant documents or emails.\n\nUser Query: {m['content']}"
                        )
                    }]

            contents.append({"role": role, "parts": parts})

        try:
            body = {
                "systemInstruction": {
                    "parts": [{
                        "text": (
                            "You are a helpful assistant. You have access to the user's Google "
                            "Drive and Gmail via a RAG system. Relevant documents and emails will "
                            "be injected into the conversation context. If you see context "
                            "provided, use it to answer the user's questions. If the user asks to "
                            "search for something and the system note says no results were found, "
                            "inform the user."
                        )
                    }]
                },
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }

            # pylint: disable=no-member
            response = self.gen_service.models().generateContent(
                model=target_model,
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
            # pylint: disable=no-else-return
            return {"error": "No response generated"}

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Gemini chat failed: %s", e)
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

            # pylint: disable=no-member
            results = self.drive_service.files().list(
                q=q,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, size, webViewLink, iconLink, "
                       "createdTime, modifiedTime)"
            ).execute()

            return results.get('files', [])
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Drive list failed: %s", e)
            return []

    def upload_file(self, name: str, content: bytes, mime_type: str,
                    folder_id: str = None) -> Dict[str, Any]:
        """Upload a file to Google Drive."""
        if not self.drive_service:
            return {"error": "Not authenticated"}

        try:
            file_metadata = {'name': name}
            if folder_id:
                file_metadata['parents'] = [folder_id]
            media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=True)
            # pylint: disable=no-member
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            ).execute()
            return file
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Drive upload failed: %s", e)
            return {"error": str(e)}

    def delete_drive_file(self, file_id: str) -> Dict[str, Any]:
        """Delete a file or folder from Google Drive."""
        if not self.drive_service:
            return {"error": "Not authenticated"}

        try:
            # pylint: disable=no-member
            self.drive_service.files().delete(fileId=file_id).execute()
            return {"success": True}
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Drive delete failed: %s", e)
            return {"error": str(e)}

    def create_drive_folder(self, name: str, parent_id: str = None) -> Dict[str, Any]:
        """Create a folder in Google Drive."""
        if not self.drive_service:
            return {"error": "Not authenticated"}

        try:
            file_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id:
                file_metadata['parents'] = [parent_id]
            # pylint: disable=no-member
            file = self.drive_service.files().create(
                body=file_metadata,
                fields='id, name, mimeType'
            ).execute()
            return file
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Drive folder creation failed: %s", e)
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
        """Get available grounding modes (only if properly authenticated)."""
        modes = []
        
        # Allow google_gemini as soon as we have credentials; services may lazy-init later.
        if self.creds:
            modes.append("google_gemini")
        
        # Only enable Vertex AI if dependencies AND configuration are present
        if HAS_VERTEX_DEPS and os.getenv("VERTEX_PROJECT_ID") and os.getenv("VERTEX_DATA_STORE_ID"):
            # Also check if we have credentials for Vertex
            if self.creds:
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
        """Upsert a document (stub)."""
        # For Drive, "upsert" might mean uploading a file or creating a GDoc
        return {"status": "skipped", "reason": "Managed by Google Drive"}

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        """Index a path (stub)."""
        return {"status": "skipped", "reason": "Managed by Google Drive"}

    def load_store(self) -> bool:
        """Load the store (stub)."""
        return True

    def save_store(self) -> bool:
        """Save the store (stub)."""
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        """List documents (stub)."""
        # Would list files in the specific Drive folder
        return []

    def rebuild_index(self) -> None:
        """Rebuild the index (stub)."""


    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank passages (stub)."""
        # Gemini handles relevance internally
        return passages

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        """Verify grounding (stub)."""
        # Gemini has a specific 'grounding' API we can use here
        return {"verified": True, "confidence": 0.9}

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        """Delete documents (stub)."""
        return {"status": "skipped"}

    def flush_cache(self) -> Dict[str, Any]:
        """Flush cache (stub)."""
        return {"status": "ok"}

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        status = "ok"
        if not self.gen_service or not self.drive_service:
            status = "warning"

        return {
            "status": status,
            "backend": "google-gemini",
            "model": "gemini-1.5-pro",
            "offline_mode": status == "warning"
        }
