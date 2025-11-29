"""
Google OAuth2 Authentication Manager.

Handles the OAuth2 flow for Google APIs (Drive, Gemini/Generative Language).
Manages client secrets, token storage, and credential refreshing.
"""
import logging
import os
from typing import Optional

from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError, TransportError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

# Scopes required for the application
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/generative-language.retriever',
    'https://www.googleapis.com/auth/generative-language.tuning',
    'https://www.googleapis.com/auth/cloud-platform'
]

# Fallback to specific Generative Language scope if cloud-platform is too broad
# SCOPES = [
#     'https://www.googleapis.com/auth/drive.readonly',
#     'https://www.googleapis.com/auth/generative-language'
# ]

class GoogleAuthManager:
    """Manages Google OAuth2 authentication flow."""

    def __init__(self,
                 secrets_path: str = "secrets/client_secrets.json",
                 token_path: str = "secrets/token.json"):
        """Initialize the authentication manager."""
        self.secrets_path = secrets_path
        self.token_path = token_path
        self.creds: Optional[Credentials] = None

    def authenticate(self) -> Optional[Credentials]:
        """
        Get valid user credentials from storage.

        Does NOT trigger interactive flow automatically anymore (use web flow).
        """
        self.creds = None

        # 1. Try to load existing token
        if os.path.exists(self.token_path):
            try:
                # Don't specify SCOPES here. Let it load whatever scopes are in the file.
                # We will check scopes manually if needed, or let the API call fail.
                # Actually, google.auth doesn't enforce scopes on load if not provided,
                # but mismatch causes issues if provided.
                self.creds = Credentials.from_authorized_user_file(self.token_path)
                
                # Check if loaded creds have all required scopes
                # This is tricky because scopes can be partial.
                # For now, rely on the refresh flow or explicit re-auth if API fails.
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to load existing token: %s", exc)
                # Delete invalid token file to prevent repeated errors
                try:
                    os.remove(self.token_path)
                    logger.info("Deleted invalid token file.")
                except OSError:
                    pass
                self.creds = None
                # Delete invalid token file to prevent repeated errors
                try:
                    os.remove(self.token_path)
                    logger.info("Deleted invalid token file.")
                except OSError:
                    pass
                self.creds = None

        # 2. Refresh if expired
        if self.creds and self.creds.expired and self.creds.refresh_token:
            try:
                logger.info("Refreshing expired Google OAuth token...")
                self.creds.refresh(Request())
                # Save refreshed token
                with open(self.token_path, 'w', encoding='utf-8') as token:
                    token.write(self.creds.to_json())
            except TransportError as exc:
                logger.warning(
                    "Token refresh failed due to network issue: %s. Keeping token for retry.", exc)
                # Keep the file, but don't return invalid creds for now
                self.creds = None
            except RefreshError as exc:
                logger.warning("Token refresh failed: %s.", exc)
                # Only delete if it's likely a permanent error (invalid_grant)
                # If it's a temporary server error (5xx), we might want to keep it,
                # but RefreshError usually means the token is bad.
                # We'll be conservative and only delete on explicit invalid_grant/unauthorized.
                error_str = str(exc).lower()
                if ("invalid_grant" in error_str or
                        "unauthorized_client" in error_str or
                        "access_denied" in error_str):
                    try:
                        os.remove(self.token_path)
                        logger.info("Deleted invalid token file after failed refresh.")
                    except OSError:
                        pass
                else:
                    logger.info("Keeping token file in case failure is temporary.")
                self.creds = None
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Token refresh failed (unexpected): %s.", exc)
                # Delete invalid token file
                try:
                    os.remove(self.token_path)
                    logger.info("Deleted invalid token file after failed refresh.")
                except OSError:
                    pass
                self.creds = None

        return self.creds

    def get_credentials(self) -> Optional[Credentials]:
        """Get valid credentials, authenticating if necessary."""
        if not self.creds or not self.creds.valid:
            return self.authenticate()
        return self.creds

    def flow_from_client_secrets(self, redirect_uri: str):
        """Create a Flow instance for web-based auth."""
        if not os.path.exists(self.secrets_path):
            raise FileNotFoundError(
                f"Client secrets file not found at {self.secrets_path}")

        return InstalledAppFlow.from_client_secrets_file(
            self.secrets_path,
            SCOPES,
            redirect_uri=redirect_uri
        )

    def save_credentials(self, creds: Credentials):
        """Save credentials to token.json."""
        self.creds = creds
        with open(self.token_path, 'w', encoding='utf-8') as token:
            token.write(creds.to_json())
        logger.info("Token saved to %s", self.token_path)

    def logout(self):
        """Clear credentials from memory and delete token file."""
        self.creds = None
        if os.path.exists(self.token_path):
            try:
                os.remove(self.token_path)
                logger.info("Deleted token file at %s", self.token_path)
            except Exception as exc:
                logger.error("Error deleting token file: %s", exc)
                raise
