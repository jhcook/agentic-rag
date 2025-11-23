# Google Gemini + Drive Integration (Premium)

This configuration enables the "Native RAG" mode, bypassing the local vector store and using Google Gemini 1.5 Pro's long context window (up to 2M tokens) to answer questions directly from your Google Drive documents.

## Prerequisites

1.  **Google Cloud Project**: You need a Google Cloud project.
2.  **Enable APIs**: You must enable the following APIs for your project:
    *   **Google Drive API**
    *   **Generative Language API** (for Gemini)
    *   *To enable them:* Go to [APIs & Services > Library](https://console.cloud.google.com/apis/library) in the Google Cloud Console, search for each API, and click "Enable".
3.  **OAuth Credentials**: `client_secrets.json` (Desktop App credentials) from the Google Cloud Console.

## Setup

1.  **Install Dependencies**:
    The required packages are already in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Add the following to your `.env` file:
    ```bash
    RAG_MODE=google
    ```

3.  **Authentication**:
    1.  Download your OAuth 2.0 Client ID JSON file from Google Cloud Console.
    2.  Save it as `client_secrets.json` in the root of the project.
    3.  Run the application. It will open a browser window for you to log in with your Google account.
    4.  A `token.json` file will be created to store your session.

## How it Works

-   **Search**: Instead of searching a local FAISS index, the system searches your Google Drive files using the Drive API.
-   **Grounded Answer**:
    1.  Retrieves the full text of relevant documents from Drive.
    2.  Passes the *entire* content to Gemini 1.5 Pro.
    3.  Gemini generates an answer based *only* on those documents, providing high-accuracy grounding without the information loss typical of chunking/embedding.

## Switching Back

To switch back to the local open-source version, simply set:
```bash
RAG_MODE=local
```
in your `.env` file (or remove the variable).

## Troubleshooting

### "Access blocked: App has not completed the Google verification process"

If you see this error (Error 403: access_denied), it means your Google Cloud Project is in **Testing** mode, and your email address has not been added to the list of allowed test users.

**Fix:**
1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Select your project.
3.  Navigate to **APIs & Services** > **OAuth consent screen**.
4.  Under the **Test users** section, click **+ ADD USERS**.
5.  Enter your email address (e.g., `jhcook@gmail.com`) and click **Save**.
6.  Retry the login process in the application.

### "Google Drive API has not been used in project..."

If you see an error like `Google Drive API has not been used in project ... or it is disabled`, it means you skipped enabling the API in the Prerequisites step.

**Fix:**
1.  Visit the link provided in the error message (usually `https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=YOUR_PROJECT_ID`).
2.  Click **Enable**.
3.  Wait a few minutes for the change to propagate.

