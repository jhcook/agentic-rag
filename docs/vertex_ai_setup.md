# Vertex AI Agent Builder Setup (Enterprise)

This guide explains how to set up Google Cloud Vertex AI Agent Builder (formerly Gen App Builder) for use with the "Vertex AI Agent" mode in Agentic RAG. This mode provides enterprise-grade retrieval, grounding, and citation capabilities.

## Prerequisites

1.  **Google Cloud Project**: You need a Google Cloud project with billing enabled.
2.  **Vertex AI API**: Enable the Vertex AI API and Agent Builder API.

## Step 1: Create a Data Store

1.  Go to the [Agent Builder Console](https://console.cloud.google.com/gen-app-builder/engines).
2.  Click **Create App**.
3.  Select **Search** as the type of app.
4.  Select **Generic** (or specific content type if applicable) and turn on **Enterprise edition features** if needed.
5.  **Data Store**:
    *   Click **Create New Data Store**.
    *   Choose your source (e.g., **Google Drive**, **Cloud Storage**, or **Website**).
    *   Follow the prompts to connect your data source (e.g., select a Drive folder).
    *   Click **Create**.
6.  Select the data store you just created and click **Create**.

## Step 2: Get Configuration Values

You will need three values to configure the Agentic RAG UI:

### 1. Project ID
*   This is your Google Cloud Project ID (not the name).
*   You can find it in the top navigation bar of the Google Cloud Console or on the Dashboard.

### 3. Location
*   This is the region where you created your Data Store (e.g., `us-central1`, `global`).
*   If you are unsure, `global` is the default for many Agent Builder apps, but `us-central1` is common for Vertex AI features. Check your App details.

### 3. Data Store ID
*   Go to the [Agent Builder Console](https://console.cloud.google.com/gen-app-builder/engines).
*   Click on your App name.
*   Go to the **Data** tab.
*   The **Data Store ID** is often visible in the URL or settings.
*   *Alternative*: If you are using the "Data Stores" view directly, the ID is listed there.
*   **Format**: It is usually a string like `my-data-store-id_12345`.

## Step 3: Configure Agentic RAG

1.  Open the Agentic RAG UI.
2.  Go to the **Settings** tab.
3.  Expand the **Gemini + Google Drive** section.
4.  Scroll down to **Vertex AI Configuration**.
5.  Enter your **Project ID**, **Location**, and **Data Store ID**.
6.  Click **Save Vertex Configuration**.

## Step 4: Enable Vertex AI Mode

1.  Go to the **Dashboard** tab.
2.  In the **Active Provider** card, click the dropdown.
3.  Select **Vertex AI Agent**.
4.  The system will now use Vertex AI for all search and chat operations.

## Troubleshooting

*   **Permission Denied**: Ensure the account you authenticated with (via `client_secrets.json`) has the `Vertex AI User` or `Discovery Engine Editor` role in IAM for your project.
*   **Not Found**: Double-check your Data Store ID and Location. A mismatch will cause 404 errors.
