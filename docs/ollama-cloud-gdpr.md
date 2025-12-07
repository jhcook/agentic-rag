# Ollama Cloud GDPR Compliance Documentation

**Last Updated**: 2025-12-07  
**Status**: Active

---

## 1. Third-Party Data Processor

### Ollama Cloud as Data Processor

**Ollama Cloud** (operated by Ollama, Inc.) is a third-party data processor that receives and processes personal data on behalf of this application when users enable cloud mode.

**Processor Details**:
- **Name**: Ollama Cloud
- **Service**: Cloud-hosted LLM (Large Language Model) API service
- **Endpoint**: `https://ollama.com` (default)
- **Contact**: See [Ollama Privacy Policy](https://ollama.com/privacy)

---

## 2. Lawful Basis for Processing

**Lawful Basis**: **Contractual Necessity** and **Legitimate Interest**

- **Contractual Necessity**: Users explicitly enable cloud mode and agree to send data to Ollama Cloud as part of the service functionality
- **Legitimate Interest**: Processing is necessary for the performance of the AI service requested by the user

**User Consent**: Users must explicitly:
1. Enable "cloud" or "auto" mode in settings
2. Provide API key for authentication
3. Acknowledge that data will be sent to Ollama Cloud

**Alternative**: Users can use "local" mode to process data entirely locally without sending data to third parties.

---

## 3. Data Categories Processed

### Personal Data Sent to Ollama Cloud

When cloud mode is enabled, the following data categories may be sent to Ollama Cloud:

1. **User Queries**: Text queries and questions entered by users
2. **Document Content**: Text content from indexed documents (when used as context for LLM)
3. **Chat History**: Conversation messages and context (when using chat functionality)
4. **Metadata**: 
   - Model selection preferences
   - API usage patterns (implicitly, through API calls)

### Data NOT Sent to Ollama Cloud

- User authentication credentials (API keys are used for authentication but not stored by Ollama Cloud in retrievable form)
- File metadata (filenames, paths, etc.) - only text content is sent
- User identification information (unless present in document content)
- System configuration (endpoints, settings)

---

## 4. Data Flow Lifecycle

### Source → Use → Storage → Retention → Deletion

#### Source
- **User Input**: Queries, chat messages entered by user
- **Indexed Documents**: Text content extracted from user's documents
- **Configuration**: Model preferences, API keys (for authentication only)

#### Use
- **Purpose**: LLM processing to generate responses, summaries, and answers
- **Processing**: 
  - Text generation
  - Question answering
  - Document summarization
  - Semantic search enhancement

#### Storage
- **Location**: Ollama Cloud servers (location not specified by Ollama)
- **Duration**: Unknown - see Retention section
- **Format**: Text data in API request/response format

#### Retention
- **Ollama Cloud Policy**: Not explicitly documented by Ollama
- **Assumption**: Data may be retained for:
  - Service operation (temporary, during request processing)
  - Logging and debugging (duration unknown)
  - Model improvement (if applicable, per Ollama's privacy policy)

**Recommendation**: Users should assume data may be retained by Ollama Cloud according to their privacy policy. For sensitive data, use local mode.

#### Deletion
- **User Control**: Users can:
  - Switch to local mode to stop sending data
  - Delete API key to disable cloud access
  - Request data deletion from Ollama (per their privacy policy)
- **Automatic**: No automatic deletion mechanism implemented
- **Ollama Cloud**: Users must contact Ollama directly for data deletion requests

---

## 5. Data Minimization

### Minimization Measures

1. **Mode Selection**: Users can choose local mode to avoid sending any data
2. **Selective Usage**: Only document content relevant to queries is sent (not entire document corpus)
3. **No PII Extraction**: System does not extract or send PII separately - only content as provided by user
4. **Optional Feature**: Cloud mode is opt-in, not required for core functionality

### Recommendations for Users

- **Sensitive Data**: Use local mode for processing sensitive personal data
- **Public Data**: Cloud mode is suitable for non-sensitive, public, or anonymized content
- **Regulated Industries**: Healthcare, finance, legal - consider local mode or data redaction

---

## 6. User Rights (GDPR Articles 15-22)

### Right of Access (Article 15)
- Users can view their configuration and API key (masked) in settings
- Users cannot directly access data stored by Ollama Cloud
- **Action Required**: Contact Ollama Cloud for data access requests

### Right to Rectification (Article 16)
- Users can update their API key and configuration
- Users cannot modify data already sent to Ollama Cloud
- **Action Required**: Contact Ollama Cloud for data rectification

### Right to Erasure (Article 17 - "Right to be Forgotten")
- Users can delete their API key to stop future data transmission
- Users can switch to local mode
- **Action Required**: Contact Ollama Cloud to request deletion of historical data

### Right to Restrict Processing (Article 18)
- Users can disable cloud mode at any time
- Switching to local mode immediately stops data transmission
- Historical data remains subject to Ollama Cloud's retention policy

### Right to Data Portability (Article 20)
- Users can export their configuration (API key excluded for security)
- Document content is stored locally and can be exported
- Data sent to Ollama Cloud is not directly exportable through this application

### Right to Object (Article 21)
- Users can object to processing by disabling cloud mode
- No automated decision-making or profiling is performed

---

## 7. Data Transfer and Safeguards

### Cross-Border Transfers
- **Destination**: Ollama Cloud servers (location not specified)
- **Safeguards**: 
  - HTTPS/TLS encryption in transit
  - API key authentication
  - No data transfer to local mode

### Technical Safeguards
- **Encryption in Transit**: All API calls use HTTPS/TLS
- **Authentication**: Bearer token (API key) authentication
- **No Storage**: This application does not store data sent to Ollama Cloud
- **Local Fallback**: Auto mode falls back to local processing if cloud unavailable

---

## 8. Data Breach Procedures

### If Ollama Cloud Reports a Breach
1. **Notification**: Ollama Cloud will notify affected users per their policy
2. **Our Action**: 
   - Notify users via application notification (if contact information available)
   - Recommend changing API keys
   - Recommend switching to local mode if concerned

### If Our System Has a Breach Affecting API Keys
1. **Immediate**: Revoke affected API keys
2. **Notification**: Notify users to regenerate API keys
3. **Documentation**: Log incident per security procedures

---

## 9. Privacy by Design

### Design Principles Applied

1. **Default to Local**: System defaults to local mode (no third-party data transmission)
2. **Explicit Opt-In**: Users must explicitly enable cloud mode
3. **Clear Documentation**: This document explains data processing
4. **User Control**: Users can disable cloud mode at any time
5. **Minimal Data**: Only necessary data (queries, relevant document content) is sent

---

## 10. Contact and Complaints

### Data Protection Inquiries
- **Application**: See main application support channels
- **Ollama Cloud**: See [Ollama Privacy Policy](https://ollama.com/privacy) for their contact information

---

## 11. API Key Storage, Retention, and Deletion (Ollama Cloud)

**Personal data stored by this application**
- Ollama Cloud API key (secret) plus endpoint URL, HTTPS proxy, and CA bundle path (hostnames can be personal data)
- Storage location: local file `secrets/ollama_cloud_config.json` under the application base directory, written with restrictive permissions (rw-------)

**Purpose of storage**
- Authenticate calls to Ollama Cloud and keep connectivity settings between sessions
- Avoid repeated credential prompts while maintaining secure, authenticated requests

**Lawful basis for storing the key (GDPR Art. 6)**
- **Contractual necessity**: Required to deliver the opt-in Ollama Cloud inference feature the user requests
- **Legitimate interest**: Security and continuity of the service (stable authenticated access); local-only mode remains available as an alternative

**Retention**
- Stored locally until the user clears or overwrites the values; there is no automatic time-based deletion of the config file
- Removing the key stops further transmission to Ollama Cloud; any data already sent to Ollama Cloud remains subject to Ollama's retention policy

**Deletion / right to erasure for stored API keys**
- Clear the stored key by sending an empty string (`""`) or `null` for `api_key` to `POST /api/ollama/cloud-config`. Example:

  ```bash
  curl -X POST "http://localhost:8001/api/ollama/cloud-config" \
       -H "Content-Type: application/json" \
       -d '{"api_key": "", "endpoint": null, "proxy": null, "ca_bundle": null}'
  ```
- Optional: delete `secrets/ollama_cloud_config.json` to remove all stored cloud configuration values
- Switching to local mode also prevents any further transmission to Ollama Cloud
- For historical data held by Ollama Cloud, submit a deletion request to Ollama via their privacy policy

### Supervisory Authority
Users have the right to lodge a complaint with their local data protection supervisory authority.

---

## 12. Updates to This Policy

This document will be updated if:
- Data processing practices change
- Ollama Cloud's privacy policy changes significantly
- Legal requirements change
- New features are added that affect data processing

**Last Review**: 2025-12-07  
**Next Review**: 2026-06-07 (or as needed)

---

## Appendix: Quick Reference

| Aspect | Details |
|--------|---------|
| **Processor** | Ollama Cloud (Ollama, Inc.) |
| **Lawful Basis** | Contractual necessity, legitimate interest |
| **Data Categories** | Queries, document content, chat history |
| **Retention** | Local config: until user clears it; Ollama Cloud: see their policy (not documented) |
| **Deletion** | Local: POST `/api/ollama/cloud-config` with empty `api_key` or delete `secrets/ollama_cloud_config.json`; Ollama Cloud: request via their policy |
| **User Control** | Can disable cloud mode at any time |
| **Default** | Local mode (no third-party transmission) |
| **Encryption** | HTTPS/TLS in transit |
| **Safeguards** | API key authentication, local fallback |

---

**Note**: This documentation is based on current understanding of Ollama Cloud's service. For the most up-to-date information about Ollama Cloud's data processing practices, refer to [Ollama's Privacy Policy](https://ollama.com/privacy).

