# Phase 2 Premium Architecture

**Version:** 1.0  
**Date:** December 18, 2025  
**Status:** Planning  
**Author:** @Architect, @Scribe

## Executive Summary

This document defines the architecture for **Premium Capabilities** that extend the free, local-first Agentic RAG application into a paid, multi-tenant SaaS offering. The core principle is that the **free tier remains fully functional offline** while premium features (sync, collaboration, integrations) require authentication and a valid license.

---

## Licensing Model

### Tier Definitions

| Tier | License Key | Price | Features |
|------|-------------|-------|----------|
| **Free** | None required | $0 | Local-only, single device, unlimited documents, full RAG capabilities |
| **Personal Sync** | Required | $9/mo | Multi-device sync, cloud backup, mobile access, conversation history across devices |
| **Teams** | Required | $25/user/mo | Workspaces, shared documents, RBAC, real-time collaboration, Slack/Teams/Discord |
| **Enterprise** | Required | Custom | SSO/SAML, tenant isolation, audit logs, compliance, SLAs, dedicated support |

### License Enforcement

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Start                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Check for license key in config/settings.json               │
│  2. If no key → Free tier (local-only mode)                     │
│  3. If key present → Validate via licensing API                 │
│  4. Cache validation result (24h TTL for offline grace)         │
│  5. Enable/disable features based on tier                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Chat Storage Strategy

### Storage by Tier

| Tier | Primary Storage | Local Cache | Sync |
|------|-----------------|-------------|------|
| **Free** | SQLite (`cache/chat_history.db`) | N/A | None |
| **Personal Sync+** | PostgreSQL (Supabase) | SQLite (offline queue) | Cloud ↔ Desktop ↔ Mobile |
| **Teams / Enterprise** | PostgreSQL (Supabase) | SQLite (offline queue) | Multi-user real-time |

### Rationale

1. **Free tier stays on SQLite**  
   - Works fully offline without cloud dependency
   - Docker/pgvector already required for vector search, but chat doesn't need it
   - Zero friction for local-only users

2. **Premium tiers migrate to PostgreSQL**  
   - Enables cross-device sync via Supabase Realtime
   - Enables Row-Level Security for workspace isolation
   - Enables multi-user conversation attribution
   - SQLite becomes local cache for offline resilience

3. **Hybrid offline-first model**  
   - Desktop app writes to SQLite immediately (instant UI)
   - Background sync pushes to PostgreSQL
   - Conflict resolution uses server-assigned `seq` ordering
   - Offline queue replays on reconnect

### Migration Path

```
Free → Personal Sync:
1. User upgrades and authenticates
2. Existing SQLite chat history offered for import
3. One-time migration to PostgreSQL
4. SQLite retained as offline cache

Free → Teams:
1. Same as above, plus workspace assignment
2. Migrated chats go to user's personal workspace
3. User can share/move conversations later
```

### Real-Time Message Delivery

Premium tiers use **Supabase Realtime** for instant message delivery:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Message Send Flow                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐  1. POST /messages   ┌──────────────┐              │
│  │  Alice  │ ────────────────────►│  REST API    │              │
│  │ Desktop │                      │  (FastAPI)   │              │
│  └─────────┘                      └──────┬───────┘              │
│                                          │                       │
│                                   2. INSERT                      │
│                                          ▼                       │
│                               ┌──────────────────┐              │
│                               │   PostgreSQL     │              │
│                               │  conversation_   │              │
│                               │    messages      │              │
│                               └────────┬─────────┘              │
│                                        │                         │
│                               3. Trigger Realtime                │
│                                        ▼                         │
│                               ┌──────────────────┐              │
│                               │ Supabase Realtime│              │
│                               │  (WebSocket Hub) │              │
│                               └────────┬─────────┘              │
│                                        │                         │
│                    ┌───────────────────┼───────────────────┐    │
│                    │                   │                   │    │
│             4. Broadcast        4. Broadcast        4. Broadcast│
│                    ▼                   ▼                   ▼    │
│              ┌─────────┐        ┌─────────┐        ┌─────────┐  │
│              │  Alice  │        │   Bob   │        │  Carol  │  │
│              │ Mobile  │        │ Desktop │        │  Slack  │  │
│              └─────────┘        └─────────┘        └─────────┘  │
│                                                                  │
│  Latency: ~50-100ms from INSERT to all clients                  │
└─────────────────────────────────────────────────────────────────┘
```

**Client Subscription (React/TypeScript):**

```typescript
// Subscribe to conversation messages
const channel = supabase
  .channel(`conversation:${conversationId}`)
  .on(
    'postgres_changes',
    {
      event: 'INSERT',
      schema: 'public',
      table: 'conversation_messages',
      filter: `conversation_id=eq.${conversationId}`
    },
    (payload) => {
      // New message received instantly
      const newMessage = payload.new as ConversationMessage;
      addMessageToUI(newMessage);
    }
  )
  .subscribe();
```

**Optimistic UI (Instant Feel):**

```typescript
async function sendMessage(content: string) {
  // 1. Generate client-side ID for optimistic update
  const clientMessageId = crypto.randomUUID();
  
  // 2. Show message immediately in UI (optimistic)
  addMessageToUI({
    id: clientMessageId,
    content,
    status: 'sending',
    sender_user_id: currentUser.id
  });
  
  // 3. Write to local SQLite cache (offline support)
  await localDb.messages.add({ clientMessageId, content, synced: false });
  
  // 4. POST to server
  const response = await api.post('/messages', {
    conversation_id: conversationId,
    content,
    client_message_id: clientMessageId
  });
  
  // 5. Update UI with server-assigned ID and seq
  updateMessageInUI(clientMessageId, {
    id: response.id,
    seq: response.seq,
    status: 'sent'
  });
  
  // 6. Mark as synced in local cache
  await localDb.messages.update(clientMessageId, { synced: true });
}
```

**Delivery Guarantees:**

| Guarantee | Mechanism |
|-----------|-----------|
| **Ordering** | Server-assigned `seq` per conversation |
| **Deduplication** | `client_message_id` prevents double-sends on retry |
| **Persistence** | PostgreSQL is source of truth |
| **Offline** | SQLite queue replays on reconnect |
| **Real-time** | WebSocket broadcast to all subscribers |

---

## Authentication Architecture

### Supported Authentication Methods

| Method | Tiers | Provider |
|--------|-------|----------|
| Email/Password | Personal+, Teams+, Enterprise | Supabase Auth |
| Magic Link | Personal+, Teams+, Enterprise | Supabase Auth |
| Google OAuth | Personal+, Teams+, Enterprise | Google Identity |
| GitHub OAuth | Personal+, Teams+, Enterprise | GitHub OAuth Apps |
| Microsoft OAuth | Teams+, Enterprise | Azure AD / Entra ID |
| SAML 2.0 SSO | Enterprise | Okta, Azure AD, OneLogin, etc. |
| OIDC | Enterprise | Any OIDC provider |

### Authentication Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Desktop    │     │    Mobile    │     │   Web Portal │
│     App      │     │     App      │     │   (React)    │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │      Auth Gateway       │
              │  (Supabase Auth / SSO)  │
              └────────────┬────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │   Supabase   │ │   Azure AD   │ │  SAML IdP    │
   │     Auth     │ │   (OAuth)    │ │  (Okta etc)  │
   └──────────────┘ └──────────────┘ └──────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │      JWT Issued         │
              │  (includes user_id,     │
              │   tenant_id, roles)     │
              └─────────────────────────┘
```

### JWT Token Structure

```json
{
  "sub": "user_uuid",
  "email": "user@example.com",
  "aud": "authenticated",
  "role": "authenticated",
  "app_metadata": {
    "tenant_id": "tenant_uuid",
    "tier": "teams",
    "roles": ["workspace_admin"],
    "permissions": ["read:documents", "write:documents", "manage:members"]
  },
  "user_metadata": {
    "full_name": "Jane Doe",
    "avatar_url": "https://..."
  },
  "iat": 1734500000,
  "exp": 1734503600
}
```

---

## Multi-User Conversation Architecture (Teams+)

### Problem Statement

In **Teams** and **Enterprise** tiers, multiple users share workspaces and can participate in the same conversation threads. The AI agent must know:

1. **When to respond**: Only when explicitly addressed or when configured to participate
2. **When to stay silent**: During human-to-human discussions
3. **How to attribute**: Track who said what and who the agent responded to
4. **How to interrupt**: Whether proactive assistance is enabled and under what conditions

### Conversation Participants Model

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONVERSATION                                 │
│  id: conv_123                                                    │
│  workspace_id: ws_456                                            │
│  type: 'multi_user' | 'direct' | 'agent_only'                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PARTICIPANTS:                                                   │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐       │
│  │  User: Alice   │ │  User: Bob     │ │  Agent: @rag   │       │
│  │  role: human   │ │  role: human   │ │  role: agent   │       │
│  │  joined: t1    │ │  joined: t2    │ │  joined: t0    │       │
│  └────────────────┘ └────────────────┘ └────────────────┘       │
│                                                                  │
│  MESSAGES:                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ [Alice]: Hey Bob, do you know how the auth system works? │   │
│  │ [Bob]: I think it uses JWT, but @rag can explain better  │   │
│  │ [@rag → Bob]: The auth system uses JWT tokens with...    │   │
│  │ [Alice]: Thanks! What about refresh tokens?              │   │
│  │ [@rag → Alice]: Refresh tokens are handled by...         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Participation Modes

| Mode | Trigger | Behavior | Use Case |
|------|---------|----------|----------|
| **Explicit Mention** | `@rag`, `@assistant` | Only responds when directly mentioned | Default for team chats |
| **Question Detection** | Message ends with `?` and no human responded in 30s | Auto-responds to unanswered questions | Helpful but non-intrusive |
| **Always Listening** | Every message | Responds to everything | 1:1 agent chats only |
| **Proactive** | Detects relevant context | Interjects with helpful info | Enterprise feature, opt-in |
| **Silent Observer** | Never | Logs and indexes but never responds | Compliance/archival channels |

### Message Attribution Schema

```sql
-- Extended message model for multi-user conversations
CREATE TABLE conversation_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  
  -- Ordering
  seq BIGINT GENERATED ALWAYS AS IDENTITY,
  
  -- Participant identification
  sender_type TEXT NOT NULL CHECK (sender_type IN ('user', 'agent', 'system')),
  sender_user_id UUID REFERENCES users(id),  -- NULL if sender_type = 'agent' or 'system'
  sender_agent_id TEXT,  -- 'rag', 'assistant', custom agent name
  
  -- Addressing (who is this message directed to?)
  reply_to_message_id UUID REFERENCES conversation_messages(id),
  mentioned_user_ids UUID[],  -- @alice, @bob
  mentioned_agent BOOLEAN DEFAULT FALSE,  -- @rag was mentioned
  
  -- Content
  content_markdown TEXT NOT NULL,
  display_markdown TEXT,
  citations JSONB DEFAULT '[]'::jsonb,
  
  -- Message classification
  kind TEXT NOT NULL CHECK (kind IN (
    'user_message',           -- Human message
    'agent_response',         -- Direct reply to mention
    'agent_proactive',        -- Unsolicited agent message
    'agent_grounded',         -- Answer with citations
    'system_notification',    -- Join/leave, errors
    'human_to_human'          -- Classified as H2H (agent should not respond)
  )),
  
  -- Agent decision metadata
  agent_triggered_by TEXT,    -- 'mention', 'question', 'proactive', 'follow_up'
  agent_addressed_to UUID,    -- Which user the agent is responding to
  
  -- Standard fields
  created_at TIMESTAMPTZ DEFAULT NOW(),
  metadata JSONB
);

CREATE INDEX idx_messages_conv_seq ON conversation_messages(conversation_id, seq);
CREATE INDEX idx_messages_sender ON conversation_messages(sender_user_id);
CREATE INDEX idx_messages_mentioned ON conversation_messages USING GIN (mentioned_user_ids);
```

### Agent Response Decision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    New Message Received                          │
│  From: Alice                                                     │
│  Content: "Hey Bob, what do you think about the new design?"    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              1. Check Agent Participation Mode                   │
│  Workspace setting: 'explicit_mention'                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              2. Check for Agent Mention                          │
│  Contains @rag or @assistant? NO                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │ No mention
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              3. Check for Human Addressee                        │
│  Message mentions @Bob? YES → Human-to-human conversation       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              4. Classify and Store                               │
│  kind = 'human_to_human'                                        │
│  Agent action: SILENT (do not respond)                          │
└─────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    New Message Received                          │
│  From: Bob                                                       │
│  Content: "@rag what's the authentication flow for mobile?"     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              1. Check for Agent Mention                          │
│  Contains @rag? YES                                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              2. Extract Query                                    │
│  Query: "what's the authentication flow for mobile?"            │
│  Addressed to: Agent                                            │
│  Reply context: Bob                                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              3. Execute RAG Pipeline                             │
│  Search → Retrieve → Rerank → Generate                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              4. Store and Broadcast Response                     │
│  kind = 'agent_grounded'                                        │
│  agent_triggered_by = 'mention'                                 │
│  agent_addressed_to = Bob's user_id                             │
│  Send to all conversation participants                          │
└─────────────────────────────────────────────────────────────────┘
```

### Workspace Agent Settings

```sql
-- Agent behavior configuration per workspace
CREATE TABLE workspace_agent_settings (
  workspace_id UUID PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
  
  -- Participation mode
  participation_mode TEXT NOT NULL DEFAULT 'explicit_mention'
    CHECK (participation_mode IN (
      'explicit_mention',    -- Only responds to @rag
      'question_detection',  -- Responds to unanswered questions
      'proactive',           -- May interject with relevant info
      'silent'               -- Never responds, just observes
    )),
  
  -- Agent identity
  agent_display_name TEXT DEFAULT 'RAG Assistant',
  agent_avatar_url TEXT,
  agent_mention_trigger TEXT DEFAULT '@rag',  -- How to summon the agent
  
  -- Proactive settings (if mode = 'proactive')
  proactive_confidence_threshold FLOAT DEFAULT 0.85,  -- Min confidence to interject
  proactive_cooldown_seconds INTEGER DEFAULT 300,      -- Min time between proactive messages
  proactive_max_per_hour INTEGER DEFAULT 5,            -- Rate limit
  
  -- Question detection settings (if mode = 'question_detection')
  question_wait_seconds INTEGER DEFAULT 30,  -- Wait for human response before agent responds
  question_patterns TEXT[],                  -- Custom patterns beyond '?'
  
  -- Threading behavior
  follow_up_enabled BOOLEAN DEFAULT TRUE,    -- Agent responds to follow-ups without re-mention
  follow_up_window_minutes INTEGER DEFAULT 5, -- Time window for follow-ups
  
  -- Boundaries
  blocked_topics TEXT[],                     -- Topics agent should not discuss
  require_grounding BOOLEAN DEFAULT TRUE,    -- Only respond if sources found
  
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Conversation Threading

For complex multi-user discussions, we support **threads** (sub-conversations):

```
Main Conversation: "Q4 Planning"
├── [Alice]: Let's discuss the Q4 roadmap
├── [Bob]: I think we should focus on mobile
├── [Alice]: @rag what features are most requested?
├── [@rag → Alice]: Based on user feedback, the top requests are...
│
└── Thread: "Mobile Auth Discussion" (started by Bob)
    ├── [Bob]: How should we handle mobile auth?
    ├── [Carol]: I suggest OAuth with PKCE
    ├── [Bob]: @rag explain PKCE flow
    └── [@rag → Bob]: PKCE (Proof Key for Code Exchange) is...
```

```sql
-- Thread support
ALTER TABLE conversations ADD COLUMN parent_conversation_id UUID REFERENCES conversations(id);
ALTER TABLE conversations ADD COLUMN thread_title TEXT;
ALTER TABLE conversations ADD COLUMN is_thread BOOLEAN DEFAULT FALSE;

CREATE INDEX idx_conversations_parent ON conversations(parent_conversation_id);
```

### Agent Interruption Policies (Proactive Mode)

When `participation_mode = 'proactive'`, the agent may interject. These policies control when:

| Policy | Description | Example |
|--------|-------------|---------|
| **Factual Correction** | User states something incorrect that's in the knowledge base | "Actually, the deadline was changed to March 15th" |
| **Missing Context** | Discussion would benefit from documented info | "FYI, there's a design doc for this..." |
| **Stale Information** | Reference to outdated document | "Note: That document was superseded by..." |
| **Action Item** | Detects a question/task that matches indexed how-to | "I can help with that deployment process" |

**Interruption Guardrails:**

```python
async def should_agent_interject(
    message: Message,
    conversation: Conversation,
    settings: WorkspaceAgentSettings
) -> tuple[bool, str | None]:
    """
    Decide if agent should proactively respond.
    Returns (should_respond, reason).
    """
    # Never interrupt if mode is not proactive
    if settings.participation_mode != 'proactive':
        return False, None
    
    # Rate limiting
    recent_interjections = await count_agent_messages(
        conversation.id, 
        since=datetime.now() - timedelta(hours=1),
        kind='agent_proactive'
    )
    if recent_interjections >= settings.proactive_max_per_hour:
        return False, "rate_limit_exceeded"
    
    # Cooldown check
    last_interjection = await get_last_agent_message(conversation.id, kind='agent_proactive')
    if last_interjection:
        elapsed = datetime.now() - last_interjection.created_at
        if elapsed.total_seconds() < settings.proactive_cooldown_seconds:
            return False, "cooldown_active"
    
    # Check if humans are actively conversing (back-to-back human messages)
    recent_messages = await get_recent_messages(conversation.id, limit=3)
    if all(m.sender_type == 'user' for m in recent_messages):
        human_gap = (recent_messages[0].created_at - recent_messages[-1].created_at).total_seconds()
        if human_gap < 60:  # Active human conversation
            return False, "humans_actively_conversing"
    
    # Check for relevant knowledge match
    relevance_score = await check_knowledge_relevance(message.content)
    if relevance_score < settings.proactive_confidence_threshold:
        return False, "low_relevance"
    
    return True, "relevant_knowledge_found"
```

### Real-Time Typing Indicators

For multi-user conversations, show who is typing (including the agent when generating):

```typescript
// Real-time presence for typing indicators
interface TypingState {
  conversation_id: string;
  user_id: string | 'agent';
  display_name: string;
  started_at: Date;
}

// Supabase Realtime subscription
supabase
  .channel(`typing:${conversationId}`)
  .on('presence', { event: 'sync' }, () => {
    const state = channel.presenceState<TypingState>();
    setTypingUsers(Object.values(state).flat());
  })
  .subscribe();
```

### UI Considerations

1. **Message Attribution**: Each message shows sender avatar + name
2. **Agent Responses**: Visually distinct (different color, "AI" badge)
3. **Addressed To**: Show "→ Bob" when agent responds to a specific user
4. **Thread Collapse**: Threads can be expanded/collapsed
5. **Mention Autocomplete**: `@` triggers user/agent picker
6. **Typing Indicators**: Show who is currently typing
7. **Agent Thinking**: Show "RAG Assistant is searching..." during generation

---

## Authorization Model (RBAC + ABAC)

### Core Concepts

| Concept | Description |
|---------|-------------|
| **User** | An authenticated identity (person or service account) |
| **Tenant** | Top-level isolation boundary (Enterprise tier) |
| **Workspace** | A collaborative space within a tenant (Teams+ tier) |
| **Role** | A named collection of permissions assigned to users |
| **Permission** | An action that can be performed on a resource type |
| **Policy** | A rule that grants or denies permissions based on context |
| **Group** | A collection of users that share role assignments |

### Tenant Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         PLATFORM                                 │
│  (Agentic RAG SaaS - all tenants, billing, platform admins)     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    TENANT (Acme Corp)                      │  │
│  │  • Enterprise tier license                                 │  │
│  │  • SSO config (Azure AD)                                   │  │
│  │  • Tenant policies (retention, DLP)                        │  │
│  │  • Usage quotas                                            │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │                                                            │  │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │  │
│  │  │   WORKSPACE     │  │   WORKSPACE     │                  │  │
│  │  │  (Engineering)  │  │  (Marketing)    │                  │  │
│  │  │                 │  │                 │                  │  │
│  │  │  • Members      │  │  • Members      │                  │  │
│  │  │  • Documents    │  │  • Documents    │                  │  │
│  │  │  • Chats        │  │  • Chats        │                  │  │
│  │  │  • Integrations │  │  • Integrations │                  │  │
│  │  └─────────────────┘  └─────────────────┘                  │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              TENANT (Personal - Jane Doe)                  │  │
│  │  • Personal Sync tier license                              │  │
│  │  • Single-user (implicit workspace)                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Roles Definition

### Platform Roles (Agentic RAG Staff)

| Role | Description | Permissions |
|------|-------------|-------------|
| `platform_superadmin` | Full platform access | All operations, all tenants, billing, infrastructure |
| `platform_admin` | Platform operations | View all tenants, manage support tickets, impersonate users |
| `platform_support` | Customer support | View user issues, read-only tenant data, no billing access |
| `platform_billing` | Billing operations | Manage subscriptions, invoices, refunds |

### Tenant Roles (Enterprise Tier)

| Role | Description | Permissions |
|------|-------------|-------------|
| `tenant_owner` | Tenant owner | All tenant operations, billing, SSO config, delete tenant |
| `tenant_admin` | Tenant administrator | Manage workspaces, users, policies, integrations |
| `tenant_auditor` | Compliance auditor | Read-only access to audit logs, usage reports |
| `tenant_member` | Default role | Access assigned workspaces only |

### Workspace Roles (Teams+ Tier)

| Role | Description | Permissions |
|------|-------------|-------------|
| `workspace_owner` | Workspace creator | All workspace operations, transfer ownership |
| `workspace_admin` | Workspace administrator | Manage members, roles, integrations, settings |
| `workspace_editor` | Content creator | Create/edit/delete documents, full chat access |
| `workspace_viewer` | Read-only access | View documents, read chat history, search |
| `workspace_guest` | External collaborator | Limited access, set per invitation |

---

## Permissions Matrix

### Resource Types

| Resource | Description |
|----------|-------------|
| `documents` | Indexed knowledge base documents |
| `conversations` | Chat sessions with history |
| `messages` | Individual chat messages |
| `workspaces` | Collaborative workspaces |
| `members` | Workspace membership |
| `integrations` | Slack/Teams/Discord connections |
| `settings` | Configuration options |
| `audit_logs` | Activity and security logs |
| `billing` | Subscription and payment info |

### Permission Actions

| Action | Description |
|--------|-------------|
| `create` | Create new resource |
| `read` | View resource |
| `update` | Modify resource |
| `delete` | Remove resource |
| `manage` | Administrative control |
| `share` | Share with others |
| `export` | Download/export data |

### Role-Permission Matrix

```
Permission              │ viewer │ editor │ admin │ owner │
────────────────────────┼────────┼────────┼───────┼───────┤
documents:read          │   ✓    │   ✓    │   ✓   │   ✓   │
documents:create        │   ✗    │   ✓    │   ✓   │   ✓   │
documents:update        │   ✗    │   ✓    │   ✓   │   ✓   │
documents:delete        │   ✗    │   ✗    │   ✓   │   ✓   │
documents:share         │   ✗    │   ✓    │   ✓   │   ✓   │
documents:export        │   ✓    │   ✓    │   ✓   │   ✓   │
────────────────────────┼────────┼────────┼───────┼───────┤
conversations:read      │   ✓    │   ✓    │   ✓   │   ✓   │
conversations:create    │   ✗    │   ✓    │   ✓   │   ✓   │
conversations:delete    │   ✗    │   ✗    │   ✓   │   ✓   │
────────────────────────┼────────┼────────┼───────┼───────┤
members:read            │   ✓    │   ✓    │   ✓   │   ✓   │
members:invite          │   ✗    │   ✗    │   ✓   │   ✓   │
members:remove          │   ✗    │   ✗    │   ✓   │   ✓   │
members:manage_roles    │   ✗    │   ✗    │   ✓   │   ✓   │
────────────────────────┼────────┼────────┼───────┼───────┤
integrations:read       │   ✓    │   ✓    │   ✓   │   ✓   │
integrations:manage     │   ✗    │   ✗    │   ✓   │   ✓   │
────────────────────────┼────────┼────────┼───────┼───────┤
settings:read           │   ✓    │   ✓    │   ✓   │   ✓   │
settings:update         │   ✗    │   ✗    │   ✓   │   ✓   │
────────────────────────┼────────┼────────┼───────┼───────┤
audit_logs:read         │   ✗    │   ✗    │   ✓   │   ✓   │
────────────────────────┼────────┼────────┼───────┼───────┤
billing:read            │   ✗    │   ✗    │   ✗   │   ✓   │
billing:manage          │   ✗    │   ✗    │   ✗   │   ✓   │
────────────────────────┼────────┼────────┼───────┼───────┤
workspace:delete        │   ✗    │   ✗    │   ✗   │   ✓   │
workspace:transfer      │   ✗    │   ✗    │   ✗   │   ✓   │
```

---

## Groups

Groups allow batch assignment of roles to multiple users.

### System Groups (Auto-managed)

| Group | Description |
|-------|-------------|
| `all_users` | Every authenticated user in the tenant |
| `workspace_members` | All members of a specific workspace (dynamic) |

### Custom Groups (User-defined)

| Example Group | Use Case |
|---------------|----------|
| `engineering_leads` | Engineering team leads with admin access to engineering workspaces |
| `external_contractors` | Contractors with time-limited guest access |
| `compliance_team` | Users who can view audit logs across workspaces |

---

## Policies

Policies enable fine-grained, context-aware access control (ABAC).

### Policy Types

| Policy Type | Description | Example |
|-------------|-------------|---------|
| **Tier Policy** | Feature availability by license tier | "Sync requires Personal+ tier" |
| **Workspace Policy** | Workspace-level configuration | "Only editors can upload files" |
| **Document Policy** | Document-level access | "Confidential docs require editor role" |
| **IP Policy** | Network-based restrictions | "API only from corporate VPN" |
| **Time Policy** | Temporal access control | "Guest access expires after 30 days" |
| **DLP Policy** | Data loss prevention | "Block PII in Slack messages" |

### Policy Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Authorization Request                        │
│  User: jane@acme.com                                             │
│  Action: documents:delete                                        │
│  Resource: doc_123 (workspace: engineering)                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1. License Check                              │
│  Is tenant license valid? Is tier sufficient for action?        │
│  ❌ DENY if license expired or tier insufficient                │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ✓ Pass
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. Tenant Policies                             │
│  Any tenant-level blocks? (IP restrictions, suspended users)    │
│  ❌ DENY if tenant policy blocks                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ✓ Pass
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  3. Workspace Membership                         │
│  Is user a member of "engineering" workspace?                   │
│  ❌ DENY if not a member                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ✓ Pass
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     4. Role Check                                │
│  Does user's role include "documents:delete" permission?        │
│  ❌ DENY if permission not granted                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ✓ Pass
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   5. Resource Policies                           │
│  Any document-specific restrictions? (e.g., locked, archived)   │
│  ❌ DENY if resource policy blocks                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ✓ Pass
                            ▼
              ┌─────────────────────────────┐
              │         ✅ ALLOW            │
              │   Action is authorized      │
              └─────────────────────────────┘
```

---

## Database Schema Extensions

### Users & Authentication

```sql
-- Tenant: top-level isolation boundary
CREATE TABLE tenants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  slug TEXT UNIQUE NOT NULL,
  tier TEXT NOT NULL CHECK (tier IN ('personal', 'teams', 'enterprise')),
  license_key TEXT UNIQUE,
  license_valid_until TIMESTAMPTZ,
  sso_provider TEXT,  -- 'azure_ad', 'okta', 'google', NULL
  sso_config JSONB,   -- Provider-specific config
  settings JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users: authenticated identities
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  full_name TEXT,
  avatar_url TEXT,
  auth_provider TEXT NOT NULL,  -- 'email', 'google', 'github', 'azure_ad', 'saml'
  auth_provider_id TEXT,        -- External provider user ID
  is_active BOOLEAN DEFAULT TRUE,
  last_login_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (tenant_id, email)
);

-- Roles: named permission sets
CREATE TABLE roles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  display_name TEXT,
  description TEXT,
  permissions TEXT[] NOT NULL,  -- Array of permission strings
  is_system BOOLEAN DEFAULT FALSE,  -- System roles cannot be deleted
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (tenant_id, name)
);

-- Groups: collections of users
CREATE TABLE groups (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  is_system BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (tenant_id, name)
);

-- Group membership
CREATE TABLE group_members (
  group_id UUID REFERENCES groups(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  added_at TIMESTAMPTZ DEFAULT NOW(),
  added_by UUID REFERENCES users(id),
  PRIMARY KEY (group_id, user_id)
);

-- Workspace membership with roles
CREATE TABLE workspace_members (
  workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  role_id UUID REFERENCES roles(id),
  invited_by UUID REFERENCES users(id),
  joined_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (workspace_id, user_id)
);

-- Policies: fine-grained access rules
CREATE TABLE policies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  resource_type TEXT NOT NULL,  -- 'document', 'workspace', 'integration'
  conditions JSONB NOT NULL,    -- Condition expressions
  effect TEXT NOT NULL CHECK (effect IN ('allow', 'deny')),
  priority INTEGER DEFAULT 0,   -- Higher = evaluated first
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (tenant_id, name)
);

-- Audit log for compliance
CREATE TABLE audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id),
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id UUID,
  details JSONB,
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_tenant_time ON audit_logs(tenant_id, created_at DESC);
CREATE INDEX idx_audit_user ON audit_logs(user_id, created_at DESC);
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);
```

### Default Roles (Seed Data)

```sql
-- Insert default roles for each tier
INSERT INTO roles (tenant_id, name, display_name, permissions, is_system) VALUES
-- Personal tier (single user, all permissions)
(NULL, 'personal_owner', 'Owner', ARRAY[
  'documents:*', 'conversations:*', 'settings:*'
], TRUE),

-- Teams tier workspace roles
(NULL, 'workspace_owner', 'Workspace Owner', ARRAY[
  'documents:*', 'conversations:*', 'members:*', 'integrations:*', 
  'settings:*', 'audit_logs:read', 'billing:*', 'workspace:*'
], TRUE),
(NULL, 'workspace_admin', 'Workspace Admin', ARRAY[
  'documents:*', 'conversations:*', 'members:*', 'integrations:*',
  'settings:*', 'audit_logs:read'
], TRUE),
(NULL, 'workspace_editor', 'Editor', ARRAY[
  'documents:read', 'documents:create', 'documents:update', 'documents:share',
  'conversations:*', 'members:read', 'integrations:read', 'settings:read'
], TRUE),
(NULL, 'workspace_viewer', 'Viewer', ARRAY[
  'documents:read', 'documents:export',
  'conversations:read', 'members:read', 'settings:read'
], TRUE),

-- Enterprise tenant roles
(NULL, 'tenant_owner', 'Tenant Owner', ARRAY[
  '*'  -- All permissions
], TRUE),
(NULL, 'tenant_admin', 'Tenant Admin', ARRAY[
  'workspaces:*', 'users:*', 'groups:*', 'roles:*', 'policies:*',
  'integrations:*', 'settings:*', 'audit_logs:read'
], TRUE),
(NULL, 'tenant_auditor', 'Auditor', ARRAY[
  'audit_logs:read', 'users:read', 'workspaces:read', 'documents:read'
], TRUE);
```

---

## Admin Portal Architecture

### Portal Sections by Role

| Section | Platform Staff | Tenant Owner | Tenant Admin | Workspace Admin | Member |
|---------|----------------|--------------|--------------|-----------------|--------|
| **Dashboard** | Platform metrics | Tenant usage | Workspace stats | Workspace stats | Personal stats |
| **Users** | All users | Tenant users | Workspace members | Workspace members | Profile only |
| **Workspaces** | All tenants | All workspaces | Managed workspaces | Own workspace | Own workspaces |
| **Documents** | N/A | All docs | Workspace docs | Workspace docs | Accessible docs |
| **Integrations** | Platform-wide | Tenant integrations | Workspace integrations | View only | N/A |
| **Billing** | All subscriptions | Tenant billing | N/A | N/A | N/A |
| **Audit Logs** | Platform logs | Tenant logs | Workspace logs | Workspace logs | Own actions |
| **Settings** | Platform config | Tenant settings | Workspace settings | Workspace settings | User prefs |
| **SSO Config** | Platform SSO | Tenant SSO | N/A | N/A | N/A |

### Admin Portal Pages

```
/admin
├── /dashboard              # Overview metrics
├── /users                  # User management
│   ├── /users/[id]         # User detail
│   └── /users/invite       # Invite new users
├── /workspaces             # Workspace management
│   ├── /workspaces/[id]    # Workspace detail
│   └── /workspaces/new     # Create workspace
├── /integrations           # Connected apps
│   ├── /integrations/slack
│   ├── /integrations/teams
│   └── /integrations/discord
├── /billing                # Subscription & invoices
│   ├── /billing/plans
│   ├── /billing/invoices
│   └── /billing/usage
├── /audit                  # Audit logs
├── /settings               # Configuration
│   ├── /settings/general
│   ├── /settings/security
│   └── /settings/sso
└── /profile                # User profile
```

---

## Implementation Phases

### Phase 2A: Authentication & Basic Authorization (2-3 weeks)

1. **Supabase Auth Integration**
   - Email/password + magic link
   - Google OAuth, GitHub OAuth
   - JWT token handling

2. **Basic RBAC**
   - Implement roles table and seed data
   - Permission checking middleware
   - Protected API endpoints

3. **User Portal MVP**
   - Login/signup pages
   - User profile page
   - Basic dashboard

### Phase 2B: Teams & Workspaces (2-3 weeks)

1. **Workspace Management**
   - Create/delete workspaces
   - Member invitations
   - Role assignments

2. **Row-Level Security**
   - RLS policies for all tables
   - Workspace-scoped queries

3. **Admin Portal**
   - Workspace settings
   - Member management
   - Integration connections

### Phase 2C: Enterprise Features (3-4 weeks)

1. **Multi-tenancy**
   - Tenant provisioning
   - License validation
   - Usage quotas

2. **SSO Integration**
   - Azure AD OAuth
   - SAML 2.0 support
   - OIDC configuration

3. **Audit & Compliance**
   - Audit log capture
   - Retention policies
   - Data export

### Phase 2D: Billing & Licensing (2 weeks)

1. **Stripe Integration**
   - Subscription management
   - Usage-based billing
   - Invoice generation

2. **License Enforcement**
   - Tier feature gates
   - Quota enforcement
   - Grace period handling

---

## Security Considerations

### SOC 2 Controls

| Control | Implementation |
|---------|----------------|
| Access Control | RBAC + RLS + policies |
| Authentication | MFA, SSO, password policies |
| Audit Logging | All sensitive actions logged |
| Data Encryption | TLS in transit, AES at rest |
| Session Management | JWT rotation, secure cookies |

### GDPR Compliance

| Requirement | Implementation |
|-------------|----------------|
| Data Export | `/api/users/{id}/export` endpoint |
| Right to Erasure | `/api/users/{id}/delete` cascade |
| Consent Management | Explicit consent for data processing |
| Data Minimization | Only collect necessary data |
| Processing Records | Audit logs serve as processing records |

---

## Next Steps

1. [ ] Review and approve architecture
2. [ ] Create database migrations for auth tables
3. [ ] Implement Supabase Auth integration
4. [ ] Build permission middleware
5. [ ] Create admin portal UI scaffold
6. [ ] Define Stripe products and pricing
