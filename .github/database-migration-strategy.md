# Database Migration & Scaling Strategy

**Version:** 1.1  
**Date:** December 13, 2025  
**Status:** Planning Phase  
**Update:** Added messaging platform integrations (Slack, Teams, Discord)

## Executive Summary

This document outlines the migration path from JSONL-based document storage to a production-ready, globally-replicated database architecture supporting:
- **Power users** (local desktop/laptop)
- **Small teams** (collaborative workspaces)
- **Enterprise multi-tenant** (global replication)
- **Messaging platform integrations** (Slack, Microsoft Teams, Discord with two-way sync)
- **Mobile access** (iOS/Android with offline sync)

## Current State

### Existing Architecture
- **Document Store:** JSONL file at `cache/rag_store.jsonl`
- **Vector Index:** FAISS in-memory index
- **Chat History:** SQLite at `cache/chat_history.db`
- **Concurrency:** File-based locking
- **Replication:** None (single-instance only)
- **Integrations:** MCP server for AI assistants

### Limitations
1. No multi-user collaboration
2. No mobile/remote access
3. No real-time synchronization
4. Limited scalability (file I/O bottleneck)
5. No tenant isolation for enterprise use
6. No time-series analytics storage
7. No messaging platform integrations

## Target Architecture

### Technology Stack: PostgreSQL + Supabase + Message Brokers

**Core Database:** PostgreSQL 15+ with pgvector extension  
**Sync Platform:** Supabase (self-hosted or cloud)  
**Message Queue:** Redis Streams (lightweight) or RabbitMQ (enterprise)  
**Integration Hub:** FastAPI webhooks + bot frameworks  
**Mobile SDKs:** Supabase JS/Flutter clients  
**Real-time:** PostgreSQL logical replication + WebSockets

### Why This Stack

**PostgreSQL + pgvector:**
- ✅ Native vector search (replaces FAISS)
- ✅ ACID transactions for data integrity
- ✅ Mature replication (streaming, logical, physical)
- ✅ SOC 2 compliant with proper configuration
- ✅ Excellent Python ecosystem (SQLAlchemy, asyncpg)

**Supabase:**
- ✅ Real-time sync out of the box
- ✅ Multi-tenancy via Row-Level Security (RLS)
- ✅ Built-in authentication & RBAC
- ✅ Mobile SDKs for iOS/Android
- ✅ Edge functions for global deployment
- ✅ Self-hostable (no vendor lock-in)
- ✅ Already SOC 2 Type II certified

**Redis Streams / RabbitMQ:**
- ✅ Async message processing (webhooks, bot events)
- ✅ Guaranteed delivery (webhook retries)
- ✅ Message persistence (replay events)
- ✅ Pub/sub for real-time notifications

### Deployment Tiers

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Power User (Local Desktop/Laptop)                   │
│  • Local PostgreSQL OR local Supabase instance              │
│  • Sync to cloud portal on-demand                           │
│  • Works offline, syncs when online                         │
└─────────────────────────────────────────────────────────────┘
                         ▼ Real-time Sync
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Cloud Portal / Dashboard                            │
│  • Supabase Cloud (managed PostgreSQL + real-time)         │
│  • Multi-region read replicas                               │
│  • Mobile apps connect here                                 │
│  • Webhook receiver for Slack/Teams/Discord                 │
└─────────────────────────────────────────────────────────────┘
                ▼ Team Workspace          ▼ Bot Integrations
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Small Teams                                         │
│  • Shared workspace_id (Row-Level Security filter)         │
│  • Real-time collaboration                                  │
│  • Role-based permissions (admin/editor/viewer)            │
│  • Slack channels synced to workspaces                     │
│  • @bot mentions trigger RAG queries                        │
└─────────────────────────────────────────────────────────────┘
                         ▼ Global Replication
┌─────────────────────────────────────────────────────────────┐
│ Tier 4: Enterprise Multi-Tenant                             │
│  • tenant_id column on all tables                           │
│  • Regional clusters (US-East, EU-West, APAC)              │
│  • Cross-region replication                                 │
│  • Dedicated Slack/Teams workspaces per tenant             │
│  • Enterprise Grid / Microsoft 365 integrations            │
└─────────────────────────────────────────────────────────────┘
```

### Messaging Platform Integration Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     User Interactions                         │
├──────────────────────────────────────────────────────────────┤
│  Desktop App    Mobile App    Slack    Teams    Discord      │
│      ▼              ▼           ▼        ▼         ▼         │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Integration Hub (FastAPI)                  │
├──────────────────────────────────────────────────────────────┤
│  • Webhook receivers (POST /webhooks/slack, /teams, /discord)│
│  • Event verification (HMAC signatures)                       │
│  • Rate limiting per tenant                                   │
│  • Message queue producer                                     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              Message Queue (Redis Streams)                    │
├──────────────────────────────────────────────────────────────┤
│  • slack_events stream                                        │
│  • teams_events stream                                        │
│  • discord_events stream                                      │
│  • outbound_messages stream (responses to send)              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│            Event Processors (Async Workers)                   │
├──────────────────────────────────────────────────────────────┤
│  • Parse message (extract query, mentions, attachments)      │
│  • Check permissions (user → workspace mapping)              │
│  • Execute RAG query (search + grounded answer)              │
│  • Format response (platform-specific markdown)              │
│  • Queue outbound message                                     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  PostgreSQL + Supabase                        │
├──────────────────────────────────────────────────────────────┤
│  • Documents (indexed knowledge base)                         │
│  • Conversations (chat history with platform_id)             │
│  • Integration_mappings (Slack channel ↔ workspace)          │
│  • Bot_configurations (per-tenant bot tokens)                │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│            Response Sender (Outbound Workers)                 │
├──────────────────────────────────────────────────────────────┤
│  • Consume outbound_messages stream                           │
│  • Call Slack API (chat.postMessage)                         │
│  • Call Teams API (send activity)                            │
│  • Call Discord API (create message)                         │
│  • Handle retries + failures                                  │
│  • Update message status in DB                                │
└──────────────────────────────────────────────────────────────┘
```

## Database Schema Evolution

### Phase 1: Power User - Local PostgreSQL

```sql
-- Core document storage
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  uri TEXT UNIQUE NOT NULL,
  text TEXT NOT NULL,
  embedding VECTOR(384),  -- Matches arctic-embed-xs dimensions
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_documents_embedding ON documents 
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

CREATE INDEX idx_documents_uri ON documents(uri);
CREATE INDEX idx_documents_updated_at ON documents(updated_at DESC);

-- Time-series performance metrics
CREATE TABLE performance_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  operation TEXT NOT NULL,  -- 'index', 'search', 'chat', 'embed'
  duration_ms INTEGER NOT NULL,
  token_count INTEGER,
  model TEXT,
  error TEXT,
  metadata JSONB
);

CREATE INDEX idx_metrics_timestamp ON performance_metrics(timestamp DESC);
CREATE INDEX idx_metrics_operation ON performance_metrics(operation);

-- Migrate existing chat history
ALTER TABLE sessions ADD COLUMN workspace_id UUID;
ALTER TABLE sessions ADD COLUMN created_by UUID;
ALTER TABLE messages ADD COLUMN token_count INTEGER;
ALTER TABLE messages ADD COLUMN latency_ms INTEGER;
```

### Phase 2: Teams - Workspace & Permissions

```sql
-- User accounts
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  full_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Workspaces (teams)
CREATE TABLE workspaces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  slug TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  owner_id UUID REFERENCES users(id)
);

-- Workspace membership & roles
CREATE TABLE workspace_members (
  workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
  joined_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (workspace_id, user_id)
);

-- Add workspace context to existing tables
ALTER TABLE documents ADD COLUMN workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE documents ADD COLUMN created_by UUID REFERENCES users(id);
ALTER TABLE sessions ADD COLUMN workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE sessions ADD COLUMN created_by UUID REFERENCES users(id);

-- Row-Level Security (RLS) policies
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users see workspace docs" ON documents
  FOR SELECT USING (
    workspace_id IN (
      SELECT workspace_id FROM workspace_members 
      WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "Editors can insert docs" ON documents
  FOR INSERT WITH CHECK (
    workspace_id IN (
      SELECT workspace_id FROM workspace_members 
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
    )
  );

CREATE POLICY "Admins can delete docs" ON documents
  FOR DELETE USING (
    workspace_id IN (
      SELECT workspace_id FROM workspace_members 
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );
```

### Phase 2b: Messaging Platform Integrations

```sql
-- Platform bot configurations (per workspace)
CREATE TABLE bot_configurations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
  platform TEXT NOT NULL CHECK (platform IN ('slack', 'teams', 'discord')),
  bot_token TEXT NOT NULL,  -- Encrypted bot token
  bot_user_id TEXT,  -- Platform's bot user ID
  webhook_url TEXT,  -- For outbound webhooks
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (workspace_id, platform)
);

-- Channel/thread mappings (Slack channel ↔ workspace)
CREATE TABLE integration_mappings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
  platform TEXT NOT NULL,
  platform_team_id TEXT,  -- Slack team ID, Teams tenant ID, Discord guild ID
  platform_channel_id TEXT NOT NULL,  -- Channel/conversation ID
  platform_thread_id TEXT,  -- Optional thread ID
  sync_direction TEXT NOT NULL CHECK (sync_direction IN ('inbound', 'outbound', 'bidirectional')),
  auto_respond BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (platform, platform_channel_id, platform_thread_id)
);

-- External user mappings (Slack user ↔ internal user)
CREATE TABLE external_user_mappings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  platform TEXT NOT NULL,
  platform_user_id TEXT NOT NULL,
  platform_username TEXT,
  platform_email TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (platform, platform_user_id)
);

-- Conversation history with platform context
CREATE TABLE platform_conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID REFERENCES workspaces(id),
  session_id UUID REFERENCES sessions(id),
  platform TEXT NOT NULL,
  platform_message_id TEXT UNIQUE NOT NULL,
  platform_user_id TEXT NOT NULL,
  platform_channel_id TEXT NOT NULL,
  platform_thread_id TEXT,
  direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
  content TEXT NOT NULL,
  metadata JSONB,  -- Reactions, attachments, etc.
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_platform_conv_workspace ON platform_conversations(workspace_id);
CREATE INDEX idx_platform_conv_channel ON platform_conversations(platform, platform_channel_id);
CREATE INDEX idx_platform_conv_thread ON platform_conversations(platform, platform_thread_id);
CREATE INDEX idx_platform_conv_created ON platform_conversations(created_at DESC);

-- Message queue (for async processing)
CREATE TABLE message_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  queue_name TEXT NOT NULL,  -- 'slack_events', 'outbound_messages', etc.
  payload JSONB NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  retry_count INTEGER DEFAULT 0,
  max_retries INTEGER DEFAULT 3,
  error TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  processed_at TIMESTAMPTZ
);

CREATE INDEX idx_queue_status ON message_queue(queue_name, status, created_at);
```

### Phase 3: Enterprise - Multi-Tenant & Global Replication

```sql
-- Tenant isolation
CREATE TABLE tenants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  slug TEXT UNIQUE NOT NULL,
  plan TEXT NOT NULL CHECK (plan IN ('free', 'team', 'enterprise')),
  max_users INTEGER,
  max_documents INTEGER,
  region TEXT,  -- 'us-east-1', 'eu-west-1', 'ap-southeast-1'
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add tenant_id to all tables
ALTER TABLE users ADD COLUMN tenant_id UUID REFERENCES tenants(id);
ALTER TABLE workspaces ADD COLUMN tenant_id UUID REFERENCES tenants(id);
ALTER TABLE documents ADD COLUMN tenant_id UUID REFERENCES tenants(id);
ALTER TABLE bot_configurations ADD COLUMN tenant_id UUID REFERENCES tenants(id);

-- Enterprise-grade bot configurations
ALTER TABLE bot_configurations ADD COLUMN sso_enabled BOOLEAN DEFAULT FALSE;
ALTER TABLE bot_configurations ADD COLUMN sso_provider TEXT;  -- 'okta', 'azure_ad'
ALTER TABLE bot_configurations ADD COLUMN audit_logging BOOLEAN DEFAULT TRUE;
ALTER TABLE bot_configurations ADD COLUMN data_retention_days INTEGER DEFAULT 90;

-- Slack Enterprise Grid / Microsoft 365 specific
CREATE TABLE enterprise_org_mappings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id),
  platform TEXT NOT NULL,
  org_id TEXT NOT NULL,  -- Slack Enterprise Grid org ID, M365 tenant ID
  org_name TEXT,
  is_verified BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (platform, org_id)
);

-- Partitioning for large-scale deployments
CREATE TABLE documents_timeseries (
  id UUID DEFAULT gen_random_uuid(),
  uri TEXT NOT NULL,
  text TEXT NOT NULL,
  embedding VECTOR(384),
  tenant_id UUID NOT NULL,
  workspace_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE documents_2025_12 PARTITION OF documents_timeseries
  FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Regional partitioning (for global replication)
CREATE TABLE documents_us PARTITION OF documents 
  FOR VALUES IN ('us-east-1') 
  TABLESPACE us_east_tablespace;

CREATE TABLE documents_eu PARTITION OF documents 
  FOR VALUES IN ('eu-west-1') 
  TABLESPACE eu_west_tablespace;

-- Usage metering for billing
CREATE TABLE usage_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id),
  metric_type TEXT NOT NULL,  -- 'api_calls', 'documents_indexed', 'tokens_used', 'slack_messages'
  value BIGINT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

CREATE INDEX idx_usage_tenant_time ON usage_metrics(tenant_id, timestamp DESC);
```

## Implementation Roadmap

### Phase 1: Power User (Local PostgreSQL) - 2-3 weeks

**Goal:** Replace JSONL with PostgreSQL + pgvector, maintain offline-first capability

#### Tasks
- [ ] **Database Setup**
  - Install PostgreSQL 15+ with pgvector extension
  - Create SQLAlchemy models matching new schema
  - Write migration script (JSONL → PostgreSQL)

- [ ] **Core Module Refactoring**
  - Replace `src/core/store.py` with `src/core/db/repositories/document_repo.py`
  - Migrate FAISS operations to pgvector queries
  - Update `src/core/rag_core.py` to use PostgreSQL backend
  - Add connection pooling (SQLAlchemy async engine)

- [ ] **Performance Instrumentation**
  - Implement `performance_metrics` table inserts
  - Add middleware to track operation latencies
  - Create dashboard endpoint for metrics visualization

- [ ] **Testing & Validation**
  - Unit tests for new repository layer
  - Integration tests for vector search accuracy
  - Performance benchmarks vs. JSONL baseline
  - Migration validation (data integrity checks)

#### Deliverables
```python
# New file structure
src/core/db/
├── __init__.py
├── connection.py       # Database connection pooling
├── models.py           # SQLAlchemy ORM models
├── repositories/
│   ├── __init__.py
│   ├── document_repo.py
│   ├── chat_repo.py
│   └── metrics_repo.py
└── migrations/
    └── 001_jsonl_to_postgres.py
```

#### Success Criteria
- [ ] All existing tests pass with PostgreSQL backend
- [ ] Vector search latency < 100ms for 10k documents
- [ ] Zero data loss during migration
- [ ] Backward compatibility (can still read old JSONL for recovery)

---

### Phase 2: Cloud Portal + Mobile + Messaging Integrations - 6-8 weeks

**Goal:** Deploy cloud dashboard, enable mobile access, integrate Slack/Teams/Discord

#### Tasks
- [ ] **Supabase Deployment**
  - Deploy Supabase Cloud instance (or self-hosted)
  - Configure authentication (email + OAuth providers)
  - Set up real-time subscriptions for chat messages
  - Create API keys and connection strings

- [ ] **Sync Engine**
  - Design conflict resolution strategy (last-write-wins vs. CRDT)
  - Implement delta sync protocol (only changed documents)
  - Add sync status tracking (pending, synced, conflict)
  - Handle offline queue (operations applied when reconnected)

- [ ] **REST API Enhancements**
  - Add `/api/sync/push` endpoint (desktop → cloud)
  - Add `/api/sync/pull` endpoint (cloud → desktop)
  - Add WebSocket endpoint for real-time updates
  - Implement authentication middleware

- [ ] **Slack Integration**
  - Create Slack app manifest (OAuth scopes: `chat:write`, `commands`, `app_mentions:read`)
  - Implement webhook receiver: `POST /webhooks/slack`
  - Event verification (HMAC signature validation)
  - Slash commands: `/rag search <query>`, `/rag index <url>`
  - App mentions: `@ragbot what is...?`
  - Interactive components (buttons for citations, "Ask follow-up")
  - Thread support (keep conversation context)

- [ ] **Microsoft Teams Integration**
  - Create Teams app package (Bot Framework SDK)
  - Implement webhook receiver: `POST /webhooks/teams`
  - Activity Handler (message, mention, conversation update)
  - Adaptive Cards for rich responses (citations, thumbnails)
  - Channel messages + 1:1 chats
  - File upload support (index directly from Teams)

- [ ] **Discord Integration**
  - Create Discord bot (discord.py or Interactions API)
  - Implement webhook receiver: `POST /webhooks/discord`
  - Slash commands: `/rag search`, `/rag help`
  - Message commands (right-click → Apps → "Index this message")
  - Embed messages for responses (rich formatting)
  - Channel permissions (bot only responds in allowed channels)

- [ ] **Message Queue Setup**
  - Deploy Redis (or Redis Cloud for managed)
  - Implement event producers (webhook → queue)
  - Implement event consumers (queue → RAG query → response sender)
  - Add monitoring (queue depth, processing latency)
  - Dead letter queue for failed messages

- [ ] **Mobile App Development**
  - Migrate `mobile/App.js` to TypeScript
  - Integrate Supabase mobile SDK
  - Implement offline-first storage (AsyncStorage + SQLite)
  - Add real-time chat subscriptions
  - Build file manager UI (view/search documents)

#### Deliverables
```python
# Backend additions
src/core/sync/
├── __init__.py
├── engine.py           # Bidirectional sync orchestration
├── conflict_resolver.py
└── delta_calculator.py

src/servers/websocket_server.py  # Real-time updates

src/integrations/
├── __init__.py
├── base.py             # Base integration class
├── slack/
│   ├── __init__.py
│   ├── bot.py          # Slack Bolt app
│   ├── handlers.py     # Event handlers
│   ├── formatters.py   # Message formatting
│   └── auth.py         # OAuth flow
├── teams/
│   ├── __init__.py
│   ├── bot.py          # Bot Framework adapter
│   ├── handlers.py     # Activity handlers
│   └── cards.py        # Adaptive Cards
└── discord/
    ├── __init__.py
    ├── bot.py          # Discord client
    ├── commands.py     # Slash commands
    └── embeds.py       # Rich embeds

src/workers/
├── __init__.py
├── event_processor.py  # Async worker (consume events)
└── response_sender.py  # Async worker (send responses)
```

```typescript
// Mobile app structure
mobile/src/
├── lib/
│   ├── supabase.ts     // Supabase client
│   └── sync.ts         // Offline sync logic
├── hooks/
│   ├── useRealtimeChat.ts
│   └── useDocuments.ts
└── screens/
    ├── ChatScreen.tsx
    ├── SearchScreen.tsx
    └── SettingsScreen.tsx
```

#### Success Criteria
- [ ] Mobile app can chat/search while offline
- [ ] Changes sync within 5 seconds when online
- [ ] Conflict resolution preserves both versions (no data loss)
- [ ] Real-time updates appear instantly for connected clients
- [ ] Slack bot responds within 3 seconds to mentions
- [ ] Teams bot handles file uploads and indexes content
- [ ] Discord bot supports slash commands and thread tracking
- [ ] Message queue processing < 1 second per event

---

### Phase 3: Teams - Collaboration & Permissions - 4-6 weeks

**Goal:** Enable team workspaces with role-based access control and platform mapping

#### Tasks
- [ ] **Schema Migration**
  - Deploy Phase 2b schema (bot configs, integration mappings)
  - Implement Row-Level Security policies for platforms
  - Add platform user mapping (Slack user → internal user)

- [ ] **Authentication & Authorization**
  - Integrate Supabase Auth (or custom JWT)
  - Implement RBAC middleware (check roles on API calls)
  - Add invitation flow (invite users to workspace)
  - Platform-aware auth (verify Slack user is in workspace)

- [ ] **Collaboration Features**
  - Real-time presence indicators ("User X is viewing document Y")
  - Shared conversation threads (multiple users in same chat)
  - Document access logs (audit trail)
  - @mention notifications across platforms
  - Cross-platform threading (Slack thread ↔ Teams conversation)

- [ ] **Platform Admin Features**
  - Workspace settings: connect Slack workspace
  - Channel allowlist (which channels bot can access)
  - Auto-index from Slack attachments
  - Notification preferences (DM vs. channel reply)
  - Bot analytics (queries per channel, top users)

- [ ] **UI Enhancements**
  - Workspace switcher dropdown
  - Team member management panel
  - Permission editor (assign roles)
  - Activity feed (recent changes by team members)
  - Platform integration panel (connect/disconnect Slack/Teams/Discord)

#### Deliverables
```python
# Backend additions
src/core/auth/
├── __init__.py
├── middleware.py       # JWT validation, role checks
└── policies.py         # RLS policy helpers

src/core/collaboration/
├── __init__.py
├── presence.py         # Real-time presence tracking
└── notifications.py    # Push notifications

src/integrations/admin/
├── __init__.py
├── workspace_setup.py  # OAuth flow for workspace connection
└── channel_manager.py  # Channel permissions management
```

```typescript
// UI additions
ui/src/features/
├── workspaces/
│   ├── WorkspaceSwitcher.tsx
│   ├── WorkspaceSettings.tsx
│   └── MemberManager.tsx
├── collaboration/
│   ├── PresenceIndicator.tsx
│   └── ActivityFeed.tsx
└── integrations/
    ├── IntegrationPanel.tsx
    ├── SlackConnect.tsx
    ├── TeamsConnect.tsx
    └── DiscordConnect.tsx
```

#### Success Criteria
- [ ] Users can create/join/leave workspaces
- [ ] RLS policies prevent cross-workspace data leaks
- [ ] Role permissions enforced at API + database level
- [ ] Real-time updates work for all team members
- [ ] Slack workspace OAuth flow completes successfully
- [ ] Bot only responds in allowed channels
- [ ] Cross-platform conversations linked correctly

---

### Phase 4: Enterprise Multi-Tenant - 8-12 weeks

**Goal:** Support thousands of tenants with global replication and enterprise integrations

#### Tasks
- [ ] **Tenant Isolation**
  - Implement tenant_id filtering across all queries
  - Add tenant provisioning API
  - Implement resource quotas (max users/documents per tenant)
  - Tenant-specific bot configurations (separate Slack apps per tenant)

- [ ] **Multi-Region Deployment**
  - Deploy regional PostgreSQL clusters (US, EU, APAC)
  - Configure logical replication between regions
  - Implement geo-routing (nearest region selection)
  - Add conflict resolution for multi-master writes

- [ ] **Enterprise Platform Features**
  - **Slack Enterprise Grid:**
    - Org-level bot installation
    - Discovery API (auto-detect workspaces)
    - Admin analytics (usage across all workspaces)
  - **Microsoft Teams / Microsoft 365:**
    - Microsoft Graph API integration
    - SharePoint indexing (auto-index team files)
    - Azure AD SSO
  - **Discord:**
    - Server templates for easy setup
    - Role-based channel access

- [ ] **Usage Metering & Billing**
  - Track API calls, documents indexed, tokens used, messages sent
  - Implement rate limiting per tenant plan
  - Add billing integration (Stripe/Chargebee)
  - Platform-specific metering (Slack message credits)

- [ ] **Enterprise Features**
  - SSO integration (SAML, OIDC)
  - Advanced audit logs (all actions + data changes)
  - Custom domain support (tenant.yourdomain.com)
  - SLA monitoring (uptime, latency)
  - DLP policies (block sensitive data in platform messages)

- [ ] **SOC 2 Compliance**
  - Data encryption at rest (PostgreSQL TDE)
  - TLS 1.3 for all connections
  - Automated backup + disaster recovery
  - Security incident response plan
  - Platform token encryption (HSM or AWS KMS)

#### Deliverables
```python
# Backend additions
src/core/tenancy/
├── __init__.py
├── provisioning.py     # Tenant creation/deletion
├── quotas.py           # Resource limits enforcement
└── isolation.py        # Tenant context manager

src/core/replication/
├── __init__.py
├── conflict_resolver.py
└── geo_router.py       # Route to nearest region

src/core/billing/
├── __init__.py
├── metering.py         # Usage tracking
└── stripe_integration.py

src/integrations/enterprise/
├── __init__.py
├── slack_grid.py       # Slack Enterprise Grid
├── microsoft_graph.py  # Microsoft 365 integration
└── sso.py              # SAML/OIDC providers
```

#### Success Criteria
- [ ] Support 1000+ tenants without performance degradation
- [ ] Cross-region replication lag < 1 second
- [ ] 99.9% uptime SLA
- [ ] SOC 2 Type II audit ready
- [ ] Slack Enterprise Grid org-level deployment works
- [ ] Microsoft Graph API indexes SharePoint files
- [ ] Platform tokens stored encrypted with rotation policy

---

## Messaging Platform Integration Details

### Slack Integration

**Bot Capabilities:**
- **Slash Commands:**
  - `/rag search <query>` - Search indexed documents
  - `/rag index <url>` - Index a new document
  - `/rag help` - Show available commands
  - `/rag status` - Show bot status and indexed doc count

- **App Mentions:**
  - `@ragbot what is RAG?` - Natural language queries
  - `@ragbot summarize https://example.com` - Summarize URL

- **Interactive Components:**
  - Buttons: "👍 Helpful", "👎 Not helpful", "🔄 Ask follow-up"
  - Select menus: Choose from top 5 matching documents
  - Modals: Configure bot settings per channel

- **Events to Subscribe:**
  - `app_mention` - Bot mentions
  - `message.channels` - Channel messages (if auto-respond enabled)
  - `file_shared` - Auto-index shared files (PDFs, docs)
  - `message.im` - Direct messages to bot

**OAuth Scopes Required:**
```
chat:write          # Send messages
commands            # Slash commands
app_mentions:read   # Read mentions
channels:history    # Read channel messages
files:read          # Read shared files
users:read          # Get user info
team:read           # Get workspace info
```

**Setup Flow:**
1. Admin clicks "Add to Slack" button in your portal
2. OAuth redirect to Slack with required scopes
3. User approves installation for their workspace
4. Callback receives `bot_token` and `team_id`
5. Store in `bot_configurations` table (encrypted)
6. Test connection with `auth.test` API call

### Microsoft Teams Integration

**Bot Capabilities:**
- **Commands:**
  - `@RAGBot search <query>`
  - `@RAGBot index <url>`
  - Invoke name: "RAG Assistant"

- **Adaptive Cards:**
  - Rich responses with citations
  - Thumbnail previews of documents
  - Action buttons (Save to favorites, Share, Report issue)

- **File Uploads:**
  - Upload PDF/DOCX directly in chat → auto-index
  - Real-time indexing status (progress bar in card)

- **Channel/Chat Support:**
  - Responds in channels (if mentioned or configured)
  - 1:1 chats with users
  - Group chats

**Bot Framework Configuration:**
```yaml
# manifest.json
{
  "manifestVersion": "1.13",
  "id": "<your-bot-id>",
  "name": {
    "short": "RAG Assistant",
    "full": "Agentic RAG Document Search Bot"
  },
  "description": {
    "short": "Search your team's knowledge base",
    "full": "AI-powered document search and retrieval using RAG"
  },
  "bots": [{
    "botId": "<your-bot-id>",
    "scopes": ["team", "personal", "groupchat"],
    "supportsFiles": true,
    "isNotificationOnly": false,
    "commandLists": [{
      "scopes": ["team", "personal"],
      "commands": [
        {
          "title": "search",
          "description": "Search indexed documents"
        },
        {
          "title": "index",
          "description": "Index a new URL"
        }
      ]
    }]
  }],
  "permissions": ["identity", "messageTeamMembers"]
}
```

**Setup Flow:**
1. Create Bot Framework bot in Azure Portal
2. Get `MicrosoftAppId` and `MicrosoftAppPassword`
3. Upload Teams app manifest to App Studio
4. Install app in team (side-loaded or App Source)
5. Store credentials in `bot_configurations` table

### Discord Integration

**Bot Capabilities:**
- **Slash Commands:**
  - `/rag search query:<text>` - Search documents
  - `/rag index url:<url>` - Index URL
  - `/rag help` - Show help

- **Message Commands:**
  - Right-click message → Apps → "Index this message"
  - Right-click message → Apps → "Summarize this"

- **Embeds:**
  - Rich embeds with color-coded results
  - Thumbnail images for documents
  - Footer with metadata (source, relevance score)

- **Permissions:**
  - `Send Messages`
  - `Embed Links`
  - `Attach Files`
  - `Read Message History`
  - `Use Slash Commands`

**OAuth Scopes:**
```
bot              # Bot user
applications.commands  # Slash commands
```

**Setup Flow:**
1. Create Discord application in Developer Portal
2. Create bot user, get token
3. Generate OAuth URL with required scopes
4. Admin clicks URL, authorizes bot for their server
5. Register slash commands via Discord API
6. Store bot token in `bot_configurations`

### Message Flow Example (Slack)

```
User in Slack: "@ragbot what is vector search?"
         │
         ▼
   Slack API (Event API)
         │
         ▼
POST /webhooks/slack
  Headers: X-Slack-Signature (HMAC SHA256)
  Body: { type: "event_callback", event: { type: "app_mention", ... } }
         │
         ▼
  Verify signature (secret from Slack app config)
         │
         ▼
  Extract: user_id, channel_id, thread_ts, text
         │
         ▼
  Push to Redis Stream: slack_events
  Payload: { event_id, workspace_id, user_id, query: "what is vector search?" }
         │
         ▼
  Return 200 OK (acknowledge receipt within 3 seconds)
         │
         ▼
  [Async Worker] Consume from slack_events
         │
         ▼
  Check permissions: Does user have access to workspace?
         │
         ▼
  Execute RAG query: search("what is vector search?")
         │
         ▼
  Format response (Slack Block Kit):
    {
      blocks: [
        { type: "section", text: { text: "Vector search is..." } },
        { type: "section", text: { text: "📄 Source: doc.pdf (page 5)" } },
        { type: "actions", elements: [
          { type: "button", text: "👍 Helpful", action_id: "helpful" },
          { type: "button", text: "👎 Not helpful", action_id: "not_helpful" }
        ]}
      ]
    }
         │
         ▼
  Push to Redis Stream: outbound_messages
  Payload: { platform: "slack", channel_id, thread_ts, blocks }
         │
         ▼
  [Async Worker] Consume from outbound_messages
         │
         ▼
  Call Slack API: chat.postMessage
    Headers: Authorization: Bearer <bot_token>
    Body: { channel, thread_ts, blocks }
         │
         ▼
  Store in platform_conversations table
    (platform_message_id = response.ts)
         │
         ▼
  Update message_queue status: "completed"
```

### Two-Way Sync Scenarios

#### Scenario 1: Document Indexed in Desktop → Notify Slack
```
User indexes PDF in desktop app
         │
         ▼
Document stored in PostgreSQL (workspace_id = "team-alpha")
         │
         ▼
Trigger: INSERT on documents table
         │
         ▼
Supabase real-time notification
         │
         ▼
Webhook listener: POST /internal/document-indexed
         │
         ▼
Query integration_mappings: Find Slack channels for workspace_id
         │
         ▼
Send Slack message:
  "📄 New document indexed: `research-paper.pdf`
   Topics: machine learning, embeddings
   Indexed by: @john
   [View in Portal]"
```

#### Scenario 2: Slack Thread → Persisted Conversation
```
Team discussing in Slack thread:
  User A: "@ragbot what is FAISS?"
  Bot: "FAISS is..."
  User B: "@ragbot how does it compare to pgvector?"
  Bot: "pgvector is..."
         │
         ▼
All messages stored in platform_conversations table
         │
         ▼
Linked to session_id in sessions table
         │
         ▼
Viewable in desktop app:
  Conversation History:
    ✓ Slack #engineering thread (Dec 13, 2025)
      - User A asked: "what is FAISS?"
      - Bot replied: [answer]
      - User B asked: "how does it compare to pgvector?"
      - Bot replied: [answer]
```

#### Scenario 3: Mobile → Slack Notification
```
User searches on mobile: "deployment guide"
         │
         ▼
Query executed, answer generated
         │
         ▼
If user has Slack connected:
  Send DM via Slack:
    "💡 You searched for 'deployment guide' on mobile.
     Here's the answer: [truncated]
     [Open in Slack] [Open in Portal]"
```

## Migration Strategy

### Data Migration (JSONL → PostgreSQL)

```python
# scripts/migrate_jsonl_to_postgres.py
import json
from pathlib import Path
from src.core.db.connection import get_session
from src.core.db.models import Document
from src.core.embeddings import embed_text

def migrate():
    jsonl_path = Path("cache/rag_store.jsonl")
    session = get_session()
    
    with open(jsonl_path) as f:
        for line in f:
            doc = json.loads(line)
            
            # Generate embedding if not present
            embedding = doc.get("embedding")
            if not embedding:
                embedding = embed_text(doc["text"])
            
            # Insert into PostgreSQL
            db_doc = Document(
                uri=doc["uri"],
                text=doc["text"],
                embedding=embedding,
                metadata=doc.get("metadata", {})
            )
            session.add(db_doc)
    
    session.commit()
    print(f"Migrated {session.query(Document).count()} documents")
```

### Rollback Plan
1. Keep JSONL files for 30 days post-migration
2. Export PostgreSQL to JSONL nightly (backup strategy)
3. Feature flag to switch back to JSONL mode if needed

### Zero-Downtime Deployment
1. **Dual-write period:** Write to both JSONL and PostgreSQL for 7 days
2. **Validation:** Compare results between old/new backends
3. **Cutover:** Switch read traffic to PostgreSQL
4. **Monitor:** Track error rates, rollback if errors spike
5. **Deprecate:** Stop writing to JSONL after 30 days

---

## Cost Estimation

### Infrastructure Costs

| Tier | Users | Storage | Compute | Msg Platforms | Monthly Cost |
|------|-------|---------|---------|---------------|--------------|
| Power User (Local) | 1 | 10 GB | Local machine | N/A | $0 |
| Small Team | 5-10 | 50 GB | 2 vCPU, 4 GB RAM | 1 Slack workspace | $25-150 |
| Growing Team | 10-50 | 200 GB | 4 vCPU, 8 GB RAM | Slack + Teams | $300-800 |
| Enterprise (Single Region) | 50-500 | 1 TB | 8 vCPU, 16 GB RAM | Multi-platform | $1,500-3,500 |
| Enterprise (Multi-Region) | 500-5000 | 10 TB | 3 regions, load balancing | Enterprise Grid | $6,000-20,000 |

**Additional Costs:**
- **Redis Cloud:** $0 (free tier) - $200/mo (dedicated)
- **Slack Enterprise Grid:** Included in customer's Slack plan
- **Microsoft Teams:** Included in Microsoft 365 subscription
- **Discord:** Free for bots

**Supabase Cloud Pricing:**
- Free Tier: 500 MB database, 50k monthly active users
- Pro: $25/mo (8 GB database, unlimited API requests)
- Team: $599/mo (dedicated instance, advanced security)
- Enterprise: Custom pricing

**Self-Hosted Alternative:**
- AWS RDS PostgreSQL: ~$150-500/mo (r6g.xlarge)
- AWS ElastiCache (Redis): ~$50-200/mo
- Fly.io Postgres: ~$30-200/mo (shared to dedicated)
- DigitalOcean Managed Postgres: ~$60-400/mo

### Development Costs (Updated)

| Phase | Duration | Complexity | Est. Dev Hours |
|-------|----------|------------|----------------|
| Phase 1 | 2-3 weeks | Medium | 80-120 hours |
| Phase 2 (with integrations) | 6-8 weeks | Very High | 240-320 hours |
| Phase 3 | 4-6 weeks | High | 160-240 hours |
| Phase 4 | 8-12 weeks | Very High | 320-480 hours |

**Total:** 800-1160 dev hours (5-7 months for single developer)

---

## Technical Risks & Mitigations

### Risk 1: Vector Search Performance Degradation
**Impact:** High  
**Likelihood:** Medium  
**Mitigation:**
- Benchmark pgvector vs. FAISS before full migration
- Use IVFFlat index with tuned `lists` parameter
- Consider HNSW index for better recall at scale
- Add Redis cache for hot queries
- Fallback to FAISS for local-only deployments

### Risk 2: Sync Conflicts in Team Scenarios
**Impact:** High  
**Likelihood:** High  
**Mitigation:**
- Implement Operational Transformation (OT) for text edits
- Use CRDTs for chat messages (append-only)
- Manual conflict resolution UI (show both versions)
- Add document locking for critical edits
- Version history (time-travel queries)

### Risk 3: Multi-Tenant Data Leakage
**Impact:** Critical  
**Likelihood:** Low  
**Mitigation:**
- Enforce tenant_id in all queries (SQLAlchemy middleware)
- Row-Level Security as defense-in-depth
- Automated tests for cross-tenant access attempts
- Penetration testing before production launch
- Database-level encryption (different keys per tenant)

### Risk 4: Global Replication Lag
**Impact:** Medium  
**Likelihood:** Medium  
**Mitigation:**
- Use logical replication (faster than physical)
- Route writes to primary region, reads to nearest replica
- Monitor replication lag metrics (alert if >5 seconds)
- Implement eventual consistency for non-critical data
- Add "syncing..." UI indicators for cross-region operations

### Risk 5: Mobile App Offline Complexity
**Impact:** Medium  
**Likelihood:** High  
**Mitigation:**
- Use Supabase's built-in offline support
- Limit offline document storage (last 100 docs)
- Clear conflict resolution UX ("Keep Mine" vs. "Keep Theirs")
- Automated tests for offline → online transitions
- Graceful degradation (read-only mode if sync fails)

### Risk 6: Platform API Rate Limits
**Impact:** High  
**Likelihood:** Medium  
**Mitigation:**
- **Slack:** Tier 1 (1/sec), Tier 2 (20/min), Tier 3 (50/min). Use exponential backoff.
- **Teams:** Bot messages throttled per bot. Queue outbound messages.
- **Discord:** Global limit 50/sec, per-channel 5/5s. Implement token bucket algorithm.
- Store rate limit state in Redis (distributed rate limiting)
- Graceful degradation (queue responses if rate limited)

### Risk 7: Platform Token Security
**Impact:** Critical  
**Likelihood:** Low  
**Mitigation:**
- Encrypt bot tokens at rest (AES-256, keys in AWS KMS/HashiCorp Vault)
- Token rotation policy (90-day expiry)
- Audit log all token usage
- Revoke tokens on workspace disconnect
- Secrets scanning in CI/CD (detect accidental commits)

### Risk 8: Message Queue Failures
**Impact:** Medium  
**Likelihood:** Medium  
**Mitigation:**
- Dead letter queue for failed messages (max 3 retries)
- Monitoring + alerting (queue depth >1000, processing lag >10s)
- Persistent queue (Redis AOF or RabbitMQ durable queues)
- Manual replay tool for operators
- Circuit breaker pattern (stop consuming if LLM is down)

---

## Success Metrics

### Phase 1 (Power User)
- ✅ Vector search latency: <100ms (p95)
- ✅ Index rebuild time: <10 seconds for 10k docs
- ✅ Storage efficiency: <5% overhead vs. JSONL
- ✅ Zero data loss during migration

### Phase 2 (Cloud Sync + Integrations)
- ✅ Sync latency: <5 seconds desktop → cloud
- ✅ Offline queue: >100 operations buffered
- ✅ Mobile app startup: <2 seconds
- ✅ Battery impact: <5% drain per hour of active use
- ✅ Slack bot response time: <3 seconds (p95)
- ✅ Teams bot file upload success rate: >95%
- ✅ Discord slash command latency: <2 seconds
- ✅ Message queue processing: <1 second per event (p95)
- ✅ Webhook verification failure rate: <0.1%

### Phase 3 (Teams)
- ✅ Real-time latency: <500ms for presence updates
- ✅ Permission check latency: <10ms (cached)
- ✅ Concurrent users: >20 per workspace without degradation
- ✅ Invitation acceptance: >80% conversion rate
- ✅ Platform OAuth success rate: >98%
- ✅ Cross-platform conversation linking: >95% accuracy

### Phase 4 (Enterprise)
- ✅ Tenant provisioning: <1 minute automated
- ✅ Cross-region replication lag: <1 second (p99)
- ✅ Uptime SLA: 99.9% (43 minutes downtime/month)
- ✅ Support <1000 tenants per cluster
- ✅ Slack Enterprise Grid org deployment: <5 minutes
- ✅ Microsoft Graph indexing throughput: >100 files/minute
- ✅ Platform token rotation: 100% automated

---

## Alternative Architectures Considered

### Option 1: Qdrant + MongoDB
**Pros:** Purpose-built vector DB, JSON-native document store  
**Cons:** Two systems to maintain, no built-in sync, manual auth, no platform integrations  
**Verdict:** Rejected (operational complexity)

### Option 2: MongoDB Atlas + Vector Search
**Pros:** Single system, Atlas Device Sync for mobile  
**Cons:** Vector search still beta, less mature than pgvector, no platform bot frameworks  
**Verdict:** Rejected (less proven at scale)

### Option 3: CouchDB + FAISS
**Pros:** True bi-directional sync, built for offline-first  
**Cons:** Vector search still external, smaller ecosystem, no messaging integrations  
**Verdict:** Rejected (limited team expertise)

### Option 4: AWS DynamoDB + OpenSearch
**Pros:** Serverless, infinite scale, managed vector search  
**Cons:** Vendor lock-in, complex pricing, no local dev story  
**Verdict:** Rejected (doesn't fit local-first requirement)

### Option 5: Dedicated Platform Integration Services (e.g., Zapier, Workato)
**Pros:** No-code integrations, fast setup  
**Cons:** Limited customization, expensive at scale, data residency concerns, latency  
**Verdict:** Rejected (need full control for RAG logic)

---

## References & Resources

### Documentation
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector) - Vector similarity search
- [Supabase Docs](https://supabase.com/docs) - Real-time sync, auth, RLS
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) - Async ORM
- [Row-Level Security Best Practices](https://supabase.com/docs/guides/auth/row-level-security)

### Platform SDKs
- [Slack Bolt Python](https://slack.dev/bolt-python/) - Slack app framework
- [Slack Block Kit](https://api.slack.com/block-kit) - Interactive messages
- [Bot Framework SDK](https://github.com/microsoft/botframework-sdk) - Microsoft Teams
- [discord.py](https://discordpy.readthedocs.io/) - Discord bot library
- [Discord Interactions API](https://discord.com/developers/docs/interactions/receiving-and-responding) - Slash commands

### Example Projects
- [Supabase Realtime Chat](https://github.com/supabase/supabase/tree/master/examples/slack-clone) - React real-time chat
- [pgvector Example](https://github.com/pgvector/pgvector-python) - Python integration
- [Multi-Tenant SaaS](https://github.com/supabase/supabase/tree/master/examples/multi-tenant) - RLS patterns
- [Slack Bolt Examples](https://github.com/slackapi/bolt-python/tree/main/examples) - Slack bot patterns
- [Teams Bot Samples](https://github.com/microsoft/BotBuilder-Samples/tree/main/samples/python) - Teams bot examples

### Performance Benchmarks
- [pgvector vs. FAISS](https://nirantk.com/writing/pgvector-vs-pinecone/) - Vector search comparison
- [Supabase Realtime Benchmarks](https://supabase.com/blog/benchmarking-supabase-realtime) - 1M concurrent connections
- [Slack API Rate Limits](https://api.slack.com/docs/rate-limits) - Tier limits
- [Teams Throttling](https://learn.microsoft.com/en-us/graph/throttling) - Microsoft Graph limits

---

## Next Steps

1. **Review & Approve:** Stakeholder sign-off on this plan (including platform integrations)
2. **Proof of Concept:** Build Phase 1 prototype (1 week)
3. **Platform Developer Accounts:** Create Slack app, Teams bot, Discord application
4. **Performance Testing:** Validate pgvector with real workload
5. **Detailed Design:** API contracts, sync protocol, schema DDL, webhook signatures
6. **Kickoff Phase 1:** Begin migration implementation

**Questions? Contact:** [Your team/maintainer info]
