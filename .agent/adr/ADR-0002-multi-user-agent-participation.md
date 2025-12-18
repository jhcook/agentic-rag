# ADR-0002: Multi-User Conversation and Agent Participation Model

**Status:** Accepted  
**Date:** December 18, 2025  
**Deciders:** @Architect, Product Owner  
**Technical Story:** Phase 2 Premium - Teams tier multi-user chat  

## Context and Problem Statement

The Agentic RAG application is evolving from a single-user, local-first tool to a multi-user collaborative platform. In the **Teams** and **Enterprise** tiers, multiple users will share workspaces and participate in the same conversation threads.

This creates a fundamental architectural question: **How should the AI agent participate in conversations where humans are also talking to each other?**

The agent must be intelligent enough to:
- Know when it's being addressed vs. when humans are conversing
- Avoid interrupting human-to-human discussions inappropriately
- Support configurable participation modes for different workspace cultures
- Maintain correct attribution (who said what, who the agent responded to)

## Decision Drivers

1. **User Experience**: Agent should feel helpful, not intrusive or noisy
2. **Flexibility**: Different teams have different collaboration styles
3. **Attribution**: Must always be clear who sent a message and who it's addressed to
4. **Compliance**: Audit trails require knowing who accessed what information
5. **Scalability**: Model must work for 2-person chats and 50-person channels
6. **Platform Parity**: Behavior should be consistent across web, mobile, Slack, Teams, Discord

## Considered Options

### Option 1: Always Respond (Rejected)

**Description:** Agent responds to every message in a multi-user conversation.

**Pros:**
- Simplest to implement
- Maximum agent engagement

**Cons:**
- Extremely noisy for human-to-human discussions
- Users would quickly disable or ignore the agent
- Creates a poor user experience

**Decision:** Rejected due to noise and poor UX.

### Option 2: Explicit Mention Only (Accepted as Default)

**Description:** Agent only responds when explicitly mentioned (e.g., `@rag`, `@assistant`).

**Pros:**
- Clear, predictable behavior
- Respects human discussions
- Users maintain control
- Works naturally with existing Slack/Teams/Discord patterns

**Cons:**
- May miss opportunities to help when users forget to mention
- Requires users to remember the mention syntax

**Decision:** Accepted as the **default mode** for all multi-user conversations.

### Option 3: Question Detection (Accepted as Optional Mode)

**Description:** Agent monitors for unanswered questions (messages ending with `?`) and responds after a configurable wait period (default: 30 seconds) if no human has replied.

**Pros:**
- More helpful without being intrusive
- Fills conversational gaps
- Good for async workspaces where humans may not respond quickly

**Cons:**
- May occasionally respond when a human was about to reply
- Requires NLP to detect question patterns
- Slight complexity in timing logic

**Decision:** Accepted as an **optional mode** configurable per workspace.

### Option 4: Proactive Mode (Accepted for Enterprise, with Guardrails)

**Description:** Agent actively monitors conversations and interjects when it detects:
- Factual errors it can correct
- Discussions that would benefit from documented information
- References to outdated documents
- Action items it can help with

**Pros:**
- Maximum value from indexed knowledge
- Can catch misinformation before it propagates
- Surfaces relevant context proactively

**Cons:**
- Risk of being annoying if over-triggered
- Requires high confidence thresholds
- More complex to implement correctly

**Decision:** Accepted for **Enterprise tier only** with mandatory guardrails:
- Rate limit: max 5 proactive messages per hour per conversation
- Cooldown: minimum 5 minutes between proactive interjections
- Confidence threshold: only interject if relevance score > 0.85
- Active conversation detection: never interrupt rapid human exchanges

### Option 5: Silent Observer (Accepted as Optional Mode)

**Description:** Agent never responds but continues to log and index messages.

**Pros:**
- Useful for compliance/archival channels
- Allows indexing without any interaction

**Cons:**
- No interactive value

**Decision:** Accepted as an **optional mode** for specific use cases.

## Decision

We will implement a **configurable agent participation model** with the following modes:

| Mode | Default | Tiers | Behavior |
|------|---------|-------|----------|
| `explicit_mention` | ✓ | Teams, Enterprise | Only responds to `@rag` mentions |
| `question_detection` | | Teams, Enterprise | Responds to unanswered questions after 30s |
| `proactive` | | Enterprise only | Interjects when relevant (with guardrails) |
| `silent` | | Teams, Enterprise | Never responds, only observes |

### Default Behavior

- **Single-user conversations**: Agent always responds (legacy behavior)
- **Multi-user conversations**: `explicit_mention` mode by default
- **Workspace admins** can change the mode for their workspace
- **Tenant admins** (Enterprise) can set policies restricting available modes

### Message Attribution Model

Every message will include:

```sql
sender_type         -- 'user', 'agent', 'system'
sender_user_id      -- Who sent the message (NULL for agent/system)
sender_agent_id     -- Agent identifier (NULL for humans)
reply_to_message_id -- Threading reference
mentioned_user_ids  -- Array of @mentioned users
mentioned_agent     -- Boolean: was @rag mentioned?
agent_addressed_to  -- Who the agent is responding to
kind                -- Classification: 'user_message', 'agent_response', 'human_to_human', etc.
```

### Agent Response Decision Algorithm

```
1. Check participation mode for workspace
2. If mode = 'silent': never respond
3. If mode = 'explicit_mention':
   a. Check if @rag or configured trigger was mentioned
   b. If not mentioned: classify as 'human_to_human', do not respond
   c. If mentioned: extract query, respond
4. If mode = 'question_detection':
   a. Apply explicit_mention logic first
   b. If no mention but message looks like a question:
      - Wait configured delay (default 30s)
      - If no human responded, agent responds
5. If mode = 'proactive':
   a. Apply question_detection logic first
   b. Check relevance against knowledge base
   c. Apply guardrails (rate limit, cooldown, confidence)
   d. If passes all checks, interject
```

### Follow-Up Handling

To avoid requiring repeated mentions in an ongoing exchange:

- After agent responds to a user, a **follow-up window** opens (default: 5 minutes)
- During this window, subsequent messages from the same user are assumed to be follow-ups
- Agent responds without requiring re-mention
- Window resets with each agent response
- Window closes if another user addresses the agent or the original user mentions someone else

## Consequences

### Positive

- Clear, predictable agent behavior
- Respects human collaboration patterns
- Configurable to match team culture
- Full attribution for compliance
- Platform-agnostic (works for web, mobile, Slack, Teams, Discord)

### Negative

- More complex than single-user model
- Requires workspace-level configuration
- Proactive mode requires careful threshold tuning
- Additional database columns and indexes

### Neutral

- Existing single-user flows remain unchanged
- Migration path: all existing conversations get `explicit_mention` default

## Implementation Notes

### New Database Tables

1. `workspace_agent_settings` - Per-workspace agent configuration
2. Extended `conversation_messages` with attribution fields

### New API Endpoints

1. `GET/PUT /api/workspaces/{id}/agent-settings` - Manage agent behavior
2. Messages API extended with sender/mention metadata

### UI Changes

1. Workspace settings panel for agent configuration
2. Message bubbles show sender avatar + name
3. Agent responses show "→ @user" when addressing specific user
4. Mention autocomplete (`@` triggers user/agent picker)
5. "RAG Assistant is thinking..." typing indicator

## Related Decisions

- ADR-0001: Apple Silicon M2 Pro (platform baseline)
- Phase 2 Premium Architecture (parent plan)
- Database Migration Strategy (schema evolution)

## References

- [Slack Bot Best Practices](https://api.slack.com/best-practices/user-experience)
- [Microsoft Teams Bot Design Guidelines](https://learn.microsoft.com/en-us/microsoftteams/platform/bots/design/bots)
- [Discord Bot UX Patterns](https://discord.com/developers/docs/topics/community-resources)
