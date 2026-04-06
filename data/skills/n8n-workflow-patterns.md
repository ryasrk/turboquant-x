# n8n Workflow Patterns

## The 5 Core Patterns

### 1. Webhook Processing (Most Common)
Webhook → Validate → Transform → Respond/Notify

### 2. HTTP API Integration
Trigger → HTTP Request → Transform → Action → Error Handler

### 3. Database Operations
Schedule → Query → Transform → Write → Verify

### 4. AI Agent Workflow
Trigger → AI Agent (Model + Tools + Memory) → Output

### 5. Scheduled Tasks
Schedule → Fetch → Process → Deliver → Log

## Pattern Selection Guide
- **Webhook Processing** — Receiving data from external systems, Slack commands, form submissions
- **HTTP API** — Fetching data from APIs, syncing with third-party services
- **Database** — Syncing between databases, ETL workflows
- **AI Agent** — Conversational AI, multi-step reasoning with tools
- **Scheduled Tasks** — Recurring reports, periodic data fetching

## Workflow Creation Checklist
### Planning
- [ ] Identify the pattern
- [ ] List required nodes (use search_nodes)
- [ ] Plan error handling strategy

### Implementation
- [ ] Create workflow with trigger
- [ ] Add data source nodes
- [ ] Configure credentials
- [ ] Add transformation nodes
- [ ] Add output nodes
- [ ] Configure error handling

### Validation & Deploy
- [ ] Validate each node (validate_node)
- [ ] Validate complete workflow (validate_workflow)
- [ ] Test with sample data
- [ ] Activate and monitor first executions

## Data Flow Patterns
- **Linear**: Trigger → Transform → Action → End
- **Branching**: Trigger → IF → [True/False paths]
- **Parallel**: Trigger → [Branch 1 + Branch 2] → Merge
- **Loop**: Trigger → Split in Batches → Process → Loop

## Common Gotchas
1. **Webhook data**: Nested under `$json.body` (not `$json` directly)
2. **Execution order**: Use v1 connection-based (recommended)
3. **Authentication**: Configure in Credentials section, not parameters
4. **Multiple triggers**: Only one executes — split into separate workflows

## Quick Start Examples
### Webhook → Slack
1. Webhook (POST) → 2. Set (map fields) → 3. Slack (post to #notifications)

### Scheduled Report
1. Schedule (daily 9 AM) → 2. HTTP Request → 3. Code (aggregate) → 4. Email → 5. Error Trigger → Slack

### AI Assistant
1. Webhook → 2. AI Agent (Model + Tools + Memory) → 3. Webhook Response
