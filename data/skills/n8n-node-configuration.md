# n8n Node Configuration

## Configuration Philosophy
Progressive disclosure: Start minimal, add complexity as needed.
- `get_node` with `detail: "standard"` covers 95% of use cases (~1-2K tokens)
- Average 56s between configuration edits

## Core Concepts
### 1. Operation-Aware Configuration
Not all fields are always required — it depends on resource + operation.
Example: Slack `post` needs channel+text; `update` needs messageId+text.

### 2. Property Dependencies
Fields appear/disappear based on other field values.
Example: HTTP Request `sendBody` only shows for POST/PUT/PATCH; `body` requires `sendBody=true`.

### 3. Progressive Discovery
1. `get_node({detail: "standard"})` — DEFAULT, 95% of needs
2. `get_node({mode: "search_properties", propertyQuery: "auth"})` — Find specific fields
3. `get_node({detail: "full"})` — Complete schema (use sparingly, 3-8K tokens)

## Configuration Workflow
```
1. Identify node type + operation
2. get_node (standard detail)
3. Configure required fields
4. Validate configuration
5. If field unclear → search_properties mode
6. Add optional fields as needed
7. Validate again → Deploy
```

## Common Node Patterns

### Resource/Operation Nodes (Slack, Google Sheets, Airtable)
```json
{"resource": "<entity>", "operation": "<action>", ...operation-specific fields}
```

### HTTP-Based Nodes
- POST/PUT/PATCH → sendBody available
- sendBody=true → body required
- authentication != "none" → credentials required

### Conditional Logic (IF, Switch)
- Binary operators (equals, contains) → value1 + value2
- Unary operators (isEmpty, isNotEmpty) → value1 only + singleValue: true

## Key Examples
### Slack Post Message
```json
{"resource": "message", "operation": "post", "channel": "#general", "text": "Hello!"}
```

### HTTP POST with JSON
```json
{"method": "POST", "url": "https://api.example.com", "sendBody": true, "body": {"contentType": "json", "content": {...}}}
```

## Anti-Patterns
- ❌ Over-configure upfront (start minimal instead)
- ❌ Skip validation before deploying
- ❌ Use same config for different operations
- ❌ Jump to detail="full" immediately
