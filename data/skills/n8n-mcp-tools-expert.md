# n8n MCP Tools Expert

## Tool Categories
Two sets of tools available:
1. **Built-in (n8n_*)** — Workflow CRUD, credentials, executions (cookie auth)
2. **MCP (mcp_n8n_*)** — Node search, validation, templates, documentation (no auth needed)

## Quick Reference
| Tool | Use When |
|------|----------|
| `mcp_n8n_search_nodes` | Finding nodes by keyword |
| `mcp_n8n_get_node` | Understanding node operations (detail="standard") |
| `mcp_n8n_validate_node` | Checking configurations (mode="full") |
| `n8n_create_workflow` | Creating workflows |
| `n8n_update_workflow` | Editing workflows |
| `mcp_n8n_validate_workflow` | Checking complete workflow |
| `mcp_n8n_search_templates` | Finding workflow templates |

## Critical: nodeType Formats
**Two different formats for different tools!**

### Search/Validate Tools (mcp_n8n_*)
```
"nodes-base.slack"
"nodes-base.httpRequest"
"nodes-base.webhook"
```

### Workflow Tools (n8n_*)
```
"n8n-nodes-base.slack"
"n8n-nodes-base.httpRequest"
"n8n-nodes-base.webhook"
```

## Tool Selection Workflow
1. `mcp_n8n_search_nodes({query: "slack"})` → Find node
2. `mcp_n8n_get_node({nodeType: "nodes-base.slack"})` → Understand config
3. `mcp_n8n_validate_node({nodeType, config, profile: "runtime"})` → Validate
4. `n8n_create_workflow({name, nodes, connections})` → Build
5. `mcp_n8n_validate_workflow({workflow})` → Verify
6. `n8n_update_workflow` → Iterate

## Validation Profiles
- `minimal` — Only required fields (fast, permissive)
- `runtime` — Values + types (RECOMMENDED for pre-deployment)
- `ai-friendly` — Reduce false positives (for AI config)
- `strict` — Maximum validation (production)

## Common Mistakes
1. **Wrong nodeType format**: Use `nodes-base.*` for search/validate, `n8n-nodes-base.*` for workflows
2. **Using detail="full" by default**: Use `detail="standard"` (covers 95% of use cases)
3. **Skipping validation profiles**: Always specify `profile: "runtime"`
4. **Ignoring auto-sanitization**: ALL nodes sanitized on ANY workflow update
5. **Building in one shot**: Iterate! Average 56s between edits

## Best Practices
- Use `get_node({detail: "standard"})` for most use cases
- Specify validation profile explicitly
- Include `intent` parameter in workflow updates
- Follow search → get_node → validate workflow
- Validate after every significant change
