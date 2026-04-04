# Skill: Manage Workflows Lifecycle

## When to Use
User asks to list, activate, deactivate, delete, or manage workflows.

## Available Actions

### List all workflows
```
n8n_list_workflows → shows all workflows with ID, name, active status
```

### Get full workflow details
```
n8n_get_workflow(workflow_id) → full JSON with all nodes and connections
```

### Activate/Deactivate
```
n8n_activate_workflow(workflow_id, active=true/false)
```
- Activate: enables triggers (webhooks start listening, schedules start running)
- Deactivate: stops all triggers, workflow won't execute automatically

### Trigger manual execution
```
n8n_execute_workflow(workflow_id) → runs the workflow once manually
```
- Useful for testing before activating
- Returns execution ID for tracking results

### Delete workflow
```
n8n_delete_workflow(workflow_id) → permanently removes the workflow
```
- ALWAYS confirm with user before deleting
- Cannot be undone

### Create new workflow
```
n8n_create_workflow(name, nodes, connections) → creates and returns new workflow ID
```
- Use template as base when possible
- Always include a trigger node

### Update workflow
```
n8n_update_workflow(workflow_id, nodes=..., connections=..., name=..., active=...)
```
- Can update individual fields or entire workflow JSON

## Workflow States
- **Inactive**: Created but not running. Triggers not listening.
- **Active**: Triggers enabled. Webhooks listening. Schedules firing.
- **Error**: Last execution failed. Check with `n8n_list_executions`.

## Best Practices
- Test with `n8n_execute_workflow` before activating
- Check `n8n_list_executions` after activation to verify success
- Deactivate before making significant changes to avoid partial runs
- Always report the workflow ID and URL to the user after creation
