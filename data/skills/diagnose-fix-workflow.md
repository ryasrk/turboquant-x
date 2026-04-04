# Skill: Diagnose and Fix Workflow Errors

## When to Use
User reports workflow errors, failures, or asks "why isn't my workflow working?"

## Steps

1. **Gather error data** (never ask for IDs — look them up):
   - If workspace has a workflow ID → call `n8n_workflow_status` immediately
   - Call `n8n_list_executions` with `status=error` to find failed runs
   - Call `n8n_execution_detail` on the most recent failed execution

2. **Diagnose the error**:
   - Call `n8n_diagnose_error` with the execution ID for AI-assisted analysis
   - Identify the failing node, error type, and root cause
   - Check common patterns:
     - **Authentication errors** → credentials expired or misconfigured
     - **Connection errors** → service unreachable, API endpoint changed
     - **Data errors** → missing fields, wrong format, null values
     - **Rate limiting** → too many requests, add delays
     - **Node not found** → community node needs installation

3. **Apply the fix**:
   - For credential issues → tell user to update credentials in n8n UI
   - For node parameter issues → call `n8n_update_workflow` with corrected JSON
   - For missing nodes → call `n8n_install_node`
   - For logic errors → call `n8n_get_workflow` to see full workflow, identify flow issues

4. **Verify the fix**:
   - Call `n8n_execute_workflow` to trigger a test run
   - Call `n8n_list_executions` to check if the new run succeeded
   - Report results to user

## Error Pattern Quick Reference

| Error Type | Likely Cause | Fix |
|------------|-------------|-----|
| 401/403 | Auth expired | Update credentials |
| 404 | Wrong API endpoint/URL | Fix HTTP Request node URL |
| ECONNREFUSED | Service down | Check service availability |
| TypeError | Wrong data shape | Add Set node to reshape data |
| Rate limit | Too fast | Add Wait node between calls |
| Node not found | Missing package | `n8n_install_node` |

## Key Rules
- NEVER ask "what's your execution ID?" — look it up with `n8n_list_executions`
- NEVER ask "what's your workflow ID?" — get it from workspace context
- Always show: what failed, why, and how to fix it
- If you can fix it programmatically, do it; don't just suggest
