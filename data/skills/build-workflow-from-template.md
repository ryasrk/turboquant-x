# Skill: Build Workflow from Template

## When to Use
User asks to create, build, or set up an n8n workflow for a specific use case.

## Steps

1. **Search local templates** first:
   - Call `n8n_search_templates` with relevant keywords from the user's request
   - Try broad terms first (e.g. "telegram", "email", "slack"), then specific ("telegram bot openai")

2. **If local match found** (score > 0):
   - Call `n8n_get_template_detail` with the best matching template ID
   - Review the workflow JSON — check if it matches the user's needs
   - Identify which nodes need modification (triggers, credentials, parameters)
   - Tell the user what you found and how it can be adapted

3. **If no local match**, search official gallery:
   - Call `n8n_search_official` with the user's keywords
   - If found, call `n8n_fetch_official_template` to get the full JSON
   - Same review process as above

4. **Adapt the template**:
   - Modify node parameters to match user's specific requirements
   - Keep credential placeholders (user configures later)
   - Ensure trigger type matches the use case (webhook vs schedule vs chat)
   - Update node names to be descriptive for the user's context

5. **Deploy to n8n**:
   - The template JSON from `n8n_get_template_detail` or `n8n_fetch_official_template` can be passed directly to `n8n_create_workflow` via the `workflow_json` parameter
   - If you need to override the name, pass `name` alongside `workflow_json`
   - Example: `n8n_create_workflow(workflow_json=<template_output>, name="My Custom Bot")`
   - Report the new workflow ID to the user
   - Tell the user which credentials they need to configure

6. **Install missing nodes** if needed:
   - Call `n8n_list_node_types` to verify all required nodes exist
   - Call `n8n_install_node` for any community nodes not installed

## Examples

### "Create a Telegram bot that answers with AI"
1. `n8n_search_templates(query="telegram bot AI")` → find Telegram AI bot templates
2. `n8n_get_template_detail(template_id=255)` → get raw workflow JSON
3. Adapt: update system prompt, model selection
4. `n8n_create_workflow(workflow_json=<step2_output>, name="Telegram AI Bot")` → deploy

### "Set up email monitoring with Slack alerts"  
1. `n8n_search_templates(query="email slack alert")` → find email+slack templates
2. Adapt trigger conditions, Slack channel, email filters
3. Deploy and instruct user to add Gmail + Slack credentials
