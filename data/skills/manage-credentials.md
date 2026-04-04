# Skill: Manage Credentials

## When to Use
User asks to set up, check, update, or troubleshoot credentials for n8n services.
Also use PROACTIVELY when deploying a workflow that has credential references — check and setup before activation.

## Steps

### Pre-deployment credential check (CRITICAL)
Before activating any workflow:
1. Call `n8n_get_workflow` to inspect the workflow nodes
2. Look at every node's `credentials` field to find required credential types
3. Call `n8n_list_credentials` to see what's already configured
4. For EACH missing credential type:
   a. Tell the user which credential is needed and for which node
   b. Ask for the secret values (API key, token, etc.)
   c. Call `n8n_create_credential` with the correct type, name, and data
5. After creating all credentials, use `n8n_update_workflow` to link credential IDs to nodes
6. ONLY THEN attempt activation

### List existing credentials
- Call `n8n_list_credentials` to see what's configured
- Identify which services have credentials and which are missing

### Check credential health
- Call `n8n_get_credential` with a credential ID to read its data
- Verify required fields are populated (API keys, tokens, secrets)
- Check for expired tokens or placeholder values

### Create new credentials
1. Identify the credential type needed (e.g. `slackApi`, `openAiApi`, `gmailOAuth2`)
2. Call `n8n_list_node_types` to find the exact credential type name
3. Call `n8n_create_credential` with:
   - `type`: exact credential type string
   - `name`: descriptive name
   - `data`: dict with required fields

### Update existing credentials
- Call `n8n_update_credential` with the credential ID and new data
- Common use case: rotating API keys, updating OAuth tokens

### Link credentials to workflow nodes
After creating a credential, update the workflow to reference it:
1. Call `n8n_get_workflow` to get current workflow JSON
2. Find the node that needs the credential
3. Update the node's `credentials` field: `{"credType": {"id": "CRED_ID", "name": "credType"}}`
4. Call `n8n_update_workflow` with the updated nodes

### Common Credential Types
| Service | Type | Required Fields |
|---------|------|----------------|
| OpenAI | `openAiApi` | `apiKey` |
| Slack | `slackApi` | `accessToken` |
| Gmail (OAuth) | `gmailOAuth2` | `clientId`, `clientSecret`, `refreshToken` |
| Telegram | `telegramApi` | `accessToken` |
| HTTP Header Auth | `httpHeaderAuth` | `name`, `value` |
| HTTP Basic Auth | `httpBasicAuth` | `user`, `password` |
| Webhook | `httpHeaderAuth` | `name`, `value` |
| GitHub | `githubApi` | `accessToken` |
| Postgres | `postgresApi` | `host`, `database`, `user`, `password`, `port` |
| MySQL | `mySqlApi` | `host`, `database`, `user`, `password`, `port` |
| Google Sheets | `googleSheetsOAuth2Api` | `clientId`, `clientSecret`, `refreshToken` |
| Notion | `notionApi` | `apiKey` |
| Discord | `discordApi` | `botToken` |
| SMTP | `smtp` | `host`, `port`, `user`, `password` |

## Security Rules
- Never expose full API keys in responses — show only first/last 4 chars
- When displaying credentials, mask sensitive values: `sk-...abc1`
- Remind user to use strong, unique API keys
- Suggest credential rotation if keys look old or compromised
- NEVER store credentials in workflow JSON — always use n8n credential references
