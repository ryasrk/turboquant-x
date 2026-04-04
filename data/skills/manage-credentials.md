# Skill: Manage Credentials

## When to Use
User asks to set up, check, update, or troubleshoot credentials for n8n services.

## Steps

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

## Security Rules
- Never expose full API keys in responses — show only first/last 4 chars
- When displaying credentials, mask sensitive values: `sk-...abc1`
- Remind user to use strong, unique API keys
- Suggest credential rotation if keys look old or compromised
