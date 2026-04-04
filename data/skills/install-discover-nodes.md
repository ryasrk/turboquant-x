# Skill: Install and Discover Nodes

## When to Use
User needs a specific integration that requires a community node, or asks what nodes/integrations are available.

## Steps

### Discover available nodes
1. Call `n8n_list_node_types` to see all installed nodes
2. Search with keyword: `n8n_list_node_types(search="telegram")` to filter
3. Node types follow patterns:
   - Built-in: `n8n-nodes-base.slack`, `n8n-nodes-base.gmail`
   - LangChain: `@n8n/n8n-nodes-langchain.agent`
   - Community: `n8n-nodes-<package>.<node>`

### Install community nodes
1. Identify the npm package name (e.g. `n8n-nodes-google-ai`)
2. Call `n8n_install_node(package_name="n8n-nodes-google-ai")`
3. Verify with `n8n_list_node_types(search="google-ai")`

### Common Community Nodes
| Package | Provides |
|---------|----------|
| `n8n-nodes-google-ai` | Google Gemini AI |
| `n8n-nodes-ollama` | Local Ollama LLM |
| `n8n-nodes-chatwoot` | Chatwoot support |
| `n8n-nodes-evolution-api` | WhatsApp via Evolution API |
| `n8n-nodes-browser-use` | Browser automation |

### When template has missing nodes
1. After deploying a template, check for missing nodes in the execution errors
2. Error will show `Unknown node type: n8n-nodes-<package>.<type>`
3. Extract the package name and install it
4. Re-execute the workflow

## Built-in Node Categories
- **Triggers**: webhook, schedule, manualTrigger, chatTrigger
- **Logic**: if, switch, merge, splitInBatches, filter
- **Data**: httpRequest, set, code, function
- **Communication**: slack, gmail, telegram, discord, whatsapp
- **Storage**: googleSheets, airtable, notion, postgres, mysql
- **AI**: langchain agent, openai, memory, tools, embeddings
- **File**: readBinary, writeBinary, spreadsheetFile, csv
