# n8n Expression Syntax

## Expression Format
All dynamic content uses **double curly braces**: `{{expression}}`

## Core Variables
- `{{$json.fieldName}}` — Current node output
- `{{$node["Node Name"].json.fieldName}}` — Reference other nodes (quotes required, case-sensitive)
- `{{$now.toFormat('yyyy-MM-dd')}}` — Current timestamp (Luxon)
- `{{$env.API_KEY}}` — Environment variables

## CRITICAL: Webhook Data Structure
Webhook data is **NOT** at root — it's under `.body`:
```
❌ {{$json.name}}
✅ {{$json.body.name}}
✅ {{$json.body.email}}
```

## When NOT to Use Expressions
- **Code nodes**: Use `$json.email` directly (no `{{ }}`)
- **Webhook paths**: Static paths only
- **Credential fields**: Use n8n credential system

## Common Mistakes
| Mistake | Fix |
|---------|-----|
| `$json.field` | `{{$json.field}}` |
| `{{$json.field name}}` | `{{$json['field name']}}` |
| `{{$node.HTTP Request}}` | `{{$node["HTTP Request"]}}` |
| `{{$json.name}}` (webhook) | `{{$json.body.name}}` |
| `'={{$json.email}}'` (Code node) | `$json.email` |

## Validation Rules
1. Always use `{{ }}` for dynamic content
2. Use bracket notation for spaces: `{{$json['field name']}}`
3. Node names are case-sensitive: `{{$node["HTTP Request"].json}}`
4. No nested `{{ }}` — never `{{{$json.field}}}`

## Common Patterns
```javascript
// Concatenation
Hello {{$json.body.name}}!

// In URLs
https://api.example.com/users/{{$json.body.user_id}}

// Conditional
{{$json.status === 'active' ? 'Active' : 'Inactive'}}

// Default values
{{$json.email || 'no-email@example.com'}}

// Date formatting
{{$now.toFormat('yyyy-MM-dd HH:mm')}}
{{$now.plus({days: 7}).toFormat('yyyy-MM-dd')}}
```
