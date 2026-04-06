# n8n Validation Expert

## Validation Philosophy
Validate early, validate often. Expect 2-3 validate → fix cycles (23s thinking, 58s fixing).

## Error Severity
1. **Errors** (must fix): missing_required, invalid_value, type_mismatch, invalid_expression, invalid_reference
2. **Warnings** (should fix): best_practice, deprecated, performance
3. **Suggestions** (optional): optimization, alternative

## Validation Profiles
| Profile | Use When | Strictness |
|---------|----------|------------|
| `minimal` | Quick checks during editing | Most permissive |
| `runtime` | Pre-deployment (RECOMMENDED) | Balanced |
| `ai-friendly` | AI-generated configs | Fewer false positives |
| `strict` | Production deployment | Maximum safety |

## The Validation Loop
```
1. Configure node
2. validate_node (profile: "runtime")
3. Read error messages carefully
4. Fix errors
5. validate_node again
6. Repeat until valid (usually 2-3 iterations)
```

## Common Error Types & Fixes
### missing_required
- Use `get_node` to see required fields → add missing field

### invalid_value
- Check error for allowed values → use valid option

### type_mismatch
- Convert value to correct type (e.g., string "100" → number 100)

### invalid_expression
- Add `{{ }}` if missing, use n8n Expression Syntax skill

### invalid_reference
- Check node name spelling (case-sensitive)

## Auto-Sanitization
Runs automatically on any workflow update:
- Binary operators (equals, contains) → removes singleValue
- Unary operators (isEmpty, isNotEmpty) → adds singleValue: true
- IF/Switch nodes → adds missing metadata

**Cannot auto-fix**: broken connections, branch count mismatches

## Reducing False Positives
Use `ai-friendly` profile:
```javascript
validate_node({nodeType, config, profile: "ai-friendly"})
```

## Recovery Strategies
1. **Start fresh**: Note required fields → minimal config → add incrementally
2. **Binary search**: Remove half the nodes → test → narrow down
3. **Clean connections**: `cleanStaleConnections` operation
4. **Auto-fix**: `n8n_autofix_workflow({id, applyFixes: true})`
