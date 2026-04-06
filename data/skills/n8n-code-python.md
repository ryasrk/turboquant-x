# n8n Code Python

## ⚠️ Use JavaScript for 95% of use cases
Python only when: specific stdlib functions needed, significantly more comfortable with Python.

## Quick Start
```python
items = _input.all()
processed = []
for item in items:
    processed.append({"json": {**item["json"], "processed": True}})
return processed
```

## Essential Rules
1. **Consider JavaScript first**
2. Access data: `_input.all()`, `_input.first()`, or `_input.item`
3. **CRITICAL**: Must return `[{"json": {...}}]` format
4. **CRITICAL**: Webhook data is under `_json["body"]`
5. **CRITICAL**: No external libraries (no requests, pandas, numpy)
6. Standard library only: json, datetime, re, base64, hashlib, urllib.parse, math, random, statistics

## Data Access
- `_input.all()` — Most common: arrays, batch operations
- `_input.first()` — Single objects, API responses
- `_input.item` — Each Item mode only
- `_node["Node Name"]["json"]` — Reference other nodes

## CRITICAL: Webhook Data
```python
❌ name = _json["name"]         # KeyError!
✅ name = _json["body"]["name"]  # correct!
✅ name = _json.get("body", {}).get("name")  # safe
```

## CRITICAL: No External Libraries
```python
❌ import requests   # ModuleNotFoundError!
❌ import pandas     # ModuleNotFoundError!
✅ import json, datetime, re, base64, hashlib, math, statistics
```
Workarounds: Use HTTP Request node before Code node, or switch to JavaScript.

## Return Format
```python
✅ return [{"json": {"field": value}}]         # Single result
✅ return [{"json": item["json"]} for item in items]  # Multiple
❌ return {"json": {"field": value}}            # Missing list wrapper!
```

## Top 5 Mistakes
1. **Importing external libraries** → Use HTTP Request node or JavaScript
2. **Empty/no return** → Always return data
3. **Wrong return format** → Must be `[{"json": {...}}]`
4. **Direct dict access** → Use `.get()` for safe access
5. **Webhook nesting** → Data under `["body"]`

## Best Practices
- Always use `.get()` for dictionary access
- Handle None/null explicitly
- Use list comprehensions for filtering
- Return consistent structure from all code paths
- Consider JavaScript for HTTP requests ($helpers.httpRequest)
