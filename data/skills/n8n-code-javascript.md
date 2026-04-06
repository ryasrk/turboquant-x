# n8n Code JavaScript

## Quick Start
```javascript
const items = $input.all();
const processed = items.map(item => ({
  json: { ...item.json, processed: true, timestamp: new Date().toISOString() }
}));
return processed;
```

## Essential Rules
1. Choose "Run Once for All Items" mode (95% of use cases)
2. Access data: `$input.all()`, `$input.first()`, or `$input.item`
3. **CRITICAL**: Must return `[{json: {...}}]` format
4. **CRITICAL**: Webhook data is under `$json.body` (not `$json` directly)
5. Built-ins: `$helpers.httpRequest()`, `DateTime` (Luxon), `$jmespath()`

## Data Access Patterns
- `$input.all()` — Most common: arrays, batch operations, aggregations
- `$input.first()` — Single objects, API responses
- `$input.item` — Each Item mode only
- `$node["Node Name"].json` — Reference other nodes

## CRITICAL: Webhook Data
```javascript
❌ const name = $json.name;       // undefined!
✅ const name = $json.body.name;  // correct!
```

## Return Format
```javascript
✅ return [{json: {field1: value1}}];           // Single result
✅ return items.map(i => ({json: i.json}));     // Multiple results
✅ return [];                                    // Empty result
❌ return {json: {field: value}};               // Missing array wrapper!
❌ return [{field: value}];                      // Missing json wrapper!
```

## Top 5 Mistakes
1. **Empty/no return**: Always return data
2. **Expression syntax in code**: Use `$json.field` not `"{{$json.field}}"`
3. **Wrong return wrapper**: Must be `[{json: {...}}]`
4. **Missing null checks**: Use `item.json?.user?.email || 'default'`
5. **Webhook body nesting**: Data is under `.body`

## Built-in Functions
```javascript
// HTTP requests
const response = await $helpers.httpRequest({method: 'GET', url: '...'});

// Dates (Luxon)
const now = DateTime.now();
const formatted = now.toFormat('yyyy-MM-dd');
const tomorrow = now.plus({days: 1});

// JSON queries
const adults = $jmespath(data, 'users[?age >= `18`]');
```

## When to Use Code Node
✅ Complex transformations, custom business logic, aggregation
❌ Simple field mapping → Set node
❌ Basic filtering → Filter node
❌ Simple conditionals → IF/Switch node
