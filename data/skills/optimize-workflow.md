# Skill: Optimize and Improve Workflows

## When to Use
User asks to optimize, improve, speed up, or review an existing workflow.

## Steps

1. **Get the full workflow**:
   - Call `n8n_get_workflow` to see all nodes, connections, and parameters
   - Call `n8n_list_executions` to check recent execution performance

2. **Run AI analysis**:
   - Call `n8n_suggest_improvements` with the workflow ID
   - Review suggestions for relevance to user's goals

3. **Search for better patterns**:
   - Call `n8n_search_templates` with keywords matching the workflow's purpose
   - Compare template patterns to current workflow structure
   - Identify missing optimizations (batching, error handling, caching)

4. **Common improvements**:
   - **Error handling**: Add Error Trigger node + notification for failures
   - **Rate limiting**: Add Wait nodes between API calls to avoid throttling
   - **Batching**: Use SplitInBatches for large data processing
   - **Caching**: Use Set node to store intermediate results
   - **Retries**: Configure retry on failure in HTTP Request nodes
   - **Conditional logic**: Add IF nodes to skip unnecessary processing
   - **Parallel processing**: Split data into batches for parallel execution

5. **Apply improvements**:
   - Call `n8n_update_workflow` with the improved JSON
   - Test with `n8n_execute_workflow`
   - Verify execution time and success rate

## Improvement Checklist
- [ ] Has error handling (Error Trigger + notification)
- [ ] Rate limits respected (Wait nodes between API calls)
- [ ] Large data batched (SplitInBatches)
- [ ] Credentials are not hardcoded (use credential nodes)
- [ ] Workflow is activated for production use
- [ ] Trigger type is appropriate (webhook vs schedule vs manual)
- [ ] Node names are descriptive
- [ ] Unused nodes removed
