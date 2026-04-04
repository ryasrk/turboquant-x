"""Agent tools for n8n workflow management and error diagnosis."""

from __future__ import annotations

import json
from typing import Any

from src.agent.base import Tool

# ── Template JSON cache for context-efficient piping ────────────────
# Template tools store full JSON here; create_workflow looks it up by ref key.
_template_json_cache: dict[str, str] = {}


class N8nWorkflowStatusTool(Tool):
    """Check the live status of an n8n workflow."""

    @property
    def name(self) -> str:
        return "n8n_workflow_status"

    @property
    def description(self) -> str:
        return (
            "Get the current status of an n8n workflow including whether it is active, "
            "its name, node count, and recent execution summary. "
            "Use when the user asks about a workflow's state or health."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The n8n workflow ID to check",
                },
            },
            "required": ["workflow_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        workflow_id = kwargs["workflow_id"]
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call("GET", f"/rest/workflows/{workflow_id}")
            resp.raise_for_status()
            wf = resp.json()
            data = wf.get("data", wf)

            nodes = data.get("nodes", [])
            node_types = [n.get("type", "?").split(".")[-1] for n in nodes[:10]]
            if len(nodes) > 10:
                node_types.append(f"+{len(nodes) - 10}more")

            # TOON format
            lines = [
                f"# workflow {workflow_id}",
                f"name: {data.get('name', 'Untitled')}",
                f"active: {data.get('active', False)}",
                f"nodes: {len(nodes)} [{','.join(node_types)}]",
                f"updated: {data.get('updatedAt', '?')}",
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching workflow {workflow_id}: {e}"


class N8nListExecutionsTool(Tool):
    """List recent n8n workflow executions."""

    @property
    def name(self) -> str:
        return "n8n_list_executions"

    @property
    def description(self) -> str:
        return (
            "List recent executions for an n8n workflow. Shows execution IDs, "
            "status (success/error/waiting), start time, and duration. "
            "Use when the user asks about workflow runs or wants to see execution history."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The n8n workflow ID (optional — omit for all workflows)",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status: success, error, waiting (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10, max 50)",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_manager import get_executions

        wf_id = kwargs.get("workflow_id")
        status = kwargs.get("status")
        limit = min(kwargs.get("limit", 10), 50)

        try:
            execs = await get_executions(
                workflow_id=wf_id,
                status=status,
                limit=limit,
            )
            if not execs:
                return "No executions found."

            # TOON tabular format
            lines = [f"executions[{len(execs)}]{{id,status,started,stopped,mode}}:"]
            for ex in execs:
                eid = ex.get("id", "?")
                finished = ex.get("finished", None)
                status_str = "success" if finished else "error" if finished is False else "running"
                started = ex.get("startedAt", "?")
                stopped = ex.get("stoppedAt", "-")
                mode = ex.get("mode", "-")
                lines.append(f"  {eid},{status_str},{started},{stopped},{mode}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error listing executions: {e}"


class N8nExecutionDetailTool(Tool):
    """Get detailed execution data including per-node results and errors."""

    @property
    def name(self) -> str:
        return "n8n_execution_detail"

    @property
    def description(self) -> str:
        return (
            "Get detailed information about a specific n8n execution, including "
            "per-node execution results, error messages, input/output data, and timing. "
            "Use when the user asks why a workflow failed or wants to see execution details."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "The n8n execution ID to inspect",
                },
            },
            "required": ["execution_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        execution_id = kwargs["execution_id"]
        from src.server.n8n_manager import get_execution_summary

        try:
            summary = await get_execution_summary(execution_id)
            return summary
        except Exception as e:
            return f"Error fetching execution {execution_id}: {e}"


class N8nDiagnoseErrorTool(Tool):
    """Analyze a failed n8n execution and recommend fixes."""

    @property
    def name(self) -> str:
        return "n8n_diagnose_error"

    @property
    def description(self) -> str:
        return (
            "Analyze a failed n8n workflow execution and provide diagnosis with "
            "actionable recommendations. Reads the execution error details, identifies "
            "the failing node, and suggests fixes. Use when a workflow fails and the user "
            "wants help understanding and fixing the issue."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "The n8n execution ID that failed",
                },
            },
            "required": ["execution_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        execution_id = kwargs["execution_id"]
        from src.server.n8n_manager import get_execution_detail

        try:
            detail = await get_execution_detail(execution_id)
            if not detail:
                return f"Execution {execution_id} not found."

            # detail is the execution object with keys like:
            # id, status, startedAt, stoppedAt, data, workflowData, etc.
            # detail["data"] is the run results (already parsed from JSON string)
            inner_data = detail.get("data", {})
            if not isinstance(inner_data, dict):
                inner_data = {}
            result_data = inner_data.get("resultData", {})
            if not isinstance(result_data, dict):
                result_data = {}
            run_data = result_data.get("runData", {})
            if not isinstance(run_data, dict):
                run_data = {}
            error = result_data.get("error") or detail.get("error")

            # Find failing nodes
            failing_nodes = []
            for node_name, runs in run_data.items():
                for run in (runs if isinstance(runs, list) else [runs]):
                    if isinstance(run, dict) and run.get("error"):
                        failing_nodes.append({
                            "node": node_name,
                            "error": run["error"],
                        })

            lines = [f"=== Diagnosis for Execution {execution_id} ===\n"]

            if error:
                err_msg = error if isinstance(error, str) else error.get("message", json.dumps(error))
                lines.append(f"Top-level error: {err_msg}\n")

            if failing_nodes:
                lines.append(f"Failing nodes ({len(failing_nodes)}):\n")
                for fn in failing_nodes:
                    node = fn["node"]
                    err = fn["error"]
                    err_msg = err if isinstance(err, str) else err.get("message", json.dumps(err))
                    err_type = err.get("type", "unknown") if isinstance(err, dict) else "unknown"

                    lines.append(f"  ❌ Node: {node}")
                    lines.append(f"     Error type: {err_type}")
                    lines.append(f"     Message: {err_msg}")

                    # Generate recommendations based on common error patterns
                    recs = _diagnose_node_error(node, err_msg, err_type)
                    if recs:
                        lines.append(f"     Recommendations:")
                        for r in recs:
                            lines.append(f"       → {r}")
                    lines.append("")
            elif not error:
                lines.append("No errors found — execution may have succeeded or is still running.")

            return "\n".join(lines)
        except Exception as e:
            return f"Error diagnosing execution {execution_id}: {e}"


class N8nInstallNodeTool(Tool):
    """Install a community node package in n8n."""

    @property
    def name(self) -> str:
        return "n8n_install_node"

    @property
    def description(self) -> str:
        return (
            "Install a community node package in n8n. Use when a workflow requires "
            "a node that is not installed, or the user asks to add a new integration. "
            "Provide the npm package name (e.g. 'n8n-nodes-slack')."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "package_name": {
                    "type": "string",
                    "description": "npm package name to install (e.g. 'n8n-nodes-slack')",
                },
            },
            "required": ["package_name"],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        package_name = kwargs["package_name"]
        from src.server.n8n_manager import install_community_node

        try:
            result = await install_community_node(package_name)
            return f"Successfully installed {package_name}: {json.dumps(result)}"
        except Exception as e:
            return f"Failed to install {package_name}: {e}"


class N8nListCredentialsTool(Tool):
    """List all credentials configured in n8n."""

    @property
    def name(self) -> str:
        return "n8n_list_credentials"

    @property
    def description(self) -> str:
        return (
            "List all credentials configured in n8n, including their type, name, and ID. "
            "Use when the user asks what API keys or credentials are set up, or when "
            "checking which services are connected."
        )

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call("GET", "/rest/credentials")
            resp.raise_for_status()
            data = resp.json()
            creds = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(creds, dict):
                creds = creds.get("data", [])

            if not creds:
                return "No credentials configured. Use n8n_create_credential to add one."

            # TOON tabular format
            lines = [f"credentials[{len(creds)}]{{id,name,type,created}}:"]
            for c in creds:
                cid = c.get("id", "?")
                name = c.get("name", "Untitled")
                ctype = c.get("type", "unknown")
                created = c.get("createdAt", "-")
                lines.append(f"  {cid},{name},{ctype},{created}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error listing credentials: {e}"


class N8nCreateCredentialTool(Tool):
    """Create a new credential (API key, OAuth, etc.) in n8n."""

    @property
    def name(self) -> str:
        return "n8n_create_credential"

    @property
    def description(self) -> str:
        return (
            "Create a new credential in n8n for connecting to external services. "
            "Common types: slackApi, openAiApi, gmailOAuth2, postgresApi, httpHeaderAuth, "
            "httpBasicAuth, httpQueryAuth. Provide the credential type, a display name, "
            "and the data (e.g. API key). Use when the user wants to add or configure "
            "a service connection."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Credential type (e.g. 'openAiApi', 'slackApi', 'httpHeaderAuth')",
                },
                "name": {
                    "type": "string",
                    "description": "Display name for the credential (e.g. 'My OpenAI Key')",
                },
                "data": {
                    "type": "object",
                    "description": "Credential data as key-value pairs. Examples: "
                                   "{\"apiKey\": \"sk-...\"} for OpenAI, "
                                   "{\"accessToken\": \"xoxb-...\"} for Slack, "
                                   "{\"name\": \"Authorization\", \"value\": \"Bearer ...\"} for HTTP Header Auth",
                },
            },
            "required": ["type", "name", "data"],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        cred_type = kwargs["type"]
        name = kwargs["name"]
        data = kwargs["data"]
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call(
                "POST",
                "/rest/credentials",
                json_data={"type": cred_type, "name": name, "data": data},
            )
            resp.raise_for_status()
            result = resp.json()
            cred = result.get("data", result) if isinstance(result, dict) else result
            cid = cred.get("id", "?")
            return (
                f"Created credential '{name}' (type: {cred_type}, id: {cid}).\n\n"
                f"NEXT STEP: To link this credential to a workflow node, call n8n_update_workflow "
                f"and update the node's 'credentials' field to include:\n"
                f'  "{cred_type}": {{"id": "{cid}", "name": "{name}"}}\n\n'
                f"Example: get the workflow with n8n_get_workflow, modify the relevant node's "
                f"credentials object, then call n8n_update_workflow with the updated JSON."
            )
        except Exception as e:
            return f"Failed to create credential: {e}"


class N8nDeleteCredentialTool(Tool):
    """Delete a credential from n8n."""

    @property
    def name(self) -> str:
        return "n8n_delete_credential"

    @property
    def description(self) -> str:
        return (
            "Delete a credential from n8n by its ID. Use when the user wants to remove "
            "an old or unused credential."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "credential_id": {
                    "type": "string",
                    "description": "The credential ID to delete",
                },
            },
            "required": ["credential_id"],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        cred_id = kwargs["credential_id"]
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call("DELETE", f"/rest/credentials/{cred_id}")
            resp.raise_for_status()
            return f"Deleted credential {cred_id}."
        except Exception as e:
            return f"Failed to delete credential {cred_id}: {e}"


class N8nUpdateWorkflowTool(Tool):
    """Update an existing n8n workflow's configuration."""

    @property
    def name(self) -> str:
        return "n8n_update_workflow"

    @property
    def description(self) -> str:
        return (
            "Update an n8n workflow. You can pass a complete workflow_json string "
            "(from template tools or n8n_get_workflow) to replace the entire workflow, "
            "or provide individual fields (name, nodes, connections, settings) to patch."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The n8n workflow ID to update",
                },
                "workflow_json": {
                    "type": "string",
                    "description": (
                        "Complete workflow JSON as a string — replaces nodes, connections, "
                        "and name. Can be raw JSON from template tools or n8n_get_workflow."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "New workflow name (optional, overrides workflow_json name)",
                },
                "nodes": {
                    "type": "array",
                    "description": "Full updated nodes array (optional — ignored if workflow_json provided)",
                },
                "connections": {
                    "type": "object",
                    "description": "Full updated connections object (optional — ignored if workflow_json provided)",
                },
                "settings": {
                    "type": "object",
                    "description": "Workflow settings to update (optional, merged on top)",
                },
            },
            "required": ["workflow_id"],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        workflow_id = kwargs["workflow_id"]
        from src.server.n8n_setup import n8n_api_call

        try:
            # Fetch current workflow first
            resp = await n8n_api_call("GET", f"/rest/workflows/{workflow_id}")
            resp.raise_for_status()
            wf_data = resp.json()
            wf = wf_data.get("data", wf_data) if isinstance(wf_data, dict) else wf_data

            # If workflow_json provided, parse and apply as bulk replacement
            if "workflow_json" in kwargs and kwargs["workflow_json"]:
                try:
                    parsed = N8nCreateWorkflowTool._parse_workflow_json(kwargs["workflow_json"])
                except (json.JSONDecodeError, ValueError) as e:
                    return f"Failed to parse workflow_json: {e}"
                wf["nodes"] = parsed.get("nodes", wf.get("nodes", []))
                wf["connections"] = parsed.get("connections", wf.get("connections", {}))
                if parsed.get("name"):
                    wf["name"] = parsed["name"]

            # Apply individual overrides (take priority over workflow_json)
            if "name" in kwargs:
                wf["name"] = kwargs["name"]
            if "nodes" in kwargs and "workflow_json" not in kwargs:
                wf["nodes"] = kwargs["nodes"]
            if "connections" in kwargs and "workflow_json" not in kwargs:
                wf["connections"] = kwargs["connections"]
            if "settings" in kwargs:
                wf.setdefault("settings", {}).update(kwargs["settings"])

            # Push updated workflow
            update_resp = await n8n_api_call("PUT", f"/rest/workflows/{workflow_id}", json_data=wf)
            update_resp.raise_for_status()

            updated = update_resp.json()
            updated_wf = updated.get("data", updated) if isinstance(updated, dict) else updated
            node_count = len(updated_wf.get("nodes", []))
            return f"Updated workflow '{updated_wf.get('name', workflow_id)}' ({node_count} nodes)"
        except Exception as e:
            return f"Failed to update workflow {workflow_id}: {e}"


class N8nSuggestImprovementsTool(Tool):
    """Analyze a workflow and suggest improvements."""

    @property
    def name(self) -> str:
        return "n8n_suggest_improvements"

    @property
    def description(self) -> str:
        return (
            "Analyze an n8n workflow and suggest improvements for reliability, error handling, "
            "performance, and best practices. Use when the user asks for workflow optimization "
            "or a review of their automation."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The n8n workflow ID to analyze",
                },
            },
            "required": ["workflow_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        workflow_id = kwargs["workflow_id"]
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call("GET", f"/rest/workflows/{workflow_id}")
            resp.raise_for_status()
            wf_data = resp.json()
            wf = wf_data.get("data", wf_data) if isinstance(wf_data, dict) else wf_data

            nodes = wf.get("nodes", [])
            connections = wf.get("connections", {})
            suggestions: list[str] = []

            # Check for missing error handling
            has_error_trigger = any("errorTrigger" in (n.get("type", "").lower()) for n in nodes)
            if not has_error_trigger:
                suggestions.append(
                    "Add an Error Trigger node to handle workflow failures (n8n-nodes-base.errorTrigger)"
                )

            # Check for missing retry logic
            for n in nodes:
                ntype = n.get("type", "")
                if "httpRequest" in ntype:
                    retry = n.get("parameters", {}).get("options", {}).get("retry", {})
                    if not retry:
                        suggestions.append(
                            f"Node '{n['name']}': Enable retry on HTTP requests for resilience"
                        )

            # Check for hardcoded values
            for n in nodes:
                params = json.dumps(n.get("parameters", {}))
                if "http://" in params or "https://" in params:
                    if "={{ " not in params:
                        suggestions.append(
                            f"Node '{n.get('name')}': Consider using environment variables or "
                            f"n8n expressions instead of hardcoded URLs"
                        )

            # Check for missing credentials
            for n in nodes:
                ntype = n.get("type", "")
                requires_creds = any(
                    svc in ntype.lower() for svc in
                    ("slack", "gmail", "openai", "postgres", "mysql", "http", "webhook")
                )
                if requires_creds and not n.get("credentials"):
                    suggestions.append(
                        f"Node '{n.get('name')}' ({ntype}) has no credentials linked — "
                        f"configure credentials for production use"
                    )

            # Check for unconnected nodes
            connected_targets = set()
            for _src, outputs in connections.items():
                main_outputs = outputs.get("main", [])
                for output_group in main_outputs:
                    for conn in output_group:
                        connected_targets.add(conn.get("node"))

            connected_sources = set(connections.keys())
            for n in nodes:
                name = n.get("name", "")
                ntype = n.get("type", "")
                is_trigger = "trigger" in ntype.lower() or "webhook" in ntype.lower()
                is_end_node = name not in connected_sources and not is_trigger
                is_orphan = name not in connected_targets and not is_trigger

                if is_orphan and not is_trigger:
                    suggestions.append(f"Node '{name}' appears disconnected — no input connections")

            if not suggestions:
                return f"Workflow '{wf.get('name', 'Untitled')}' looks good! No major improvements found."

            lines = [f"Suggestions for '{wf.get('name', 'Untitled')}' ({len(nodes)} nodes):\n"]
            for i, s in enumerate(suggestions, 1):
                lines.append(f"  {i}. {s}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing workflow {workflow_id}: {e}"


def _diagnose_node_error(node_name: str, error_msg: str, error_type: str) -> list[str]:
    """Generate recommendations based on common n8n error patterns."""
    recs: list[str] = []
    msg_lower = (error_msg or "").lower()

    # Authentication errors
    if any(w in msg_lower for w in ("401", "unauthorized", "authentication", "invalid token", "forbidden", "403")):
        recs.append("Check credentials: the API key or OAuth token may be expired or invalid")
        recs.append("Verify the credential is correctly linked to this node")
        recs.append("Re-authenticate the service if using OAuth2")

    # Connection errors
    if any(w in msg_lower for w in ("econnrefused", "enotfound", "timeout", "network", "connection")):
        recs.append("Check that the target service is reachable from this server")
        recs.append("Verify the URL/hostname is correct")
        recs.append("Check for firewall rules or DNS issues")

    # Rate limiting
    if any(w in msg_lower for w in ("429", "rate limit", "too many requests", "throttle")):
        recs.append("Add a wait/delay node before this node to respect rate limits")
        recs.append("Reduce batch size or add pagination")
        recs.append("Check the API's rate limit documentation")

    # Data/validation errors
    if any(w in msg_lower for w in ("validation", "required", "missing", "invalid", "schema")):
        recs.append("Check that all required fields are populated")
        recs.append("Verify input data format matches what the node expects")
        recs.append("Use an IF node to filter out incomplete data before this node")

    # Expression errors
    if any(w in msg_lower for w in ("expression", "cannot read", "undefined", "null")):
        recs.append("Check node expressions — a referenced field may not exist in the input data")
        recs.append("Use {{ $json.field ?? 'default' }} to handle missing fields")
        recs.append("Add an IF node to check data exists before processing")

    # Node not found
    if any(w in msg_lower for w in ("node type", "not found", "not installed", "unknown node")):
        recs.append("The node type may need to be installed as a community package")
        recs.append("Use the n8n_install_node tool to install the required package")

    if not recs:
        recs.append("Review the full error message and node configuration")
        recs.append("Check n8n community forum for similar error reports")
        recs.append("Try running the workflow manually with test data to isolate the issue")

    return recs


# ── Full-access tools ────────────────────────────────────────────────


class N8nListWorkflowsTool(Tool):
    """List all workflows in n8n."""

    @property
    def name(self) -> str:
        return "n8n_list_workflows"

    @property
    def description(self) -> str:
        return (
            "List all workflows in n8n with their IDs, names, active status, "
            "and node count. Use as a starting point to discover workflows."
        )

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call("GET", "/rest/workflows")
            resp.raise_for_status()
            raw = resp.json()
            wfs = raw.get("data", raw) if isinstance(raw, dict) else raw
            if isinstance(wfs, dict):
                wfs = wfs.get("data", [])
            if not isinstance(wfs, list):
                wfs = []

            if not wfs:
                return "No workflows found in n8n."

            lines = [f"Found {len(wfs)} workflow(s):\n"]
            for wf in wfs:
                wid = wf.get("id", "?")
                name = wf.get("name", "Untitled")
                active = "✓ active" if wf.get("active") else "✗ inactive"
                nodes = len(wf.get("nodes", []))
                lines.append(f"  [{wid}] {name} — {active} ({nodes} nodes)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing workflows: {e}"


class N8nGetWorkflowFullTool(Tool):
    """Get the complete workflow JSON including all node configs and credentials."""

    @property
    def name(self) -> str:
        return "n8n_get_workflow"

    @property
    def description(self) -> str:
        return (
            "Get the FULL workflow definition as JSON — all nodes with their complete "
            "parameters, connections, credentials references, and settings. "
            "Use when you need to inspect node configuration details, see what parameters "
            "are set, check credential bindings, or understand the full workflow structure."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The n8n workflow ID",
                },
            },
            "required": ["workflow_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        workflow_id = kwargs["workflow_id"]
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call("GET", f"/rest/workflows/{workflow_id}")
            resp.raise_for_status()
            raw = resp.json()
            wf = raw.get("data", raw) if isinstance(raw, dict) else raw

            # Return TOON-formatted compact summary for token efficiency
            nodes = wf.get("nodes", [])
            lines = [
                f"# workflow {workflow_id}",
                f"name: {wf.get('name', 'Untitled')}",
                f"active: {wf.get('active', False)}",
            ]

            # Nodes in TOON tabular format
            lines.append(f"\nnodes[{len(nodes)}]{{name,type,cred_ids,params}}:")
            for n in nodes:
                ntype = n.get("type", "?")
                nname = n.get("name", ntype)
                creds = n.get("credentials", {})
                params = n.get("parameters", {})
                cred_str = ";".join(f"{k}={v.get('id','?')}" for k, v in creds.items() if isinstance(v, dict)) or "-"
                key_params = {k: v for k, v in params.items() if v and k not in ("options", "additionalFields")}
                p_str = json.dumps(key_params, default=str, ensure_ascii=False) if key_params else "-"
                if len(p_str) > 150:
                    p_str = p_str[:150] + "..."
                lines.append(f"  {nname},{ntype},{cred_str},{p_str}")

            # Connections in TOON format
            conn_pairs = []
            for src_node, targets in wf.get("connections", {}).items():
                if isinstance(targets, dict):
                    for _key, target_list in targets.items():
                        if isinstance(target_list, list):
                            for group in target_list:
                                if isinstance(group, list):
                                    for t in group:
                                        conn_pairs.append(f"  {src_node},{t.get('node', '?')}")
            if conn_pairs:
                lines.append(f"\nconnections[{len(conn_pairs)}]{{src,dest}}:")
                lines.extend(conn_pairs)

            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching workflow {workflow_id}: {e}"


class N8nCreateWorkflowTool(Tool):
    """Create a brand new workflow in n8n."""

    @property
    def name(self) -> str:
        return "n8n_create_workflow"

    @property
    def description(self) -> str:
        return (
            "Create a new workflow in n8n. You can pass:\n"
            "- workflow_json='template:ID' or 'official:ID' (cached from template tools)\n"
            "- workflow_json='{...}' (raw JSON string)\n"
            "- OR name + nodes + connections separately."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "workflow_json": {
                    "type": "string",
                    "description": (
                        "Workflow JSON — either a cache reference like 'template:42' or "
                        "'official:123' (from template tools), or a raw JSON string."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "Workflow name (overrides name in workflow_json if both provided)",
                },
                "nodes": {
                    "type": "array",
                    "description": "Array of node objects (ignored if workflow_json provided)",
                },
                "connections": {
                    "type": "object",
                    "description": "Connections between nodes (ignored if workflow_json provided)",
                },
                "active": {
                    "type": "boolean",
                    "description": "Whether to activate immediately (default: false)",
                },
            },
            "required": [],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    @staticmethod
    def _parse_workflow_json(raw: str) -> dict:
        """Parse workflow JSON from a string, stripping markdown fences if present."""
        text = raw.strip()
        # Strip markdown code fences
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()
        return json.loads(text)

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_setup import n8n_api_call

        # Parse workflow_json if provided
        if "workflow_json" in kwargs and kwargs["workflow_json"]:
            raw_json = kwargs["workflow_json"].strip()

            # Check if it's a template cache reference (e.g. "template:42" or "official:123")
            if raw_json in _template_json_cache:
                raw_json = _template_json_cache.pop(raw_json)  # consume from cache

            try:
                wf = self._parse_workflow_json(raw_json)
            except (json.JSONDecodeError, ValueError) as e:
                return f"Failed to parse workflow_json: {e}"
            name = kwargs.get("name") or wf.get("name", "Untitled Workflow")
            nodes = wf.get("nodes", [])
            connections = wf.get("connections", {})
        elif "nodes" in kwargs and "connections" in kwargs:
            name = kwargs.get("name", "Untitled Workflow")
            nodes = kwargs["nodes"]
            connections = kwargs["connections"]
        else:
            return "Provide either workflow_json OR (name + nodes + connections)."

        payload = {
            "name": name,
            "nodes": nodes,
            "connections": connections,
            "active": kwargs.get("active", False),
            "settings": {"executionOrder": "v1"},
        }

        try:
            resp = await n8n_api_call("POST", "/rest/workflows", json_data=payload)
            resp.raise_for_status()
            raw = resp.json()
            wf = raw.get("data", raw) if isinstance(raw, dict) else raw
            return f"Created workflow '{wf.get('name')}' (ID: {wf.get('id')})"
        except Exception as e:
            return f"Failed to create workflow: {e}"


class N8nDeleteWorkflowTool(Tool):
    """Delete a workflow from n8n."""

    @property
    def name(self) -> str:
        return "n8n_delete_workflow"

    @property
    def description(self) -> str:
        return "Delete a workflow from n8n by its ID. This is permanent."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The workflow ID to delete",
                },
            },
            "required": ["workflow_id"],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        workflow_id = kwargs["workflow_id"]
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call("DELETE", f"/rest/workflows/{workflow_id}")
            resp.raise_for_status()
            return f"Deleted workflow {workflow_id}."
        except Exception as e:
            return f"Failed to delete workflow {workflow_id}: {e}"


class N8nActivateWorkflowTool(Tool):
    """Activate or deactivate an n8n workflow."""

    @property
    def name(self) -> str:
        return "n8n_activate_workflow"

    @property
    def description(self) -> str:
        return (
            "Activate or deactivate an n8n workflow. Active workflows run automatically "
            "based on their trigger. Use to turn workflows on or off."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The workflow ID",
                },
                "active": {
                    "type": "boolean",
                    "description": "True to activate, False to deactivate",
                },
            },
            "required": ["workflow_id", "active"],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        workflow_id = kwargs["workflow_id"]
        active = kwargs["active"]
        from src.server.n8n_setup import n8n_api_call

        try:
            if active:
                # Use the dedicated activate endpoint
                resp = await n8n_api_call("POST", f"/rest/workflows/{workflow_id}/activate")
            else:
                resp = await n8n_api_call("POST", f"/rest/workflows/{workflow_id}/deactivate")

            if resp.status_code == 400:
                # Activation failed — likely missing credentials
                error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                error_msg = error_data.get("message", resp.text[:200])
                return (
                    f"Failed to {'activate' if active else 'deactivate'} workflow: {error_msg}\n\n"
                    "HINT: This usually means the workflow has nodes with missing or invalid credentials. "
                    "Steps to fix:\n"
                    "1. Call n8n_get_workflow to inspect node credential references\n"
                    "2. Call n8n_list_credentials to see what's available\n"
                    "3. Create missing credentials with n8n_create_credential (capture the returned ID)\n"
                    "4. Update the workflow nodes to reference the correct credential IDs using n8n_update_workflow\n"
                    "5. Then retry n8n_activate_workflow"
                )

            resp.raise_for_status()
            state = "activated" if active else "deactivated"

            raw = resp.json()
            wf = raw.get("data", raw) if isinstance(raw, dict) else raw
            return f"Workflow '{wf.get('name', workflow_id)}' {state}."
        except Exception as e:
            return f"Failed to {'activate' if active else 'deactivate'} workflow: {e}"


class N8nExecuteWorkflowTool(Tool):
    """Trigger a manual execution of an n8n workflow."""

    @property
    def name(self) -> str:
        return "n8n_execute_workflow"

    @property
    def description(self) -> str:
        return (
            "Trigger a manual execution of an n8n workflow. Optionally pass input data. "
            "Use when the user wants to test-run a workflow."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The workflow ID to execute",
                },
                "data": {
                    "type": "object",
                    "description": "Optional input data to pass to the workflow trigger node",
                },
            },
            "required": ["workflow_id"],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        workflow_id = kwargs["workflow_id"]
        input_data = kwargs.get("data", {})
        from src.server.n8n_setup import n8n_api_call

        try:
            payload: dict[str, Any] = {"workflowId": workflow_id}
            if input_data:
                payload["data"] = input_data

            resp = await n8n_api_call("POST", "/rest/workflows/run", json_data=payload)
            resp.raise_for_status()
            raw = resp.json()
            result = raw.get("data", raw) if isinstance(raw, dict) else raw

            if isinstance(result, dict):
                exec_id = result.get("executionId", result.get("id", "?"))
                return f"Workflow execution started. Execution ID: {exec_id}. Use n8n_execution_detail to check results."
            return f"Workflow execution triggered: {json.dumps(result, default=str)[:500]}"
        except Exception as e:
            return f"Failed to execute workflow {workflow_id}: {e}"


class N8nGetCredentialDataTool(Tool):
    """Read the actual data/secrets stored in a credential."""

    @property
    def name(self) -> str:
        return "n8n_get_credential"

    @property
    def description(self) -> str:
        return (
            "Read the full credential data including secret values (API keys, tokens, etc.) "
            "for a specific credential by ID. Returns structured info including a JSON data "
            "field that can be passed to n8n_update_credential's data parameter."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "credential_id": {
                    "type": "string",
                    "description": "The credential ID to read",
                },
            },
            "required": ["credential_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        cred_id = kwargs["credential_id"]
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call("GET", f"/rest/credentials/{cred_id}")
            resp.raise_for_status()
            raw = resp.json()
            cred = raw.get("data", raw) if isinstance(raw, dict) else raw

            lines = [
                f"Credential: {cred.get('name', 'Untitled')}",
                f"ID: {cred_id}",
                f"Type: {cred.get('type', 'unknown')}",
                f"Created: {cred.get('createdAt', 'unknown')}",
                f"Updated: {cred.get('updatedAt', 'unknown')}",
            ]

            # Include the actual data if available
            cred_data = cred.get("data")
            if cred_data:
                lines.append(f"Data: {json.dumps(cred_data, default=str)}")
            else:
                # Try fetching with includeData parameter
                resp2 = await n8n_api_call(
                    "GET", f"/rest/credentials/{cred_id}",
                    params={"includeData": "true"},
                )
                if resp2.status_code == 200:
                    raw2 = resp2.json()
                    cred2 = raw2.get("data", raw2) if isinstance(raw2, dict) else raw2
                    d = cred2.get("data")
                    if d:
                        lines.append(f"Data: {json.dumps(d, default=str)}")
                    else:
                        lines.append("Data: [not available — n8n may encrypt at rest]")

            return "\n".join(lines)
        except Exception as e:
            return f"Error reading credential {cred_id}: {e}"


class N8nUpdateCredentialTool(Tool):
    """Update an existing credential's data."""

    @property
    def name(self) -> str:
        return "n8n_update_credential"

    @property
    def description(self) -> str:
        return (
            "Update an existing credential's name or secret data. Use when you need to "
            "change an API key, token, or other credential value. Can also rename the credential."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "credential_id": {
                    "type": "string",
                    "description": "The credential ID to update",
                },
                "name": {
                    "type": "string",
                    "description": "New display name (optional)",
                },
                "data": {
                    "type": "object",
                    "description": "New credential data (key-value pairs) — replaces existing data",
                },
            },
            "required": ["credential_id"],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        cred_id = kwargs["credential_id"]
        from src.server.n8n_setup import n8n_api_call

        try:
            # Fetch current credential
            resp = await n8n_api_call("GET", f"/rest/credentials/{cred_id}")
            resp.raise_for_status()
            raw = resp.json()
            cred = raw.get("data", raw) if isinstance(raw, dict) else raw

            # Build update payload
            payload: dict[str, Any] = {
                "name": kwargs.get("name", cred.get("name")),
                "type": cred.get("type"),
            }
            if "data" in kwargs:
                payload["data"] = kwargs["data"]

            update_resp = await n8n_api_call(
                "PATCH", f"/rest/credentials/{cred_id}", json_data=payload,
            )
            update_resp.raise_for_status()
            return f"Updated credential '{payload['name']}' (ID: {cred_id})"
        except Exception as e:
            return f"Failed to update credential {cred_id}: {e}"


class N8nGetSettingsTool(Tool):
    """Read n8n global settings."""

    @property
    def name(self) -> str:
        return "n8n_get_settings"

    @property
    def description(self) -> str:
        return (
            "Read n8n's current settings including version, license, defaults, "
            "and configuration. Use to check n8n version or global configuration."
        )

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_setup import n8n_api_call

        try:
            resp = await n8n_api_call("GET", "/rest/settings")
            resp.raise_for_status()
            raw = resp.json()
            settings = raw.get("data", raw) if isinstance(raw, dict) else raw

            if isinstance(settings, dict):
                # Extract key settings
                lines = ["n8n Settings:\n"]
                for key in (
                    "n8nVersion", "versionCli", "defaultLocale", "isDocker",
                    "databaseType", "executionMode", "pushBackend", "communityNodesEnabled",
                    "deployment", "mfaEnabled", "publicApi",
                ):
                    if key in settings:
                        lines.append(f"  {key}: {settings[key]}")

                # Show all settings on request (compact)
                lines.append(f"\nFull settings keys: {sorted(settings.keys())}")
                return "\n".join(lines)
            return json.dumps(settings, indent=2, default=str)[:3000]
        except Exception as e:
            return f"Error reading settings: {e}"


class N8nGetNodeTypesTool(Tool):
    """List available n8n node types."""

    @property
    def name(self) -> str:
        return "n8n_list_node_types"

    @property
    def description(self) -> str:
        return (
            "List all available node types installed in n8n. Use to check what integrations "
            "are available or to find the correct node type name for creating workflows."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Optional search term to filter nodes (e.g. 'telegram', 'http', 'slack')",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_manager import get_available_nodes

        search = kwargs.get("search", "").lower()

        try:
            nodes = await get_available_nodes()
            if search:
                nodes = [
                    n for n in nodes
                    if search in n.get("name", "").lower()
                    or search in n.get("displayName", "").lower()
                ]

            if not nodes:
                return f"No node types found{f' matching \"{search}\"' if search else ''}."

            # TOON tabular format
            show = nodes[:50]
            lines = [f"node_types[{len(nodes)}]{{name,display}}:"]
            for n in show:
                lines.append(f"  {n.get('name', '?')},{n.get('displayName', '')}")
            if len(nodes) > 50:
                lines.append(f"  # ...{len(nodes) - 50} more — use search param to filter")

            return "\n".join(lines)
        except Exception as e:
            return f"Error listing node types: {e}"


# ── Template tools ───────────────────────────────────────────────────


class N8nSearchTemplatesTool(Tool):
    """Search the local library of 290+ ready-to-use n8n workflow templates."""

    @property
    def name(self) -> str:
        return "n8n_search_templates"

    @property
    def description(self) -> str:
        return (
            "Search the built-in library of ~290 community n8n workflow templates "
            "by keyword, category, or node type. Returns matching templates with "
            "name, category, node types, and an ID you can use with "
            "n8n_get_template_detail to get the full workflow JSON."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords (e.g. 'telegram bot', 'email openai', 'slack webhook').",
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter by category (e.g. 'WhatsApp', 'OpenAI and LLMs', 'Gmail and Email Automation').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10).",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_templates import search_templates, get_categories

        query = kwargs.get("query", "")
        category = kwargs.get("category")
        limit = int(kwargs.get("limit", 10))

        results = search_templates(query, category=category, limit=limit)
        if not results:
            cats = get_categories()
            return (
                f"No templates found for '{query}'"
                + (f" in category '{category}'" if category else "")
                + f".\nAvailable categories: {', '.join(cats)}"
            )

        # TOON tabular format
        lines = [f"templates[{len(results)}]{{id,name,category,nodes,types}}:"]
        for t in results:
            types_short = ",".join(nt.split(".")[-1] for nt in t["node_types"][:5])
            lines.append(f"  {t['id']},{t['name']},{t['category']},{t['node_count']},{types_short}")

        lines.append("\n# use n8n_get_template_detail(template_id) to deploy")
        return "\n".join(lines)


class N8nGetTemplateDetailTool(Tool):
    """Get the full workflow JSON for a specific template from the local library."""

    @property
    def name(self) -> str:
        return "n8n_get_template_detail"

    @property
    def description(self) -> str:
        return (
            "Get an n8n workflow template by its numeric ID. Returns a compact summary "
            "and caches the full JSON. To deploy, call n8n_create_workflow with "
            "workflow_json='template:ID' (e.g. workflow_json='template:42')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "integer",
                    "description": "The numeric template ID from search results.",
                },
            },
            "required": ["template_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_templates import get_template_by_id, strip_credentials

        tid = int(kwargs.get("template_id", -1))
        result = get_template_by_id(tid)
        if not result:
            return f"Template with ID {tid} not found."

        workflow = strip_credentials(result["workflow"])
        # Cache full JSON for later use by n8n_create_workflow
        cache_key = f"template:{tid}"
        _template_json_cache[cache_key] = json.dumps(workflow)

        # Return compact summary instead of full JSON
        nodes = workflow.get("nodes", [])
        name = workflow.get("name", "Untitled")
        node_types = [n.get("type", "?").split(".")[-1] for n in nodes]
        creds_needed = set()
        for n in nodes:
            for ct in (n.get("credentials") or {}):
                creds_needed.add(ct)

        lines = [
            f"Template {tid}: {name}",
            f"Nodes ({len(nodes)}): {', '.join(node_types[:15])}{'...' if len(nodes) > 15 else ''}",
        ]
        if creds_needed:
            lines.append(f"Credentials needed: {', '.join(creds_needed)}")
        lines.append(f"\nTo deploy: n8n_create_workflow(workflow_json='{cache_key}')")
        return "\n".join(lines)


class N8nSearchOfficialTemplatesTool(Tool):
    """Search the official n8n.io template gallery online."""

    @property
    def name(self) -> str:
        return "n8n_search_official"

    @property
    def description(self) -> str:
        return (
            "Search the official n8n.io template gallery (thousands of templates) "
            "when the local library doesn't have what you need. Requires internet. "
            "Returns template IDs that can be fetched with n8n_fetch_official_template."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the official gallery.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10).",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_templates import search_official_templates

        query = kwargs.get("query", "")
        limit = int(kwargs.get("limit", 10))

        results = await search_official_templates(query, limit=limit)
        if not results:
            return f"No official templates found for '{query}'."

        # TOON tabular format
        lines = [f"official_templates[{len(results)}]{{id,name,nodes,by}}:"]
        for t in results:
            nodes_short = ",".join(t["nodes"][:4])
            lines.append(f"  {t['id']},{t['name']},{nodes_short},{t['user']}")

        lines.append("\n# use n8n_fetch_official_template(template_id) to deploy")
        return "\n".join(lines)


class N8nFetchOfficialTemplateTool(Tool):
    """Fetch the full workflow JSON of an official n8n.io template."""

    @property
    def name(self) -> str:
        return "n8n_fetch_official_template"

    @property
    def description(self) -> str:
        return (
            "Fetch an official n8n.io template by its numeric ID. Returns a compact "
            "summary and caches the full JSON. To deploy, call n8n_create_workflow "
            "with workflow_json='official:ID' (e.g. workflow_json='official:123')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "integer",
                    "description": "The official template ID from n8n_search_official results.",
                },
            },
            "required": ["template_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from src.server.n8n_templates import fetch_official_template, strip_credentials

        tid = int(kwargs.get("template_id", -1))
        workflow = await fetch_official_template(tid)
        if not workflow:
            return f"Failed to fetch official template {tid}. It may not exist or the API is unreachable."

        workflow = strip_credentials(workflow)
        # Cache full JSON for later use by n8n_create_workflow
        cache_key = f"official:{tid}"
        _template_json_cache[cache_key] = json.dumps(workflow)

        # Return compact summary
        nodes = workflow.get("nodes", [])
        name = workflow.get("name", "Untitled")
        node_types = [n.get("type", "?").split(".")[-1] for n in nodes]
        creds_needed = set()
        for n in nodes:
            for ct in (n.get("credentials") or {}):
                creds_needed.add(ct)

        lines = [
            f"Official Template {tid}: {name}",
            f"Nodes ({len(nodes)}): {', '.join(node_types[:15])}{'...' if len(nodes) > 15 else ''}",
        ]
        if creds_needed:
            lines.append(f"Credentials needed: {', '.join(creds_needed)}")
        lines.append(f"\nTo deploy: n8n_create_workflow(workflow_json='{cache_key}')")
        return "\n".join(lines)