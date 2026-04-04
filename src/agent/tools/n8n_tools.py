"""Agent tools for n8n workflow management and error diagnosis."""

from __future__ import annotations

import json
from typing import Any

from src.agent.base import Tool


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
            node_summary = ", ".join(n.get("name", n.get("type", "?")) for n in nodes[:10])
            if len(nodes) > 10:
                node_summary += f" ... (+{len(nodes) - 10} more)"

            lines = [
                f"Workflow: {data.get('name', 'Untitled')}",
                f"ID: {workflow_id}",
                f"Active: {data.get('active', False)}",
                f"Nodes ({len(nodes)}): {node_summary}",
                f"Created: {data.get('createdAt', 'unknown')}",
                f"Updated: {data.get('updatedAt', 'unknown')}",
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

            lines = [f"Found {len(execs)} execution(s):\n"]
            for ex in execs:
                eid = ex.get("id", "?")
                finished = ex.get("finished", None)
                status_str = "✓ success" if finished else "✗ error" if finished is False else "⋯ running"
                started = ex.get("startedAt", "?")
                stopped = ex.get("stoppedAt", "")
                mode = ex.get("mode", "")
                lines.append(f"  [{eid}] {status_str} | started: {started} | stopped: {stopped} | mode: {mode}")

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

            data = detail.get("data", detail)
            result_data = data.get("data", {}).get("resultData", data.get("resultData", {}))
            run_data = result_data.get("runData", {})
            error = result_data.get("error", data.get("error"))

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
