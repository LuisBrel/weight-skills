"""Model Context Protocol server exposing cartridges as agent tools.

Runs over stdio using newline-delimited JSON-RPC 2.0 — no third-party MCP
package required, so the server works inside any harness that speaks MCP
(Claude Code, Cursor, KiroCrew).

Tools
-----
``ws_list_cartridges``  list discoverable cartridges with geometry metadata
``ws_mount_skill``      mount a skill on the loaded model
``ws_unmount_skill``    unmount and verify restoration

Run with ``weight-skills-mcp --model /path/to/model --cartridges ./cartridges``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2024-11-05"


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required},
    }


TOOLS = [
    _tool(
        "ws_list_cartridges",
        "List weight-skill cartridges available on this machine.",
        {},
        [],
    ),
    _tool(
        "ws_mount_skill",
        "Mount a weight skill onto the local model with zero prompt tokens.",
        {
            "skill_id": {"type": "string", "description": "Cartridge id, e.g. 'aws-cso'."},
            "scale": {"type": "number", "description": "Strength multiplier (default 1.0)."},
        },
        ["skill_id"],
    ),
    _tool(
        "ws_unmount_skill",
        "Unmount all skills and verify bit-identical restoration of base weights.",
        {},
        [],
    ),
]


class CartridgeMCPServer:
    """Stateful MCP handler around an optional live engine."""

    def __init__(self, engine: Any | None = None, cartridge_dir: str = "cartridges") -> None:
        self.engine = engine
        self.cartridge_dir = Path(cartridge_dir)
        self.mounted: list[str] = []

    # ------------------------------------------------------------------ tools
    def _discover(self) -> list[dict[str, Any]]:
        from ..core.cartridge import Cartridge

        out = []
        if self.cartridge_dir.is_dir():
            for path in sorted(self.cartridge_dir.rglob("*.safetensors")):
                try:
                    cart = Cartridge.load(path)
                except Exception:  # noqa: BLE001
                    continue
                out.append(
                    {
                        "id": cart.skill_id,
                        "version": cart.version,
                        "rank": cart.rank,
                        "hidden_size": cart.hidden_size,
                        "layers": list(cart.layer_indices),
                        "nbytes": cart.nbytes,
                        "path": str(path),
                    }
                )
        return out

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "ws_list_cartridges":
            data = self._discover()
            return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}], "isError": False}
        if name == "ws_mount_skill":
            skill_id = str(arguments.get("skill_id", ""))
            scale = float(arguments.get("scale", 1.0))
            if self.engine is None:
                return {
                    "content": [{"type": "text", "text": "no model loaded; run with --model"}],
                    "isError": True,
                }
            try:
                report = self.engine.mount([skill_id], scale=scale)
            except KeyError as exc:
                return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
            self.mounted = list(report.get("skills", [skill_id]))
            return {"content": [{"type": "text", "text": json.dumps(report, indent=2)}], "isError": False}
        if name == "ws_unmount_skill":
            if self.engine is None:
                return {"content": [{"type": "text", "text": "no model loaded"}], "isError": True}
            report = self.engine.unmount()
            self.mounted = []
            return {"content": [{"type": "text", "text": json.dumps(report, indent=2)}], "isError": False}
        return {"content": [{"type": "text", "text": f"unknown tool {name!r}"}], "isError": True}

    # ---------------------------------------------------------------- jsonrpc
    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "weight-skills", "version": "0.1.0"},
            }
        elif method in ("notifications/initialized", "initialized"):
            return None
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = request.get("params") or {}
            result = self.call_tool(str(params.get("name", "")), dict(params.get("arguments") or {}))
        elif method == "ping":
            result = {}
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    # ------------------------------------------------------------------ stdio
    def serve(self, stdin: Any = None, stdout: Any = None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                stdout.write(
                    json.dumps(
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
                    )
                    + "\n"
                )
                stdout.flush()
                continue
            response = self.handle(request)
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="weight-skills MCP server (stdio)")
    parser.add_argument("--model", default=None, help="local model path (optional)")
    parser.add_argument("--cartridges", default="cartridges")
    args = parser.parse_args(argv)

    engine = None
    if args.model:
        from ..proxy.engine import MLXEngine

        engine = MLXEngine(model_path=args.model, cartridge_dir=args.cartridges)
    CartridgeMCPServer(engine=engine, cartridge_dir=args.cartridges).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
