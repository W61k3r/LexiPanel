#!/usr/bin/env python3
"""
Hermes Agent (Nous Research) on LexiPanel (added 2026-09-25): a llama.cpp instance as its model,
the panel's MCP server as its tools. Checks what Hermes needs of the instance and writes the two
blocks of ~/.hermes/config.yaml. Hermes' own requirements (its docs, providers / llama.cpp):
  * llama-server started with --jinja, or it ignores the `tools` parameter (LexiPanel always
    passes it; a hand-started server may not),
  * at least 64,000 tokens of context for the conversation (per slot, so CTX / PARALLEL),
  * a chat template that handles tools.
The MCP block is read-only by default (LEXIPANEL_MCP_READONLY=1): an agent that reads status,
logs and measurements needs no start/stop/restart tools until you decide it does.
"""
import os
from pathlib import Path

P = None
MIN_CTX = 64000


def bind(panel_module):
    global P
    P = panel_module


def _q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def setup(inst, host_header=None):
    with P.using_instance(inst):
        p = P.load_params()
        pid = P.server_pid()
        argv = P.live_cmdline_args() if pid else []
        props = P.api_get("/props", timeout=3) if pid else None
    props = props if isinstance(props, dict) else {}
    running = bool(pid)
    live_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    slot_ctx = P.workload.slot_ctx_of(p, live_ctx)
    port = P._argv_get(argv, ("--port",)) if argv else None
    port = port or p.get("PORT")
    bind_host = (P._argv_get(argv, ("--host",)) if argv else None) or str(p.get("HOST") or "127.0.0.1")
    lan = bind_host not in ("127.0.0.1", "localhost", "::1")
    box = (host_header or "").rsplit(":", 1)[0].strip("[]") or "127.0.0.1"
    base = f"http://{box if lan else '127.0.0.1'}:{port}/v1"
    alias = P._argv_get(argv, ("-a", "--alias")) if argv else None
    model = alias or os.path.basename(str(p.get("MODEL") or "")) or "local"
    tmpl = str(props.get("chat_template") or "")
    checks = []

    def chk(cid, ok, text):
        checks.append(dict(id=cid, ok=ok, text=text))

    chk("running", running, "the instance is running" if running else
        "the instance is not running: start it (the checks below need the live server)")
    if running:
        chk("jinja", "--jinja" in argv, "started with --jinja: Hermes' tool calls reach the model" if "--jinja" in argv
            else "started WITHOUT --jinja: llama-server ignores Hermes' tools; restart it from LexiPanel")
        chk("template", ("tools" in tmpl) if tmpl else None,
            "the chat template handles tools" if "tools" in tmpl else
            "the chat template does not mention tools: tool calls will be unreliable (Templates tab)"
            if tmpl else "the server did not report its chat template")
    chk("context", (slot_ctx or 0) >= MIN_CTX,
        f"{slot_ctx:,} tokens per conversation (Hermes needs {MIN_CTX:,})" if slot_ctx else
        "context per conversation unknown")
    chk("reachable", True if lan else None,
        f"listening on {bind_host}: reachable at {base}" if lan else
        f"listening on {bind_host} only: Hermes must run on this machine (HOST=0.0.0.0 to reach it from another)")
    auth = bool(str(p.get("API_KEY") or ""))
    home = str(P.HOME)
    mcp_path = f"{Path(P.PANEL).as_posix()}/mcp_server.py"
    user = os.path.basename(home.rstrip("/")) or "admin"
    ctx_line = f"  context_length: {min(slot_ctx, 1_000_000)}\n" if slot_ctx else ""
    yaml = ("# ~/.hermes/config.yaml  (then /reload-mcp, or restart hermes)\n"
            "model:\n  provider: custom\n"
            f"  base_url: {base}\n  default: {_q(model)}\n{ctx_line}"
            + ("  api_key: ${env:LEXIPANEL_LLM_KEY}   # this instance's API key (Parameters > API key)\n" if auth else "")
            + "\nmcp_servers:\n"
            "  lexipanel:                     # Hermes on THIS machine\n"
            "    command: python3\n"
            f"    args: [{_q(mcp_path)}]\n"
            "    env:\n      LEXIPANEL_MCP_READONLY: \"1\"   # remove to let the agent start / stop / restart\n"
            "    timeout: 180\n"
            "  # lexipanel:                   # Hermes on ANOTHER machine: the same server over ssh\n"
            "  #   command: ssh\n"
            f"  #   args: [{_q(user + '@' + box)}, \"LEXIPANEL_MCP_READONLY=1\", \"python3\", {_q(mcp_path)}]\n"
            "  #   timeout: 180\n")
    return dict(ready=all(c["ok"] for c in checks), checks=checks, yaml=yaml, base_url=base, model=model,
                slot_ctx=slot_ctx, min_ctx=MIN_CTX, running=running)
