#!/usr/bin/env python3
"""
llama.cpp's built-in web UI, per instance (added 2026-09-24).

  defaults   The UI settings a browser starts from (--ui-config-file): system message,
             reasoning display, extra request JSON, MCP servers, theme. The panel writes
             <instance dir>/webui-config.json and points UI_CONFIG_FILE at it.
             Sampling is NOT set here on purpose: the UI already takes temperature, top-p,
             top-k and min-p from the server's own settings (/props), i.e. from the
             Parameters tab and the optimizer, unless a browser user overrode them. A value
             here would override those for new browsers.
             llama.cpp applies these defaults on a browser's FIRST visit only; a browser that
             already used the UI keeps its own until "Reset to default" in the UI's settings.
             Everything here is readable by anyone who can reach the server (/props), so no
             secrets (no MCP headers, no API key).
  links      direct (http://host:port/) and behind the panel's login (https://panel/ui/<id>/).
  login      a Caddy route per instance, path-stripped, so llama-server needs no --api-prefix
             and API clients keep using http://host:port/v1. The panel writes only a route
             snippet (no password hash); webui/apply-caddy-webui.sh, run with sudo, checks it
             line by line, inserts it into /etc/caddy/Caddyfile, validates and restarts Caddy
             (restart, never reload: admin is off here), and rolls back on any failure.
  mcp proxy  --ui-mcp-proxy (UI_MCP_PROXY): llama.cpp calls it experimental and unsafe on
             untrusted networks. Off unless set.
"""
import json, os, re, subprocess
from pathlib import Path

P = None
CADDYFILE = Path("/etc/caddy/Caddyfile")
HERE = Path(__file__).resolve().parent / "webui"
BEGIN, END = "# BEGIN LexiPanel web UIs", "# END LexiPanel web UIs"
MAX_JSON = 64 * 1024

# key: (type, label, help)  - the curated form. Types: text, bool, int, choice:<a|b|c>
FIELDS = {
    "systemMessage": ("text", "System message", "Starting instruction for every new conversation."),
    "theme": ("choice:system|light|dark", "Theme", ""),
    "showThoughtInProgress": ("bool", "Show reasoning while it streams", ""),
    "excludeReasoningFromContext": ("bool", "Leave earlier reasoning out of the context",
                                    "Saves context on long chats with a reasoning model."),
    "disableReasoningParsing": ("bool", "Show reasoning as plain text", ""),
    "showMessageStats": ("bool", "Show tokens/s under each reply", ""),
    "sendOnEnter": ("bool", "Enter sends the message", ""),
    "pasteLongTextToFileLen": ("int", "Paste longer than this becomes a file (0 = never)", ""),
    "agenticMaxTurns": ("int", "Max tool-call turns per reply", ""),
    "customJson": ("json", "Extra fields for every request (JSON)",
                   'Merged into each request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}'),
}
# every key this UI generation understands; sampling keys are refused (see above)
KNOWN = set(FIELDS) | {
    "alwaysShowSidebarOnDesktop", "alwaysShowToolCallContent", "autoMicOnEmpty", "conversationTabs",
    "copyTextAttachmentsAsPlainText", "customCss", "pyInterpreterEnabled", "disableAutoScroll",
    "enableContinueGeneration", "fullHeightCodeBlocks", "jsSandboxEnabled", "maxImageMPixels",
    "mcpRequestTimeoutSeconds", "mcpServers", "mentionSearchMaxDepth", "pdfAsImage",
    "preEncodeConversation", "renderThinkingAsMarkdown", "renderUserContentAsMarkdown",
    "showAgenticTurnStats", "showBuildVersion", "showFullPathInMentions", "showModelOrgNameInTrigger",
    "showModelQuantization", "showModelTags", "showRawModelNames", "showRawOutputSwitch",
    "showSystemMessage", "symbolicMathEnabled", "titleGenerationPrompt", "titleGenerationUseFirstLine",
    "titleGenerationUseLLM"}
SAMPLING = {"temperature", "top_k", "top_p", "min_p", "typ_p", "samplers", "dynatemp_range",
            "dynatemp_exponent", "xtc_probability", "xtc_threshold", "repeat_last_n", "repeat_penalty",
            "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base", "dry_allowed_length",
            "dry_penalty_last_n", "max_tokens", "backend_sampling"}
SECRET = {"apiKey"}
_URL = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?(/[\w.\-/~%?=&:+]*)?$")
_SNIP_LINE = re.compile(
    r"^\s*(#.*"
    r"|redir /ui/[A-Za-z0-9_-]+ /ui/[A-Za-z0-9_-]+/"
    r"|handle_path /ui/[A-Za-z0-9_-]+/\* \{"
    r"|reverse_proxy 127\.0\.0\.1:\d{2,5}"
    r"|\})\s*$")


def bind(panel_module):
    global P
    P = panel_module


def _cfg_file(inst=None):
    inst = inst or P.INST()
    return Path(inst["dir"]) / "webui-config.json"


# ============================================================================
# defaults (--ui-config-file)
# ============================================================================
def _check(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("UI defaults are a JSON object")
    out, notes = {}, []
    for k, v in cfg.items():
        if k in SECRET:
            raise ValueError(f"{k}: the defaults are served to anyone who can reach the server; "
                             "never put a key in them")
        if k in SAMPLING:
            raise ValueError(f"{k}: sampling comes from the server's own settings (Parameters tab), "
                             "which the UI already follows; setting it here would override them")
        if k not in KNOWN:
            raise ValueError(f"{k}: not a setting this web UI knows")
        kind = FIELDS.get(k, ("any",))[0]
        if v is None or v == "":
            continue
        if kind == "bool" and not isinstance(v, bool):
            raise ValueError(f"{k}: true or false")
        if kind == "int" and (not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 1_000_000):
            raise ValueError(f"{k}: a whole number")
        if kind.startswith("choice:") and v not in kind[7:].split("|"):
            raise ValueError(f"{k}: one of {kind[7:].replace('|', ', ')}")
        if kind == "text" and (not isinstance(v, str) or len(v) > 20000):
            raise ValueError(f"{k}: text up to 20000 characters")
        if kind == "json":
            if isinstance(v, dict):
                v = json.dumps(v)
            try:
                if not isinstance(json.loads(v), dict):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"{k}: a JSON object")
            if re.search(r'"(api[_-]?key|authorization|token|password)"', v, re.I):
                raise ValueError(f"{k}: looks like it carries a secret; the defaults are public")
        if k == "mcpServers":
            v = _check_mcp(v)
        out[k] = v
    return out, notes


def _check_mcp(v):
    try:
        servers = json.loads(v) if isinstance(v, str) else v
    except ValueError:
        raise ValueError("mcpServers: a JSON list")
    if not isinstance(servers, list) or len(servers) > 20:
        raise ValueError("mcpServers: a list of up to 20 servers")
    clean = []
    for s in servers:
        if not isinstance(s, dict):
            raise ValueError("mcpServers: each server is an object")
        if s.get("headers"):
            raise ValueError("mcpServers: no headers here - they would be public; add them in the "
                             "browser's own MCP settings")
        url = str(s.get("url") or "").strip()
        if not _URL.match(url):
            raise ValueError(f"mcpServers: {url[:80]!r} is not an http(s) URL")
        clean.append(dict(name=str(s.get("name") or "")[:60] or url, url=url,
                          enabled=bool(s.get("enabled", True)), useProxy=bool(s.get("useProxy"))))
    return json.dumps(clean)


def get_defaults():
    f = _cfg_file()
    try:
        cfg = json.loads(f.read_text())
    except (OSError, ValueError):
        cfg = {}
    mcp = []
    try:
        mcp = json.loads(cfg.get("mcpServers") or "[]")
    except ValueError:
        pass
    return dict(file=str(f), exists=f.exists(), config=cfg, mcp=mcp, fields=FIELDS,
                param=str(P.load_params().get("UI_CONFIG_FILE") or ""))


def save_defaults(body):
    cfg = body.get("config")
    if isinstance(cfg, str):
        if len(cfg) > MAX_JSON:
            raise ValueError("too large")
        try:
            cfg = json.loads(cfg or "{}")
        except ValueError as e:
            raise ValueError(f"not JSON: {e}")
    if "mcp" in body:
        cfg = dict(cfg or {}, mcpServers=json.dumps(body.get("mcp") or []))
    clean, _notes = _check(cfg or {})
    f = _cfg_file()
    params = P.load_params()
    if clean:
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(clean, indent=1) + "\n")
        os.replace(tmp, f)
        want = str(f)
    else:
        f.unlink(missing_ok=True)
        want = ""
    if str(params.get("UI_CONFIG_FILE") or "") != want:
        P.save_params({"UI_CONFIG_FILE": want})
    return dict(get_defaults(), saved=True,
                note="Takes effect when this server next starts. Browsers that already used the UI "
                     "keep their own settings until 'Reset to default' in the UI's settings.")


# ============================================================================
# status: links, what the server serves, the Caddy route
# ============================================================================
def _probe(host, port):
    """Is the web UI up on this port, and what does the server hand it?"""
    import urllib.request
    h = "127.0.0.1" if host in ("", "0.0.0.0", None) else host
    out = dict(serving=None, sampling=None, ui_settings=None, build=None)
    try:
        req = urllib.request.Request(f"http://{h}:{port}/", headers={"Accept": "text/html",
                                                                     "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=3) as r:
            out["serving"] = r.status == 200 and "html" in (r.headers.get("Content-Type") or "")
    except Exception as e:
        out["serving"] = False if "404" in str(e) else None
    try:
        with urllib.request.urlopen(f"http://{h}:{port}/props", timeout=3) as r:
            d = json.loads(r.read())
        p = ((d.get("default_generation_settings") or {}).get("params") or {})
        out["sampling"] = {k: (round(p[k], 3) if isinstance(p.get(k), float) else p.get(k))
                           for k in ("temperature", "top_p", "top_k", "min_p") if k in p}
        out["ui_settings"] = d.get("ui_settings")
        out["build"] = d.get("build_info")
    except Exception:
        pass
    return out


def routes():
    """The Caddy routes the panel wants: one per llama.cpp instance whose UI is not off."""
    out = []
    for rec in P.list_instances():
        if rec.get("error") or (rec.get("engine") or "llama.cpp") != "llama.cpp" or not rec.get("port"):
            continue
        with P.using_instance(P.get_instance(rec["id"])):
            p = P.load_params()
        if str(p.get("WEBUI") or "") == "off" or not re.fullmatch(r"[A-Za-z0-9_-]+", rec["id"]):
            continue
        out.append(dict(id=rec["id"], port=int(rec["port"]), api_key=bool(p.get("API_KEY"))))
    return out


def snippet(rs=None):
    rs = routes() if rs is None else rs
    lines = [f"    {BEGIN} (written by the panel; change it there, then re-run the apply script)"]
    for r in rs:
        lines += [f"    redir /ui/{r['id']} /ui/{r['id']}/",
                  f"    handle_path /ui/{r['id']}/* {{",
                  f"        reverse_proxy 127.0.0.1:{r['port']}",
                  "    }"]
    lines.append(f"    {END}")
    return "\n".join(lines) + "\n"


def _live_block():
    try:
        txt = CADDYFILE.read_text()
    except OSError:
        return None, False
    m = re.search(re.escape(BEGIN) + r".*?" + re.escape(END), txt, re.S)
    return (m.group(0) if m else ""), True


def _norm(block):
    return [l.strip() for l in block.splitlines() if l.strip() and not l.strip().startswith("#")]


def caddy_state():
    want = snippet()
    live, readable = _live_block()
    snip_file = HERE / "caddy-webui.snippet"
    if not readable:
        state = "unreadable"
    elif not live:
        state = "not installed"
    elif _norm(live) == _norm(want):
        state = "installed"
    else:
        state = "out of date"
    return dict(state=state, snippet=want, file=str(snip_file),
                apply_cmd=f"sudo bash {HERE / 'apply-caddy-webui.sh'}",
                remove_cmd=f"sudo bash {HERE / 'apply-caddy-webui.sh'} --remove",
                routes=[r["id"] for r in routes()])


def write_snippet(body=None):
    """Write the snippet the apply script reads. No secrets in it."""
    HERE.mkdir(exist_ok=True)
    s = snippet()
    for line in s.splitlines():
        if not _SNIP_LINE.match(line):
            raise ValueError(f"refusing to write an unexpected line: {line!r}")
    tmp = HERE / "caddy-webui.snippet.tmp"
    tmp.write_text(s)
    os.replace(tmp, HERE / "caddy-webui.snippet")
    return caddy_state()


def status():
    inst = P.INST()
    p = P.load_params()
    port = p.get("PORT")
    host = "0.0.0.0" if inst.get("legacy") else str(p.get("HOST") or "127.0.0.1")
    probe = _probe(host, port) if port else {}
    legacy_ignored = set(P.legacy_ignored_keys()) if inst.get("legacy") else set()
    return dict(instance=inst["id"], port=port, host=host, lan=host not in ("127.0.0.1", "localhost"),
                webui=str(p.get("WEBUI") or ""), mcp_proxy=str(p.get("UI_MCP_PROXY") or ""),
                api_key=bool(p.get("API_KEY")),
                launcher_ignores=sorted({"WEBUI", "UI_CONFIG_FILE", "UI_MCP_PROXY"} & legacy_ignored),
                probe=probe, defaults=get_defaults(), caddy=caddy_state())


def set_flags(body):
    """WEBUI and UI_MCP_PROXY for this instance: '', 'on' or 'off'."""
    changed = {}
    for k in ("WEBUI", "UI_MCP_PROXY"):
        if k in body:
            v = str(body.get(k) or "")
            if v not in ("", "on", "off"):
                raise ValueError(f"{k}: '', on or off")
            changed[k] = v
    if changed:
        P.save_params(changed)
    return status()
