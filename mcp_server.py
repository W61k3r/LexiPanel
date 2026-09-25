#!/usr/bin/env python3
"""
LexiPanel MCP server: the panel's JSON API as Model Context Protocol tools, so an AI
client can see and drive this box (rewritten 2026-09-25, the first version
could not be reached and called functions that do not exist).

One tool table, two ways in:
  stdio   python3 mcp_server.py
          newline-delimited JSON-RPC on stdin/stdout, for clients that start a command
          (Claude Code, Claude Desktop, ...). It calls the panel at LEXIPANEL_URL
          (default http://127.0.0.1:8090), so run it on the box itself or through ssh:
            claude mcp add lexipanel -- ssh admin@box python3 /home/admin/panel/mcp_server.py
          To go through Caddy instead: LEXIPANEL_URL=https://box, LEXIPANEL_USER and
          LEXIPANEL_PASSWORD (the panel login), LEXIPANEL_CA=<Caddy root.crt> to trust its CA.
  HTTP    POST /api/mcp on the panel: MCP's Streamable HTTP transport, JSON responses.
          Behind Caddy it needs the panel login like every other route:
            claude mcp add --transport http lexipanel https://box/api/mcp \\
                --header "Authorization: Basic $(printf admin:PASSWORD | base64)"
          GET /api/mcp/tools and POST /api/mcp/call {"name","arguments"} are the same tools
          as plain JSON, for scripts.

Every tool is one or two calls to documented panel routes (API.md), so the panel's own
validation, guard rails and audit log apply unchanged. LEXIPANEL_MCP_READONLY=1 hides every
tool that changes something. There is deliberately no file-write tool: models, templates and
settings have their own validated routes, and a free write into the home folder is one prompt
injection away from ~/.bashrc. Stdlib only.
"""
import threading
import base64, json, os, re, ssl, sys, urllib.error, urllib.parse, urllib.request

VERSION = "1.0.0"
_caller = threading.local()          # in the panel: the user a request came from (multi-user mode)
PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = ("LexiPanel runs local AI servers (llama.cpp, stable-diffusion.cpp, audio.cpp, Camelid) "
                "on this box's GPUs. Most tools take `instance` (default 'main'; list_instances names "
                "them). Read before you change: get_status, get_params and launch_plan show what a "
                "start would do; estimate_memory checks a change fits before set_params saves it. "
                "Starts can be refused by the panel's safety rails; the refusal says why.")
P = None                      # the panel module when served in-process
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


def bind(panel_module):
    global P
    P = panel_module


# ============================================================================
# talking to the panel
# ============================================================================
class ToolError(Exception):
    pass


def _base():
    if P is not None:
        host = P.BIND if P.BIND not in ("0.0.0.0", "::", "") else "127.0.0.1"
        return f"http://{host}:{P.PORT}"
    return (os.environ.get("LEXIPANEL_URL") or "http://127.0.0.1:8090").rstrip("/")


def _ctx():
    ca = os.environ.get("LEXIPANEL_CA")
    return ssl.create_default_context(cafile=ca) if ca else None


def api(method, path, query=None, body=None, raw=False, timeout=120):
    """One panel call. Returns parsed JSON (or bytes with raw=True); a panel error
    ({"error": ...} or HTTP >= 400) becomes a ToolError with the panel's own words."""
    q = {k: v for k, v in (query or {}).items() if v is not None and v != ""}
    url = _base() + path + ("?" + urllib.parse.urlencode(q, doseq=True) if q else "")
    data = json.dumps(body if body is not None else {}).encode() if method == "POST" else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    who = getattr(_caller, "who", None)
    if P is not None and who and getattr(P, "auth", None):            # in the panel: act as the caller
        req.add_header("X-LexiPanel-Internal", f"{P.auth.INTERNAL}:{who}")
    if P is None and os.environ.get("LEXIPANEL_API_KEY"):
        req.add_header("Authorization", "Bearer " + os.environ["LEXIPANEL_API_KEY"])
    user, pw = os.environ.get("LEXIPANEL_USER"), os.environ.get("LEXIPANEL_PASSWORD")
    if P is None and user and pw:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
            payload = r.read()
    except urllib.error.HTTPError as e:
        txt = e.read()[:2000].decode(errors="replace")
        try:
            txt = json.loads(txt).get("error") or txt
        except (ValueError, AttributeError):
            pass
        raise ToolError(f"panel said HTTP {e.code}: {txt}")
    except (urllib.error.URLError, OSError) as e:
        raise ToolError(f"cannot reach the panel at {_base()}: {getattr(e, 'reason', e)}")
    if raw:
        return payload
    try:
        out = json.loads(payload)
    except ValueError:
        raise ToolError("the panel answered with something that is not JSON")
    if isinstance(out, dict) and out.get("error") and len(out) <= 3:
        raise ToolError(str(out["error"]))
    return out


def _inst(a):
    iid = str(a.get("instance") or "main")
    if not _ID.match(iid):
        raise ToolError(f"bad instance id {iid!r}")
    return iid


def _pick(d, keys):
    return {k: d.get(k) for k in keys if isinstance(d, dict) and k in d}


# ============================================================================
# tools
# ============================================================================
S_INST = {"instance": {"type": "string", "description": "Instance id (default 'main'). list_instances names them."}}


def _obj(props=None, required=()):
    return {"type": "object", "properties": dict(props or {}), **({"required": list(required)} if required else {}),
            "additionalProperties": False}


def t_list_instances(a):
    d = api("GET", "/api/instances")
    return dict(instances=[_pick(i, ("id", "name", "engine", "device", "devices", "port", "running", "legacy",
                                     "unit", "state", "model")) for i in d.get("instances") or []],
                devices=[_pick(x, ("pci", "vendor", "driver", "name", "vram_total_mib", "backends", "usable"))
                         for x in d.get("devices") or []])


def t_get_status(a):
    d = api("GET", "/api/status", dict(inst=_inst(a)))
    return _pick(d, ("running", "pid", "health", "engine", "vram_mib", "gtt_mib", "vram_total_mib",
                     "mem_avail_mb", "live_tps", "generating", "unit", "guard", "throughput", "live",
                     "instance", "gpus", "uptime"))


def t_list_servers(a):
    return api("GET", "/api/servers")


def _ctl(verb):
    def run(a):
        return api("POST", f"/api/{verb}", dict(inst=_inst(a)))
    return run


def t_get_params(a):
    return api("GET", "/api/params", dict(inst=_inst(a), backend=a.get("backend")))


def _merged(a):
    iid = _inst(a)
    new = a.get("params")
    if not isinstance(new, dict) or not new:
        raise ToolError("params: an object of SETTING: value, e.g. {\"CTX\": 65536}")
    cur = api("GET", "/api/params", dict(inst=iid))
    unknown = sorted(k for k in new if k not in cur)
    if unknown:
        raise ToolError("not settings of this instance: " + ", ".join(unknown[:20])
                        + " (get_params lists the valid ones)")
    merged = dict(cur)                     # BACKEND included: a partial save would otherwise
    merged.update(new)                     # fall back to vulkan's settings file
    return iid, cur, merged


def t_set_params(a):
    """Merge into the saved settings (never replace them), then save. The panel refuses
    values its safety rails reject; nothing restarts unless restart is true."""
    iid, cur, merged = _merged(a)
    saved = api("POST", "/api/params", dict(inst=iid), merged).get("params") or {}
    changed = {k: dict(before=cur.get(k), after=saved.get(k)) for k in a["params"] if cur.get(k) != saved.get(k)}
    out = dict(saved=True, changed=changed)
    if a.get("restart"):
        out["restart"] = api("POST", "/api/restart", dict(inst=iid))
    return out


def t_estimate_memory(a):
    if a.get("params"):
        iid, _cur, merged = _merged(a)
        return api("POST", "/api/estimate", dict(inst=iid), merged)
    return api("GET", "/api/estimate", dict(inst=_inst(a)))


def t_launch_plan(a):
    return api("GET", "/api/launch-plan", dict(inst=_inst(a)))


def t_list_models(a):
    return api("GET", "/api/models")


def t_gpu_status(a):
    return api("GET", "/api/gpus", dict(which="all"))


def t_gpu_tuning(a):
    d = api("GET", "/api/gpu-tune")
    for c in d.get("cards") or []:
        c["settings"] = [_pick(r, ("knob", "target", "value", "choices")) for r in c.get("settings") or []]
    d["history"] = [_pick(h, ("id", "instance", "model_name", "label", "verdict", "settings", "summary",
                              "started", "error")) for h in d.get("history") or []]
    d.pop("backups", None)
    return d


def t_start_gpu_benchmark(a):
    body = {k: a[k] for k in ("instance", "depth", "n_predict", "reps", "label") if a.get(k) is not None}
    body["instance"] = _inst(a)
    r = api("POST", "/api/gpu-tune/bench/start", None, body)
    return _pick(r, ("id", "instance", "state", "step", "depth", "n_predict", "reps", "settings"))


def t_get_gpu_benchmark(a):
    d = api("GET", "/api/gpu-tune/bench")
    act = d.get("active")
    if act:
        act.pop("series", None)
    return dict(active=act, history=[_pick(h, ("id", "instance", "model_name", "label", "verdict", "settings",
                                               "summary", "started", "error")) for h in d.get("history") or []])


def t_get_stats(a):
    return api("GET", "/api/stats", dict(inst=_inst(a)))


def t_get_diagnostics(a):
    d = api("GET", "/api/diagnostics", dict(inst=_inst(a)))
    return _pick(d, ("checks", "cmdline", "baseline"))


def t_get_crash_report(a):
    return api("GET", "/api/crash-report")


def t_get_logs(a):
    n = max(10, min(int(a.get("lines") or 200), 5000))
    d = api("GET", "/api/logs", dict(inst=_inst(a), n=max(1000, n * 400)))
    which = a.get("which") or "engine"
    if which not in ("engine", "launch"):
        raise ToolError("which: engine or launch")
    return dict(which=which, lines=(d.get(which) or "").splitlines()[-n:])


def t_get_power(a):
    d = api("GET", "/api/power")
    return dict(helper=_pick(d.get("helper") or {}, ("installed", "sudo_ok", "current", "install_cmd")),
                settings=[_pick(r, ("knob", "target", "device", "value")) for r in d.get("table") or []
                          if r.get("target")],
                boot_profile=(d.get("persisted") or {}).get("name"), drift=d.get("drift"),
                profiles=[_pick(p, ("id", "name", "description", "builtin")) for p in d.get("profiles") or []])


def t_apply_power_profile(a):
    pid = str(a.get("profile") or "")
    if not re.fullmatch(r"[\w-]{1,80}", pid):
        raise ToolError("profile: an id from get_power's profiles")
    return api("POST", "/api/power/apply", None, dict(profile=pid))


def t_start_depth_curve(a):
    body = {k: a[k] for k in ("depths", "n_predict", "reps") if a.get(k) is not None}
    r = api("POST", "/api/curve/start", dict(inst=_inst(a)), body)
    return _pick(r, ("id", "instance", "state", "step", "depths", "n_predict", "reps"))


def t_get_depth_curve(a):
    d = api("GET", "/api/curve", dict(inst=_inst(a)))
    act = d.get("active")
    return dict(active=_pick(act, ("id", "state", "step", "summary", "below_threshold_at", "error")) if act else None,
                history=[_pick(h, ("id", "model_name", "summary", "below_threshold_at", "started", "state"))
                         for h in (d.get("history") or [])[:10]])


def t_optimizer_status(a):
    d = api("GET", "/api/optimize/status")
    return dict(active=d.get("active"), runs=[_pick(r, ("id", "instance", "goal", "state", "started", "best"))
                                              for r in (d.get("runs") or [])[:10]])


def t_get_workload(a):
    d = api("GET", "/api/workload", dict(inst=_inst(a), days=a.get("days")))
    env, af = d.get("envelope") or {}, d.get("autofit") or {}
    env.pop("hours", None)                       # 168 cells; idle_hours and the findings carry it
    return dict(envelope=env, findings=d.get("findings"),
                autofit=dict(settings=af.get("settings"), due=af.get("due"), gate=af.get("gate"),
                             running=af.get("running"),
                             proposals=[_pick(x, ("id", "kind", "label", "diffs", "gain", "workload_gain", "quality",
                                                  "outcome")) for x in af.get("proposals") or []],
                             recent=[_pick(x, ("id", "kind", "started", "state", "decision", "outcome", "label",
                                               "gain", "verify", "rollback")) for x in (af.get("experiments") or [])[:10]]))


def t_apply_workload_proposal(a):
    xid = str(a.get("id") or "")
    if not re.fullmatch(r"x\d{1,12}", xid):
        raise ToolError("id: a proposal id from get_workload")
    return api("POST", "/api/workload/proposal/apply", dict(inst=_inst(a)),
               dict(id=xid, restart=bool(a.get("restart"))))


def t_list_files(a):
    return api("GET", "/api/files/list", dict(path=str(a.get("path") or "")))


def t_read_text_file(a):
    cap = max(1024, min(int(a.get("max_bytes") or 65536), 262144))
    data = api("GET", "/api/files/download", dict(path=str(a.get("path") or "")), raw=True)
    if b"\x00" in data[:8192]:
        raise ToolError("that is a binary file; this tool reads text only")
    return dict(path=a.get("path"), bytes=len(data), truncated=len(data) > cap,
                text=data[:cap].decode("utf-8", errors="replace"))


RO = dict(readOnlyHint=True, openWorldHint=False)
RW = dict(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
RISKY = dict(readOnlyHint=False, destructiveHint=True, openWorldHint=False)

TOOLS = [
    ("list_instances", "Every instance (engine, device(s), port, running or not) and every GPU in the box.",
     _obj(), RO, t_list_instances),
    ("get_status", "One instance's live state: running, health, VRAM, live tokens/s, systemd unit, start guard, GPUs.",
     _obj(S_INST), RO, t_get_status),
    ("list_servers", "Every inference server running on the box, whatever started it.", _obj(), RO, t_list_servers),
    ("start_instance", "Start an instance. The panel refuses a plan that would not fit or would crash-loop and says why.",
     _obj(S_INST), RW, _ctl("start")),
    ("stop_instance", "Stop an instance (its server process ends; in-flight requests fail).",
     _obj(S_INST), RISKY, _ctl("stop")),
    ("restart_instance", "Restart an instance. Refused while main is still loading or its crash-loop guard is locked.",
     _obj(S_INST), RISKY, _ctl("restart")),
    ("get_params", "An instance's saved settings (the params.env the next start uses).",
     _obj(dict(S_INST, backend={"type": "string", "description": "vulkan, rocm, cuda or cpu: that backend's saved copy"})),
     RO, t_get_params),
    ("set_params", "Change some settings of an instance: merged into the saved ones, checked by the panel, saved. "
                   "Takes effect at the next start; restart=true restarts now. Run estimate_memory first.",
     _obj(dict(S_INST, params={"type": "object", "description": "SETTING: value pairs, e.g. {\"CTX\": 65536}"},
               restart={"type": "boolean", "description": "restart the instance after saving (default false)"}),
          ("params",)), RISKY, t_set_params),
    ("estimate_memory", "VRAM / RAM estimate for the saved settings, or for changed ones without saving them.",
     _obj(dict(S_INST, params={"type": "object", "description": "optional SETTING: value changes to try"})),
     RO, t_estimate_memory),
    ("launch_plan", "The exact argv and environment a start would use, with the errors that would refuse it.",
     _obj(S_INST), RO, t_launch_plan),
    ("list_models", "Model files on disk: size, kind, which instance uses each.", _obj(), RO, t_list_models),
    ("gpu_status", "Live per-GPU numbers: VRAM, load, temperature, power, PCIe link, which instances use it.",
     _obj(), RO, t_gpu_status),
    ("gpu_tuning", "GPU Tuning view: every card's clocks, OverDrive ranges, sensors, tuning settings, "
                   "benchmark history and evidence-based advice.", _obj(), RO, t_gpu_tuning),
    ("start_gpu_benchmark", "Benchmark a running llama.cpp instance's GPU(s): repeated fixed request, decode t/s, "
                            "tokens per joule, peak temperatures. Changes no setting.",
     _obj(dict(S_INST, depth={"type": "integer", "description": "prompt tokens (256-65536, default 2048)"},
               n_predict={"type": "integer", "description": "generated tokens per run (32-2048, default 256)"},
               reps={"type": "integer", "description": "runs (1-30, default 5)"},
               label={"type": "string", "description": "a note saved with the result"})), RW, t_start_gpu_benchmark),
    ("get_gpu_benchmark", "The running GPU benchmark (if any) and recent results.", _obj(), RO, t_get_gpu_benchmark),
    ("get_stats", "Request, token and speed statistics of an instance.", _obj(S_INST), RO, t_get_stats),
    ("get_diagnostics", "Pass / warn / fail checks with evidence and the fix, and the live command line.",
     _obj(S_INST), RO, t_get_diagnostics),
    ("get_crash_report", "Boot history with clean vs hard stops and the settings in force at each crash.",
     _obj(), RO, t_get_crash_report),
    ("get_logs", "The last lines of an instance's engine or launch log.",
     _obj(dict(S_INST, which={"type": "string", "enum": ["engine", "launch"]},
               lines={"type": "integer", "description": "10-5000, default 200"})), RO, t_get_logs),
    ("get_power", "Power options: every power setting's live value, the boot profile, drift, the saved profiles.",
     _obj(), RO, t_get_power),
    ("apply_power_profile", "Apply a saved power profile now (live; the boot profile is unchanged). Needs the power helper.",
     _obj({"profile": {"type": "string", "description": "profile id from get_power"}}, ("profile",)), RISKY,
     t_apply_power_profile),
    ("start_depth_curve", "Measure decode tokens/s against context depth on a running llama.cpp instance.",
     _obj(dict(S_INST, depths={"type": "array", "items": {"type": "integer"}},
               n_predict={"type": "integer"}, reps={"type": "integer"})), RW, t_start_depth_curve),
    ("get_depth_curve", "The running depth-curve measurement and recent curves of an instance.",
     _obj(S_INST), RO, t_get_depth_curve),
    ("optimizer_status", "The optimizer's active run and recent runs.", _obj(), RO, t_optimizer_status),
    ("get_workload", "What this instance's real traffic looks like (context depth, output length, concurrency, "
                     "idle hours, decode against the measured curve), the findings with their evidence, and "
                     "auto-fit's state: what is due, why it is or is not starting, pending proposals.",
     _obj(dict(S_INST, days={"type": "integer", "description": "window in days (default 14)"})), RO, t_get_workload),
    ("apply_workload_proposal", "Apply an auto-fit proposal (saved settings; restart=true restarts the instance "
                                "now). The change is then checked against real traffic.",
     _obj(dict(S_INST, id={"type": "string", "description": "proposal id from get_workload"},
               restart={"type": "boolean"}), ("id",)), RISKY, t_apply_workload_proposal),
    ("list_files", "List a folder of the panel's home folder (hidden system folders are not reachable).",
     _obj({"path": {"type": "string", "description": "relative to the home folder; empty for its top"}}),
     RO, t_list_files),
    ("read_text_file", "Read a text file from the home folder (logs, templates, configs). Text only, capped.",
     _obj({"path": {"type": "string"}, "max_bytes": {"type": "integer", "description": "1024-262144, default 65536"}},
          ("path",)), RO, t_read_text_file),
]


def readonly():
    return os.environ.get("LEXIPANEL_MCP_READONLY", "").lower() in ("1", "true", "yes")


def _visible():
    return [t for t in TOOLS if not (readonly() and not t[3].get("readOnlyHint"))]


def list_tools():
    return [dict(name=n, description=d, inputSchema=s, annotations=h) for n, d, s, h, _f in _visible()]


def call_tool(name, arguments):
    """Run one tool. Returns the tool's result; raises ToolError for a failed call and
    KeyError for a tool that does not exist (or is hidden by read-only mode)."""
    t = next((t for t in _visible() if t[0] == name), None)
    if t is None:
        raise KeyError(name)
    if arguments is not None and not isinstance(arguments, dict):
        raise ToolError("arguments must be an object")
    try:
        return t[4](arguments or {})
    except (TypeError, ValueError) as e:
        raise ToolError(f"bad arguments: {e}")


# ============================================================================
# JSON-RPC (MCP)
# ============================================================================
def _err(mid, code, msg):
    return dict(jsonrpc="2.0", id=mid, error=dict(code=code, message=msg))


def handle(msg):
    """One JSON-RPC message (or a batch list). Returns the response, or None when the
    message was only notifications."""
    if isinstance(msg, list):
        out = [r for r in (handle(m) for m in msg) if r is not None]
        return out or None
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method", ""), str):
        return _err(None, -32600, "invalid request")
    method, mid, note = msg.get("method"), msg.get("id"), "id" not in msg
    params = msg.get("params") or {}
    if method is None:                     # a response from the client (we send no requests): ignore
        return None
    if method.startswith("notifications/"):
        return None
    try:
        if method == "initialize":
            want = params.get("protocolVersion")
            result = dict(protocolVersion=want if want in PROTOCOLS else PROTOCOLS[0],
                          capabilities=dict(tools=dict(listChanged=False)),
                          serverInfo=dict(name="lexipanel", title="LexiPanel", version=VERSION),
                          instructions=INSTRUCTIONS)
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = dict(tools=list_tools())
        elif method == "tools/call":
            name = params.get("name")
            try:
                out = call_tool(name, params.get("arguments"))
                result = dict(content=[dict(type="text", text=json.dumps(out, indent=1, default=str))],
                              isError=False)
            except ToolError as e:
                result = dict(content=[dict(type="text", text=str(e))], isError=True)
            except KeyError:
                return None if note else _err(mid, -32602, f"unknown tool: {name!r}")
        else:
            return None if note else _err(mid, -32601, f"method not found: {method}")
    except Exception as e:                 # never let one bad call kill the stdio loop
        return None if note else _err(mid, -32603, f"internal error: {e}")
    return None if note else dict(jsonrpc="2.0", id=mid, result=result)


def serve_stdio(inp=sys.stdin, out=sys.stdout):
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            resp = _err(None, -32700, "parse error")
        else:
            resp = handle(msg)
        if resp is not None:
            out.write(json.dumps(resp, default=str) + "\n")
            out.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    serve_stdio()
