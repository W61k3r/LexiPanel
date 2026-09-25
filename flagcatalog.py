#!/usr/bin/env python3
"""
Every option the ACTIVE build accepts, read from its own --help (added 2026-09-23).

The curated parameter form covers the options that matter most, with guards,
estimates and explanations. Upstream adds and renames options every few days,
and a hand-written list goes stale with the next daily build. This module
closes the gap: it parses `llama-server --help` / `sd-server --help` from the
build the instance will actually launch, and returns every option the curated
form does not already own. The UI shows them grouped by upstream section, with
upstream's own description, and writes the choices into EXTRA_ARGS / SD_EXTRA,
which the launch plan already validates and passes through.

Left out on purpose:
  * options the curated form already sets (any alias, or its --no- form)
  * options the plan owns or forbids (model, port, host, log file, draft model)
  * options upstream marks as removed, and one-shot actions (--help, --version,
    --list-devices, --completion-bash)
  * the "use default model X" presets that download weights from the internet
"""
import os, re, shlex, subprocess
import flaghelp as H
from pathlib import Path

P = None
_cache = {}

ACTIONS = {"-h", "--help", "--usage", "--version", "--list-devices", "--completion-bash",
           "--cache-list",
           # a secret would sit in the visible command line; the panel stores the
           # Hugging Face token itself (Download tab)
           "--hf-token", "-hft"}
# upstream's own words say these are unsafe on an exposed server
CAUTION = re.compile(r"do not enable in untrusted|untrusted environments|security|"
                     r"arbitrary|expose", re.I)
SD_OWNED = {"--listen-ip", "--listen-port", "-l", "--diffusion-model", "-m", "--model", "--llm",
            "--llm_vision", "--vae", "--clip_l", "--t5xxl", "--backend", "--offload-to-cpu",
            "--diffusion-fa", "--vae-tiling", "--mmap", "--max-vram", "-t", "--threads", "-W",
            "--width", "-H", "--height", "--steps", "--cfg-scale", "--sampling-method", "-s",
            "--seed", "--log-level", "--scheduler", "--flow-shift", "-n", "--negative-prompt",
            "-v", "--verbose", "--qwen2vl", "--qwen2vl_vision", "--clip-on-cpu", "--vae-on-cpu",
            "--control-net-cpu"}


def bind(panel_module):
    global P
    P = panel_module


def parse_help(text):
    """[{section, flags, arg, desc}] from a llama.cpp- or sd.cpp-style --help."""
    out, section, cur = [], "General", None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        s = line.strip()
        m = re.match(r"^-{3,}\s*(.*?)\s*-{3,}$", s)
        if m:                                           # ----- common params -----
            section, cur = m.group(1).strip() or section, None
            continue
        if not raw.startswith(" ") and not s.startswith("-") and s.endswith(":"):
            section, cur = s[:-1], None                 # "Context Options:"
            continue
        indent = len(raw) - len(raw.lstrip())
        if s.startswith("-") and indent <= 6:
            left, _, desc = re.sub(r",\s+", ", ", s).partition("  ")
            flags = re.findall(r"(?<![\w-])(--?[a-zA-Z][\w-]*)", left)
            rest = re.sub(r"(?<![\w-])--?[a-zA-Z][\w-]*,?", " ", left).strip()
            cur = dict(section=section, flags=flags, arg=rest or None, desc=desc.strip())
            out.append(cur)
        elif cur is not None and indent > 6:
            cur["desc"] = (cur["desc"] + " " + s).strip()
    for o in out:
        if not o["arg"]:
            # sd.cpp writes the type into the description: "--seed  RNG seed ..."
            m = re.match(r"^<(\w+)>\s*", o["desc"])
            if m:
                o["arg"] = f"<{m.group(1)}>"
                o["desc"] = o["desc"][m.end():]
    return out


def _help(binary, bindir):
    exe = Path(bindir) / binary
    try:
        key = (str(exe), exe.stat().st_mtime)
    except OSError:
        return None, None
    if key not in _cache:
        r = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, LD_LIBRARY_PATH=str(bindir)))
        _cache[key] = r.stdout + r.stderr
    return _cache[key], str(exe)


def _llama_known():
    """Flags the curated llama.cpp form emits (any alias counts)."""
    known = set()
    for row in P.LAUNCH_FLAGS + P.SPEC_FLAGS + P.VISION_FLAGS:
        for x in row:
            for y in (x if isinstance(x, (list, tuple)) else [x]):
                if isinstance(y, str) and y.startswith("-"):
                    known.add(y)
    src = Path(P.__file__).read_text()
    # flags the plan writes literally (e.g. "--jinja", "--reasoning-format")
    known |= set(re.findall(r'"(--?[a-zA-Z][\w-]*)"', src[src.index("def _llama_launch_plan"):
                                                        src.index("def _render_flags")]))
    return known


def catalog(inst=None):
    inst = inst or P.INST()
    engine = inst.get("engine") or "llama.cpp"
    with P.using_instance(inst):
        vals = P.load_params()
    backend = str(vals.get("BACKEND") or "vulkan")
    if engine == "sd.cpp":
        bindir = P.engines.ENGINES["sd.cpp"].active(backend)
        binary, known, owned, key = "sd-server", SD_OWNED, SD_OWNED, "SD_EXTRA"
    else:
        bindir = P.active_build(backend)
        binary, known, key = "llama-server", _llama_known(), "EXTRA_ARGS"
        owned = set(P.EXTRA_FORBIDDEN)
    if not bindir:
        return dict(engine=engine, error=f"no active {engine} {backend} build", sections=[])
    text, exe = _help(binary, bindir)
    if text is None:
        return dict(engine=engine, error=f"{binary} not found in {bindir}", sections=[])
    opts = parse_help(text)
    total = len(opts)
    shown, excluded, curated = [], 0, 0
    for o in opts:
        fl = set(o["flags"])
        base = {re.sub(r"^--no-", "--", f) for f in fl} | fl
        if base & known:
            curated += 1
            continue
        per_request = engine == "sd.cpp" and fl & H.SD_PER_REQUEST
        if per_request or fl & (owned | ACTIONS) or "has been removed" in o["desc"] or \
                "can download weights" in o["desc"] or not fl:
            excluded += 1
            continue
        env = re.search(r"\(env: ([A-Z0-9_]+)\)", o["desc"])
        default = re.search(r"\(default: ([^)]*)\)", o["desc"])
        choices = re.search(r"one of \[([^\]]+)\]", o["desc"]) or \
            (re.search(r"\{([\w, |-]+)\}", o["arg"] or "") if o["arg"] else None)
        long_flag = next((f for f in o["flags"] if f.startswith("--") and not f.startswith("--no-")),
                         o["flags"][0])
        neg = next((f for f in o["flags"] if f.startswith("--no-")), None)
        takes = bool(o["arg"]) or (engine == "sd.cpp" and not (fl & H.SD_SWITCHES))
        upstream = re.sub(r"\s*\(env: [A-Z0-9_]+\)", "", o["desc"])
        shown.append(dict(section=o["section"], flag=long_flag, flags=o["flags"], neg=neg,
                          arg=o["arg"], takes_value=takes,
                          plain=H.plain(long_flag, upstream, engine),
                          desc=re.sub(r"\s*\(env: [A-Z0-9_]+\)", "", o["desc"]),
                          env=env.group(1) if env else None,
                          default=default.group(1) if default else None,
                          choices=[c.strip() for c in re.split(r"[,|]", choices.group(1))]
                          if choices else None,
                          experimental="experimental" in o["desc"].lower(),
                          caution=bool(CAUTION.search(o["desc"]))))
    sections = []
    for o in shown:
        if not sections or sections[-1]["name"] != o["section"]:
            sections.append(dict(name=o["section"], options=[]))
        sections[-1]["options"].append(o)
    try:
        current = shlex.split(str(vals.get(key) or ""))
    except ValueError:
        current = []
    return dict(engine=engine, binary=exe, build=Path(bindir).name, key=key, current=current,
                counts=dict(total=total, curated=curated, other=len(shown), excluded=excluded),
                sections=sections)
