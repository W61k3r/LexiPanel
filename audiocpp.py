#!/usr/bin/env python3
"""
audio.cpp instances: parameters, models, launch plan, and runs (added 2026-09-23).

An instance whose instance.json says engine="audio.cpp" runs audiocpp_server
(github.com/0xShug0/audio.cpp): TTS, voice cloning, ASR, music, sound effects,
source separation and more, all on ggml. It shares everything instance-shaped
with the other engines (systemd --user unit, devices, ports, the failed-start
counter, launch log, start/stop). What differs lives here:

  * server parameters: AC_* keys in the instance's params.env
  * which models it serves: audio-models.json in the instance dir, one entry
    per installed package, with its task and its load/session/default-request
    options. The option forms are generated from the build's model_specs/*.json,
    which carry type, range, default and a description for every option.
  * packages are installed with the build's own tools/model_manager_v2.py
    (stdlib Python, no pip) into ~/audiocpp/models, shared by all instances
  * launch_plan(): argv + env + the server config JSON the launcher writes
  * runs: every task goes through POST /v1/tasks/run; outputs are kept in
    ~/audiocpp/outputs/<instance>/

Facts learned bringing it up on LexiPanel (v0.8.1), do not rediscover:
  * load and session options MUST be keyed "<family>.<name>" in the server
    config. The unprefixed form is accepted and silently ignored.
    Request options are plain names.
  * a GGUF package is loaded by its .gguf FILE: two precisions of one model
    share a target directory, and a directory with several GGUFs is ambiguous.
  * the Kokoro/Kitten/Piper/sanoTTS style frontends dlopen libespeak-ng. This
    host has no system copy, so ~/audiocpp/deps holds one unpacked from the
    Ubuntu .debs, and AUDIOCPP_ESPEAK_LIBRARY/_DATA point at it. A system copy
    (apt install libespeak-ng1 espeak-ng-data) wins when present.
  * the Linux release links ggml statically and enumerates every Vulkan ICD,
    the Intel iGPU included (trap 4), so the plan pins by ICD like llama.cpp.
  * one server drives one device.
"""
import base64, glob, json, os, re, shlex, subprocess, threading, time, urllib.error, urllib.request, uuid
from pathlib import Path

P = None
E = None                          # engines module

DEFAULTS = dict(
    BACKEND="vulkan", PORT=8085, HOST="127.0.0.1",
    AC_THREADS=4, AC_MAX_LOADED=1, AC_IDLE_UNLOAD_S=600, AC_MIN_FREE_MB=1024,
    AC_BUSY_TIMEOUT_S=300, AC_LAZY=1, AC_UI=1, AC_LOG=1,
    RAM_FLOOR_MB=2048, AC_EXTRA="",
)
BACKENDS = ("vulkan", "cpu")
GROUPS = ["Backend", "Memory & residency", "Server"]
# enum presets from upstream docs/maintainers/model_specs.md (v0.8.1)
ENUM_PRESETS = dict(
    weight_type_full=["native", "f32", "f16", "bf16", "q8_0"],
    weight_type_conv=["native", "f32", "f16"],
    weight_type_codec_q8=["native", "f32", "f16", "q8_0"],
    text_chunk_mode_full=["word_budget", "tag_aware", "japanese", "endline"],
    perf_mode_flash_attention=["off", "flash_attention"],
    best_of_n_language=["auto", "en", "ja"],
)
# what each server task means, in words
TASK_WORDS = dict(tts="text to speech", clon="voice cloning (speak in a reference voice)",
                  vdes="voice design (describe the voice in words)", gen="music / sound generation",
                  asr="speech to text", vc="voice conversion", s2s="audio to audio",
                  sep="source separation (split vocals / stems)", vad="voice activity detection",
                  diar="speaker diarization (who spoke when)", align="forced alignment",
                  midi="audio to MIDI / score", svc="singing voice conversion")
UPLOAD_MAX = 50 * 1048576


def bind(panel_module, engines_module):
    global P, E
    P, E = panel_module, engines_module


def home():
    return P.HOME / "audiocpp"


def models_root():
    return home() / "models"


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
def _m(group, label, tip, type="text", options=None, strict=False, unit=None, numeric=False):
    d = dict(group=group, label=label, tip=tip, type=type)
    if options is not None:
        d["options"] = options
    if strict:
        d["strict"] = True
    if unit:
        d["unit"] = unit
    if numeric:
        d["numeric"] = True
    return d


def meta():
    flag = lambda f: f"<br><br><span style='opacity:.75'>Flag: {f}</span>"
    return {
        "BACKEND": _m("Backend", "Compute backend",
                      "What runs the models. <b>vulkan</b> uses the graphics card picked for this "
                      "instance. <b>cpu</b> uses no graphics card: small speech models (Kokoro, "
                      "Kitten, Piper, Supertonic) are fast enough on the CPU, music and big "
                      "voice-cloning models are not. Upstream tunes audio.cpp for CUDA, so on "
                      "Vulkan some models are slower or unsupported." + flag("--backend"),
                      "select", list(BACKENDS), strict=True),
        "AC_THREADS": _m("Backend", "CPU threads",
                         "How many CPU cores the CPU parts use. This box has 8; leave some for "
                         "the chat model and the panel." + flag("--threads"), "int", numeric=True),
        "AC_MAX_LOADED": _m("Memory & residency", "Models kept loaded",
                            "How many models may sit in memory at once. When one more is needed, "
                            "the least recently used idle one is unloaded first. <b>1</b> means "
                            "one at a time, the safe choice while the chat model owns most of "
                            "the card. <b>0</b> = no limit." + flag("--max-loaded-models"),
                            "int", numeric=True),
        "AC_IDLE_UNLOAD_S": _m("Memory & residency", "Unload when idle after",
                               "Frees every loaded model after this many seconds without a "
                               "request, handing the memory back to the chat model. The next "
                               "request loads it again (seconds for speech models, longer for "
                               "music). <b>0</b> = never." + flag("--idle-unload-ms"),
                               "int", unit="s", numeric=True),
        "AC_MIN_FREE_MB": _m("Memory & residency", "Keep free",
                             "Refuse to load a model unless this much host RAM and graphics "
                             "memory is still free afterwards, so a load fails cleanly instead "
                             "of starving the chat model. <b>0</b> = no check."
                             + flag("--min-free-memory-mb"), "int", unit="MiB", numeric=True),
        "AC_LAZY": _m("Memory & residency", "Load on first use",
                      "Load each model only when it is first asked for, instead of all at "
                      "start. Keeps start-up fast and memory low.", "bool"),
        "RAM_FLOOR_MB": _m("Memory & residency", "Host RAM floor",
                           "The launcher kills audiocpp_server if available host RAM drops "
                           "below this. The host has hard-locked from running out of RAM.",
                           "int", unit="MB", numeric=True),
        "AC_BUSY_TIMEOUT_S": _m("Server", "Busy timeout",
                                "A request waiting this long for a busy model fails instead of "
                                "queueing forever. Long music jobs hold the model for minutes, "
                                "so raise it if requests pile up behind them. <b>0</b> = wait "
                                "forever." + flag("--busy-timeout-ms"), "int", unit="s",
                                numeric=True),
        "AC_UI": _m("Server", "Built-in web UI",
                    "audio.cpp's own browser UI (Arena, voice library, microphone input) on the "
                    "instance's port. It has no password: keep the listen address on "
                    "127.0.0.1 and reach it through an SSH tunnel." + flag("--ui / --no-ui"),
                    "bool"),
        "AC_LOG": _m("Server", "Framework log",
                     "Write audio.cpp's own log to the engine log (Logs tab)." + flag("--log"),
                     "bool"),
        "PORT": _m("Server", "Port", "Where audiocpp_server listens." + flag("--port"), "int",
                   numeric=True),
        "HOST": _m("Server", "Listen address",
                   "audiocpp_server has no authentication: keep 127.0.0.1 and use it through "
                   "the panel." + flag("--host"), "select", ["127.0.0.1", "0.0.0.0"]),
        "AC_EXTRA": _m("Server", "Extra arguments",
                       "Anything else audiocpp_server accepts, passed as-is (shell-quoted)."),
    }


def _coerce(vals):
    for k, v in list(vals.items()):
        d = DEFAULTS.get(k)
        if isinstance(d, int) and str(v).lstrip("-").isdigit():
            vals[k] = int(v)
    return vals


def load_params(inst):
    cur = dict(DEFAULTS)
    cur.update(P._read_env_file(inst["dir"] / "params.env"))
    return _coerce(cur)


def write_params(inst, vals, note=""):
    body = ["# GENERATED by the admin panel - do not hand-edit.",
            f"# audio.cpp instance '{inst['id']}'. Read by panel/instance_launch.py.",
            "# The models it serves are in audio-models.json next to this file.",
            *([f"# {note}"] if note else []), ""]
    for k in DEFAULTS:
        v = vals.get(k, DEFAULTS[k])
        body.append(f"{k}={P._env_quote(v)}")
    (inst["dir"] / "params.env").write_text("\n".join(body) + "\n")


def save_params(inst, new):
    cur = load_params(inst)
    unknown = sorted(set(new) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"not audio.cpp settings: {', '.join(unknown)}")
    cur.update(new)
    cur = _coerce(cur)
    if cur["BACKEND"] not in BACKENDS:
        raise ValueError(f"BACKEND must be one of {', '.join(BACKENDS)}")
    try:
        port = int(cur["PORT"])
        assert 1024 <= port <= 65535 and port not in (8090, 8091, 8092)
    except (ValueError, AssertionError):
        raise ValueError("PORT must be 1024-65535 and not the panel's own ports")
    for k in ("AC_THREADS",):
        if int(cur[k]) < 1:
            raise ValueError(f"{k} must be at least 1")
    for k in ("AC_MAX_LOADED", "AC_IDLE_UNLOAD_S", "AC_MIN_FREE_MB", "AC_BUSY_TIMEOUT_S"):
        if int(cur[k]) < 0:
            raise ValueError(f"{k} must be 0 or more")
    try:
        shlex.split(str(cur.get("AC_EXTRA") or ""))
    except ValueError as e:
        raise ValueError(f"AC_EXTRA does not parse: {e}")
    write_params(inst, cur)
    return cur


# --------------------------------------------------------------------------
# build, specs, loaders, packages
# --------------------------------------------------------------------------
_cache = {}                        # (kind, bindir) -> (mtime, value)


def bindir_for(backend):
    """The Vulkan build also carries the CPU backend, so a CPU instance can
    use either; a dedicated CPU build wins when one is active."""
    e = E.ENGINES["audio.cpp"]
    if backend == "cpu":
        return e.active("cpu") or e.active("vulkan")
    return e.active(backend)


def any_bindir():
    return bindir_for("vulkan") or bindir_for("cpu")


def _cached(kind, bindir, fn):
    try:
        mt = Path(bindir, "audiocpp_server").stat().st_mtime
    except OSError:
        return fn()
    c = _cache.get((kind, bindir))
    if c and c[0] == mt:
        return c[1]
    v = fn()
    _cache[(kind, bindir)] = (mt, v)
    return v


def specs(bindir):
    def load():
        out = {}
        for f in sorted(glob.glob(f"{bindir}/model_specs/*.json")):
            try:
                d = json.loads(Path(f).read_text())
            except (OSError, ValueError):
                continue
            if d.get("family"):
                out[d["family"]] = d
        return out
    return _cached("specs", bindir, load) if bindir else {}


def loaders(bindir):
    """family -> {tasks: {task: [modes]}, api_endpoints} as THIS build reports."""
    def load():
        try:
            r = subprocess.run([f"{bindir}/audiocpp_cli", "--list-loaders", "--json"],
                               capture_output=True, text=True, timeout=60)
            return json.loads(r.stdout).get("loaders", {})
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {}
    return _cached("loaders", bindir, load) if bindir else {}


def installed_packages():
    """package id -> {dir, files, installed_at}, from the manager's markers."""
    out = {}
    for m in models_root().glob("*/.audiocpp-package-*.json"):
        try:
            d = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        pid = d.get("package_id") or m.name[len(".audiocpp-package-"):-5]
        out[pid] = dict(dir=str(m.parent), files=sorted(d.get("files") or {}),
                        installed_at=d.get("installed_at_unix"))
    return out


def _sizes_file():
    return home() / "package-sizes.json"


def _sizes():
    try:
        return json.loads(_sizes_file().read_text())
    except (OSError, ValueError):
        return {}


def package_path(pkg, inst_rec):
    """What the server config's "path" should be for an installed package: its
    one .gguf when it has exactly one, else its directory."""
    d = Path(inst_rec["dir"])
    strip = (pkg.get("strip_prefix") or "").rstrip("/")
    ggufs = [f for f in inst_rec["files"] if f.endswith(".gguf")]
    if len(ggufs) == 1:
        rel = ggufs[0][len(strip) + 1:] if strip and ggufs[0].startswith(strip + "/") else ggufs[0]
        if (d / rel).is_file():
            return str(d / rel)
        hits = list(d.rglob(Path(rel).name))
        if hits:
            return str(hits[0])
    return str(d)


def _pkg_bytes(pkg, inst_rec):
    if not inst_rec:
        return None
    p = Path(package_path(pkg, inst_rec))
    try:
        return p.stat().st_size if p.is_file() else P._dir_bytes(p)
    except OSError:
        return None


def catalog():
    bindir = any_bindir()
    sp, ld, have, sizes = specs(bindir), loaders(bindir), installed_packages(), _sizes()
    fams = []
    for fam, d in sorted(sp.items(), key=lambda kv: (kv[1].get("category") or "", kv[0])):
        ui = d.get("ui") or {}
        pk = []
        for p in d.get("packages") or []:
            ir = have.get(p["id"])
            pk.append(dict(id=p["id"], display_name=p.get("display_name") or p["id"],
                           precision=p.get("precision"), format=p.get("format"),
                           default=bool(p.get("default")),
                           recommended=ui.get("recommended_package") == p["id"],
                           installed=bool(ir), path=package_path(p, ir) if ir else None,
                           size_bytes=(sizes.get(p["id"]) or {}).get("size_bytes"),
                           disk_bytes=_pkg_bytes(p, ir),
                           gated=bool(((d.get("package_defaults") or {}).get("download") or {})
                                      .get("gated"))))
        lt = (ld.get(fam) or {}).get("tasks") or {}
        fams.append(dict(family=fam, display_name=d.get("display_name") or fam,
                         description=d.get("description") or "", category=d.get("category"),
                         status=d.get("status"), spec_tasks=d.get("tasks") or [],
                         server_tasks={t: m for t, m in lt.items()},
                         task_words={t: TASK_WORDS.get(t, t) for t in lt},
                         api_endpoints=(ld.get(fam) or {}).get("api_endpoints") or [],
                         languages=d.get("languages") or [],
                         voices=ui.get("builtin_voices") or [],
                         default_voice=ui.get("default_voice"),
                         capabilities=d.get("capabilities") or {},
                         docs=ui.get("docs") or [], packages=pk,
                         options={s: _opts(o) for s, o in (d.get("options") or {}).items()},
                         runnable=bool(lt)))
    return dict(families=fams, build=bindir, models_root=str(models_root()),
                espeak=espeak_env() is not None,
                note=None if bindir else "no audio.cpp build installed - install one on the "
                                          "Builds tab")


def _opts(rows):
    out = []
    for o in rows or []:
        o = dict(o)
        if o.get("type") == "enum" and not o.get("values") and o.get("preset") in ENUM_PRESETS:
            o["values"] = ENUM_PRESETS[o["preset"]]
        out.append(o)
    return out


def _package(pid):
    for fam, d in specs(any_bindir()).items():
        for p in d.get("packages") or []:
            if p["id"] == pid:
                return fam, d, p
    raise ValueError(f"unknown audio.cpp package {pid!r}")


def _manager(*args, timeout=None, **kw):
    bindir = any_bindir()
    if not bindir:
        raise ValueError("no audio.cpp build installed - install one on the Builds tab")
    cmd = ["python3", f"{bindir}/tools/model_manager_v2.py", "--specs-dir",
           f"{bindir}/model_specs", *args]
    if kw.get("popen"):
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout or 120)


def check_sizes(family):
    """Ask Hugging Face for the download size of every package of one family.
    Network, a few seconds; cached in ~/audiocpp/package-sizes.json."""
    d = specs(any_bindir()).get(family)
    if not d:
        raise ValueError(f"unknown audio.cpp family {family!r}")
    ids = [p["id"] for p in d.get("packages") or []]
    r = _manager("sizes", *ids, timeout=120)
    try:
        rows = json.loads(r.stdout)
    except ValueError:
        raise ValueError("size check failed: " + (r.stderr or r.stdout)[-300:])
    cur = _sizes()
    for x in rows:
        cur[x["id"]] = dict(size_bytes=x.get("size_bytes"), state=x.get("state"),
                            message=x.get("message"), at=int(time.time()))
    home().mkdir(exist_ok=True)
    _sizes_file().write_text(json.dumps(cur, indent=1))
    return dict(ok=True, sizes={x["id"]: x.get("size_bytes") for x in rows})


def install_package(pid):
    fam, d, p = _package(pid)
    if pid in installed_packages():
        raise ValueError(f"{pid} is already installed")
    for rec in P._downloads.values():
        if rec.get("kind") == "audio-model" and rec.get("package") == pid and \
                rec["status"] not in ("done", "failed"):
            raise ValueError(f"{pid} is already downloading")
    models_root().mkdir(parents=True, exist_ok=True)
    did = "acmodel-" + str(int(time.time() * 1000))
    total = (_sizes().get(pid) or {}).get("size_bytes") or 0
    rec = dict(id=did, url=f"hf:{((d.get('package_defaults') or {}).get('download') or {}).get('repo')}",
               name=f"audio.cpp model {pid}", dest=str(models_root() / (p.get("target_directory") or pid)),
               kind="audio-model", package=pid, family=fam, status="running", pct=0.0,
               downloaded=0, total=total, started=time.time(), error=None, log=[])
    with P._lock:
        P._downloads[did] = rec

    def run():
        try:
            proc = _manager("install", pid, "--models-root", str(models_root()), "--progress",
                            popen=True)
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode(errors="replace").rstrip()
                m = re.match(r"AUDIOCPP_PROGRESS downloaded=(\d+) total=(\d+)", line)
                if m:
                    rec["downloaded"], rec["total"] = int(m.group(1)), int(m.group(2)) or rec["total"]
                    rec["pct"] = round(rec["downloaded"] * 100 / max(1, rec["total"]), 1)
                elif line:
                    rec["log"] = (rec["log"] + [line])[-20:]
            rc = proc.wait()
            if rc != 0 or pid not in installed_packages():
                raise RuntimeError("; ".join(rec["log"][-3:]) or f"model manager exit {rc}")
            rec.update(status="done", pct=100.0)
        except Exception as e:
            rec.update(status="failed", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    return {k: v for k, v in rec.items() if k != "log"}


def _referencing(pid):
    out = []
    for iid in P.instance_ids():
        try:
            inst = P.get_instance(iid)
        except ValueError:
            continue
        if inst.get("engine") == "audio.cpp" and any(m.get("package") == pid
                                                     for m in load_models(inst)):
            out.append(iid)
    return out


def uninstall_package(pid):
    _package(pid)
    if pid not in installed_packages():
        raise ValueError(f"{pid} is not installed")
    users = _referencing(pid)
    if users:
        raise ValueError(f"{pid} is served by instance(s) {', '.join(users)}; remove it from "
                         "their model list first")
    r = _manager("uninstall", pid, "--models-root", str(models_root()), timeout=300)
    if r.returncode != 0:
        raise ValueError("uninstall failed: " + (r.stderr or r.stdout)[-300:])
    return dict(ok=True, removed=pid)


# --------------------------------------------------------------------------
# which models an instance serves
# --------------------------------------------------------------------------
def _models_file(inst):
    return inst["dir"] / "audio-models.json"


def load_models(inst):
    try:
        return json.loads(_models_file(inst).read_text()).get("models", [])
    except (OSError, ValueError):
        return []


def _clean_opts(rows, given, where):
    """Keep only options the spec declares, coerced to their type; empty = unset."""
    by = {o["name"]: o for o in rows or []}
    out = {}
    for k, v in (given or {}).items():
        if v in ("", None):
            continue
        o = by.get(k)
        if not o:
            raise ValueError(f"{where}: {k!r} is not an option of this model")
        t = o.get("type")
        try:
            if t == "int":
                v = int(v)
            elif t == "float":
                v = float(v)
            elif t == "bool":
                v = str(v).lower() in ("1", "true", "yes", "on") if not isinstance(v, bool) else v
            else:
                v = str(v)
        except ValueError:
            raise ValueError(f"{where}: {k} must be {t}")
        if t in ("int", "float"):
            if o.get("min") is not None and v < o["min"]:
                raise ValueError(f"{where}: {k} must be at least {o['min']}")
            if o.get("max") is not None and v > o["max"]:
                raise ValueError(f"{where}: {k} must be at most {o['max']}")
        vals = o.get("values") or ENUM_PRESETS.get(o.get("preset") or "")
        if t == "enum" and vals and v not in vals:
            raise ValueError(f"{where}: {k} must be one of {', '.join(vals)}")
        out[k] = v
    return out


def save_models(inst, rows):
    bindir = any_bindir()
    sp, ld, have = specs(bindir), loaders(bindir), installed_packages()
    out, seen = [], set()
    for r in rows or []:
        pid = str(r.get("package") or "")
        fam, d, _p = _package(pid)
        if pid in seen:
            raise ValueError(f"{pid} is listed twice")
        seen.add(pid)
        tasks = (ld.get(fam) or {}).get("tasks") or {}
        task = str(r.get("task") or next(iter(tasks), ""))
        if task not in tasks:
            raise ValueError(f"{pid}: task {task!r} is not one this build offers for {fam} "
                             f"({', '.join(tasks) or 'none'})")
        mode = str(r.get("mode") or "offline")
        if mode not in tasks[task]:
            raise ValueError(f"{pid}: {task} runs {'/'.join(tasks[task])}, not {mode}")
        o = d.get("options") or {}
        out.append(dict(package=pid, family=fam, task=task, mode=mode,
                        enabled=bool(r.get("enabled", True)),
                        load_options=_clean_opts(o.get("load"), r.get("load_options"), f"{pid} load"),
                        session_options=_clean_opts(o.get("session"), r.get("session_options"),
                                                    f"{pid} session"),
                        default_request_options=_clean_opts(o.get("request"),
                                                            r.get("default_request_options"),
                                                            f"{pid} request")))
    missing = [m["package"] for m in out if m["package"] not in have]
    tmp = _models_file(inst).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(models=out), indent=1) + "\n")
    os.replace(tmp, _models_file(inst))
    return dict(models=out, not_installed=missing)


def server_config(inst, models):
    have = installed_packages()
    cfg_models = []
    for m in models:
        if not m.get("enabled", True):
            continue
        fam, _d, p = _package(m["package"])
        ir = have.get(m["package"])
        pref = lambda opts: {f"{fam}.{k}": v for k, v in (opts or {}).items()}
        e = dict(id=m["package"], family=fam, path=package_path(p, ir) if ir else "",
                 task=m["task"], mode=m.get("mode") or "offline")
        if m.get("load_options"):
            e["load_options"] = pref(m["load_options"])
        if m.get("session_options"):
            e["session_options"] = pref(m["session_options"])
        if m.get("default_request_options"):
            e["default_request_options"] = dict(m["default_request_options"])
        cfg_models.append(e)
    return cfg_models


# --------------------------------------------------------------------------
# launch plan
# --------------------------------------------------------------------------
def espeak_env():
    """Env that lets audio.cpp dlopen eSpeak-ng: none needed with a system
    copy, ~/audiocpp/deps otherwise, None when neither exists."""
    for lib in ("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1", "/lib/x86_64-linux-gnu/libespeak-ng.so.1"):
        if Path(lib).exists():
            return {}
    L = home() / "deps/root/usr/lib/x86_64-linux-gnu"
    if (L / "libespeak-ng.so.1").exists() and (L / "espeak-ng-data").is_dir():
        return dict(AUDIOCPP_ESPEAK_LIBRARY=str(L / "libespeak-ng.so.1"),
                    AUDIOCPP_ESPEAK_DATA=str(L / "espeak-ng-data"),
                    LD_LIBRARY_PATH=f"{L}:{L}/pulseaudio")
    return None


_vk_seen = {}


def _vk_check(bindir, env, dev):
    """Does this build, with this pinning, see exactly `dev` as Vulkan:0?"""
    key = (bindir, env.get("VK_DRIVER_FILES"), dev["pci"])
    if key in _vk_seen:
        return _vk_seen[key]
    e = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items() if k.startswith(("VK_", "GGML_")))
    out = P.sh(f"env {e} timeout 60 {shlex.quote(bindir)}/audiocpp_server --backend vulkan "
               "--list-devices 2>/dev/null", timeout=70)
    names = re.findall(r'^Vulkan:(\d+)\s+"([^"]*)"', out, re.M)
    ok = len(names) == 1 and P._vk_matches(dev, names[0][1])
    res = None if ok else (f"with pinning the build sees {', '.join(n for _, n in names) or 'no'} "
                           f"Vulkan device(s), expected only {dev['name']}")
    _vk_seen[key] = res
    return res


def launch_plan(inst, values=None):
    v = dict(values or load_params(inst))
    errors, warnings = [], []
    backend = str(v.get("BACKEND") or "vulkan")
    bindir = bindir_for(backend)
    if not bindir:
        errors.append(f"no active audio.cpp {backend} build - install one on the Builds tab")
    alld = {d["pci"]: d for d in P.gpu_devices(probe=False)}
    devs = []
    for pci in inst["devices"]:
        if pci == "cpu":
            continue
        if pci not in alld:
            errors.append(f"device {pci} is not in this machine")
        else:
            devs.append(alld[pci])
    if len(devs) > 1:
        errors.append("audio.cpp drives one device per server; give this instance one GPU "
                      "(make a second instance for another card)")
    if backend == "cpu" and devs:
        warnings.append("BACKEND=cpu: the selected GPU is not used at all")
    if backend != "cpu" and not devs and not any(p not in alld for p in inst["devices"] if p != "cpu"):
        errors.append(f"BACKEND={backend} needs a GPU; this instance is CPU-only - pick BACKEND=cpu")

    models = load_models(inst)
    enabled = [m for m in models if m.get("enabled", True)]
    if not enabled:
        errors.append("no models: add at least one installed model on the Parameters tab")
    have = installed_packages()
    cfg_models = []
    try:
        cfg_models = server_config(inst, enabled)
    except ValueError as e:
        errors.append(str(e))
    for m in enabled:
        if m["package"] not in have:
            errors.append(f"{m['package']} is not installed (Parameters tab -> Models)")
    for c in cfg_models:
        if c["path"] and not Path(c["path"]).exists():
            errors.append(f"{c['id']}: {c['path']} does not exist")

    env = dict(GGML_VK_ALLOW_SYSMEM_FALLBACK="0")
    esp = espeak_env()
    if esp:
        env.update(esp)
    elif esp is None:
        warnings.append("no eSpeak-ng: Kokoro, Kitten, Piper and sanoTTS will fail to speak. "
                        "sudo apt install libespeak-ng1 espeak-ng-data")
    device_index = None
    if backend == "cpu":
        # the Vulkan build still opens every ICD at start, the XTX and the
        # iGPU included; with no ICD it runs CPU-only and touches no card
        env.update(VK_DRIVER_FILES="/nonexistent.json", VK_ICD_FILENAMES="/nonexistent.json")
    if backend == "vulkan" and len(devs) == 1:
        venv, _m, verr = P.vulkan_pinning(devs, None, check_runtime=False)
        errors += verr
        env.update({k: val for k, val in venv.items() if k.startswith(("VK_", "GGML_"))})
        device_index = 0
        if bindir and not verr:
            bad = _vk_check(bindir, env, devs[0])
            if bad:
                errors.append(bad)

    # memory: with one model resident at a time the peak is the largest one
    sizes = []
    for c in cfg_models:
        try:
            p = Path(c["path"])
            sizes.append(p.stat().st_size if p.is_file() else P._dir_bytes(p))
        except OSError:
            pass
    sizes = sorted((s // 1048576 for s in sizes), reverse=True)
    keep = int(v.get("AC_MAX_LOADED") or 0)
    peak_mib = sum(sizes[:keep] if keep else sizes)
    avail = P._meminfo_mb("MemAvailable")
    floor = int(v.get("RAM_FLOOR_MB") or 2048)
    if backend == "cpu" and avail and peak_mib and peak_mib + floor > avail:
        errors.append(f"the resident models need ~{peak_mib} MiB of host RAM; only {avail} MiB "
                      f"is available with the {floor} MiB floor - this box has hard-locked from "
                      "running out of RAM")
    if devs and backend != "cpu":
        free = P.sdcpp._free_vram_mib(devs[0])
        need = int(peak_mib * 1.3) + 512
        if free is not None and peak_mib and need > free:
            other = [s.get("instance") or f"pid {s['pid']}" for s in P.list_servers()
                     if any(g.get("pdev") == devs[0]["pci"] for g in s.get("gpu") or [])]
            warnings.append(f"{P._short_gpu(devs[0]['name'])} has ~{free} MiB free"
                            + (f" (in use by {', '.join(map(str, other))})" if other else "")
                            + f"; the largest resident model set needs ~{need} MiB. With "
                            "GGML_VK_ALLOW_SYSMEM_FALLBACK=0 a load that does not fit fails "
                            "(the request errors) rather than running from host RAM.")

    port = int(v.get("PORT") or 0)
    for other in P.instance_ids():
        if other == inst["id"]:
            continue
        try:
            o = P.get_instance(other)
        except ValueError:
            continue
        if str(P._read_env_file(o["dir"] / "params.env").get("PORT") or
               (P.DEFAULTS["PORT"] if o["legacy"] else "")) == str(port):
            warnings.append(f"PORT {port} is shared with instance '{other}'")

    cfg_file = str(Path(inst["rundir"]) / "audiocpp-server.json")
    cfg = dict(host=str(v.get("HOST") or "127.0.0.1"), port=port,
               lazy_load=_truthy(v.get("AC_LAZY")), models=cfg_models)
    argv = [f"{bindir or '<no build>'}/audiocpp_server", "--config", cfg_file,
            "--host", cfg["host"], "--port", str(port),
            "--backend", backend if devs or backend == "cpu" else "cpu",
            "--threads", str(v.get("AC_THREADS") or 4),
            "--max-loaded-models", str(v.get("AC_MAX_LOADED") or 0),
            "--idle-unload-ms", str(int(v.get("AC_IDLE_UNLOAD_S") or 0) * 1000),
            "--min-free-memory-mb", str(v.get("AC_MIN_FREE_MB") or 0),
            "--busy-timeout-ms", str(int(v.get("AC_BUSY_TIMEOUT_S") or 0) * 1000),
            "--ui" if _truthy(v.get("AC_UI")) else "--no-ui"]
    if device_index is not None:
        argv += ["--device", str(device_index)]
    if _truthy(v.get("AC_LOG")):
        argv.append("--log")
    try:
        argv += shlex.split(str(v.get("AC_EXTRA") or ""))
    except ValueError as e:
        errors.append(f"AC_EXTRA does not parse: {e}")
    if str(v.get("HOST")) in ("0.0.0.0", "::"):
        warnings.append("audiocpp_server has no authentication and is bound to every interface")
    return dict(engine="audio.cpp", argv=argv, env=env, errors=errors, warnings=warnings,
                backend=backend, bindir=bindir, rundir=str(inst["rundir"]),
                config_file=cfg_file, config=cfg,
                device=dict(name=devs[0]["name"] if devs else "CPU",
                            pci=devs[0]["pci"] if devs else "cpu"),
                placement=[f"vulkan{device_index}" if device_index is not None else "cpu"],
                weights_mib=sum(sizes), peak_mib=peak_mib, host_ram_mib=peak_mib if backend == "cpu" else 0,
                vulkan_map=[], template_copy=None, api_key=None)


def _truthy(x):
    return str(x).strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# the running server
# --------------------------------------------------------------------------
def _http(host, port, path, body=None, timeout=10):
    h = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    req = urllib.request.Request(f"http://{h}:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read() or b"{}")
        except ValueError:
            detail = {}
        return dict(_error=e.code, detail=detail)
    except Exception:
        return None


def _err_text(r):
    d = (r or {}).get("detail") or r or {}
    e = d.get("error") if isinstance(d, dict) else None
    return (e.get("message") if isinstance(e, dict) else e) or str(d)[:300]


def describe(pid, argv):
    """/api/servers row for an audiocpp_server process."""
    g = lambda *names: P._argv_get(argv, names)
    host, port = g("--host") or "127.0.0.1", g("--port") or "8080"
    exe = os.path.realpath(f"/proc/{pid}/exe") if os.path.exists(f"/proc/{pid}/exe") else argv[0]
    h = _http(host, port, "/health", timeout=1.5)
    ms = _http(host, port, "/v1/models", timeout=1.5) or {}
    loaded = [m["id"] for m in ms.get("data") or [] if m.get("loaded")]
    names = [m["id"] for m in ms.get("data") or []]
    health = "ok" if isinstance(h, dict) and h.get("status") == "ok" else \
        ("no answer" if h is None else "loading")
    return dict(engine="audio.cpp", model=", ".join(names), model_name=", ".join(loaded) or
                (f"{len(names)} model(s), none loaded" if names else None),
                binary=exe, host=host, port=int(port) if str(port).isdigit() else port,
                backend=g("--backend") or "cpu", health=health, ctx=None, kv=None, ngl=None,
                parallel=None, spec=None, mmproj=None, cache_ram=None, draft=None, alias=None,
                batch=None, ubatch=None, device=g("--device"), log_file=None, slots=None,
                slots_busy=None, live_tps=None, visible_devices={})


def _server_addr(inst):
    with P.using_instance(inst):
        pid = P.server_pid()
    if not pid:
        raise ValueError(f"{inst['id']} is not running - press Start")
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    return (P._argv_get(argv, ("--host",)) or "127.0.0.1",
            P._argv_get(argv, ("--port",)) or "8080")


def health(inst):
    try:
        host, port = _server_addr(inst)
    except ValueError:
        return None
    h = _http(host, port, "/health", timeout=2)
    if h is None:
        return dict(status="loading")
    if h.get("_error"):
        return dict(status="error", code=h["_error"])
    ms = _http(host, port, "/v1/models", timeout=2) or {}
    return dict(status="ok", models=[dict(id=m["id"], loaded=m.get("loaded"), task=m.get("task"))
                                     for m in ms.get("data") or []])


def unload_all(inst):
    host, port = _server_addr(inst)
    r = _http(host, port, "/v1/tasks/unload_all_models", {}, timeout=60)
    if r is None or r.get("_error"):
        raise ValueError(f"unload failed: {_err_text(r)}")
    return r


def voices(inst, model):
    host, port = _server_addr(inst)
    r = _http(host, port, "/v1/audio/voices?model=" + urllib.request.quote(model), timeout=10)
    return (r or {}).get("voices") or []


# --------------------------------------------------------------------------
# uploads, runs, outputs
# --------------------------------------------------------------------------
_jobs = {}
_jlock = threading.Lock()


def _dir(kind, iid):
    d = home() / kind / iid
    d.mkdir(parents=True, exist_ok=True)
    return d


def upload(inst, body):
    name = str(body.get("name") or "audio.wav")
    try:
        data = base64.b64decode(str(body.get("data") or "").split(",", 1)[-1], validate=False)
    except ValueError:
        raise ValueError("upload is not base64")
    if not data:
        raise ValueError("empty upload")
    if len(data) > UPLOAD_MAX:
        raise ValueError(f"upload is over {UPLOAD_MAX // 1048576} MB")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{name} is not a WAV file; audio.cpp reads WAV input")
    fn = f"{uuid.uuid4().hex[:12]}.wav"
    (_dir("uploads", inst["id"]) / fn).write_bytes(data)
    return dict(ok=True, file=fn, bytes=len(data), original=name)


def _upload_path(iid, fn):
    if not fn:
        return None
    if not re.match(r"^[0-9a-f]{12}\.wav$", fn):
        raise ValueError("bad upload name")
    p = _dir("uploads", iid) / fn
    if not p.is_file():
        raise ValueError("that upload is gone - upload it again")
    return str(p)


def run(inst, body):
    host, port = _server_addr(inst)
    model = str(body.get("model") or "")
    m = next((x for x in load_models(inst) if x["package"] == model and x.get("enabled", True)), None)
    if not m:
        raise ValueError(f"{model!r} is not an enabled model of this instance")
    fam, d, _p = _package(model)
    req = {}
    for k in ("text", "voice_id", "language", "reference_text", "lyrics", "emotion",
              "target_text", "style_ref_text"):
        v = body.get(k)
        if v not in (None, ""):
            req[k] = str(v)
    for k in ("speaking_rate", "duration_seconds"):
        if body.get(k) not in (None, ""):
            req[k] = float(body[k])
    for k in ("voice_ref", "audio", "source_audio", "target_voice"):
        pth = _upload_path(inst["id"], body.get(k))
        if pth:
            req[k] = pth
    req["options"] = _clean_opts((d.get("options") or {}).get("request"), body.get("options"),
                                 "request")
    if not (req.get("text") or req.get("audio") or req.get("lyrics")):
        raise ValueError("nothing to run: give text, lyrics or an input audio file")
    jid = uuid.uuid4().hex[:12]
    rec = dict(id=jid, instance=inst["id"], model=model, family=fam, task=m["task"],
               status="running", submitted=time.time(), finished=None, request=req,
               files=[], text=None, timing=None, error=None)
    with _jlock:
        _jobs[jid] = rec
        for old in sorted(_jobs, key=lambda k: _jobs[k]["submitted"])[:-50]:
            _jobs.pop(old, None)

    def go():
        r = _http(host, port, "/v1/tasks/run", dict(model=model, request=req), timeout=3600)
        if r is None:
            rec.update(status="failed", error="audiocpp_server did not answer (crashed, or "
                       "the run took over an hour)", finished=time.time())
            return
        if r.get("_error") or r.get("error"):
            rec.update(status="failed", error=_err_text(r if r.get("_error") else dict(detail=r)),
                       finished=time.time())
            return
        out = _dir("outputs", inst["id"])
        stamp = time.strftime("%Y%m%d_%H%M%S")
        clips = []
        if r.get("audio"):
            clips.append(("", r["audio"]))
        for n in r.get("named_audio_outputs") or []:
            clips.append((re.sub(r"[^\w-]", "_", str(n.get("id") or "out"))[:32], n.get("audio")))
        for suffix, b64 in clips:
            if not b64:
                continue
            fn = f"{stamp}_{jid}{'_' + suffix if suffix else ''}.wav"
            (out / fn).write_bytes(base64.b64decode(b64))
            rec["files"].append(fn)
        if r.get("text"):
            rec["text"] = r["text"]
            (out / f"{stamp}_{jid}.txt").write_text(r["text"])
        rec["timing"] = r.get("timing")
        rec.update(status="completed", finished=time.time())
        meta = dict(model=model, family=fam, task=m["task"], request=req, files=rec["files"],
                    text=rec["text"], timing=rec["timing"],
                    seconds=round(rec["finished"] - rec["submitted"], 1),
                    created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (out / f"{stamp}_{jid}.json").write_text(json.dumps(meta, indent=1))

    threading.Thread(target=go, daemon=True).start()
    return _public(rec)


def _public(rec):
    r = dict(rec)
    r["elapsed_s"] = round((rec["finished"] or time.time()) - rec["submitted"], 1)
    return r


def job(jid):
    with _jlock:
        rec = _jobs.get(jid)
    if not rec:
        raise ValueError("no such job")
    return _public(rec)


def outputs(iid, limit=30):
    d = _dir("outputs", iid)
    items = []
    for m in sorted(d.glob("*.json"), reverse=True)[:limit]:
        try:
            meta = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        rq = meta.get("request") or {}
        items.append(dict(files=meta.get("files", []), text=meta.get("text"),
                          model=meta.get("model"), task=meta.get("task"),
                          prompt=rq.get("text") or rq.get("lyrics"),
                          voice=rq.get("voice_id"), seconds=meta.get("seconds"),
                          timing=meta.get("timing"), created=meta.get("created")))
    active = [_public(j) for j in _jobs.values() if j["instance"] == iid and not j["finished"]]
    return dict(items=items, active=active)


def output_path(iid, name):
    if not re.match(r"^[\w.-]+\.(wav|txt)$", name or ""):
        raise ValueError("bad file name")
    f = _dir("outputs", iid) / name
    if not f.is_file():
        raise ValueError("no such file")
    return f
