#!/usr/bin/env python3
"""
Inference engines: installed builds, upstream releases and a daily updater
(added 2026-09-23).

The panel started as a llama.cpp panel. Image, video and audio generation
bring more engines, and each one needs the same three things: which builds
are on disk, which one each backend launches from, and a way to fetch new
ones. This module keeps those per ENGINE rather than hard-wiring llama.cpp.

  llama.cpp       wraps the panel's existing build code (builds.env, ~/llama),
                  so main's bash scripts keep reading exactly what they read.
  sd.cpp          stable-diffusion.cpp release zips, unpacked to ~/sdcpp.
  camelid         Camelid (Rust GGUF engine) tagged releases, ~/camelid; the
                  download is checked against the published .sha256.
  audio.cpp       audio.cpp tagged release tarballs (vX.Y.Z), unpacked to
                  ~/audiocpp. Linux ships CPU and Vulkan only (added
                  2026-09-23).
                  Upstream publishes a release on EVERY master commit (several
                  a day) and ships Linux Vulkan/ROCm/CPU builds but NO Linux
                  CUDA build. The Vulkan build drives the RTX 2060 through the
                  NVIDIA driver's own Vulkan ICD, with no toolkit to install.

Updater policy per engine (panel/engine-updates.json):
  enabled   off by default; nothing is fetched until the operator turns it on
  at        "HH:MM" UTC, once per day
  flavours  which builds to keep current, e.g. ["vulkan"]
  mode      "hold": download and verify, then wait for the operator to press
                    Activate (default)
            "auto": also make it the active build. Running servers are NEVER
                    restarted; the new build is used at their next start.
  keep      prune to this many updater-installed builds per flavour (0 = never
            prune). Only builds the updater itself installed are candidates;
            an active build or one a running process was launched from is
            never removed.
"""
import json, os, re, shutil, subprocess, tarfile, threading, time, urllib.request, zipfile
from pathlib import Path

P = None                        # the panel module, bound by panel.py
ENGINES = {}
UPDATES_FILE = STATE_FILE = ACTIVE_FILE = None
_lock = threading.RLock()
_run_lock = threading.Lock()    # one update run at a time, scheduled or manual
_rel_cache = {}

DEFAULT_POLICY = dict(enabled=False, at="04:30", flavours=[], mode="hold", keep=0)
MARKER = ".LexiPanel-build.json"


def bind(panel_module):
    global P, UPDATES_FILE, STATE_FILE, ACTIVE_FILE
    P = panel_module
    UPDATES_FILE = P.PANEL / "engine-updates.json"
    STATE_FILE = P.PANEL / "engine-updates-state.json"
    ACTIVE_FILE = P.PANEL / "engine-builds.json"
    for e in (LlamaCpp(), SdCpp(), AudioCpp(), Camelid()):
        ENGINES[e.id] = e


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def github_releases(repo, limit=12):
    """Raw release list, cached 10 min per repo (unauthenticated: 60 req/h/IP)."""
    now = time.time()
    with _lock:
        c = _rel_cache.get(repo)
        if c and now - c[0] < 600:
            return c[1]
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases?per_page={int(limit)}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "LexiPanel-panel"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = json.load(r)
    with _lock:
        _rel_cache[repo] = (now, raw)
    return raw


def _running_binaries():
    """Every executable path and argv[0] of every process we can read."""
    out = set()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            out.add(os.path.realpath(d / "exe"))
        except OSError:
            pass
        try:
            a0 = (d / "cmdline").read_bytes().split(b"\0", 1)[0].decode(errors="ignore")
            if a0:
                out.add(os.path.realpath(a0))
        except OSError:
            pass
    return out


def _in_use(bindir):
    bd = os.path.realpath(bindir)
    return any(b.startswith(bd + os.sep) for b in _running_binaries())


# ============================================================================
# engines
# ============================================================================
class Engine:
    id = label = repo = ""
    flavours = ()
    notes = ""

    def describe(self):
        return dict(id=self.id, label=self.label, repo=self.repo,
                    flavours=list(self.flavours), notes=self.notes)

    # subclasses implement these
    def releases(self):            raise NotImplementedError
    def installed(self):           raise NotImplementedError
    def active(self, flavour):     raise NotImplementedError
    def activate(self, flavour, bindir): raise NotImplementedError
    def delete(self, bindir):      raise NotImplementedError
    def install(self, tag, flavour, by="operator"): raise NotImplementedError

    def release(self, tag):
        rel = next((r for r in self.releases() if r["tag"] == tag), None)
        if not rel:
            raise ValueError(f"{self.label}: no release {tag!r} among the recent ones")
        return rel


class LlamaCpp(Engine):
    """Thin adapter over panel.py's original llama.cpp build code."""
    id, label, repo = "llama.cpp", "llama.cpp", "ggml-org/llama.cpp"
    flavours = ("vulkan", "rocm", "cuda", "cpu")
    notes = ("Active builds are recorded in panel/builds.env, which main's launch "
             "scripts source. A new build is used the next time a server on that "
             "backend starts.")

    def releases(self):
        return P.github_releases()

    def installed(self):
        out = []
        for b in P.scan_builds():
            out.append(dict(b, tag=f"b{b['build']}", engine=self.id))
        return out

    def active(self, flavour):
        return P.active_build(flavour)

    def activate(self, flavour, bindir):
        P.set_active_build(flavour, bindir)

    def delete(self, bindir):
        return P.delete_build(bindir)

    def install(self, tag, flavour, by="operator"):
        rel = self.release(tag)
        a = rel["flavours"].get(flavour)
        if not a:
            raise ValueError(f"{tag} has no {flavour} build for this box")
        rec = P.start_build_install(a["url"], flavour, rel["build"], a.get("runtime_url"))
        rec["engine"] = self.id
        return rec


class ReleaseEngine(Engine):
    """An engine installed from GitHub release archives into ~/<root_name>/,
    one directory per (release, flavour), each holding a server binary.

    Subclasses say how to read a release (parse_release), what a directory
    is (_flavour_of), how to name it (dest_name) and how to prove an
    unpacked build runs (verify). Download, unpack, activate and delete are
    shared, so every engine gets the same guards."""
    root_name = server = ""
    archive = "zip"                 # "zip" or "tar"
    dir_build_re = None             # build number from a dir name with no marker

    @property
    def root(self):
        return P.HOME / self.root_name

    def parse_release(self, rel):   raise NotImplementedError
    def dest_name(self, rel, flavour): raise NotImplementedError
    def verify(self, bd, flavour):  raise NotImplementedError

    def releases(self):
        out = [r for r in (self.parse_release(rel) for rel in github_releases(self.repo)) if r]
        out.sort(key=lambda r: -r["build"])
        return out

    def _flavour_of(self, d):
        raise NotImplementedError

    def installed(self):
        act = _read_json(ACTIVE_FILE, {}).get(self.id, {})
        out = []
        try:
            dirs = sorted(d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith("."))
        except OSError:
            dirs = []
        for d in dirs:
            flav = self._flavour_of(d)
            if not flav:
                continue
            meta = _read_json(d / MARKER, {})
            m = self.dir_build_re.match(d.name) if self.dir_build_re else None
            out.append(dict(engine=self.id, dir=str(d), bindir=str(d), name=d.name,
                            tag=meta.get("tag") or d.name,
                            build=int(meta.get("build") or (m.group(1) if m else 0)),
                            flavour=flav, size_mb=round(P._dir_bytes(d) / 1048576, 1),
                            installed=meta.get("installed"), by=meta.get("by"),
                            active=act.get(flav) == str(d), chosen=act.get(flav) == str(d)))
        out.sort(key=lambda r: (-r["build"], r["flavour"]))
        for flav in {r["flavour"] for r in out}:
            if not any(r["chosen"] for r in out if r["flavour"] == flav):
                first = next(r for r in out if r["flavour"] == flav)
                first["active"] = True                 # newest installed = default
                first["default"] = True
        return out

    def active(self, flavour):
        """The chosen build for this flavour; with none chosen, the newest one
        installed, so a first install works without a trip to the Builds tab."""
        return self.chosen(flavour) or self.default(flavour)

    def default(self, flavour):
        return next((b["bindir"] for b in self.installed() if b["flavour"] == flavour), None)

    def chosen(self, flavour):
        bd = _read_json(ACTIVE_FILE, {}).get(self.id, {}).get(flavour)
        return bd if bd and Path(bd, self.server).is_file() else None

    def activate(self, flavour, bindir):
        if flavour not in self.flavours:
            raise ValueError(f"unknown flavour {flavour!r}")
        got = self._flavour_of(bindir)
        if got != flavour:
            raise ValueError(f"{bindir} is a {got or 'non-' + self.id} build, not {flavour}")
        with _lock:
            cur = _read_json(ACTIVE_FILE, {})
            cur.setdefault(self.id, {})[flavour] = str(Path(bindir))
            _atomic_json(ACTIVE_FILE, cur)

    def delete(self, bindir):
        d = Path(bindir).resolve()
        if d.parent != self.root.resolve():
            raise ValueError(f"build is not under ~/{self.root_name}")
        for f in self.flavours:
            if self.chosen(f) == str(d):
                raise ValueError(f"that build is the active {f} build; activate another first")
        if _in_use(d):
            raise ValueError("a running process was launched from that build")
        shutil.rmtree(d)
        return dict(deleted=str(d))

    def _unpack(self, apath, stage):
        stage.mkdir()
        if self.archive == "zip":
            with zipfile.ZipFile(apath) as z:
                for info in z.infolist():
                    target = (stage / info.filename).resolve()
                    if stage.resolve() not in target.parents:
                        raise RuntimeError(f"archive entry escapes the build dir: {info.filename}")
                    z.extract(info, stage)
                    mode = info.external_attr >> 16
                    if mode and not info.is_dir():
                        os.chmod(target, mode & 0o755)
        else:
            with tarfile.open(apath) as t:
                # the "data" filter refuses absolute paths, .. and links out of stage
                t.extractall(stage, filter="data")

    def install(self, tag, flavour, by="operator"):
        rel = self.release(tag)
        a = rel["flavours"].get(flavour)
        if not a:
            raise ValueError(f"{tag} has no Linux {flavour} build")
        if not re.match(r"^https://github\.com/", a["url"]):
            raise ValueError("asset url is not a github.com release asset")
        dest = self.root / self.dest_name(rel, flavour)
        if dest.exists():
            raise ValueError(f"{dest.name} is already installed")
        self.root.mkdir(exist_ok=True)
        did = f"{self.root_name}build-" + str(int(time.time() * 1000))
        apath = self.root / f".dl-{did}.{'zip' if self.archive == 'zip' else 'tar.gz'}"
        stage = self.root / f".stage-{did}"
        rec = dict(id=did, url=a["url"], name=f"{self.id} {rel['tag']} {flavour}", dest=str(dest),
                   kind="build", engine=self.id, flavour=flavour, build=rel["build"],
                   tag=rel["tag"], status="running", pct=0.0, downloaded=0,
                   total=int(a["size_mb"] * 1048576), started=time.time(), error=None)
        with P._lock:
            P._downloads[did] = rec

        def run():
            try:
                p = subprocess.Popen(["curl", "-fL", "--retry", "3", "-o", str(apath), a["url"]],
                                     stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
                while p.poll() is None:
                    time.sleep(1)
                    try:
                        rec["downloaded"] = apath.stat().st_size
                        rec["pct"] = round(rec["downloaded"] * 100 / max(1, rec["total"]), 1)
                    except FileNotFoundError:
                        pass
                if p.returncode != 0:
                    raise RuntimeError(p.stderr.read().decode(errors="ignore")[-400:]
                                       or f"curl exit {p.returncode}")
                if a.get("sha256_url"):                   # upstream publishes one: check it
                    rec["status"] = "verifying"
                    with urllib.request.urlopen(a["sha256_url"], timeout=30) as r:
                        want = r.read().decode().split()[0].lower()
                    import hashlib
                    h = hashlib.sha256()
                    with open(apath, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            h.update(chunk)
                    if h.hexdigest() != want:
                        raise RuntimeError("SHA-256 mismatch: the download is not what upstream published")
                rec.update(status="unpacking", pct=100.0)
                self._unpack(apath, stage)
                # accept a single top-level folder as well as the flat layout
                bd = stage
                if not (bd / self.server).is_file():
                    subs = [s for s in stage.iterdir() if s.is_dir() and (s / self.server).is_file()]
                    if not subs:
                        raise RuntimeError(f"archive contains no {self.server}")
                    bd = subs[0]
                rec["status"] = "testing"
                self.verify(bd, flavour)
                (bd / MARKER).write_text(json.dumps(dict(
                    engine=self.id, tag=rel["tag"], build=rel["build"], sha=rel.get("sha"),
                    flavour=flavour, asset=a["name"], installed=_now_iso(), by=by), indent=1))
                got = self._flavour_of(bd)
                if got != flavour:
                    raise RuntimeError(f"archive is a {got} build, expected {flavour}")
                bd.rename(dest)
                rec.update(status="done", bindir=str(dest))
            except Exception as e:
                rec.update(status="failed", error=str(e))
            finally:
                try:
                    apath.unlink()
                except OSError:
                    pass
                shutil.rmtree(stage, ignore_errors=True)

        threading.Thread(target=run, daemon=True).start()
        return rec


class SdCpp(ReleaseEngine):
    id, label, repo = "sd.cpp", "stable-diffusion.cpp", "leejet/stable-diffusion.cpp"
    flavours = ("vulkan", "rocm", "cpu")
    notes = ("Upstream publishes a build on every commit, several a day. There is no "
             "Linux CUDA build: the Vulkan build runs on both the 7900 XTX (RADV) and "
             "the RTX 2060 (NVIDIA's Vulkan driver).")
    root_name, server, archive = "sdcpp", "sd-server", "zip"
    dir_build_re = re.compile(r"^sd-(\d+)-")
    ASSET_RE = {
        "vulkan": re.compile(r"^sd-master-([0-9a-f]+)-bin-Linux-Ubuntu-[\d.]+-x86_64-vulkan\.zip$"),
        "rocm":   re.compile(r"^sd-master-([0-9a-f]+)-bin-Linux-Ubuntu-[\d.]+-x86_64-rocm-[\d.]+\.zip$"),
        "cpu":    re.compile(r"^sd-master-([0-9a-f]+)-bin-Linux-Ubuntu-[\d.]+-x86_64\.zip$"),
    }
    TAG_RE = re.compile(r"^master-(\d+)-([0-9a-f]+)$")
    FLAVOUR_LIB = (("libggml-vulkan.so", "vulkan"), ("libggml-hip.so", "rocm"),
                   ("libggml-cuda.so", "cuda"))

    def parse_release(self, rel):
        m = self.TAG_RE.match(rel.get("tag_name") or "")
        if not m:
            return None
        fl = {}
        for a in rel.get("assets", []):
            for flav, rx in self.ASSET_RE.items():
                if rx.match(a["name"]):
                    fl[flav] = dict(name=a["name"], url=a["browser_download_url"],
                                    size_mb=round(a["size"] / 1048576, 1))
        return fl and dict(tag=rel["tag_name"], build=int(m.group(1)), sha=m.group(2),
                           published=rel.get("published_at"), flavours=fl) or None

    def dest_name(self, rel, flavour):
        return f"sd-{rel['build']}-{rel['sha']}-{flavour}"

    def _flavour_of(self, d):
        try:
            names = {p.name for p in Path(d).iterdir()}
        except OSError:
            return None
        if "sd-server" not in names:
            return None
        for lib, flav in self.FLAVOUR_LIB:
            if lib in names:
                return flav
        return "cpu"

    def verify(self, bd, flavour):
        t = subprocess.run([str(bd / "sd-server"), "--help"], capture_output=True,
                           text=True, timeout=60,
                           env=dict(os.environ, LD_LIBRARY_PATH=str(bd)))
        if "Usage" not in (t.stdout + t.stderr):
            raise RuntimeError("sd-server --help did not run: " + (t.stderr or t.stdout)[-300:])


class Camelid(ReleaseEngine):
    id, label, repo = "camelid", "Camelid", "timtoole02/Camelid"
    flavours = ("linux",)
    notes = ("Rust-native GGUF engine with its own chat/agent web UI and an OpenAI-compatible "
             "API. The Linux build carries CUDA and falls back to the CPU; there is no Vulkan "
             "or ROCm, so on AMD cards it runs on the CPU. Supports a curated list of models "
             "(see its catalog on the Parameters tab). Downloads are checked against "
             "upstream's published SHA-256.")
    root_name, server, archive = "camelid", "camelid", "tar"
    TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
    ASSET = "camelid-linux-x86_64.tar.gz"

    def parse_release(self, rel):
        tag = rel.get("tag_name") or ""
        m = self.TAG_RE.match(tag)
        if not m:
            return None
        assets = {a["name"]: a for a in rel.get("assets", [])}
        a = assets.get(self.ASSET)
        if not a:
            return None
        sha = assets.get(self.ASSET + ".sha256")
        build = int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3))
        return dict(tag=tag, build=build, sha=None, published=rel.get("published_at"),
                    flavours={"linux": dict(name=a["name"], url=a["browser_download_url"],
                                            size_mb=round(a["size"] / 1048576, 1),
                                            sha256_url=sha and sha["browser_download_url"])})

    def dest_name(self, rel, flavour):
        return f"cm-{rel['tag']}-{flavour}"

    def _flavour_of(self, d):
        return "linux" if (Path(d) / self.server).is_file() else None

    def verify(self, bd, flavour):
        os.chmod(bd / self.server, 0o755)
        t = subprocess.run([str(bd / self.server), "--version"], capture_output=True,
                           text=True, timeout=60, env=dict(os.environ, LD_LIBRARY_PATH=str(bd)))
        if not (t.stdout + t.stderr).lower().startswith("camelid"):
            raise RuntimeError("camelid --version did not run: " + (t.stderr or t.stdout)[-300:])


class AudioCpp(ReleaseEngine):
    id, label, repo = "audio.cpp", "audio.cpp", "0xShug0/audio.cpp"
    flavours = ("vulkan", "cpu")
    notes = ("Speech (TTS, voice cloning, ASR), music and sound effects on ggml. Tagged "
             "releases every few days. Linux ships CPU and Vulkan builds only; the Vulkan "
             "build also carries the CPU backend. Upstream tunes for CUDA, so Vulkan "
             "coverage and speed vary by model.")
    root_name, server, archive = "audiocpp", "audiocpp_server", "tar"
    dir_build_re = None
    TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
    BINARIES = ("audiocpp_server", "audiocpp_cli", "audiocpp_gguf")

    def parse_release(self, rel):
        tag = rel.get("tag_name") or ""
        m = self.TAG_RE.match(tag)
        if not m:
            return None
        fl = {}
        for a in rel.get("assets", []):
            for flav in self.flavours:
                # the plain build, not "-portable" (a dynamic-CPU-variant repack)
                if a["name"] == f"audio-{tag}-bin-ubuntu-x64-{flav}.tar.gz":
                    fl[flav] = dict(name=a["name"], url=a["browser_download_url"],
                                    size_mb=round(a["size"] / 1048576, 1))
        build = int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3))
        return fl and dict(tag=tag, build=build, sha=None,
                           published=rel.get("published_at"), flavours=fl) or None

    def dest_name(self, rel, flavour):
        return f"ac-{rel['tag']}-{flavour}"

    def _flavour_of(self, d):
        d = Path(d)
        if not (d / self.server).is_file():
            return None
        # the Linux builds link ggml statically, so there is no backend .so to
        # look at: trust the install marker, then the directory name
        flav = _read_json(d / MARKER, {}).get("flavour")
        if flav in self.flavours:
            return flav
        return next((f for f in self.flavours if d.name.endswith("-" + f)), None)

    def verify(self, bd, flavour):
        for b in self.BINARIES:
            if (bd / b).is_file():
                os.chmod(bd / b, 0o755)       # the tarball ships them 0644
        t = subprocess.run([str(bd / self.server), "--version"], capture_output=True,
                           text=True, timeout=60)
        out = t.stdout + t.stderr
        if "audio.cpp" not in out:
            raise RuntimeError(f"{self.server} --version did not run: " + out[-300:])
        backends = re.search(r"^backends:\s*(.*)$", out, re.M)
        if flavour == "vulkan" and "vulkan" not in (backends.group(1) if backends else ""):
            raise RuntimeError(f"build reports backends {backends and backends.group(1)!r}, "
                               "no vulkan")


# ============================================================================
# catalog API
# ============================================================================
def get(eid):
    e = ENGINES.get(eid)
    if not e:
        raise ValueError(f"unknown engine {eid!r}")
    return e


def policy(eid):
    p = dict(DEFAULT_POLICY)
    p.update(_read_json(UPDATES_FILE, {}).get(eid, {}))
    return p


def set_policy(eid, body):
    e = get(eid)
    p = policy(eid)
    if "enabled" in body:
        p["enabled"] = bool(body["enabled"])
    if "at" in body:
        at = str(body["at"]).strip()
        if not re.match(r"^([01]\d|2[0-3]):[0-5]\d$", at):
            raise ValueError("time must be HH:MM (24h, UTC)")
        p["at"] = at
    if "flavours" in body:
        fl = [str(f) for f in body["flavours"] or []]
        bad = [f for f in fl if f not in e.flavours]
        if bad:
            raise ValueError(f"{e.label} has no {', '.join(bad)} build")
        p["flavours"] = fl
    if "mode" in body:
        if body["mode"] not in ("hold", "auto"):
            raise ValueError("mode must be hold or auto")
        p["mode"] = body["mode"]
    if "keep" in body:
        k = int(body["keep"])
        if k < 0 or k > 50:
            raise ValueError("keep must be 0 (never prune) to 50")
        if k == 1:
            raise ValueError("keep 1 would leave nothing to roll back to; use 0 or 2+")
        p["keep"] = k
    if p["enabled"] and not p["flavours"]:
        raise ValueError("pick at least one flavour to keep current")
    with _lock:
        cur = _read_json(UPDATES_FILE, {})
        cur[eid] = p
        _atomic_json(UPDATES_FILE, cur)
    return p


def _state():
    return _read_json(STATE_FILE, {})


def _save_state(st):
    with _lock:
        _atomic_json(STATE_FILE, st)


def _held(eid):
    """Builds the updater fetched in hold mode that are still present and not active."""
    e = get(eid)
    st = _state().get(eid, {})
    inst = {b["bindir"]: b for b in e.installed()}
    out = []
    for h in st.get("held", []):
        b = inst.get(h["bindir"])
        if b and getattr(e, "chosen", e.active)(b["flavour"]) != b["bindir"]:
            out.append(dict(h, build=b["build"], tag=b["tag"]))
    return out


def describe_all():
    out = []
    for eid, e in ENGINES.items():
        try:
            inst = e.installed()
        except Exception as ex:
            inst, err = [], str(ex)
        else:
            err = None
        st = _state().get(eid, {})
        out.append(dict(e.describe(), installed=inst, error=err,
                        active={f: e.active(f) for f in e.flavours},
                        policy=policy(eid), held=_held(eid),
                        last_run=st.get("last_run"), last_result=st.get("last_result"),
                        log=st.get("log", [])[-30:], running=_run_lock.locked()))
    return out


def releases(eid):
    e = get(eid)
    rels = e.releases()
    have = {(b["build"], b["flavour"]) for b in e.installed()}
    for r in rels:
        r["installed"] = [f for f in r["flavours"] if (r["build"], f) in have]
    return rels


def install(eid, tag, flavour):
    e = get(eid)
    return e.install(tag, flavour)


def activate(eid, flavour, bindir):
    e = get(eid)
    e.activate(flavour, bindir)
    st = _state()
    s = st.setdefault(eid, {})
    s["held"] = [h for h in s.get("held", []) if h["bindir"] != str(bindir)]
    _save_state(st)
    return dict(ok=True, active=e.active(flavour))


def delete(eid, bindir):
    return get(eid).delete(bindir)


# ============================================================================
# updater
# ============================================================================
def _wait_download(rec, timeout=3600):
    t0 = time.time()
    while rec["status"] not in ("done", "failed") and time.time() - t0 < timeout:
        time.sleep(2)
    if rec["status"] != "done":
        raise RuntimeError(rec.get("error") or f"download still {rec['status']} after {timeout}s")
    return rec["bindir"]


def run_update(eid, reason="manual"):
    """Bring each chosen flavour of one engine up to its newest release."""
    e = get(eid)
    if not _run_lock.acquire(blocking=False):
        raise ValueError("an update run is already in progress")
    log = []
    # This run's view of its own bookkeeping; merged back into the state file
    # at the end so a concurrent Activate from the UI is not overwritten.
    mine = dict(held=[], installed_by_updater=[], last_result="error")

    def say(msg):
        log.append(f"{_now_iso()} {msg}")

    try:
        pol = policy(eid)
        say(f"{reason} check of {e.label} ({', '.join(pol['flavours']) or 'no flavours'}, "
            f"mode {pol['mode']})")
        try:
            rels = e.releases()
        except Exception as ex:
            say(f"could not list releases: {ex}")
            return log
        have = {(b["build"], b["flavour"]): b for b in e.installed()}
        changed = False
        for flav in pol["flavours"]:
            rel = next((r for r in rels if flav in r["flavours"]), None)
            if not rel:
                say(f"{flav}: no recent release carries a {flav} build")
                continue
            if (rel["build"], flav) in have:
                cur = have[(rel["build"], flav)]
                say(f"{flav}: newest is {rel['tag']}, already installed")
                if pol["mode"] == "auto" and e.active(flav) != cur["bindir"]:
                    e.activate(flav, cur["bindir"])
                    say(f"{flav}: activated {cur['name']}")
                continue
            if pol["mode"] == "hold" and hasattr(e, "chosen") and not e.chosen(flav) \
                    and e.default(flav):
                # with nothing chosen, the newest build is the default - pin the
                # current one so the download waits for Activate as promised
                e.activate(flav, e.default(flav))
                say(f"{flav}: pinned {Path(e.default(flav)).name} as the active build")
            say(f"{flav}: installing {rel['tag']}")
            try:
                bindir = _wait_download(e.install(rel["tag"], flav, by="updater"))
            except Exception as ex:
                say(f"{flav}: install failed: {ex}")
                continue
            changed = True
            mine["installed_by_updater"].append(bindir)
            if pol["mode"] == "auto":
                try:
                    e.activate(flav, bindir)
                    say(f"{flav}: {rel['tag']} is now active; running servers keep their "
                        "current build until their next start")
                except Exception as ex:
                    say(f"{flav}: installed but could not activate: {ex}")
            else:
                mine["held"].append(dict(flavour=flav, bindir=bindir, tag=rel["tag"],
                                         at=_now_iso()))
                say(f"{flav}: {rel['tag']} installed and verified, waiting for Activate")
        mine["last_result"] = "updated" if changed else "up to date"
        return log
    finally:
        with _lock:
            st = _state()
            s = st.setdefault(eid, {})
            s["held"] = s.get("held", []) + mine["held"]
            s["installed_by_updater"] = sorted(set(s.get("installed_by_updater", []))
                                               | set(mine["installed_by_updater"]))
            s["last_result"] = mine["last_result"]
            s["last_run"] = _now_iso()
            _save_state(st)
            try:
                pol = policy(eid)
                if pol["keep"]:
                    _prune(e, s, pol, say)
            finally:
                s["log"] = (s.get("log", []) + log)[-200:]
                _save_state(st)
        _run_lock.release()


def _prune(e, s, pol, say):
    mine = set(s.get("installed_by_updater", []))
    for flav in pol["flavours"]:
        builds = [b for b in e.installed() if b["flavour"] == flav and b["bindir"] in mine]
        builds.sort(key=lambda b: -b["build"])
        for b in builds[pol["keep"]:]:
            if e.active(flav) == b["bindir"]:
                continue
            try:
                e.delete(b["bindir"])
                mine.discard(b["bindir"])
                say(f"{flav}: pruned {b['name']}")
            except Exception as ex:
                say(f"{flav}: kept {b['name']} ({ex})")
    s["installed_by_updater"] = sorted(mine)


def run_update_async(eid, reason="manual"):
    get(eid)
    if _run_lock.locked():
        raise ValueError("an update run is already in progress")
    threading.Thread(target=lambda: _safe_run(eid, reason), daemon=True).start()
    return dict(ok=True, started=True)


def _safe_run(eid, reason):
    try:
        run_update(eid, reason)
    except Exception:
        pass


def scheduler():
    """Panel thread: once per UTC day at each engine's time, run its update."""
    while True:
        try:
            now = time.gmtime()
            today, hm = time.strftime("%Y-%m-%d", now), time.strftime("%H:%M", now)
            for eid in list(ENGINES):
                pol = policy(eid)
                if not pol["enabled"] or not pol["flavours"]:
                    continue
                st = _state().get(eid, {})
                if st.get("last_scheduled_day") == today or hm < pol["at"]:
                    continue
                full = _state()
                full.setdefault(eid, {})["last_scheduled_day"] = today
                _save_state(full)
                _safe_run(eid, "scheduled")
        except Exception:
            pass
        time.sleep(30)
