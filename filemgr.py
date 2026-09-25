#!/usr/bin/env python3
"""
File manager for the home folder (added 2026-09-23): the Files tab.

Browse, upload (files and whole folders), download (a file, or folders and
multi-selections as a streamed zip), new folder, rename, delete, copy / cut /
paste and drag-to-move, all confined to ROOT.

ROOT is the home folder (operator request, 2026-09-23: models, templates, logs,
backups and uploads all live there). Guard rails, because this is a web page:
  * confinement: every path is ROOT-relative and resolved with symlinks
    followed; anything outside ROOT is refused ("../" or a symlink out).
  * hidden top-level entries (.ssh, .config, .claude, ...) are invisible and
    unreachable: they hold login keys, systemd units and credentials, and
    writing there would bypass the terminal's own sign-in.
  * READ_ONLY top-level folders (panel/) can be browsed and downloaded, never
    changed, so the running panel cannot be broken from inside itself.
  * delete / move / rename refuse anything a process holds open or mapped
    (a running model, a log being written) and any model, projector, draft or
    template an instance's settings or a running server refer to.
ROOT itself can be listed but never renamed, moved or deleted.

Uploads stream straight to a hidden ".upload-*.part" file in the target folder
and are renamed into place only when complete, so a dropped connection never
leaves a half file under the real name. An existing name is never overwritten
silently: the new file gets " (2)", " (3)", ... unless the request asks to
overwrite.
"""
import os, re, shutil, time, uuid, zipfile
from pathlib import Path

P = None
ROOT = None
CHUNK = 1 << 20
FREE_MARGIN = 1 << 30                     # keep 1 GiB free after an upload


def bind(panel_module):
    global P, ROOT
    P = panel_module
    # LEXIPANEL_FILES_ROOT points the Files tab somewhere else (tests, other boxes)
    ROOT = Path(os.environ.get("LEXIPANEL_FILES_ROOT") or P.HOME)


READ_ONLY = ("panel",)                    # top-level folders that are browse/download only
# quick-jump buttons; only those that exist are offered
PLACES = [("Home", ""), ("Uploads", "uploads"), ("Models & templates", "models"),
          ("Image models", "models/sd"), ("Audio models", "audiocpp/models"),
          ("Image outputs", "sdcpp/outputs"), ("Audio outputs", "audiocpp/outputs"),
          ("Logs", "llama_logs"), ("Thermal logs", "rigmon/logs"),
          ("Support bundles", "support-bundles"), ("Backups", "backups"),
          ("Launch scripts", "llama"), ("Panel (read-only)", "panel")]


def places():
    root = _root()
    return [dict(label=l, path=r) for l, r in PLACES if (root / r).is_dir()]


def _root():
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT.resolve()


def resolve(rel, must_exist=True):
    """ROOT-relative path from a request -> absolute Path inside ROOT."""
    root = _root()
    rel = str(rel or "").replace("\\", "/").strip("/")
    if "\x00" in rel:
        raise ValueError("bad path")
    p = (root / rel).resolve() if rel else root
    if p != root and root not in p.parents:
        raise ValueError("path is outside the home folder")
    if p != root and p.relative_to(root).parts[0].startswith("."):
        raise ValueError("hidden system folders (.ssh, .config, ...) are not reachable from here")
    if must_exist and not p.exists():
        raise ValueError(f"{rel or '/'} does not exist")
    return p


def entry(rel):
    """The entry itself (NOT following a final symlink), inside ROOT. Delete,
    rename and move act on this, so removing a link never touches its target."""
    rel = str(rel or "").replace("\\", "/").strip("/")
    if not rel:
        raise ValueError("the uploads folder itself cannot be changed")
    parent_rel, _, name = rel.rpartition("/")
    parent = resolve(parent_rel)
    name = _check_name(name)
    p = parent / name
    if not os.path.lexists(p):
        raise ValueError(f"{rel} does not exist")
    return p


def root_ok(p):
    root = _root()
    return p == root or (root in p.parents and not p.relative_to(root).parts[0].startswith("."))


def read_only(p):
    """Is this absolute path inside a READ_ONLY top-level folder?"""
    try:
        parts = Path(p).relative_to(_root()).parts
    except ValueError:
        return False
    return bool(parts) and parts[0] in READ_ONLY


def _writable(p, what):
    if read_only(p):
        raise ValueError(f"{Path(p).relative_to(_root()).parts[0]}/ is read-only here "
                         f"(it is the running panel); cannot {what}")


def in_use(p):
    """Processes holding p, or anything under it, open, mapped or as cwd."""
    target = os.path.realpath(p)
    pre = target.rstrip(os.sep) + os.sep
    hit = lambda t: t == target or t.startswith(pre)
    me, users = os.getpid(), {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            found = hit(os.readlink(d / "cwd"))
            if not found:
                for fd in (d / "fd").iterdir():
                    try:
                        if hit(os.readlink(fd)):
                            found = True
                            break
                    except OSError:
                        pass
            if not found:
                with open(d / "maps") as m:
                    for line in m:
                        i = line.find("/")
                        if i >= 0 and hit(line[i:].rstrip("\n").removesuffix(" (deleted)")):
                            found = True
                            break
            if found:
                users[int(d.name)] = (d / "comm").read_text().strip()
        except OSError:
            continue
    return users


def _check_free(p, what):
    """Refuse changing something a process is using or a config points at."""
    if os.path.islink(p):
        return                                  # the link, not its target
    users = in_use(p)
    if users:
        who = ", ".join(f"{c} (pid {pid})" for pid, c in list(users.items())[:3])
        raise ValueError(f"cannot {what} {Path(p).name}: in use by {who}")
    target = os.path.realpath(p)
    pre = target.rstrip(os.sep) + os.sep
    try:
        refs = P.model_references()
    except Exception:
        refs = {}
    held = [(k, v) for k, v in refs.items() if k == target or k.startswith(pre)]
    if held:
        k, why = held[0]
        more = f" (+{len(held) - 1} more file(s))" if len(held) > 1 else ""
        raise ValueError(f"cannot {what} {Path(p).name}: {Path(k).name} is set in "
                         f"{', '.join(why[:3])}{' …' if len(why) > 3 else ''}{more}. "
                         "Change that setting first.")


def rel_of(p):
    r = str(Path(p).resolve().relative_to(_root()))
    return "" if r == "." else r


def _check_name(name):
    name = str(name or "").strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"bad name {name!r}")
    if len(name.encode()) > 255:
        raise ValueError("name is too long")
    return name


def _unique(dirpath, name):
    """name, or 'name (2).ext', 'name (3).ext', ... whichever is free."""
    if not (dirpath / name).exists():
        return name
    stem, ext = (name, "") if (dirpath / name).is_dir() or "." not in name[1:] else \
        (name[:name.rfind(".")], name[name.rfind("."):])
    stem = re.sub(r" \(\d+\)$", "", stem)
    i = 2
    while (dirpath / f"{stem} ({i}){ext}").exists():
        i += 1
    return f"{stem} ({i}){ext}"


def _count(p):
    try:
        with os.scandir(p) as it:
            files = dirs = 0
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    dirs += 1
                else:
                    files += 1
            return files, dirs
    except OSError:
        return 0, 0


def listing(rel):
    d = resolve(rel)
    if not d.is_dir():
        raise ValueError(f"{rel} is not a folder")
    out = []
    at_root = d == _root()
    with os.scandir(d) as it:
        for e in it:
            if e.name.startswith(".upload-") and e.name.endswith(".part"):
                continue                                   # uploads in flight
            if at_root and e.name.startswith("."):
                continue                                   # hidden system folders
            if e.is_symlink():
                tgt = Path(e.path).resolve()
                if not root_ok(tgt):
                    # points outside ~/uploads: show the link, reveal nothing about its target
                    lst = e.stat(follow_symlinks=False)
                    out.append(dict(name=e.name, dir=False, mtime=int(lst.st_mtime), link=True,
                                    outside=True, hidden=e.name.startswith("."), size=None))
                    continue
            try:
                st = e.stat(follow_symlinks=True)
            except OSError:
                continue
            is_dir = e.is_dir(follow_symlinks=True)
            item = dict(name=e.name, dir=is_dir, mtime=int(st.st_mtime),
                        link=e.is_symlink(), hidden=e.name.startswith("."),
                        ro=read_only(Path(d) / e.name))
            if is_dir:
                item["files"], item["dirs"] = _count(e.path)
                item["size"] = None
            else:
                item["size"] = st.st_size
            out.append(item)
    du = shutil.disk_usage(d)
    return dict(path=rel_of(d), root=str(_root()), entries=out, ro=read_only(d), places=places(),
                disk=dict(free=du.free, total=du.total),
                totals=dict(folders=sum(1 for x in out if x["dir"]),
                            files=sum(1 for x in out if not x["dir"]),
                            bytes=sum(x["size"] or 0 for x in out)))


def mkdir(rel, name):
    d = resolve(rel)
    _writable(d, "create a folder here")
    name = _check_name(name)
    t = d / name
    if t.exists():
        raise ValueError(f"{name} already exists")
    t.mkdir()
    return dict(ok=True, path=rel_of(t))


def rename(rel, new_name):
    src = entry(rel)
    _writable(src, "rename")
    _check_free(src, "rename")
    new_name = _check_name(new_name)
    dst = src.parent / new_name
    if os.path.lexists(dst):
        raise ValueError(f"{new_name} already exists here")
    src.rename(dst)
    return dict(ok=True, path=rel_of(dst.parent) + ("/" if rel_of(dst.parent) else "") + new_name)


def delete(rels):
    done = []
    for rel in rels or []:
        p = entry(rel)
        _writable(p, "delete")
        _check_free(p, "delete")
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
        done.append(rel)
    return dict(ok=True, deleted=done)


def paste(op, rels, dest_rel):
    """Copy or move sources into dest. A name clash gets ' (2)' etc.; moving
    or copying a folder into itself is refused."""
    if op not in ("copy", "move"):
        raise ValueError("op must be copy or move")
    dest = resolve(dest_rel)
    if not dest.is_dir():
        raise ValueError("paste target is not a folder")
    _writable(dest, "paste into it")
    out = []
    for rel in rels or []:
        src = entry(rel)
        if src.is_dir() and not src.is_symlink() and (dest == src or src in dest.parents):
            raise ValueError(f"cannot {op} {src.name} into itself")
        if op == "move" and src.parent == dest:
            continue                                        # already there
        if op == "move":
            _writable(src, "move")
            _check_free(src, "move")
        name = _unique(dest, src.name)
        tgt = dest / name
        if op == "move":
            shutil.move(str(src), str(tgt))
        elif src.is_dir() and not src.is_symlink():
            shutil.copytree(src, tgt, symlinks=True)
        else:
            shutil.copy2(src, tgt, follow_symlinks=False)
        out.append((rel_of(dest) + "/" if rel_of(dest) else "") + name)
    return dict(ok=True, op=op, created=out)


# --------------------------------------------------------------------------
# streaming upload / download (called from the HTTP handler)
# --------------------------------------------------------------------------
def upload_target(dir_rel, relname, overwrite=False):
    """Where an upload lands. relname may carry sub-folders (a folder upload
    sends 'photos/2026/a.jpg'); each part is checked and created."""
    d = resolve(dir_rel)
    if not d.is_dir():
        raise ValueError("upload target is not a folder")
    _writable(d, "upload here")
    parts = [p for p in str(relname or "").replace("\\", "/").split("/") if p]
    if not parts:
        raise ValueError("no file name")
    for part in parts:
        _check_name(part)
    for part in parts[:-1]:
        d = d / part
        if d.exists() and not d.is_dir():
            raise ValueError(f"{part} exists and is not a folder")
        d.mkdir(exist_ok=True)
        resolve(rel_of(d))                                  # still inside ROOT?
    name = parts[-1] if overwrite else _unique(d, parts[-1])
    return d, name


def receive(rfile, length, dir_rel, relname, overwrite=False):
    if length is None or length < 0:
        raise ValueError("upload needs a Content-Length")
    d, name = upload_target(dir_rel, relname, overwrite)
    free = shutil.disk_usage(d).free
    if length + FREE_MARGIN > free:
        raise ValueError(f"not enough disk space: {length >> 20} MiB upload, "
                         f"{free >> 20} MiB free")
    part = d / f".upload-{uuid.uuid4().hex[:10]}.part"
    left = length
    try:
        with open(part, "wb") as f:
            while left > 0:
                buf = rfile.read(min(CHUNK, left))
                if not buf:
                    raise ValueError("upload interrupted")
                f.write(buf)
                left -= len(buf)
        os.replace(part, d / name)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return dict(ok=True, path=rel_of(d / name), bytes=length)


def check_zip(rels):
    """Vet a zip request BEFORE any response headers are sent."""
    if not rels:
        raise ValueError("nothing selected")
    paths = [resolve(r) for r in rels]
    if _root() in paths:
        raise ValueError("pick folders inside Home to zip, not the whole home folder")
    return paths


def zip_name(rels):
    if len(rels) == 1:
        base = Path(rels[0]).name or "home"
    else:
        base = f"files-{time.strftime('%Y%m%d-%H%M%S')}"
    return f"{base}.zip"


def stream_zip(rels, out):
    """Write a zip of files/folders to a non-seekable stream (the socket)."""
    paths = [resolve(r) for r in rels]
    if _root() in paths:
        raise ValueError("pick folders inside Home to zip, not the whole home folder")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in paths:
            base = p.parent
            if p.is_dir():
                for dirpath, dirnames, files in os.walk(p):
                    dirnames[:] = sorted(dirnames)
                    rel_dir = Path(dirpath).relative_to(base)
                    if not files and not dirnames:
                        z.writestr(str(rel_dir) + "/", b"")
                    for fn in sorted(files):
                        fp = Path(dirpath) / fn
                        if fn.startswith(".upload-") and fn.endswith(".part"):
                            continue
                        if not root_ok(Path(os.path.realpath(fp))):
                            continue                # a link out of Home or into .ssh etc.
                        try:
                            z.write(fp, str(rel_dir / fn))
                        except OSError:
                            pass
            else:
                z.write(p, p.name)              # resolve() already vetted the target
