#!/usr/bin/env python3
"""
LexiPanel Fit recipes (added 2026-09-24): read how a GGUF was made, then make it again.

  explain  Given a quantized GGUF and the full-precision source it was cut from, find the
           llama-quantize command that reproduces every tensor's format. The file's ftype
           names the stock mix; a dry run of that mix is the baseline; each difference
           becomes an override (output layer, embeddings, MTP block, per-layer or per-role
           patterns). A final dry run proves it tensor by tensor. Header-only: seconds, no GPU.
  recipes  Saved commands. The MTP block is a symbolic rule resolved from each source's
           header, so a recipe carries across models of a family. Two built-ins
           ("stock + imatrix" and "MAX") and any number explained or imported; export/import
           is plain JSON so others can replicate a build.
  build    llama-quantize with the recipe's flags as a Fit job (CPU; the server keeps serving).

The importance matrix changes the formats llama.cpp picks, not only the rounding (e.g. IQ4_XS
raises the first eighth of ffn_down to Q5_K only WITHOUT one). Dry runs therefore use a
neutral one-entry stand-in when a recipe needs an imatrix; a real one is needed to build.
Findings behind this: ~/fitquant/FINDINGS.md §13-14.
"""
import hashlib, json, os, re, shlex, shutil, struct, subprocess, time
from pathlib import Path

P = None                    # panel module
F = None                    # fitquant module

MIB = 1048576

# llama.h ftype enum -> the name llama-quantize takes
FTYPES = {1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
          11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
          17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
          23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
          29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
          38: "MXFP4_MOE"}
FTYPE_NAMES = set(FTYPES.values()) - {"F16", "BF16"}
BASE_MENU = ["IQ4_XS", "Q4_K_M", "Q4_K_S", "IQ4_NL", "Q4_0", "Q5_K_M", "Q6_K", "Q8_0",
             "IQ3_M", "Q3_K_M"]
# per-tensor types --tensor-type / --output-tensor-type / --token-embedding-type accept
TENSOR_TYPES = {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K",
                "Q4_K", "Q5_K", "Q6_K", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S",
                "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "TQ1_0", "TQ2_0", "MXFP4"}
UNQUANT = {"F32": 4, "F16": 2, "BF16": 2}
SCOPES = ("tensor", "layers", "role")       # applied in this order; the MTP rule sits
                                            # between layers and role (first match wins)
_PAT_OK = re.compile(r"^[A-Za-z0-9_.\\^$()|+*?\[\]{}-]{1,600}$")
_ID_OK = re.compile(r"^[0-9A-Za-z_.-]{1,120}$")

BUILTINS = [
    dict(id="builtin-stock-imatrix", builtin=True, name="Stock mix + importance matrix",
         aka="DavidAU's \"LOW-MTP\" files", ftype="IQ4_XS", imatrix="required",
         output=None, embed=None, mtp=None, rules=[],
         description="llama.cpp's own mix for the chosen base, cut with an importance matrix "
                     "and nothing else. On Qwen3.8-27B IQ4_XS this is byte-for-byte the layout "
                     "of DavidAU's LOW-MTP file, the fastest file measured on the XTX (78.7 t/s)."),
    dict(id="builtin-max", builtin=True, name="MAX: output at source precision, MTP block Q8_0",
         aka="DavidAU's \"MAX-MTP\" files", ftype="IQ4_XS", imatrix="required",
         output="source", embed=None, mtp="Q8_0", rules=[],
         description="The stock mix plus two overrides: the output layer kept at the source's "
                     "precision (BF16, --leave-output-tensor) and every MTP-block matrix at Q8_0. "
                     "Measured on the XTX: 17% slower than the stock mix for no gain in draft "
                     "acceptance; the output layer is most of the cost."),
]


def bind(panel_module, fit_module):
    global P, F
    P, F = panel_module, fit_module


def _dir():
    return F._fit_dir("recipes")


# ============================================================================
# helpers
# ============================================================================
def standin_imatrix():
    """A neutral one-entry legacy imatrix. llama.cpp only asks whether an imatrix is
    present when it picks formats, so this reproduces a real one's type choices in a dry
    run (verified: 14,589 MiB and 866/866 types on DavidAU's LOW-MTP IQ4_XS). Never used
    for a real build."""
    f = F._fit_dir("cache") / "standin-imatrix.dat"
    if not f.exists():
        name, vals = b"__lexipanel_standin__", [1.0] * 256
        f.write_bytes(struct.pack("<i", 1) + struct.pack("<i", len(name)) + name
                      + struct.pack("<ii", 1, len(vals)) + struct.pack(f"<{len(vals)}f", *vals)
                      + struct.pack("<i", 1) + struct.pack("<i", 7) + b"standin")
    return f


def _dry(src, ftype, flags):
    q = F.quantize_bin()
    if not q:
        raise ValueError("no llama-quantize: install a llama.cpp CPU build in the Builds tab")
    argv = ["nice", "-n", "10", str(q), "--dry-run"] + list(flags) + [str(src), ftype]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    txt = r.stdout + r.stderr
    rows = {}
    for line in txt.splitlines():
        m = F._DRY_LINE.match(line.strip())
        if m:
            name, _shape, st, smib, qmib, qt = m.groups()
            rows[name] = dict(type=(qt or st).upper(), src_type=st.upper(),
                              mib=float(qmib) if qmib else float(smib), quantized=bool(qmib))
    if not rows:
        raise ValueError(f"llama-quantize --dry-run gave no tensor list (exit {r.returncode}): "
                         f"{txt.strip()[-400:]}")
    tot = re.search(r"quant size\s*=\s*([\d.]+) MiB", txt)
    return rows, (float(tot.group(1)) if tot else sum(x["mib"] for x in rows.values()))


def _counts(kv):
    arch, n_layer, nextn = F._arch_counts(kv)
    n_main = (n_layer - nextn) if n_layer else None
    return arch, n_layer, nextn, n_main


def _under(path, roots):
    rp = os.path.realpath(str(path or ""))
    return any(rp == os.path.realpath(r) or rp.startswith(os.path.realpath(r) + os.sep)
               for r in roots)


def _model_path(p):
    p = str(p or "")
    if not p.endswith(".gguf") or not os.path.isfile(p) or not _under(p, [P.HOME]):
        raise ValueError("pick a .gguf file under your home directory")
    return p


def _source_paths():
    return {s["path"] for s in F.sources()}


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")[:60] or "recipe"


def _cmd(q, flags, src, out, ftype, imatrix=None):
    argv = [str(q)] + (["--imatrix", str(imatrix)] if imatrix else []) + list(flags) \
        + [str(src), str(out), ftype]
    return " ".join(shlex.quote(a) for a in argv)


def _layer_alt(layers):
    return "(" + "|".join(str(x) for x in sorted(layers)) + ")"


# ============================================================================
# a recipe -> llama-quantize flags for one source
# ============================================================================
def resolve(rec, kv):
    """(flags, notes). The MTP rule becomes an anchored block pattern from the source's
    header; rules apply tensor -> layers -> MTP -> role, since the first match wins."""
    _arch, n_layer, nextn, n_main = _counts(kv)
    flags, notes = [], []
    out = rec.get("output")
    if out == "source":
        flags.append("--leave-output-tensor")
    elif out:
        flags += ["--output-tensor-type", out.lower()]
    if rec.get("embed"):
        flags += ["--token-embedding-type", rec["embed"].lower()]
    rules = rec.get("rules") or []
    pats = [(r["pattern"], r["type"]) for s in ("tensor", "layers") for r in rules
            if (r.get("scope") or "layers") == s]
    if rec.get("mtp"):
        if nextn and n_layer:
            pats.append((r"^blk\." + _layer_alt(range(n_main, n_layer)) + r"\.", rec["mtp"]))
        else:
            notes.append("the recipe sets the MTP block, but this model has none: skipped")
    pats += [(r["pattern"], r["type"]) for r in rules if r.get("scope") == "role"]
    for pat, t in pats:
        flags += ["--tensor-type", f"{pat}={t.lower()}"]
    if n_main is not None:
        hi = [int(x) for r in rules if r.get("scope") == "layers"
              for x in re.findall(r"\d+", r["pattern"].split(r"\.", 2)[1] if r"\." in r["pattern"] else "")]
        if hi and max(hi) >= n_main:
            notes.append(f"the recipe names layer {max(hi)}, but this model has {n_main} main "
                         "layers: those rules will not match here")
    return flags, notes


def _validate(rec):
    """Normalise and check a recipe (built, explained or imported). Raises ValueError."""
    if not isinstance(rec, dict):
        raise ValueError("a recipe is a JSON object")
    out = dict(format="lexipanel-recipe/1")
    out["name"] = str(rec.get("name") or "recipe").strip()[:100]
    out["description"] = str(rec.get("description") or "")[:2000]
    out["aka"] = str(rec.get("aka") or "")[:200]
    ft = str(rec.get("ftype") or "").upper()
    if ft not in FTYPE_NAMES:
        raise ValueError(f"ftype must be one of {', '.join(sorted(FTYPE_NAMES))}")
    out["ftype"] = ft
    im = str(rec.get("imatrix") or "none")
    if im not in ("required", "recommended", "none"):
        raise ValueError("imatrix: required, recommended or none")
    out["imatrix"] = im
    o = rec.get("output")
    o = None if o in (None, "", "stock") else str(o)
    if o and o != "source" and o.upper() not in TENSOR_TYPES:
        raise ValueError("output: source, a tensor type, or empty")
    out["output"] = o if o in (None, "source") else o.upper()
    for k in ("embed", "mtp"):
        v = rec.get(k)
        v = None if v in (None, "", "stock") else str(v).upper()
        if v and v not in TENSOR_TYPES:
            raise ValueError(f"{k}: a tensor type or empty")
        out[k] = v
    rules = []
    for r in (rec.get("rules") or [])[:400]:
        pat, t = str(r.get("pattern") or ""), str(r.get("type") or "").upper()
        scope = str(r.get("scope") or "layers")
        if not _PAT_OK.match(pat) or pat.startswith("-") or "=" in pat:
            raise ValueError(f"bad pattern {pat[:80]!r}")
        try:
            re.compile(pat)
        except re.error as e:
            raise ValueError(f"pattern {pat[:80]!r}: {e}")
        if t not in TENSOR_TYPES:
            raise ValueError(f"rule type {t!r} is not a tensor type")
        if scope not in SCOPES:
            raise ValueError(f"rule scope: {', '.join(SCOPES)}")
        rules.append(dict(pattern=pat, type=t, scope=scope, why=str(r.get("why") or "")[:300]))
    out["rules"] = rules
    prov = rec.get("provenance")
    if isinstance(prov, dict):
        out["provenance"] = json.loads(json.dumps(prov, default=str))
    if isinstance(rec.get("check"), dict):
        out["check"] = rec["check"]
    if isinstance(rec.get("explain"), list):
        out["explain"] = [str(x)[:600] for x in rec["explain"][:40]]
    return out


def _save(rec):
    base = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + _slug(rec["name"]).lower()
    rid, i = base, 2
    while (_dir() / f"{rid}.json").exists():          # same name within the same second
        rid, i = f"{base}-{i}", i + 1
    rec = dict(rec, id=rid, builtin=False, saved=F._now())
    F._atomic(_dir() / f"{rid}.json", rec)
    return rec


def get(rid):
    rid = str(rid or "")
    for b in BUILTINS:
        if b["id"] == rid:
            return dict(b, format="lexipanel-recipe/1")
    if not _ID_OK.match(rid):
        raise ValueError("bad recipe id")
    r = F._read(_dir() / f"{rid}.json")
    if not r:
        raise ValueError(f"no recipe {rid}")
    return r


def summary(r):
    bits = []
    if r.get("output"):
        bits.append("output " + ("at source precision" if r["output"] == "source" else r["output"]))
    if r.get("embed"):
        bits.append(f"embeddings {r['embed']}")
    if r.get("mtp"):
        bits.append(f"MTP block {r['mtp']}")
    if r.get("rules"):
        bits.append(f"{len(r['rules'])} pattern rule" + ("s" if len(r["rules"]) != 1 else ""))
    return "; ".join(bits) or "stock mix, no overrides"


def list_recipes():
    out = []
    for r in [dict(b) for b in BUILTINS] + [F._read(f) for f in
                                            sorted(_dir().glob("*.json"), reverse=True)]:
        if not r:
            continue
        prov = r.get("provenance") or {}
        out.append(dict(id=r.get("id"), name=r.get("name"), aka=r.get("aka"),
                        builtin=bool(r.get("builtin")), ftype=r.get("ftype"),
                        imatrix=r.get("imatrix"), summary=summary(r),
                        description=r.get("description"), from_name=prov.get("from_name"),
                        from_file=prov.get("from_file"), imported=bool(prov.get("imported")),
                        check=r.get("check"), saved=r.get("saved")))
    return out


_models_cache = {}          # path -> ((mtime, size), row or None); the Fit tab asks every 30 s


def explain_models():
    """Quantized GGUFs the operator can ask about (projectors and sources left out)."""
    out = []
    for d in (P.MODELS, F.out_dir()):
        if not Path(d).is_dir():
            continue
        for f in sorted(Path(d).glob("*.gguf")):
            m = re.search(r"-(\d{5})-of-\d{5}\.gguf$", f.name)
            if (m and m.group(1) != "00001") or f.name.startswith("mmproj"):
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            key = (st.st_mtime_ns, st.st_size)
            hit = _models_cache.get(str(f))
            if hit and hit[0] == key:
                if hit[1]:
                    out.append(hit[1])
                continue
            row = None
            hdr = F.gguf_tensors(f)
            if hdr:
                kv = hdr[0]
                ft = FTYPES.get(kv.get("general.file_type"))
                quant = any(t["type"] not in UNQUANT for t in hdr[1].values())
                if kv.get("general.architecture") != "clip" and kv.get("general.type") != "imatrix" \
                        and (quant or ft not in (None, "F16", "BF16")):
                    row = dict(path=str(f), name=f.name, ftype=ft,
                               gib=round((P.model_bytes(f) or st.st_size) / 2**30, 2),
                               imatrix=bool(kv.get("quantize.imatrix.file")
                                            or kv.get("quantize.imatrix.entries_count")))
            _models_cache[str(f)] = (key, row)
            if row:
                out.append(row)
    return out


# ============================================================================
# explain: reverse-engineer a GGUF against its source
# ============================================================================
def _same_weights(tm, ts, cap=512 * MIB):
    """Hash the tensors both files keep unquantized (norms, SSM constants...). Identical
    bytes = cut from the same weights. Exact sizes from the shapes: the header's
    offset gaps include alignment padding."""
    names = [n for n in tm if n in ts and tm[n]["type"] in UNQUANT and tm[n]["type"] == ts[n]["type"]
             and tm[n]["shape"] == ts[n]["shape"]]
    checked = same = used = 0
    for n in names:
        size = UNQUANT[tm[n]["type"]]
        for d in tm[n]["shape"]:
            size *= int(d)
        if used + 2 * size > cap:
            break
        try:
            hs = []
            for t in (tm[n], ts[n]):
                with open(t["file"], "rb") as f:
                    f.seek(t["off"])
                    hs.append(hashlib.sha1(f.read(size)).digest())
        except (OSError, KeyError):
            continue
        used += 2 * size
        checked += 1
        same += hs[0] == hs[1]
    return dict(checked=checked, identical=same,
                verdict=None if not checked else (same == checked))


def _match_source(tm):
    """Sources under ~/models/src with the same tensor names and shapes; the one with
    identical unquantized tensors first."""
    best = []
    for s in F.sources():
        hdr = F.gguf_tensors(s["path"])
        if not hdr:
            continue
        ts = hdr[1]
        if set(ts) != set(tm) or any(ts[n]["shape"] != tm[n]["shape"] for n in tm):
            continue
        sw = _same_weights(tm, ts, cap=64 * MIB)
        best.append((1 if sw["verdict"] else 0, s["path"]))
    return max(best)[1] if best else None


def _role_table(tm, n_main):
    by = {}
    for n, t in tm.items():
        role, il = F.role_of(n, n_main)
        if t["type"] in UNQUANT and role in ("other", "ssm_small"):
            continue
        r = by.setdefault(role, dict(types={}, mib=0.0, layers={}))
        r["types"][t["type"]] = r["types"].get(t["type"], 0) + 1
        r["mib"] += t["bytes"] / MIB
        if il is not None:
            r["layers"].setdefault(t["type"], []).append(il)
    return [dict(role=k, label=F.ROLE_LABEL.get(k, k), mib=round(v["mib"], 1), types=v["types"],
                 layers={t: F._ranges(sorted(set(ls))) for t, ls in v["layers"].items()})
            for k, v in sorted(by.items(), key=lambda kv: -kv[1]["mib"])]


def _stem(n):
    m = F._BLK.match(n)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2) + (m.group(3) or "")


def _group_text(names, n_main, types_of):
    """'FFN down projection layers 0-7 at Q5_K' style text for a set of tensors."""
    g = {}
    for n in names:
        role, il = F.role_of(n, n_main)
        g.setdefault((role, types_of(n)), []).append(il)
    parts = []
    for (role, t), ils in sorted(g.items(), key=lambda kv: str(kv[0])):
        ls = [x for x in ils if x is not None]
        parts.append(f"{F.ROLE_LABEL.get(role, role)}"
                     + (f" layers {F._ranges(sorted(set(ls)))}" if ls else "") + f" at {t}")
    return "; ".join(parts)


def explain(body):
    model = _model_path(body.get("model"))
    hm = F.gguf_tensors(model)
    if not hm:
        raise ValueError("cannot read that file's GGUF header")
    kv_m, tm = hm
    arch, n_layer, nextn, n_main = _counts(kv_m)
    ftype = str(body.get("ftype") or "").upper() or FTYPES.get(kv_m.get("general.file_type"))
    if ftype in (None, "F16", "BF16") or ftype not in FTYPE_NAMES:
        raise ValueError("the file does not name a quantization mix (general.file_type); "
                         "pick the base mix it was made with")
    imx = {k.split("quantize.imatrix.", 1)[1]: v for k, v in kv_m.items()
           if k.startswith("quantize.imatrix.")}
    roles = _role_table(tm, n_main)
    base = dict(model=model, model_name=Path(model).name, arch=arch, ftype=ftype,
                n_layer=n_layer, nextn=nextn, imatrix_meta=imx or None, roles=roles,
                file_mib=round(sum(t["bytes"] for t in tm.values()) / MIB, 1))

    src = str(body.get("source") or "")
    if src:
        if src not in _source_paths():
            raise ValueError("the source must be a full-precision GGUF under ~/models/src")
    else:
        src = _match_source(tm)
    if not src:
        return dict(base, recipe=None, error="no full-precision source here has this file's "
                    "tensor layout. Fetch the model's original weights (Fit, Upstream watch: "
                    "phase A), then explain again. Formats per tensor group are shown below.")
    kv_s, ts = F.gguf_tensors(src)
    if set(ts) != set(tm):
        raise ValueError(f"{Path(src).name} has a different tensor set "
                         f"({len(set(tm) - set(ts))} missing, {len(set(ts) - set(tm))} extra)")
    same = _same_weights(tm, ts)

    si = str(standin_imatrix())
    rows_i, _ = _dry(src, ftype, ["--imatrix", si])
    rows_n, _ = _dry(src, ftype, [])
    diff_i = [n for n in tm if rows_i[n]["type"] != tm[n]["type"]]
    diff_n = [n for n in tm if rows_n[n]["type"] != tm[n]["type"]]
    imx_changes = [n for n in rows_i if rows_i[n]["type"] != rows_n[n]["type"]]
    use_i = bool(imx) or len(diff_i) < len(diff_n)
    stock = rows_i if use_i else rows_n
    rec = dict(name=str(body.get("name") or f"{Path(model).stem} (explained)")[:100],
               ftype=ftype, output=None, embed=None, mtp=None, rules=[],
               imatrix=("required" if use_i and imx_changes else "recommended" if use_i else "none"))
    lines = []
    if imx:
        lines.append(f"Made with an importance matrix: {imx.get('file') or '?'}"
                     + (f", dataset {imx['dataset']}" if imx.get("dataset") else "")
                     + (f", {imx['chunks_count']} chunks" if imx.get("chunks_count") else "")
                     + (f", {imx['entries_count']} entries" if imx.get("entries_count") else "")
                     + ". The file records its name only; the data is not inside it.")
    elif use_i:
        lines.append("The file records no importance matrix, but its layout matches llama.cpp's "
                     "choices WITH one, so one was used.")
    if use_i and imx_changes:
        lines.append(f"The imatrix changes the layout here: without one, llama.cpp would put "
                     f"{len(imx_changes)} tensors elsewhere ({_group_text(imx_changes, n_main, lambda n: rows_n[n]['type'])}). "
                     "A replica needs an imatrix to match.")

    diffs = {n: tm[n]["type"] for n in tm if stock[n]["type"] != tm[n]["type"]}
    if "output.weight" in diffs:
        t = diffs.pop("output.weight")
        if t == ts["output.weight"]["type"]:
            rec["output"] = "source"
            lines.append(f"Output layer left at the source's {t} (--leave-output-tensor, "
                         f"{tm['output.weight']['bytes'] / MIB:,.0f} MiB); the stock mix "
                         f"would use {stock['output.weight']['type']}.")
        else:
            rec["output"] = t
            lines.append(f"Output layer at {t} (--output-tensor-type); the stock mix would use "
                         f"{stock['output.weight']['type']}.")
    if "token_embd.weight" in diffs:
        rec["embed"] = diffs.pop("token_embd.weight")
        lines.append(f"Token embeddings at {rec['embed']} (--token-embedding-type); the stock "
                     f"mix would use {stock['token_embd.weight']['type']}.")
    if nextn and n_main is not None:
        mtp = [n for n in tm if F.role_of(n, n_main)[0] == "mtp" and stock[n]["quantized"]]
        if mtp and any(n in diffs for n in mtp):
            # one block rule at the most common format, exact-name exceptions for the rest
            # (tensor rules are applied before the block rule, so they win)
            cnt = {}
            for n in mtp:
                cnt[tm[n]["type"]] = cnt.get(tm[n]["type"], 0) + 1
            rec["mtp"] = max(cnt, key=lambda t: (cnt[t], t))
            exc = [n for n in mtp if tm[n]["type"] != rec["mtp"]]
            for n in mtp:
                diffs.pop(n, None)
            for n in exc:
                rec["rules"].append(dict(pattern="^" + re.escape(n) + "$", type=tm[n]["type"],
                                         scope="tensor", why=f"MTP block exception (block rule {rec['mtp']})"))
            blk = f"block {n_main}{'-' + str(n_layer - 1) if nextn > 1 else ''}"
            lines.append(f"MTP block ({blk}): " + (f"every matrix at {rec['mtp']}." if not exc else
                         f"{len(mtp) - len(exc)} of {len(mtp)} matrices at {rec['mtp']}; "
                         + ", ".join(f"{_stem(n)[1]} at {tm[n]['type']}" for n in exc) + "."))
        elif mtp:
            lines.append("MTP block: the stock mix's choice ("
                         + ", ".join(f"{t}×{sum(tm[n]['type'] == t for n in mtp)}"
                                     for t in sorted({tm[n]['type'] for n in mtp})) + ").")
    # the rest: per role (stem), one rule per target format
    by_stem = {}
    for n, t in diffs.items():
        il, stem = _stem(n)
        if stem is None:
            rec["rules"].append(dict(pattern="^" + re.escape(n) + "$", type=t, scope="tensor",
                                     why=f"stock would use {stock[n]['type']}"))
            continue
        by_stem.setdefault(stem, {}).setdefault(t, []).append(il)
    for stem, per_t in sorted(by_stem.items()):
        pool = [n for n in tm if _stem(n)[1] == stem
                and not (rec["mtp"] and F.role_of(n, n_main)[0] == "mtp")]
        for t, ils in sorted(per_t.items()):
            role = F.role_of(next(n for n in pool if _stem(n)[0] == ils[0]), n_main)[0]
            label = F.ROLE_LABEL.get(role, stem)
            if role in ("ssm_small", "other", "mtp"):
                label += f" ({stem.removesuffix('.weight')})"
            was = sorted({stock[n]["type"] for n in pool if _stem(n)[0] in ils})
            if all(tm[n]["type"] == t for n in pool):
                rec["rules"].append(dict(pattern=r"^blk\.\d+\." + re.escape(stem) + "$", type=t,
                                         scope="role", why=f"every {label} at {t}"))
                lines.append(f"{label}: every layer at {t} (stock: {', '.join(was)} in layers "
                             f"{F._ranges(sorted(ils))}).")
            else:
                rec["rules"].append(dict(pattern=r"^blk\." + _layer_alt(ils) + r"\." + re.escape(stem) + "$",
                                         type=t, scope="layers",
                                         why=f"{label} layers {F._ranges(sorted(ils))}"))
                lines.append(f"{label}: layers {F._ranges(sorted(ils))} at {t} "
                             f"(stock: {', '.join(was)}).")

    # prove it; anything still off gets an exact-name rule, then prove again
    rec = _validate(rec)
    flags, _notes = resolve(rec, kv_s)
    dry_imx = ["--imatrix", si] if rec["imatrix"] != "none" else []
    rows, tot = _dry(src, ftype, dry_imx + flags)
    bad = [n for n in tm if rows[n]["type"] != tm[n]["type"]]
    if bad:
        rec["rules"] = [dict(pattern="^" + re.escape(n) + "$", type=tm[n]["type"], scope="tensor",
                             why="fix-up: the grouped rules missed it") for n in bad] + rec["rules"]
        rec = _validate(rec)
        flags, _notes = resolve(rec, kv_s)
        rows, tot = _dry(src, ftype, dry_imx + flags)
        bad = [n for n in tm if rows[n]["type"] != tm[n]["type"]]
    if not any((rec["output"], rec["embed"], rec["mtp"], rec["rules"])):
        lines.insert(0, f"This is llama.cpp's stock {ftype} mix"
                        + (" with an importance matrix" if rec["imatrix"] != "none" else "")
                        + ": nothing was hand-tuned.")
    else:
        lines.insert(0, f"Base: llama.cpp's stock {ftype} mix"
                        + (" with an importance matrix" if rec["imatrix"] != "none" else "")
                        + f", plus {summary(rec)}.")
    if same["verdict"] is True:
        lines.append(f"Same weights as {Path(src).name}: all {same['checked']} unquantized "
                     "tensors are byte-identical, so a replica can match exactly.")
    elif same["verdict"] is False:
        lines.append(f"{same['checked'] - same['identical']} of {same['checked']} unquantized "
                     f"tensors differ from {Path(src).name}: same layout, different weights "
                     "(another revision or fine-tune). The recipe still applies; the file won't match.")
    file_mib = sum(t["bytes"] for t in tm.values()) / MIB
    rec["check"] = dict(at=F._now(), matched=len(tm) - len(bad), total=len(tm),
                        mismatches=bad[:20], quant_mib=round(tot, 1), file_mib=round(file_mib, 1),
                        source=src, quantizer_build=F._build_no(F.quantize_bin()))
    rec["provenance"] = dict(from_file=model, from_name=Path(model).name,
                             general_name=kv_m.get("general.name"), imatrix_meta=imx or None,
                             source=src, source_name=Path(src).name, same_weights=same,
                             arch=arch, n_layer=n_layer, nextn=nextn, explained=F._now())
    rec["explain"] = lines
    rec["description"] = f"Explained from {Path(model).name}."
    rec = _save(rec)
    return dict(base, recipe=rec, source=src,
                command=_cmd("llama-quantize", flags, "SOURCE.gguf", "OUT.gguf", ftype,
                             "IMATRIX.gguf" if rec["imatrix"] != "none" else None))


# ============================================================================
# check (dry run) and build
# ============================================================================
def _prep(body):
    rec = get(body.get("recipe"))
    src = str(body.get("source") or "")
    if src not in _source_paths():
        raise ValueError("pick a full-precision source under ~/models/src")
    ftype = str(body.get("ftype") or rec["ftype"]).upper()
    if ftype not in FTYPE_NAMES:
        raise ValueError("unknown base mix")
    imatrix = str(body.get("imatrix") or "")
    if imatrix and not (os.path.isfile(imatrix) and _under(imatrix, [F.imatrix_dir()])):
        raise ValueError("pick an importance matrix from the Fit list (build or import one)")
    kv = F.gguf_tensors(src)[0]
    flags, notes = resolve(rec, kv)
    return rec, src, ftype, imatrix, kv, flags, notes


def check(body):
    rec, src, ftype, imatrix, kv, flags, notes = _prep(body)
    used = imatrix or (str(standin_imatrix()) if rec["imatrix"] != "none" else "")
    rows, tot = _dry(src, ftype, (["--imatrix", used] if used else []) + flags)
    _arch, _n, _x, n_main = _counts(kv)
    tm = {n: dict(type=r["type"], bytes=r["mib"] * MIB) for n, r in rows.items()}
    res = dict(quant_mib=round(tot, 1), roles=_role_table(tm, n_main), notes=notes,
               command=_cmd(F.quantize_bin() or "llama-quantize", flags, src, "OUT.gguf", ftype,
                            imatrix or ("IMATRIX.gguf" if rec["imatrix"] != "none" else None)))
    if not imatrix and rec["imatrix"] == "required":
        res["notes"].append("sized with a neutral stand-in imatrix: a real build needs one, or "
                            "llama.cpp picks different formats for some tensors")
    orig = (rec.get("provenance") or {}).get("from_file")
    if orig and os.path.exists(orig):
        hdr = F.gguf_tensors(orig)
        if hdr and set(hdr[1]) == set(rows):
            bad = [n for n in rows if rows[n]["type"] != hdr[1][n]["type"]]
            res["vs_original"] = dict(name=Path(orig).name, matched=len(rows) - len(bad),
                                      total=len(rows), mismatches=bad[:20])
    return res


def build(body):
    rec, src, ftype, imatrix, kv, flags, notes = _prep(body)
    if rec["imatrix"] == "required" and not imatrix and not body.get("allow_no_imatrix"):
        raise ValueError("this recipe needs an importance matrix: without one llama.cpp picks "
                         "different formats for some tensors. Pick one (Fit, Importance matrix: "
                         "build or import), or tick 'build without one'.")
    name = str(body.get("name") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,200}\.gguf", name):
        raise ValueError("name: letters, digits and ._+- only, ending in .gguf")
    out = F.out_dir() / name
    if out.exists():
        raise ValueError(f"{out} exists; delete it first or pick another name")
    q = F.quantize_bin()
    if not q:
        raise ValueError("no llama-quantize (Builds tab: install a CPU build)")
    # the dry run takes exactly the build's flags, so its per-tensor types are the prediction
    rows, tot = _dry(src, ftype, (["--imatrix", imatrix] if imatrix else []) + flags)
    need_gb = tot * MIB / 1e9 + 1
    free = F._free_gb(F.out_dir()) - F.DISK_RESERVE_GB
    if free < need_gb:
        raise ValueError(f"not enough disk: the file needs {need_gb:.1f} GB and {free:.1f} GB is "
                         f"free above the {F.DISK_RESERVE_GB} GB reserve")
    threads = int(body.get("threads") or max(1, (os.cpu_count() or 4) - 2))
    orig = (rec.get("provenance") or {}).get("from_file")

    def work(job):
        F.out_dir().mkdir(parents=True, exist_ok=True)
        for n in notes:
            F._jlog(job, "note: " + n)
        F._jlog(job, f"recipe {rec['name']} ({summary(rec)}) on {Path(src).name}, base {ftype}; "
                     f"predicted {tot:,.0f} MiB")
        part = Path(str(out) + ".part")
        argv = ["nice", "-n", "19", "ionice", "-c3", str(q)] \
            + (["--imatrix", imatrix] if imatrix else []) + flags + [src, str(part), ftype, str(threads)]
        try:
            F._run_cmd(job, argv, label=f"quantizing to {name}", progress_re=r"\[\s*\d+/\s*\d+\]\s+\S+")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, out)
        hdr = F.gguf_tensors(out)
        got = {n: t["type"] for n, t in (hdr[1] if hdr else {}).items()}
        bad = [n for n, r in rows.items() if got.get(n) and got[n] != r["type"]]
        side = dict(fit=1, built=F._now(), goal="recipe", recipe=rec["id"], recipe_name=rec["name"],
                    base=ftype, source=src, source_rev=F._source_rev(src), quantizer=str(q),
                    quantizer_build=F._build_no(q), imatrix=imatrix or None,
                    predicted_mib=round(tot, 1), file_bytes=out.stat().st_size,
                    mismatched_types=len(bad), replica_of=orig if orig and os.path.exists(orig) else None,
                    command=_cmd(q, flags, src, out, ftype, imatrix or None), kld=None)
        F._atomic(Path(str(out) + ".fit.json"), side)
        job["output"] = str(out)
        F._jlog(job, f"built {out} ({out.stat().st_size / 2**30:.2f} GiB); "
                     + (f"{len(bad)} tensors differ from the dry run" if bad else "every tensor matches the dry run"))
    return F._start_job("build", f"Build {name} ({rec['name']})", work, recipe=rec["id"], output=str(out))


# ============================================================================
# import / export / delete
# ============================================================================
def export(rid):
    """A recipe to hand to someone else: this box's paths (home directory, user name)
    are cut down to file names; the maker's recorded imatrix name is kept."""
    r = json.loads(json.dumps(get(rid), default=str))
    for k in ("id", "saved", "builtin"):
        r.pop(k, None)
    prov = r.get("provenance") or {}
    for k in ("from_file", "source"):
        if prov.get(k):
            prov[k] = None
    if isinstance(r.get("check"), dict) and r["check"].get("source"):
        r["check"]["source"] = Path(str(r["check"]["source"])).name
    r["exported"] = F._now()
    return r


def import_recipe(body):
    rec = body.get("recipe")
    if isinstance(rec, str):
        try:
            rec = json.loads(rec)
        except ValueError:
            raise ValueError("not JSON")
    rec = _validate(rec)
    prov = dict(rec.get("provenance") or {})
    prov["imported"] = F._now()
    if prov.get("from_file") and not os.path.exists(str(prov["from_file"])):
        prov["from_file"] = None             # someone else's path: keep the name only
    rec["provenance"] = prov
    return _save(rec)


def delete(body):
    rid = str(body.get("id") or "")
    if rid.startswith("builtin-"):
        raise ValueError("built-in recipes cannot be deleted")
    if not _ID_OK.match(rid) or not (_dir() / f"{rid}.json").exists():
        raise ValueError("no such recipe")
    (_dir() / f"{rid}.json").unlink()
    return dict(ok=True)


# ============================================================================
# importing an importance matrix made elsewhere
# ============================================================================
def _imatrix_entries(path):
    """(format, entry names, chunks) of a GGUF or legacy .dat imatrix; ValueError if neither."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"GGUF":
        hdr = F.gguf_tensors(path)
        if not hdr or hdr[0].get("general.type") != "imatrix":
            raise ValueError("a GGUF, but not an importance matrix (general.type)")
        names = sorted({n[:-len(".in_sum2")] for n in hdr[1] if n.endswith(".in_sum2")})
        return "gguf", names, hdr[0].get("imatrix.chunk_count")
    names = []
    with open(path, "rb") as f:
        try:
            n = struct.unpack("<i", f.read(4))[0]
            if not 0 < n <= 200_000:
                raise ValueError
            for _ in range(n):
                ln = struct.unpack("<i", f.read(4))[0]
                if not 0 < ln <= 1024:
                    raise ValueError
                names.append(f.read(ln).decode("utf-8", "replace"))
                _ncall, nval = struct.unpack("<ii", f.read(8))
                if not 0 < nval <= 10_000_000:
                    raise ValueError
                f.seek(4 * nval, 1)
            tail = f.read(4)
            chunks = struct.unpack("<i", tail)[0] if len(tail) == 4 else None
        except (struct.error, ValueError):
            raise ValueError("neither a GGUF imatrix nor a legacy .dat imatrix")
    return "dat", sorted(names), chunks


def import_imatrix(body):
    path = str(body.get("path") or "")
    if not os.path.isfile(path) or not _under(path, [P.HOME]):
        raise ValueError("give the path of an imatrix file under your home directory")
    if os.path.getsize(path) > 4 * 2**30:
        raise ValueError("over 4 GiB: not an importance matrix")
    fmt, names, chunks = _imatrix_entries(path)
    ext = ".gguf" if fmt == "gguf" else ".dat"
    name = str(body.get("name") or Path(path).stem).strip()
    name = re.sub(r"[^A-Za-z0-9._+-]", "_", name)[:120].removesuffix(ext) + ".imported" + ext
    dest = F.imatrix_dir() / name
    if dest.exists():
        raise ValueError(f"{name} is already imported")
    cov = None
    src = str(body.get("source") or "")
    if src:
        if src not in _source_paths():
            raise ValueError("unknown source")
        tbl = F.size_table(src)
        n_main = (tbl["n_layer"] - tbl["nextn"]) if tbl.get("n_layer") else None
        want = [n for n, r in tbl["tensors"].items() if r.get("quantized") and len(r.get("shape") or []) >= 2]
        have = set(names)
        miss = [n for n in want if n not in have]
        cov = dict(source=Path(src).name, covered=len(want) - len(miss), total=len(want),
                   missing_mtp=sum(F.role_of(n, n_main)[0] == "mtp" for n in miss),
                   missing_output=sum(F.role_of(n, n_main)[0] == "output" for n in miss))
    shutil.copy2(path, dest)
    F._atomic(Path(str(dest) + ".json"), dict(
        built=F._now(), reference_kind="imported", imported_from=path, format=fmt,
        entries=len(names), chunks=chunks, coverage=cov,
        note="Made elsewhere. An imatrix from another fine-tune of the same base loads (same "
             "tensor names) but measured that model's activations, not this one's."))
    return dict(ok=True, path=str(dest), entries=len(names), coverage=cov)
