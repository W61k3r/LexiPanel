#!/usr/bin/env python3
"""
LexiPanel Fit: hardware-fitted requants (phases C-F, added 2026-09-24).

Phase A (~/fitquant/phaseA-source.sh) turns a model's full-precision upload
into a BF16 GGUF under ~/models/src/<name>/. Phase B measures how fast each
integer format runs on each card. This module is everything after that, as a
panel tab:

  C  solver    picks every tensor's format inside the card's REAL budget:
               the running model's footprint is known-good, so the budget is
               "what the current model's weights use + the free VRAM measured
               right now - a margin you choose". Sizes are exact: llama-quantize
               --dry-run reports every tensor's size per format in ~0.2 s.
  D  build     llama-quantize with a per-tensor type file, at low priority on
     verify    the CPU; verification is an optimizer run in "models" mode (the
               same agentic/coding suites, speed probes and depth curve, with
               the saved settings snapshotted and restored).
  E  parts     output layer, MTP head, embeddings and projector are chosen
               separately; sd.cpp and audio.cpp models convert here too.
  F  watch     a daily check of each source's upstream revision and of the
               quantizer build. It only reports; it never downloads.

Quality is ranked with PUBLISHED SENSITIVITY PRIORS (which tensors hurt most
when squeezed) until KL divergence is measured on this box. Every plan says
so. The index is relative to the model the target instance runs today.

Nothing here changes a setting or a running server except the bench and
verify jobs, which stop and ALWAYS restart the instance they measure.
"""
import hashlib, json, math, os, re, shutil, signal, struct, subprocess, threading, time
import urllib.request
from pathlib import Path

P = None                    # panel module
E = None                    # engines module
O = None                    # optimizer module

_lock = threading.RLock()
_job = None                 # the active job (one at a time: they share CPU, disk and GPU)
_proc = None                # its running subprocess
_cancel = threading.Event()

MIB = 1048576
KEEP_FREE_DEFAULT = 512     # MiB left free on each card. The live "perfect fit" ran at 163.
DISK_RESERVE_GB = 20        # never fill the disk past this

# ---------------------------------------------------------------------------
# formats
# nominal bits/weight, and the "effective bits" the quality prior uses. The
# i-quants' non-linear grids do better than their size says; Q4_0 worse.
# Without an importance matrix the i-quant bonus is smaller (it is here).
# ---------------------------------------------------------------------------
TYPES = {
    "Q2_K": (2.625, 2.45), "IQ3_S": (3.4375, 3.35), "Q3_K": (3.4375, 3.2),
    "Q4_0": (4.5, 4.0), "IQ4_NL": (4.5, 4.35), "IQ4_XS": (4.25, 4.3), "Q4_K": (4.5, 4.4),
    "Q5_K": (5.5, 5.4), "Q6_K": (6.5625, 6.4), "Q8_0": (8.5, 8.3),
    "BF16": (16.0, 16.0), "F16": (16.0, 16.0), "F32": (32.0, 32.0),
}
# formats the solver may use; each gets one dry run per source
TABLE_TYPES = ["IQ3_S", "Q3_K", "Q4_0", "IQ4_NL", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]
BASE_CHOICES = ["IQ4_XS", "Q4_K", "IQ4_NL", "Q4_0", "Q5_K", "Q6_K", "IQ3_S", "Q3_K"]
BENCH_TYPES = ["Q4_0", "IQ4_NL", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K"]
GGML_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
             9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
             16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
             22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
             29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
EXTRA_BITS = {"Q4_1": (5.0, 4.5), "Q5_0": (5.5, 5.0), "Q5_1": (6.0, 5.3), "IQ3_XXS": (3.06, 3.0),
              "IQ2_XXS": (2.06, 2.0), "IQ2_XS": (2.31, 2.25), "IQ2_S": (2.5, 2.4),
              "IQ1_S": (1.56, 1.5), "IQ1_M": (1.75, 1.7), "MXFP4": (4.25, 3.9)}

# ---------------------------------------------------------------------------
# the quality prior: relative damage per parameter when a tensor is squeezed.
# From llama.cpp's own k-quant mixtures (which bump attn_v / ffn_down and the
# first and last layers) and published per-tensor sensitivity studies. The
# output layer shapes every token's distribution; the MTP head only drafts,
# but its precision is the draft acceptance rate, i.e. speed.
# ---------------------------------------------------------------------------
IMPORTANCE = dict(output=8.0, mtp=3.0, attn_v=3.0, attn_qkv=2.5, ffn_down=2.0,
                  attn_output=1.5, ssm_out=1.5, attn_k=1.5, attn_q=1.2, ffn_up=1.0,
                  ffn_gate=1.0, attn_gate=1.0, ssm_small=2.0, embed=0.5, experts=0.7,
                  shexp=1.5, other=1.0)
EDGE_ROLES = {"attn_v", "attn_qkv", "ffn_down"}
ROLE_LABEL = dict(output="output layer (next-token logits)", mtp="MTP draft head",
                  attn_v="attention values", attn_qkv="linear-attention QKV",
                  ffn_down="FFN down projection", attn_output="attention output",
                  ssm_out="linear-attention output", attn_k="attention keys",
                  attn_q="attention queries", ffn_up="FFN up", ffn_gate="FFN gate",
                  attn_gate="attention gate", ssm_small="linear-attention gates (tiny)",
                  embed="token embeddings", experts="MoE experts", shexp="shared expert",
                  other="other")

VENDOR_GUIDE = [
    dict(vendor="amd", title="AMD RDNA3 (7900 XTX / XT / GRE, 7800, 7700, 7600)",
         measured_here=True,
         text="llama.cpp runs 4-bit weights with 8-bit activations (the INT8 path), so 4-bit "
              "wins through memory bandwidth, not INT4 math: decode reads every GPU weight once "
              "per token. Smaller files decode faster almost linearly. No FP8/FP4 hardware; FP4 "
              "formats only save memory. Measure Q4_0, IQ4_NL, IQ4_XS and Q4_K; spend saved "
              "bytes on the output layer, attention values and FFN down.",
         formats=["IQ4_XS", "Q4_K", "IQ4_NL", "Q4_0", "Q6_K", "Q8_0"]),
    dict(vendor="amd4", title="AMD RDNA4 (9070 XT / 9070 / 9060)", measured_here=False,
         text="Adds FP8 and faster INT8 WMMA. Published results show some k-quants slower than "
              "on RDNA3 and Q4_0/Q8_0 strong. Unverified on this box: benchmark before trusting.",
         formats=["Q4_0", "IQ4_XS", "Q4_K", "Q8_0"]),
    dict(vendor="nvidia", title="NVIDIA Turing / Ampere / Ada (RTX 20/30/40)", measured_here=False,
         text="INT8 tensor cores run the k-quants and i-quants through int8 MMQ kernels. CUDA's "
              "k-quant and IQ4_XS kernels are mature; decode is still bandwidth-bound. Not yet "
              "measured on this box: run Measure formats on the card.",
         formats=["Q4_K", "IQ4_XS", "Q5_K", "Q6_K", "Q8_0"]),
    dict(vendor="nvidia5", title="NVIDIA Blackwell (RTX 50)", measured_here=False,
         text="Native FP4 (NVFP4/MXFP4) tensor cores; llama.cpp merged the NVFP4 path in April "
              "2026. MXFP4 is only offered as the MoE preset here. Unverified on this box.",
         formats=["MXFP4", "Q4_K", "IQ4_XS", "Q8_0"]),
    dict(vendor="intel", title="Intel Arc (Alchemist / Battlemage Xe2)", measured_here=False,
         text="SYCL and Vulkan both work; Q4_0 and Q8_0 have the 'reorder' fast kernels, so they "
              "often beat k-quants despite the size. No FP8. Unverified on this box. Integrated "
              "GPUs are never used as inference devices.",
         formats=["Q4_0", "Q8_0", "Q4_K"]),
    dict(vendor="cpu", title="CPU (AVX2 / AVX-512)", measured_here=False,
         text="Repacked Q4_0 / IQ4_NL (the runtime converts them to interleaved layouts) are the "
              "fastest CPU formats; memory bandwidth decides the rest.",
         formats=["Q4_0", "IQ4_NL", "Q4_K", "Q8_0"]),
]


def bind(panel_module, engines_module, optimizer_module):
    global P, E, O
    P, E, O = panel_module, engines_module, optimizer_module


def _fit_dir(*sub):
    d = P.PANEL.joinpath("fit", *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir():
    return P.MODELS / "fit"


def test_dir():
    return P.MODELS / "fit-test"


def src_root():
    return P.MODELS / "src"


def fitq_home():
    return P.HOME / "fitquant"


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _bits(t):
    t = str(t).upper()
    return TYPES.get(t) or EXTRA_BITS.get(t) or (16.0, 16.0)


def _free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


# ============================================================================
# tools
# ============================================================================
def _build_no(bindir):
    m = re.search(r"b(\d{4,6})", str(bindir or ""))
    return int(m.group(1)) if m else None


def quantize_bin():
    """llama-quantize from the active CPU build, else the newest build that has one."""
    cands = []
    act = P.active_build("cpu")
    if act:
        cands.append(Path(act))
    try:
        cands += sorted((Path(p).parent for p in (P.LLAMA).glob("*/*/llama-quantize")),
                        key=lambda d: -(_build_no(d) or 0))
    except OSError:
        pass
    for d in cands:
        if (d / "llama-quantize").is_file():
            return d / "llama-quantize"
    return None


def bench_bin(backend):
    act = P.active_build(backend)
    if act and (Path(act) / "llama-bench").is_file():
        return Path(act) / "llama-bench"
    return None


def converter():
    """phase A's private converter (uv venv + llama.cpp checkout), if set up."""
    home = fitq_home()
    py = home / "venv" / "bin" / "python"
    conv = next(iter(sorted(home.glob("llama.cpp-b*/convert_hf_to_gguf.py"))), None)
    script = next((s for s in (home / "phaseA-source.sh", P.PANEL / "fit" / "phaseA-source.sh")
                   if s.exists()), home / "phaseA-source.sh")
    return dict(python=str(py) if py.exists() else None,
                convert=str(conv) if conv else None,
                phase_a=str(script) if script.exists() else None)


# ============================================================================
# GGUF header: tensor names, types, shapes, sizes (all shards)
# ============================================================================
_hdr_cache = {}


def gguf_tensors(path):
    """(kv scalars, {name: dict(type, shape, bytes)}) or None."""
    shards = P.model_shards(path)
    try:
        key = tuple((str(s), s.stat().st_mtime_ns, s.stat().st_size) for s in shards)
    except OSError:
        return None
    if key in _hdr_cache:
        return _hdr_cache[key]
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    kv, out = {}, {}
    try:
        for n_shard, shard in enumerate(shards):
            end = shard.stat().st_size
            with open(shard, "rb") as f:
                if f.read(4) != b"GGUF":
                    return None
                f.read(4)
                n_t, n_kv = struct.unpack("<QQ", f.read(16))
                rd = lambda: f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
                align = 32
                for _ in range(n_kv):
                    k = rd()
                    t = struct.unpack("<I", f.read(4))[0]
                    if t == 8:
                        val = rd()
                    elif t == 9:
                        et, ln = struct.unpack("<IQ", f.read(12))
                        if et == 8:
                            for _i in range(ln):
                                rd()
                        else:
                            f.read(SZ.get(et, 4) * ln)
                        val = None
                    elif t == 6:
                        val = struct.unpack("<f", f.read(4))[0]
                    elif t == 12:
                        val = struct.unpack("<d", f.read(8))[0]
                    else:
                        val = int.from_bytes(f.read(SZ.get(t, 4)), "little",
                                             signed=t in (1, 3, 5, 11))
                    if n_shard == 0 and not (isinstance(val, str) and len(val) > 512):
                        kv[k] = val
                    if k == "general.alignment" and isinstance(val, int) and val:
                        align = val
                infos = []
                for _ in range(n_t):
                    name = rd()
                    nd = struct.unpack("<I", f.read(4))[0]
                    dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
                    ty, off = struct.unpack("<IQ", f.read(12))
                    infos.append((off, name, ty, dims))
                data = (f.tell() + align - 1) // align * align
            infos.sort()
            for i, (off, name, ty, dims) in enumerate(infos):
                nxt = infos[i + 1][0] if i + 1 < len(infos) else end - data
                out[name] = dict(type=GGML_TYPE.get(ty, f"type{ty}"), shape=dims,
                                 bytes=max(nxt - off, 0))
    except (OSError, struct.error, UnicodeDecodeError):
        return None
    res = (kv, out)
    if len(_hdr_cache) > 8:
        _hdr_cache.clear()
    _hdr_cache[key] = res
    return res


def _arch_counts(kv):
    arch = kv.get("general.architecture") or ""
    n = kv.get(f"{arch}.block_count")
    nextn = kv.get(f"{arch}.nextn_predict_layers") or 0
    return arch, (int(n) if n else None), int(nextn or 0)


_BLK = re.compile(r"^blk\.(\d+)\.(.+?)(\.weight|\.bias)?$")


def role_of(name, n_main):
    if name.startswith("output.") and "norm" not in name:
        return "output", None
    if name.startswith(("token_embd", "per_layer_token_embd")):
        return "embed", None
    m = _BLK.match(name)
    if not m:
        return "other", None
    il, stem = int(m.group(1)), m.group(2)
    if n_main is not None and il >= n_main:
        return "mtp", il
    if "_exps" in stem:
        return "experts", il
    if "_shexp" in stem:
        return "shexp", il
    if stem in ("ssm_alpha", "ssm_beta", "ssm_ba"):
        return "ssm_small", il
    for r in ("attn_qkv", "attn_v", "attn_k", "attn_q", "attn_output", "attn_gate",
              "ffn_down", "ffn_up", "ffn_gate", "ssm_out"):
        if stem == r:
            return r, il
    return "other", il


def importance(role, il, n_main):
    w = IMPORTANCE.get(role, 1.0)
    if role in EDGE_ROLES and il is not None and n_main:
        if il < max(1, n_main // 8) or il >= n_main - max(1, n_main // 8):
            w *= 1.4
    return w


def _nparams(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _loss(imp, nparams, t):
    return imp * nparams * 4.0 ** (-_bits(t)[1])


# ============================================================================
# sources and the per-format size table (llama-quantize --dry-run)
# ============================================================================
def sources():
    """Full-precision GGUFs under ~/models/src, with their phase A manifest."""
    out = []
    root = src_root()
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        man = _read(d / "manifest.json", {}) or {}
        for f in sorted(d.glob("*.gguf")):
            if f.name.endswith(".part") or re.search(r"-0000[2-9]-of-", f.name):
                continue
            hdr = gguf_tensors(f)
            if not hdr:
                continue
            kv, ts = hdr
            types = {t["type"] for t in ts.values()}
            if not types & {"BF16", "F16", "F32"} or types - {"BF16", "F16", "F32", "I32", "I64"}:
                continue            # already quantized: a requant stacks rounding error
            arch, n_layer, nextn = _arch_counts(kv)
            out.append(dict(path=str(f), name=f.name, dir=d.name, arch=arch,
                            n_layer=n_layer, nextn=nextn,
                            gib=round(P.model_bytes(f) / 2**30, 2),
                            repo=man.get("repo"), revision=man.get("revision"),
                            checked=man.get("checked"),
                            params_b=round(sum(_nparams(t["shape"]) for t in ts.values()) / 1e9, 2)))
    return out


_DRY_LINE = re.compile(r"^\[\s*\d+/\s*\d+\]\s+(\S+)\s+- \[([^\]]*)\], type =\s*(\S+), "
                       r"size =\s*([\d.]+) MiB(?: ->\s*([\d.]+) MiB \((\S+)\))?")


def _dry_run(src, base, type_file=None, pure=True):
    q = quantize_bin()
    if not q:
        raise ValueError("no llama-quantize: install a llama.cpp CPU build in the Builds tab")
    argv = ["nice", "-n", "10", str(q), "--dry-run"]
    if pure:
        argv.append("--pure")
    if type_file:
        argv += ["--tensor-type-file", str(type_file)]
    argv += [str(src), base]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    txt = r.stdout + r.stderr
    rows = {}
    for line in txt.splitlines():
        m = _DRY_LINE.match(line.strip())
        if m:
            name, shape, st, smib, qmib, qt = m.groups()
            rows[name] = dict(shape=[int(x) for x in shape.split(",") if x.strip()],
                              src_type=st.upper(), src_mib=float(smib),
                              mib=float(qmib) if qmib else float(smib),
                              type=(qt or st).upper(), quantized=bool(qmib))
    if not rows:
        raise ValueError(f"llama-quantize --dry-run gave no tensor list (exit {r.returncode}): "
                         f"{txt.strip()[-400:]}")
    tot = re.search(r"quant size\s*=\s*([\d.]+) MiB", txt)
    return rows, (float(tot.group(1)) if tot else sum(x["mib"] for x in rows.values()))


def size_table(src):
    """{"tensors": {name: {shape, src_type, src_mib, quantized}},
        "sizes": {TYPE: {name: MiB}}, "actual": {TYPE: {name: type it really got}}}.
    One dry run per format, cached per (file, quantizer build)."""
    src = Path(src)
    st = src.stat()
    q = quantize_bin()
    key = hashlib.sha1(f"{src}|{st.st_size}|{st.st_mtime_ns}|{q}".encode()).hexdigest()[:16]
    cf = _fit_dir("cache") / f"sizes-{key}.json"
    cached = _read(cf)
    if cached and set(TABLE_TYPES) <= set(cached.get("sizes", {})):
        return cached
    tensors, sizes, actual = {}, {}, {}
    for t in TABLE_TYPES:
        rows, _tot = _dry_run(src, t)
        sizes[t], actual[t] = {}, {}
        for name, r in rows.items():
            tensors.setdefault(name, dict(shape=r["shape"], src_type=r["src_type"],
                                          src_mib=r["src_mib"], quantized=r["quantized"]))
            if r["quantized"]:
                sizes[t][name] = r["mib"]
                actual[t][name] = r["type"]
    hdr = gguf_tensors(src)
    kv = hdr[0] if hdr else {}
    arch, n_layer, nextn = _arch_counts(kv)
    res = dict(source=str(src), key=key, quantizer=str(q), built=_now(), arch=arch,
               n_layer=n_layer, nextn=nextn, tensors=tensors, sizes=sizes, actual=actual)
    _atomic(cf, res)
    return res


# ============================================================================
# card speed profiles (phase B data)
# ============================================================================
def _card_id(pci):
    try:
        base = Path("/sys/bus/pci/devices") / pci
        return (base / "vendor").read_text().strip()[2:] + "-" + (base / "device").read_text().strip()[2:]
    except OSError:
        return None


def _card_file(card, backend):
    return _fit_dir("cards") / f"{card}-{backend}.json"


def card_profile(pci, backend):
    card = _card_id(pci)
    return _read(_card_file(card, backend)) if card else None


def _fit_speed(types):
    """Decode time per token = a + sum(bytes / bw[type]). Fit `a` across the
    measured formats (a shared overhead: norms, attention, launches), then
    each format's effective bandwidth. Per depth, `a` absorbs attention cost."""
    pts = [(v["gpu_bytes"], 1.0 / v["tg"]["0"]) for v in types.values()
           if v.get("gpu_bytes") and (v.get("tg") or {}).get("0")]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    k = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx if sxx else 0
    a = my - k * mx
    a = min(max(a, 0.0), 0.6 * min(p[1] for p in pts))
    bw = {}
    for t, v in types.items():
        tg0 = (v.get("tg") or {}).get("0")
        if v.get("gpu_bytes") and tg0 and 1 / tg0 > a:
            bw[t] = v["gpu_bytes"] / (1 / tg0 - a)
    depth_a = {}
    depths = sorted({int(d) for v in types.values() for d in (v.get("tg") or {})})
    for d in depths:
        xs = [1 / v["tg"][str(d)] - v["gpu_bytes"] / bw[t] for t, v in types.items()
              if t in bw and (v.get("tg") or {}).get(str(d))]
        if xs:
            xs.sort()
            depth_a[str(d)] = max(xs[len(xs) // 2], 0.0)
    return dict(a=a, depth_a=depth_a, bw=bw, fitted=_now(),
                note=f"fixed cost {a * 1000:.2f} ms/token; effective bandwidth " +
                     ", ".join(f"{t} {b / 2**30:.0f} GiB/s" for t, b in sorted(bw.items(), key=lambda x: -x[1])))


def import_bench(files, pci, backend, source=None, by="import"):
    """Fold llama-bench JSONL files into the card profile for `pci`."""
    card = _card_id(pci)
    if not card:
        raise ValueError(f"device {pci} is not present")
    cf = _card_file(card, backend)
    prof = _read(cf) or dict(card=card, backend=backend, types={}, runs=[])
    rec = P._device_record(pci) or {}
    prof.update(name=rec.get("name") or pci, pci=pci, vendor=rec.get("vendor"))
    table = None
    if source and Path(source).exists():
        try:
            table = size_table(source)
        except Exception:
            table = None
    n = 0
    for fn in files:
        for line in open(fn):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            mf = d.get("model_filename") or ""
            m = re.search(r"pure-([A-Z0-9_]+)\.gguf$", mf)
            t = m.group(1) if m else (str(d.get("model_type") or "").split()[-4:-3] or ["?"])[0]
            t = t.upper()
            slot = prof["types"].setdefault(t, dict(tg={}, pp={}))
            kind = "pp" if d.get("n_prompt") else "tg"
            slot[kind][str(int(d.get("n_depth") or 0))] = round(float(d.get("avg_ts") or 0), 2)
            slot["file_bytes"] = int(d.get("model_size") or 0)
            slot["build"] = d.get("build_number")
            slot["gpu_info"] = d.get("gpu_info")
            # GPU-resident bytes: the input embeddings stay on the CPU
            emb = None
            hdr = gguf_tensors(mf) if mf and os.path.exists(mf) else None
            if hdr:
                emb = sum(x["bytes"] for k, x in hdr[1].items() if k.startswith("token_embd"))
            elif table and t in table["sizes"]:
                emb = int(sum(v for k, v in table["sizes"][t].items()
                              if k.startswith("token_embd")) * MIB)
            if emb is not None:
                slot["gpu_bytes"] = max(slot["file_bytes"] - emb, 0)
            n += 1
    prof["runs"] = (prof.get("runs") or [])[-20:] + [dict(at=_now(), by=by, rows=n,
                                                          files=[str(f) for f in files])]
    prof["model"] = _fit_speed(prof["types"])
    prof["updated"] = _now()
    _atomic(cf, prof)
    return prof


def _auto_import():
    """First use: fold phase B's script results in, if they exist and no profile does."""
    res = fitq_home() / "results" / "phaseB"
    files = sorted(res.glob("*.jsonl")) if res.is_dir() else []
    if not files:
        return None
    marker = _fit_dir("cards") / ".phaseB-imported"
    newest = max(f.stat().st_mtime for f in files)
    if marker.exists() and marker.stat().st_mtime >= newest:
        return None
    gpu = ""
    try:
        gpu = json.loads(open(files[0]).readline()).get("gpu_info") or ""
    except (OSError, ValueError):
        pass
    want = "amd" if re.search(r"AMD|Radeon", gpu) else "nvidia" if re.search(r"NVIDIA|GeForce|RTX", gpu) else "intel"
    devs = [d for d in P.gpu_devices(probe=False) if d["vendor"] == want and d["pci"] != "cpu"]
    if len(devs) != 1:
        return None
    src = next((s["path"] for s in sources()), None)
    prof = import_bench(files, devs[0]["pci"], "vulkan", source=src, by="phase B script")
    marker.write_text(_now())
    return prof


def cards():
    out = []
    for f in sorted(_fit_dir("cards").glob("*.json")):
        p = _read(f)
        if p:
            out.append(p)
    return out


# ============================================================================
# the target: budget per device, reference model
# ============================================================================
def _vram_now(pci):
    """(total MiB, used MiB) measured right now, or (None, None)."""
    base = Path("/sys/bus/pci/devices") / pci
    try:
        tot = int((base / "mem_info_vram_total").read_text()) // MIB
        used = int((base / "mem_info_vram_used").read_text()) // MIB
        return tot, used
    except (OSError, ValueError):
        pass
    try:
        r = P._nvidia_smi(pci)
        if r:
            return int(float(r["mem_total_mib"])), int(float(r["mem_used_mib"]))
    except Exception:
        pass
    return None, None


def target(iid, ctx=None, keep_free=KEEP_FREE_DEFAULT):
    """Everything the solver needs about where the model will run."""
    inst = P.get_instance(iid)
    if inst.get("engine") in P.GEN_ENGINES:
        raise ValueError(f"{iid} runs {inst['engine']}; Fit plans llama.cpp models")
    with P.using_instance(inst):
        params = dict(P.load_params())
        backend = str(params.get("BACKEND") or "vulkan")
        model = str(params.get("MODEL") or "")
        pid = P.server_pid()
        running_model = None
        if pid:
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
                running_model = P._argv_get(argv, ("-m", "--model"))
            except OSError:
                pass
        ref_pl = P.estimate_placement(params) if model and os.path.exists(model) else None
        est = None
        try:
            est = P.estimate(params)
        except Exception:
            pass
        devs = []
        if ref_pl and ref_pl["devices"]:
            fps = P._device_footprints(params, ref_pl)
            for d, fp in zip(ref_pl["devices"], fps):
                tot, used = _vram_now(d["pci"]) if d.get("pci") else (None, None)
                live = bool(pid and running_model and os.path.realpath(running_model) ==
                            os.path.realpath(model) and used is not None)
                if live:
                    free, basis = tot - used, "measured now, with this model loaded"
                else:
                    free, basis = fp["headroom_mib"], "estimated (the instance is not running this model)"
                devs.append(dict(name=d["name"], pci=d["pci"], label=d["label"], vendor=d["vendor"],
                                 vram_total_mib=tot or d["vram_total_mib"],
                                 ref_weights_mib=d["weights_mib"], free_mib=int(free),
                                 kv_frac=d["kv_frac"], basis=basis, live=live,
                                 kv_mib=fp["kv_mib"]))
        elif backend != "cpu":
            head = (est or {}).get("vram", {}).get("headroom_mib")
            dev = (inst.get("devices") or [inst.get("device")])[0]
            rec = P._device_record(dev) or {}
            devs.append(dict(name="GPU0", pci=dev, label=rec.get("name"), vendor=rec.get("vendor"),
                             vram_total_mib=rec.get("vram_total_mib"), ref_weights_mib=0,
                             free_mib=int(head or 0), kv_frac=1.0, live=False, kv_mib=0,
                             basis="estimated (no readable reference model)"))
    cur_ctx = int(params.get("CTX") or 0)
    new_ctx = int(ctx or cur_ctx or 0)
    kv_delta = 0.0
    if new_ctx != cur_ctx and cur_ctx:
        try:
            per_tok, _fixed, _b = P._kv_model(model, backend)
            kvt = str(params.get("KV_TYPE") or "q5_1")
            kv_delta = (cur_ctx - new_ctx) * per_tok * (P.KV_BPW.get(kvt, 6.0) / 6.0)
        except Exception:
            kv_delta = 0.0
    for d in devs:
        d["kv_freed_mib"] = round(kv_delta * d["kv_frac"])
        d["budget_mib"] = int(d["ref_weights_mib"] + d["free_mib"] + d["kv_freed_mib"] - keep_free)
    return dict(instance=iid, instance_name=inst.get("name"), backend=backend, params=params,
                model=model, running=bool(pid), running_model=running_model, ctx=new_ctx,
                cur_ctx=cur_ctx, keep_free_mib=keep_free, devices=devs,
                cpu_only=backend == "cpu" or not devs)


def _where(params, names_sizes):
    """{tensor: device index or None (CPU)} for this instance's placement rules."""
    pl = P.estimate_placement(params, layout=({}, names_sizes), detail=True)
    return (pl or {}).get("where") or {}


# ============================================================================
# phase C: the solver
# ============================================================================
def _ladder(role, base, avail):
    """Formats a tensor of this role may take, ascending by size."""
    up = ["Q5_K", "Q6_K", "Q8_0", "BF16"]
    down = ["IQ3_S", "Q3_K"] if role not in ("output", "mtp", "attn_v", "embed") else []
    lad = [t for t in down if t in avail] + [base] + [t for t in up if t in avail or t == "BF16"]
    out = []
    for t in sorted(set(lad), key=lambda t: _bits(t)[0]):
        if _bits(t)[0] < _bits(base)[0] and t not in down:
            continue
        out.append(t)
    return out


class _Plan:
    def __init__(self, tbl, where, devs, card_models, base, n_main):
        self.tbl, self.where, self.devs, self.cm, self.base = tbl, where, devs, card_models, base
        self.n_main = n_main
        self.names = [n for n, t in tbl["tensors"].items() if t["quantized"]]
        self.info = {}
        for n in self.names:
            r, il = role_of(n, n_main)
            self.info[n] = dict(role=r, il=il, imp=importance(r, il, n_main),
                                np=_nparams(tbl["tensors"][n]["shape"]))
        self.types = {n: base for n in self.names}

    def mib(self, n, t):
        if t in ("BF16", "F16"):
            return self.tbl["tensors"][n]["src_mib"] * (1.0 if self.tbl["tensors"][n]["src_type"] in ("BF16", "F16") else 0.5)
        return self.tbl["sizes"].get(t, {}).get(n)

    def real_type(self, n, t):
        if t in ("BF16", "F16"):
            return t
        return self.tbl["actual"].get(t, {}).get(n, t)

    def dev_of(self, n):
        return self.where.get(n)

    def dev_mib(self):
        per = [0.0] * len(self.devs)
        cpu = 0.0
        for n in self.names:
            d = self.dev_of(n)
            m = self.mib(n, self.types[n]) or 0
            if d is None:
                cpu += m
            else:
                per[d] += m
        # non-quantized tensors (norms, ssm_a, conv) keep their size
        for n, t in self.tbl["tensors"].items():
            if t["quantized"]:
                continue
            d = self.dev_of(n)
            if d is None:
                cpu += t["src_mib"]
            else:
                per[d] += t["src_mib"]
        return per, cpu

    def secs(self, n, t, depth="0"):
        d = self.dev_of(n)
        if d is None:
            return 0.0                          # CPU-resident: input embeddings, a row lookup
        m = self.cm[d]
        if not m:
            return 0.0
        bw = m["bw"].get(self.real_type(n, t))
        if not bw:
            known = sorted(m["bw"].items(), key=lambda kv: abs(_bits(kv[0])[0] - _bits(t)[0]))
            bw = known[0][1] if known else 800 * MIB * 1024
        return (self.mib(n, t) or 0) * MIB / bw

    def decode(self, depth="0"):
        if not any(self.cm):
            return None
        m0 = next(m for m in self.cm if m)
        a = m0["depth_a"].get(depth, m0["a"]) if depth != "0" else m0["a"]
        t = a + sum(self.secs(n, self.types[n]) for n in self.names)
        return round(1.0 / t, 1) if t > 0 else None

    def loss(self):
        return sum(_loss(i["imp"], i["np"], self.real_type(n, self.types[n]))
                   for n, i in self.info.items())


def _ref_metrics(ref_path, plan, n_main):
    """The same numbers for the model the instance runs today."""
    hdr = gguf_tensors(ref_path) if ref_path and os.path.exists(ref_path) else None
    if not hdr:
        return None
    kv, ts = hdr
    loss, secs, missing = 0.0, 0.0, 0
    m0 = next((m for m in plan.cm if m), None)
    for n, info in plan.info.items():
        t = ts.get(n)
        if not t:
            missing += 1
            continue
        loss += _loss(info["imp"], info["np"], t["type"])
        d = plan.dev_of(n)
        if d is not None and plan.cm[d]:
            bw = plan.cm[d]["bw"].get(t["type"])
            if not bw:
                known = sorted(plan.cm[d]["bw"].items(),
                               key=lambda kvp: abs(_bits(kvp[0])[0] - _bits(t["type"])[0]))
                bw = known[0][1] if known else None
            if bw:
                secs += t["bytes"] / bw
    mix = {}
    for n, info in plan.info.items():
        t = ts.get(n)
        if t:
            mix.setdefault(info["role"], {}).setdefault(t["type"], 0)
            mix[info["role"]][t["type"]] += 1
    dec = {}
    if m0:
        for depth in ["0"] + sorted(m0["depth_a"], key=int):
            a = m0["a"] if depth == "0" else m0["depth_a"][depth]
            dec[depth] = round(1.0 / (a + secs), 1) if secs else None
    return dict(path=ref_path, name=Path(ref_path).name, loss=loss, decode=dec,
                missing=missing, mix=mix,
                output_type=(ts.get("output.weight") or {}).get("type"),
                file_mib=round(P.model_bytes(ref_path) / MIB))


def _greedy(plan, key, allowed, budgets, stop=None, time_cap=None):
    """Upgrade one ladder step at a time, best `key` ratio first, while every
    device stays inside its budget (and the time cap, when given)."""
    ladders = {n: _ladder(plan.info[n]["role"], plan.base, plan.tbl["sizes"]) for n in plan.names}
    per, _cpu = plan.dev_mib()
    tsec = sum(plan.secs(n, plan.types[n]) for n in plan.names)
    m0 = next((m for m in plan.cm if m), None)
    a = m0["a"] if m0 else 0.0
    steps = 0
    while True:
        if stop and stop():
            break
        best, best_r = None, 0.0
        for n in plan.names:
            if n in allowed and not allowed[n]:
                continue
            lad = ladders[n]
            cur = plan.types[n]
            i = lad.index(cur) if cur in lad else None
            if i is None or i + 1 >= len(lad):
                continue
            nxt = lad[i + 1]
            m1 = plan.mib(n, nxt)
            if m1 is None:
                continue
            dm = m1 - (plan.mib(n, cur) or 0)
            d = plan.dev_of(n)
            if d is not None and per[d] + dm > budgets[d]:
                continue
            ds = plan.secs(n, nxt) - plan.secs(n, cur)
            if time_cap is not None and a + tsec + ds > time_cap:
                continue
            info = plan.info[n]
            gain = _loss(info["imp"], info["np"], plan.real_type(n, cur)) - \
                _loss(info["imp"], info["np"], plan.real_type(n, nxt))
            if gain <= 0:
                continue
            cost = ds if key == "time" and d is not None else max(dm, 1e-6)
            r = gain / max(cost, 1e-9)
            if r > best_r:
                best, best_r = (n, nxt, dm, ds, d), r
        if not best:
            break
        n, nxt, dm, ds, d = best
        plan.types[n] = nxt
        if d is not None:
            per[d] += dm
        tsec += ds
        steps += 1
        if steps > 20000:
            break
    return steps


def _shrink(plan, budgets):
    """The base format does not fit: drop the least important tensors below it."""
    per, _ = plan.dev_mib()
    over = [per[i] - budgets[i] for i in range(len(budgets))]
    if all(o <= 0 for o in over):
        return True
    ladders = {n: _ladder(plan.info[n]["role"], plan.base, plan.tbl["sizes"]) for n in plan.names}
    while any(o > 0 for o in over):
        best, best_r = None, None
        for n in plan.names:
            d = plan.dev_of(n)
            if d is None or over[d] <= 0:
                continue
            lad = ladders[n]
            cur = plan.types[n]
            i = lad.index(cur) if cur in lad else 0
            if i == 0:
                continue
            nxt = lad[i - 1]
            m1 = plan.mib(n, nxt)
            if m1 is None:
                continue
            saved = (plan.mib(n, cur) or 0) - m1
            if saved <= 0:
                continue
            info = plan.info[n]
            hurt = _loss(info["imp"], info["np"], plan.real_type(n, nxt)) - \
                _loss(info["imp"], info["np"], plan.real_type(n, cur))
            r = hurt / saved
            if best_r is None or r < best_r:
                best, best_r = (n, nxt, saved, d), r
        if not best:
            return False
        n, nxt, saved, d = best
        plan.types[n] = nxt
        over[d] -= saved
    return True


def _apply_components(plan, comp, where_cpu_embed):
    """Phase E pins. Returns {tensor: False} for tensors the greedy must not move."""
    frozen = {}
    for n, info in plan.info.items():
        r = info["role"]
        choice = None
        if r == "output":
            choice = comp.get("output") or "auto"
            if choice == "auto":
                plan.types[n] = "Q6_K" if _bits("Q6_K")[0] > _bits(plan.base)[0] else plan.base
                continue
        elif r == "mtp":
            choice = comp.get("mtp") or "auto"
            if choice == "auto":
                plan.types[n] = "Q8_0"
                continue
        elif r == "embed":
            choice = comp.get("embed") or "auto"
            if choice == "auto":
                # input embeddings live on the CPU: VRAM-free and not read per token,
                # so they cost only host RAM
                plan.types[n] = "Q8_0" if where_cpu_embed else plan.base
                frozen[n] = False
                continue
        elif r == "ssm_small":
            plan.types[n] = "Q8_0"              # a few KiB each; never worth the risk
            frozen[n] = False
            continue
        if choice and choice != "auto":
            plan.types[n] = plan.base if choice == "base" else choice
            frozen[n] = False
    return frozen


GOALS = {
    "faster": "Same quality as today's model (by the prior), as fast as possible",
    "better": "Same decode speed as today's model, best quality that fits",
    "max": "Best quality that fits the VRAM budget, speed permitting",
}


def solve(body):
    """Phase C. body: source, instance, goals[], base (auto|TYPE), keep_free_mib,
    ctx, components{output, mtp, embed}. Returns one plan per goal, saved."""
    src = str(body.get("source") or "")
    if not src or not os.path.exists(src):
        raise ValueError("pick a full-precision source (run phase A first)")
    iid = str(body.get("instance") or "main")
    keep = int(body.get("keep_free_mib") if body.get("keep_free_mib") not in (None, "") else KEEP_FREE_DEFAULT)
    if not 0 <= keep <= 8192:
        raise ValueError("keep_free_mib 0-8192")
    ctx = int(body.get("ctx") or 0) or None
    try:
        _auto_import()              # phase B script results, if not folded in yet
    except Exception:
        pass
    tgt = target(iid, ctx=ctx, keep_free=keep)
    tbl = size_table(src)
    arch, n_layer, nextn = tbl["arch"], tbl["n_layer"], tbl["nextn"]
    n_main = (n_layer - nextn) if n_layer else None
    names_sizes = {n: int(t["src_mib"] * MIB) for n, t in tbl["tensors"].items()}
    params = dict(tgt["params"], MODEL=src)
    with P.using_instance(P.get_instance(iid)):
        where = _where(params, names_sizes) if not tgt["cpu_only"] else {}
    devs = tgt["devices"]
    cms, notes, warns = [], [], []
    for d in devs:
        prof = card_profile(d["pci"], tgt["backend"]) if d.get("pci") else None
        cms.append((prof or {}).get("model"))
        d["card_profile"] = bool(prof and prof.get("model"))
        if not d["card_profile"]:
            warns.append(f"{d['name']} ({d.get('label') or d['pci']}): no measured format speeds. "
                         "Speed predictions are off for it; run 'Measure formats' first.")
    budgets = [d["budget_mib"] for d in devs]
    # base format: the fastest measured 4-bit format per GiB on the main card
    base = str(body.get("base") or "auto").upper()
    main_prof = next((card_profile(d["pci"], tgt["backend"]) for d in devs if d.get("pci")), None)
    if base == "AUTO":
        cand = []
        for t in BASE_CHOICES:
            v = ((main_prof or {}).get("types") or {}).get(t) or {}
            if (v.get("tg") or {}).get("0") and _bits(t)[0] <= 4.5:
                cand.append((v["tg"]["0"], t))
        base = max(cand)[1] if cand else "Q4_K"
        why_base = (f"{base} is the fastest 4-bit format measured on this card "
                    f"({max(cand)[0]} t/s decode at depth 0" + (", and the smallest" if base == "IQ4_XS" else "") + ")"
                    if cand else f"{base}: no card measurements, so the safe default")
    else:
        if base not in TABLE_TYPES:
            raise ValueError(f"base must be auto or one of {', '.join(TABLE_TYPES)}")
        why_base = f"{base}: chosen by you"
    comp = dict(body.get("components") or {})
    for k, allowed_v in (("output", ("auto", "base", "Q6_K", "Q8_0", "BF16")),
                         ("mtp", ("auto", "base", "Q6_K", "Q8_0", "BF16")),
                         ("embed", ("auto", "base", "Q4_K", "Q6_K", "Q8_0", "BF16"))):
        v = str(comp.get(k) or "auto")
        v = v if v in ("auto", "base") else v.upper()
        if v not in allowed_v:
            raise ValueError(f"components.{k} must be one of {', '.join(allowed_v)}")
        comp[k] = v
    goals = [g for g in (body.get("goals") or ["faster", "better", "max"]) if g in GOALS]
    if not goals:
        raise ValueError(f"goals: any of {', '.join(GOALS)}")
    embed_cpu = any(where.get(n) is None for n in tbl["tensors"] if n.startswith("token_embd"))

    probe = _Plan(tbl, where, devs, cms, base, n_main)
    ref = _ref_metrics(tgt["model"], probe, n_main)
    if ref and ref["missing"]:
        warns.append(f"today's model is missing {ref['missing']} of the source's tensors "
                     "(different architecture or no MTP head?); the comparison is approximate")
    plans = []
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    for goal in goals:
        pl = _Plan(tbl, where, devs, cms, base, n_main)
        frozen = _apply_components(pl, comp, embed_cpu)
        fits = _shrink(pl, budgets)
        glog = []
        if not fits:
            glog.append("does not fit even with the least important tensors at 3 bits: lower "
                        "the context, the margin, or offload layers (NGL)")
        elif goal == "max":
            _greedy(pl, "bytes", frozen, budgets)
        elif goal == "better":
            cap = None
            if ref and ref["decode"].get("0"):
                cap = 1.0 / ref["decode"]["0"]
            else:
                glog.append("no reference speed: filling the budget instead")
            _greedy(pl, "bytes", frozen, budgets, time_cap=cap)
        elif goal == "faster":
            target_loss = ref["loss"] if ref else None
            if target_loss is None:
                glog.append("no reference model to match: this is the base format with its protections")
            else:
                _greedy(pl, "time", frozen, budgets, stop=lambda: pl.loss() <= target_loss)
                if pl.loss() > target_loss * 1.0005:
                    glog.append("could not reach today's quality index within the budget")
        plans.append(_finish_plan(pl, goal, tgt, tbl, ref, base, why_base, comp, stamp,
                                  src, warns + glog, notes, fits))
    return dict(plans=plans, target=_public_target(tgt), reference=_public_ref(ref),
                source=src, base=base)


def _public_target(t):
    return {k: v for k, v in t.items() if k != "params"}


def _public_ref(ref):
    if not ref:
        return None
    return {k: v for k, v in ref.items() if k != "loss"}


def _finish_plan(pl, goal, tgt, tbl, ref, base, why_base, comp, stamp, src, warns, notes, fits):
    per, cpu = pl.dev_mib()
    loss = pl.loss()
    by_role = {}
    for n, info in pl.info.items():
        r = by_role.setdefault(info["role"], dict(types={}, mib=0.0, count=0, layers={}))
        t = pl.real_type(n, pl.types[n])
        r["types"][t] = r["types"].get(t, 0) + 1
        r["mib"] += pl.mib(n, pl.types[n]) or 0
        r["count"] += 1
        if info["il"] is not None:
            r["layers"].setdefault(t, []).append(info["il"])
    roles = []
    for role, r in sorted(by_role.items(), key=lambda kv: -kv[1]["mib"]):
        ranges = {t: _ranges(sorted(set(ls))) for t, ls in r["layers"].items()}
        roles.append(dict(role=role, label=ROLE_LABEL.get(role, role), mib=round(r["mib"], 1),
                          count=r["count"], types=r["types"], layers=ranges,
                          importance=IMPORTANCE.get(role, 1.0)))
    decode = {}
    m0 = next((m for m in pl.cm if m), None)
    if m0:
        decode["0"] = pl.decode("0")
        for d in sorted(m0["depth_a"], key=int):
            if d != "0":
                decode[d] = pl.decode(d)
    explain = [f"Base format: {why_base}. Every tensor starts there."]
    o = next((x for x in roles if x["role"] == "output"), None)
    if o:
        ot = next(iter(o["types"]))
        explain.append(
            f"Output layer: {ot} ({o['mib']:.0f} MiB). It is read in full for every token and "
            "shapes every probability, so it gets the most bits per byte of any tensor"
            + (f"; today's model keeps it at {ref['output_type']}" if ref and ref.get("output_type") else "") + ".")
    mt = next((x for x in roles if x["role"] == "mtp"), None)
    if mt:
        explain.append(f"MTP draft head: {', '.join(f'{k}×{v}' for k, v in mt['types'].items())}. "
                       "Its precision is the draft acceptance rate, which is decode speed.")
    em = next((x for x in roles if x["role"] == "embed"), None)
    if em:
        et = next(iter(em["types"]))
        explain.append(f"Token embeddings: {et} ({em['mib']:.0f} MiB). "
                       + ("llama.cpp keeps the input embeddings on the CPU, so they cost host RAM, "
                          "not VRAM or decode time." if any(pl.dev_of(n) is None for n in pl.names
                                                             if n.startswith('token_embd'))
                          else "They sit on a GPU with this placement."))
    ups = [x for x in roles if x["role"] not in ("output", "mtp", "embed", "ssm_small")
           and any(t != base for t in x["types"])]
    for x in ups[:6]:
        parts = [f"{t} layers {x['layers'].get(t, '')}".strip() for t in x["types"] if t != base]
        explain.append(f"{x['label']}: " + "; ".join(parts) +
                       f" (sensitivity {x['importance']}×"
                       + (", 1.4× more in the first and last eighth of layers" if x["role"] in EDGE_ROLES else "")
                       + ").")
    if goal == "faster":
        explain.append("Goal: stop upgrading as soon as the quality index matches today's model, "
                       "spending the fewest decode-milliseconds to get there.")
    elif goal == "better":
        explain.append("Goal: best quality whose predicted decode is no slower than today's model.")
    else:
        explain.append("Goal: fill the VRAM budget with the upgrades that buy the most quality "
                       "per MiB, whatever it costs in speed.")
    devices = []
    for i, d in enumerate(tgt["devices"]):
        devices.append(dict(name=d["name"], label=d.get("label"), weights_mib=round(per[i]),
                            budget_mib=d["budget_mib"], ref_weights_mib=d["ref_weights_mib"],
                            spare_mib=round(d["budget_mib"] - per[i]),
                            predicted_free_mib=round(d["budget_mib"] - per[i] + tgt["keep_free_mib"]),
                            basis=d["basis"]))
    rel = round(loss / ref["loss"], 3) if ref and ref["loss"] else None
    pid = f"{stamp}-{goal}"
    plan = dict(id=pid, created=_now(), goal=goal, goal_label=GOALS[goal], source=src,
                source_name=Path(src).name, instance=tgt["instance"], base=base,
                components=comp, ctx=tgt["ctx"], cur_ctx=tgt["cur_ctx"],
                keep_free_mib=tgt["keep_free_mib"], fits=fits, devices=devices,
                cpu_mib=round(cpu), file_mib=round(sum(per) + cpu),
                predicted_decode=decode, quality_index=rel,
                quality_basis="published sensitivity priors (no KL divergence measured yet); "
                              "1.00 = today's model, lower is better",
                roles=roles, explain=explain, warnings=list(dict.fromkeys(warns)), notes=notes,
                reference=_public_ref(ref), n_main=pl.n_main,
                types={n: pl.real_type(n, t) for n, t in pl.types.items()})
    plan["suggested_name"] = _suggest_name(src, goal, base)
    _atomic(_fit_dir("plans") / f"{pid}.json", plan)
    return plan


def _ranges(xs):
    out, start, prev = [], None, None
    for x in xs:
        if start is None:
            start = prev = x
        elif x == prev + 1:
            prev = x
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = x
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


def _suggest_name(src, goal, base):
    stem = re.sub(r"[-_.]?(BF16|F16|F32)$", "", Path(src).stem, flags=re.I)
    return f"{stem}-Fit-{goal}-{base}.gguf"


def plan(pid):
    if not re.fullmatch(r"[0-9A-Za-z_.-]+", pid or ""):
        raise ValueError("bad plan id")
    p = _read(_fit_dir("plans") / f"{pid}.json")
    if not p:
        raise ValueError(f"no plan {pid}")
    return p


def list_plans(limit=30):
    out = []
    for f in sorted(_fit_dir("plans").glob("*.json"), reverse=True)[:limit]:
        p = _read(f)
        if p:
            out.append({k: p.get(k) for k in ("id", "created", "goal", "goal_label", "source_name",
                                               "instance", "base", "file_mib", "predicted_decode",
                                               "quality_index", "fits", "suggested_name")})
    return out


def type_file(p, path):
    """The --tensor-type-file for a plan: every tensor not at the base format,
    as an anchored exact name (llama-quantize matches with regex search, so
    'output.weight' alone would also hit every attn_output.weight)."""
    lines = []
    for n, t in sorted(p["types"].items()):
        if t != p["base"]:
            lines.append(f"^{re.escape(n)}$={t.lower()}")
    Path(path).write_text("\n".join(lines) + "\n")
    return len(lines)


def check_plan(pid):
    """Dry-run the plan through llama-quantize: exact size, and proof that
    every override landed."""
    p = plan(pid)
    tf = _fit_dir("plans") / f"{pid}.types.txt"
    n = type_file(p, tf)
    rows, tot = _dry_run(p["source"], p["base"], tf, pure=True)
    bad = [(k, v["type"], p["types"][k]) for k, v in rows.items()
           if v["quantized"] and k in p["types"] and v["type"] != p["types"][k].upper()]
    p["dry_run"] = dict(at=_now(), overrides=n, quant_mib=round(tot, 1), mismatches=bad[:20],
                        n_mismatch=len(bad))
    _atomic(_fit_dir("plans") / f"{pid}.json", p)
    return p["dry_run"]


# ============================================================================
# jobs (one at a time)
# ============================================================================
def _job_file(jid):
    return _fit_dir("jobs") / f"{jid}.json"


def _job_log(jid):
    return _fit_dir("jobs") / f"{jid}.log"


def _jlog(job, msg):
    line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
    with open(_job_log(job["id"]), "a") as f:
        f.write(line + "\n")
    with _lock:
        job["last"] = msg[-300:]


def _jsave(job):
    _atomic(_job_file(job["id"]), {k: v for k, v in job.items() if not k.startswith("_")})


def _run_cmd(job, argv, env=None, cwd=None, label=None, progress_re=None):
    global _proc
    if _cancel.is_set():
        raise InterruptedError("cancelled")
    _jlog(job, "$ " + " ".join(str(a) for a in argv))
    with _lock:
        job["step"] = label or Path(str(argv[0])).name
    _jsave(job)
    full_env = dict(os.environ, **(env or {}))
    with open(_job_log(job["id"]), "ab") as lf:
        pr = subprocess.Popen([str(a) for a in argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env=full_env, cwd=cwd, start_new_session=True)
        with _lock:
            _proc = pr
        tail = []
        for raw in pr.stdout:
            lf.write(raw)
            line = raw.decode(errors="replace").rstrip()
            tail.append(line)
            if len(tail) > 2000:
                del tail[:1000]
            if progress_re:
                m = re.search(progress_re, line)
                if m:
                    with _lock:
                        job["progress"] = m.group(0)[-120:]
        rc = pr.wait()
        with _lock:
            _proc = None
    if _cancel.is_set():
        raise InterruptedError("cancelled")
    if rc != 0:
        raise RuntimeError(f"{Path(str(argv[0])).name} exited {rc}: " + " | ".join(tail[-4:]))
    return tail


def _start_job(kind, title, fn, **info):
    global _job
    with _lock:
        if _job and not _job.get("_done"):
            raise ValueError(f"a Fit job is already running: {_job['title']}")
        if kind in ("bench", "verify", "imatrix") and getattr(getattr(P, "benchlab", None), "active", lambda: False)():
            raise ValueError("a Bench run is active; wait for it or stop it")
        if kind in ("bench", "verify", "imatrix"):
            try:
                if (O.status().get("active") or {}).get("state") in ("running", "starting"):
                    raise ValueError("the optimizer is running; it owns the server right now")
            except AttributeError:
                pass
        jid = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{kind}"
        job = dict(id=jid, kind=kind, title=title, state="running", started=_now(),
                   finished=None, step="starting", progress=None, last=None, **info)
        _job = job
        _cancel.clear()
    _jsave(job)

    def run():
        try:
            fn(job)
            job["state"] = "done"
        except InterruptedError:
            job["state"] = "cancelled"
            _jlog(job, "cancelled")
        except Exception as e:
            job["state"] = "failed"
            job["error"] = str(e)[:2000]
            _jlog(job, f"FAILED: {e}")
        finally:
            job["finished"] = _now()
            job["step"] = None
            job["_done"] = True
            _jsave(job)
    threading.Thread(target=run, daemon=True, name=f"fit-{kind}").start()
    return {k: v for k, v in job.items() if not k.startswith("_")}


def cancel():
    with _lock:
        if not _job or _job.get("_done"):
            raise ValueError("no Fit job is running")
        _cancel.set()
        pr = _proc
    if pr and pr.poll() is None:
        try:
            os.killpg(pr.pid, signal.SIGTERM)
        except OSError:
            pass
    return dict(ok=True)


def job(jid):
    if not re.fullmatch(r"[0-9A-Za-z_.-]+", jid or ""):
        raise ValueError("bad job id")
    with _lock:
        if _job and _job["id"] == jid:
            j = {k: v for k, v in _job.items() if not k.startswith("_")}
        else:
            j = _read(_job_file(jid))
    if not j:
        raise ValueError(f"no job {jid}")
    try:
        with open(_job_log(jid), "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 12000))
            j["log"] = f.read().decode(errors="replace")
    except OSError:
        j["log"] = ""
    return j


def jobs(limit=15):
    out = []
    for f in sorted(_fit_dir("jobs").glob("*.json"), reverse=True)[:limit]:
        j = _read(f)
        if j:
            out.append({k: j.get(k) for k in ("id", "kind", "title", "state", "started",
                                               "finished", "step", "error", "output")})
    with _lock:
        live = {k: v for k, v in _job.items() if not k.startswith("_")} if _job and not _job.get("_done") else None
    return live, out


def recover_on_startup():
    """A job the panel did not finish: say so. A bench job may have left its
    instance stopped; the flag tells the UI to offer the restart."""
    for f in _fit_dir("jobs").glob("*.json"):
        j = _read(f)
        if j and j.get("state") == "running":
            j["state"] = "interrupted"
            j["finished"] = _now()
            j["error"] = "the panel restarted while this job ran"
            for part in _fit_dir("jobs").glob(f"{j['id']}*.part"):
                part.unlink(missing_ok=True)
            _atomic(f, j)
    for part in out_dir().glob("*.gguf.part") if out_dir().is_dir() else []:
        try:
            part.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# build (phase D, first half)
# ---------------------------------------------------------------------------
def build(body):
    p = plan(str(body.get("plan") or ""))
    name = str(body.get("name") or p.get("suggested_name") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,200}\.gguf", name):
        raise ValueError("name: letters, digits and ._+- only, ending in .gguf")
    out = out_dir() / name
    if out.exists() and not body.get("overwrite"):
        raise ValueError(f"{out} exists; delete it first or pick another name")
    q = quantize_bin()
    if not q:
        raise ValueError("no llama-quantize (Builds tab: install a CPU build)")
    need_gb = p["file_mib"] * MIB / 1e9 + 1
    free = _free_gb(out_dir()) - DISK_RESERVE_GB
    if free < need_gb:
        raise ValueError(f"not enough disk: the file needs {need_gb:.1f} GB and {free:.1f} GB is free "
                         f"above the {DISK_RESERVE_GB} GB reserve. Delete test quants or old "
                         "candidates first (Fit tab, Disk).")
    threads = int(body.get("threads") or max(1, (os.cpu_count() or 4) - 2))
    imatrix = str(body.get("imatrix") or "")
    if imatrix and not os.path.exists(imatrix):
        raise ValueError(f"no imatrix at {imatrix}")

    def work(job):
        out_dir().mkdir(parents=True, exist_ok=True)
        tf = _fit_dir("plans") / f"{p['id']}.types.txt"
        n = type_file(p, tf)
        _jlog(job, f"plan {p['id']}: {n} tensors off the {p['base']} base; "
                   f"predicted {p['file_mib']} MiB")
        part = Path(str(out) + ".part")
        argv = ["nice", "-n", "19", "ionice", "-c3", str(q), "--pure", "--tensor-type-file", str(tf)]
        if imatrix:
            argv += ["--imatrix", imatrix]
        argv += [p["source"], str(part), p["base"], str(threads)]
        try:
            _run_cmd(job, argv, label=f"quantizing to {name}",
                     progress_re=r"\[\s*\d+/\s*\d+\]\s+\S+")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, out)
        hdr = gguf_tensors(out)
        got = {n: t["type"] for n, t in (hdr[1] if hdr else {}).items()}
        bad = [n for n, t in p["types"].items() if got.get(n) and got[n] != t.upper()]
        side = dict(fit=1, built=_now(), plan=p["id"], goal=p["goal"], base=p["base"],
                    source=p["source"], source_rev=_source_rev(p["source"]),
                    quantizer=str(q), quantizer_build=_build_no(q), imatrix=imatrix or None,
                    predicted_decode=p["predicted_decode"], quality_index=p["quality_index"],
                    quality_basis=p["quality_basis"], instance=p["instance"],
                    components=p["components"], ctx=p["ctx"], file_bytes=out.stat().st_size,
                    mismatched_types=len(bad), kld=None)
        _atomic(Path(str(out) + ".fit.json"), side)
        job["output"] = str(out)
        _jlog(job, f"built {out} ({out.stat().st_size / 2**30:.2f} GiB); "
                   + (f"{len(bad)} tensors differ from the plan" if bad else "every tensor matches the plan"))
    return _start_job("build", f"Build {name}", work, plan=p["id"], output=str(out))


def _source_rev(src):
    man = _read(Path(src).parent / "manifest.json", {}) or {}
    return man.get("revision")


# ---------------------------------------------------------------------------
# bench (phase B, generalised to any instance's cards)
# ---------------------------------------------------------------------------
def bench(body):
    iid = str(body.get("instance") or "main")
    src = str(body.get("source") or "")
    if not os.path.exists(src):
        raise ValueError("pick a full-precision source to make the test quants from")
    types = [t.upper() for t in (body.get("types") or BENCH_TYPES)]
    for t in types:
        if t not in TABLE_TYPES:
            raise ValueError(f"unknown format {t}")
    depths = [int(d) for d in (body.get("depths") or [0, 32768])]
    reps = max(1, min(int(body.get("reps") or 2), 5))
    keep = bool(body.get("keep_files"))
    inst = P.get_instance(iid)
    with P.using_instance(inst):
        params = dict(P.load_params())
        backend = str(params.get("BACKEND") or "vulkan")
        lp = P.launch_plan(params, check_runtime=False)
        devnames = P._instance_dev_names(backend)
    if backend == "cpu" or not devnames:
        raise ValueError("this instance has no GPU to measure")
    bb = bench_bin(backend)
    if not bb:
        raise ValueError(f"the active {backend} build has no llama-bench")
    q = quantize_bin()
    tbl = size_table(src)
    stem = re.sub(r"[-_.]?(BF16|F16|F32)$", "", Path(src).stem, flags=re.I)
    env = {k: str(v) for k, v in (lp.get("env") or {}).items()}
    env["GGML_VK_ALLOW_SYSMEM_FALLBACK"] = "0"      # a format that does not fit fails loudly
    kvt = str(params.get("KV_TYPE") or "f16")
    fa = "on" if str(params.get("FLASH_ATTN") or "on") in ("on", "1", "true", "auto") else "off"

    def work(job):
        test_dir().mkdir(parents=True, exist_ok=True)
        made = []
        todo = list(types)
        was_running = False
        try:
            while todo:
                # as many formats as the disk holds, then measure them
                chunk, room = [], _free_gb(test_dir()) - DISK_RESERVE_GB
                for t in todo:
                    f = test_dir() / f"{stem}-pure-{t}.gguf"
                    need = 0 if f.exists() else sum(tbl["sizes"][t].values()) * MIB / 1e9 + 0.5
                    if need <= room or not chunk:
                        chunk.append(t)
                        room -= need
                todo = [t for t in todo if t not in chunk]
                for t in chunk:
                    f = test_dir() / f"{stem}-pure-{t}.gguf"
                    if f.exists():
                        continue
                    if _free_gb(test_dir()) - DISK_RESERVE_GB < sum(tbl["sizes"][t].values()) * MIB / 1e9:
                        raise RuntimeError("out of disk for the test quants")
                    part = Path(str(f) + ".part")
                    _run_cmd(job, ["nice", "-n", "19", "ionice", "-c3", str(q), "--pure", src,
                                   str(part), t, str(max(1, (os.cpu_count() or 4) - 2))],
                             label=f"quantizing {t} test file (CPU; the server keeps serving)",
                             progress_re=r"\[\s*\d+/\s*\d+\]")
                    os.replace(part, f)
                    made.append(f)
                with P.using_instance(inst):
                    if P.server_pid():
                        was_running = True
                        _jlog(job, f"stopping {iid} to free the card")
                        ok, msg = P.stop_server()
                        if not ok and P.server_pid():
                            raise RuntimeError(f"could not stop {iid}: {msg}")
                        time.sleep(4)
                for dev_name, rec in devnames:
                    out_files = []
                    for t in chunk:
                        f = test_dir() / f"{stem}-pure-{t}.gguf"
                        res = _fit_dir("jobs") / f"{job['id']}-{rec.get('pci', dev_name).replace(':', '_')}-{t}.jsonl"
                        tail = _run_cmd(job, [str(bb), "-m", str(f), "-dev", dev_name, "-ngl", "99",
                                              "-fa", fa, "-ctk", kvt, "-ctv", kvt,
                                              "-b", str(params.get("BATCH") or 2048),
                                              "-ub", str(params.get("UBATCH") or 512),
                                              "-p", "512", "-n", "128", "-d", ",".join(map(str, depths)),
                                              "-r", str(reps), "-o", "jsonl"],
                                        env=env, label=f"measuring {t} on {dev_name}")
                        res.write_text("\n".join(l for l in tail if l.startswith("{")) + "\n")
                        out_files.append(res)
                    import_bench(out_files, rec["pci"], backend, source=src, by=f"job {job['id']}")
                if was_running:
                    _restart(job, inst, iid)
                    was_running = False
                if not keep:
                    for f in list(made):
                        if any(f.name.endswith(f"-pure-{t}.gguf") for t in chunk):
                            f.unlink(missing_ok=True)
                            made.remove(f)
        finally:
            if was_running:
                _restart(job, inst, iid)
    return _start_job("bench", f"Measure {len(types)} formats on {iid}", work, instance=iid)


def _restart(job, inst, iid):
    with P.using_instance(inst):
        if P.server_pid():
            return
        _jlog(job, f"starting {iid} again")
        ok, msg = P.start_server()
        _jlog(job, f"start: {msg}")
        t0 = time.time()
        while time.time() - t0 < 600:
            if P.server_pid():
                try:
                    with urllib.request.urlopen(P.api_base() + "/health", timeout=3) as r:
                        if r.status == 200:
                            _jlog(job, f"{iid} healthy after {int(time.time() - t0)} s")
                            return
                except Exception:
                    pass
            time.sleep(3)
        _jlog(job, f"!! {iid} did not report healthy within 10 min - check its log")


# ---------------------------------------------------------------------------
# phase E: projector and other engines
# ---------------------------------------------------------------------------
def convert(body):
    kind = str(body.get("kind") or "")
    src = str(body.get("input") or "")
    if not os.path.exists(src):
        raise ValueError(f"no such file {src}")
    ty = str(body.get("type") or "q8_0")
    name = str(body.get("name") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,200}\.gguf", name):
        raise ValueError("name: letters, digits and ._+- only, ending in .gguf")
    out = out_dir() / name
    if out.exists():
        raise ValueError(f"{out} exists")
    if not re.fullmatch(r"[A-Za-z0-9_]{2,10}", ty):
        raise ValueError("bad type")
    rules = str(body.get("rules") or "").strip()
    if kind == "mmproj":
        q = quantize_bin()
        if not q:
            raise ValueError("no llama-quantize")
        argv = ["nice", "-n", "19", str(q), src, None, ty.upper()]
    elif kind == "sd.cpp":
        eng = E.get("sd.cpp")
        bd = eng.active("cpu") or eng.active("vulkan")
        if not bd or not (Path(bd) / "sd-cli").is_file():
            raise ValueError("no sd.cpp build with sd-cli (Builds tab)")
        argv = ["nice", "-n", "19", str(Path(bd) / "sd-cli"), "-M", "convert", "-m", src,
                "-o", None, "--type", ty]
        if rules:
            if not re.fullmatch(r"[\w\\.^$*+?()\[\]|=,-]{1,400}", rules):
                raise ValueError("rules: pattern=type pairs, comma separated")
            argv += ["--tensor-type-rules", rules]
    elif kind == "audio.cpp":
        eng = E.get("audio.cpp")
        bd = eng.active("cpu") or eng.active("vulkan")
        if not bd or not (Path(bd) / "audiocpp_gguf").is_file():
            raise ValueError("no audio.cpp build with audiocpp_gguf (Builds tab)")
        argv = ["nice", "-n", "19", str(Path(bd) / "audiocpp_gguf"), "--input", src,
                "--output", None, "--type", ty.lower()]
        for r in [x.strip() for x in rules.split(",") if x.strip()]:
            if not re.fullmatch(r"[\w.*-]{1,120}=[a-z0-9_]{2,8}", r):
                raise ValueError(f"keep rule {r!r}: prefix*=type")
            argv += ["--keep-type", r]
    else:
        raise ValueError("kind: mmproj, sd.cpp or audio.cpp")

    def work(job):
        out_dir().mkdir(parents=True, exist_ok=True)
        part = Path(str(out) + ".part.gguf") if kind != "mmproj" else Path(str(out) + ".part")
        a = [str(part) if x is None else x for x in argv]
        try:
            _run_cmd(job, a, label=f"converting to {name}")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, out)
        _atomic(Path(str(out) + ".fit.json"), dict(fit=1, kind=kind, built=_now(), source=src,
                                                    type=ty, rules=rules or None,
                                                    file_bytes=out.stat().st_size))
        job["output"] = str(out)
        _jlog(job, f"wrote {out} ({out.stat().st_size / MIB:.0f} MiB)")
    return _start_job("convert", f"Convert {Path(src).name} for {kind}", work, output=str(out))


# ---------------------------------------------------------------------------
# phase D, second half: verify through the optimizer ("models" mode)
# ---------------------------------------------------------------------------
def verify(body):
    iid = str(body.get("instance") or "main")
    models = [str(m) for m in (body.get("models") or [])]
    if not models:
        raise ValueError("pick at least one model to verify")
    for m in models:
        if not os.path.exists(m):
            raise ValueError(f"no such model {m}")
    with _lock:
        if _job and not _job.get("_done"):
            raise ValueError(f"a Fit job is running ({_job['title']}); verify when it is done")
    return O.start(iid, dict(phases=["models"], models=models,
                             budget=str(body.get("budget") or "quick"),
                             goal=str(body.get("goal") or "agentic"),
                             curve_depths=body.get("curve_depths") or [8192, 32768, 131072],
                             force=bool(body.get("force"))))


def verifications():
    """Optimizer runs in models mode, flattened per model file."""
    out = {}
    for r in O.list_runs():
        try:
            run = O.read_run(r["id"])
        except ValueError:
            continue
        if "models" not in (run.get("opts") or {}).get("phases", []):
            continue
        base = next((c for c in run.get("candidates", []) if c.get("phase") == "baseline"), None)
        for c in run.get("candidates", []):
            m = (c.get("launch") or {}).get("MODEL") or (run.get("model") if c is base else None)
            if not m:
                continue
            out.setdefault(m, []).append(dict(
                run=run["id"], state=run.get("state"), at=c.get("finished") or run.get("started"),
                status=c.get("status"), detail=c.get("detail"), cand=c["id"],
                baseline=c is base, metrics=c.get("metrics"), score=c.get("score"),
                recommended=run.get("recommended") == c["id"]))
    return out


# ---------------------------------------------------------------------------
# fitted models on disk
# ---------------------------------------------------------------------------
def fitted():
    out = []
    d = out_dir()
    ver = verifications()
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.gguf")):
        side = _read(Path(str(f) + ".fit.json"), {}) or {}
        out.append(dict(path=str(f), name=f.name, gib=round(f.stat().st_size / 2**30, 2),
                        side=side, verified=ver.get(str(f), []),
                        in_use=_in_use(f)))
    return out


def _in_use(f):
    rp = os.path.realpath(f)
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            for a in argv:
                if a and a.endswith(b".gguf") and os.path.realpath(a.decode(errors="ignore")) == rp:
                    return True
    except OSError:
        pass
    for iid in [i["id"] for i in P.list_instances()]:
        try:
            with P.using_instance(P.get_instance(iid)):
                if os.path.realpath(str(P.load_params().get("MODEL") or "")) == rp:
                    return True
        except Exception:
            continue
    return False


def delete(body):
    path = Path(str(body.get("path") or ""))
    rp = Path(os.path.realpath(path))
    allowed = [Path(os.path.realpath(out_dir())), Path(os.path.realpath(test_dir()))]
    if not any(str(rp).startswith(str(a) + os.sep) for a in allowed) or rp.suffix != ".gguf":
        raise ValueError("only files under ~/models/fit and ~/models/fit-test can be deleted here")
    if not rp.exists():
        raise ValueError("no such file")
    if _in_use(rp):
        raise ValueError("an instance uses this model (running or saved as its MODEL)")
    size = rp.stat().st_size
    rp.unlink()
    Path(str(rp) + ".fit.json").unlink(missing_ok=True)
    return dict(ok=True, freed_gb=round(size / 1e9, 1))


def disk():
    items = []
    for d, kind in ((test_dir(), "test"), (out_dir(), "fitted")):
        if d.is_dir():
            for f in sorted(d.glob("*.gguf")):
                items.append(dict(path=str(f), name=f.name, kind=kind,
                                  gb=round(f.stat().st_size / 1e9, 1)))
    return dict(free_gb=round(_free_gb(P.MODELS), 1), reserve_gb=DISK_RESERVE_GB, files=items)


def save_profile(body):
    model = str(body.get("model") or "")
    iid = str(body.get("instance") or "main")
    name = str(body.get("name") or "").strip() or ("fit-" + Path(model).stem[-40:])
    if not os.path.exists(model):
        raise ValueError("no such model")
    with P.using_instance(P.get_instance(iid)):
        vals = dict(P.load_params(), MODEL=model)
        res = P.save_profile(name, vals)
    return dict(ok=True, profile=(res or {}).get("name", name),
                note="saved as a model profile: loading it fills the Parameters form; "
                     "nothing changes until you press Save there")


# ---------------------------------------------------------------------------
# importance matrix: which weights carry signal on real text
# ---------------------------------------------------------------------------
CALIB_GLOBS = [                      # (glob, share): code-heavy, like an agent's context
    ("/usr/lib/python3*/**/*.py", 0.55),
    ("/usr/lib/python3/dist-packages/**/*.py", 0.2),
    ("/usr/share/doc/**/*.md", 0.1),
    ("/usr/share/common-licenses/*", 0.05),
    ("/usr/share/doc/**/README*", 0.1),
]


def imatrix_dir():
    return _fit_dir("imatrix")


def imatrices():
    out = []
    for f in sorted(imatrix_dir().glob("*.gguf")):
        side = _read(Path(str(f) + ".json"), {}) or {}
        out.append(dict(path=str(f), name=f.name, mib=round(f.stat().st_size / MIB, 1), **side))
    return out


def _calibration(job, tokens, extra):
    """A deterministic text mix of about `tokens` tokens (3.4 chars/token for code)."""
    import glob as _g
    want = int(tokens * 3.4)
    f = _fit_dir("cache") / f"calib-{tokens}-{hashlib.sha1('|'.join(extra).encode()).hexdigest()[:8]}.txt"
    if f.exists() and f.stat().st_size >= want * 0.9:
        return f
    parts, have = [], 0
    for path in extra:                          # the operator's own text comes first, whole
        try:
            t = Path(path).read_text(errors="ignore")
        except OSError:
            continue
        parts.append(t)
        have += len(t)
    for pat, share in CALIB_GLOBS:
        quota, got = int(want * share), 0
        files = sorted(_g.glob(pat, recursive=True))
        step = max(1, len(files) // 4000)
        for fn in files[::step]:
            if got >= quota:
                break
            try:
                if os.path.getsize(fn) > 400_000:
                    continue
                t = open(fn, errors="ignore").read()
            except OSError:
                continue
            chunk = f"\n\n# ===== {os.path.basename(fn)} =====\n" + t[:60_000]
            parts.append(chunk)
            got += len(chunk)
        have += got
    f.write_text("".join(parts)[:max(want, have)])
    _jlog(job, f"calibration text: {f.stat().st_size / 1e6:.1f} MB, about "
               f"{int(f.stat().st_size / 3.4 / 1000)}k tokens")
    return f


def imatrix(body):
    """Build (if needed) a Q8_0 reference from the source, stop the instance,
    run llama-imatrix over calibration text with -fit, restart the instance."""
    iid = str(body.get("instance") or "main")
    src = str(body.get("source") or "")
    if not os.path.exists(src):
        raise ValueError("pick the full-precision source")
    ref_kind = str(body.get("reference") or "q8_0")
    tokens = max(50_000, min(int(body.get("tokens") or 300_000), 2_000_000))
    extra = [str(x) for x in (body.get("calibration_files") or []) if str(x).strip()]
    for x in extra:
        if not os.path.isfile(x):
            raise ValueError(f"no such calibration file {x}")
    inst = P.get_instance(iid)
    with P.using_instance(inst):
        params = dict(P.load_params())
        backend = str(params.get("BACKEND") or "vulkan")
        lp = P.launch_plan(params, check_runtime=False)
        devnames = P._instance_dev_names(backend)
    if ref_kind == "current":
        ref = str(params.get("MODEL") or "")
        if not os.path.exists(ref):
            raise ValueError("the instance's MODEL does not exist")
    elif ref_kind == "q8_0":
        stem = re.sub(r"[-_.]?(BF16|F16|F32)$", "", Path(src).stem, flags=re.I)
        ref = str(test_dir() / f"{stem}-pure-Q8_0.gguf")
        if not os.path.exists(ref):
            need = sum(size_table(src)["sizes"]["Q8_0"].values()) * MIB / 1e9 + 1
            if _free_gb(test_dir()) - DISK_RESERVE_GB < need:
                raise ValueError(f"the Q8_0 reference needs {need:.0f} GB; "
                                 f"{_free_gb(test_dir()) - DISK_RESERVE_GB:.0f} GB is free above "
                                 "the reserve. Delete test quants first, or use the current model.")
    else:
        raise ValueError("reference: q8_0 or current")
    bd = P.active_build(backend)
    ib = Path(bd) / "llama-imatrix" if bd else None
    if not ib or not ib.is_file():
        raise ValueError(f"the active {backend} build has no llama-imatrix")
    q = quantize_bin()
    env = {k: str(v) for k, v in (lp.get("env") or {}).items()}
    env["GGML_VK_ALLOW_SYSMEM_FALLBACK"] = "0"
    name = re.sub(r"[-_.]?(BF16|F16|F32)$", "", Path(src).stem, flags=re.I) + \
        f"-{ref_kind}-{tokens // 1000}k.imatrix.gguf"
    out = imatrix_dir() / name

    def work(job):
        if ref_kind == "q8_0" and not os.path.exists(ref):
            test_dir().mkdir(parents=True, exist_ok=True)
            part = ref + ".part"
            try:
                _run_cmd(job, ["nice", "-n", "19", "ionice", "-c3", str(q), "--pure", src, part,
                               "Q8_0", str(max(1, (os.cpu_count() or 4) - 2))],
                         label="building the Q8_0 reference (CPU; the server keeps serving)",
                         progress_re=r"\[\s*\d+/\s*\d+\]")
            except BaseException:
                Path(part).unlink(missing_ok=True)
                raise
            os.replace(part, ref)
        calib = _calibration(job, tokens, extra)
        was = False
        try:
            with P.using_instance(inst):
                if P.server_pid():
                    was = True
                    _jlog(job, f"stopping {iid} to free the card")
                    ok, msg = P.stop_server()
                    if not ok and P.server_pid():
                        raise RuntimeError(f"could not stop {iid}: {msg}")
                    time.sleep(4)
            argv = [str(ib), "-m", ref, "-f", str(calib), "-o", str(out) + ".part.gguf",
                    "-c", "512", "-b", "512", "--process-output", "-fa", "on", "-fit", "on",
                    "-fitt", "512", "--no-ppl", "-ofreq", "20"]
            if devnames:
                argv += ["-dev", ",".join(n for n, _r in devnames)]
            t0 = time.time()
            _run_cmd(job, argv, env=env, label="collecting the importance matrix (GPU)",
                     progress_re=r"\[\d+\][^\n]{0,80}|computing over \d+ chunks[^\n]*|\d+(\.\d+)? seconds per pass[^\n]*")
            os.replace(str(out) + ".part.gguf", out)
            _atomic(Path(str(out) + ".json"), dict(built=_now(), source=src, reference=ref,
                                                   reference_kind=ref_kind, tokens=tokens,
                                                   calibration=str(calib), extra=extra,
                                                   minutes=round((time.time() - t0) / 60, 1)))
            job["output"] = str(out)
            _jlog(job, f"wrote {out} in {(time.time() - t0) / 60:.1f} min")
        finally:
            Path(str(out) + ".part.gguf").unlink(missing_ok=True)
            if was:
                _restart(job, inst, iid)
    return _start_job("imatrix", f"Importance matrix for {Path(src).name}", work,
                      instance=iid, output=str(out))


# ============================================================================
# phase F: upstream watch
# ============================================================================
_UPD = None


def _upd_file():
    return _fit_dir() / "updates.json"


def check_updates(by="operator"):
    res = dict(checked=_now(), by=by, sources=[], fitted=[])
    active_q = quantize_bin()
    for s in sources():
        row = dict(path=s["path"], repo=s["repo"], revision=s["revision"])
        if s["repo"]:
            try:
                req = urllib.request.Request(f"https://huggingface.co/api/models/{s['repo']}",
                                             headers={"User-Agent": "LexiPanel-Fit"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    meta = json.loads(r.read())
                row["upstream"] = meta.get("sha")
                row["upstream_modified"] = meta.get("lastModified")
                row["new_revision"] = bool(meta.get("sha") and s["revision"] and meta["sha"] != s["revision"])
            except Exception as e:
                row["error"] = str(e)[:200]
        res["sources"].append(row)
    for f in fitted():
        side = f["side"]
        if not side.get("plan"):
            continue
        row = dict(path=f["path"], name=f["name"], source_rev=side.get("source_rev"),
                   quantizer_build=side.get("quantizer_build"),
                   active_quantizer_build=_build_no(active_q))
        src_now = _source_rev(side.get("source") or "")
        row["source_changed"] = bool(src_now and side.get("source_rev") and src_now != side["source_rev"])
        up = next((s for s in res["sources"] if s["path"] == side.get("source")), None)
        row["upstream_newer"] = bool(up and up.get("new_revision"))
        row["newer_quantizer"] = bool(row["active_quantizer_build"] and side.get("quantizer_build")
                                      and row["active_quantizer_build"] > side["quantizer_build"])
        res["fitted"].append(row)
    _atomic(_upd_file(), res)
    return res


def updates():
    return _read(_upd_file())


def scheduler():
    """Panel thread: once a day at 05:10 UTC, check upstream. Reports only."""
    while True:
        try:
            u = updates() or {}
            today = time.strftime("%Y-%m-%d", time.gmtime())
            if time.strftime("%H:%M", time.gmtime()) >= "05:10" and \
                    not str(u.get("checked") or "").startswith(today):
                check_updates(by="daily check")
        except Exception:
            pass
        time.sleep(300)


def fetch_source(body):
    """Phase A as a job: download (pinned revision) + convert. Big: ~110 GB."""
    ph = converter()["phase_a"]
    if not ph:
        raise ValueError("~/fitquant/phaseA-source.sh is missing")
    repo = str(body.get("repo") or "")
    rev = str(body.get("revision") or "")
    name = str(body.get("name") or "")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("repo: owner/name")
    if rev and not re.fullmatch(r"[0-9a-f]{7,40}", rev):
        raise ValueError("revision: a commit hash")
    if name and not re.fullmatch(r"[\w.+-]{1,120}", name):
        raise ValueError("bad name")
    if _free_gb(src_root()) < 120 + DISK_RESERVE_GB:
        raise ValueError(f"phase A needs about 120 GB free (safetensors + BF16); "
                         f"{_free_gb(src_root()):.0f} GB is free")
    env = {"REPO": repo}
    if rev:
        env["REV"] = rev
    if name:
        env["NAME"] = name

    def work(job):
        _run_cmd(job, ["nice", "-n", "19", "ionice", "-c3", "bash", ph, "--yes"], env=env,
                 label=f"phase A: {repo}", progress_re=r"\d+%|MiB|GB")
    return _start_job("source", f"Fetch and convert {repo}", work)


# ============================================================================
# status / readiness
# ============================================================================
def readiness(srcs, cps):
    q = quantize_bin()
    conv = converter()
    free = _free_gb(P.MODELS)
    try:
        mem = int([l for l in Path("/proc/meminfo").read_text().splitlines()
                   if l.startswith("MemAvailable")][0].split()[1]) // 1024
    except Exception:
        mem = None
    items = [
        dict(id="source", ok=bool(srcs), label="A full-precision source (BF16/F16 GGUF)",
             detail=(f"{len(srcs)} found under ~/models/src" if srcs else
                     "none: run phase A (below, or ~/fitquant/phaseA-source.sh)"),
             why="Every Fit quant is cut from full precision. Requantizing someone's Q4 stacks "
                 "rounding error on rounding error."),
        dict(id="quantize", ok=bool(q), label="llama-quantize (CPU build)",
             detail=str(q) if q else "Builds tab: install a llama.cpp CPU build",
             why="Quantizing runs on the CPU at low priority, so the server keeps serving."),
        dict(id="cards", ok=bool(cps), label="Measured format speeds for your cards",
             detail=(", ".join(f"{c.get('name', c['card'])[:40]}: {len(c.get('types', {}))} formats"
                               for c in cps) if cps else "none yet: 'Measure formats'"),
             why="Formats are not equally fast on every card. The solver spends bytes only "
                 "where this card's measured speed says they are cheap."),
        dict(id="disk", ok=free - DISK_RESERVE_GB > 20, warn=free - DISK_RESERVE_GB < 60,
             label="Disk space", detail=f"{free:.0f} GB free on the models disk "
                                        f"({DISK_RESERVE_GB} GB is kept in reserve)",
             why="A 27B candidate is 14-20 GB; format tests take one file per format."),
        dict(id="ram", ok=(mem or 0) > 4000, label="Host RAM for quantizing",
             detail=f"{mem} MB available" if mem else "unknown",
             why="llama-quantize streams the source; a few GB is enough, but it shares RAM with "
                 "the running server."),
        dict(id="imatrix", ok=bool(imatrices()), warn=not imatrices(), optional=True,
             label="Importance matrix (optional)",
             detail=(", ".join(i["name"] for i in imatrices()) if imatrices() else
                     "none: builds round every weight equally; i-quants lose a little without one"),
             why="An imatrix records which weights matter on real text. Making one needs the "
                 "model running on the GPU for a while."),
        dict(id="kld", ok=False, warn=True, optional=True, label="KL divergence (tabled)",
             detail="not measured: the quality index is a prior until it is",
             why="Measures how far each candidate's token probabilities drift from the source's."),
        dict(id="converter", ok=bool(conv["convert"] and conv["python"]), optional=True,
             label="Converter for new sources (phase A)",
             detail=conv["convert"] or "set up by the first phase A run",
             why="Turns safetensors into the BF16 GGUF the solver works from."),
    ]
    return items


def status():
    try:
        _auto_import()
    except Exception:
        pass
    srcs = sources()
    cps = cards()
    live, recent = jobs()
    return dict(readiness=readiness(srcs, cps), sources=srcs, cards=cps, plans=list_plans(),
                job=live, jobs=recent, fitted=fitted(), updates=updates(), disk=disk(),
                imatrices=imatrices(),
                goals=GOALS, vendors=VENDOR_GUIDE, base_choices=BASE_CHOICES,
                bench_types=BENCH_TYPES, keep_free_default=KEEP_FREE_DEFAULT,
                instances=[dict(id=i["id"], name=i.get("name"), engine=i.get("engine"))
                           for i in P.list_instances()])
