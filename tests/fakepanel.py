"""A stand-in for the panel module, for the workload / auto-fit tests. The engine-log
patterns and depth buckets are read from panel.py itself (not copied), so the tests always
exercise the parser the panel really uses."""
import ast, contextlib, os, re, types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _panel_constants():
    tree = ast.parse((HERE.parent / "panel.py").read_text())
    want = {"TIMING_RE", "EVAL_RE", "ACCEPT_RE", "RELEASE_RE", "CURVE_BUCKETS"}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in want:
            name, v = node.targets[0].id, node.value
            if isinstance(v, ast.Call):                       # re.compile("...")
                out[name] = re.compile(ast.literal_eval(v.args[0]))
            else:
                out[name] = ast.literal_eval(v)
    assert set(out) == want, f"panel.py constants not found: {want - set(out)}"
    return out


def make(root, params=None, instances=("main",)):
    """A fake panel rooted at `root` (a Path). Each instance's engine log is
    root/logs/<id>.log; tweak fp.live_argv / fp.params / fp.baseline as a test needs."""
    root = Path(root)
    (root / "panel").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    fp = types.SimpleNamespace(**_panel_constants())
    fp.PANEL = root / "panel"
    fp.params = {i: dict(dict(CTX="131072", PARALLEL="1", SPEC_TYPE="draft-mtp", KV_TYPE="q8_0",
                              MODEL="/m/Qwen-27B.gguf", BACKEND="vulkan"), **(params or {})) for i in instances}
    fp.live_argv = {i: ["/opt/llama/b10985-vulkan/llama-server", "-m", "/m/Qwen-27B.gguf", "-c", "131072",
                        "--port", "8081", "--spec-type", "draft-mtp"] for i in instances}
    fp.baseline = None
    fp.archives = []
    fp.kv_per_token = 0.04                                       # MiB per context token, for estimate()
    fp.vram_total = 24000
    fp._cur = None

    class _Log:
        def __str__(self):
            return str(root / "logs" / f"{fp._cur}.log")
    fp.ENGINE_LOG = _Log()

    @contextlib.contextmanager
    def using_instance(inst):
        prev = fp._cur
        fp._cur = inst["id"] if isinstance(inst, dict) else inst
        try:
            yield
        finally:
            fp._cur = prev
    fp.using_instance = using_instance
    fp.instance_ids = lambda: list(instances)
    fp.get_instance = lambda iid: dict(id=iid, name=iid, engine="llama.cpp", legacy=iid == "main",
                                       device="0000:03:00.0", devices=["0000:03:00.0"])
    fp.load_params = lambda b=None: dict(fp.params[fp._cur])
    fp.live_cmdline_args = lambda: list(fp.live_argv.get(fp._cur) or [])
    fp._build_of = lambda binary: (re.search(r"/(b\d+)[-/]", str(binary)) or [None, None])[1]

    def _argv_get(argv, names):
        for i, a in enumerate(argv):
            if a in names and i + 1 < len(argv):
                return argv[i + 1]
        return None
    fp._argv_get = _argv_get
    fp.measured_baseline = lambda inst=None: fp.baseline

    def expected_at(base, depth):
        pts = sorted((q["depth"], q["decode_tps"]) for q in (base or {}).get("points") or [])
        if not pts or depth is None:
            return None
        if depth <= pts[0][0]:
            return pts[0][1]
        for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
            if depth <= d1:
                return round(v0 + (v1 - v0) * (depth - d0) / (d1 - d0), 1)
        return pts[-1][1]
    fp.expected_at = expected_at

    def estimate(p):
        kv = int(p.get("CTX") or 0) * fp.kv_per_token
        used = 15000 + kv
        return dict(vram=dict(kv_mib=round(kv), headroom_mib=round(fp.vram_total - used)))
    fp.estimate = estimate
    fp._log_files_for_curve = lambda backend=None: ([Path(str(fp.ENGINE_LOG))] + list(fp.archives), 0)
    fp.server_pid = lambda: 4242
    return fp


def req_lines(depth, new=512, out=128, dec=40.0, acc=None, task=1):
    """Engine-log lines of one completed request, in llama-server's print_timing format."""
    p = f"slot print_timing: id  0 | task {task} |"
    lines = [f"{p} prompt eval time = {new / 1.5:10.2f} ms / {new:5d} tokens (    0.67 ms per token,  1500.00 tokens per second)",
             f"{p}        eval time = {out / dec * 1000:10.2f} ms / {out:5d} tokens (   25.00 ms per token, {dec:8.2f} tokens per second)"]
    if acc is not None:
        lines.append(f"{p} draft acceptance = {acc:.5f} (   80 accepted /   100 generated), mean len = 2.10")
    lines.append(f"slot      release: id  0 | task {task} | stop processing: n_tokens = {depth}, truncated = 0")
    return lines


def write_log(fp, iid, lines, mode="a"):
    path = Path(fp.PANEL).parent / "logs" / f"{iid}.log"
    with open(path, mode) as f:
        f.write("".join(l + "\n" for l in lines))
    return path
