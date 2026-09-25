"""How fast the pretend model runs under given launch settings: shared by the fake
llama-server (what the optimizer measures) and the demo's "real traffic" (what users see)."""
import random

FLAGS = {"-c": "CTX", "-ub": "UBATCH", "-b": "BATCH", "--spec-draft-n-max": "SPEC_N_MAX",
         "--spec-draft-p-min": "SPEC_P_MIN", "--cache-reuse": "CACHE_REUSE", "--threads": "THREADS",
         "-np": "PARALLEL", "--spec-type": "SPEC_TYPE", "--cache-type-k": "KV_TYPE"}
UB = {"256": 0.95, "512": 1.0, "1024": 1.07, "2048": 1.03}
SPEC_N = {"1": 0.97, "2": 1.0, "3": 0.99, "4": 0.96}


def params_of(argv):
    return {FLAGS[a]: argv[i + 1] for i, a in enumerate(argv[:-1]) if a in FLAGS}


def speed(p):
    return UB.get(str(p.get("UBATCH")), 1.0) * SPEC_N.get(str(p.get("SPEC_N_MAX")), 1.0)


def decode(depth, p, noise=0.006, rnd=random):
    return round((43.0 - depth / 3600) * speed(p) * (1 + rnd.uniform(-noise, noise)), 2)


def prefill(depth, p):
    return round(max(300.0, 1650 - depth / 90) * (1.04 if str(p.get("UBATCH")) in ("1024", "2048") else 1.0), 1)


def log_lines(depth, new, out, dec, pre, task):
    """llama-server's print_timing lines of one finished request."""
    h = f"slot print_timing: id  0 | task {task} |"
    return [f"{h} prompt eval time = {new / pre * 1000:10.2f} ms / {new:5d} tokens ({1000 / pre:8.2f} ms per token, {pre:8.2f} tokens per second)",
            f"{h}        eval time = {out / dec * 1000:10.2f} ms / {out:5d} tokens ({1000 / dec:8.2f} ms per token, {dec:8.2f} tokens per second)",
            f"{h} draft acceptance = 0.71000 (   71 accepted /   100 generated), mean len = 2.40",
            f"slot      release: id  0 | task {task} | stop processing: n_tokens = {depth}, truncated = 0"]
