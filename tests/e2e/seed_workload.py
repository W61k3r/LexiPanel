#!/usr/bin/env python3
"""Seed a panel run dir with two weeks of plausible agentic traffic for instance main:
requests.jsonl, activity.jsonl, and an autofit.json with one confirmed change and one
pending proposal.   seed_workload.py <panel_dir>"""
import json, math, random, sys, time
from pathlib import Path

d = Path(sys.argv[1]) / "workload" / "main"
d.mkdir(parents=True, exist_ok=True)
rnd = random.Random(7)
now = time.time()
h_now = int(now // 3600)
FP = "fp-demo-1"
t1, t2 = now - 9 * 86400, now - 20 * 3600


def rate(lt):
    """requests per hour by local weekday/hour"""
    wd, hr = lt.tm_wday, lt.tm_hour
    if wd < 5:
        if 8 <= hr < 18:
            return 22
        if 18 <= hr < 23:
            return 6
        if hr == 23 or hr == 7:
            return 1
        return 0
    return 5 if 10 <= hr < 17 else 0


reqs, act = [], []
for h in range(h_now - 14 * 24, h_now):
    lt = time.localtime(h * 3600)
    n = rnd.randint(int(rate(lt) * 0.6), int(rate(lt) * 1.3)) if rate(lt) else 0
    busy = 0.0
    for _ in range(n):
        depth = min(60000, int(rnd.lognormvariate(math.log(17000), 0.75)))
        out = max(20, int(rnd.lognormvariate(math.log(320), 0.8)))
        pn = max(30, int(rnd.lognormvariate(math.log(900), 1.0)))
        old = h * 3600 < t1                                   # before the UBATCH change 9 days ago
        dec = max(8.0, (43.0 - depth / 3600) * (0.955 if old else 1.0) + rnd.gauss(0, 1.2))
        pre = max(300.0, 1650 - depth / 90 + rnd.gauss(0, 40))
        acc = min(0.95, max(0.3, rnd.gauss(0.71, 0.07)))
        drafted = int(out * 0.9)
        t = int(h * 3600 + rnd.uniform(0, 3599))
        reqs.append(dict(prompt_ms=round(pn / pre * 1000, 1), prompt_tokens=pn, prompt_tps=round(pre, 2),
                         eval_ms=round(out / dec * 1000, 1), eval_tokens=out, decode_tps=round(dec, 2),
                         accept=round(acc, 5), accepted=int(drafted * acc), drafted=drafted, mean_len=2.4,
                         depth=depth, t=t, fp="fp-demo-0" if old else FP, slot_ctx=131072))
        busy += pn / pre + out / dec
    act.append(dict(h=h, samples=3600.0, busy=round(min(3500, busy), 1), full=round(min(3500, busy), 1),
                    max_busy=1 if n else 0, n_slots=1))
reqs.sort(key=lambda r: r["t"])
(d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
(d / "activity.jsonl").write_text("".join(json.dumps(a) + "\n" for a in act))
summ = dict(model="Qwen3.8-27B-IQ4_XS.gguf", ctx="131072", parallel="1", kv="q8_0", spec="draft-mtp",
            ubatch="512", batch="2048", cache_reuse="256", build="b7123")
(d / "configs.json").write_text(json.dumps({
    "fp-demo-0": dict(summary=summ, first=int(now - 14 * 86400), last=int(t1)),
    FP: dict(summary=dict(summ, ubatch="1024"), first=int(t1), last=int(now))}))


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))

af = dict(settings=dict(mode="propose"), dismissed={}, rejected={}, experiments=[
    dict(id=f"x{int(t1)}", kind="tune", reason="first tune for this workload", manual=False, run_id="r-demo-1",
         started=iso(t1), started_t=int(t1), finished=iso(t1 + 3900), state="done", decision="applied",
         applied_by="you", fp_before="fp-demo-0", fp_after=FP, label="UBATCH=1024", diffs=dict(UBATCH="1024"),
         gain=0.041, quality=dict(before=0.93, after=0.93), workload_tps=dict(before=31.2, after=32.6),
         outcome="applied UBATCH=1024: decode at your depths +4.5%, quality unchanged",
         verify=dict(state="confirmed", ratio=1.038, matched=212, note="real decode at matched depths, 7 days")),
    dict(id=f"x{int(t2)}", kind="reshape", reason="measure the findings' candidates", manual=False, run_id="r-demo-2",
         started=iso(t2), started_t=int(t2), finished=iso(t2 + 2700), state="done", decision="proposal",
         proposal="pending", fp_before=FP, label="CTX=81920", diffs=dict(CTX="81920"),
         gain=0.012, quality=dict(before=0.93, after=0.93), workload_tps=dict(before=32.6, after=33.0),
         outcome="proposal: CTX=81920 (changes more than speed settings)"),
])
(d / "autofit.json").write_text(json.dumps(af, indent=1))
print(f"seeded {len(reqs)} requests, {len(act)} hours into {d}")
