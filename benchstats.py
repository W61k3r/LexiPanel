#!/usr/bin/env python3
"""
Statistics for LexiPanel's benchmark platform (benchlab.py). Pure functions, standard
library only, no I/O: every verdict the platform gives is computed here and unit-tested.

Speeds are compared as LOG RATIOS (ln b/a). A 5 % speed-up and a 5 % slow-down are then the
same distance from zero, noise is roughly symmetric, and "percent faster" is exp(mean) - 1.

The decision rule, used everywhere:
  better     the whole confidence interval is above zero AND the estimate reaches the margin
  worse      the mirror image
  small      a real difference (interval excludes zero) that is smaller than the margin
  same       the whole interval sits inside +-margin: equivalent for practical purposes
  undecided  none of the above yet: measure more, or report "no detectable difference (+-x %)"

Comparisons are SEQUENTIAL with pre-registered looks (LOOKS pairs). Each look uses
alpha / len(looks) (Bonferroni), so stopping at the first clear answer cannot inflate the
false-win rate, however often we peek. selftest() then measures the real false-win rate and
the detection power on this machine's own noise instead of trusting the theory.
"""
import math, random, statistics

LOOKS = (3, 5, 8, 12)          # pairs at which a comparison may stop
ALPHA = 0.05                   # overall false-win rate a comparison is allowed
POWER = 0.8


# ============================================================================
# distributions (no scipy on the box)
# ============================================================================
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    """Inverse normal CDF: Acklam's rational approximation plus one Halley step (~1e-12)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    lo = 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - lo:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    if df <= 0:
        raise ValueError("df must be > 0")
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p, df):
    """Inverse Student-t CDF by bisection on t_cdf (called a few times per decision)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if df > 1e6:
        return norm_ppf(p)
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return (lo + hi) / 2


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p-value (sum of outcomes no more likely than k)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    lim = pmf[k] * (1 + 1e-9)
    return min(1.0, sum(q for q in pmf if q <= lim))


# ============================================================================
# descriptive
# ============================================================================
def _num(xs):
    return [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def describe(xs):
    xs = _num(xs)
    if not xs:
        return dict(n=0)
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else None
    return dict(n=len(xs), mean=m, sd=sd, median=statistics.median(xs), min=min(xs), max=max(xs),
                cv=(sd / m) if sd is not None and m else None)


def pct(log_ratio):
    """ln(b/a) -> percent change of b over a."""
    return None if log_ratio is None else (math.exp(log_ratio) - 1.0) * 100.0


def margin_log(margin):
    """A relative margin (0.02 = 2 %) as a log-ratio distance."""
    return math.log1p(abs(margin))


# ============================================================================
# paired comparison (the lab: A and B measured back to back)
# ============================================================================
def paired(a, b, alpha=ALPHA):
    """b against a for positive metrics, from matched pairs. Returns the log-ratio mean, its
    t-interval at 1 - alpha and the same as percentages. Needs >= 2 pairs for an interval."""
    d = [math.log(y / x) for x, y in zip(a, b) if x and y and x > 0 and y > 0]
    n = len(d)
    if n == 0:
        return dict(n=0)
    m = statistics.fmean(d)
    out = dict(n=n, mean_log=m, pct=pct(m))
    if n < 2:
        return out
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    q = t_ppf(1 - alpha / 2, n - 1)
    lo, hi = m - q * se, m + q * se
    out.update(sd_log=sd, se=se, df=n - 1, alpha=alpha, lo_log=lo, hi_log=hi, pct_lo=pct(lo), pct_hi=pct(hi))
    return out


def verdict(est, lo, hi, margin, higher_better=True):
    """Classify an estimate and its interval (all log ratios, b over a). `higher_better`
    False for times (a lower b is an improvement)."""
    if est is None or lo is None or hi is None:
        return "undecided"
    if not higher_better:
        est, lo, hi = -est, -hi, -lo
    m = margin_log(margin)
    if lo > 0:
        return "better" if est >= m else "small"
    if hi < 0:
        return "worse" if est <= -m else "small"
    if lo > -m and hi < m:
        return "same"
    return "undecided"


def sequential(a, b, looks=LOOKS, alpha=ALPHA, margin=0.02, higher_better=True):
    """Where a paired comparison stands after len(pairs) pairs. It may only stop at a look;
    each look spends alpha/len(looks). State 'stop' carries the final verdict; at the last
    look an unclear result stops as 'undecided' with the precision reached."""
    n = min(len(a), len(b))
    looks = tuple(sorted(looks))
    a_look = alpha / len(looks)
    res = paired(a[:n], b[:n], alpha=a_look)
    res.update(looks=list(looks), alpha_look=a_look, margin=margin)
    if n not in looks and n < looks[-1]:
        res.update(state="continue", verdict=None, next_look=next(k for k in looks if k > n))
        return res
    v = verdict(res.get("mean_log"), res.get("lo_log"), res.get("hi_log"), margin, higher_better)
    last = n >= looks[-1]
    if v != "undecided" or last:
        res.update(state="stop", verdict=v)
        if v == "undecided" and res.get("lo_log") is not None:
            res["precision_pct"] = max(abs(pct(res["lo_log"])), abs(pct(res["hi_log"])))
    else:
        res.update(state="continue", verdict=None, next_look=next(k for k in looks if k > n))
    return res


def mde(sd_log, n, alpha=ALPHA, power=POWER):
    """Smallest relative effect a paired test with n pairs detects with `power`."""
    if not sd_log or n < 2:
        return None
    q = t_ppf(1 - alpha / 2, n - 1) + t_ppf(power, n - 1)
    return math.expm1(q * sd_log / math.sqrt(n))


def pairs_needed(sd_log, effect, alpha=ALPHA, power=POWER, max_n=500):
    """Pairs a paired test needs to detect a relative `effect` (0.03 = 3 %)."""
    if not sd_log:
        return 2
    target = math.log1p(abs(effect))
    for n in range(2, max_n + 1):
        q = t_ppf(1 - alpha / 2, n - 1) + t_ppf(power, n - 1)
        if q * sd_log / math.sqrt(n) <= target:
            return n
    return None


# ============================================================================
# self-test: this machine's own noise, resampled
# ============================================================================
def simulate(diffs, shift=0.0, looks=LOOKS, alpha=ALPHA, margin=0.02, trials=2000, seed=1):
    """Run the sequential rule `trials` times on pairs resampled from measured A/A differences
    (log ratios, centred to a true effect of zero) plus a known `shift` (0.03 = B is 3 %
    faster). Returns how often each verdict came out and the mean pairs used."""
    d = _num(diffs)
    if len(d) < 3:
        raise ValueError("need at least 3 A/A differences")
    mu = statistics.fmean(d)
    base = [x - mu for x in d]
    s = math.log1p(shift)
    rng = random.Random(seed)
    counts = dict(better=0, worse=0, same=0, small=0, undecided=0)
    used = 0
    for _ in range(trials):
        a, b = [], []
        for _k in range(max(looks)):
            a.append(1.0)
            b.append(math.exp(rng.choice(base) + s))
            r = sequential(a, b, looks, alpha, margin)
            if r["state"] == "stop":
                break
        counts[r["verdict"]] += 1
        used += len(a)
    out = {k: v / trials for k, v in counts.items()}
    out.update(shift=shift, trials=trials, mean_pairs=used / trials)
    return out


def selftest(diffs, looks=LOOKS, alpha=ALPHA, margin=0.02, shifts=(0.0, 0.01, 0.02, 0.03, 0.05, 0.08),
             trials=1000, seed=1):
    """The platform's own error rates on this machine: the false-win rate (shift 0 declared
    'better' or 'worse') and the detection curve. `smallest_reliable` is the smallest shift
    tested that is declared 'better' at least POWER of the time."""
    curve = [simulate(diffs, s, looks, alpha, margin, trials, seed) for s in shifts]
    zero = next((c for c in curve if c["shift"] == 0.0), None)
    reliable = next((c["shift"] for c in curve if c["shift"] > 0 and c["better"] >= POWER), None)
    return dict(false_win=(zero["better"] + zero["worse"]) if zero else None,
                allowed=alpha, curve=curve, smallest_reliable=reliable, n_diffs=len(_num(diffs)))


# ============================================================================
# pass/fail tasks, paired (same task + seed on both sides)
# ============================================================================
def mcnemar(a_ok, b_ok, alpha=ALPHA):
    """Paired pass/fail outcomes. Only discordant pairs carry information: b_only = B passed
    where A failed. Returns the exact two-sided p-value and the pass-rate difference
    (B - A) with an Agresti-Min style interval."""
    pairs = [(bool(x), bool(y)) for x, y in zip(a_ok, b_ok)]
    n = len(pairs)
    if n == 0:
        return dict(n=0)
    a_only = sum(1 for x, y in pairs if x and not y)
    b_only = sum(1 for x, y in pairs if y and not x)
    k = a_only + b_only
    p = binom_two_sided(min(a_only, b_only), k) if k else 1.0
    # Agresti & Min (2005): add 1/2 to each cell for the interval
    n2 = n + 2.0
    b2, c2 = b_only + 0.5, a_only + 0.5
    diff = (b2 - c2) / n2
    se = math.sqrt(max((b2 + c2) - (b2 - c2) ** 2 / n2, 0.0)) / n2
    z = norm_ppf(1 - alpha / 2)
    return dict(n=n, a_pass=sum(x for x, _ in pairs), b_pass=sum(y for _, y in pairs),
                a_only=a_only, b_only=b_only, p=p, diff=(b_only - a_only) / n,
                lo=diff - z * se, hi=diff + z * se, significant=p < alpha)


# ============================================================================
# least squares (the traffic analysis)
# ============================================================================
def _solve_inverse(A):
    """Inverse of a small symmetric matrix by Gauss-Jordan with partial pivoting; None when
    singular."""
    n = len(A)
    M = [list(map(float, row)) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col]:
                f = M[r][col]
                M[r] = [v - f * w for v, w in zip(M[r], M[col])]
    return [row[n:] for row in M]


def ols(X, y):
    """Ordinary least squares. X: rows of regressors (include the intercept column yourself)."""
    n, p = len(X), len(X[0]) if X else 0
    if n <= p:
        return None
    XtX = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    Xty = [sum(X[k][i] * y[k] for k in range(n)) for i in range(p)]
    inv = _solve_inverse(XtX)
    if inv is None:
        return None
    beta = [sum(inv[i][j] * Xty[j] for j in range(p)) for i in range(p)]
    resid = [y[k] - sum(beta[i] * X[k][i] for i in range(p)) for k in range(n)]
    df = n - p
    s2 = sum(r * r for r in resid) / df
    return dict(beta=beta, se=[math.sqrt(max(inv[i][i] * s2, 0.0)) for i in range(p)],
                resid_sd=math.sqrt(s2), df=df, n=n)


def _bucket(depth, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= depth < edges[i + 1]:
            return i
    return len(edges) - 1


def traffic_noise(rows, edges):
    """Request-to-request decode noise at matched depth (log scale), raw and after
    accounting for MTP draft acceptance. rows: dicts with decode_tps, depth, accept."""
    rows = [r for r in rows if (r.get("decode_tps") or 0) > 0 and r.get("depth") is not None]
    if len(rows) < 10:
        return dict(n=len(rows))
    groups = {}
    for r in rows:
        groups.setdefault(_bucket(r["depth"], edges), []).append(r)
    raw, cx, cy = [], [], []
    for g in groups.values():
        ys = [math.log(r["decode_tps"]) for r in g]
        my = statistics.fmean(ys)
        raw += [y - my for y in ys]
        acc = [r.get("accept") for r in g]
        if all(a is not None for a in acc):
            ma = statistics.fmean(acc)
            cx += [a - ma for a in acc]
            cy += [y - my for y in ys]
    dof = len(rows) - len(groups)
    out = dict(n=len(rows), buckets=len(groups))
    if dof < 2:
        return out
    sraw = math.sqrt(sum(e * e for e in raw) / dof)
    out["raw_sd"] = sraw
    sxx = sum(x * x for x in cx)
    if len(cx) > len(groups) + 2 and sxx > 0:
        slope = sum(x * y for x, y in zip(cx, cy)) / sxx
        res = [y - slope * x for x, y in zip(cx, cy)]
        sadj = math.sqrt(sum(e * e for e in res) / max(len(cx) - len(groups) - 1, 1))
        out.update(adjusted_sd=sadj, acceptance_slope=slope,
                   explained=1 - (sadj * sadj) / (sraw * sraw) if sraw else None,
                   pct_per_0_1_acceptance=pct(slope * 0.1))
    return out


def traffic_effect(before, after, edges, adjust=True, alpha=ALPHA):
    """Real decode after a configuration change against before it, from independent
    requests: log decode = depth-bucket level (+ acceptance slope) + change. Returns the
    change as a ratio with its interval and the requests per side needed for 3 %."""
    rows = [(0, r) for r in before] + [(1, r) for r in after]
    rows = [(g, r) for g, r in rows if (r.get("decode_tps") or 0) > 0 and r.get("depth") is not None
            and (not adjust or r.get("accept") is not None)]
    n0 = sum(1 for g, _ in rows if g == 0)
    n1 = len(rows) - n0
    if n0 < 3 or n1 < 3:
        return dict(n_before=n0, n_after=n1, verdict="not enough data")
    present = sorted({_bucket(r["depth"], edges) for _, r in rows})
    X, y = [], []
    for g, r in rows:
        b = _bucket(r["depth"], edges)
        x = [1.0] + [1.0 if b == k else 0.0 for k in present[1:]]
        if adjust:
            x.append(float(r["accept"]))
        x.append(float(g))
        X.append(x)
        y.append(math.log(r["decode_tps"]))
    fit = ols(X, y)
    if fit is None:
        return dict(n_before=n0, n_after=n1, verdict="not enough data")
    eff, se = fit["beta"][-1], fit["se"][-1]
    q = t_ppf(1 - alpha / 2, fit["df"])
    lo, hi = eff - q * se, eff + q * se
    return dict(n_before=n0, n_after=n1, adjusted=adjust, ratio=math.exp(eff), pct=pct(eff),
                pct_lo=pct(lo), pct_hi=pct(hi), lo_log=lo, hi_log=hi, mean_log=eff,
                resid_sd=fit["resid_sd"], per_side_for_3pct=requests_needed(fit["resid_sd"], 0.03, alpha))


def requests_needed(resid_sd, effect, alpha=ALPHA, power=POWER):
    """Independent requests per side to detect a relative `effect` with a two-sample test."""
    if not resid_sd:
        return None
    z = norm_ppf(1 - alpha / 2) + norm_ppf(power)
    return math.ceil(2 * (z * resid_sd / math.log1p(abs(effect))) ** 2)


# ============================================================================
# what a user waits for
# ============================================================================
def request_seconds(decode_by_depth, weights, output_tokens, prompt_tokens=0, prefill_tps=None):
    """Seconds a typical request takes: its new prompt tokens at the prefill rate plus its
    output at the workload's effective decode rate (harmonic, weighted by how often requests
    land at each depth). decode_by_depth: {depth: t/s}; weights: {depth: weight}."""
    pairs = [(w, decode_by_depth.get(d)) for d, w in weights.items() if w > 0]
    if not pairs or any(not t for _w, t in pairs):
        return None
    eff = sum(w for w, _t in pairs) / sum(w / t for w, t in pairs)
    pre = prompt_tokens / prefill_tps if prompt_tokens and prefill_tps else 0.0
    return pre + max(output_tokens, 1) / eff
