"""Benchmark suite for the optimizer: agentic and coding work, graded objectively.

Every task is graded by a program, never by eye or by another model:
  code   the reply's python block is run against hidden unit tests
  tool   the tool calls are parsed and checked against a schema, and for the
         repair task the proposed edit is applied to the file and executed
  json   the reply must parse as the requested JSON and carry the right facts
  needle long-context retrieval over a synthetic repository, two turns

Only the OpenAI-compatible /v1/chat/completions surface is used, so the suite
does not care which engine serves the model.

Code runs in bubblewrap (no network, no view of /home, fresh /tmp) when bwrap
is installed, and always under CPU/memory/file-size rlimits and a timeout.
"""
import json, os, random, re, shutil, subprocess, tempfile
try:
    import resource
except ImportError:
    resource = None                                          # Windows has no resource module

SANDBOX_TIMEOUT = 25
_BWRAP = shutil.which("bwrap")


# ============================================================================
# sandbox
# ============================================================================
def _limits():
    if resource is None:
        return
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
    os.setsid()


def run_python(files, entry="run_tests.py"):
    """Run entry with files {name: text} in an isolated dir. -> (ok, detail)."""
    d = tempfile.mkdtemp(prefix="LexiPanel-opt-")
    try:
        for name, text in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        cmd = ["python3", "-E", "-s", "-X", "utf8", entry]
        if _BWRAP:
            cmd = [_BWRAP, "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
                   "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
                   "--ro-bind", "/etc/alternatives", "/etc/alternatives",
                   "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                   "--bind", d, "/work", "--chdir", "/work",
                   "--unshare-all", "--die-with-parent"] + cmd
        try:
            r = subprocess.run(cmd, cwd=d, capture_output=True, text=True,
                               timeout=SANDBOX_TIMEOUT, preexec_fn=_limits,
                               env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp"})
        except subprocess.TimeoutExpired:
            return False, f"timed out after {SANDBOX_TIMEOUT}s"
        out = (r.stdout + r.stderr).strip()
        ok = r.returncode == 0 and "ALL TESTS PASSED" in r.stdout
        return ok, ("passed" if ok else out[-1200:] or f"exit {r.returncode}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def extract_code(text, want=None):
    """Last fenced python block, preferring one that defines `want`."""
    blocks = re.findall(r"```(?:python3?|py)?[ \t]*\r?\n(.*?)```", text or "", re.S)
    if not blocks:
        return None
    if want:
        for b in reversed(blocks):
            if re.search(rf"^\s*(def|class)\s+{re.escape(want)}\b", b, re.M):
                return b
    return blocks[-1]


def _msg(resp):
    try:
        return resp["choices"][0]["message"] or {}
    except (KeyError, IndexError, TypeError):
        return {}


def _finish(resp):
    try:
        return resp["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def _tool_calls(resp):
    """-> list of (name, args dict | None, raw args)."""
    out = []
    for tc in _msg(resp).get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            args = raw
        else:
            try:
                args = json.loads(raw or "")
            except (ValueError, TypeError):
                args = None
        out.append((fn.get("name"), args if isinstance(args, dict) else None, raw))
    return out


# ============================================================================
# coding tasks
# ============================================================================
CODE_SYSTEM = ("You are a senior Python engineer working inside an automated coding agent. "
               "Reply with exactly one ```python fenced code block containing the complete, "
               "self-contained solution (standard library only). No prose outside the block.")


def _code_task(tid, title, name, spec, tests, sanity=False, weight=1.0):
    def grade(resp):
        if _finish(resp) == "length":
            return False, "truncated (hit max_tokens)"
        code = extract_code(_msg(resp).get("content"), name)
        if not code:
            return False, "no python code block in the reply"
        return run_python({"solution.py": code,
                           "run_tests.py": tests.strip() + "\nprint('ALL TESTS PASSED')\n"})
    return dict(id=tid, kind="code", title=title, weight=weight, sanity=sanity, grade=grade,
                messages=[{"role": "system", "content": CODE_SYSTEM},
                          {"role": "user", "content": spec.strip()}])


CODE_TASKS = [
    _code_task("code.duration", "Parse duration strings", "parse_duration", """
Write `parse_duration(text: str) -> int` returning a number of seconds.
The input is one or more components of the form <non-negative integer><unit>, where unit is
one of d (days), h (hours), m (minutes), s (seconds). Components may be separated by
whitespace. Examples: "45s" -> 45, "1h30m" -> 5400, "1h 2m 3s" -> 3723, "2d" -> 172800.
Raise ValueError for anything else, including an empty string, a bare number such as "10",
unknown units, or stray characters such as "1h-2m".
""", """
from solution import parse_duration
assert parse_duration("45s") == 45
assert parse_duration("90m") == 5400
assert parse_duration("1h30m") == 5400
assert parse_duration("1h 2m 3s") == 3723
assert parse_duration("2d") == 172800
assert parse_duration("1d1h1m1s") == 90061
for bad in ["", "abc", "10", "5x", "1h-2m", "m5"]:
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"no ValueError for {bad!r}")
""", sanity=True),

    _code_task("code.lru", "LRU cache", "LRUCache", """
Implement `class LRUCache` with `__init__(self, capacity: int)`, `get(self, key)` and
`put(self, key, value)`. `get` returns the value or -1 when absent. `put` inserts or updates;
when the cache is over capacity it evicts the least recently used key. Both `get` and `put`
count as a use. Both operations must be O(1). A capacity below 1 raises ValueError.
""", """
from solution import LRUCache
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1
assert c.get(3) == 3 and c.get(4) == 4
c.put(3, 30)
c.put(5, 5)
assert c.get(4) == -1 and c.get(3) == 30 and c.get(5) == 5
z = LRUCache(1); z.put("a", 1); z.put("b", 2)
assert z.get("a") == -1 and z.get("b") == 2
try:
    LRUCache(0)
except ValueError:
    pass
else:
    raise AssertionError("capacity 0 must raise ValueError")
"""),

    _code_task("code.intervals", "Merge intervals", "merge_intervals", """
Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]`. Merge all overlapping
intervals and return them sorted by start. Intervals that touch ([1, 4] and [4, 5]) merge.
Do not modify the input list or its inner lists. Return a new list of two-element lists.
""", """
from solution import merge_intervals
assert merge_intervals([]) == []
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]
assert merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]
src = [[3,4],[1,2]]
merge_intervals(src)
assert src == [[3,4],[1,2]], "input was mutated"
"""),

    _code_task("code.bugfix", "Fix a failing function", "chunked", """
This function is failing its tests. Fix it and return the corrected function.

```python
def chunked(seq, size):
    \"\"\"Split seq into lists of length `size`; the last chunk may be shorter.\"\"\"
    out = []
    for i in range(0, len(seq) - size, size):
        out.append(list(seq[i:i + size]))
    return out
```

Failing cases reported by CI:
- chunked([1, 2, 3, 4, 5], 2) returned [[1, 2], [3, 4]], expected [[1, 2], [3, 4], [5]]
- chunked([1, 2], 2) returned [], expected [[1, 2]]

It must also accept any sequence (lists, strings, ranges) and raise ValueError when size < 1.
""", """
from solution import chunked
assert chunked([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]
assert chunked([1,2], 2) == [[1,2]]
assert chunked([], 3) == []
assert chunked("abcde", 3) == [["a","b","c"],["d","e"]]
assert chunked(range(4), 1) == [[0],[1],[2],[3]]
for bad in (0, -1):
    try:
        chunked([1], bad)
    except ValueError:
        pass
    else:
        raise AssertionError("size < 1 must raise ValueError")
""", sanity=True),

    _code_task("code.toposort", "Dependency install order", "topo_order", """
Write `topo_order(deps: dict[str, list[str]]) -> list[str]`. `deps` maps a package to the
packages it depends on. Return an install order containing every package exactly once
(including packages that only appear as dependencies) where each package comes after all of
its dependencies. Raise ValueError whose message contains the word "cycle" if there is a
dependency cycle. It must handle dependency chains thousands of packages deep.
""", """
from solution import topo_order
def check(deps):
    order = topo_order(deps)
    nodes = set(deps) | {d for v in deps.values() for d in v}
    assert sorted(order) == sorted(nodes), "wrong or duplicated packages"
    pos = {n: i for i, n in enumerate(order)}
    for n, ds in deps.items():
        for d in ds:
            assert pos[d] < pos[n], f"{d} must come before {n}"
check({"app": ["web", "db"], "web": ["http"], "db": [], "http": []})
check({"a": ["b"], "b": ["c"], "c": ["d"]})
check({})
check({"x": [], "y": []})
for cyc in ({"a": ["a"]}, {"a": ["b"], "b": ["c"], "c": ["a"]}):
    try:
        topo_order(cyc)
    except ValueError as e:
        assert "cycle" in str(e).lower(), "message must mention cycle"
    else:
        raise AssertionError("cycle not detected")
check({f"n{i}": [f"n{i+1}"] for i in range(5000)})
""", sanity=True),

    _code_task("code.jsonpath", "Nested path lookup", "get_path", """
Write `get_path(obj, path: str, default=None)`. `path` is a dotted path with optional list
indexes, e.g. "a.b[2].c", "list[1][0]" or "[0].name". Dict segments are string keys; [n]
segments are non-negative list indexes. Return `default` when any segment is missing, out of
range, or applied to the wrong type. An empty path returns `obj` itself.
""", """
from solution import get_path
data = {"a": {"b": [{"c": 1}, {"c": 2}, {"c": 3, "d": [10, 20]}]}, "x": None, "list": [[1, 2], [3]]}
assert get_path(data, "a.b[2].c") == 3
assert get_path(data, "a.b[2].d[1]") == 20
assert get_path(data, "list[1][0]") == 3
assert get_path(data, "a.b[9].c", "dflt") == "dflt"
assert get_path(data, "a.zz", 0) == 0
assert get_path(data, "x") is None
assert get_path(data, "x.y", "d") == "d"
assert get_path(data, "a.b.c", "d") == "d"
assert get_path([{"k": "v"}], "[0].k") == "v"
assert get_path(data, "") is data
"""),

    _code_task("code.tokenbucket", "Rate limiter with injected clock", "TokenBucket", """
Implement `class TokenBucket` with `__init__(self, rate: float, capacity: float,
clock=time.monotonic)` and `allow(self, n: float = 1) -> bool`. The bucket starts full, refills
continuously at `rate` tokens per second up to `capacity`, and `allow` consumes n tokens and
returns True only if n tokens are available (otherwise it consumes nothing and returns False).
A request larger than capacity is always False. Read time only through `clock()`.
""", """
from solution import TokenBucket
t = [100.0]
b = TokenBucket(rate=2.0, capacity=5, clock=lambda: t[0])
assert all(b.allow() for _ in range(5))
assert not b.allow()
t[0] += 0.5
assert b.allow() and not b.allow()
t[0] += 10
assert b.allow(5) and not b.allow()
t[0] += 1.0
assert b.allow(2) and not b.allow(1)
assert not b.allow(6)
t[0] += 100
assert not b.allow(6)
assert b.allow(5)
"""),

    _code_task("code.semver", "SemVer precedence", "compare_versions", """
Write `compare_versions(a: str, b: str) -> int` returning -1, 0 or 1 by Semantic Versioning
2.0.0 precedence: compare MAJOR.MINOR.PATCH numerically; a pre-release version has lower
precedence than the release; pre-release identifiers compare dot by dot, numeric identifiers
numerically, alphanumeric ones in ASCII order, numeric lower than alphanumeric, and a shorter
set is lower when all preceding identifiers are equal. Build metadata (+...) is ignored.
""", """
from solution import compare_versions as c
assert c("1.2.10", "1.2.9") == 1
assert c("1.0.0", "1.0.0") == 0
assert c("2.0.0", "10.0.0") == -1
assert c("1.0.0-alpha", "1.0.0") == -1
assert c("1.0.0", "1.0.0-rc.1") == 1
chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
         "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
for lo, hi in zip(chain, chain[1:]):
    assert c(lo, hi) == -1, (lo, hi)
    assert c(hi, lo) == 1, (hi, lo)
assert c("1.0.0+build.5", "1.0.0+other") == 0
"""),
]


# ============================================================================
# agentic tool-use tasks
# ============================================================================
AGENT_SYSTEM = ("You are a coding agent working on a repository through tools. Act by calling "
                "tools; do not describe an action you could take with a tool. When no tool is "
                "needed, answer the user directly and briefly.")


def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required,
                       "additionalProperties": False}}}


AGENT_TOOLS = [
    _fn("read_file", "Read a file from the repository.",
        {"path": {"type": "string"}}, ["path"]),
    _fn("search_code", "Search the repository for a string or regex.",
        {"query": {"type": "string"}, "glob": {"type": "string"}}, ["query"]),
    _fn("run_tests", "Run tests. target is a test file path or pytest node id.",
        {"target": {"type": "string"}}, ["target"]),
    _fn("replace_in_file", "Replace the first exact occurrence of `old` with `new` in a file.",
        {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
        ["path", "old", "new"]),
    _fn("write_file", "Overwrite a file with new content.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
]

AUTH_PY = '''import os


def auth_header(token: str) -> str:
    """Return the Authorization header value for an API token."""
    if not token:
        raise ValueError("empty token")
    return "bearer " + token.strip()
'''


def _grade_first_action(resp):
    calls = _tool_calls(resp)
    if not calls:
        return False, "no tool call (described the action instead of taking it)"
    name, args, raw = calls[0]
    if args is None:
        return False, f"tool arguments are not valid JSON: {str(raw)[:200]}"
    if name != "run_tests":
        return False, f"first call was {name}, expected run_tests"
    if "tests/test_auth.py" not in str(args.get("target", "")):
        return False, f"run_tests target {args.get('target')!r} is not tests/test_auth.py"
    if any(n in ("write_file", "replace_in_file") for n, _, _ in calls):
        return False, "edited files before running the tests"
    return True, "run_tests(tests/test_auth.py)"


def _grade_repair(resp):
    calls = _tool_calls(resp)
    content = AUTH_PY
    edited = False
    for name, args, raw in calls:
        if name in ("write_file", "replace_in_file") and args is None:
            return False, f"edit arguments are not valid JSON: {str(raw)[:200]}"
        if name == "replace_in_file":
            if str(args.get("path", "")).lstrip("./") != "src/auth.py":
                return False, f"edited {args.get('path')!r}, expected src/auth.py"
            old = str(args.get("old", ""))
            if not old or old not in content:
                return False, "replace_in_file `old` text does not occur in the file"
            content = content.replace(old, str(args.get("new", "")), 1)
            edited = True
        elif name == "write_file":
            if str(args.get("path", "")).lstrip("./") != "src/auth.py":
                return False, f"wrote {args.get('path')!r}, expected src/auth.py"
            content = str(args.get("content", ""))
            edited = True
    if not edited:
        return False, "no edit to src/auth.py" + ("" if calls else " (no tool call at all)")
    return run_python({"auth.py": content, "run_tests.py": """
from auth import auth_header
assert auth_header("abc123") == "Bearer abc123", auth_header("abc123")
assert auth_header("  xyz ") == "Bearer xyz"
try:
    auth_header("")
except ValueError:
    pass
else:
    raise AssertionError("empty token must still raise ValueError")
print('ALL TESTS PASSED')
"""})


TICKET_TOOL = [{"type": "function", "function": {
    "name": "create_ticket", "description": "Create an issue in the tracker.",
    "parameters": {"type": "object", "additionalProperties": False,
                   "required": ["title", "priority", "labels", "estimate_hours"],
                   "properties": {
                       "title": {"type": "string"},
                       "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                       "labels": {"type": "array", "items": {"type": "string"}},
                       "estimate_hours": {"type": "number"},
                       "assignee": {"type": ["string", "null"]}}}}}]


def _grade_ticket(resp):
    calls = [c for c in _tool_calls(resp) if c[0] == "create_ticket"]
    if len(calls) != 1:
        return False, f"expected one create_ticket call, got {len(calls)}"
    _, a, raw = calls[0]
    if a is None:
        return False, f"arguments are not valid JSON: {str(raw)[:200]}"
    extra = set(a) - {"title", "priority", "labels", "estimate_hours", "assignee"}
    if extra:
        return False, f"unknown argument(s) {sorted(extra)}"
    if not isinstance(a.get("title"), str) or not a["title"].strip():
        return False, "title missing"
    if a.get("priority") != "high":
        return False, f"priority {a.get('priority')!r}, expected 'high'"
    labels = a.get("labels")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        return False, "labels must be a list of strings"
    if not {"backend", "auth"} <= {x.lower() for x in labels}:
        return False, f"labels {labels} missing backend/auth"
    est = a.get("estimate_hours")
    if isinstance(est, bool) or not isinstance(est, (int, float)) or est != 3:
        return False, f"estimate_hours {est!r} must be the number 3"
    if a.get("assignee") not in (None, ""):
        return False, f"assignee {a.get('assignee')!r}, should be left unassigned"
    return True, "schema-exact"


def _grade_restraint(resp):
    if _tool_calls(resp):
        return False, "called a tool for a question that needed none"
    text = (_msg(resp).get("content") or "").strip()
    if "conflict" not in text.lower():
        return False, f"answer does not say 'conflict': {text[:160]!r}"
    if len(text) > 400:
        return False, f"answer is {len(text)} chars, asked for one sentence"
    return True, "answered directly"


DIFF = """diff --git a/requirements.txt b/requirements.txt
--- a/requirements.txt
+++ b/requirements.txt
@@ -3,2 +3,3 @@ fastapi==0.115.0
 pydantic==2.9.2
+requests==2.32.3
 uvicorn==0.30.6
diff --git a/src/client.py b/src/client.py
--- a/src/client.py
+++ b/src/client.py
@@ -12,7 +12,8 @@ class Client:
     def fetch(self, url):
-        return urlopen(url, timeout=5).read()
+        import requests
+        return requests.get(url, timeout=30).content
"""


def _grade_json(resp):
    text = (_msg(resp).get("content") or "").strip()
    try:
        obj = json.loads(text)
    except ValueError:
        return False, f"reply is not bare JSON: {text[:160]!r}"
    if not isinstance(obj, dict):
        return False, "top level is not an object"
    if set(obj) != {"files_changed", "adds_dependency", "risk"}:
        return False, f"keys {sorted(obj)} != files_changed, adds_dependency, risk"
    if sorted(obj["files_changed"] or []) != ["requirements.txt", "src/client.py"]:
        return False, f"files_changed {obj['files_changed']}"
    if obj["adds_dependency"] is not True:
        return False, "adds_dependency must be true"
    if obj["risk"] not in ("low", "medium", "high"):
        return False, f"risk {obj['risk']!r} not in enum"
    return True, "valid"


TOOL_TASKS = [
    dict(id="agent.first-action", kind="tool", title="Take the first step with a tool",
         weight=1.0, sanity=True, tools=AGENT_TOOLS, grade=_grade_first_action,
         messages=[{"role": "system", "content": AGENT_SYSTEM},
                   {"role": "user", "content": "CI reports failures in tests/test_auth.py. "
                                               "Run just that test file first."}]),
    dict(id="agent.repair", kind="tool", title="Repair a bug from tool results",
         weight=1.5, sanity=False, tools=AGENT_TOOLS, grade=_grade_repair,
         messages=[{"role": "system", "content": AGENT_SYSTEM},
                   {"role": "user", "content": "tests/test_auth.py is failing. Fix it."},
                   {"role": "assistant", "content": "", "tool_calls": [{
                       "id": "call_1", "type": "function", "function": {
                           "name": "run_tests", "arguments": '{"target": "tests/test_auth.py"}'}}]},
                   {"role": "tool", "tool_call_id": "call_1", "content":
                       "FAILED tests/test_auth.py::test_header - AssertionError: "
                       "assert 'bearer abc123' == 'Bearer abc123'\n1 failed, 6 passed in 0.21s"},
                   {"role": "assistant", "content": "", "tool_calls": [{
                       "id": "call_2", "type": "function", "function": {
                           "name": "read_file", "arguments": '{"path": "src/auth.py"}'}}]},
                   {"role": "tool", "tool_call_id": "call_2", "content": AUTH_PY}]),
    dict(id="agent.schema", kind="tool", title="Schema-exact tool arguments",
         weight=1.0, sanity=True, tools=TICKET_TOOL, grade=_grade_ticket,
         messages=[{"role": "system", "content": "You manage the issue tracker through tools."},
                   {"role": "user", "content": "File a ticket: the login endpoint returns 500 when "
                    "the password contains unicode. It's high priority, tag it backend and auth, "
                    "should take about three hours, and leave it unassigned."}]),
    dict(id="agent.restraint", kind="tool", title="No tool when none is needed",
         weight=0.5, sanity=False, tools=AGENT_TOOLS, grade=_grade_restraint,
         messages=[{"role": "system", "content": AGENT_SYSTEM},
                   {"role": "user", "content": "Quick question, no need to touch the repo: what "
                                               "does HTTP status 409 mean? One sentence."}]),
    dict(id="agent.json", kind="json", title="Bare JSON review of a diff",
         weight=1.0, sanity=False, grade=_grade_json,
         messages=[{"role": "system", "content": "You are a code review bot whose output is parsed "
                    "by a program."},
                   {"role": "user", "content": "Review this diff. Respond with only a JSON object "
                    "- no prose, no code fences - with exactly these keys: files_changed (array of "
                    "paths), adds_dependency (boolean), risk (\"low\", \"medium\" or \"high\").\n\n"
                    + DIFF}]),
]


# ============================================================================
# long context: synthetic repository with two needles
# ============================================================================
_NOUNS = ["invoice", "ledger", "shipment", "account", "session", "payout", "refund", "order",
          "customer", "catalog", "voucher", "tenant", "webhook", "report", "quota", "audit"]
_VERBS = ["load", "sync", "validate", "merge", "archive", "publish", "score", "reconcile"]


def _module(i, rnd, extra=""):
    n1, n2 = rnd.sample(_NOUNS, 2)
    v1, v2 = rnd.sample(_VERBS, 2)
    return (f"# ===== file: services/svc_{i:03d}.py =====\n"
            f'"""Service {i}: {v1}s {n1} records for the {n2} pipeline."""\n'
            f"import logging\nfrom .common import retry, emit\n\n"
            f"LOG = logging.getLogger(__name__)\n"
            f"MAX_BATCH_{i:03d} = {rnd.randint(50, 900)}\nTIMEOUT_S_{i:03d} = {rnd.randint(5, 120)}\n"
            f"{extra}\n"
            f"def {v1}_{n1}_{i:03d}(client, key):\n"
            f'    """{v1.capitalize()} {n1} rows for key."""\n'
            f"    rows = client.fetch({n1!r}, key, limit=MAX_BATCH_{i:03d})\n"
            f"    return [r for r in rows if r.get('active')]\n\n"
            f"@retry(times=3)\n"
            f"def {v2}_{n2}_{i:03d}(client, items):\n"
            f"    for item in items:\n"
            f"        emit('{n2}.{v2}', item)\n"
            f"    LOG.debug('%d items', len(items))\n"
            f"    return len(items)\n\n")


def build_repo(target_chars, seed=7):
    """Synthetic repo of ~target_chars with two needles. -> (text, facts)."""
    rnd = random.Random(seed)
    per = len(_module(0, random.Random(0))) + 20
    n = max(12, target_chars // per)
    k1, k2 = int(n * 0.35), int(n * 0.70)
    budget = rnd.randint(1000, 9999)
    fname = f"drain_{rnd.choice(_NOUNS)}_queue"
    parts = []
    for i in range(n):
        extra = ""
        if i == k1:
            extra = f"RETRY_BUDGET_MS = {budget}\n"
        if i == k2:
            extra = (f"\ndef {fname}(client):\n"
                     f"    pending = client.pending()\n"
                     f"    _flush_outbox(client)\n"
                     f"    return pending\n")
        parts.append(_module(i, rnd, extra))
    return "".join(parts), dict(module=f"svc_{k1:03d}", value=budget,
                                fn_module=f"svc_{k2:03d}", fn=fname)


NEEDLE_SYSTEM = "You answer questions about the repository below exactly and briefly."


def needle_messages(repo, facts, nonce, turn=1, answer1=None):
    q1 = ("In which module is RETRY_BUDGET_MS defined, and what is its value? Reply exactly in the "
          "form module:value, for example svc_001:1234.")
    q2 = (f"Which function in {facts['fn_module']}.py calls _flush_outbox? "
          "Reply with just the function name.")
    msgs = [{"role": "system", "content": f"[run {nonce}] {NEEDLE_SYSTEM}"},
            {"role": "user", "content": repo + "\n\nQuestion: " + q1}]
    if turn == 2:
        msgs += [{"role": "assistant", "content": answer1 or f"{facts['module']}:{facts['value']}"},
                 {"role": "user", "content": q2}]
    return msgs


def grade_needle(resp, facts, turn):
    text = re.sub(r"<think>.*?</think>", "", _msg(resp).get("content") or "", flags=re.S).strip()
    if _finish(resp) == "length" and not text:
        return False, "truncated before answering"
    if turn == 1:
        want = f"{facts['module']}:{facts['value']}"
        ok = want in text.replace(" ", "").replace(".py", "")
    else:
        want = facts["fn"]
        ok = want in text
    return ok, ("found" if ok else f"expected {want}, got {text[:120]!r}")


# ============================================================================
# selection
# ============================================================================
ALL_TASKS = CODE_TASKS + TOOL_TASKS
QUICK_IDS = {"code.duration", "code.lru", "code.bugfix", "code.toposort",
             "agent.first-action", "agent.repair", "agent.schema", "agent.restraint", "agent.json"}


def tasks_for(which):
    if which == "sanity":
        return [t for t in ALL_TASKS if t["sanity"]]
    if which == "core":
        return [t for t in ALL_TASKS if t["id"] in QUICK_IDS]
    return list(ALL_TASKS)


def describe():
    return [dict(id=t["id"], kind=t["kind"], title=t["title"], weight=t["weight"],
                 sanity=t["sanity"], quick=t["id"] in QUICK_IDS) for t in ALL_TASKS] + [
        dict(id="context.needle", kind="needle", title="Long-context retrieval, two turns",
             weight=1.0, sanity=False, quick=True)]


SPEED_PROMPT = [
    {"role": "system", "content": CODE_SYSTEM},
    {"role": "user", "content": "Write a complete Python module implementing an asyncio job "
     "scheduler with priorities, retries with exponential backoff and jitter, cancellation, "
     "per-job timeouts, and structured logging. Include type hints and docstrings."}]
