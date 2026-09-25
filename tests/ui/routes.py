#!/usr/bin/env python3
"""Every GET route API.md lists must answer something other than HTTP 500.
  routes.py <port>"""
import re, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = ("/api/backup", "releases", "/api/cm/catalog", "/api/debug-bundle")   # network or heavy
routes = [r for r in re.findall(r"^\| GET \| `(/api/[^`]+)`", (ROOT / "API.md").read_text(), re.M)
          if not any(s in r for s in SKIP)]
bad = []
for r in routes:
    try:
        code = urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}{r}", timeout=60).status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:
        code = f"no answer ({e})"
    if code == 500 or not isinstance(code, int):
        bad.append(f"{code} {r}")
print("\n".join(f"FAIL {b}" for b in bad) + ("\n" if bad else "") + f"{len(routes) - len(bad)}/{len(routes)} GET routes answer without a 500")
sys.exit(1 if bad or len(routes) < 40 else 0)
