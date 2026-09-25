#!/usr/bin/env python3
"""Syntax-check every inline <script> of static/index.html, and gg/gg.js, with node --check.
One stray character in index.html's scripts stops every tab but Status (what broke a development build before 1.0.0)."""
import re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
html = (ROOT / "static/index.html").read_text()
blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
bad = 0
with tempfile.TemporaryDirectory() as d:
    files = []
    for i, b in enumerate(blocks):
        f = Path(d) / f"index-script-{i}.js"
        f.write_text(b)
        files.append((f"static/index.html <script> #{i + 1} (line {html[:html.find(b)].count(chr(10)) + 1})", f))
    files += [(str(p.relative_to(ROOT)), p) for p in sorted((ROOT / "gg").glob("*.js"))]
    for name, f in files:
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        print(("ok   " if r.returncode == 0 else "FAIL ") + name)
        if r.returncode:
            bad += 1
            print("     " + r.stderr.strip().replace("\n", "\n     ")[:1500])
print(f"{len(files) - bad}/{len(files)} scripts parse")
sys.exit(1 if bad or not blocks else 0)
