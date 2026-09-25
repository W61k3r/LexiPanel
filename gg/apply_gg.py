#!/usr/bin/env python3
"""Bake Graph Gauntlet into a LexiPanel static/index.html (idempotent).

usage: apply_gg.py <index.html> <gg.css> <gg.js>
Inserts: the CSS before </style>; a one-line data hand-off in lineChart() and
drawCurveChart(); the per-card button hook before the first refresh(); the
game script before </body>. Re-running replaces the previous GG block.
"""
import re, sys
from pathlib import Path

html_p, css_p, js_p = map(Path, sys.argv[1:4])
s = html_p.read_text()
css, js = css_p.read_text().rstrip() + "\n", js_p.read_text().rstrip() + "\n"

def put(s, marker_re, new, anchor, before=True):
    """Replace an existing marked block, or insert `new` next to `anchor` (must be unique)."""
    m = re.search(marker_re, s, re.S)
    if m:
        return s[:m.start()] + new + s[m.end():]
    assert s.count(anchor) == 1, f"anchor not unique: {anchor[:50]!r}"
    i = s.index(anchor)
    return s[:i] + new + s[i:] if before else s[:i + len(anchor)] + new + s[i + len(anchor):]

s = put(s, r"/\* --- GG: Graph Gauntlet.*?(?=</style>)", css, "</style>")

HOOK = """// --- GG: Graph Gauntlet buttons, one per chart card (the game itself is the last <script>)
const GG_CHARTS=[['#ch_decode','decode rate'],['#curvechart','decode vs depth'],['#ch_vram','VRAM used'],
  ['#ch_decode2','decode rate'],['#ch_prefill','prefill rate'],['#ch_accept','draft acceptance'],
  ['#gt_chart','GPU board power']];
function ggButtons(){
  for(const [sel,label] of GG_CHARTS){
    const el=$(sel), card=el&&el.closest('.card'), h=card&&card.querySelector('h2');
    if(!h||h.querySelector('.ggbtn')) continue;
    const b=document.createElement('button'); b.type='button'; b.className='ggbtn'; b.textContent='▶ GG';
    b.title='Graph Gauntlet: race along this graph';
    b.onclick=()=>window.GG&&GG.toggle(card,{label, button:b, series:()=>el._ggData||[]});
    h.appendChild(b);
  }
}
ggButtons();
// --- end GG hook
"""
s = put(s, r"// --- GG: Graph Gauntlet buttons.*?// --- end GG hook\n", HOOK,
        "refresh(); setInterval(refresh,5000);")

# hand the plotted numbers to the game (the chart itself is unchanged)
if "el._ggData=data;" not in s:
    a = "function lineChart(el, data, o){\n"
    assert s.count(a) == 1
    s = s.replace(a, a + "  el._ggData=data;                       // GG: the track starts from this series\n")
if "el._ggData=sm.map" not in s:
    a = "function drawCurveChart(el, sm, thr){\n"
    assert s.count(a) == 1
    s = s.replace(a, a + "  el._ggData=sm.map(p=>p.decode_tps);    // GG: the track starts from this curve\n")

GAME = "<script>\n" + js + "</script>\n"
s = put(s, r"<script>\n// === GG: Graph Gauntlet.*?</script>\n", GAME, "</body></html>")

html_p.write_text(s)
print("GG applied to", html_p)
