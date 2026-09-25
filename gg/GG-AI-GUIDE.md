# GG: Graph Gauntlet. A guide for an AI rebuilding or extending it

Graph Gauntlet is a small runner game built into LexiPanel's web page. Every chart card gets a
**▶ GG** button. Pressing it swaps that chart for the game, and the chart's own recent numbers
become the first 30 m of track. Read this whole file before changing the game. The rules in
**Hard constraints** exist because breaking them breaks the panel, not just the game.

## Where it lives

LexiPanel's UI is a single file, `static/index.html`: plain HTML, CSS and JavaScript, with no
build step and no libraries. The panel's Python server only serves that one file, so GG is baked
into it in four marked places:

| Place in `static/index.html` | Marker | What it does |
|---|---|---|
| End of the `<style>` block | `/* --- GG: Graph Gauntlet` | All GG styles. Colours come from the panel's CSS variables (`--bg`, `--fg`, `--line`, `--series-1`, `--fail`, `--warn`, `--accent`). |
| First line of `lineChart()` and `drawCurveChart()` | `el._ggData=` | Gives the plotted numbers to the game. The charts are otherwise unchanged. |
| Just before the first `refresh();` | `// --- GG: Graph Gauntlet buttons` … `// --- end GG hook` | `GG_CHARTS` lists the chart elements that get a button. `ggButtons()` adds one button to each card's `<h2>`. |
| Last `<script>` before `</body>` | `// === GG: Graph Gauntlet` | The game itself: one IIFE that exposes `window.GG`. |

`python3 gg/apply_gg.py static/index.html gg/gg.css gg/gg.js` inserts or replaces all four
blocks, and running it again is harmless. In this repository, `gg/gg.css` and `gg/gg.js` are the
source and `static/index.html` ships with them already baked in; edit the sources, then re-run
the apply script.

## Hard constraints: do not break these

1. **Never render inside a chart element** (`#ch_decode`, `#curvechart`, …). The panel's
   `refresh()` rewrites their `innerHTML` every 5 s. The game inserts its own `.gghost` right
   after the card's `<h2>`, and the card gets the class `ggon`. The CSS then hides every other
   child of the card while the chart keeps updating out of sight.
2. **Zero CPU when not playing.** The `requestAnimationFrame` loop must fully stop when the game
   is paused, over or closed, when the panel tab changes (`host.offsetParent===null`), and when
   the browser tab is hidden (`visibilitychange`). Nothing may keep ticking in the background.
3. **No dependencies, no network, no asset files.** No libraries, CDNs, images or audio files.
   Sound is synthesized with WebAudio, off by default, and only created after a click.
4. **One game at a time.** Opening GG on another card closes the first one.
5. **The keyboard only while the game has focus.** Key handlers sit on `.gghost` (tabindex=0),
   never on `document`, so the panel's own shortcuts (such as the Files tab's Ctrl+A/C/X/V and
   Delete) keep working.
6. **Keep the engine pure.** `makeWorld`, `step` and `drawLine` touch no DOM. The tests run
   them in plain Node.
7. **Browser storage is optional.** Only the record is stored, under
   `localStorage['lexipanel.gg.record']`, and every access is wrapped in try/catch.
8. **The page's API guard.** The panel refuses cross-site requests (`Handler._foreign`). GG
   makes no requests at all. Keep it that way.

## The game

The world is measured in metres with y pointing up. The view shows 12 m top to bottom, with the
runner 30% of the way across. Physics runs at a fixed 120 Hz step.

- **Runner:** a white stick figure that runs right by itself. Speed is 6 m/s plus 0.02 m/s per
  metre travelled, capped at 14 m/s. Gravity is 26 m/s².
- **Surfaces:** the generated ground (drawn like a chart series, with a line and a soft fill)
  and the player's lines. While running, the runner follows the highest surface within ±0.35 m
  of its feet. Running off the end of a surface launches it along that surface's slope, so
  upward ramps become jumps. Lines steeper than a slope of 1.7 (about 60°) can't be run on; the
  runner passes through them.
- **Drawing:** drag with the mouse, a finger or a pen to make one **straight** line. The preview
  shows its length out of the allowed length. Lines under 0.5 m are ignored and cost nothing.
  Longer drags are clipped to the allowed length (4.2 m at the start).
- **Lines:** you start with 3. A line comes back once it is 20 m behind the runner, so
  "Available lines" really means how many you can have out at once.
- **Pickups:** **+ LINE** (red) permanently adds one line, up to 6. **+ LENGTH** (yellow) adds
  1 m of allowed length, up to 9 m. Some float low and are collected by running through them;
  others are high and need a ramp.
- **Hazards:**
  - Pits: falling below y = −7 ends the run.
  - Spike beds: red triangles. Touching them at below 0.45 m ends the run.
  - Archers: orange stick figures. Each carries 1 to 3 arrows depending on distance, visibly
    draws its bow for 0.6 s, then lobs a slow, high arrow that leads the runner, with some aim
    error. **Any drawn line stops an arrow.** The practical defence is a flat roof overhead,
    because the runner outruns any wall drawn in front of it. Running into an archer knocks it
    over.
- **Difficulty** rises over the first 600 m: wider pits, longer spike beds, more arrows, better aim.
- **Score:** metres travelled, shown as `0141 METRES`, with the speed in m/s under it.
- **Game over:** a box shows `Your score`, `Record`, the actual cause of death, **Try again**
  and **Back to graph**. The possible causes are "fell off the graph", "ran onto the spikes"
  and "An archer's arrow found your runner".
- **Controls:** Pause, Restart, Sound, Graph (close). Keys, while the game has focus: P pauses
  or resumes, R restarts, Esc goes back to the graph.

All tunables are in the `T` object at the top of `gg.js`. Change the numbers there, not
scattered through the code.

## Adding a chart

Add `['#elementId','label']` to `GG_CHARTS`. If the chart isn't drawn by `lineChart()`, set
`el._ggData = [numbers…]` wherever it is drawn. With no data, the game simply starts on
generated track.

## Tests: run both before shipping any change

Tests are in `gg/tests/`. They need Node 20 or newer, with `npm i jsdom@26` in `gg/tests/`.

1. `node gg/tests/engine.test.js` runs 23 checks: seeding, running, pits, line rules, ramps, arrows
   and roofs, recycling, pickups. It also plays 300 generated courses with a simple bot.
   Expect a median of about 100 m and a 90th percentile over 200 m. A big change in either
   direction means the balance moved.
2. `node gg/tests/ui.test.js http://127.0.0.1:8190/` runs 21 checks against a **scratch copy** of the
   panel, never the live one. Start the copy with
   `LEXIPANEL_PANEL_DIR=<copy> PANEL_PORT=8190 python3 <copy>/panel.py`, after removing
   `gpu-power.json` and `engine-updates*.json` from the copy so it can't act on the real
   hardware. The test covers:
   - the buttons, and that the game never renders inside a chart
   - start, drawing lines, and pausing on a tab switch with the loop stopped
   - one game at a time, and Esc
   - game over, the saved record and Try again
   - surviving a panel refresh, with zero page errors

jsdom has no canvas or layout. The test records canvas calls and fakes element sizes, so how
the game **looks** still has to be checked in a real browser.
