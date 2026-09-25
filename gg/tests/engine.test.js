// Engine tests for Graph Gauntlet: pure logic, no DOM.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const src = fs.readFileSync(__dirname + '/../gg.js', 'utf8');
const ctx = { window: {}, document: { addEventListener() {} }, Math, console };
vm.createContext(ctx); vm.runInContext(src, ctx);
const E = ctx.window.GG._engine, T = E.T;
let pass = 0; const ok = (c, m) => { assert.ok(c, m); pass++; };

function run(w, secs, bot) { const n = Math.round(secs / T.DT); for (let i = 0; i < n && !w.over; i++) { if (bot) bot(w); E.step(w, T.DT); } return w; }

// 1. seeding: the chart's shape becomes the first stretch, with bounded slopes
{
  const series = [10, 40, 12, 45, 50, 5, 30, 31, 33, 60];
  const w = E.makeWorld(1, series);
  ok(w.seeded, 'series seeds the track');
  const steep = w.ground.filter(g => g.x1 < 40).map(g => Math.abs((g.y2 - g.y1) / (g.x2 - g.x1)));
  ok(Math.max(...steep) <= 0.56, 'seeded slopes stay runnable: ' + Math.max(...steep).toFixed(2));
  const w2 = E.makeWorld(1, []); ok(!w2.seeded, 'no data still gives a course');
  const w3 = E.makeWorld(1, [5, 5, 5, 5]); ok(w3.seeded && w3.ground.every(g => isFinite(g.y1 + g.y2)), 'flat data is fine');
}
// 2. running: moves right, speeds up, stays on the ground
{
  const w = E.makeWorld(2, []);
  run(w, 3);
  ok(!w.over && w.dist > 17 && w.dist < 20, 'runs ~18 m in 3 s: ' + w.dist.toFixed(1));
  ok(w.r.vx > T.SPEED0, 'speed grows with distance');
  ok(w.r.on, 'on the ground');
}
// 3. a pit with no line: falls and dies with the right reason
{
  let died = null;
  for (let seed = 3; seed < 40 && !died; seed++) {
    const w = E.makeWorld(seed, []); run(w, 60);
    if (w.over && /fell/.test(w.over.why)) died = w;
  }
  ok(died, 'an unbridged pit kills: ' + (died && died.over.why));
}
// 4. drawLine: consumes, clamps to max length, refuses at 0, refuses tiny lines
{
  const w = E.makeWorld(4, []);
  ok(!E.drawLine(w, 10, 0, 10.2, 0), 'a 0.2 m line is refused');
  ok(w.lines === T.LINES0, 'refused line costs nothing');
  ok(E.drawLine(w, 10, 1, 30, 1), 'long line accepted');
  const l = w.drawn[0]; ok(Math.abs(Math.hypot(l.x2 - l.x1, l.y2 - l.y1) - T.LEN0) < 1e-9, 'clamped to 4.2 m');
  ok(E.drawLine(w, 12, 1, 8, 2) && w.drawn[1].x1 < w.drawn[1].x2, 'right-to-left line is normalised');
  w.lines = 0; ok(!E.drawLine(w, 1, 1, 3, 1) && w.lines === 0, 'no lines left -> refused');
}
// 5. a ramp lifts the runner and launches it upward
{
  const w = E.makeWorld(5, []);
  const gy = E.yAt(w.ground[0], 4);
  w.ground = [{ x1: 0, y1: gy, x2: 200, y2: gy, kind: 'ground' }]; w.spikes = []; w.archers = []; w.pickups = [];
  w.genX = 1e9; w.r.on = w.ground[0];
  E.drawLine(w, 4, gy - 0.1, 7, gy + 2.2);
  let maxY = -9, launched = false;
  for (let i = 0; i < 240; i++) { E.step(w, T.DT); maxY = Math.max(maxY, w.r.y); if (w.events.includes('launch')) launched = true; }
  ok(maxY > gy + 2.2, 'ramp + launch clears the ramp top: ' + (maxY - gy).toFixed(2) + ' m');
  ok(launched && !w.over && w.r.on, 'launched and landed again');
}
// 6. a drawn line blocks an arrow; without it the arrow kills
{
  const mk = () => { const w = E.makeWorld(6, []); w.ground = [{ x1: -50, y1: 0, x2: 500, y2: 0, kind: 'ground' }];
    w.spikes = []; w.pickups = []; w.genX = 1e9; w.r.y = 0; w.r.on = w.ground[0];
    w.archers = [{ x: 22, y: 0, next: 0, dead: false, fell: 0, quiver: 2, aim: 0 }]; w.rnd = () => 0.5; return w; };  // rnd 0.5 = no aim error
  const a = mk(); run(a, 4);
  ok(a.over && /arrow/.test(a.over.why), 'unshielded runner is shot: ' + (a.over && a.over.why));
  const b = mk(); b.lines = 6;
  // a tall shield in front of the runner, moving with it (bot redraws when needed)
  // a roof: a flat line overhead that the runner passes under and the falling arrow hits
  run(b, 4, w => { const inc = w.arrows.find(x => !x.dead && x.x > w.r.x && x.x - w.r.x < 12);
    if (inc && !w.drawn.some(l => l.x2 > w.r.x + 1 && l.y1 > w.r.y + 1.8)) E.drawLine(w, w.r.x + 1, w.r.y + 2.2, w.r.x + 5.2, w.r.y + 2.2); });
  ok(!(b.over && /arrow/.test(b.over.why)), 'a line in the way blocks the arrow');
}
// 7b. lines come back once they are far behind
{
  const w = E.makeWorld(8, []); w.ground=[{x1:-50,y1:0,x2:900,y2:0,kind:'ground'}]; w.spikes=[]; w.archers=[]; w.pickups=[]; w.genX=1e9; w.r.y=0; w.r.on=w.ground[0];
  w.lines = w.owned = 2; E.drawLine(w, 3, 3, 6, 3); E.drawLine(w, 3, 5, 6, 5);
  ok(w.lines === 0, 'both lines out');
  run(w, 5);
  ok(w.lines === 2 && w.drawn.length === 0, 'both back after they fall 20 m behind');
}
// 7. pickups add lines and length
{
  const w = E.makeWorld(7, []); const g = w.ground[0];
  w.pickups = [{ x: 6, y: E.yAt(g, 6) + 1.0, kind: 'line', got: false }, { x: 7, y: E.yAt(g, 7) + 1.0, kind: 'len', got: false }];
  run(w, 1.2);
  ok(w.lines === T.LINES0 + 1 && w.owned === T.LINES0 + 1 && Math.abs(w.maxLen - (T.LEN0 + T.LEN_STEP)) < 1e-9, '+LINE and +LENGTH apply');
}
// 8. playability: a bot that plays like a careful beginner
{
  function farSide(w, x0, y0) {           // first solid ground after a hazard, else null
    for (let x = x0 + 0.5; x < x0 + 9; x += 0.25) {
      const f = E.floorAt(w, x, y0, 2.5, 2.5, null);
      if (f && f.kind === 'ground' && !w.spikes.some(s => x > s.x1 - 0.3 && x < s.x2 + 0.3)) return { x, y: E.yAt(f, x) };
    }
    return null;
  }
  function bot(w) {
    const r = w.r;
    // shield: a steep line (runner passes through it) between runner and an incoming arrow
    const inc = w.arrows.find(a => !a.dead && a.x > r.x && a.x - r.x < 12);
    if (inc && w.lines > 0 && !w.drawn.some(l => l.x2 > r.x + 1 && Math.min(l.y1, l.y2) > r.y + 1.8))
      E.drawLine(w, r.x + 1, r.y + 2.2, r.x + 1 + Math.min(4.2, w.maxLen), r.y + 2.2);   // a roof
    if (!r.on) return;
    const look = r.x + r.vx * 0.3;
    const gap = !E.floorAt(w, look, r.y, 1.2, 0.6, null);
    const spike = w.spikes.find(s => s.x2 > r.x && s.x1 < look + 0.6);
    if (!(gap || spike) || w.drawn.some(l => l.x2 > r.x && l.x1 < look + 1 && Math.abs(l.x2 - l.x1) > 1)) return;  // already built
    const far = farSide(w, spike ? spike.x2 : look, r.y);
    const sx = r.x + 0.25, sy = r.y - 0.05;
    if (far && Math.hypot(far.x - sx, far.y + 0.3 - sy) <= w.maxLen && !spike)
      E.drawLine(w, sx, sy, far.x + 0.3, far.y + 0.05);                 // plain bridge
    else E.drawLine(w, sx, sy, sx + w.maxLen * 0.85, sy + w.maxLen * 0.5); // ramp: jump it
  }
  const dist = [], why = {};
  for (let seed = 100; seed < 400; seed++) {
    const w = E.makeWorld(seed, []); run(w, 120, bot);
    dist.push(w.dist); const k = w.over ? w.over.why : 'alive after 120 s'; why[k] = (why[k] || 0) + 1;
  }
  dist.sort((a, b) => a - b);
  const med = dist[dist.length >> 1];
  console.log('   bot over 300 courses: p10', dist[30].toFixed(0), 'm, median', med.toFixed(0), 'm, p90', dist[270].toFixed(0), 'm', why);
  ok(med > 90 && dist[270] > 200, 'a simple bot gets past the opening; good runs go far');
  ok(dist[270] < 5000, 'and it still ends: the gauntlet gets harder');
}
console.log(`engine: ${pass} checks passed`);
