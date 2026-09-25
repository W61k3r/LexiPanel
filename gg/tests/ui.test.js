// UI smoke test: the real panel page (scratch copy on :8190) with GG baked in.
const { JSDOM, VirtualConsole } = require('jsdom');
const assert = require('assert');
const URL = process.argv[2] || 'http://127.0.0.1:8190/';
const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => /Could not load iframe: "[^"]*\/terminal\//.test(e.message) || errors.push('jsdomError: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' | ')));
vc.on('error', (...a) => errors.push('console.error: ' + a.join(' ')));
const sleep = ms => new Promise(r => setTimeout(r, ms));
let pass = 0; const ok = (c, m) => { assert.ok(c, m); pass++; console.log('   ok ', m); };

(async () => {
  const dom = await JSDOM.fromURL(URL, { runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true,
    virtualConsole: vc, beforeParse(win) {
      // jsdom has no canvas and no layout: record canvas calls, give elements a size,
      // and make offsetParent honour display:none the way a browser does
      win.HTMLCanvasElement.prototype.getContext = function () {
        const calls = this._calls = this._calls || [];
        return new Proxy({}, { get: (t, k) => k in t ? t[k] : (...a) => { calls.push(k); }, set: (t, k, v) => (t[k] = v, true) });
      };
      win.Element.prototype.getBoundingClientRect = function () { return { left: 0, top: 0, width: 800, height: 380, right: 800, bottom: 380 }; };
      Object.defineProperty(win.HTMLElement.prototype, 'offsetParent', { get() {
        for (let e = this; e && e.nodeType === 1; e = e.parentElement) if (win.getComputedStyle(e).display === 'none') return null;
        return this.parentElement; } });
      win.addEventListener('error', e => errors.push('window.error: ' + e.message));
      win.fetch = (u, o) => fetch(new win.URL(u, URL).href, o);   // jsdom has no fetch
    } });
  const w = dom.window, d = w.document, $ = s => d.querySelector(s);
  await sleep(2500);                                   // first refresh() round trip
  ok(typeof w.GG === 'object' && w.GG.toggle, 'GG loaded');
  const btns = [...d.querySelectorAll('.ggbtn')];
  ok(btns.length === 7, '7 chart cards got a ▶ GG button: ' + btns.map(b => b.closest('.card').querySelector('h2').firstChild.textContent).join(', '));

  const b = $('#ch_decode').closest('.card').querySelector('.ggbtn');
  b.click();
  const host = $('.gghost'), card = b.closest('.card');
  ok(host && card.classList.contains('ggon') && host.previousElementSibling.tagName === 'H2', 'game opens next to the chart, not inside it');
  ok($('#ch_decode').contains(host) === false, 'chart element untouched');
  ok(b.textContent.includes('Graph'), 'button flips to "■ Graph"');
  ok(/Start/.test(host.querySelector('.ggover').textContent), 'start screen shown');
  const cv = host.querySelector('canvas');
  ok((cv._calls || []).length > 50, 'first frame drawn (' + (cv._calls || []).length + ' canvas calls)');

  host.querySelector('[data-a="start"]').click();
  await sleep(1500);
  const m1 = +host.querySelector('[data-k="m"]').textContent;
  ok(m1 >= 5, 'runner moving: ' + m1 + ' m after 1.5 s');

  const linesTxt = () => host.querySelector('[data-k="lines"]').textContent;
  const before = linesTxt();
  const ev = (type, x, y) => cv.dispatchEvent(new w.MouseEvent(type, { clientX: x, clientY: y, button: 0, bubbles: true }));
  ev('pointerdown', 300, 300); ev('pointermove', 420, 250); ev('pointerup', 420, 250);
  ok(linesTxt() !== before && /Available lines: 2/.test(linesTxt()), 'drawing a line uses one: ' + before + ' -> ' + linesTxt());
  ev('pointerdown', 300, 300); ev('pointerup', 302, 300);
  ok(/Available lines: 2/.test(linesTxt()), 'a click without a drag costs nothing');

  // switching panel tab pauses it (and stops the loop)
  d.querySelector('nav button[data-t="gpu"]').click();
  await sleep(300);
  d.querySelector('nav button[data-t="status"]').click();
  await sleep(200);
  ok(/Paused/.test(host.querySelector('.ggover').textContent), 'leaving the tab pauses the game');
  const callsPaused = cv._calls.length; await sleep(600);
  ok(cv._calls.length === callsPaused, 'no drawing while paused (loop stopped)');

  host.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'p', bubbles: true }));
  await sleep(300);
  ok(!host.querySelector('.ggover').classList.contains('on'), 'P resumes');

  // opening another card's game closes this one
  const b2 = $('#ch_vram').closest('.card').querySelector('.ggbtn');
  b2.click();
  ok(d.querySelectorAll('.gghost').length === 1 && !card.classList.contains('ggon') && b.textContent.includes('GG'),
     'only one game at a time; the first card is back to its graph');
  const host2 = $('.gghost');
  host2.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  ok(!d.querySelector('.gghost') && !d.querySelector('.card.ggon'), 'Esc closes and restores the graph');

  // let a game run into the first pit: game over with score, record and cause
  b.click(); const h3 = $('.gghost'); h3.querySelector('[data-a="start"]').click();
  for (let i = 0; i < 40 && !/Your score/.test(h3.querySelector('.ggover').textContent); i++) await sleep(500);
  const over = h3.querySelector('.ggover').textContent.replace(/\s+/g, ' ');
  ok(/Your score: \d+ m\s*Record: \d+ m/.test(over) && /Try again/.test(over), 'game over: ' + over.trim().slice(0, 120));
  ok(+w.localStorage.getItem('lexipanel.gg.record') > 0, 'record saved: ' + w.localStorage.getItem('lexipanel.gg.record'));
  ok(/archers passed: \d+/.test(over) && /arrows dodged: \d+/.test(over) && /This session: 1 run/.test(over), 'run stats and session on the game-over screen');
  ok(JSON.parse(w.localStorage.getItem('lexipanel.gg.bests') || '{}').m > 0, 'personal bests saved: ' + w.localStorage.getItem('lexipanel.gg.bests'));
  ok(JSON.parse(w.localStorage.getItem('lexipanel.gg.stats') || '{}').runs === 1, 'lifetime totals saved');
  h3.querySelector('[data-a="again"]').click(); await sleep(400);
  ok(+h3.querySelector('[data-k="m"]').textContent < 10 && !h3.querySelector('.ggover').classList.contains('on'), 'Try again restarts');
  h3.querySelector('[data-a="close"]').click();
  ok(!d.querySelector('.gghost'), 'Graph button closes');

  // the panel's 5 s refresh must not break while a game is open
  b.click(); await sleep(6000);
  ok($('.gghost') && $('#ch_decode'), 'survives a panel refresh');
  w.GG.close();

  ok(errors.length === 0, 'no page errors' + (errors.length ? ':\n' + errors.join('\n') : ''));
  console.log(`ui: ${pass} checks passed`);
  w.close(); process.exit(0);
})().catch(e => { console.error('FAIL:', e.message); if (errors.length) console.error(errors.join('\n')); process.exit(1); });
