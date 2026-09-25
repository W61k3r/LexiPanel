// The Workload tab against two weeks of seeded traffic (serve.sh --seed-workload): tiles, depth
// bars, heatmap and its tooltips, table view, saving settings, a refused window, dismissing a
// proposal, a refused "Tune now" (no server running).   node workload.test.js <port>
const { chromium } = require('playwright');
const assert = require('assert');
const port = process.argv[2];
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('dialog', d => d.accept());
  const ok = (c, m) => { assert.ok(c, m); console.log('ok   ' + m); };
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForTimeout(800);
  await page.click('nav button[data-t="workload"]');
  await page.waitForSelector('#wl_tiles .stat', { timeout: 15000 });
  ok((await page.$$('#wl_tiles .stat')).length >= 5, 'summary tiles');
  ok((await page.$$('#wl_depth tr[data-i]')).length >= 3, 'depth histogram rows');
  ok((await page.$$('#wl_heat .wlcell')).length === 168, '168 hours in the heatmap');
  ok((await page.$$('#wl_heat .wlcell.wlidle')).length > 20, 'idle windows learned');
  ok((await page.$$eval('#wl_find b', b => b.map(x => x.textContent))).some(t => /never reached/.test(t)), 'context finding');
  await page.hover('#wl_depth .wlbar');
  await page.waitForTimeout(200);
  ok(/requests/.test(await page.$eval('.wltip', t => t.innerText)), 'bar tooltip');
  await page.click('#wl_heat details summary');
  ok((await page.$$('#wl_heat details tr')).length === 8, 'table view of the heatmap');
  await page.selectOption('#af_mode', 'propose');
  await page.click('#workload button[onclick="afSave()"]');
  await page.waitForTimeout(800);
  ok((await page.textContent('#af_msg')) === 'saved', 'settings saved');
  const bad = await page.evaluate(async () => (await fetch('/api/workload/settings?inst=main', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ window: '25-99' }) })).json());
  ok(/window/.test(bad.error || ''), 'a bad window is refused');
  await page.click('#af_props button:has-text("Dismiss")');
  await page.waitForTimeout(800);
  ok(/No proposals/.test(await page.textContent('#af_props')), 'proposal dismissed');
  await page.click(`#workload button[onclick="afRun('tune')"]`);
  await page.waitForTimeout(800);
  ok(/not running/.test(await page.textContent('#af_msg')), '"Tune now" refused without a server');
  ok(errors.length === 0, 'no JavaScript errors' + (errors.length ? ': ' + errors.join('; ') : ''));
  await browser.close();
})().catch(e => { console.error('FAIL ' + e.message); process.exit(1); });
