// Open the panel in headless Chromium, click every tab: each must show content and raise no
// JavaScript error.   node tabs.test.js <port>
const { chromium } = require('playwright');
const port = process.argv[2] || '18190';
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error' && !/404 \(Not Found\)/.test(m.text())) errors.push('console: ' + m.text()); });
  page.on('response', r => { const u = new URL(r.url());
    if (u.pathname.startsWith('/api/') && r.status() >= 500) errors.push(`HTTP ${r.status()} ${u.pathname}`); });
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  await page.waitForTimeout(1500);
  const tabs = await page.$$eval('nav button[data-t]', bs => bs.map(b => b.dataset.t));
  let fail = tabs.length < 10 ? 1 : 0;
  for (const t of tabs) {
    const e0 = errors.length;
    await page.click(`nav button[data-t="${t}"]`);
    await page.waitForTimeout(1200);
    const chars = await page.$eval(`#${t}`, el => getComputedStyle(el).display !== 'none' && el.innerText.trim().length).catch(() => 0);
    const errs = errors.slice(e0);
    const bad = !chars || errs.length;
    fail += bad ? 1 : 0;
    console.log(`${bad ? 'FAIL' : 'ok  '} ${t.padEnd(10)} ${chars ? chars + ' chars' : 'EMPTY OR MISSING'}` + (errs.length ? '\n       ' + errs.join('\n       ') : ''));
  }
  await browser.close();
  console.log(`${tabs.length - fail}/${tabs.length} tabs ok`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
