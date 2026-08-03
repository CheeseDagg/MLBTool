#!/usr/bin/env node
/* selftest.js — headless check of mlb/index.html's render layer.
 *
 * WHY THIS FILE EXISTS. This repo has a dozen Python selftests and had nothing at all
 * covering the ~520-line render block that turns slate.json into the page. That gap is
 * not theoretical: it is exactly where the worst defect in the repo lived.
 *
 *   renderCal() — 87 lines that grade the HR board bucket by bucket against the
 *   25,128-batter-game season replay, report whether each ingredient earns its keep,
 *   and state in plain English whether both betting screens are losing — ended with
 *       $('hrCal').innerHTML = h;
 *   There is no element with id "hrCal" on the page, and nothing ever called renderCal.
 *   Below it, the win%-model backtest panel wrote to $('btBody'), also nonexistent,
 *   guarded by `if(_bt)` so it failed in total silence. And data/hr_backtest_panel.json,
 *   the strongest evidence this repo owns, was published on every backtest run and
 *   fetched by no page.
 *
 * A number nobody can see is not a safeguard, and "the code exists" is not the same
 * claim as "the code runs". This harness extracts the SHIPPED <script> block out of
 * index.html — not a copy — stubs the DOM and fetch, drives render() against the real
 * committed slate.json and the real committed panel, and asserts the output landed in
 * elements that actually exist in the markup.
 */
const fs = require('fs');
const path = require('path');
const HERE = path.join(__dirname, 'mlb');

const html = fs.readFileSync(path.join(HERE, 'index.html'), 'utf8');

let failures = 0;
function fail(msg) { console.log('  FAIL: ' + msg); failures++; }
function ok(msg) { console.log('  ok: ' + msg); }
function check(cond, msg) { cond ? ok(msg) : fail(msg); }

/* The page ships three script blocks (main render, line shop, K props). Only the
 * first defines render(); the other two are self-contained fetch-and-paint widgets. */
const blocks = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = blocks.filter(b => /function render\s*\(/.test(b));
if (main.length !== 1) { fail(`expected exactly 1 script block defining render(), found ${main.length}`); process.exit(1); }
// Strip the boot fetch; the harness drives render() itself.
const src = main[0].replace(/fetch\('data\/slate\.json'[\s\S]*$/, '');

/* ---------------------------------------------------------------- DOM stub */
/* IDS THAT REALLY EXIST. The whole point of this file is that writing to a
 * nonexistent id is silent, so the stub must NOT invent elements on demand the way
 * the soccer harness does. getElementById returns null for anything not in the
 * shipped markup, exactly like a browser. */
const REAL_IDS = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));

function makeDom() {
  const els = {};
  const mk = id => ({
    id, innerHTML: '', textContent: '', style: {}, className: '', value: '',
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    appendChild() {}, querySelectorAll: () => [], querySelector: () => null,
    addEventListener() {}, checked: false, dataset: {},
  });
  const doc = {
    getElementById: id => {
      if (!REAL_IDS.has(id)) return null;      // <- the bug class this file exists for
      return els[id] || (els[id] = mk(id));
    },
    querySelectorAll: () => [],
    querySelector: () => null,
    createElement: () => mk('_tmp'),
    addEventListener() {},
    body: mk('body'),
  };
  return { els, doc };
}

/* fetch stub: serve real committed files out of mlb/data, 404 anything else. */
function makeFetch(overrides) {
  const pending = [];
  const f = (url) => {
    const key = String(url).replace(/^data\//, '');
    let body;
    if (overrides && Object.prototype.hasOwnProperty.call(overrides, key)) {
      body = overrides[key];
    } else {
      const p = path.join(HERE, 'data', key);
      body = fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : undefined;
    }
    const pr = body === undefined
      ? Promise.resolve({ ok: false, status: 404, json: () => Promise.reject(new Error('404')) })
      : Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });
    pending.push(pr);
    return pr;
  };
  f.settle = async () => { for (let i = 0; i < 8; i++) await new Promise(r => setImmediate(r)); };
  return f;
}

async function runRender(slate, overrides) {
  const { els, doc } = makeDom();
  const fetchStub = makeFetch(overrides);
  const errors = [];
  const con = {
    log: () => {}, warn: () => {}, info: () => {},
    error: (...a) => errors.push(a.map(String).join(' ')),
  };
  const sandbox = {
    document: doc,
    window: { addEventListener() {}, location: { hash: '' }, matchMedia: () => ({ matches: false, addEventListener() {} }) },
    location: { hash: '' },
    console: con,
    fetch: fetchStub,
    setTimeout, clearTimeout, setInterval: () => 0, clearInterval: () => {},
    Math, JSON, Date, Number, String, Array, Object, Set, Map, Boolean,
    isFinite, isNaN, parseFloat, parseInt, encodeURIComponent, decodeURIComponent,
    MutationObserver: function () { this.observe = function () {}; this.disconnect = function () {}; },
  };
  const names = Object.keys(sandbox);
  const body = src + '\n;render(__SLATE__); return {els:__ELS__};';
  const fn = new Function(...names, '__SLATE__', '__ELS__', body);
  fn(...names.map(n => sandbox[n]), slate, els);
  await fetchStub.settle();
  return { els, errors, html: id => (els[id] ? els[id].innerHTML : null) };
}

const SLATE = JSON.parse(fs.readFileSync(path.join(HERE, 'data', 'slate.json'), 'utf8'));
const PANEL = JSON.parse(fs.readFileSync(path.join(HERE, 'data', 'hr_backtest_panel.json'), 'utf8'));
const clone = o => JSON.parse(JSON.stringify(o));

(async () => {

/* ------------------------------------- 0) the ids the renderers write to exist */
console.log('0) every id the render layer writes to is really in the markup');
// Catch the whole bug class, not just today's two instances: pull every $('...')
// target out of the shipped source and confirm the markup carries it.
const targets = [...src.matchAll(/\$\('([A-Za-z0-9_-]+)'\)/g)].map(m => m[1]);
const missing = [...new Set(targets)].filter(id => !REAL_IDS.has(id));
check(missing.length === 0,
  'no $(id) in the render block points at a nonexistent element' +
  (missing.length ? ' — MISSING: ' + missing.join(', ') : ''));

/* -------------------------------------------- 1) the real committed slate renders */
console.log('1) render the committed slate.json');
let r = await runRender(clone(SLATE));
check(r.errors.length === 0, 'render() threw nothing' + (r.errors.length ? ': ' + r.errors[0] : ''));
check((r.html('slateBody') || '').length > 50, 'the slate board painted');
check((r.html('hrBody') || '').length > 50, 'the HR board painted');

/* ------------------------------- 2) the calibration panel is no longer a no-op */
console.log('2) renderCal reaches the page');
const cal = r.html('hrCal');
check(cal !== null, 'hrCal exists and was written to (it was dead code for months)');
check(/graded batter-games/.test(cal || ''), 'the graded-rows headline rendered');
check(/THE VERDICT/.test(cal || ''),
  'the season panel was FETCHED and the verdict block rendered from it');
// The season replay is the whole reason this panel is worth showing. If the fetch
// silently 404s the panel must say so rather than quietly printing the live-only view
// as if it were the season read.
check(new RegExp(String(PANEL.n)).test(cal || '') || /full season/.test(cal || ''),
  'the panel quotes the season replay, not just this week');
check(!/undefined|NaN/.test(cal || ''), 'no "undefined"/"NaN" leaked into the calibration panel');

/* --------------------------- 3) it degrades when the season replay is missing */
console.log('3) a missing season panel degrades instead of lying');
r = await runRender(clone(SLATE), { 'hr_backtest_panel.json': undefined });
check(r.errors.length === 0, 'a 404 on the season panel does not throw');
const calNo = r.html('hrCal') || '';
check(/graded batter-games/.test(calNo), 'the live calibration still renders with no season data');
check(!/THE VERDICT/.test(calNo),
  'with no season replay the panel does NOT print a season verdict it cannot support');
check(!/undefined|NaN/.test(calNo), 'no "undefined"/"NaN" in the degraded panel');

/* -------------------------------- 4) the win%-model backtest panel reaches the page */
console.log('4) the win% backtest panel reaches the page');
r = await runRender(clone(SLATE));
const bt = r.html('btBody');
check(bt !== null, 'btBody exists and was written to (it was a silent no-op)');
check(/games predicted/.test(bt || '') && /Brier score/.test(bt || ''),
  'accuracy and Brier rendered');
check(new RegExp(String(SLATE.backtest.n)).test(bt || ''),
  'it reports the slate\'s own n, not a placeholder');

// A slate published before backtest_block() ran carries no "backtest" key. Before
// today that was survivable only because the element was missing; now it exists, so
// b.n would throw and take the rest of render() with it.
console.log('5) a slate with no backtest block does not take the page down');
const noBt = clone(SLATE); delete noBt.backtest;
r = await runRender(noBt);
check(r.errors.length === 0, 'render() survives a slate with no backtest key');
check(/carries no/.test(r.html('btBody') || ''), 'it says the block is missing rather than blanking');
check((r.html('hrBody') || '').length > 50, 'the HR board still painted (render did not abort)');

// Same for hr_cal: no graded rows yet is a legitimate state.
console.log('6) a slate with no hr_cal does not take the page down');
const noCal = clone(SLATE); delete noCal.hr_cal;
r = await runRender(noCal);
check(r.errors.length === 0, 'render() survives a slate with no hr_cal');
check((r.html('hrBody') || '').length > 50, 'the HR board still painted');

/* --------------------------------------------------- 7) no stale-cache fetches */
console.log('7) every data fetch is no-store');
const fetches = [...html.matchAll(/fetch\('(data\/[^']+)'([^)]*)\)/g)];
check(fetches.length > 0, 'found the data fetches');
const cached = fetches.filter(m => !/no-store/.test(m[2])).map(m => m[1]);
check(cached.length === 0,
  'no data fetch can be served from browser cache' +
  (cached.length ? ' — CACHEABLE: ' + cached.join(', ') : ''));

console.log(failures ? `\nFAILED (${failures})` : '\nALL GREEN');
process.exit(failures ? 1 : 0);

})();
