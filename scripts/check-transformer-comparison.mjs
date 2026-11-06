// Focused offline UI check for the merged Company Interviews comparison.
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
});
try {
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
  await context.route('**/*', route => /^https?:/.test(route.request().url())
    ? route.abort() : route.continue());
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const url = pathToFileURL(path.join(root, '08_company_interviews.html')).href;
  const id = 'original-transformer-bert-t5-and-gpt';
  await page.goto(`${url}#${id}`);
  const audit = await page.evaluate(headingId => {
    const heading = document.getElementById(headingId);
    const nodes = [];
    for (let node = heading?.nextElementSibling; node && !/^H[1-5]$/.test(node.tagName);
      node = node.nextElementSibling) nodes.push(node);
    return {
      found: !!heading,
      open: heading?.closest('details')?.open,
      tableCount: nodes.filter(node => node.classList.contains('table-scroll')).length,
      tableRows: nodes.reduce((count, node) => count + node.querySelectorAll('tr').length, 0),
      mathCount: nodes.reduce((count, node) => count + node.querySelectorAll('.katex').length, 0),
      mathErrors: document.querySelectorAll('.katex-error').length,
      overflow: document.documentElement.scrollWidth > innerWidth + 1,
      ids: [...document.querySelectorAll('[id]')].map(node => node.id),
      searchCount: window.STUDY_BOOK.search.filter(entry => entry.page === 'interviews'
        && entry.title === 'Original Transformer, BERT, T5, and GPT').length,
      comparisonFits: (() => {
        const wrapper = document.getElementById('transformer-family-comparison')?.nextElementSibling;
        return !!wrapper && wrapper.scrollWidth <= wrapper.clientWidth + 1;
      })(),
    };
  }, id);
  assert.equal(audit.found, true);
  assert.equal(audit.open, true);
  assert.equal(audit.mathErrors, 0);
  assert.equal(audit.overflow, false);
  assert.equal(audit.tableCount, 1);
  assert.equal(audit.tableRows, 9);
  assert.ok(audit.mathCount >= 15);
  assert.equal(audit.ids.length, new Set(audit.ids).size);
  assert.equal(audit.searchCount, 1);
  assert.equal(audit.comparisonFits, true, 'All four model columns fit the desktop reading width.');
  for (const old of ['gpt-and-bert-pre-training', 't5-compared-with-gpt']) {
    assert.equal(await page.locator(`h5[id="${old}"]`).count(), 0);
    await page.goto(`${url}#${old}`);
    assert.equal(await page.locator(`#${old}`).evaluate(node => node.closest('details')?.open), true);
  }
  await page.goto(`${url}#${id}`);
  await page.locator(`#${id}`).scrollIntoViewIfNeeded();
  await page.screenshot({path: '/private/tmp/company-transformer-comparison-desktop.png', animations: 'disabled'});
  const mobile = await context.newPage();
  await mobile.setViewportSize({width: 390, height: 844});
  await mobile.goto(`${url}#${id}`);
  await mobile.locator(`#${id}`).scrollIntoViewIfNeeded();
  assert.equal(await mobile.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
  assert.equal(await mobile.locator('#transformer-family-comparison + .table-scroll').evaluate(
    node => node.scrollWidth > node.clientWidth), true, 'Mobile scroll stays inside the comparison table.');
  await mobile.screenshot({path: '/private/tmp/company-transformer-comparison-mobile.png', animations: 'disabled'});
  assert.deepEqual(errors, []);
  console.log(`PASS merged comparison: ${audit.tableRows} table rows, ${audit.mathCount} rendered math expressions, both old bookmarks, desktop/mobile overflow and JS checks.`);
} finally {
  await browser.close();
}
