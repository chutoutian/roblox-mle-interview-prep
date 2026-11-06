import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// The number of questions, counted from the markdown source, independently of book.js.
function countMlQuestions() {
  const md = fs.readFileSync(path.join(root, '02_ml_domain.md'), 'utf8');
  const chapter = md.slice(md.indexOf('\n# 2. Interview Questions'), md.indexOf('\n# 3. '));
  return chapter.split('\n').filter(line => /^\*\*\d+\. .*\?\*\*$/.test(line)).length;
}
function countCompanyQuestions() {
  const lines = fs.readFileSync(path.join(root, '08_company_interviews.md'), 'utf8').split('\n');
  return lines.filter((line, i) => /^#{2,6} /.test(line) && (lines.slice(i + 1).find(next => next.trim()) ?? '').startsWith('**Asked:**')).length;
}

const lists = [
  {name: 'ML interview questions', file: '02_ml_domain.html', page: 'ml', expected: countMlQuestions(), min: 60},
  {name: 'Company interview questions', file: '08_company_interviews.html', page: 'interviews', expected: countCompanyQuestions(), min: 70},
];

// Per page: every question carries one "reviewed" checkbox, keyed by its text (never its number or heading id);
// each chapter shows its own count; ticks survive a reload; names read well; it fits a phone; and it still
// works, without saving, when localStorage throws.
async function audit(page) {
  return page.evaluate(() => {
    const slug = text => text.toLowerCase().replace(/^\d+(?:[.-]\d+)*[.\s-]+/, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    const chapters = [...document.querySelectorAll('.chapter')].map(chapter => {
      const boxes = [...chapter.querySelectorAll('.review-check input[type="checkbox"]')];
      return {
        id: chapter.dataset.chapter, total: boxes.length, ticked: boxes.filter(box => box.checked).length,
        count: chapter.querySelector('.review-count')?.textContent.trim() ?? null,
        boxes: boxes.map(box => {
          const host = box.closest('label').parentElement;
          const text = document.getElementById(box.getAttribute('aria-describedby') ?? '');
          const heading = /^H[3-6]$/.test(host.tagName);
          return {
            key: box.dataset.reviewKey, label: box.getAttribute('aria-label'), first: host.firstElementChild === box.closest('label'),
            text: text?.textContent.trim() ?? null, heading,
            asked: heading && /^Asked:/.test(host.nextElementSibling?.firstElementChild?.textContent.trim() ?? ''),
            named: heading ? host.getAttribute('aria-labelledby') === text?.id : true,
            expectedKey: heading ? slug(chapter.dataset.chapter) + '/' + slug(text?.textContent ?? '') : slug(text?.textContent ?? ''),
          };
        }),
      };
    }).filter(chapter => chapter.total);
    return chapters;
  });
}

export async function checkReviewChecklists({page, target}) {
  const viewport = page.viewportSize();
  for (const list of lists) {
    assert.ok(list.expected >= list.min, `${list.name}: at least ${list.min} questions in the source (found ${list.expected})`);
    await page.goto(target(list.file));
    await page.waitForFunction(() => !!window.STUDY_BOOK);
    const prefix = `study-quiz:${list.page}:review:`;
    const saved = await page.evaluate(prefix => Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(prefix)).map(key => [key, localStorage.getItem(key)])), prefix);
    try {
      await page.evaluate(prefix => Object.keys(localStorage).filter(key => key.startsWith(prefix)).forEach(key => localStorage.removeItem(key)), prefix);
      await page.reload();
      await page.waitForFunction(() => !!window.STUDY_BOOK);
      let chapters = await audit(page);
      const boxes = chapters.flatMap(chapter => chapter.boxes);
      assert.equal(boxes.length, list.expected, `${list.name}: one checkbox per question`);
      for (const chapter of chapters) assert.equal(chapter.count, `Reviewed 0 of ${chapter.total} questions`, `${list.name}: chapter ${chapter.id} counts its own questions`);
      assert.equal(new Set(boxes.map(box => box.key)).size, boxes.length, `${list.name}: unique storage keys`);
      assert.deepEqual(boxes.filter(box => box.key !== box.expectedKey).map(box => box.key), [], `${list.name}: keys come from the question text (and company), not numbers or ids`);
      assert.ok(boxes.every(box => box.first && box.label === 'Mark as reviewed' && box.text && box.named), `${list.name}: each checkbox leads its question, reads "Mark as reviewed" and is described by the question; headings keep their own name`);
      if (list.page === 'interviews') assert.ok(boxes.every(box => box.heading && box.asked), `${list.name}: checkboxes sit in the headings followed by "Asked:"`);
      else assert.ok(boxes.every(box => !box.heading && /^\d+\. .*\?$/.test(box.text)), `${list.name}: checkboxes sit in the numbered question paragraphs`);

      await page.locator('[data-expand]').click();
      const target1 = page.locator('.review-check input').nth(1);
      await target1.check();
      chapters = await audit(page);
      assert.equal(chapters.reduce((sum, chapter) => sum + chapter.ticked, 0), 1, `${list.name}: one tick`);
      assert.ok(chapters.every(chapter => chapter.count === `Reviewed ${chapter.ticked} of ${chapter.total} questions`), `${list.name}: ticking updates the chapter count`);
      assert.equal(await target1.evaluate(node => node.closest('.reviewed') !== null), true, `${list.name}: a reviewed question is marked`);
      await page.reload();
      await page.waitForFunction(() => !!window.STUDY_BOOK);
      assert.equal(await page.locator('.review-check input').nth(1).isChecked(), true, `${list.name}: the tick survives a reload`);
      assert.equal(await page.locator('.review-check input').nth(0).isChecked(), false, `${list.name}: other questions stay unticked`);
      chapters = await audit(page);
      assert.ok(chapters.every(chapter => chapter.count === `Reviewed ${chapter.ticked} of ${chapter.total} questions`), `${list.name}: the counts survive a reload`);

      await page.setViewportSize({width: 390, height: 844});
      await page.locator('[data-expand]').click().catch(() => {});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${list.name}: fits a 390px phone`);
    } finally {
      await page.setViewportSize(viewport);
      await page.evaluate(([prefix, saved]) => {
        Object.keys(localStorage).filter(key => key.startsWith(prefix)).forEach(key => localStorage.removeItem(key));
        for (const [key, value] of Object.entries(saved)) localStorage.setItem(key, value);
      }, [prefix, saved]);
    }

    // Private mode: localStorage throws. The boxes still render and count; a toast says nothing is saved.
    const privateContext = await page.context().browser().newContext({viewport: {width: 1280, height: 900}});
    try {
      await privateContext.addInitScript(() => Object.defineProperty(window, 'localStorage', {configurable: true, get() { throw new DOMException('denied', 'SecurityError'); }}));
      const privatePage = await privateContext.newPage();
      const errors = [];
      privatePage.on('pageerror', error => errors.push(error.message));
      await privatePage.goto(target(list.file));
      await privatePage.waitForFunction(() => !!window.STUDY_BOOK);
      assert.equal(await privatePage.locator('.review-check input').count(), list.expected, `${list.name}: checkboxes render when storage is blocked`);
      await privatePage.locator('[data-expand]').click();
      await privatePage.locator('.review-check input').first().check();
      assert.match((await privatePage.locator('.review-count').first().textContent()).trim(), /^Reviewed 1 of \d+ questions$/, `${list.name}: the count works when storage is blocked`);
      assert.match(await privatePage.locator('.toast').textContent(), /cannot be saved/, `${list.name}: blocked storage is reported`);
      assert.deepEqual(errors, [], `${list.name}: no page errors when storage is blocked`);
    } finally {
      await privateContext.close();
    }
    console.log(`PASS ${list.name}: ${list.expected} questions, each with a "reviewed" checkbox keyed by its text, per-chapter counts, reload persistence, accessible names, a 390px layout, and graceful blocked storage.`);
  }
}
