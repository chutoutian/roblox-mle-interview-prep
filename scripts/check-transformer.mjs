import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {chromium} from 'playwright';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const interviewId = '2-interview-questions--transformers--llms';
const transformerId = '3-transformer-from-input-to-output';
const vlmId = '4-vlm-from-image-and-question-to-answer';
const generativeId = '6-generative-models--learned-compression';
const chapterTitles = ['1. Basics', '2. Interview Questions — Transformers & LLMs', '3. Transformer: From Input to Output', '4. VLM: From Image and Question to Answer', '5. Post-Training: LoRA', '6. Generative Models & Learned Compression', '7. Inference', '8. Scaling Laws & Transformer Accounting', '9. GPUs & the Roofline Model', '10. Precision, Efficiency & Deployment', '11. Parallelism', '12. Metrics, Evaluation & Debugging', '13. Rapid-Fire Drill — ML', '14. Coding Reverse Index — ML'];
// Chapter 2 is the interview Q&A: numbered bold questions with answers, grouped in six sections, no code.
const interview = {
  chapterId: interviewId, min: 116,
  sections: ['2.1 Positions', '2.2 Normalization, residuals and training stability', '2.3 Attention and efficient attention', '2.4 Architecture and modern LLMs', '2.5 Training and inference at scale', '2.6 Interview Questions — Computer Vision'],
  first: 'Why is an added positional encoding absolute while RoPE is relative?',
  required: ['How do BatchNorm, LayerNorm and RMSNorm differ', 'Why do Transformers use LayerNorm instead of BatchNorm', 'Why did the original Transformer use post-norm'],
  anchors: ['75-efficient-attention', '76-cnn-vs-transformer', '92-tokenization', '26-interview-questions--computer-vision', ...Array.from({length: 50}, (_, i) => `cv-q${String(i + 1).padStart(2, '0')}`), ...Array.from({length: 5}, (_, i) => `cv-case${String(i + 1).padStart(2, '0')}`)],
};
const basicsSections = ['1.1 Neural Net Basics', '1.2 Mathy Things & Losses', '1.3 Training Mechanics', '1.4 CNNs & Other Architectures'];
const formerBasics = ['1-neural-net-basics', '2-mathy-things--losses', '3-training-mechanics', '8-cnns--other-architectures'];
// The "from input to output" chapters share one format: bold steps with equations and a shape
// table. VLM, LoRA and Inference then embed the runnable code from a tested file; the Transformer
// chapter shows no code (it is read in VS Code: the .py and its notebook). Its former "Beyond this
// example" section became interview questions (chapter 2) and a decoder-only GPT-2 walkthrough.
const walkthroughs = [
  {
    name: 'Transformer', chapterId: transformerId, shots: 'transformer',
    sections: ['Rotary positions (RoPE)', 'Decoder-only LLMs: GPT-2, step by step', 'Vision Transformers, from ViT to today'],
    code: false,
    gpt2: {section: 'decoder-only-llms-gpt-2-step-by-step', steps: 7, total: '124,439,808'},
    rope: {section: 'rotary-positions-rope', leads: ['Where it happens.', 'Notation.', 'The rule.', 'Why the score sees only the distance.', '1D: text.', '2D: images.', '3D: video, and words and images in one sequence.', 'The three cases side by side.'], tables: 12, symbols: 8},
    anchors: ['9-llm-internals', '91-architecture-decoder-only-the-2026-default', 'decoder-only-llms', 'gpt2-params', 'gpt2-vs-modern', 'gpt2-shapes', 'transformer-math', 'transformer-shapes', 'pytorch-implementation', 'step-1-images', 'rotary-positions-rope', 'rope-where', 'rope-notation', 'rope-1d', 'rope-2d', 'rope-3d', 'rope-cases', 'rope-compare', 'coding-scaled-attention', 'coding-mha', 'coding-causal-mask', 'coding-rope', 'coding-transformer-block'],
    shapes: {anchor: 'transformer-shapes', rows: 13, contains: '[4, 10]'},
    source: 'transformer_encoder_decoder.py', tests: 'test_transformer_encoder_decoder.py',
    display: 22, inline: 60,
  },
  {
    name: 'VLM', chapterId: vlmId, shots: 'vlm',
    sections: ['Classic models, from CLIP to today', 'PyTorch implementation: ViT + projector + LLM'],
    models: {anchor: 'vlm-model-line', first: 'CLIP', last: 'Gemma 4', rows: 15, ways: 3},
    codeSection: 'pytorch-implementation-vit--projector--llm',
    anchors: ['vlm-math', 'vlm-shapes', 'vlm-models', 'vlm-model-line', 'coding-vlm', 'pytorch-implementation-vit--projector--llm', '10-vlm--mllm', '102-vlm-architecture-families', '132-vlm-architecture-families'],
    shapes: {anchor: 'vlm-shapes', rows: 9, contains: '[576, 4096]'},
    source: 'vlm_vit_projector_llm.py', tests: 'test_vlm_vit_projector_llm.py',
    regions: ['vlm-imports', 'vlm-patches', 'vlm-blocks', 'vlm-vision', 'vlm-model', 'vlm-demo'],
    review: 'vlm-model', output: /step 8 .*\[1, 11, 13\][\s\S]*stage 1 \(projector only[\s\S]*a dog <eos>/, display: 12, inline: 40,
  },
  {
    name: 'LoRA', chapterId: '5-post-training-lora', shots: 'lora', steps: 10,
    sections: ['PyTorch implementation: LoRA on the toy Transformer'],
    codeSection: 'pytorch-implementation-lora-on-the-toy-transformer',
    anchors: ['lora-math', 'lora-where', 'lora-notation', 'lora-memory', 'lora-shapes', 'coding-lora', '116-lora-and-peft', '11-post-training-sft-rlhf-dpo-grpo'],
    shapes: {anchor: 'lora-shapes', rows: 8, contains: '[4096, 16]'},
    source: 'lora_linear.py', tests: 'test_lora_linear.py',
    regions: ['lora-imports', 'lora-linear', 'lora-apply', 'lora-demo'],
    review: 'lora-linear', output: /LoRA r=2 on 12 matrices: 384 of 2578[\s\S]*frozen weights unchanged: True[\s\S]*merged: same logits True/, display: 7, inline: 40,
  },
  {
    name: 'Inference', chapterId: '7-inference', shots: 'inference', steps: 5,
    sections: ['PyTorch implementation: KV cache and sampling'],
    codeSection: 'pytorch-implementation-kv-cache-and-sampling',
    anchors: ['inference-math', 'inference-notation', 'inference-shapes', 'coding-inference', 'coding-kv-cache', 'coding-llm-sampling', '51-kv-cache--the-thing-that-dominates-inference', '52-prefill-vs-decode', '53-sampling', '54-batching-packing-and-scheduling', '93-kv-cache--the-thing-that-dominates-inference', '94-prefill-vs-decode', '95-sampling'],
    shapes: {anchor: 'inference-shapes', rows: 5, contains: '4.5 ms'},
    source: 'inference_kv_cache.py', tests: 'test_inference_kv_cache.py',
    regions: ['inference-imports', 'inference-model', 'inference-cache', 'inference-sampling', 'inference-demo'],
    review: 'inference-cache', output: /cache matches full recompute: True[\s\S]*greedy with cache == greedy without cache: True/, display: 8, inline: 60,
  },
];
// The vision transformer section of Chapter 2: one example image, each variant as input ⟶ output
// with its equations, a cost table and the model line. Its code (vision_transformers.py) is read in VS Code, not embedded.
const vision = {
  section: 'vision-transformers-from-vit-to-today',
  anchors: ['vision-transformers', 'vit-example', 'vit-cost', 'vit-model-line', 'coding-vit'],
  leads: ['ViT', 'DeiT', 'Swin', 'PVT', 'MAE', 'DINO', 'ViTDet and SAM', 'TimeSformer and ViViT', 'NaViT', 'DiT'],
  models: {first: 'ViT', last: 'DINOv3', rows: 16},
  source: 'vision_transformers.py', tests: 'test_vision_transformers.py',
  display: 16,
};

async function checkVision(page, preview) {
  const chapter = page.locator(`#chapter-${transformerId}`);
  for (const id of vision.anchors) assert.equal(await chapter.locator(`[id="${id}"]`).count(), 1, `${id}: anchor inside the Transformer chapter`);
  // Everything between the section heading and the next heading.
  const section = await chapter.locator(`[id="${vision.section}"]`).evaluate(heading => {
    const nodes = [];
    for (let node = heading.nextElementSibling; node && !/^H[23]$/.test(node.tagName); node = node.nextElementSibling) nodes.push(node);
    const leads = nodes.filter(node => node.tagName === 'P' && node.firstElementChild?.tagName === 'STRONG' && node.firstElementChild.textContent.includes(' · '))
      .map(node => ({name: node.firstElementChild.textContent.split(' · ')[0], arrow: [...node.querySelectorAll('.katex-html')].some(math => math.textContent.includes('⟶')) || node.textContent.includes('⟶')}));
    const displays = nodes.flatMap(node => [...node.querySelectorAll('.katex-display')]);
    return {
      leads, displays: displays.length,
      wide: displays.filter(block => block.scrollWidth > block.clientWidth + 1).map(block => block.textContent.slice(0, 40)),
      errors: nodes.flatMap(node => [...node.querySelectorAll('.katex-error')]).length,
      tables: nodes.filter(node => node.matches('.table-scroll')).map(node => ({anchor: node.previousElementSibling?.id ?? '', fits: node.scrollWidth <= node.clientWidth + 1})),
      code: nodes.filter(node => node.matches('.code-block')).map(node => ({language: node.querySelector('.code-lang')?.textContent.trim(), text: node.querySelector('code').textContent})),
    };
  });
  assert.deepEqual(section.leads.map(lead => lead.name), vision.leads, 'Vision: one lead per variant, in order');
  assert.ok(section.leads.every(lead => lead.arrow), 'Vision: every variant states its input ⟶ output');
  assert.ok(section.displays >= vision.display, `Vision: display equations (found ${section.displays})`);
  assert.equal(section.errors, 0, 'Vision: no KaTeX errors');
  assert.deepEqual(section.wide, [], 'Vision: every equation fits the desktop column');
  assert.deepEqual(section.tables, [{anchor: 'vit-cost', fits: true}, {anchor: 'vit-model-line', fits: true}], 'Vision: the cost table and the model line read without panning');

  const models = chapter.locator('[id="vit-model-line"] + .table-scroll');
  const rows = await models.locator('tbody tr').evaluateAll(nodes => nodes.map(row => ({name: row.cells[0].textContent.trim(), year: Number(row.cells[0].textContent.match(/(20\d\d)\s*$/)?.[1]), link: row.cells[0].querySelector('a[href^="https://arxiv.org/abs/"]') !== null, cells: row.cells.length})));
  assert.equal(rows.length, vision.models.rows, 'Vision: model line rows');
  assert.ok(rows[0].name.startsWith(vision.models.first) && rows.at(-1).name.startsWith(vision.models.last), `Vision: the line runs from ${vision.models.first} to ${vision.models.last}`);
  assert.ok(rows.every(row => row.link && row.cells === 5 && row.year >= 2020), 'Vision: every model has a year and an arXiv link');
  assert.deepEqual(rows.map(row => row.year), rows.map(row => row.year).sort((a, b) => a - b), 'Vision: models in chronological order');
  assert.equal(await models.evaluate(node => getComputedStyle(node.querySelector('td')).whiteSpace), 'normal', 'Vision: the model line wraps');

  // The code is read in VS Code (vision_transformers.py and its tests), so the section embeds none.
  assert.deepEqual(section.code, [], 'Vision: no implementation code or recorded output in the section');
  await capture(page, chapter.locator(`[id="vit-model-line"] + .table-scroll`), preview, 'vision-models-desktop.png');
  return section;
}

const componentStyle = 'html{scroll-behavior:auto!important}.topbar,.read-progress,.skip-link,.sidebar,.sidebar-backdrop,.toast{visibility:hidden!important}';

async function openContent(page) {
  await page.locator('article details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
}

async function noOverflow(page, label) {
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, label);
}

async function capture(page, locator, preview, name, maxHeight = Infinity) {
  const viewport = page.viewportSize();
  const box = await locator.boundingBox();
  assert.ok(box, `${name}: visible screenshot target`);
  const height = Math.min(Math.ceil(box.height), maxHeight);
  const tall = Math.max(viewport.height, height + 180);
  await page.setViewportSize({width: viewport.width, height: tall});
  try {
    if (height < box.height) {
      // Only the top of a very tall element: align it with the viewport top, then clip the viewport.
      await locator.evaluate(node => { document.documentElement.style.scrollBehavior = 'auto'; node.scrollIntoView({block: 'start'}); });
      const top = await locator.boundingBox();
      const y = Math.max(0, top.y);
      await page.screenshot({path: path.join(preview, name), clip: {x: Math.max(0, top.x), y, width: top.width, height: Math.min(height, tall - y)}, style: componentStyle, animations: 'disabled'});
      await page.evaluate(() => { document.documentElement.style.scrollBehavior = ''; });
    } else {
      await locator.screenshot({path: path.join(preview, name), style: componentStyle, animations: 'disabled'});
    }
  } finally {
    await page.setViewportSize(viewport);
  }
}

function regions(config) {
  const lines = fs.readFileSync(path.join(root, config.source), 'utf8').split(/\r?\n/);
  const names = lines.flatMap(line => line.match(/^# BEGIN ([a-z-]+)$/)?.slice(1) || []);
  assert.deepEqual(names, config.regions, `${config.source}: named teaching regions in equation order`);
  return names.map(name => {
    const start = lines.indexOf(`# BEGIN ${name}`), end = lines.indexOf(`# END ${name}`);
    assert.ok(start >= 0 && end > start, `Valid code region: ${name}`);
    return {name, code: lines.slice(start + 1, end).join('\n').trim()};
  });
}

async function checkPageLinks(page) {
  const audit = await page.evaluate(() => ({
    ids: [...document.querySelectorAll('[id]')].map(node => node.id),
    hrefs: [...document.querySelectorAll('a[href]')].map(node => node.getAttribute('href')),
    language: document.documentElement.lang,
    chinese: /[㐀-䶿一-鿿豈-﫿]/u.test(document.documentElement.outerHTML),
  }));
  assert.equal(audit.language, 'en');
  assert.equal(audit.chinese, false, 'ML content is English only');
  assert.deepEqual(audit.ids.filter((id, i, all) => all.indexOf(id) !== i), [], 'No duplicate ML IDs');
  const cachedIds = new Map([[fileURLToPath(page.url()), new Set(audit.ids)]]);
  for (const href of audit.hrefs) {
    if (!href) continue;
    const url = new URL(href, page.url());
    if (url.protocol !== 'file:') continue;
    const file = fileURLToPath(url);
    assert.ok(fs.existsSync(file), `Local link exists: ${href}`);
    if (!url.hash || !file.endsWith('.html')) continue;
    if (!cachedIds.has(file)) {
      const html = fs.readFileSync(file, 'utf8');
      cachedIds.set(file, new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1])));
    }
    assert.ok(cachedIds.get(file).has(decodeURIComponent(url.hash.slice(1))), `Fragment exists: ${href}`);
  }
}

async function mathAudit(scope) {
  return scope.evaluate(node => ({
    display: node.querySelectorAll('.katex-display').length,
    inline: node.querySelectorAll('.katex').length - node.querySelectorAll('.katex-display .katex').length,
    errors: node.querySelectorAll('.katex-error').length,
    overflowing: [...node.querySelectorAll('.katex-display')].filter(block => getComputedStyle(block).overflowX !== 'auto').length,
  }));
}

// The four foundation chapters were merged into "1. Basics", the VLM chapter was rewritten and
// moved next to the Transformer, and the rest renumbered twice: structure, old bookmarks and
// saved reading progress must all survive.
async function checkBookLayout(page, target) {
  const meta = await page.evaluate(() => window.STUDY_BOOK.docs.ml);
  assert.deepEqual(meta.chapters.map(chapter => chapter.title), chapterTitles, 'Fourteen chapters: foundations, interview questions, the walkthroughs, then generative models as chapter 6');
  assert.equal(await page.locator('details.chapter[data-chapter]').count(), chapterTitles.length);
  assert.equal(meta.drills, 18, 'Coding exercises: six generative-model and compression folds live in chapter 5');
  const basics = page.locator('#chapter-1-basics');
  assert.deepEqual(await basics.locator('.prose h3').evaluateAll(nodes => nodes.map(node => node.textContent.replace(/#\s*$/, '').trim())), basicsSections);
  assert.equal(await basics.locator('details.coding-drill').count(), 8, 'Basics keeps the eight exercises of its four former chapters');
  for (const text of ['Backpropagation', 'Cross-entropy', 'SGD vs Adam', 'Normalization', 'Convolution basics', 'State-space models']) {
    assert.ok((await basics.locator('.prose h4, .prose h5').allTextContents()).some(title => title.includes(text)), `Basics still contains: ${text}`);
  }

  const aliases = Object.entries(meta.legacyAliases);
  assert.ok(aliases.length >= 70, `Former heading ids are kept as aliases (found ${aliases.length})`);
  const unresolved = await page.evaluate(entries => entries.filter(([alias, destination]) => document.getElementById(alias)?.parentElement.closest('[id]')?.id !== destination).map(([alias]) => alias), aliases);
  assert.deepEqual(unresolved, [], 'Every former heading id resolves inside its new heading');
  for (const [alias, chapter] of [['3-training-mechanics', '1-basics'], ['81-convolution-basics', '1-basics'], ['4-transformer-from-input-to-output', transformerId], ['2-transformer-from-input-to-output', transformerId], ['3-inference', '7-inference'], ['4-inference', '7-inference'], ['5-inference', '7-inference'], ['6-inference', '7-inference'], ['102-data-parallelism-and-fsdp--zero', '11-parallelism'], ['10-parallelism', '11-parallelism'], ['85-diffusion', generativeId], ['1-learned--generative-compression', generativeId], ['21-vq-vae', generativeId], ['5-generative-models--learned-compression', generativeId], ['541-the-core-idea', generativeId], ['13-multimodality--vlm', vlmId], ['10-multimodality--vlm', vlmId], ['10-vlm-from-image-and-question-to-answer', vlmId], ['3-vlm-from-image-and-question-to-answer', vlmId], ['132-vlm-architecture-families', vlmId], ['86-lora-and-peft', '5-post-training-lora'], ['84-dpo', '5-post-training-lora'], ['8-post-training-lora', '5-post-training-lora'], ['4-post-training-lora', '5-post-training-lora'], ['71-count-parameters-flops-and-memory', '8-scaling-laws--transformer-accounting'], ['115-evaluating-generative-and-llm-outputs', '12-metrics-evaluation--debugging'], ['12-rapid-fire-drill--ml', '13-rapid-fire-drill--ml'], ['13-coding-reverse-index--ml', '14-coding-reverse-index--ml']]) {
    await page.goto(target('02_ml_domain.html') + '#' + alias);
    const found = await page.locator(`[id="${alias}"]`).evaluate(node => { const owner = node.closest('[data-chapter]'); return [owner?.dataset.chapter, owner?.open]; });
    assert.deepEqual(found, [chapter, true], `${alias}: an old bookmark opens its new chapter`);
  }

  const saved = await page.evaluate(() => localStorage.getItem('study-progress-v1'));
  const migrate = async state => {
    await page.evaluate(value => localStorage.setItem('study-progress-v1', JSON.stringify(value)), state);
    await page.goto(target('02_ml_domain.html'));
    await page.waitForFunction(() => !!window.STUDY_BOOK);
    return page.evaluate(() => JSON.parse(localStorage.getItem('study-progress-v1')));
  };
  try {
    const complete = await migrate({ml: {done: [...formerBasics, '5-inference'], last: '33-normalization'}, network: {done: ['1-network-basics']}});
    assert.ok(complete.ml.done.includes('1-basics'), 'Basics is read once all four former chapters were read');
    assert.ok(complete.ml.done.includes('7-inference'), 'A chapter keeps its completion through every renumbering');
    assert.ok(!complete.ml.done.includes(transformerId), 'An unread chapter stays unread');
    assert.equal(complete.ml.last, '133-normalization', 'The resume location follows its section');
    assert.deepEqual(complete.network.done, ['1-network-basics'], 'Other books are untouched');
    assert.equal(await page.locator('#chapter-1-basics [data-mark]').getAttribute('aria-pressed'), 'true');
    const partial = await migrate({ml: {done: formerBasics.slice(0, 3)}});
    assert.ok(!partial.ml.done.includes('1-basics'), 'A partially read merged chapter must not become complete');
    const rewritten = await migrate({ml: {done: ['13-multimodality--vlm', '10-multimodality--vlm'], last: '10-multimodality--vlm', layoutVersion: 1}});
    assert.ok(!rewritten.ml.done.includes(vlmId), 'The rewritten VLM chapter is new material and is not marked read');
    assert.equal(rewritten.ml.last, vlmId, 'The resume location follows the renamed chapter');
    const moved = await migrate({ml: {done: ['10-vlm-from-image-and-question-to-answer', '9-parallelism', '8-post-training-lora'], last: '31-kv-cache--the-thing-that-dominates-inference', layoutVersion: 2}});
    assert.ok(moved.ml.done.includes(vlmId) && moved.ml.done.includes('11-parallelism') && moved.ml.done.includes('5-post-training-lora'), 'Moving a chapter keeps its completion');
    assert.equal(moved.ml.last, '7-inference', 'A former inference section resumes at the rewritten chapter');
    const again = await migrate({ml: {done: ['12-parallelism'], layoutVersion: meta.navigationMigration.version}});
    assert.ok(!again.ml.done.includes('11-parallelism'), 'Migration runs once per layout version');
    const generative = await migrate({ml: {done: ['8-generative-models', '9-precision-efficiency--deployment', '13-coding-reverse-index--ml'], last: '85-diffusion', layoutVersion: 5}});
    assert.deepEqual(generative.ml.done.sort(), ['10-precision-efficiency--deployment', '14-coding-reverse-index--ml'], 'Renumbered chapters keep their completion through three layouts; the old generative chapter is not marked read');
    assert.equal(generative.ml.layoutVersion, meta.navigationMigration.version);
    assert.equal(await page.locator(`[id="${generative.ml.last}"]`).evaluate(node => node.closest('[data-chapter]')?.dataset.chapter), generativeId, 'A bookmark inside the old generative chapter resumes inside chapter 6');
    const sixth = await migrate({ml: {done: ['5-inference', '9-parallelism'], last: '9-parallelism', layoutVersion: 6}});
    assert.deepEqual(sixth.ml.done.sort(), ['11-parallelism', '7-inference'], 'A version-6 state moves up by two chapters across two layouts');
    assert.equal(sixth.ml.last, '11-parallelism');
    // Version 8 inserted the interview chapter as chapter 2: chapters 2-13 became 3-14.
    const seventh = await migrate({ml: {done: ['2-transformer-from-input-to-output', '6-inference', '12-rapid-fire-drill--ml'], last: '71-count-parameters-flops-and-memory', layoutVersion: 7}});
    assert.deepEqual(seventh.ml.done.sort(), ['13-rapid-fire-drill--ml', transformerId, '7-inference'].sort(), 'A version-7 state moves up by one chapter');
    assert.ok(!seventh.ml.done.includes(interviewId), 'The new interview chapter is not marked read');
    assert.equal(seventh.ml.last, '81-count-parameters-flops-and-memory', 'A section bookmark follows its renumbered heading');
    assert.equal(seventh.ml.layoutVersion, 8);
  } finally {
    await page.evaluate(value => value === null ? localStorage.removeItem('study-progress-v1') : localStorage.setItem('study-progress-v1', value), saved);
  }
  await page.goto(target('02_ml_domain.html'));
  await page.waitForFunction(() => !!window.STUDY_BOOK);
}

// The interview chapter: six sections of numbered questions, each answered, no code, KaTeX intact.
async function checkInterview(page) {
  const chapter = page.locator(`#chapter-${interview.chapterId}`);
  assert.equal(await chapter.count(), 1, 'Interview: chapter id');
  assert.deepEqual(await chapter.locator('.prose h3').evaluateAll(nodes => nodes.map(node => node.textContent.replace(/#\s*$/, '').trim())), interview.sections, 'Interview: the six question groups');
  for (const id of interview.anchors) assert.equal(await chapter.locator(`[id="${id}"]`).count(), 1, `${id}: anchor inside the interview chapter`);
  const qa = await chapter.locator('.prose').evaluate(prose => {
    // book.js puts a "reviewed" checkbox in front of each question's bold lead.
    const lead = node => node.firstElementChild?.matches('label.review-check') ? node.firstElementChild.nextElementSibling : node.firstElementChild;
    const isQuestion = node => node?.tagName === 'P' && lead(node)?.tagName === 'STRONG' && node.firstChild === node.firstElementChild && /^\d+\. .*\?$/.test(node.textContent.trim());
    const nodes = [...prose.children];
    return {
      questions: nodes.filter(isQuestion).map(node => ({text: node.textContent.trim(), answered: !!node.nextElementSibling && !isQuestion(node.nextElementSibling) && node.nextElementSibling.textContent.trim().length > 40, checkbox: !!node.querySelector(':scope > label.review-check > input[type="checkbox"]')})),
      code: prose.querySelectorAll('.code-block').length,
      errors: prose.querySelectorAll('.katex-error').length,
    };
  });
  const numbers = qa.questions.map(question => Number(question.text.match(/^(\d+)\./)[1]));
  assert.ok(qa.questions.length >= interview.min, `Interview: at least ${interview.min} questions (found ${qa.questions.length})`);
  assert.deepEqual(numbers, numbers.map((_, i) => i + 1), 'Interview: questions are numbered 1..N');
  assert.deepEqual(qa.questions.filter(question => !question.answered).map(question => question.text), [], 'Interview: every question has an answer');
  assert.deepEqual(qa.questions.filter(question => !question.checkbox).map(question => question.text), [], 'Interview: every question has a reviewed checkbox');
  assert.equal(qa.questions[0].text.replace(/^1\. /, ''), interview.first, 'Interview: the first question');
  for (const prefix of interview.required) assert.ok(qa.questions.some(question => question.text.replace(/^\d+\. /, '').startsWith(prefix)), `Interview: question "${prefix}..."`);
  assert.equal(qa.code, 0, 'Interview: the answers contain no code');
  assert.equal(qa.errors, 0, 'Interview: no KaTeX errors');
  assert.deepEqual(await chapter.locator('.table-scroll').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 1).length), 0, 'Interview: the comparison table fits the desktop column');
  const cv = await chapter.locator('[id="26-interview-questions--computer-vision"]').evaluate(heading => {
    const nodes = [];
    for (let node = heading.nextElementSibling; node; node = node.nextElementSibling) nodes.push(node);
    const all = nodes.flatMap(node => [node, ...node.querySelectorAll('*')]);
    return {
      text: nodes.map(node => node.textContent).join('\n'),
      questions: nodes.filter(node => node.tagName === 'P' && /^\d+\. .*\?$/.test(node.textContent.trim())).map(node => Number(node.textContent.match(/^(\d+)\./)[1])),
      details: all.filter(node => node.tagName === 'SUMMARY').map(node => node.textContent.trim()),
      math: all.filter(node => node.classList.contains('katex')).length,
      badLinks: all.filter(node => node.tagName === 'A' && node.getAttribute('href')?.startsWith('#cv-') && !document.getElementById(node.getAttribute('href').slice(1))).map(node => node.getAttribute('href')),
    };
  });
  assert.deepEqual(cv.questions, Array.from({length: 55}, (_, i) => i + 62), 'CV: 50 core questions and five design cases');
  assert.equal(/[\u3400-\u9fff]/u.test(cv.text), false, 'CV: English content only');
  assert.equal(/GenUI|DIL\.useState|<Cite/.test(cv.text), false, 'CV: no source UI or internal citations');
  assert.ok(cv.details.includes('Detailed CNN versus fully connected example'), 'CV: latest CNN follow-up retained');
  assert.ok(cv.details.includes('Additional prompts from the conversation'), 'CV: alternate question prompts retained');
  assert.ok(cv.math >= 50, `CV: equations rendered (found ${cv.math})`);
  assert.deepEqual(cv.badLinks, [], 'CV: all question and case cross-links resolve');
  await capture(page, chapter.locator('.prose'), path.join(root, '.preview-check'), 'interview-desktop.png', 1600);
  await capture(page, chapter.locator('[id="cv-q01"] + p + p + details'), path.join(root, '.preview-check'), 'cv-cnn-desktop.png', 1600);
  const cvHeading = chapter.locator('[id="26-interview-questions--computer-vision"]');
  await cvHeading.evaluate(node => { document.documentElement.style.scrollBehavior = 'auto'; node.scrollIntoView({block: 'start'}); });
  await page.screenshot({path: path.join(root, '.preview-check', 'cv-section-desktop.png'), animations: 'disabled'});
  await page.evaluate(() => { document.documentElement.style.scrollBehavior = ''; });
  return qa.questions.length;
}

// Structure, equations, shape table and embedded code of one walkthrough chapter (desktop, light).
async function checkWalkthrough(page, config, preview) {
  const chapter = page.locator(`#chapter-${config.chapterId}`);
  assert.equal(await chapter.count(), 1, `${config.name}: chapter id`);
  const headings = await chapter.locator('.prose h3').evaluateAll(nodes => nodes.map(node => node.textContent.replace(/#\s*$/, '').trim()));
  assert.deepEqual(headings, config.sections, `${config.name}: sections after the walkthrough`);
  assert.equal(await chapter.locator('.prose h4, .prose h5, .prose h6').count(), 0, `${config.name}: no nested sub-sections`);
  const steps = await chapter.locator('.prose p').evaluateAll(nodes => nodes
    .filter(node => node.firstElementChild?.tagName === 'STRONG' && /^Step \d+ · /.test(node.firstElementChild.textContent))
    .map(node => ({number: Number(node.firstElementChild.textContent.match(/^Step (\d+)/)[1]), arrow: [...node.querySelectorAll('.katex-html')].some(math => math.textContent.includes('⟶'))})));
  assert.deepEqual(steps.map(step => step.number), Array.from({length: config.steps ?? 11}, (_, i) => i + 1), `${config.name}: ${config.steps ?? 11} steps, in order`);
  assert.ok(steps.every(step => step.arrow), `${config.name}: every step states its input ⟶ output`);
  for (const id of config.anchors) assert.equal(await chapter.locator(`[id="${id}"]`).count(), 1, `${id}: anchor inside the ${config.name} chapter`);

  if (config.gpt2) {
    // The decoder-only walkthrough: seven "GPT-2 step k" leads with input ⟶ output, and the exact parameter count.
    const gpt2 = await chapter.locator(`[id="${config.gpt2.section}"]`).evaluate(heading => {
      const nodes = [];
      for (let node = heading.nextElementSibling; node && node.tagName !== 'H3'; node = node.nextElementSibling) nodes.push(node);
      return {
        steps: nodes.filter(node => node.tagName === 'P' && /^GPT-2 step \d+ · /.test(node.firstElementChild?.textContent ?? ''))
          .map(node => ({number: Number(node.firstElementChild.textContent.match(/^GPT-2 step (\d+)/)[1]), arrow: [...node.querySelectorAll('.katex-html')].some(math => math.textContent.includes('⟶'))})),
        params: nodes.find(node => node.matches('.table-scroll') && node.previousElementSibling?.id === 'gpt2-params')?.textContent ?? '',
      };
    });
    assert.deepEqual(gpt2.steps.map(step => step.number), Array.from({length: config.gpt2.steps}, (_, i) => i + 1), `${config.name}: GPT-2 steps 1..${config.gpt2.steps}, in order`);
    assert.ok(gpt2.steps.every(step => step.arrow), `${config.name}: every GPT-2 step states its input ⟶ output`);
    assert.ok(gpt2.params.includes(config.gpt2.total), `${config.name}: the GPT-2 parameter table totals ${config.gpt2.total}`);
  }
  const math = await mathAudit(chapter);
  assert.ok(math.display >= config.display, `${config.name}: display equations rendered (found ${math.display})`);
  assert.ok(math.inline >= config.inline, `${config.name}: inline math rendered (found ${math.inline})`);
  assert.equal(math.errors, 0, `${config.name}: no KaTeX render errors`);
  assert.equal(math.overflowing, 0, `${config.name}: wide equations scroll inside their block`);
  assert.deepEqual(await chapter.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 1).map(node => node.textContent.slice(0, 40))), [], `${config.name}: every equation fits the desktop column without panning`);
  // Every table that follows a named anchor must read on a desktop column without panning.
  assert.deepEqual(await chapter.locator('.table-scroll').evaluateAll(nodes => nodes.filter(node => node.previousElementSibling?.matches('a[id]') && node.scrollWidth > node.clientWidth + 1).map(node => node.previousElementSibling.id)), [], `${config.name}: anchored tables fit the desktop column`);
  const table = chapter.locator(`[id="${config.shapes.anchor}"] ~ .table-scroll`).first();
  assert.equal(await table.locator('tbody tr').count(), config.shapes.rows, `${config.name}: the shape table lists every step`);
  assert.ok((await table.textContent()).includes(config.shapes.contains), `${config.name}: shape table contains ${config.shapes.contains}`);

  if (config.models) {
    // The model comparison: chronological, every model linked to its source, readable without panning.
    const models = chapter.locator(`[id="${config.models.anchor}"] + .table-scroll`);
    const rows = await models.locator('tbody tr').evaluateAll(nodes => nodes.map(row => ({name: row.cells[0].textContent.trim(), year: Number(row.cells[0].textContent.match(/(20\d\d)\s*$/)?.[1]), link: row.cells[0].querySelector('a[href^="https://"]') !== null, cells: row.cells.length})));
    assert.equal(rows.length, config.models.rows, `${config.name}: model line rows`);
    assert.ok(rows[0].name.startsWith(config.models.first) && rows.at(-1).name.startsWith(config.models.last), `${config.name}: the line runs from ${config.models.first} to ${config.models.last}`);
    assert.ok(rows.every(row => row.link && row.cells === 5 && row.year >= 2021), `${config.name}: every model has a year and a source link`);
    assert.deepEqual(rows.map(row => row.year), rows.map(row => row.year).sort((a, b) => a - b), `${config.name}: models are in chronological order`);
    assert.deepEqual(await models.evaluate(node => [node.scrollWidth <= node.clientWidth + 1, getComputedStyle(node.querySelector('td')).whiteSpace]), [true, 'normal'], `${config.name}: the model line wraps instead of panning on desktop`);
    assert.equal(await chapter.locator(`[id="${config.models.anchor}"]`).evaluate(node => { let list = node.previousElementSibling; while (list && list.tagName !== 'OL') list = list.previousElementSibling; return list?.children.length; }), config.models.ways, `${config.name}: the ways for the image to reach the LLM`);
  }

  let review;
  if (config.code === false) {
    // The code is read in VS Code: no Python, shell or recorded program output anywhere in the chapter.
    const code = await chapter.locator('.code-block').evaluateAll(nodes => nodes.map(node => node.querySelector('.code-lang')?.textContent.trim() ?? ''));
    assert.deepEqual(code, [], `${config.name}: no code blocks in the chapter`);
  } else {
  // Code: the regions of the runnable file are embedded once each, in order, and copy intact.
  const blocks = await chapter.locator('.code-block:has(code.language-python)').all();
  const blockTexts = await Promise.all(blocks.map(block => block.locator('code').textContent()));
  let previous = -1;
  for (const region of regions(config)) {
    const matches = blockTexts.flatMap((text, i) => text.trim() === region.code ? [i] : []);
    assert.equal(matches.length, 1, `${region.name}: exactly one complete source-matched embedded region`);
    const block = blocks[matches[0]];
    const home = config.extras?.[region.name];
    if (home) {
      // A region that belongs to a later section of the chapter, outside the main listing.
      assert.equal(await block.evaluate(node => { let heading = node.previousElementSibling; while (heading && heading.tagName !== 'H3') heading = heading.previousElementSibling; return heading?.id; }), home, `${region.name}: embedded in its own section`);
    } else {
      assert.ok(matches[0] > previous, `${region.name}: embedded in file order`);
      previous = matches[0];
    }
    await block.locator('.copy-code').click();
    assert.equal(await page.evaluate(() => window.__transformerCopied), blockTexts[matches[0]], `${region.name}: full clipboard payload`);
    if (region.name === config.review) review = block;
  }
  assert.ok(review, `${config.name}: implementation available for visual review`);
  const output = chapter.locator(`[id="${config.codeSection}"] ~ .code-block:has(.code-lang:text-is("text"))`).first();
  assert.match(await output.locator('code').textContent(), config.output, `${config.name}: the recorded demo output`);
  }
  if (config.rope) {
    // The RoPE section: where it happens and the notation come before the rule, then the three cases.
    const section = await chapter.locator(`[id="${config.rope.section}"]`).evaluate(heading => {
      const nodes = [];
      for (let node = heading.nextElementSibling; node && node.tagName !== 'H3'; node = node.nextElementSibling) nodes.push(node);
      return {
        leads: nodes.filter(node => node.tagName === 'P' && node.firstElementChild?.tagName === 'STRONG' && node.firstChild === node.firstElementChild).map(node => node.firstElementChild.textContent),
        tables: nodes.filter(node => node.matches('.table-scroll')).map(node => ({anchor: node.previousElementSibling?.id ?? '', fits: node.scrollWidth <= node.clientWidth + 1, rows: node.querySelectorAll('tbody tr').length})),
        wideEquations: nodes.flatMap(node => [...node.querySelectorAll('.katex-display')]).filter(block => block.scrollWidth > block.clientWidth + 1).map(block => block.textContent.slice(0, 40)),
      };
    });
    assert.deepEqual(section.wideEquations, [], 'RoPE: every equation fits the desktop column without panning');
    assert.deepEqual(section.leads.filter(lead => config.rope.leads.includes(lead)), config.rope.leads, 'RoPE: place and notation first, then the rule, then 1D, 2D and 3D');
    assert.equal(section.tables.length, config.rope.tables, 'RoPE: tables');
    assert.ok(section.tables.every(table => table.anchor.startsWith('rope-') && table.fits), `RoPE: every table reads without panning: ${JSON.stringify(section.tables.filter(table => !table.fits))}`);
    assert.equal(section.tables.find(table => table.anchor === 'rope-notation').rows, config.rope.symbols, 'RoPE: every symbol is defined');
    assert.equal(section.tables.find(table => table.anchor === 'rope-compare').rows, 5, 'RoPE: comparison with added positions');
  }

  await capture(page, chapter.locator('.prose'), preview, `${config.shots}-math-desktop.png`, 1800);
  if (review) await capture(page, review, preview, `${config.shots}-code-desktop.png`);
  return {config, chapter, review, math};
}

export async function checkTransformer({page, context, target, preview}) {
  fs.mkdirSync(preview, {recursive: true});
  const errors = [];
  const onError = error => errors.push(error.message);
  page.on('pageerror', onError);
  try {
    await page.setViewportSize({width: 1440, height: 1000});
    await page.goto(target('02_ml_domain.html'));
    await page.waitForFunction(() => !!window.STUDY_BOOK);
    await checkBookLayout(page, target);
    assert.equal(await page.locator(`#chapter-${transformerId}`).evaluate(node => node.open), true, 'The Transformer chapter opens by default');
    assert.equal(await page.locator(`.book-move-note a[href="#${transformerId}"], .book-move-note a[href="#${interviewId}"]`).count(), 2, 'The start note links to the walkthrough and to the interview questions');
    await openContent(page);
    await checkPageLinks(page);
    await noOverflow(page, 'ML desktop page must not overflow');

    // Equations are static KaTeX HTML rendered at build time; the stylesheet and fonts are local.
    assert.equal(await page.locator('.katex-error').count(), 0, 'No KaTeX render errors anywhere on the page');
    assert.equal(await page.locator('script[src*="katex" i]').count(), 0, 'Math must not need a runtime script');
    assert.equal(await page.locator('link[rel="stylesheet"][href="assets/katex/katex.min.css"]').count(), 1, 'Offline KaTeX stylesheet');
    const fonts = await page.evaluate(async () => {
      await document.fonts.ready;
      return [...document.fonts].map(font => `${font.family.replace(/["']/g, '')}:${font.status}`).filter(font => font.startsWith('KaTeX'));
    });
    assert.ok(fonts.some(font => /^KaTeX_(Main|Math):loaded$/.test(font)), `KaTeX fonts load from assets without a network: ${fonts.join(', ') || 'none declared'}`);

    await page.evaluate(() => Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: async text => { window.__transformerCopied = text; }}}));
    const questions = await checkInterview(page);
    const results = [];
    for (const config of walkthroughs) results.push(await checkWalkthrough(page, config, preview));
    const visionSection = await checkVision(page, preview);

    await page.locator('html').evaluate(node => { node.dataset.theme = 'dark'; });
    for (const {config, chapter} of results) await capture(page, chapter.locator('.prose'), preview, `${config.shots}-math-dark.png`, 1200);
    await noOverflow(page, 'ML dark page must not overflow');
    await page.locator('html').evaluate(node => { node.dataset.theme = 'light'; });

    await page.emulateMedia({media: 'print'});
    for (const {config, chapter, review} of results) {
      assert.ok(await chapter.locator('.katex-display').first().isVisible(), `${config.name}: equations print`);
      if (review) assert.ok(await review.isVisible(), `${config.name}: code prints`);
    }
    await page.emulateMedia({media: 'screen'});

    await page.setViewportSize({width: 390, height: 844});
    await noOverflow(page, 'ML page fits a 390px mobile viewport');
    await page.locator('[id="26-interview-questions--computer-vision"]').evaluate(node => { document.documentElement.style.scrollBehavior = 'auto'; node.scrollIntoView({block: 'start'}); });
    await page.screenshot({path: path.join(preview, 'cv-section-mobile.png'), animations: 'disabled'});
    await page.evaluate(() => { document.documentElement.style.scrollBehavior = ''; });
    for (const {config, chapter, review, math} of results) {
      assert.equal((await mathAudit(chapter)).display, math.display, `${config.name}: every equation survives the mobile layout`);
      await capture(page, chapter.locator('.prose'), preview, `${config.shots}-math-mobile.png`, 2400);
      if (review) await capture(page, review, preview, `${config.shots}-code-mobile.png`);
    }

    for (const file of [...walkthroughs.flatMap(config => [config.source, config.tests]), vision.source, vision.tests]) {
      const viewer = `source-${file.replace(/\.py$/, '.html')}`;
      assert.ok(fs.existsSync(path.join(root, viewer)), `${viewer}: generated source viewer`);
      await page.goto(target(viewer));
      assert.equal(await page.locator(`.source-code a[download][href="${file}"]`).count(), 1, `${viewer}: source toolbar downloads the exact source file`);
      const downloadLinks = await page.locator('a[download]').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
      assert.deepEqual(downloadLinks, [file, file], `${viewer}: header and source toolbar both download the exact source file`);
      const source = fs.readFileSync(path.join(root, file), 'utf8').split('\n');
      const displayed = await page.locator('.source-line').evaluateAll(nodes => nodes.map(node => {
        const clone = node.cloneNode(true); clone.querySelector('.line-number')?.remove(); return clone.textContent;
      }));
      assert.deepEqual(displayed, source.map(line => line || ' '), `${viewer}: every source line matches`);
      await noOverflow(page, `${viewer}: mobile source viewer must not overflow`);
    }

    const staticContext = await context.browser().newContext({javaScriptEnabled: false, viewport: {width: 390, height: 844}});
    try {
      await staticContext.route('**/*', route => /^https?:/.test(route.request().url()) ? route.abort() : route.continue());
      const staticPage = await staticContext.newPage();
      await staticPage.goto(target('02_ml_domain.html'));
      assert.equal(await staticPage.locator(`#chapter-${transformerId}`).evaluate(node => node.open), true, 'No-JS readers land on the open Transformer chapter');
      assert.ok(await staticPage.locator(`#chapter-${transformerId} .katex-display`).first().isVisible(), 'No-JS reading shows the equations');
      for (const {config, math} of results) {
        const staticMath = await mathAudit(staticPage.locator(`#chapter-${config.chapterId}`));
        assert.deepEqual([staticMath.display, staticMath.inline, staticMath.errors], [math.display, math.inline, 0], `${config.name}: math is static HTML, identical without JavaScript`);
      }
      await noOverflow(staticPage, 'No-JS ML page must fit mobile');
    } finally {
      await staticContext.close();
    }
    assert.deepEqual(errors, [], 'No ML page errors');
    console.log(`PASS ML layout: fourteen chapters, interview questions as chapter 2 (${questions} answered questions in ${interview.sections.length} groups), generative models as chapter 6; old bookmarks and reading progress migrate across four layouts.`);
    console.log(`PASS Vision transformers: ${vision.leads.length} variants from ViT to DiT (${visionSection.displays} display equations), cost table, ${vision.models.rows}-model line, no embedded code (${vision.source} is read in VS Code), two source viewers.`);
    for (const {config, math} of results) console.log(`PASS ${config.name}: one ${config.steps ?? 11}-step walkthrough (${math.display} display + ${math.inline} inline KaTeX equations, offline fonts), shape table, ${config.code === false ? 'no embedded code (read in VS Code), a decoder-only GPT-2 walkthrough' : `${config.regions.length} embedded code regions and copies`}, two source viewers, desktop/mobile, dark, print, no-JS, links and IDs.`);
  } finally {
    page.off('pageerror', onError);
    await page.setViewportSize({width: 1440, height: 1000});
  }
}

async function main() {
  const preview = path.join(root, '.preview-check');
  const macChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  const executablePath = process.env.TRANSFORMER_TEST_CHROME || (fs.existsSync(macChrome) ? macChrome : undefined);
  const browser = await chromium.launch({headless: true, ...(executablePath ? {executablePath} : {})});
  try {
    const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
    await context.route('**/*', route => /^https?:/.test(route.request().url()) ? route.abort() : route.continue());
    const page = await context.newPage();
    const target = name => pathToFileURL(path.join(root, name)).href;
    await checkTransformer({page, context, target, preview});
  } finally {
    await browser.close();
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
