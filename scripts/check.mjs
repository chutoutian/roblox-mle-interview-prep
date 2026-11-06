import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { checkNetworking } from './check-network.mjs';
import { checkLeetCode } from './check-leetcode.mjs';
import { checkJpeg } from './check-jpeg.mjs';
import { checkInterviewPrep } from './check-interview-prep.mjs';
import { checkTransformer } from './check-transformer.mjs';
import { checkReviewChecklists } from './check-review-checklists.mjs';
import { execFileSync } from 'node:child_process';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
execFileSync('python3',[path.join(root,'scripts/check-binary-search.py')],{cwd:root,stdio:'inherit',timeout:30000});
execFileSync('python3',[path.join(root,'scripts/check-linked-list.py')],{cwd:root,stdio:'inherit',timeout:30000});
execFileSync('python3',[path.join(root,'scripts/check-tree-dfs.py')],{cwd:root,stdio:'inherit',timeout:30000});
execFileSync('python3',[path.join(root,'scripts/check-backtracking.py')],{cwd:root,stdio:'inherit',timeout:30000});
execFileSync('python3',[path.join(root,'scripts/check-dijkstra.py')],{cwd:root,stdio:'inherit',timeout:30000});
execFileSync('python3',[path.join(root,'scripts/check-dfs-memo.py')],{cwd:root,stdio:'inherit',timeout:30000});
execFileSync('python3',[path.join(root,'scripts/check-company-interviews.py')],{cwd:root,stdio:'inherit',timeout:120000});
const preview=path.join(root,'.preview-check');fs.mkdirSync(preview,{recursive:true});
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
const context=await browser.newContext({viewport:{width:1440,height:1000}});
await context.route('**/*',route=>/^https?:/.test(route.request().url())?route.abort():route.continue());
const page=await context.newPage();
const errors=[];page.on('pageerror',error=>errors.push(error.message));
const target=name=>pathToFileURL(path.join(root,name)).href;
const pages=fs.readdirSync(root).filter(name=>name.endsWith('.html'));
const cache=new Map(), broken=[];
try {
  for(const name of pages) {
    await page.goto(target(name));
    await page.waitForFunction(()=>!!window.STUDY_BOOK);
    const audit=await page.evaluate(()=>({
      ids:[...document.querySelectorAll('[id]')].map(x=>x.id),
      hrefs:[...document.querySelectorAll('a[href]')].map(x=>x.getAttribute('href')),
      overflow:document.documentElement.scrollWidth>innerWidth+1,
      language:document.documentElement.lang,
      hasChinese:/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/u.test(document.documentElement.outerHTML),
      details:document.querySelectorAll('details.chapter').length
    }));
    cache.set(name,audit);
    const duplicates=audit.ids.filter((id,i,a)=>a.indexOf(id)!==i);
    assert.deepEqual(duplicates,[],`${name}: duplicate IDs`);
    assert.equal(audit.overflow,false,`${name}: desktop horizontal overflow`);
    assert.equal(audit.language,'en',`${name}: document language must be English`);
    assert.equal(audit.hasChinese,false,`${name}: untranslated Chinese text`);
  }
  for(const [name,audit] of cache) for(const href of audit.hrefs) {
    if(!href||/^[a-z]+:/i.test(href))continue;
    const [rawFile,rawHash]=href.split('#');
    const file=decodeURIComponent(rawFile||name), hash=rawHash?decodeURIComponent(rawHash):null;
    if(!fs.existsSync(path.join(root,file))) {broken.push(`${name} → missing ${href}`);continue;}
    if(hash&&cache.has(file)&&!cache.get(file).ids.includes(hash))broken.push(`${name} → missing # ${href}`);
  }
  assert.deepEqual(broken,[],'Internal links must resolve');
  for(const entry of await page.evaluate(()=>window.STUDY_BOOK.search)) {
    const [file,hash]=entry.url.split('#');
    assert.ok(cache.get(file)?.ids.includes(hash),`Search destination must exist: ${entry.url}`);
  }
  console.log(`PASS ${pages.length} English-only offline pages: no untranslated Chinese, JS errors, overflow, duplicate IDs or broken internal links.`);
  await checkTransformer({page,context,target,preview});
  await checkReviewChecklists({page,target});
  await page.goto(target('01_video_domain.html'));
  const videoMeta=await page.evaluate(()=>window.STUDY_BOOK.docs.video);
  assert.equal(videoMeta.chapters.length,3,'Video book keeps vision, image codecs and video codecs; 360/VR, hardware, the drill and the reverse index were removed');
  assert.equal(videoMeta.drills,15,'The video book retains fifteen exercises after removing the VBV simulator');
  assert.deepEqual(videoMeta.chapters.slice(0,3).map(c=>c.title),['1. Human Vision & Color','2. Image Codecs','3. Video Codecs']);
  const videoChapter=page.locator('#chapter-3-video-codecs');
  assert.deepEqual(await videoChapter.locator(':scope > .chapter-body > .prose > h3').evaluateAll(nodes=>nodes.map(node=>node.id)),[
    '31-the-encode-pipeline',
    '32-rate-control-cbr-vbr-cqp-and-buffers',
    '33-frame-types-gop-and-reference-structures',
    '34-intra-prediction',
    '35-inter-prediction-and-motion-estimation',
    '36-block-partitioning-across-standards',
    '37-residual-transforms-and-quantization',
    '38-entropy-coding-and-cabac',
    '39-rdo--rate-distortion-optimization',
    '310-compression-artifacts-and-in-loop-filters'
  ],'Rate control follows the encode pipeline and later sections are numbered consecutively');
  assert.ok(!(await videoChapter.locator('h3,h4,h5,h6').allTextContents()).some(title=>/Scalability|VBV buffer simulator/i.test(title)),'Removed scalability and VBV simulator headings must not be rendered');
  assert.equal(await page.locator('[id="14-color-primaries-transfer-function-matrix"]').count(),0,'Removed color-tags section must not be rendered');
  assert.ok(!(await page.locator('article').textContent()).includes('The Universal Compression Pipeline'),'The standalone pipeline chapter must be removed');
  for(const [id,chapter] of [
    ['coding-2-d-dct--idct-ready','2-image-codecs'],
    ['coding-quantization--dequantization-ready','2-image-codecs'],
    ['coding-huffman-encode--decode-ready','2-image-codecs'],
    ['coding-jpeg-like-block-pipeline-ready','2-image-codecs'],
    ['coding-integer-arithmetic-coder-ready','3-video-codecs'],
    ['coding-closed-loop-ip-encoder-toy','3-video-codecs']
  ]) assert.equal(await page.locator(`[id="${id}"]`).evaluate(el=>el.closest('[data-chapter]')?.dataset.chapter),chapter,`${id}: wrong chapter`);
  assert.ok((await page.locator('#chapter-3-video-codecs').textContent()).includes('CABAC'));
  for(const [alias,destination] of [
    ['311-rate-control-modes-and-the-vbv-model','32-rate-control-cbr-vbr-cqp-and-buffers'],
    ['coding-vbv-buffer-simulator-ready','32-rate-control-cbr-vbr-cqp-and-buffers'],
    ['32-frame-types-gop-and-reference-structures','33-frame-types-gop-and-reference-structures'],
    ['310-scalability-svc-simulcast-and-layered-coding','3-video-codecs']
  ]) assert.equal(videoMeta.legacyAliases[alias],destination,`${alias}: retain the previous video bookmark`);
  for(const [alias,destination] of Object.entries(videoMeta.legacyAliases)) {
    assert.equal(await page.locator(`[id="${alias}"]`).evaluate(el=>el.parentElement.closest('[id]')?.id),destination,`${alias}: wrong legacy destination`);
  }
  for(const alias of ['4-video-coding-core','31-jpeg','46-rdo--rate-distortion-optimization','311-rate-control-modes-and-the-vbv-model','coding-vbv-buffer-simulator-ready','32-frame-types-gop-and-reference-structures','310-scalability-svc-simulcast-and-layered-coding']) {
    await page.goto(target('01_video_domain.html')+'#'+alias);
    assert.equal(await page.locator(`[id="${alias}"]`).evaluate(el=>{
      const chapter=el.closest('[data-chapter]');
      return !!chapter?.open;
    }),true,`${alias}: old bookmarks must reveal the new chapter`);
  }
  const savedProgress=await page.evaluate(()=>localStorage.getItem('study-progress-v1'));
  try {
    await page.evaluate(()=>localStorage.setItem('study-progress-v1',JSON.stringify({
      video:{done:['2-the-universal-compression-pipeline','3-image-codecs','4-video-coding-core','10-video-quality-assessment'],last:'4-video-coding-core'},
      ml:{done:['1-neural-net-basics']}
    })));
    await page.reload();
    const migrated=await page.evaluate(()=>JSON.parse(localStorage.getItem('study-progress-v1')));
    for(const chapter of ['2-image-codecs','3-video-codecs']) assert.ok(migrated.video.done.includes(chapter));
    await page.goto(target('01_video_domain.html')+'#10-video-quality-assessment');
    await page.waitForURL(url=>url.pathname.endsWith('/04_resume_drill.html')&&url.hash==='#6-tiktok--ugc-short-video-quality-assessment-via-mllm');
    await page.goto(target('01_video_domain.html'));
    assert.equal(migrated.video.last,'3-video-codecs');
    assert.deepEqual(migrated.ml.done,['1-neural-net-basics'],'Migration must leave other books untouched');
    await page.evaluate(()=>localStorage.setItem('study-progress-v1',JSON.stringify({video:{done:['3-image-codecs','4-video-coding-core'],last:'3-image-codecs'}})));
    await page.reload();
    const partial=await page.evaluate(()=>JSON.parse(localStorage.getItem('study-progress-v1')).video);
    assert.ok(!partial.done.includes('2-image-codecs')&&!partial.done.includes('3-video-codecs'),'Partially read merged chapters must not become complete');
    assert.equal(partial.last,'2-image-codecs');
    await page.locator('[data-resume]').click();
    await page.waitForURL(/#2-image-codecs$/);
    assert.equal(await page.locator('#chapter-2-image-codecs').evaluate(el=>el.open),true);
    await page.locator('#chapter-2-image-codecs [data-mark]').click();
    await page.reload();
    assert.equal(await page.locator('#chapter-2-image-codecs [data-mark]').getAttribute('aria-pressed'),'true','Migration must not overwrite newer reading progress');
  } finally {
    await page.evaluate(saved=>saved===null?localStorage.removeItem('study-progress-v1'):localStorage.setItem('study-progress-v1',saved),savedProgress);
    await page.goto(target('01_video_domain.html'));
  }
  console.log('PASS reorganized codec chapters, retained exercises, legacy bookmarks, and reading-progress migration.');
  await checkJpeg({page,context,target,preview});
  await checkNetworking({page,context,target,preview,cache});
  await checkLeetCode({page,context,target,preview});
  await checkInterviewPrep({page,context,target,preview});
  await page.goto(target('01_video_domain.html'));
  await page.locator('[data-collapse]').click();
  assert.equal(await page.locator('article details[open]').count(),0);
  await page.locator('[data-expand]').click();
  assert.equal(await page.locator('article details:not([open])').count(),0);
  await page.locator('[data-collapse]').click();
  const destination=await page.locator('[data-toc]').nth(2).getAttribute('href');   // the last of the three video chapters
  await page.locator('[data-toc]').nth(2).click();
  const id=decodeURIComponent(destination.slice(1));
  assert.equal(await page.locator('[data-chapter]').evaluateAll((nodes,id)=>nodes.find(n=>n.dataset.chapter===id)?.open,id),true);
  const done=page.locator(`[data-chapter="${id}"] [data-mark]`);
  await done.click();assert.equal(await done.getAttribute('aria-pressed'),'true');
  await page.reload();assert.equal(await done.getAttribute('aria-pressed'),'true');
  await page.locator('[data-theme-toggle]').click();
  assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  await page.reload();assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  await page.locator('[data-theme-toggle]').click();
  await page.locator('[data-font-size]').click();
  await page.locator('[data-font-size]').click();
  assert.equal(await page.locator('html').evaluate(el=>el.style.getPropertyValue('--reader-font-size')),'17px');
  await page.reload();
  assert.equal(await page.locator('html').evaluate(el=>el.style.getPropertyValue('--reader-font-size')),'17px');
  await page.locator('[data-font-size]').click();
  await page.locator('[data-focus]').click();assert.equal(await page.locator('body').evaluate(e=>e.classList.contains('focus-mode')),true);
  await page.locator('[data-focus]').click();
  await page.locator('[data-open-search]').first().click();
  await page.locator('#search-input').fill('KV cache');
  await page.waitForFunction(()=>document.querySelector('.search-status').textContent.startsWith('Found'));
  assert.ok(await page.locator('.search-result').count()>0);
  await page.locator('[data-scope="ml"]').click();
  assert.ok((await page.locator('.result-path').allTextContents()).every(t=>t==='ML & LLMs'));
  await page.locator('#search-input').press('ArrowDown');
  assert.equal(await page.locator('#search-input').getAttribute('aria-activedescendant'),'search-result-0');
  assert.equal(await page.locator('#search-result-0').getAttribute('aria-selected'),'true');
  await page.locator('#search-input').press('Enter');
  await page.waitForURL(/02_ml_domain\.html#/);
  assert.equal(await page.evaluate(()=>{
    let node=document.getElementById(decodeURIComponent(location.hash.slice(1)));
    for(;node;node=node.parentElement) if(node.tagName==='DETAILS'&&!node.open)return false;
    return true;
  }),true);
  await page.locator('[data-open-search]').first().click();
  await page.locator('#search-input').fill('zzzzzzzz-no-such-term');
  await page.waitForFunction(()=>document.querySelector('.search-status').textContent.includes('Found 0'));
  assert.equal(await page.locator('.search-result').count(),0);
  await page.locator('[data-close-search]').click();
  await page.goto(target('02_ml_domain.html')+'#ml-coding-companion');
  await page.locator('a[href="#coding-linear-backward"]').first().click();
  assert.equal(await page.locator('#coding-linear-backward').evaluate(a=>{
    const marker=a.parentElement.tagName==='P'?a.parentElement:a;
    return marker.nextElementSibling?.open;
  }),true,'Legacy coding links must also open their exercise');
  await page.locator('[data-open-search]').first().click();
  await page.locator('#search-input').fill('dependency chain');
  await page.waitForFunction(()=>document.querySelector('.result-title')?.textContent.includes('Reading guide'));
  await page.locator('.search-result').first().click();
  assert.equal(await page.locator('#reading-overview').evaluate(d=>d.open),true,'Search must include and reveal introductory notes');
  await page.locator('[data-expand]').click();
  // Stub only the OS clipboard sink, then verify the exact snippet sent to it.
  await page.evaluate(()=>Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{window.__copiedCode=text;}}}));
  const copy=page.locator('.copy-code').first();
  const expected=await copy.locator('xpath=ancestor::div[contains(@class,"code-block")]//code').textContent();
  await copy.click();assert.equal(await page.evaluate(()=>window.__copiedCode),expected);
  const quiz=page.locator('.drill-check input').first();
  if(await quiz.count()){await quiz.check();await page.reload();assert.equal(await quiz.isChecked(),true);}
  console.log('PASS collapse, expand, deep links, completion persistence, themes, font sizes, focus mode, search, clipboard payload and self-test state.');

  // The bilinear 8-to-2 diagram must teach the spatial mapping, not merely
  // decorate the exercise. Verify its data model, all four states, native
  // keyboard activation, responsive composition, print visibility, and theme.
  await page.goto(target('01_video_domain.html'));
  const bilinear=page.locator('figure[data-bilinear-diagram]');
  assert.equal(await bilinear.count(),1,'There must be exactly one bilinear diagram');
  assert.equal(await bilinear.getAttribute('id'),'bilinear-8-to-2');
  const bilinearDrill=bilinear.locator('xpath=ancestor::details[contains(concat(" ", normalize-space(@class), " "), " coding-drill ")][1]');
  assert.equal(await bilinearDrill.count(),1,'The bilinear diagram must be inside a coding drill');
  assert.match(await bilinearDrill.locator('summary').first().textContent(),/bilinear/i,'The diagram must belong to the bilinear coding drill');
  await bilinear.evaluate(node=>{
    for(let parent=node.parentElement;parent;parent=parent.parentElement) {
      if(parent.tagName==='DETAILS') parent.open=true;
    }
  });

  const source=bilinear.locator('[data-bilinear-source]');
  const output=bilinear.locator('[data-bilinear-output]');
  assert.equal(await source.count(),1,'The diagram needs one 8x8 source SVG');
  assert.equal(await output.count(),1,'The diagram needs one 2x2 output control');
  const sourceCells=source.locator('[data-source-row][data-source-col]');
  assert.equal(await sourceCells.count(),64,'The source SVG must expose all 64 cells');
  const sourceGrid=await sourceCells.evaluateAll(cells=>cells.map(cell=>{
    const localText=cell.matches('text')?cell.textContent:cell.querySelector('text')?.textContent;
    const siblingText=[...(cell.parentElement?.children||[])].find(child=>child.tagName?.toLowerCase()==='text')?.textContent;
    const evidence=[cell.dataset.value,localText,siblingText,cell.getAttribute('aria-label')].filter(Boolean).join(' ');
    return {
      row:Number(cell.dataset.sourceRow),
      col:Number(cell.dataset.sourceCol),
      numbers:[...evidence.matchAll(/-?\d+(?:\.\d+)?/g)].map(match=>Number(match[0]))
    };
  }));
  const sourceByCoordinate=new Map(sourceGrid.map(cell=>[`${cell.row},${cell.col}`,cell]));
  assert.equal(sourceByCoordinate.size,64,'Every source coordinate must be unique');
  for(let row=0;row<8;row++) for(let col=0;col<8;col++) {
    const cell=sourceByCoordinate.get(`${row},${col}`);
    assert.ok(cell,`Missing source cell (${row},${col})`);
    assert.ok(cell.numbers.includes(row*8+col),`Source cell (${row},${col}) must display ${row*8+col}`);
  }

  const bilinearCases=[
    {index:'0',letter:'A',coordinate:'(1.5,1.5)',rows:[1,2],cols:[1,2],values:[9,10,17,18],output:'13.5'},
    {index:'1',letter:'B',coordinate:'(1.5,5.5)',rows:[1,2],cols:[5,6],values:[13,14,21,22],output:'17.5'},
    {index:'2',letter:'C',coordinate:'(5.5,1.5)',rows:[5,6],cols:[1,2],values:[41,42,49,50],output:'45.5'},
    {index:'3',letter:'D',coordinate:'(5.5,5.5)',rows:[5,6],cols:[5,6],values:[45,46,53,54],output:'49.5'}
  ];
  const bilinearButtons=bilinear.locator('button[data-bilinear-pixel]');
  assert.equal(await bilinearButtons.count(),4,'The output must contain four native buttons');
  assert.equal(await bilinear.locator('[data-bilinear-panel]').count(),4,'The diagram must contain four explanation panels');
  assert.equal(await bilinear.locator('[data-bilinear-group]').count(),4,'The source SVG must contain four selectable groups');
  assert.equal(await source.locator('[data-source-row][data-source-col].is-active').count(),4,'Exactly four source cells must be selected initially');
  assert.equal((await bilinearButtons.evaluateAll(nodes=>nodes.filter(node=>node.getAttribute('aria-pressed')==='true').length)),1,'Exactly one output button must be selected initially');

  const containsNumericToken=(text,value)=>new RegExp(`(^|[^\\d.])${String(value).replace('.', '\\.')}(?=$|[^\\d.])`).test(text);
  const assertBilinearSelection=async item=>{
    const button=bilinear.locator(`button[data-bilinear-pixel="${item.index}"]`);
    await button.click();
    assert.equal(await button.getAttribute('aria-pressed'),'true',`${item.letter} must expose its selected state`);
    assert.match((await button.textContent()).replace(/\s+/g,' '),new RegExp(`${item.letter}.*${item.output.replace('.', '\\.')}`),`${item.letter} must display output ${item.output}`);
    for(const other of bilinearCases) {
      const expected=other.index===item.index;
      assert.equal(await bilinear.locator(`button[data-bilinear-pixel="${other.index}"]`).getAttribute('aria-pressed'),String(expected),`Only ${item.letter} may be pressed`);
      assert.equal(await bilinear.locator(`[data-bilinear-panel="${other.index}"]`).evaluate(node=>node.hidden),!expected,`Only panel ${item.letter} may be visible`);
      assert.equal(await bilinear.locator(`[data-bilinear-group="${other.index}"]`).evaluate(node=>node.classList.contains('is-active')),expected,`Only source group ${item.letter} may be active`);
    }
    const expectedCoordinates=item.rows.flatMap(row=>item.cols.map(col=>`${row},${col}`)).sort();
    const activeCoordinates=(await source.locator('[data-source-row][data-source-col].is-active').evaluateAll(cells=>cells.map(cell=>`${cell.dataset.sourceRow},${cell.dataset.sourceCol}`))).sort();
    assert.deepEqual(activeCoordinates,expectedCoordinates,`${item.letter} must highlight the correct 2x2 source neighborhood`);
    const panel=bilinear.locator(`[data-bilinear-panel="${item.index}"]`);
    const panelLabel=(await panel.getAttribute('aria-label'))?.trim();
    assert.ok(panelLabel,`${item.letter} panel needs an aria-label`);
    assert.equal((await bilinear.locator('[data-bilinear-description]').textContent()).trim(),panelLabel,`The SVG description must announce panel ${item.letter}`);
    const panelText=(await panel.textContent()).replace(/\s+/g,' ').trim();
    const compactPanelText=panelText.replace(/\s+/g,'');
    assert.ok(compactPanelText.includes(item.coordinate),`${item.letter} panel must show mapped coordinate ${item.coordinate}`);
    assert.ok(panelText.includes('0.25'),`${item.letter} panel must show the 0.25 bilinear weights`);
    for(const value of [...item.values,Number(item.output)]) {
      assert.ok(containsNumericToken(panelText,value),`${item.letter} formula must include numeric value ${value}`);
    }
  };
  for(const item of bilinearCases) await assertBilinearSelection(item);

  const keyboardEnter=bilinear.locator('button[data-bilinear-pixel="0"]');
  await keyboardEnter.focus();
  await keyboardEnter.press('Enter');
  assert.equal(await keyboardEnter.getAttribute('aria-pressed'),'true','Enter must activate a bilinear output button');
  const keyboardSpace=bilinear.locator('button[data-bilinear-pixel="3"]');
  await keyboardSpace.focus();
  await keyboardSpace.press('Space');
  assert.equal(await keyboardSpace.getAttribute('aria-pressed'),'true','Space must activate a bilinear output button');

  await page.setViewportSize({width:1440,height:1000});
  await page.locator('html').evaluate(html=>html.dataset.theme='light');
  await bilinear.locator('button[data-bilinear-pixel="0"]').click();
  const [sourceBox,outputBox]=await Promise.all([source.boundingBox(),output.boundingBox()]);
  assert.ok(sourceBox&&outputBox&&sourceBox.x<outputBox.x,'Desktop diagram must place the 8x8 source to the left of the 2x2 output');
  const componentScreenshotStyle='.topbar,.read-progress,.skip-link,.sidebar,.sidebar-backdrop,.toast{visibility:hidden!important}';
  await bilinear.screenshot({path:path.join(preview,'bilinear-desktop.png'),style:componentScreenshotStyle,animations:'disabled'});
  await page.locator('html').evaluate(html=>html.dataset.theme='dark');
  await bilinear.screenshot({path:path.join(preview,'bilinear-dark.png'),style:componentScreenshotStyle,animations:'disabled'});

  await page.setViewportSize({width:390,height:844});
  await page.locator('html').evaluate(html=>html.dataset.theme='light');
  assert.equal(await bilinear.evaluate(node=>node.scrollWidth>node.clientWidth+1),false,'Bilinear diagram must not overflow internally at 390px');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,'Bilinear diagram must not make the mobile document overflow');
  await bilinear.screenshot({path:path.join(preview,'bilinear-mobile.png'),style:componentScreenshotStyle,animations:'disabled'});
  await page.emulateMedia({media:'print'});
  assert.deepEqual(await bilinearButtons.evaluateAll(nodes=>nodes.map(node=>getComputedStyle(node).display==='none')),[false,false,false,false],'Print CSS must retain all four bilinear output buttons');
  await page.emulateMedia({media:'screen'});
  console.log('PASS interactive bilinear diagram data, states, keyboard controls, responsive layout, themes and print visibility.');

  await page.setViewportSize({width:1440,height:1000});
  await page.goto(target('index.html'));
  await page.screenshot({path:path.join(preview,'home-desktop.png')});
  await page.locator('.book-row').filter({hasText:'ML & LLMs'}).click();
  await page.waitForURL(/02_ml_domain\.html$/);
  await page.screenshot({path:path.join(preview,'ml-desktop.png')});
  await page.goto(target('01_video_domain.html'));
  await page.screenshot({path:path.join(preview,'video-desktop.png')});
  for(const chapter of ['2-image-codecs','3-video-codecs']) {
    await page.goto(target('01_video_domain.html')+'#'+chapter);
    assert.equal(await page.locator(`[data-chapter="${chapter}"]`).evaluate(el=>el.open),true);
    await page.waitForFunction(id=>{
      const top=document.getElementById(id).getBoundingClientRect().top;
      const headerBottom=document.querySelector('.topbar').getBoundingClientRect().bottom;
      return top>=headerBottom&&top<=headerBottom+26;
    },chapter,{timeout:5000});
    await page.screenshot({path:path.join(preview,chapter+'-desktop.png'),animations:'disabled'});
  }
  await page.setViewportSize({width:390,height:844});
  for(const name of ['index.html','01_video_domain.html','02_ml_domain.html','04_resume_drill.html','05_networking_streaming.html','06_leetcode_notes.html','07_system_design.html','08_company_interviews.html']) {
    await page.goto(target(name));
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`${name}: mobile overflow`);
    await page.locator('[data-expand]').click();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`${name}: expanded mobile overflow`);
  }
  await page.goto(target('index.html'));
  await page.locator('[data-menu]').click();
  assert.equal(await page.locator('[data-menu]').getAttribute('aria-expanded'),'true');
  await page.locator('#sidebar a[href="01_video_domain.html"]').click();
  await page.waitForURL(/01_video_domain\.html$/);
  await page.screenshot({path:path.join(preview,'video-mobile.png')});
  await page.goto(target('index.html'));
  await page.screenshot({path:path.join(preview,'home-mobile.png')});
  assert.deepEqual(errors,[],'Browser JavaScript errors');
  console.log('PASS 390px mobile layouts and navigation. Screenshots saved to .preview-check/.');
} finally {await browser.close();}
