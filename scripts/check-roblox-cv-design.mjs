import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {chromium} from 'playwright';
import {cvDesignCases} from './build-roblox-cv-design.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const directory=path.join(root,'ml_modeling_practice/roblox_cv');
const preview=path.join(root,'.preview-check/cv-design');
fs.mkdirSync(preview,{recursive:true});
const sequence=Array.from({length:21},(_,i)=>i+1);
const pages=[{file:'index.html',mode:'index'},...cvDesignCases.flatMap(item=>['questions','solution'].map(mode=>({file:`${item.slug}_${mode}.html`,mode,slug:item.slug})))];
for(const item of cvDesignCases)for(const mode of ['questions','solution']){
  const file=`${item.slug}_${mode}.md`, source=fs.readFileSync(path.join(directory,file),'utf8');
  assert.deepEqual([...source.matchAll(/\*\*Q(\d+)\./g)].map(match=>Number(match[1])),sequence,`${file}:21 matching follow-ups`);
  assert.deepEqual([...source.matchAll(/^#{2,3} ([1-7])\. /gm)].map(match=>Number(match[1])),[1,2,3,4,5,6,7],`${file}:seven aligned stages`);
  for(const time of ['0–4','4–10','10–15','15–20','20–26','26–31','31–35'])assert.ok(source.includes(time),`${file}: ${time}`);
  assert.equal(/[\u3400-\u9fff]/u.test(source),false,`${file}: English`);
  assert.equal(/\\[\[\]()]/.test(source),false,`${file}: supported math delimiters`);
  if(mode==='solution')assert.ok(source.split(/\s+/).length>=1400,`${file}: substantive worked answer`);
}
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
const context=await browser.newContext({viewport:{width:1440,height:1000}});
await context.route('**/*',route=>/^https?:/.test(route.request().url())?route.abort():route.continue());
const page=await context.newPage();
const errors=[];
page.on('pageerror',error=>errors.push(error.message));
try {
  for(const item of pages){
    await page.goto(pathToFileURL(path.join(directory,item.file)).href);
    await page.evaluate(async()=>{await document.fonts.ready;document.documentElement.style.scrollBehavior='auto';});
    const audit=await page.evaluate(()=>({
      language:document.documentElement.lang,
      chinese:/[\u3400-\u9fff]/u.test(document.body.textContent),
      ids:[...document.querySelectorAll('[id]')].map(node=>node.id),
      links:[...document.querySelectorAll('a[href]')].map(node=>node.getAttribute('href')),
      overflow:document.documentElement.scrollWidth>innerWidth+1,
      math:document.querySelectorAll('.katex').length,
      mathErrors:document.querySelectorAll('.katex-error').length,
      wideTables:[...document.querySelectorAll('.table-scroll')].filter(node=>node.scrollWidth>node.clientWidth+1).length,
    }));
    assert.equal(audit.language,'en');assert.equal(audit.chinese,false);
    assert.equal(new Set(audit.ids).size,audit.ids.length,`${item.file}:unique IDs`);
    assert.equal(audit.overflow,false,`${item.file}:desktop fits`);
    assert.equal(audit.mathErrors,0,`${item.file}:valid math`);
    assert.equal(audit.wideTables,0,`${item.file}:desktop tables wrap without panning`);
    for(const href of audit.links){
      if(!href||/^[a-z]+:/i.test(href))continue;
      const [file,hash]=href.split('#');const target=path.resolve(directory,file||item.file);
      assert.ok(fs.existsSync(target),`${item.file}:missing ${href}`);
      if(hash&&target.endsWith('.html')){
        const html=fs.readFileSync(target,'utf8');
        assert.ok(html.includes(`id="${decodeURIComponent(hash)}"`),`${item.file}:missing anchor ${href}`);
      }
    }
    if(item.mode==='questions'){
      assert.equal(await page.locator('[data-question]').count(),21,`${item.file}:review checkboxes`);
      await page.locator('[data-question="q1"]').check();
      await page.reload();
      assert.equal(await page.locator('[data-question="q1"]').isChecked(),true,`${item.file}:saved review`);
      await page.locator('[data-question="q1"]').uncheck();
      assert.match(await page.locator('.design-review-count').textContent(),/Reviewed 0 of 21/);
    }else if(item.mode==='solution'){
      assert.equal(await page.locator('.followup-answer').count(),21,`${item.file}:21 answer targets`);
      assert.ok(audit.math>=10,`${item.file}:rendered formulas and shapes`);
      assert.ok(await page.locator('.design-diagram').count()>=1,`${item.file}:end-to-end flow`);
    }
    await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));
    if(item.mode!=='questions')await page.screenshot({path:path.join(preview,item.file.replace('.html','-desktop.png')),animations:'disabled'});
    if(item.mode==='solution'){
      const modelHeading=page.locator('article :is(h2,h3)').filter({hasText:/^4\. Model design/}).first();
      await modelHeading.evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
      await page.screenshot({path:path.join(preview,`${item.slug}-model-desktop.png`),animations:'disabled'});
      const lossHeading=page.locator('article :is(h2,h3)').filter({hasText:/^5\. Loss/}).first();
      await lossHeading.evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
      await page.screenshot({path:path.join(preview,`${item.slug}-loss-desktop.png`),animations:'disabled'});
    }
    await page.locator('[data-design-theme]').click();
    const toggled=await page.locator('html').getAttribute('data-theme');
    assert.ok(['light','dark'].includes(toggled));
    await page.locator('[data-design-theme]').click();
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`${item.file}:390px fits`);
    if(item.mode==='questions'||item.mode==='index')await page.screenshot({path:path.join(preview,item.file.replace('.html','-mobile.png')),animations:'disabled'});
    await page.emulateMedia({media:'print'});
    assert.equal(await page.locator('.design-sidebar').isVisible(),false,`${item.file}:print hides navigation`);
    if(item.mode==='solution')assert.equal(await page.locator('.katex').first().isVisible(),true,`${item.file}:math prints`);
    await page.emulateMedia({media:'screen'});
    await page.setViewportSize({width:1440,height:1000});
    console.log(`PASS ${item.file}: links, English, desktop/mobile, math, print and review controls.`);
  }
  assert.deepEqual(errors,[],'No page errors');
  const blocked=await browser.newContext();
  await blocked.addInitScript(()=>Object.defineProperty(window,'localStorage',{get(){throw new DOMException('blocked','SecurityError');}}));
  try{
    const blockedPage=await blocked.newPage();
    await blockedPage.goto(pathToFileURL(path.join(directory,pages[1].file)).href);
    await blockedPage.locator('[data-question="q1"]').check();
    assert.match(await blockedPage.locator('.design-review-count').textContent(),/cannot be saved/);
  }finally{await blocked.close();}
  const noJs=await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
  try{
    const staticPage=await noJs.newPage();
    await staticPage.goto(pathToFileURL(path.join(directory,'02_game_genre_solution.html')).href);
    assert.ok(await staticPage.locator('.katex').count()>10);
    assert.equal(await staticPage.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  }finally{await noJs.close();}
  console.log('PASS CV design pack:63 basic questions,63 corresponding answers,3 architecture/loss flows,offline/no-JS math,blocked-storage fallback.');
}finally{await context.close();await browser.close();}
