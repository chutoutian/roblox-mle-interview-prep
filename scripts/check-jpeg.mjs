import assert from 'node:assert/strict';
import path from 'node:path';
import { jpegStages } from './jpeg-diagram.mjs';

export async function checkJpeg({page,context,target,preview}) {
  const sectionIds=[
    '21-jpeg-overall-pipeline',
    '22-color-conversion-and-chroma-subsampling',
    '23-blocks-and-level-shift',
    '24-dct',
    '25-quantization',
    '26-zigzag-and-entropy-coding',
    '27-jpeg-file-and-decoding',
    '28-modern-image-codecs'
  ];
  await page.goto(target('01_video_domain.html')+'#21-jpeg-overall-pipeline');
  const chapter=page.locator('#chapter-2-image-codecs');
  const diagram=chapter.locator('figure#jpeg-encoding-pipeline[data-jpeg-diagram]');
  assert.equal(await diagram.count(),1,'Image Codecs needs one overall JPEG pipeline');
  assert.equal(await diagram.evaluate(node=>!!node.closest('.coding-drill')),false,'The overview must not be hidden inside an exercise');
  assert.deepEqual(await chapter.locator(':scope > .chapter-body > .prose > h3').evaluateAll(nodes=>nodes.map(node=>node.id)),sectionIds,'Explain the JPEG steps in encoding order');
  assert.equal(await diagram.evaluate(node=>{
    const first=document.getElementById('21-jpeg-overall-pipeline');
    const next=document.getElementById('22-color-conversion-and-chroma-subsampling');
    return !!(first.compareDocumentPosition(node)&Node.DOCUMENT_POSITION_FOLLOWING)
      &&!!(node.compareDocumentPosition(next)&Node.DOCUMENT_POSITION_FOLLOWING);
  }),true,'The overview diagram belongs directly in section 2.1');
  assert.ok(!(await chapter.textContent()).includes('Image codecs exploit spatial redundancy'),'Remove the generic introductory paragraph');
  assert.equal(await page.locator('.coding-drill').count(),15,'The video coding exercises of chapters 1 to 3 after removing the VBV simulator');
  assert.equal(await page.locator('[id="24-jpeg"]').evaluate(node=>node.parentElement.closest('[id]')?.id),sectionIds[0],'Keep the old JPEG bookmark working');

  assert.equal(jpegStages.length,9);
  assert.equal(await diagram.locator('ol.jpeg-steps > li.jpeg-step').count(),9);
  assert.deepEqual(await diagram.locator('.jpeg-step-number').allTextContents(),['1','2','3','4','5','6','7','8','9']);
  assert.equal(await diagram.locator('.jpeg-branch').count(),2,'Show separate DC and AC paths before Huffman coding');
  assert.equal(await diagram.locator('.jpeg-step-tag').count(),2,'Label optional chroma subsampling and quantization as lossy');
  assert.match(jpegStages[3].output,/64 samples → 64 coefficients/,'DCT alone does not reduce the coefficient count');
  const stageLinks=diagram.locator('.jpeg-step-heading > a');
  assert.equal(await stageLinks.count(),9,'Every JPEG stage must link to its explanation');
  assert.deepEqual(await stageLinks.allTextContents(),jpegStages.map(stage=>stage.title));
  const destinations=await stageLinks.evaluateAll(nodes=>nodes.map(node=>node.getAttribute('href').slice(1)));
  assert.deepEqual(destinations,jpegStages.map(stage=>stage.href.slice(1)));
  for(const destination of destinations) {
    assert.ok(sectionIds.includes(destination),`Unexpected JPEG stage destination: ${destination}`);
    assert.equal(await chapter.locator(`[id="${destination}"]`).count(),1);
  }
  for(const link of await stageLinks.all()) {
    const href=await link.getAttribute('href');
    await link.click();
    assert.equal(await page.evaluate(()=>location.hash),href);
    assert.equal(await chapter.evaluate(node=>node.open),true);
  }
  await stageLinks.first().focus();
  await stageLinks.first().press('Enter');
  assert.equal(await page.evaluate(()=>location.hash),'#'+destinations[0],'Stage links must support keyboard navigation');

  const componentStyle='html{scroll-behavior:auto!important}.topbar,.read-progress,.skip-link,.sidebar,.sidebar-backdrop,.toast{visibility:hidden!important}';
  async function capture(name) {
    const viewport=page.viewportSize();
    const height=Math.ceil((await diagram.boundingBox()).height);
    // A figure taller than its viewport can otherwise capture a blank bottom strip.
    await page.setViewportSize({width:viewport.width,height:Math.max(viewport.height,height+180)});
    try {
      await diagram.screenshot({path:path.join(preview,name),style:componentStyle,animations:'disabled'});
    } finally {await page.setViewportSize(viewport);}
  }
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  await capture('jpeg-pipeline-desktop.png');
  const originalTheme=await page.locator('html').getAttribute('data-theme');
  await page.locator('[data-theme-toggle]').click();
  assert.notEqual(await page.locator('html').getAttribute('data-theme'),originalTheme);
  for(const link of await stageLinks.all())assert.ok(await link.isVisible());
  await capture('jpeg-pipeline-dark.png');
  await page.locator('[data-theme-toggle]').click();

  await page.locator('[data-collapse]').click();
  await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));
  await page.emulateMedia({media:'print'});
  assert.ok(await diagram.isVisible(),'The pipeline must be printable');
  for(const link of await stageLinks.all())assert.ok(await link.isVisible());
  await page.emulateMedia({media:'screen'});
  await page.evaluate(()=>window.dispatchEvent(new Event('afterprint')));
  if(!await chapter.evaluate(node=>node.open))await chapter.locator(':scope > summary').click();

  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,'JPEG mobile page must not overflow');
  const box=await diagram.boundingBox();
  assert.ok(box.x>=0&&box.x+box.width<=391,'JPEG diagram must fit the mobile viewport');
  for(const link of await stageLinks.all())assert.ok(await link.isVisible());
  await capture('jpeg-pipeline-mobile.png');
  await page.setViewportSize({width:1440,height:1000});

  const offline=await context.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
  try {
    await offline.route('**/*',route=>/^https?:/.test(route.request().url())?route.abort():route.continue());
    const staticPage=await offline.newPage();
    await staticPage.goto(target('01_video_domain.html'));
    const staticChapter=staticPage.locator('#chapter-2-image-codecs');
    if(!await staticChapter.evaluate(node=>node.open))await staticChapter.locator(':scope > summary').click();
    const staticDiagram=staticChapter.locator('#jpeg-encoding-pipeline');
    assert.ok(await staticDiagram.isVisible(),'The full pipeline must be readable without JavaScript');
    assert.equal(await staticDiagram.locator('a[href^="#"]').count(),9);
    assert.equal(await staticPage.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  } finally {await offline.close();}

  await page.goto(target('01_video_domain.html')+'#codec-overview');
  const videoChapter=page.locator('#chapter-3-video-codecs');
  if(!await videoChapter.evaluate(node=>node.open))await videoChapter.locator(':scope > summary').click();
  const codec=videoChapter.locator('figure#codec-pipeline-figure');
  assert.equal(await codec.count(),1,'Section 3.1 needs one overall encoder figure');
  assert.equal(await codec.evaluate(node=>!!(node.compareDocumentPosition(document.getElementById('codec-step-1'))&Node.DOCUMENT_POSITION_FOLLOWING)),true,'The encoder figure comes before Step 1');
  const codecReport=await codec.evaluate(fig=>{
    const svg=fig.querySelector('svg');
    const nodes=[...svg.querySelectorAll('a.codec-node')];
    const steps=nodes.map(a=>a.getAttribute('href').replace('#codec-step-',''));
    const inside=(inner,outer)=>inner.left>=outer.left-0.5&&inner.right<=outer.right+0.5&&inner.top>=outer.top-0.5&&inner.bottom<=outer.bottom+0.5;
    const hit=(a,b)=>a.left<b.right&&b.left<a.right&&a.top<b.bottom&&b.top<a.bottom;
    const shapes=[...svg.querySelectorAll('a.codec-node rect, .codec-sum')].map(node=>node.getBoundingClientRect());
    return {
      steps:[...new Set(steps)].sort((a,b)=>a-b).join(','),
      missing:steps.filter(step=>!document.getElementById('codec-step-'+step)),
      overflowing:nodes.filter(a=>[...a.querySelector('.codec-box').children].some(child=>!inside(child.getBoundingClientRect(),a.querySelector('rect').getBoundingClientRect()))).map(a=>a.getAttribute('aria-label')),
      covered:[...svg.querySelectorAll('.codec-label:not(.codec-buffer-text) > span')].filter(label=>shapes.some(shape=>hit(label.getBoundingClientRect(),shape))).map(label=>label.textContent),
      katexErrors:fig.querySelectorAll('.katex-error').length
    };
  });
  assert.equal(codecReport.steps,'1,2,3,4,6,7,8,9,10,11,12','Every step is a box except 5, the subtractor');
  assert.deepEqual(codecReport.missing,[],'Every box links to its Step paragraph');
  assert.deepEqual(codecReport.overflowing,[],'Box text must fit inside its box');
  assert.deepEqual(codecReport.covered,[],'Arrow labels must not sit on a box');
  assert.equal(codecReport.katexErrors,0);
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,'The encoder figure scrolls inside its own box on mobile');
  await page.setViewportSize({width:1440,height:1000});
  console.log('PASS JPEG-first chapter order, nine linked pipeline stages, retained exercises, keyboard, dark mode, mobile, print and no-JS reading; the section 3.1 encoder figure links all eleven boxes and fits its labels.');
}
