import assert from 'node:assert/strict';
import path from 'node:path';

const books=[
  {id:'system-design',file:'07_system_design.html',label:'System Design',chapters:7,query:'webhook'}
];

async function revealChapters(page) {
  await page.locator('[data-expand]').click();
  assert.equal(await page.locator('article details:not([open])').count(),0);
}

async function noOverflow(page,description) {
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,description);
}

async function screenshotAt(page,heading,file,preview) {
  await heading.evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
  await page.screenshot({
    path:path.join(preview,file),animations:'disabled',
    style:'html{scroll-behavior:auto!important}.toast{visibility:hidden!important}'
  });
}

export async function checkInterviewPrep({page,context,target,preview}) {
  const metadata={};
  const savedViewport=page.viewportSize();
  await page.setViewportSize({width:1440,height:1000});
  await page.goto(target('index.html'));
  const savedProgress=await page.evaluate(()=>localStorage.getItem('study-progress-v1'));
  try {
    assert.equal(await page.locator('.book-row').count(),await page.evaluate(()=>Object.keys(window.STUDY_BOOK.docs).filter(id=>id!=='home').length));
    for(const book of books) {
      await page.goto(target('index.html'));
      await page.locator(`.book-row[href="${book.file}"]`).click();
      await page.waitForURL(url=>url.pathname.endsWith('/'+book.file));
      assert.equal(await page.locator(`#sidebar a[href="${book.file}"]`).getAttribute('aria-current'),'page');
      metadata[book.id]=await page.evaluate(id=>window.STUDY_BOOK.docs[id],book.id);
      assert.equal(metadata[book.id].chapters.length,book.chapters,`${book.label}: chapter count`);
      assert.equal(await page.locator('details.chapter[data-chapter]').count(),book.chapters);
      await page.locator('[data-open-search]').first().click();
      await page.locator(`[data-scope="${book.id}"]`).click();
      await page.locator('#search-input').fill(book.query);
      await page.waitForFunction(label=>{
        const results=[...document.querySelectorAll('.result-path')];
        return document.querySelector('.search-status').textContent.startsWith('Found')&&results.length>0&&results.every(node=>node.textContent===label);
      },book.label);
      await page.locator('[data-close-search]').click();
      await revealChapters(page);
      await noOverflow(page,`${book.label}: desktop overflow`);
      const firstId=metadata[book.id].chapters[0].id;
      const mark=page.locator(`[data-chapter="${firstId}"] [data-mark]`);
      if(await mark.getAttribute('aria-pressed')!=='true')await mark.click();
      await page.reload();
      assert.equal(await page.locator(`[data-chapter="${firstId}"] [data-mark]`).getAttribute('aria-pressed'),'true',`${book.label}: reading progress must persist`);
    }

    await page.setViewportSize({width:1440,height:1000});
    await page.goto(target('07_system_design.html'));
    await revealChapters(page);
    const designChapters=metadata['system-design'].chapters;
    const diagrams=page.locator('article pre code.language-text,article pre code.language-plaintext');
    assert.ok(await diagrams.count()>=3,'System Design needs at least three readable architecture diagrams');
    const reportedCases=designChapters.filter(chapter=>/Nuro Example:/.test(chapter.title));
    assert.equal(reportedCases.length,3,'Three reported-topic cases should be the main SD path');
    assert.ok(reportedCases.some(chapter=>/Webhook/.test(chapter.title)));
    assert.ok(reportedCases.some(chapter=>/Vehicle Metrics/.test(chapter.title)));
    assert.ok(reportedCases.some(chapter=>/CPU\/GPU Video Processing/.test(chapter.title)));
    const liveId='5-worked-example-low-latency-vehicle-video';
    const optionalId='6-optional-video-reference';
    assert.equal(designChapters[4].id,liveId,'The live-video worked example precedes optional references');
    assert.equal(designChapters[5].id,optionalId,'Older video material remains optional');
    const liveCase=page.locator(`[data-chapter="${liveId}"]`);
    const liveText=await liveCase.textContent();
    for(const [concept,pattern] of [
      ['evidence boundary',/role-inspired[\s\S]*not a verified Nuro interview question/i],
      ['camera-to-viewer explanation',/camera[\s\S]*viewer/i],
      ['latency budget',/latency budget/i],
      ['bandwidth adaptation',/bandwidth/i],
      ['bounded buffering',/bounded/i],
      ['codec-aware frame dropping',/keyframe/i],
      ['connection recovery',/reconnect/i],
      ['stale-video detection',/stale/i],
      ['capture-time freshness',/capture.{0,15}time|capture.{0,15}timestamp/i],
      ['failure validation',/failure|fault/i],
      ['interview recap',/reference answer|spoken|recap/i]
    ])assert.match(liveText,pattern,`Live vehicle video needs ${concept}`);
    assert.ok(await liveCase.locator('pre code.language-text,pre code.language-plaintext').count()>=3,'Explain the live-video design with multiple readable diagrams');
    const optionalAliases=[
      '4-optional-video-reference','5-optional-video-reference',
      '2-building-blocks-you-actually-need','41-a-few-terms-to-look-up-when-needed','51-a-few-terms-to-look-up-when-needed',
      '3-practice-live-video-for-a-remote-operator','42-live-vehicle-video','52-live-vehicle-video',
      '4-practice-an-on-vehicle-camera-processing-pipeline','43-on-vehicle-camera-buffers','53-on-vehicle-camera-buffers',
      '5-practice-resumable-video-upload-and-replay','44-resumable-video-upload','54-resumable-video-upload'
    ];
    for(const alias of optionalAliases)assert.equal(await page.locator(`[id="${alias}"]`).count(),1,`Preserve Optional Video bookmark: ${alias}`);
    for(const oldChapter of ['4-optional-video-reference','5-optional-video-reference']) {
      await page.evaluate(({oldChapter})=>localStorage.setItem('study-progress-v1',JSON.stringify({'system-design':{done:[oldChapter,'4-nuro-example-cpugpu-video-processing'],last:'52-live-vehicle-video'},ml:{done:['1-neural-net-basics']}})),{oldChapter});
      await page.reload();
      const migrated=await page.evaluate(()=>JSON.parse(localStorage.getItem('study-progress-v1')));
      assert.ok(migrated['system-design'].done.includes(optionalId),'Preserve completion of the existing optional reference');
      assert.ok(!migrated['system-design'].done.includes(liveId),'The newly added live-video example must remain unread');
      assert.ok(migrated['system-design'].done.includes('4-nuro-example-cpugpu-video-processing'),'Preserve the CPU/GPU chapter completion');
      assert.equal(migrated['system-design'].last,'62-live-vehicle-video','Resume at the moved reference, not the new chapter');
      assert.deepEqual(migrated.ml.done,['1-neural-net-basics'],'Do not alter another book during migration');
    }
    for(const alias of ['5-optional-video-reference','52-live-vehicle-video']) {
      await page.goto(target('07_system_design.html')+'#'+alias);
      assert.equal(await page.locator(`[data-chapter="${optionalId}"]`).evaluate(node=>node.open),true,`${alias}: open the moved Optional chapter`);
    }
    await page.locator('[data-open-search]').first().click();
    await page.locator('[data-scope="system-design"]').click();
    await page.locator('#search-input').fill('stale');
    const liveSearch=page.locator('.search-result[href^="07_system_design.html#5"]');
    await liveSearch.first().waitFor();
    assert.ok((await page.locator('.result-path').allTextContents()).every(label=>label==='System Design'),'Live-video search stays scoped to System Design');
    await liveSearch.first().click();
    await page.waitForURL(url=>url.hash.startsWith('#5'));
    assert.equal(await liveCase.evaluate(node=>node.open),true,'Search opens the new live-video chapter');
    await liveCase.locator('[data-mark]').click();
    await page.reload();
    assert.equal(await liveCase.locator('[data-mark]').getAttribute('aria-pressed'),'true','Live-video reading progress persists');
    assert.equal(await page.locator(`[data-chapter="${optionalId}"] [data-mark]`).getAttribute('aria-pressed'),'true','Reading the new chapter preserves optional-reference completion');
    await revealChapters(page);
    const videoCase=page.locator('[data-chapter="4-nuro-example-cpugpu-video-processing"]');
    const videoText=await videoCase.textContent();
    for(const [concept,pattern] of [
      ['one-hour interview agenda',/60[- ]minute/i],
      ['interview reference answer',/reference answer/i],
      ['processing stages',/stages?/i],
      ['worker leases',/leases?/i],
      ['backpressure',/backpressure/i]
    ])assert.match(videoText,pattern,`CPU/GPU video processing needs ${concept}`);
    const journey=videoText.indexOf('4.5 Explain the high-level user journey');
    const architecture=videoText.indexOf('4.6 Build the architecture one need at a time');
    const interfaces=videoText.indexOf('4.7 Translate the picture into APIs and records');
    assert.ok(journey>=0&&journey<architecture&&architecture<interfaces,'Explain the user journey and architecture before API syntax');
    assert.ok(videoText.indexOf('POST /batches')>interfaces,'Do not introduce POST before the interface explanation');
    for(const concept of ['Who is the client?','What is a batch?','Why introduce a manifest?','Translate the original diagram','GPU batch']) {
      assert.ok(videoText.includes(concept),`Missing beginner bridge: ${concept}`);
    }
    for(const alias of [
      '41-start-with-one-video-and-six-basic-concepts',
      '45-define-the-interfaces-and-durable-records--minutes-1722',
      '46-draw-the-complete-architecture--minutes-2228',
      '47-walk-video-17-through-the-system--minutes-2833'
    ])assert.equal(await videoCase.locator(`[id="${alias}"]`).count(),1,`Preserve video-processing bookmark: ${alias}`);
    for(const chapter of reportedCases) {
      assert.ok(await page.locator(`[data-chapter="${chapter.id}"] pre code.language-text,[data-chapter="${chapter.id}"] pre code.language-plaintext`).count()>0,`${chapter.title}: missing architecture diagram`);
    }
    assert.equal(await page.locator('h2,h3,h4').filter({hasText:/Three mocks to do first|Practice Plan and Further Reading|Building Blocks You Actually Need/}).count(),0,'Remove the generic mock/prerequisite sequence');
    await screenshotAt(page,page.locator(`[data-chapter="${designChapters[0].id}"] > summary`),'system-design-framework-desktop.png',preview);
    await screenshotAt(page,page.locator(`[data-chapter="${designChapters[2].id}"] > summary`),'system-design-case-desktop.png',preview);
    await screenshotAt(page,page.locator(`[data-chapter="${reportedCases[0].id}"] > summary`),'nuro-webhook-walkthrough-desktop.png',preview);
    await screenshotAt(page,videoCase.locator(':scope > summary'),'nuro-video-processing-desktop.png',preview);
    await screenshotAt(page,liveCase.locator(':scope > summary'),'vehicle-live-video-desktop.png',preview);
    await page.locator('[data-collapse]').click();
    await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));
    await page.emulateMedia({media:'print'});
    assert.equal(await page.locator('article details:not([open])').count(),0);
    for(const diagram of await diagrams.all())assert.ok(await diagram.isVisible(),'Architecture diagrams must remain printable');
    await page.emulateMedia({media:'screen'});
    await page.evaluate(()=>window.dispatchEvent(new Event('afterprint')));
    await revealChapters(page);
    await page.setViewportSize({width:390,height:844});
    await noOverflow(page,'System Design: mobile overflow');
    await screenshotAt(page,page.locator(`[data-chapter="${designChapters[0].id}"] > summary`),'system-design-framework-mobile.png',preview);
    await screenshotAt(page,page.locator(`[data-chapter="${designChapters[2].id}"] > summary`),'system-design-case-mobile.png',preview);
    await screenshotAt(page,videoCase.locator(':scope > summary'),'nuro-video-processing-mobile.png',preview);
    await screenshotAt(page,liveCase.locator(':scope > summary'),'vehicle-live-video-mobile.png',preview);

    const offline=await context.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
    try {
      await offline.route('**/*',route=>/^https?:/.test(route.request().url())?route.abort():route.continue());
      const staticPage=await offline.newPage();
      for(const book of books) {
        await staticPage.goto(target(book.file));
        for(const chapter of await staticPage.locator('details.chapter').all()) {
          if(!await chapter.evaluate(node=>node.open))await chapter.locator(':scope > summary').click();
        }
        for(const drill of await staticPage.locator('.coding-drill').all()) {
          if(!await drill.evaluate(node=>node.open))await drill.locator(':scope > summary').click();
        }
        for(const listing of await staticPage.locator('details.source-inventory').all()) {
          if(!await listing.evaluate(node=>node.open))await listing.locator(':scope > summary').click();
          assert.ok(await listing.locator('table').isVisible(),'Report index must be readable without JavaScript');
        }
        for(const code of await staticPage.locator('article pre code').all())assert.ok(await code.isVisible(),`${book.label}: code and diagrams must remain readable without JavaScript`);
        if(book.id==='system-design') {
          const staticLive=staticPage.locator(`[data-chapter="${liveId}"]`);
          assert.ok(await staticLive.locator(':scope > summary').isVisible(),'The live-video example remains accessible without JavaScript');
          assert.match(await staticLive.textContent(),/stale/i,'Freshness guidance is present in the offline page');
        }
        await noOverflow(staticPage,`${book.label}: no-JS mobile overflow`);
      }
    } finally {await offline.close();}
  } finally {
    await page.emulateMedia({media:'screen'});
    await page.evaluate(saved=>saved===null?localStorage.removeItem('study-progress-v1'):localStorage.setItem('study-progress-v1',saved),savedProgress);
    if(savedViewport)await page.setViewportSize(savedViewport);
  }
  console.log('PASS System Design book: home link, scoped search, reading progress, three reported-topic design walkthroughs plus role-inspired live video, legacy bookmarks/progress, diagrams, print, mobile and no-JS.');
}
