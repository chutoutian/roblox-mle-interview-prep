import assert from 'node:assert/strict';
import path from 'node:path';
import { networkDiagramIds, calculateAbrScenario, networkModels } from './network-diagrams.mjs';

export async function checkNetworking({page,context,target,preview,cache}) {
  await page.goto(target('05_networking_streaming.html'));
  const books=await page.evaluate(()=>window.STUDY_BOOK.docs);
  assert.equal(books.network.chapters.length,10);
  assert.equal(books.network.drills,1);
  assert.equal(books.video.drills+books.network.drills,16,'The remaining video and networking exercises, after the VBV simulator was removed');
  assert.equal(books.video.chapters.length,3,'Video keeps vision, image codecs and video codecs');
  assert.ok(!books.video.chapters.some(c=>/Streaming|Real-Time/.test(c.title)),'Standalone transport chapters must move out of Video');
  assert.equal(await page.locator('figure[data-network-diagram]').count(),11);
  assert.deepEqual(await page.locator('figure[data-network-diagram]').evaluateAll(nodes=>nodes.map(n=>n.dataset.networkDiagram).sort()),[...networkDiagramIds].sort());
  assert.equal(await page.locator('#sidebar a[href="05_networking_streaming.html"]').getAttribute('aria-current'),'page');
  assert.equal(await page.locator('[data-scope="network"]').count(),1);
  for(const destination of Object.values(books.video.redirects)) {
    const [file,id]=destination.split('#');
    assert.ok(cache.get(file)?.ids.includes(id),`Moved destination must exist: ${destination}`);
  }
  for(const [alias,expected] of [
    ['76-abr-algorithms','51-abr-algorithms'],
    ['86-abr-algorithms','51-abr-algorithms'],
    ['83-cloud-gaming--cloud-vr-pipeline','66-interactive-media-pipeline']
  ]) {
    await page.goto(target('01_video_domain.html')+'#'+alias);
    await page.waitForURL(url=>url.pathname.endsWith('/05_networking_streaming.html')&&url.hash==='#'+expected);
    assert.equal(await page.locator(`[id="${expected}"]`).evaluate(el=>el.closest('[data-chapter]').open),true);
  }
  await page.goto(target('01_video_domain.html'));
  const saved=await page.evaluate(()=>localStorage.getItem('study-progress-v1'));
  try {
    await page.evaluate(()=>localStorage.setItem('study-progress-v1',JSON.stringify({
      video:{layoutVersion:2,done:['7-streaming--delivery','8-real-time-vr360-and-cloud-gaming','9-video-quality-assessment'],last:'7-streaming--delivery'},
      ml:{done:['1-neural-net-basics']}
    })));
    await page.reload();
    const migrated=await page.evaluate(()=>JSON.parse(localStorage.getItem('study-progress-v1')));
    assert.equal(migrated.video.layoutVersion,8);
    assert.deepEqual(migrated.network.done,[],'Expanded networking material must not be automatically marked complete');
    assert.equal(migrated.network.last,'4-hls-dash-and-cmaf');
    assert.deepEqual(migrated.ml.done,['1-neural-net-basics']);
    await page.locator('[data-resume]').click();
    await page.waitForURL(url=>url.pathname.endsWith('/05_networking_streaming.html')&&url.hash==='#4-hls-dash-and-cmaf');
    await page.locator('#chapter-4-hls-dash-and-cmaf [data-mark]').click();
    await page.reload();
    assert.equal(await page.locator('#chapter-4-hls-dash-and-cmaf [data-mark]').getAttribute('aria-pressed'),'true');
  } finally {
    await page.evaluate(value=>value===null?localStorage.removeItem('study-progress-v1'):localStorage.setItem('study-progress-v1',value),saved);
  }
  // Use a separate, disposable browser context so seeded historical quiz data
  // cannot alter the reader state used by the remaining checks.
  const quizContext=await context.browser().newContext();
  try {
    await quizContext.route('**/*',route=>/^https?:/.test(route.request().url())?route.abort():route.continue());
    const quizPage=await quizContext.newPage();
    await quizPage.goto(target('01_video_domain.html'));
    await quizPage.evaluate(questions=>{
      localStorage.clear();
      localStorage.setItem('study-quiz:video:11-rapid-fire-drill--video:0','true');
      localStorage.setItem('study-quiz:video:12-rapid-fire-drill--video:1','true');
      localStorage.setItem('study-quiz:video:11-rapid-fire-drill--video:40','true');
      localStorage.setItem('study-quiz:video:10-rapid-fire-drill--video:0','false');
      for(const question of questions) localStorage.setItem('study-quiz:video:11-rapid-fire-drill--video:'+question.index,'true');
    },books.video.networkSplitMigration.deliveryQuestions);
    await quizPage.reload();
    await quizPage.goto(target('05_networking_streaming.html'));
    const networkQuestions=quizPage.locator('#chapter-9-rapid-fire-drill--networking ol > li');
    assert.equal(await networkQuestions.count(),28);
    const texts=await networkQuestions.allTextContents();
    const migratedIndices=books.video.networkSplitMigration.deliveryQuestions.map(question=>{
      const index=texts.findIndex(text=>text.replace(/\s+/g,' ').trim()===(question.currentText||question.text));
      assert.ok(index>=0,`Migrated quiz question must exist: ${question.text}`);
      return index;
    });
    for(const index of migratedIndices) assert.equal(await networkQuestions.nth(index).locator('input').isChecked(),true);
    assert.equal(await networkQuestions.nth(0).locator('input').isChecked(),false,'New networking questions must not inherit unrelated video answers');
    await quizPage.locator('#chapter-9-rapid-fire-drill--networking').evaluate(node=>node.open=true);
    await networkQuestions.nth(migratedIndices[0]).locator('input').uncheck();
    await quizPage.reload();
    assert.equal(await networkQuestions.nth(migratedIndices[0]).locator('input').isChecked(),false);
  } finally {await quizContext.close();}
  await page.goto(target('index.html'));
  assert.equal(await page.locator('.book-row').count(),Object.keys(books).filter(id=>id!=='home').length);
  await page.locator('.book-row[href="05_networking_streaming.html"]').click();
  await page.waitForURL(/05_networking_streaming\.html$/);
  await page.locator('[data-open-search]').first().click();
  await page.locator('[data-scope="network"]').click();
  await page.locator('#search-input').fill('TCP');
  await page.waitForFunction(()=>document.querySelector('.search-status').textContent.startsWith('Found'));
  assert.ok(await page.locator('.search-result').count()>0);
  assert.ok((await page.locator('.result-path').allTextContents()).every(t=>t==='Networking & Streaming'));
  await page.locator('[data-close-search]').click();
  await page.locator('[data-expand]').click();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  for(const id of networkDiagramIds) {
    const figure=page.locator(`[data-network-diagram="${id}"]`);
    const label=await figure.getAttribute('aria-labelledby');
    assert.ok(label&&await page.locator(`[id="${label}"]`).count(),`${id}: accessible figure title`);
    const controls=figure.locator('[data-network-choice]');
    for(let index=0;index<await controls.count();index++) {
      const button=controls.nth(index), key=await button.getAttribute('data-network-choice');
      await button.click();
      assert.equal(await figure.getAttribute('data-network-state'),key);
      assert.equal(await figure.locator('[data-network-choice][aria-pressed="true"]').count(),1);
      assert.equal(await figure.locator('[data-network-panel]:not([hidden])').count(),1);
      assert.equal(await figure.locator(`[data-network-panel="${key}"]`).isVisible(),true);
      assert.ok((await figure.locator('[data-network-live]').textContent()).trim().length>15);
    }
    if(await controls.count()) {
      await controls.first().focus(); await controls.first().press('Enter');
      assert.equal(await figure.getAttribute('data-network-state'),await controls.first().getAttribute('data-network-choice'));
      await controls.last().focus(); await controls.last().press('Space');
      assert.equal(await figure.getAttribute('data-network-state'),await controls.last().getAttribute('data-network-choice'));
    }
  }
  const packetPath=page.locator('[data-network-diagram="packet-path"]');
  const protocolLayers=['application','transport','network','link','physical'];
  assert.equal(await packetPath.getAttribute('id'),'net-packet-path','Existing packet-path anchors must remain valid');
  assert.equal(await packetPath.locator('[data-network-choice]').count(),0,'Packet encapsulation is one continuous example');
  assert.deepEqual(await packetPath.locator('[data-encapsulation-layer]').evaluateAll(nodes=>nodes.map(node=>node.dataset.encapsulationLayer)),protocolLayers);
  for(const layer of protocolLayers) {
    const bytes=packetPath.locator(`[data-encapsulation-layer="${layer}"] .network-layer-bytes`);
    assert.equal(await bytes.count(),1,`${layer}: show the contents carried by this layer`);
    assert.ok((await bytes.textContent()).trim().length>0,`${layer}: byte contents must not be empty`);
  }
  assert.equal(await packetPath.locator('[data-protocol-stack]').count(),3);
  for(const endpoint of ['sender','receiver']) {
    const stack=packetPath.locator(`[data-protocol-stack="${endpoint}"]`);
    assert.equal(await stack.count(),1);
    assert.deepEqual(await stack.locator('[data-stack-layer]').evaluateAll(nodes=>nodes.map(node=>node.dataset.stackLayer)),protocolLayers,`${endpoint}: preserve all five layers`);
  }
  assert.deepEqual(await packetPath.locator('[data-protocol-stack="router"] [data-stack-layer]').evaluateAll(nodes=>nodes.map(node=>node.dataset.stackLayer)),['network','link','physical'],'The transit router must not have application or transport layers');
  const packetModel=networkModels.packetPath;
  assert.equal(packetModel.request,'GET /hello HTTP/1.1\r\nHost: example.test\r\n\r\n');
  assert.equal(packetModel.requestBytes,43);
  assert.equal(Buffer.byteLength(packetModel.request,'ascii'),packetModel.requestBytes,'HTTP byte count must include each CRLF');
  assert.equal(Buffer.byteLength(packetModel.request,'utf8'),packetModel.requestBytes,'The example request must contain only ASCII bytes');
  assert.equal(packetModel.tcpSequence,1001);
  assert.equal(packetModel.nextAck,1044);
  assert.equal(packetModel.tcpSequence+packetModel.requestBytes,packetModel.nextAck,'ACK points to the next byte after this request');
  assert.equal(packetModel.tcpHeaderBytes,20);
  assert.equal(packetModel.ipHeaderBytes,20);
  assert.equal(packetModel.ethernetHeaderBytes,14);
  assert.equal(packetModel.ethernetFcsBytes,4);
  const segmentBytes=packetModel.requestBytes+packetModel.tcpHeaderBytes;
  const ipPacketBytes=segmentBytes+packetModel.ipHeaderBytes;
  assert.equal(segmentBytes,63);
  assert.equal(ipPacketBytes,83);
  assert.equal(packetModel.ethernetHeaderBytes+ipPacketBytes+packetModel.ethernetFcsBytes,101,'Ethernet frame size includes its header and FCS');
  assert.equal(packetModel.senderTtl,64);
  assert.equal(packetModel.receiverTtl,63);
  assert.equal(packetModel.senderTtl-packetModel.receiverTtl,1,'One forwarding router consumes one TTL');
  const packetText=await packetPath.textContent();
  for(const field of ['GET /hello HTTP/1.1','Host: example.test','51514','1001','1044','192.0.2.10','198.51.100.20']) {
    assert.ok(packetText.includes(field),`Packet-path example must display ${field}`);
  }
  const abr=page.locator('[data-network-diagram="abr-buffer"]');
  for(const [throughput,buffer,rate,download,newBuffer,stall] of [
    [2,6,2,4,6,0],[5,6,4,3.2,6.8,0],[10,6,8,3.2,6.8,0],
    [2,2,2,4,4,2],[5,2,4,3.2,4,1.2],[10,2,8,3.2,4,1.2]
  ]) {
    const result=calculateAbrScenario(throughput,buffer);
    assert.equal(result.representationMbps,rate);
    assert.equal(result.downloadSeconds,download);
    assert.ok(Math.abs(result.newBufferSeconds-newBuffer)<1e-10);
    assert.ok(Math.abs(result.stallSeconds-stall)<1e-10);
    await abr.locator(`[data-network-choice="t${throughput}-b${buffer}"]`).click();
    const panel=abr.locator('[data-network-panel]:not([hidden])');
    assert.match(await panel.locator('.network-stall').textContent(),stall>0?/STALL:/:/NO STALL:/);
    assert.equal(await panel.locator('.network-buffer-equation > div').last().locator('strong').textContent(),`${newBuffer} s`);
  }
  assert.equal(networkModels.realtime.reduce((sum,stage)=>sum+stage.milliseconds,0),150);
  assert.equal(networkModels.lowLatency.segmentSeconds/networkModels.lowLatency.partSeconds,4);
  assert.throws(()=>calculateAbrScenario(0,2),RangeError);
  await page.locator('[data-network-diagram="tcp-handshake"] [data-network-choice="ack"]').click();
  assert.match(await page.locator('[data-network-diagram="tcp-handshake"] [data-network-panel="ack"]').textContent(),/ack=501/);
  await page.emulateMedia({media:'print'});
  assert.ok((await page.locator('[data-network-panel]').evaluateAll(nodes=>nodes.every(n=>getComputedStyle(n).display!=='none'))),'Printed figures must include every scenario');
  await page.emulateMedia({media:'screen'});
  const componentStyle='.topbar,.read-progress,.skip-link,.sidebar,.sidebar-backdrop,.toast{visibility:hidden!important}';
  for(const id of networkDiagramIds) {
    const figure=page.locator(`[data-network-diagram="${id}"]`);
    if(await figure.locator('[data-network-choice]').count()) await figure.locator('[data-network-choice]').first().click();
    await figure.screenshot({path:path.join(preview,`network-${id}-desktop.png`),style:componentStyle,animations:'disabled'});
  }
  await page.locator('[data-theme-toggle]').click();
  await abr.screenshot({path:path.join(preview,'network-abr-dark.png'),style:componentStyle,animations:'disabled'});
  await packetPath.screenshot({path:path.join(preview,'network-packet-path-dark.png'),style:componentStyle,animations:'disabled'});
  await page.locator('[data-theme-toggle]').click();
  await page.setViewportSize({width:390,height:844});
  for(const id of networkDiagramIds) {
    const figure=page.locator(`[data-network-diagram="${id}"]`);
    const controls=figure.locator('[data-network-choice]');
    for(let index=0;index<await controls.count();index++) {
      await controls.nth(index).click();
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`${id}: mobile scenario overflow`);
    }
    assert.ok((await figure.boundingBox()).width<=390,`${id}: figure must fit the viewport`);
  }
  for(const id of ['packet-path','tcp-handshake','dash-flow','abr-buffer','webrtc-connect']) {
    await page.locator(`[data-network-diagram="${id}"]`).screenshot({path:path.join(preview,`network-${id}-mobile.png`),style:componentStyle,animations:'disabled'});
  }
  await page.setViewportSize({width:1440,height:1000});
  const offline=await context.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
  try {
    await offline.route('**/*',route=>/^https?:/.test(route.request().url())?route.abort():route.continue());
    const staticPage=await offline.newPage();
    await staticPage.goto(target('05_networking_streaming.html'));
    await staticPage.locator('details.chapter').evaluateAll(nodes=>nodes.forEach(n=>n.open=true));
    assert.equal(await staticPage.locator('[data-network-diagram]').count(),11);
    assert.equal(await staticPage.locator('[data-network-panel][hidden]').count(),0);
    assert.ok(await staticPage.locator('[data-network-panel]').evaluateAll(nodes=>nodes.every(n=>getComputedStyle(n).display!=='none')));
    assert.equal(await staticPage.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  } finally {await offline.close();}
  console.log('PASS all eleven networking diagrams, eighteen interactive scenarios, ABR math, keyboard, print, dark mode, mobile and no-JS fallback.');
  console.log('PASS separate networking book, moved content, cross-book bookmarks, reading and quiz progress migration, home and search integration.');
}
