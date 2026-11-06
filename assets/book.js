(() => {
  'use strict';
  const data = window.STUDY_BOOK;
  if (!data) return;
  const $ = (s, root=document) => root.querySelector(s);
  const $$ = (s, root=document) => [...root.querySelectorAll(s)];
  const lookup = (object,key) => object&&Object.hasOwn(object,key)?object[key]:undefined;
  const page = document.body.dataset.page;
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const storage = {
    get(key, fallback) {try {return JSON.parse(localStorage.getItem(key)) ?? fallback;} catch {return fallback;}},
    set(key, value) {try {localStorage.setItem(key, JSON.stringify(value));return true;} catch {return false;}}
  };
  let state = storage.get('study-progress-v1', {});
  if (!state || typeof state !== 'object' || Array.isArray(state)) state = {};
  // Keep bookmarks/resume locations useful after chapter renumbering. A merged
  // chapter is complete only if all of its former content chapters were read.
  const migration=data.docs.video?.navigationMigration, videoState=state.video;
  if(migration&&videoState&&typeof videoState==='object'&&(!Number.isInteger(videoState.layoutVersion)||videoState.layoutVersion<migration.version)) {
    const done=new Set(Array.isArray(videoState.done)?videoState.done:[]);
    for(const [chapter,previous] of Object.entries(migration.prerequisites)) {
      if(previous.every(id=>done.has(id))) done.add(chapter);
    }
    videoState.done=[...done];
    videoState.last=lookup(migration.aliases,videoState.last)||videoState.last;
    videoState.layoutVersion=migration.version;
    storage.set('study-progress-v1',state);
  }
  // Preserve prior completion/resume locations through the cross-book split.
  // New networking chapters contain new material, so do not mark them read
  // merely because the older, shorter streaming chapter was completed.
  const split=data.docs.video?.networkSplitMigration;
  if(split&&videoState&&typeof videoState==='object'&&(!Number.isInteger(videoState.layoutVersion)||videoState.layoutVersion<split.version)) {
    const done=new Set(Array.isArray(videoState.done)?videoState.done:[]);
    for(const [previous,current] of Object.entries(split.chapterMap)) if(done.has(previous)) done.add(current);
    videoState.done=[...done];
    videoState.last=lookup(split.videoAliases,videoState.last)||videoState.last;
    const moved=lookup(data.docs.video.redirects,videoState.last);
    if(moved&&!getPageState('network').last) getPageState('network').last=moved.split('#')[1];
    videoState.layoutVersion=split.version;
    storage.set('study-progress-v1',state);
  }
  // Removing chapters (Codec Standards, then Rate Control) renumbered every later video chapter; apply each renumbering in order.
  for(const renumber of data.docs.video?.renumberMigrations||[]) {
    if(!(videoState&&typeof videoState==='object'&&(!Number.isInteger(videoState.layoutVersion)||videoState.layoutVersion<renumber.version))) continue;
    const done=new Set(Array.isArray(videoState.done)?videoState.done:[]);
    for(const [previous,current] of Object.entries(renumber.chapterMap)) if(done.has(previous)) done.add(current);
    videoState.done=[...done];
    videoState.last=lookup(renumber.aliases,videoState.last)||videoState.last;
    videoState.layoutVersion=renumber.version;
    storage.set('study-progress-v1',state);
  }
  // Research & resume: the Earlier Work chapter was removed, so Behavioral Bank and
  // Questions to Ask Them moved from 9 and 10 to 8 and 9. Completion and resume follow them.
  const resumeMigration=data.docs.resume?.navigationMigration, resumeState=state.resume;
  if(resumeMigration&&resumeState&&typeof resumeState==='object'&&(!Number.isInteger(resumeState.layoutVersion)||resumeState.layoutVersion<resumeMigration.version)) {
    const done=Array.isArray(resumeState.done)?resumeState.done:[];
    resumeState.done=[...new Set(done.map(id=>resumeMigration.chapterMap[id]||id))];
    resumeState.last=lookup(resumeMigration.aliases,resumeState.last)||resumeState.last;
    resumeState.layoutVersion=resumeMigration.version;
    storage.set('study-progress-v1',state);
  }
  // ML: the four foundation chapters (1, 2, 3 and 8) became "1. Basics" and the other
  // chapters were renumbered. Basics is complete only if all four were read.
  const mlMigration=data.docs.ml?.navigationMigration, mlState=state.ml;
  if(mlMigration&&mlState&&typeof mlState==='object'&&(!Number.isInteger(mlState.layoutVersion)||mlState.layoutVersion<mlMigration.version)) {
    const done=new Set(Array.isArray(mlState.done)?mlState.done:[]);
    const stored=Number.isInteger(mlState.layoutVersion)?mlState.layoutVersion:0;
    // The prerequisites describe the layouts before version 5 (merged Basics, renumbered chapters).
    if(stored<5) for(const [chapter,previous] of Object.entries(mlMigration.prerequisites)) {
      if(previous.every(id=>done.has(id))) done.add(chapter);
    }
    // Later layouts rename whole chapters (version 6: generative models left; version 7: they returned as
    // chapter 5 and the chapters after it moved up by one). Apply each map the stored state predates, in order.
    let last=mlState.last;
    for(const step of mlMigration.chapterMaps||[]) if(stored<step.version) {
      for(const [previous,current] of Object.entries(step.map)) if(done.has(previous)) { done.delete(previous); if(current) done.add(current); }
      last=lookup(step.map,last)||last;
    }
    mlState.done=[...done];
    mlState.last=lookup(mlMigration.aliases,last)||last;
    mlState.layoutVersion=mlMigration.version;
    storage.set('study-progress-v1',state);
  }
  // Inserting the live-video walkthrough moves Optional Video Reference from
  // chapter 5 (formerly 4) to 6. Preserve that reference's progress, never infer
  // completion of the newly inserted worked example from its chapter number.
  const designState=state['system-design'];
  if(data.docs['system-design']?.chapters.some(chapter=>chapter.id==='6-optional-video-reference')&&designState&&typeof designState==='object'&&(!Number.isInteger(designState.layoutVersion)||designState.layoutVersion<1)) {
    const done=new Set(Array.isArray(designState.done)?designState.done:[]);
    if(['4-optional-video-reference','5-optional-video-reference'].some(id=>done.has(id))) done.add('6-optional-video-reference');
    const aliases={
      '4-optional-video-reference':'6-optional-video-reference',
      '5-optional-video-reference':'6-optional-video-reference',
      '2-building-blocks-you-actually-need':'61-a-few-terms-to-look-up-when-needed',
      '41-a-few-terms-to-look-up-when-needed':'61-a-few-terms-to-look-up-when-needed',
      '51-a-few-terms-to-look-up-when-needed':'61-a-few-terms-to-look-up-when-needed',
      '3-practice-live-video-for-a-remote-operator':'62-live-vehicle-video',
      '42-live-vehicle-video':'62-live-vehicle-video',
      '52-live-vehicle-video':'62-live-vehicle-video',
      '4-practice-an-on-vehicle-camera-processing-pipeline':'63-on-vehicle-camera-buffers',
      '43-on-vehicle-camera-buffers':'63-on-vehicle-camera-buffers',
      '53-on-vehicle-camera-buffers':'63-on-vehicle-camera-buffers',
      '5-practice-resumable-video-upload-and-replay':'64-resumable-video-upload',
      '44-resumable-video-upload':'64-resumable-video-upload',
      '54-resumable-video-upload':'64-resumable-video-upload'
    };
    designState.done=[...done];
    designState.last=lookup(aliases,designState.last)||designState.last;
    designState.layoutVersion=1;
    storage.set('study-progress-v1',state);
  }
  // LeetCode: Tree DFS (3), Backtracking (4) and DFS + Memoization (6) became one chapter,
  // "3. Recursion"; Dijkstra moved from 5 to 4 and Notes Inbox from 7 to 5 (it was 2 to 6 in
  // older layouts). The merged chapter is complete only if all three former chapters were read,
  // and a resume position inside one of them lands on the section that holds its part.
  const leetcodeState=state.leetcode, recursionId='3-recursion-tree-dfs-backtracking-memoization';
  if(data.docs.leetcode?.chapters.some(chapter=>chapter.id===recursionId)&&leetcodeState&&typeof leetcodeState==='object'&&(!Number.isInteger(leetcodeState.layoutVersion)||leetcodeState.layoutVersion<4)) {
    const chapterMap={'2-notes-inbox':'5-notes-inbox','3-notes-inbox':'5-notes-inbox','4-notes-inbox':'5-notes-inbox','6-notes-inbox':'5-notes-inbox','7-notes-inbox':'5-notes-inbox','5-dijkstra-shortest-path':'4-dijkstra-shortest-path'};
    const merged=['3-tree-dfs','4-backtracking','6-dfs--memoization'];
    const resume={...chapterMap,'3-tree-dfs':'32-tree-dfs','4-backtracking':'33-backtracking','6-dfs--memoization':'34-dfs--memoization'};
    const before=Array.isArray(leetcodeState.done)?leetcodeState.done:[];
    const done=new Set(before.filter(id=>!merged.includes(id)).map(id=>lookup(chapterMap,id)||id));
    if(merged.every(id=>before.includes(id))) done.add(recursionId);
    leetcodeState.done=[...done];
    leetcodeState.last=lookup(resume,leetcodeState.last)||leetcodeState.last;
    leetcodeState.layoutVersion=4;
    storage.set('study-progress-v1',state);
  }
  let toastTimer;
  function toast(message) {
    const el=$('.toast'); el.textContent=message;el.classList.add('visible');
    clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('visible'),2200);
  }
  function getPageState(id) {
    if (!state[id] || typeof state[id]!=='object') state[id]={};
    if (!Array.isArray(state[id].done)) state[id].done=[];
    return state[id];
  }
  function updateProgress() {
    let total=0, complete=0;
    for(const [id,book] of Object.entries(data.docs)) {
      const valid=new Set(book.chapters.map(c=>c.id));
      const done=getPageState(id).done.filter(x=>valid.has(x));
      if(id!=='home') {total+=valid.size;complete+=done.length;}
      const item=$(`[data-book-progress="${id}"]`);
      if(item) item.textContent=done.length?`${done.length} / ${valid.size} chapters read`:'Not started';
      if(id===page) {
        if($('[data-page-progress]')) $('[data-page-progress]').textContent=`${done.length} / ${valid.size} chapters read`;
        $$('[data-mark]').forEach(button=>{
          const checked=done.includes(button.dataset.mark);
          button.setAttribute('aria-pressed',String(checked));
          $('span',button).textContent=checked?'Read · Undo':'Mark as read';
          button.closest('.chapter')?.classList.toggle('is-read',checked);
        });
      }
    }
    const percent=total?Math.round(complete/total*100):0;
    $('[data-total-progress]').textContent=percent+'%';
    $('[data-total-bar]').style.width=percent+'%';
    $('[data-total-count]').textContent=`${complete} / ${total} chapters read`;
    if($('[data-resume]')&&getPageState(page).last) $('[data-resume]').firstChild.textContent='Continue reading ';
  }
  updateProgress();
  const sizes=[{label:'Default',px:13},{label:'Medium',px:15},{label:'Large',px:17}];
  let sizeIndex=storage.get('study-font-size',0);
  if(!Number.isInteger(sizeIndex)||sizeIndex<0||sizeIndex>2) sizeIndex=0;
  function applyFontSize(){
    document.documentElement.style.setProperty('--reader-font-size',sizes[sizeIndex].px+'px');
    if($('[data-font-size]')) $('[data-font-size]').textContent='Text · '+sizes[sizeIndex].label;
  }
  applyFontSize();
  $$('[data-bilinear-diagram]').forEach(diagram=>{
    const select=index=>{
      const row=4*Math.floor(index/2)+1, col=4*(index%2)+1;
      $$('[data-bilinear-pixel]',diagram).forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.bilinearPixel)===index)));
      $$('[data-bilinear-panel]',diagram).forEach(panel=>panel.hidden=Number(panel.dataset.bilinearPanel)!==index);
      $$('[data-bilinear-group]',diagram).forEach(group=>group.classList.toggle('is-active',Number(group.dataset.bilinearGroup)===index));
      $$('[data-source-row]',diagram).forEach(cell=>{
        const y=Number(cell.dataset.sourceRow), x=Number(cell.dataset.sourceCol);
        cell.classList.toggle('is-active',y>=row&&y<=row+1&&x>=col&&x<=col+1);
      });
      $('[data-bilinear-description]',diagram).textContent=$(`[data-bilinear-panel="${index}"]`,diagram).getAttribute('aria-label');
    };
    $$('[data-bilinear-pixel]',diagram).forEach(button=>button.addEventListener('click',()=>select(Number(button.dataset.bilinearPixel))));
  });
  $('[data-font-size]')?.addEventListener('click',()=>{sizeIndex=(sizeIndex+1)%sizes.length;applyFontSize();storage.set('study-font-size',sizeIndex);});
  $$('[data-mark]').forEach(button=>button.addEventListener('click',()=>{
    const p=getPageState(page), id=button.dataset.mark;
    p.done=p.done.includes(id)?p.done.filter(x=>x!==id):[...p.done,id];
    if(!storage.set('study-progress-v1',state)) toast('Progress cannot be saved in this browser. You can still keep reading.');
    updateProgress();
  }));
  function revealHash(scroll=true) {
    if(!location.hash) return;
    let id;try{id=decodeURIComponent(location.hash.slice(1));}catch{return;}
    const moved=lookup(data.docs[page]?.redirects,id);
    if(moved) {location.replace(moved);return;}
    const target=document.getElementById(id);
    if(!target) return;
    for(let el=target;el;el=el.parentElement) if(el.tagName==='DETAILS') el.open=true;
    // Markdown can wrap a standalone legacy anchor in a paragraph.
    let alias=target;
    if(target.tagName==='A'&&target.parentElement?.tagName==='P'&&!target.parentElement.textContent.trim()) alias=target.parentElement;
    if(alias.nextElementSibling?.tagName==='DETAILS') alias.nextElementSibling.open=true;
    const scrollTarget=target.hasAttribute('data-heading-alias')?target.parentElement.closest('[id]')||target:target;
    if(scroll) requestAnimationFrame(()=>scrollTarget.scrollIntoView({block:'start',behavior:'auto'}));
  }
  document.addEventListener('click',event=>{
    const link=event.target.closest('a[href^="#"]');
    if(link) {closeMenu();setTimeout(()=>revealHash(),0);}
  });
  window.addEventListener('hashchange',()=>revealHash());
  revealHash();
  $('[data-resume]')?.addEventListener('click',()=>{
    const id=getPageState(page).last || data.docs[page]?.chapters[0]?.id;
    if(id) {location.hash=id;revealHash();}
  });
  $('[data-expand]')?.addEventListener('click',()=>{
    $$('article details').forEach(d=>d.open=true);toast('All chapters and exercises expanded');
  });
  $('[data-collapse]')?.addEventListener('click',()=>{
    $$('article details').forEach(d=>d.open=false);toast('All collapsed. Select a chapter to read.');
  });
  // Save only deliberate browsing, avoiding overwriting the resume location during page load.
  let hasInteracted=false;
  ['wheel','touchstart','keydown','click'].forEach(type=>window.addEventListener(type,()=>hasInteracted=true,{passive:true,once:true}));
  const observer=new IntersectionObserver(entries=>{
    for(const entry of entries) if(entry.isIntersecting) {
      const chapter=entry.target.closest('[data-chapter]');
      if(!chapter) continue;
      const id=chapter.dataset.chapter;
      $$('[data-toc]').forEach(a=>a.classList.toggle('active',a.dataset.toc===id));
      if(hasInteracted) {getPageState(page).last=id;storage.set('study-progress-v1',state);}
    }
  },{rootMargin:'-65px 0px -60% 0px',threshold:0});
  $$('.chapter-summary,.chapter-body h3,.chapter-body h4').forEach(h=>observer.observe(h));
  let scheduled=false;
  function readingProgress(){
    const max=document.documentElement.scrollHeight-innerHeight;
    $('#read-progress').style.width=(max>0?scrollY/max*100:0)+'%';scheduled=false;
  }
  window.addEventListener('scroll',()=>{if(!scheduled){requestAnimationFrame(readingProgress);scheduled=true;}},{passive:true});
  window.addEventListener('resize',readingProgress);readingProgress();
  $$('details').forEach(d=>d.addEventListener('toggle',readingProgress));
  $$('[data-theme-toggle]').forEach(button=>button.addEventListener('click',()=>{
    const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';
    document.documentElement.dataset.theme=theme;
    try{localStorage.setItem('study-theme',theme);}catch{}
    button.setAttribute('aria-label',theme==='dark'?'Switch to light mode':'Switch to dark mode');
  }));
  $('[data-focus]')?.addEventListener('click',event=>{
    const active=document.body.classList.toggle('focus-mode');
    event.currentTarget.setAttribute('aria-pressed',String(active));
    toast(active?'Focus mode on. Use the top-right button to exit.':'Navigation restored');
  });
  function closeMenu(){document.body.classList.remove('menu-open');$('[data-menu]')?.setAttribute('aria-expanded','false');}
  $('[data-menu]')?.addEventListener('click',()=>{
    const active=document.body.classList.toggle('menu-open');$('[data-menu]').setAttribute('aria-expanded',String(active));
  });
  $('[data-close-menu]')?.addEventListener('click',closeMenu);
  let beforePrint=[];
  window.addEventListener('beforeprint',()=>{beforePrint=$$('article details').map(d=>[d,d.open]);beforePrint.forEach(([d])=>d.open=true);});
  window.addEventListener('afterprint',()=>beforePrint.forEach(([d,open])=>d.open=open));
  $('[data-print]')?.addEventListener('click',()=>window.print());
  $$('.copy-code').forEach(button=>button.addEventListener('click',async()=>{
    const text=$('code',button.closest('.code-block')).textContent;
    try {
      if(navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
      else {
        const textarea=document.createElement('textarea');textarea.value=text;textarea.style.cssText='position:fixed;opacity:0';document.body.append(textarea);textarea.select();
        const copied=document.execCommand('copy');textarea.remove();if(!copied) throw new Error('copy failed');
      }
      button.textContent='Copied ✓';setTimeout(()=>button.textContent='Copy code',1600);
    } catch {toast('Clipboard access unavailable. Select the code to copy it manually.');}
  }));
  // Turn the existing numbered rapid-fire lists into locally saved self-test checklists.
  $$('.chapter').filter(c=>/rapid.fire/i.test($('.chapter-summary',c)?.textContent||'')).forEach(chapter=>{
    const quizMigrationKey='study-quiz-network-split-v3:'+page;
    const migrateQuiz=(page==='video'||page==='network')&&!storage.get(quizMigrationKey,false);
    $$('ol > li',chapter).forEach((li,i)=>{
      const key=page+':'+chapter.dataset.chapter+':'+i;
      // The rapid-fire chapter moved from 10 to 9 to 8 to 6 as chapters were removed; questions kept their order.
      if(page==='video'&&storage.get('study-quiz:'+key,null)===null) {
        for(const priorChapter of ['8-rapid-fire-drill--video','9-rapid-fire-drill--video','10-rapid-fire-drill--video']) {
          if(priorChapter===chapter.dataset.chapter) continue;
          const prior=storage.get('study-quiz:video:'+priorChapter+':'+i,null);
          if(prior!==null) {storage.set('study-quiz:'+key,prior);break;}
        }
      }
      if(migrateQuiz&&storage.get('study-quiz:'+key,null)===null) {
        const question=li.textContent.replace(/\s+/g,' ').trim();
        const previousIndex=page==='video'?(i<31?i:i+9):split?.deliveryQuestions.find(q=>q.text===question||q.currentText===question)?.index;
        if(previousIndex!==undefined) {
          for(const oldChapter of ['11-rapid-fire-drill--video','12-rapid-fire-drill--video']) {
            const prior=storage.get('study-quiz:video:'+oldChapter+':'+previousIndex,null);
            if(prior!==null) {storage.set('study-quiz:'+key,prior);break;}
          }
        }
      }
      const label=document.createElement('label');label.className='drill-check';
      const input=document.createElement('input');input.type='checkbox';input.checked=storage.get('study-quiz:'+key,false);input.setAttribute('aria-label','Mark this question as mastered');
      label.append(input);li.prepend(label);li.classList.toggle('answered',input.checked);
      input.addEventListener('change',()=>{storage.set('study-quiz:'+key,input.checked);li.classList.toggle('answered',input.checked);});
    });
    if(migrateQuiz) storage.set(quizMigrationKey,true);
  });
  // Interview questions: a locally saved "reviewed" checkbox before every question, and a running count per chapter.
  // ML chapter 2 marks its numbered bold questions; the company-interviews book marks every heading whose first
  // paragraph is "Asked:". Keys come from the question text without its number (company questions also carry the
  // company), never from numbers or heading ids, so renumbering or adding a company keeps every tick in place.
  const reviewKey=text=>text.toLowerCase().replace(/^\d+(?:[.-]\d+)*[.\s-]+/,'').replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'');
  const reviewText=host=>[...host.childNodes].filter(node=>!(node.nodeType===1&&node.matches('.heading-anchor,.review-check'))).map(node=>node.textContent).join('').trim();
  const reviewLists=[];
  if(page==='ml') {
    const chapter=$('#chapter-2-interview-questions--transformers--llms');
    if(chapter) reviewLists.push({chapter,keyOf:host=>reviewKey(host.firstElementChild.textContent),
      hosts:$$('.prose > p',chapter).filter(p=>p.firstElementChild?.tagName==='STRONG'&&p.firstChild===p.firstElementChild&&/^\d+\.\s.*\?$/.test(p.textContent.trim()))});
  }
  if(page==='interviews') $$('.chapter').forEach(chapter=>reviewLists.push({chapter,
    keyOf:host=>reviewKey(chapter.dataset.chapter||'')+'/'+reviewKey(reviewText(host)),
    legacyKeyOf:host=>reviewKey(host.id),   // the first version keyed company questions by heading id
    hosts:$$('.prose :is(h3,h4,h5,h6)',chapter).filter(heading=>{const next=heading.nextElementSibling;return next?.tagName==='P'&&/^Asked:/.test(next.firstElementChild?.textContent.trim()||'');})}));
  const reviewSeen=new Set();
  let reviewIndex=0;
  reviewLists.filter(list=>list.hosts.length).forEach(({chapter,hosts,keyOf,legacyKeyOf})=>{
    const count=document.createElement('p');count.className='review-count';count.setAttribute('aria-live','polite');
    const boxes=[];
    const update=()=>{count.textContent=`Reviewed ${boxes.filter(box=>box.checked).length} of ${boxes.length} questions`;};
    hosts.forEach(host=>{
      let key=keyOf(host)||'question';
      while(reviewSeen.has(key)) key+='-2';
      reviewSeen.add(key);
      const storeKey='study-quiz:'+page+':review:'+key;
      if(legacyKeyOf&&storage.get(storeKey,null)===null) {
        const legacy=storage.get('study-quiz:'+page+':review:'+legacyKeyOf(host),null);
        if(legacy!==null) storage.set(storeKey,legacy);
      }
      // The question keeps its own accessible name; the checkbox reads "Mark as reviewed", described by the question.
      let text=host.firstElementChild;
      if(/^H[3-6]$/.test(host.tagName)) {
        text=document.createElement('span');
        [...host.childNodes].filter(node=>!(node.nodeType===1&&node.matches('.heading-anchor'))).forEach(node=>text.append(node));
        host.prepend(text);
      }
      text.id||=('review-q-'+(++reviewIndex));
      if(/^H[3-6]$/.test(host.tagName)) host.setAttribute('aria-labelledby',text.id);
      const label=document.createElement('label');label.className='review-check';label.title='Mark as reviewed';
      const input=document.createElement('input');input.type='checkbox';input.checked=storage.get(storeKey,false)===true;input.dataset.reviewKey=key;
      input.setAttribute('aria-label','Mark as reviewed');input.setAttribute('aria-describedby',text.id);
      label.append(input);host.prepend(label);host.classList.toggle('reviewed',input.checked);boxes.push(input);
      input.addEventListener('change',()=>{
        if(!storage.set(storeKey,input.checked)) toast('Progress cannot be saved in this browser. You can still keep reading.');
        host.classList.toggle('reviewed',input.checked);update();
      });
    });
    $('.prose',chapter)?.prepend(count);update();
  });
  // Offline full-text search: no fetch, backend or network access is required.
  const dialog=$('.search-dialog'), input=$('#search-input'), results=$('.search-results');
  const normalized=data.search.map(item=>({...item,haystack:(item.title+'\n'+item.text).toLowerCase(),titleLower:item.title.toLowerCase()}));
  let scope='all',selected=-1,previousFocus=null,searchTimer;
  function openSearch(preset=''){
    previousFocus=document.activeElement;closeMenu();if(!dialog.open) dialog.showModal();input.value=preset;renderSearch();input.focus();
  }
  function closeSearch(){dialog.close();previousFocus?.focus();}
  $$('[data-open-search]').forEach(b=>b.addEventListener('click',()=>openSearch()));
  $$('[data-search-preset]').forEach(b=>b.addEventListener('click',()=>openSearch(b.dataset.searchPreset)));
  $('[data-close-search]').addEventListener('click',closeSearch);
  dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeSearch();}});
  dialog.addEventListener('close',()=>{input.removeAttribute('aria-activedescendant');input.setAttribute('aria-expanded','false');});
  $$('[data-scope]').forEach(b=>b.addEventListener('click',()=>{
    scope=b.dataset.scope;$$('[data-scope]').forEach(x=>x.classList.toggle('active',x===b));renderSearch();input.focus();
  }));
  function renderSearch(){
    selected=-1;input.removeAttribute('aria-activedescendant');input.setAttribute('aria-expanded','true');const q=input.value.trim().toLowerCase();
    if(!q){results.innerHTML='<div class="search-empty"><strong>What would you like to review?</strong><p>Try KV cache, chroma, motion vector, DPO, or Coding.</p></div>';$('.search-status').textContent='Enter keywords to search all books';return;}
    const words=q.split(/\s+/);
    const hits=normalized.filter(x=>(scope==='all'||x.page===scope)&&words.every(w=>x.haystack.includes(w))).map(x=>({...x,score:(x.titleLower.includes(q)?12:0)+words.filter(w=>x.titleLower.includes(w)).length*4})).sort((a,b)=>b.score-a.score);
    $('.search-status').textContent=`Found ${hits.length} ${hits.length===1?'result':'results'}${hits.length>60?' · Showing the first 60':''}`;
    results.innerHTML=hits.slice(0,60).map((x,i)=>{
      const matchAt=Math.max(0,x.text.toLowerCase().indexOf(words[0]));
      const start=Math.max(0,matchAt-50), snippet=(start?'…':'')+x.text.slice(start,start+185).replace(/\s+/g,' ');
      return `<a class="search-result" id="search-result-${i}" role="option" aria-selected="false" href="${esc(x.url)}"><span class="result-path">${esc(x.pageTitle)}</span><strong class="result-title">${esc(x.title)}</strong><span class="result-context">${esc(snippet)}</span></a>`;
    }).join('')||'<div class="search-empty"><strong>No matching notes</strong><p>Try a shorter keyword or switch the scope to All.</p></div>';
    $$('.search-result',results).forEach(link=>link.addEventListener('click',()=>{dialog.close();setTimeout(revealHash,0);}));
  }
  input.addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(renderSearch,90);});
  dialog.addEventListener('keydown',event=>{
    const links=$$('.search-result',results);
    if(['ArrowDown','ArrowUp'].includes(event.key)&&links.length){
      event.preventDefault();selected=(selected+(event.key==='ArrowDown'?1:-1)+links.length)%links.length;
      links.forEach((a,i)=>{a.classList.toggle('selected',i===selected);a.setAttribute('aria-selected',String(i===selected));});
      input.setAttribute('aria-activedescendant',links[selected].id);links[selected].scrollIntoView({block:'nearest'});
    }else if(event.key==='Enter'&&document.activeElement===input&&links.length){event.preventDefault();links[Math.max(0,selected)].click();}
  });
  document.addEventListener('keydown',event=>{
    if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='k'){event.preventDefault();openSearch();}
    if(event.key==='Escape') closeMenu();
  });
  window.addEventListener('storage',event=>{if(event.key==='study-progress-v1'){state=storage.get('study-progress-v1',{});updateProgress();}});
})();
