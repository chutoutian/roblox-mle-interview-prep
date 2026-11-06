import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Marked } from 'marked';
import hljs from 'highlight.js';
import katex from 'katex';
import { renderBilinearDiagram } from './bilinear-diagram.mjs';
import { renderJpegDiagram } from './jpeg-diagram.mjs';
import { renderCodecDiagram } from './codec-diagram.mjs';
import { renderNetworkDiagram, networkDiagramIds } from './network-diagrams.mjs';
import { renderTransformerDiagram, transformerDiagramIds } from './transformer-diagrams.mjs';
import { buildCvDesignPack } from './build-roblox-cv-design.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const write = (name, value) => fs.writeFileSync(path.join(root, name), value);
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
const videoNavigation = JSON.parse(read('scripts/video-navigation.json'));
const networkNavigation = JSON.parse(read('scripts/network-navigation.json'));
const mlNavigation = JSON.parse(read('scripts/ml-navigation.json'));
// ML chapter 8 (generative models) moved to 03_video_ml: its ids, and every alias that ended at them, redirect across books.
const mlRedirects = {...(mlNavigation.redirects||{})};
for (const [alias, target] of Object.entries(mlNavigation.aliases)) if (mlRedirects[target]) { mlRedirects[alias] = mlRedirects[target]; delete mlNavigation.aliases[alias]; }
const resumeNavigation = JSON.parse(read('scripts/resume-navigation.json'));
const videoAliases = {...networkNavigation.videoAliases};
const videoRedirects = {...networkNavigation.redirects};
for (const [alias, previous] of Object.entries(videoNavigation.aliases)) {
  if (videoRedirects[previous]) videoRedirects[alias] = videoRedirects[previous];
  else videoAliases[alias] = videoAliases[previous] || previous;
}
// Removing chapters (Codec Standards, then Rate Control) renumbered every later chapter: walk each renumbering in order so every alias points at the current ids.
for (const step of videoNavigation.renumbers || []) {
  for (const [alias, target] of Object.entries(videoAliases)) videoAliases[alias] = step.aliases[target] || target;
  for (const [alias, target] of Object.entries(step.aliases)) videoAliases[alias] = target;
  // A chapter that moved to another book: its ids, and every alias that ends at them, become cross-book redirects.
  for (const [id, destination] of Object.entries(step.redirects || {})) videoRedirects[id] = destination;
  for (const [alias, target] of Object.entries(videoAliases)) if (step.redirects?.[target]) { videoRedirects[alias] = step.redirects[target]; delete videoAliases[alias]; }
}
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const plain = s => String(s).replace(/\$\$[\s\S]*?\$\$/g, ' ').replace(/\$(?!\s)(?:\\\$|[^$\n])+?(?<!\s)\$(?!\d)/g, ' ').replace(/<[^>]*>/g, '').replace(/!?(\[([^\]]+)\])\([^)]*\)/g, '$2').replace(/[`*_]/g, '').trim();
const slug = s => plain(s).toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
const reference = 'https://alisawuffles.notion.site/alisa-s-book-of-llms';
// Math: $$...$$ blocks and $...$ spans become static KaTeX HTML at build time; the stylesheet and
// fonts are copied to assets/katex so the pages stay offline. A bad formula fails the build.
const inlineMathRule = /^\$(?!\s)((?:\\\$|[^$\n])+?)(?<!\s)\$(?!\d)/;
const blockMathRule = /^\$\$[ \t]*\n?([\s\S]+?)\n?[ \t]*\$\$[ \t]*(?:\n+|$)/;
const renderMath = (tex, displayMode) => katex.renderToString(tex, {displayMode, throwOnError: true});
const mathExtension = {extensions: [
  {name: 'mathBlock', level: 'block',
    start(src) { return src.match(/(?:^|\n)\$\$/)?.index; },
    tokenizer(src) { const match = blockMathRule.exec(src); if (match) return {type: 'mathBlock', raw: match[0], text: match[1].trim()}; },
    renderer(token) { return `<div class="math-block">${renderMath(token.text, true)}</div>\n`; }},
  {name: 'mathInline', level: 'inline',
    start(src) { return src.match(/\$(?!\s)/)?.index; },
    tokenizer(src) { const match = inlineMathRule.exec(src); if (match) return {type: 'mathInline', raw: match[0], text: match[1].replace(/\\\$/g, '$')}; },
    renderer(token) { return renderMath(token.text, false); }},
]};
const iconPaths = {
  book: '<path d="M4 4h6a3 3 0 0 1 3 3v14a4 4 0 0 0-4-2H4z"/><path d="M13 7a3 3 0 0 1 3-3h5v15h-5a4 4 0 0 0-3 2"/>',
  video: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m10 8 6 4-6 4zM7 4v16M17 4v16"/>',
  network: '<rect x="9" y="2" width="6" height="5" rx="1"/><rect x="2" y="17" width="6" height="5" rx="1"/><rect x="16" y="17" width="6" height="5" rx="1"/><path d="M12 7v5M5 17v-5h14v5"/>',
  ml: '<rect x="4" y="7" width="16" height="13" rx="3"/><path d="M12 3v4M2 12h2m16 0h2M9 16h6"/><circle cx="8" cy="11" r=".7"/><circle cx="16" cy="11" r=".7"/>',
  cross: '<path d="M4 6h7v7H4zM13 11h7v7h-7zM7 16v4h10M17 8V4H7"/>',
  resume: '<path d="M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7"/>',
  search: '<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  theme: '<path d="M20 14A8 8 0 0 1 10 4a8 8 0 1 0 10 10Z"/>',
  focus: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  code: '<path d="m8 5-6 7 6 7m8-14 6 7-6 7M14 3l-4 18"/>'
};
const icon = name => `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${iconPaths[name]||iconPaths.book}</svg>`;
const docs = [
  {id:'home',file:'readme.md',out:'index.html',icon:'book',label:'Home',title:"Lingzhi’s book of Video & ML",eyebrow:'A PERSONAL STUDY BOOK',desc:'From video pixels to multimodal models. Understand the principles, write the code, and explain your research clearly.'},
  {id:'video',file:'01_video_domain.md',out:'01_video_domain.html',icon:'video',label:'Video domain',title:'Video domain',eyebrow:'01 / PIXELS → IMAGE CODECS → VIDEO CODECS',desc:'Pixels, image codecs, and the video encode pipeline with its rate control. Understand how frames are represented and compressed.'},
  {id:'ml',file:'02_ml_domain.md',out:'02_ml_domain.html',icon:'ml',label:'ML & LLMs',title:'ML & LLMs',eyebrow:'02 / TENSORS → TRANSFORMERS → SYSTEMS',desc:'Follow an input through a complete Transformer: embeddings, multi-head attention, normalization, feed-forward layers, and the decoder. Then connect the model to training and inference systems.'},
  {id:'resume',file:'04_resume_drill.md',out:'04_resume_drill.html',icon:'resume',label:'Research & resume',title:'Research & resume',eyebrow:'04 / KNOW YOUR OWN WORK',desc:'Practice a 90-second project pitch and prepare for follow-up questions. Explain the insight, evidence, and limitations of each project.'},
  {id:'leetcode',file:'06_leetcode_notes.md',out:'06_leetcode_notes.html',icon:'code',label:'LeetCode Notes',title:'LeetCode Notes',eyebrow:'06 / TEMPLATES & EXAMPLES',desc:'Notes for coding practice.'},
  {id:'system-design',file:'07_system_design.md',out:'07_system_design.html',icon:'network',label:'System Design',title:'System Design',eyebrow:'07 / REQUIREMENTS → DESIGN → TRADE-OFFS',desc:'A practical framework for designing systems.'},
  {id:'interviews',file:'08_company_interviews.md',out:'08_company_interviews.html',icon:'cross',label:'Company Interviews',title:'Company Interviews',eyebrow:'08 / LEETCODE → ML → SYSTEM DESIGN → BQ',desc:'Interview questions by company: LeetCode, ML, system design and behavioral, each with its source, the candidate\'s level and an answer.'},
  {id:'network',file:'05_networking_streaming.md',out:'05_networking_streaming.html',icon:'network',label:'Networking & Streaming',title:'Networking & Streaming',eyebrow:'09 / PACKETS → PROTOCOLS → PLAYBACK',desc:'Follow a packet from sender to screen. Learn TCP, UDP, QUIC, adaptive streaming, and real-time media through diagrams and worked examples.'},
];
const documentLinks = Object.fromEntries(docs.map(d=>[d.file,d.out]));
const search = [];
const sourceFiles = new Set();
const allMeta = {};
const sourcePage = file => 'source-' + file.replace(/\.py$/,'.html').replace(/\.cpp$/,'-cpp.html');
function practiceRegion(file, id, language) {
  const prefix=language==='python'?'#':'//';
  const lines=read(file).split(/\r?\n/);
  const begins=lines.flatMap((line,i)=>line.trim()===`${prefix} BEGIN ${id}`?[i]:[]);
  const ends=lines.flatMap((line,i)=>line.trim()===`${prefix} END ${id}`?[i]:[]);
  if(begins.length!==1||ends.length!==1||ends[0]<=begins[0]) throw new Error(`Invalid practice region: ${file} ${id}`);
  return '\n```'+language+'\n'+lines.slice(begins[0]+1,ends[0]).join('\n').trim()+'\n```\n';
}
function expandPracticeCode(markdown) {
  return markdown
    .replace(/<!-- transformer-code: ([a-z-]+) -->/g, (_, id) => practiceRegion('transformer_encoder_decoder.py',id,'python'))
    .replace(/<!-- vlm-code: ([a-z-]+) -->/g, (_, id) => practiceRegion('vlm_vit_projector_llm.py',id,'python'))
    .replace(/<!-- vit-code: ([a-z-]+) -->/g, (_, id) => practiceRegion('vision_transformers.py',id,'python'))
    .replace(/<!-- lora-code: ([a-z-]+) -->/g, (_, id) => practiceRegion('lora_linear.py',id,'python'))
    .replace(/<!-- inference-code: ([a-z-]+) -->/g, (_, id) => practiceRegion('inference_kv_cache.py',id,'python'))
}
function rewriteLink(href) {
  if (/^(?:https?:|mailto:|#)/i.test(href)) return href;
  let [file, hash] = href.replace(/^\.\//,'').split('#');
  if (documentLinks[file]) return documentLinks[file] + (hash ? '#' + hash : '');
  if (/\.(?:py|cpp)$/.test(file) && fs.existsSync(path.join(root,file))) {
    sourceFiles.add(file);
    return sourcePage(file) + (hash ? '#' + hash : '');
  }
  return href;
}
function parseDoc(doc) {
  const md = new Marked({gfm:true,breaks:false});
  md.use(mathExtension);
  const ids = new Set();
  const headings = [];
  const legacyAliases = doc.id==='video' ? videoAliases : doc.id==='ml' ? mlNavigation.aliases : doc.id==='resume' ? resumeNavigation.aliases : {};
  const aliasesFor = id => Object.entries(legacyAliases).filter(([,target])=>target===id).map(([alias])=>`<span id="${esc(alias)}" data-heading-alias></span>`).join('');
  let first = true;
  const tokens = md.lexer(expandPracticeCode(read(doc.file)));
  md.walkTokens(tokens, token=>{
    if (token.type!=='heading') return;
    if (first) { first=false; token.skipTitle=true; return; }
    const base=slug(token.text)||'section';
    let id=base, n=1;
    while(ids.has(id)) id=`${base}-${n++}`;
    ids.add(id);token.bookId=id;
    token.explicitId=false;
    headings.push({id,title:plain(token.text),depth:token.depth});
  });
  for(const [alias,target] of Object.entries(legacyAliases)) {
    if(!ids.has(target)) throw new Error(`${doc.file}: alias ${alias} has no destination ${target}`);
    if(ids.has(alias)) throw new Error(`${doc.file}: alias ${alias} duplicates a heading`);
    ids.add(alias);
  }
  md.use({renderer:{
    html({text}) {
      if (text.trim()==='<!-- bilinear-8-to-2 -->') return renderBilinearDiagram();
      if (text.trim()==='<!-- jpeg-encoding-pipeline -->') return renderJpegDiagram();
      if (text.trim()==='<!-- codec-encode-pipeline -->') return renderCodecDiagram();
      const transformerDiagram=text.trim().match(/^<!-- transformer-diagram: ([a-z-]+) -->$/);
      if (transformerDiagram) {
        if(!transformerDiagramIds.includes(transformerDiagram[1])) throw new Error(`Unknown transformer diagram: ${transformerDiagram[1]}`);
        return renderTransformerDiagram(transformerDiagram[1]);
      }
      const networkDiagram=text.trim().match(/^<!-- network-diagram: ([a-z-]+) -->$/);
      if (networkDiagram) {
        if(!networkDiagramIds.includes(networkDiagram[1])) throw new Error(`Unknown network diagram: ${networkDiagram[1]}`);
        return renderNetworkDiagram(networkDiagram[1]);
      }
      return text;
    },
    heading(token) {
      if (token.skipTitle) return '';
      const depth=Math.min(6,token.depth+1);
      return `<h${depth}${token.explicitId?'':` id="${esc(token.bookId)}"`}>${aliasesFor(token.bookId)}${this.parser.parseInline(token.tokens)}<a class="heading-anchor" href="#${esc(token.bookId)}" aria-label="Link to ${esc(plain(token.text))}">#</a></h${depth}>\n`;
    },
    code({text,lang}) {
      const language=(lang||'').split(/\s/)[0];
      const html=language&&hljs.getLanguage(language)?hljs.highlight(text,{language,ignoreIllegals:true}).value:esc(text);
      return `<div class="code-block"><div class="code-toolbar"><span class="code-lang">${esc(language||'text')}</span><button class="copy-code button" type="button">Copy code</button></div><pre><code${language?` class="language-${esc(language)}"`:''}>${html}</code></pre></div>\n`;
    },
    link({href,title,tokens}) {
      const external=/^https?:/i.test(href);
      if (/^(?:javascript|data|vbscript):/i.test(href)) return this.parser.parseInline(tokens);
      return `<a href="${esc(rewriteLink(href))}"${title?` title="${esc(title)}"`:''}${external?' target="_blank" rel="noopener noreferrer"':''}>${this.parser.parseInline(tokens)}</a>`;
    },
    table(token) {
      const cell=(c,i,tag)=>`<${tag}${token.align[i]?` style="text-align:${token.align[i]}"`:''}>${this.parser.parseInline(c.tokens)}</${tag}>`;
      return `<div class="table-scroll" tabindex="0" role="region" aria-label="Horizontally scrollable table"><table><thead><tr>${token.header.map((c,i)=>cell(c,i,'th')).join('')}</tr></thead><tbody>${token.rows.map(r=>`<tr>${r.map((c,i)=>cell(c,i,'td')).join('')}</tr>`).join('')}</tbody></table></div>`;
    }
  }});
  // Standalone anchors are sometimes lexed as paragraphs. Normalize both forms before
  // splitting chapters, so an alias immediately before H1 travels with the new chapter.
  const anchorsOnly=/^\s*(?:<a\s+id=["'][^"']+["']\s*><\/a>\s*)+$/;
  for (const t of tokens) if(anchorsOnly.test(t.raw||'')) {
    const aliases=t.raw.replace(/<a\s+id=["']([^"']+)["']\s*><\/a>/g,(full,id)=>ids.has(id)?'':full);
    t.type='html';t.text=aliases;t.raw=aliases;t.block=true;delete t.tokens;
  }
  const chapters=[]; let intro=[]; let current=null;
  for(const token of tokens) {
    if(token.skipTitle) continue;
    if(token.type==='heading'&&token.depth===1) {
      const previous=current?current.tokens:intro;
      const aliases=[];
      while(previous.length && (previous.at(-1).type==='space'||(previous.at(-1).type==='html'&&/^\s*<a\s+id=/i.test(previous.at(-1).text)))) aliases.unshift(previous.pop());
      current={id:token.bookId,title:plain(token.text),token,tokens:aliases};chapters.push(current);
    } else (current?current.tokens:intro).push(token);
  }
  if(!chapters.length) {chapters.push({id:'notes',title:'Notes',tokens:intro});intro=[];}
  function body(tokens) {
    // Keep coding exercises compact; the concept, prompt and answer stay together.
    let html='', i=0;
    while(i<tokens.length) {
      const t=tokens[i];
      if(t.type==='heading'&&t.depth>=3&&/^Coding:/i.test(t.text)) {
        const inner=[]; let j=i+1;
        while(j<tokens.length && !(tokens[j].type==='heading'&&tokens[j].depth<=t.depth) && !(tokens[j].type==='html'&&/^<a\s+id=/i.test(tokens[j].raw))) inner.push(tokens[j++]);
        html+=`<details class="coding-drill"${t.explicitId?'':` id="${esc(t.bookId)}"`}><summary><span>${md.parseInline(t.text)}</span><span class="drill-hint">View exercise</span></summary><div class="drill-body">${md.parser(inner)}</div></details>`;
        i=j;
      } else {html+=md.parser([t]);i++;}
    }
    return html;
  }
  // Build small search records per heading, with text from the whole corresponding subsection.
  let entry={page:doc.id,pageTitle:doc.label,title:doc.label+' · Reading guide and overview',url:doc.out+'#reading-overview',text:''};
  for(const t of tokens) {
    if(t.skipTitle) continue;
    if(t.type==='heading') {
      if(entry) search.push(entry);
      entry={page:doc.id,pageTitle:doc.label,title:plain(t.text),url:doc.out+'#'+t.bookId,text:''};
    } else if(entry) entry.text+=plain(t.raw||'')+'\n';
  }
  if(entry) search.push(entry);
  allMeta[doc.id]={title:doc.label,url:doc.out,chapters:chapters.map(c=>({id:c.id,title:c.title})),sections:headings.length,drills:headings.filter(h=>/^Coding:/i.test(h.title)).length};
  if(doc.id==='video') {
    allMeta[doc.id].navigationMigration=videoNavigation;
    allMeta[doc.id].networkSplitMigration=networkNavigation;
    allMeta[doc.id].renumberMigrations=videoNavigation.renumbers;
    allMeta[doc.id].legacyAliases=videoAliases;
    allMeta[doc.id].redirects=videoRedirects;
  }
  if(doc.id==='resume') {
    allMeta[doc.id].navigationMigration=resumeNavigation;
    allMeta[doc.id].legacyAliases=resumeNavigation.aliases;
  }
  if(doc.id==='ml') {
    allMeta[doc.id].navigationMigration=mlNavigation;
    allMeta[doc.id].legacyAliases=mlNavigation.aliases;
    allMeta[doc.id].redirects=mlRedirects;
  }
  const chapterHTML=chapters.map((c,i)=>{
    const count=c.tokens.filter(t=>t.type==='heading'&&t.depth===2).length;
    const title=c.title.replace(/^\d+\.\s*/, '');
    const initiallyOpen=doc.id==='ml' ? c.id==='3-transformer-from-input-to-output' : i===0&&doc.id!=='home';
    return `<details class="chapter" id="chapter-${esc(c.id)}" data-chapter="${esc(c.id)}"${initiallyOpen?' open':''}>
      <summary class="chapter-summary"${c.token?.explicitId?'':` id="${esc(c.id)}"`}><span class="chapter-number">${String(i+1).padStart(2,'0')}</span><span class="chapter-title">${aliasesFor(c.id)}${esc(title)}</span><span class="chapter-state">${count?`${count} ${count===1?'section':'sections'}`:'Notes'}</span></summary>
      <div class="chapter-body"><div class="chapter-tools"><button type="button" class="button mark-read" data-mark="${esc(c.id)}" aria-pressed="false">${icon('check')}<span>Mark as read</span></button><a class="button" href="#${esc(c.id)}">Chapter link ↗</a></div><div class="prose">${body(c.tokens)}</div></div></details>`;
  }).join('\n');
  const movedNote=doc.id==='video'?`<aside class="book-move-note">${Object.keys(videoRedirects).map(id=>`<span id="${esc(id)}" data-moved-anchor></span>`).join('')}<span>Looking for TCP, HLS/DASH, WebRTC, or CDN delivery?</span> <a href="05_networking_streaming.html">Open Networking &amp; Streaming ${icon('arrow')}</a> <span>Quality assessment now lives in the UGC-VQA chapter of Research &amp; resume.</span> <a href="04_resume_drill.html#6-tiktok--ugc-short-video-quality-assessment-via-mllm">Open ML-Based Quality Assessment ${icon('arrow')}</a></aside>`:'';
  const mlStart=doc.id==='ml'?'<aside class="book-move-note">Start with <a href="#3-transformer-from-input-to-output">the step-by-step Transformer walkthrough</a> or jump to <a href="#2-interview-questions--transformers--llms">the interview questions</a>. <a href="#1-basics">Chapter 1</a> collects the basics to refer back to.</aside>':'';
  return {chapters,headings,content:`${movedNote}${mlStart}${intro.length?`<details class="chapter overview" id="reading-overview"><summary class="chapter-summary"><span class="chapter-title">Reading guide & contents</span></summary><div class="chapter-body prose">${body(intro)}</div></details>`:''}${chapterHTML}`};
}
const parsed=Object.fromEntries(docs.map(d=>[d.id,parseDoc(d)]));
for (const [alias, destination] of [...Object.entries(videoRedirects), ...Object.entries(mlRedirects)]) {
  const [file,id]=destination.split('#'), target=docs.find(d=>d.out===file);
  const hasId=target&&(parsed[target.id].headings.some(h=>h.id===id)||parsed[target.id].content.includes(` id="${id}"`));
  if(!hasId) throw new Error(`Moved bookmark ${alias} has no destination ${destination}`);
}
function nav(doc) {
  return `<a href="index.html" class="brand">${icon('book')}<span>Video & ML<br><small>Lingzhi’s study book</small></span></a>
    <button class="nav-search" type="button" data-open-search>${icon('search')}<span>Search all notes</span><kbd>⌘ K</kbd></button>
    <nav aria-label="Book navigation"><p class="nav-label">WORKSPACE</p>${docs.map(d=>`<a class="nav-link ${d.id===doc.id?'active':''}" ${d.id===doc.id?'aria-current="page"':''} href="${d.out}">${icon(d.icon)}<span>${d.label}</span>${d.id!=='home'?`<small>${allMeta[d.id].chapters.length}</small>`:''}</a>`).join('')}
    <p class="nav-label">YOUR PRACTICE</p><button class="nav-link" type="button" data-search-preset="Coding">${icon('resume')}<span>Coding exercises</span></button><button class="nav-link" type="button" data-search-preset="Rapid-Fire">${icon('check')}<span>Oral self-test</span></button></nav>
    <div class="sidebar-progress"><div><span>Reading progress</span><strong data-total-progress>0%</strong></div><div class="progress-track"><div class="track-fill" data-total-bar></div></div><small data-total-count>0 chapters read</small></div>
    <div class="sidebar-footer"><a class="nav-link" href="${reference}" target="_blank" rel="noopener noreferrer"><span>↗</span><span>Alisa’s book of LLMs</span></a><small>Read offline · Progress saved in this browser</small></div>`;
}
function homeIntro() {
  return `<div class="callout home-callout"><span>✳</span><div><strong>Interview notes to return to</strong><p>Understand the concept, open a coding exercise, then try the rapid-fire questions. Mark chapters as read and pick up where you left off.</p></div></div>
    <div class="section-label"><h2>Choose a book</h2><span>${docs.length-1} books, one learning path</span></div>
    <div class="book-list">${docs.filter(d=>d.id!=='home').map(d=>`<a class="book-row" href="${d.out}"><span class="book-icon">${icon(d.icon)}</span><div class="book-info"><h3>${d.label}</h3><p>${d.desc}</p></div><div class="book-meta"><span>${allMeta[d.id].chapters.length} chapters${allMeta[d.id].drills?` · ${allMeta[d.id].drills} ${allMeta[d.id].drills===1?'exercise':'exercises'}`:''}</span><small data-book-progress="${d.id}">Not started</small></div>${icon('arrow')}</a>`).join('')}</div>
    <div class="section-label"><h2>Suggested reading</h2><span>From fundamentals to your research</span></div><div class="learning-path"><a href="01_video_domain.html"><small>01</small><strong>Understand video</strong><span>Pixels → Image codecs → Video codecs</span></a><a href="02_ml_domain.html"><small>02</small><strong>Build ML foundations</strong><span>Foundations → LLM → Systems</span></a><a href="04_resume_drill.html"><small>04</small><strong>Explain your research</strong><span>Insight → Evidence → Limits</span></a><a href="06_leetcode_notes.html"><small>06</small><strong>Practice algorithms</strong><span>Templates → Examples → Practice</span></a><a href="05_networking_streaming.html"><small>09</small><strong>Follow the network</strong><span>Packets → Streaming → Playback</span></a></div>
    <p class="source-note">The reading layout and ML learning sequence are inspired by <a href="${reference}" target="_blank" rel="noopener noreferrer">Alisa’s book of LLMs ↗</a>. The video and research books retain the original notes, with full content and links to local code below.</p><div class="section-label"><h2>Interview field notes</h2><span>Strategy · Roles · Review checklists</span></div>`;
}
function template(doc, {content,chapters=[]}, source=false) {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>${esc(doc.title)} · Video & ML</title><meta name="description" content="${esc(doc.desc)}"><link rel="stylesheet" href="assets/book.css"><link rel="stylesheet" href="assets/code.css"><link rel="stylesheet" href="assets/katex/katex.min.css">${doc.id==='video'?'<link rel="stylesheet" href="assets/jpeg.css">':''}${doc.id==='network'?'<link rel="stylesheet" href="assets/network.css">':''}${doc.id==='ml'?'<link rel="stylesheet" href="assets/transformer.css">':''}<script>try{document.documentElement.dataset.theme=localStorage.getItem('study-theme')||'light'}catch{}</script><script src="assets/book-data.js" defer></script><script src="assets/book.js" defer></script>${doc.id==='network'?'<script src="assets/network.js" defer></script>':''}${doc.id==='ml'?'<script src="assets/transformer.js" defer></script>':''}</head>
<body data-page="${doc.id}"><a class="skip-link" href="#main-content">Skip to content</a><div class="read-progress" aria-hidden="true"><div class="track-fill" id="read-progress"></div></div>
<aside class="sidebar" id="sidebar">${nav(doc)}</aside><button class="sidebar-backdrop" aria-label="Close navigation" type="button" data-close-menu></button>
<header class="topbar"><div class="breadcrumb"><button class="icon-button mobile-menu" aria-label="Open navigation" aria-expanded="false" aria-controls="sidebar" type="button" data-menu>${icon('menu')}</button><a href="index.html">${icon('book')}<span>Study book</span></a><span class="breadcrumb-divider">/</span><span>${esc(doc.label)}</span></div><div class="topbar-actions"><button class="icon-button" type="button" data-open-search aria-label="Search notes">${icon('search')}</button><button class="icon-button" type="button" data-focus aria-label="Focus mode" aria-pressed="false" title="Focus mode">${icon('focus')}</button><button class="icon-button" type="button" data-theme-toggle aria-label="Switch to dark mode" title="Toggle color theme">${icon('theme')}</button><button class="button print-button" type="button" data-print>Print / PDF</button></div></header>
<main class="main" id="main-content"><div class="page-shell"><header class="page-header"><div class="page-icon">${icon(doc.icon)}</div><p class="eyebrow">${doc.eyebrow}</p><h1>${esc(doc.title)}</h1><p class="page-description">${esc(doc.desc)}</p><div class="page-meta"><span>${source?(doc.file.endsWith('.cpp')?'C++ SOURCE':'PYTHON SOURCE'):doc.id==='home'?'PERSONAL KNOWLEDGE BASE':`${chapters.length} CHAPTERS`}</span><span>·</span><a href="${esc(doc.file)}" download>Download source ↗</a>${doc.id!=='home'&&!source?'<span>·</span><span data-page-progress>0 chapters read</span>':''}</div></header>
${doc.id==='home'?homeIntro():''}
${doc.id==='ml'?`<p class="source-note">The learning sequence follows <a href="${reference}" target="_blank" rel="noopener noreferrer">Alisa’s book of LLMs ↗</a>, connecting the original interview notes with principles, formulas, code, and system costs.</p>`:''}
${!source?`<div class="page-actions"><button class="button primary-button" type="button" data-resume>Start reading ${icon('arrow')}</button><div><button class="button" type="button" data-font-size aria-label="Adjust reading font size">Text · Default</button><button class="button" type="button" data-expand>Expand all</button><button class="button" type="button" data-collapse>Collapse all</button></div></div>`:''}
<div class="reading-layout"><article class="reading-content">${content}</article>${!source?`<nav class="toc" aria-label="Table of contents"><p class="nav-label">ON THIS PAGE</p>${chapters.map((c,i)=>`<a href="#${esc(c.id)}" data-toc="${esc(c.id)}"><span>${String(i+1).padStart(2,'0')}</span>${esc(c.title.replace(/^\d+\.\s*/,''))}</a>`).join('')}<a href="#main-content" class="toc-top">↑ Back to top</a></nav>`:''}</div>
<footer class="site-footer"><span>Video & ML · Notes, code & questions.</span><a href="#main-content">Back to top ↑</a></footer></div></main>
<dialog class="search-dialog" aria-label="Search all notes"><div class="search-head">${icon('search')}<input id="search-input" type="search" role="combobox" aria-autocomplete="list" aria-controls="search-results" aria-expanded="false" placeholder="Search concepts, questions, code…" aria-label="Search keywords" autocomplete="off"><button class="button" type="button" data-close-search>Esc</button></div><div class="search-filters" role="group" aria-label="Search scope"><button class="button active" type="button" data-scope="all">All</button>${docs.filter(d=>d.id!=='home').map(d=>`<button class="button" type="button" data-scope="${d.id}">${d.label}</button>`).join('')}</div><p class="search-status" role="status" aria-live="polite">Enter keywords to search all books</p><div class="search-results" id="search-results" role="listbox" aria-label="Search results"></div><div class="search-footer">↑ ↓ Select · Enter Open · Esc Close</div></dialog><div class="toast" role="status" aria-live="polite"></div></body></html>`;
}
for(const doc of docs) {write(doc.out,template(doc,parsed[doc.id]));console.log(`${doc.out}: ${parsed[doc.id].chapters.length} chapters`);}
write('readme.html',read('index.html'));
for(const file of sourceFiles) {
  const code=read(file);
  // Highlight each line independently so multiline spans cannot cross line-anchor wrappers.
  const language=file.endsWith('.cpp')?'cpp':'python';
  const lines=code.split('\n').map(line=>hljs.highlight(line,{language,ignoreIllegals:true}).value);
  const content=`<div class="source-code"><div class="code-toolbar"><span>${esc(file)}</span><a href="${esc(file)}" download class="button">Download ${esc(path.extname(file))}</a></div><pre><code>${lines.map((line,i)=>`<span class="source-line" id="L${i+1}"><a class="line-number" href="#L${i+1}" aria-label="Line ${i+1}">${i+1}</a>${line||' '}</span>`).join('\n')}</code></pre></div>`;
  write(sourcePage(file),template({id:'code',label:file,title:file,file,icon:'resume',eyebrow:'LOCAL CODING COMPANION',desc:'Local practice source code. The READY / TOY / TODO labels in the notes describe the status of each implementation.'},{content},true));
}
write('assets/book-data.js',`window.STUDY_BOOK=${JSON.stringify({docs:allMeta,search}).replace(/</g,'\\u003c')};\n`);
fs.copyFileSync(path.join(root,'node_modules/highlight.js/styles/github.css'),path.join(root,'assets/code.css'));
const katexDist=path.join(root,'node_modules/katex/dist');
fs.mkdirSync(path.join(root,'assets/katex/fonts'),{recursive:true});
fs.copyFileSync(path.join(katexDist,'katex.min.css'),path.join(root,'assets/katex/katex.min.css'));
for(const font of fs.readdirSync(path.join(katexDist,'fonts')).filter(name=>name.endsWith('.woff2'))) fs.copyFileSync(path.join(katexDist,'fonts',font),path.join(root,'assets/katex/fonts',font));
buildCvDesignPack();
console.log(`Built ${docs.length+1} reading pages, ${sourceFiles.size} source viewers, ${search.length} search entries.`);
