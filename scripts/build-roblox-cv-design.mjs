// Offline reading companions for original CV modeling mock documents.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Marked} from 'marked';
import katex from 'katex';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const directory = path.join(root, 'ml_modeling_practice/roblox_cv');
export const cvDesignCases = [
  {slug: '01_visual_moderation', title: 'Visual safety moderation', walkthrough: true},
  {slug: '02_game_genre', title: 'Game genre classification', walkthrough: true},
  {slug: '03_multimodal_asset_retrieval', title: 'Text / image asset retrieval', walkthrough: true},
];
const summarySlug = '04_interview_playbook';
const esc = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const slug = value => value.replace(/<[^>]+>/g, '').toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
const inlineMathRule = /^\$(?!\s)((?:\\\$|[^$\n])+?)(?<!\s)\$(?!\d)/;
const blockMathRule = /^\$\$[ \t]*\n?([\s\S]+?)\n?[ \t]*\$\$[ \t]*(?:\n+|$)/;
const mathExtension = {extensions: [
  {name:'mathBlock', level:'block', start(source){return source.match(/(?:^|\n)\$\$/)?.index;},
    tokenizer(source){const m=blockMathRule.exec(source);if(m)return{type:'mathBlock',raw:m[0],text:m[1].trim()};},
    renderer(token){return `<div class="math-block">${katex.renderToString(token.text,{displayMode:true,throwOnError:true})}</div>\n`;}},
  {name:'mathInline', level:'inline', start(source){return source.match(/\$(?!\s)/)?.index;},
    tokenizer(source){const m=inlineMathRule.exec(source);if(m)return{type:'mathInline',raw:m[0],text:m[1].replace(/\\\$/g,'$')};},
    renderer(token){return katex.renderToString(token.text,{displayMode:false,throwOnError:true});}},
]};

function render(file, mode) {
  const source = fs.readFileSync(path.join(directory, file), 'utf8');
  if (mode!=='summary'&&/[\u3400-\u9fff]/u.test(source)) throw new Error(`${file}: prose must be English`);
  const title = source.match(/^# (.+)$/m)?.[1];
  if (!title) throw new Error(`${file}: missing title`);
  const headings = [], used = new Set();
  const md = new Marked({gfm:true});
  md.use(mathExtension);
  md.use({renderer: {
    heading(token) {
      if(token.depth===1)return '';
      const question=mode==='detailed'&&token.text.match(/^Q(\d+)\./);
      const base=question?`q${question[1]}`:slug(token.text)||'section';let id=base,n=1;
      while(used.has(id))id=`${base}-${n++}`;
      used.add(id);headings.push({id,text:token.text.replace(/\*|`/g,''),depth:token.depth});
      return `<h${token.depth} id="${esc(id)}">${this.parser.parseInline(token.tokens)}<a class="heading-anchor" href="#${esc(id)}" aria-label="Link to section">#</a></h${token.depth}>\n`;
    },
    code({text,lang}) {return `<div class="design-diagram" role="region" tabindex="0" aria-label="${esc(lang==='text'?'System flow diagram':'Code example')}"><pre><code>${esc(text)}</code></pre></div>\n`;},
    link({href,title,tokens}) {
      if(/^(?:javascript|data|vbscript):/i.test(href))return this.parser.parseInline(tokens);
      const local=href.replace(/^README\.md(?=#|$)/,'index.html').replace(/^(0[1-4]_[a-z_]+)\.md(?=#|$)/,'$1.html');
      return `<a href="${esc(local)}"${title?` title="${esc(title)}"`:''}${/^https?:/.test(local)?' target="_blank" rel="noopener noreferrer"':''}>${this.parser.parseInline(tokens)}</a>`;
    },
    table(token) {
      const cell=(c,tag)=>`<${tag}>${this.parser.parseInline(c.tokens)}</${tag}>`;
      return `<div class="table-scroll" tabindex="0" role="region" aria-label="Comparison table"><table><thead><tr>${token.header.map(c=>cell(c,'th')).join('')}</tr></thead><tbody>${token.rows.map(r=>`<tr>${r.map(c=>cell(c,'td')).join('')}</tr>`).join('')}</tbody></table></div>`;
    },
  }});
  let content=md.parse(source);
  if(mode==='questions')content=content.replace(/<li>\s*<strong>Q(\d+)\.<\/strong>/g,(_,n)=>`<li class="design-question" id="q${n}"><label class="design-check"><input type="checkbox" data-question="q${n}" aria-label="Mark follow-up Q${n} as reviewed"></label><strong>Q${n}.</strong>`);
  if(mode==='solution')content=content.replace(/<p><strong>Q(\d+)\.([^<]*)<\/strong>/g,(_,n,title)=>`<p class="followup-answer" id="q${n}"><strong>Q${n}.${title}</strong>`);
  return {title,content,headings};
}

function template({file,mode,caseInfo}) {
  const {title,content,headings}=render(file,mode);
  const isHome=mode==='index';
  const detailed=mode==='detailed';
  const summary=mode==='summary';
  const nav=cvDesignCases.map((item,i)=>`<section class="case-nav"><p>${String(i+1).padStart(2,'0')} · ${esc(item.title)}</p><a href="${item.slug}_questions.html"${caseInfo?.slug===item.slug&&mode==='questions'?' aria-current="page"':''}>Questions</a><a href="${item.slug}_solution.html"${caseInfo?.slug===item.slug&&mode==='solution'?' aria-current="page"':''}>Reference answer</a>${item.walkthrough?`<a href="${item.slug}_detailed.html"${caseInfo?.slug===item.slug&&detailed?' aria-current="page"':''}>Detailed walkthrough</a>`:''}</section>`).join('')+`<section class="case-nav"><p>04 · Interview playbook</p><a href="${summarySlug}.html"${summary?' aria-current="page"':''}>Pacing & answer structure</a></section>`;
  const paired=caseInfo?`<div class="document-tabs"><a href="${caseInfo.slug}_questions.html"${mode==='questions'?' aria-current="page"':''}>Question document</a><a href="${caseInfo.slug}_solution.html"${mode==='solution'?' aria-current="page"':''}>Reference answer</a>${caseInfo.walkthrough?`<a href="${caseInfo.slug}_detailed.html"${detailed?' aria-current="page"':''}>Detailed walkthrough</a>`:''}</div>`:'';
  return `<!doctype html>
<html lang="${summary?'zh-CN':'en'}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>${esc(title)} · Video &amp; ML</title><link rel="stylesheet" href="../../assets/book.css"><link rel="stylesheet" href="../../assets/katex/katex.min.css"><link rel="stylesheet" href="../../assets/cv-modeling.css"><script src="../../assets/cv-modeling.js" defer></script></head>
<body data-page="ml-design" data-design="${esc(caseInfo?.slug||(summary?summarySlug:'index'))}" data-mode="${mode}"><a class="skip-link" href="#main-content">Skip to content</a>
<header class="design-topbar"><a href="../../index.html">Video &amp; ML</a><span>/</span><a href="index.html">ML Modeling Knowledge</a><div><button type="button" data-design-theme aria-label="Toggle color theme">Theme</button><button type="button" data-design-print>Print / PDF</button></div></header>
<div class="design-layout"><aside class="design-sidebar"><a class="design-home" href="index.html"${isHome?' aria-current="page"':''}>CV design practice</a>${nav}<div class="source-nav"><a href="../../08_company_interviews.html#13-system-design">Company interview topics ↗</a><a href="../../02_ml_domain.html#26-interview-questions--computer-vision">CV foundations · 2.6 ↗</a></div></aside>
<main id="main-content"><header class="design-header"><p class="eyebrow">${summary?'04 · INTERVIEW PLAYBOOK':isHome?'THREE ORIGINAL MOCKS':mode==='questions'?'PRACTICE · QUESTIONS ONLY':detailed?'DETAILED WALKTHROUGH · EXAMPLES & MATH':'REFERENCE · WORKED DESIGN'}</p><h1>${esc(title)}</h1><p class="design-meta">${summary?'中文总结 · English speaking templates · 45-minute interview':detailed?'English · Spoken interview answers · Q1–Q21 · examples and equations':'45 minutes · 35-minute technical core · English · CV / multimodal'}</p>${paired}<a class="source-download" href="${esc(file)}" download>Download Markdown source ↗</a>${mode==='questions'?'<p class="design-review-count" aria-live="polite">Review checks are available when JavaScript is enabled.</p>':''}</header><article class="prose design-prose">${content}</article><footer class="design-footer"><a href="index.html">All three mocks</a><a href="#main-content">Back to top ↑</a></footer></main>
<nav class="design-toc" aria-label="Document sections"><p class="nav-label">ON THIS PAGE</p>${headings.filter(h=>h.depth===2||(/^[1-7]\. /.test(h.text))).map(h=>`<a href="#${esc(h.id)}">${esc(h.text)}</a>`).join('')}</nav></div>
</body></html>`;
}

export function buildCvDesignPack() {
  const docs=[{file:'README.md',mode:'index'},...cvDesignCases.flatMap(caseInfo=>(caseInfo.walkthrough?['questions','solution','detailed']:['questions','solution']).map(mode=>({file:`${caseInfo.slug}_${mode}.md`,mode,caseInfo}))),{file:`${summarySlug}.md`,mode:'summary'}];
  const prepared=docs.map(doc=>({out:doc.mode==='index'?'index.html':doc.file.replace(/\.md$/,'.html'),html:template(doc)}));
  for(const item of prepared)fs.writeFileSync(path.join(directory,item.out),item.html);
  console.log(`CV modeling design: ${cvDesignCases.length} question/answer pairs + overview + ${cvDesignCases.filter(item=>item.walkthrough).length} detailed walkthroughs + interview playbook (offline HTML).`);
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))buildCvDesignPack();
