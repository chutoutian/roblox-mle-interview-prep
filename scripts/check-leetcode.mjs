import assert from 'node:assert/strict';
import path from 'node:path';

async function checkDrillComplexity(drill) {
  const title=await drill.locator(':scope > summary').textContent();
  const body=drill.locator(':scope > .drill-body');
  assert.equal(await body.count(),1,'Each Coding exercise needs its own body: '+title);
  for(const label of ['Time complexity:','Space complexity:']) {
    const paragraph=body.locator('p').filter({hasText:new RegExp('^'+label+'\\s*\\S')});
    assert.equal(await paragraph.count(),1,'Each Coding exercise needs one nonempty '+label+' paragraph: '+title);
    assert.ok(await paragraph.isVisible(),label+' must be readable in the expanded exercise: '+title);
    assert.equal(await paragraph.evaluate(node=>{
      const codes=node.closest('.drill-body').querySelectorAll('pre code');
      const lastCode=codes[codes.length-1];
      return !!lastCode&&!!(lastCode.compareDocumentPosition(node)&Node.DOCUMENT_POSITION_FOLLOWING);
    }),true,label+' must follow the exercise code, inside its fold: '+title);
  }
}

export async function checkLeetCode({page,context,target,preview}) {
  await page.goto(target('index.html'));
  assert.equal(await page.locator('.book-row').count(),await page.evaluate(()=>Object.keys(window.STUDY_BOOK.docs).filter(id=>id!=='home').length));
  await page.locator('.book-row[href="06_leetcode_notes.html"]').click();
  await page.waitForURL(/06_leetcode_notes\.html$/);
  assert.equal(await page.locator('#sidebar a[href="06_leetcode_notes.html"]').getAttribute('aria-current'),'page');
  const metadata=await page.evaluate(()=>window.STUDY_BOOK.docs.leetcode);
  assert.equal(metadata.chapters.length,5);
  assert.equal(metadata.drills,39);
  assert.deepEqual(metadata.chapters.map(chapter=>chapter.id),['1-binary-search','2-doubly-linked-lists','3-recursion-tree-dfs-backtracking-memoization','4-dijkstra-shortest-path','5-notes-inbox']);
  assert.match(await page.locator('#chapter-5-notes-inbox').textContent(),/No additional topics yet/);
  assert.equal(await page.locator('[data-bs-lab]').count(),0,'The simplified notes must not render the old interactive lab');
  assert.equal(await page.locator('script[src="assets/binary-search.js"],link[href="assets/binary-search.css"]').count(),0);
  await page.locator('[data-open-search]').first().click();
  await page.locator('[data-scope="leetcode"]').click();
  await page.locator('#search-input').fill('lower bound');
  await page.waitForFunction(()=>document.querySelector('.search-status').textContent.startsWith('Found'));
  assert.ok(await page.locator('.search-result').count()>0);
  assert.ok((await page.locator('.result-path').allTextContents()).every(text=>text==='LeetCode Notes'));
  await page.locator('#search-input').fill('LRU Cache');
  await page.waitForFunction(()=>document.querySelector('.search-status').textContent.startsWith('Found')&&[...document.querySelectorAll('.search-result')].some(node=>node.textContent.includes('LRU Cache')));
  assert.ok((await page.locator('.result-path').allTextContents()).every(text=>text==='LeetCode Notes'));
  await page.locator('#search-input').fill('Tree DFS');
  await page.waitForFunction(()=>[...document.querySelectorAll('.search-result')].some(node=>['06_leetcode_notes.html#3-recursion-tree-dfs-backtracking-memoization','06_leetcode_notes.html#32-tree-dfs'].includes(node.getAttribute('href'))));
  assert.ok((await page.locator('.result-path').allTextContents()).every(text=>text==='LeetCode Notes'));
  await page.locator('#search-input').fill('BST deletion');
  await page.waitForFunction(()=>[...document.querySelectorAll('.search-result .result-title')].some(node=>node.textContent.includes('BST deletion')));
  assert.ok((await page.locator('.result-path').allTextContents()).every(text=>text==='LeetCode Notes'));
  await page.locator('.search-result').filter({has:page.locator('.result-title').filter({hasText:'BST deletion'})}).first().click();
  await page.waitForFunction(()=>document.querySelector('#chapter-3-recursion-tree-dfs-backtracking-memoization').open);
  assert.equal(await page.evaluate(()=>{
    let node=document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if(!node)return false;
    for(;node;node=node.parentElement)if(node.tagName==='DETAILS'&&!node.open)return false;
    return true;
  }),true,'A Tree DFS search result must reveal its destination');

  await page.locator('[data-open-search]').first().click();
  await page.locator('[data-scope="leetcode"]').click();
  await page.locator('#search-input').fill('Network Delay');
  await page.waitForFunction(()=>[...document.querySelectorAll('.search-result')].some(node=>(node.getAttribute('href')||'').startsWith('06_leetcode_notes.html#42-743-network-delay-time')));
  assert.ok((await page.locator('.result-path').allTextContents()).every(text=>text==='LeetCode Notes'));
  await page.locator('#search-input').fill('Word Break');
  await page.waitForFunction(()=>[...document.querySelectorAll('.search-result')].some(node=>(node.getAttribute('href')||'').startsWith('06_leetcode_notes.html#346-139-word-break')));
  assert.ok((await page.locator('.result-path').allTextContents()).every(text=>text==='LeetCode Notes'));
  await page.locator('#search-input').fill('Permutations II');
  const permutationResult=page.locator('.search-result').filter({has:page.locator('.result-title').filter({hasText:'Permutations II'})}).first();
  await permutationResult.waitFor();
  assert.ok((await page.locator('.result-path').allTextContents()).every(text=>text==='LeetCode Notes'));
  await permutationResult.click();
  await page.waitForFunction(()=>document.querySelector('#chapter-3-recursion-tree-dfs-backtracking-memoization').open);
  assert.equal(await page.evaluate(()=>{
    let node=document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if(!node)return false;
    for(;node;node=node.parentElement)if(node.tagName==='DETAILS'&&!node.open)return false;
    return true;
  }),true,'A Backtracking search result must reveal its destination and code fold');

  for(const alias of ['2-notes-inbox','3-notes-inbox','4-notes-inbox','6-notes-inbox','7-notes-inbox']) {
    await page.locator('[data-collapse]').click();
    await page.goto(target('06_leetcode_notes.html')+'#'+alias);
    assert.equal(await page.locator('[id="'+alias+'"]').evaluate(node=>node.closest('[data-chapter]')?.dataset.chapter),'5-notes-inbox','Old inbox bookmarks must remain inside Notes Inbox');
    await page.waitForFunction(()=>document.querySelector('#chapter-5-notes-inbox').open);
  }
  await page.goto(target('06_leetcode_notes.html'));
  const savedProgress=await page.evaluate(()=>localStorage.getItem('study-progress-v1'));
  try {
    for(const oldInbox of ['2-notes-inbox','3-notes-inbox','4-notes-inbox','6-notes-inbox','7-notes-inbox']) {
      await page.evaluate(oldInbox=>localStorage.setItem('study-progress-v1',JSON.stringify({
        leetcode:{done:['1-binary-search','3-tree-dfs',oldInbox],last:oldInbox},
        ml:{done:['1-neural-net-basics']}
      })),oldInbox);
      await page.reload();
      const migrated=await page.evaluate(()=>JSON.parse(localStorage.getItem('study-progress-v1')));
      assert.deepEqual(migrated.leetcode.done,['1-binary-search','5-notes-inbox'],'The inbox completion moves; one former recursion chapter alone does not complete the merged chapter');
      assert.equal(migrated.leetcode.last,'5-notes-inbox','Resume at the moved inbox, not Dijkstra');
      assert.equal(migrated.leetcode.layoutVersion,4);
      assert.deepEqual(migrated.ml.done,['1-neural-net-basics'],'LeetCode migration must not alter another book');
      assert.equal(await page.locator('#chapter-3-recursion-tree-dfs-backtracking-memoization [data-mark]').getAttribute('aria-pressed'),'false','A partly read merged chapter stays unread');
      await page.locator('[data-resume]').click();
      await page.waitForURL(/#5-notes-inbox$/);
      assert.equal(await page.locator('#chapter-5-notes-inbox').evaluate(node=>node.open),true);
      await page.locator('#chapter-5-notes-inbox [data-mark]').click();
      await page.reload();
      assert.equal(await page.locator('#chapter-5-notes-inbox [data-mark]').getAttribute('aria-pressed'),'false','Migration must not overwrite newer inbox progress');
      await page.goto(target('06_leetcode_notes.html'));
    }
    // All three former chapters read: the merged chapter is read, Dijkstra keeps its completion,
    // and a resume inside Backtracking lands on the first backtracking section.
    for(const [done,last,expected,resume] of [
      [['3-tree-dfs','4-backtracking','6-dfs--memoization','5-dijkstra-shortest-path'],'4-backtracking',['4-dijkstra-shortest-path','3-recursion-tree-dfs-backtracking-memoization'],'33-backtracking'],
      [['3-tree-dfs','6-dfs--memoization'],'6-dfs--memoization',[],'34-dfs--memoization'],
      [['2-doubly-linked-lists'],'3-tree-dfs',['2-doubly-linked-lists'],'32-tree-dfs']
    ]) {
      await page.evaluate(([done,last])=>localStorage.setItem('study-progress-v1',JSON.stringify({leetcode:{done,last,layoutVersion:3}})),[done,last]);
      await page.reload();
      const migrated=await page.evaluate(()=>JSON.parse(localStorage.getItem('study-progress-v1')));
      assert.deepEqual(migrated.leetcode.done,expected,'Merged-chapter completion needs all three former chapters: '+done.join(', '));
      assert.equal(migrated.leetcode.last,resume,'Resume follows the former chapter to its section of the merged chapter');
      assert.equal(await page.locator('[id="'+resume+'"]').evaluate(node=>node.tagName),'H3','The former chapter is now a section of the merged chapter');
      assert.equal(migrated.leetcode.layoutVersion,4);
      assert.equal(await page.locator('[id="'+resume+'"]').evaluate(node=>node.closest('[data-chapter]')?.dataset.chapter),'3-recursion-tree-dfs-backtracking-memoization');
    }
  } finally {
    await page.evaluate(saved=>saved===null?localStorage.removeItem('study-progress-v1'):localStorage.setItem('study-progress-v1',saved),savedProgress);
    await page.goto(target('06_leetcode_notes.html'));
  }

  await page.locator('[data-expand]').click();
  const drills=page.locator('.coding-drill');
  assert.equal(await drills.count(),39);
  for(const drill of await drills.all())await checkDrillComplexity(drill);
  assert.equal(await page.locator('#chapter-1-binary-search .coding-drill').count(),4);
  const linkedChapter=page.locator('#chapter-2-doubly-linked-lists');
  assert.equal(await linkedChapter.locator('.coding-drill').count(),3);
  const treeChapter=page.locator('#chapter-3-recursion-tree-dfs-backtracking-memoization');
  assert.equal(await treeChapter.locator('.coding-drill').count(),26,'Recursion keeps the 9 tree DFS, 7 backtracking and 10 memoization folds');
  assert.deepEqual((await treeChapter.locator(':scope > .chapter-body > .prose > h3').allTextContents()).map(text=>text.replace(/#$/,'')),[
    '3.1 One recursion, three questions',
    '3.2 Tree DFS',
    '3.3 Backtracking',
    '3.4 DFS + memoization'
  ]);
  assert.deepEqual((await treeChapter.locator(':scope > .chapter-body > .prose > h4').allTextContents()).map(text=>text.replace(/#$/,'')),[
    '3.2.1 Choose the return contract',
    '3.2.2 Return a number: subtree information',
    '3.2.3 Return bool: check a condition',
    '3.2.4 Return node: rebuild a subtree',
    '3.2.5 Return node: find an answer',
    '3.2.6 Return None: collect paths',
    '3.2.7 Return a tuple: combine states',
    '3.2.8 Return value versus final answer',
    '3.2.9 Traversal order and common mistakes',
    '3.3.1 First decide whether order and length matter',
    '3.3.2 The shared pattern: choose, explore, undo',
    '3.3.3 Subsets: save every node',
    '3.3.4 Combinations: save only when the size reaches k',
    '3.3.5 Permutations: fill positions with unused indices',
    '3.3.6 Allow reusing an element: i instead of i + 1',
    '3.3.7 Duplicate inputs: skip duplicate branches, not valid repeated values',
    '3.3.8 Template cheat sheet and common mistakes',
    '3.4.1 When memoization works',
    '3.4.2 The shared template',
    '3.4.3 377. Combination Sum IV: count ordered ways, combine with +',
    '3.4.4 279. Perfect Squares: fewest pieces, combine with min',
    '3.4.5 300. Longest Increasing Subsequence: best length, combine with max',
    '3.4.6 139. Word Break: feasibility, combine with or',
    '3.4.7 416. Partition Equal Subset Sum: take or skip, combine with or',
    '3.4.8 152. Maximum Product Subarray: return two values, combine with max and min',
    '3.4.9 Finding the state, and common mistakes'
  ],'Recursion: numbered subsections under each of the three parts');
  const treeHelper=treeChapter.locator('pre code').filter({hasText:'class TreeNode:'});
  const contractTable=treeChapter.locator('[id="321-choose-the-return-contract"] ~ .table-scroll').first();
  const deletionDiagram=treeChapter.locator('pre code.language-text').filter({hasText:'Delete 3:'});
  assert.equal(await deletionDiagram.count(),1,'BST deletion needs a before/after tree diagram');
  assert.ok(await contractTable.evaluate(el=>el.scrollWidth<=el.clientWidth+1),'The return-contract overview should wrap on desktop');
  assert.equal(await treeHelper.count(),1);
  assert.equal(await treeHelper.evaluate(node=>!!node.closest('.coding-drill')),false,'The shared TreeNode helper must be readable outside a coding fold');
  const treeSnippets=await treeChapter.locator('.coding-drill pre code').allTextContents();
  for(const name of ['max_depth','same_tree','is_valid_bst','insert_bst','delete_bst','lowest_common_ancestor','path_sum','is_balanced','diameter']) {
    assert.ok(treeSnippets.some(code=>code.includes('def '+name+'(')),'Missing rendered Tree DFS template: '+name);
  }
  const backtrackingChapter=treeChapter;
  const backtrackingTables=backtrackingChapter.locator('.table-scroll');
  assert.equal(await backtrackingTables.count(),7,'Recursion: three overview tables, the tree return contracts, two backtracking tables and the memoization states');
  const overviewTableCount=3; // the 3.1 tables may pan on a phone; every other table wraps everywhere
  for(const table of await backtrackingTables.all())assert.ok(await table.evaluate(el=>el.scrollWidth<=el.clientWidth+1),'Every recursion table should wrap on desktop');
  const backtrackingSnippets=await backtrackingChapter.locator('.coding-drill pre code').allTextContents();
  for(const name of ['subsets','combine','permute','combination_sum','subsets_with_dup','combination_sum2','permute_unique']) {
    assert.ok(backtrackingSnippets.some(code=>code.includes('def '+name+'(')),'Missing rendered Backtracking template: '+name);
  }
  const backtrackingText=await backtrackingChapter.textContent();
  for(const [concept,pattern] of [
    ['start index',/\bstart\b/],
    ['used flags',/\bused\b/],
    ['same-depth duplicate pruning',/same[- ](?:recursion[- ]|tree[- ])?(?:depth|level)/i],
    ['copying each answer',/path\.copy\(\)|path\[:\]/],
    ['undoing a choice',/path\.pop\(\)/]
  ])assert.match(backtrackingText,pattern,'Backtracking notes must explain '+concept);
  const backtrackingDiagrams=backtrackingChapter.locator('pre code.language-text,pre code.language-plaintext');
  assert.ok(await backtrackingDiagrams.count()>0,'Backtracking needs a readable choice-tree diagram');
  for(const diagram of await backtrackingDiagrams.all())assert.ok(await diagram.isVisible());
  const dijkstraChapter=page.locator('#chapter-4-dijkstra-shortest-path');
  assert.equal(await dijkstraChapter.locator('.coding-drill').count(),6);
  const dijkstraTables=dijkstraChapter.locator('.table-scroll');
  assert.equal(await dijkstraTables.count(),2,'Dijkstra needs its variant comparison and when-to-use tables');
  for(const table of await dijkstraTables.all())assert.ok(await table.evaluate(el=>el.scrollWidth<=el.clientWidth+1),'Both Dijkstra tables should wrap on desktop');
  const dijkstraSnippets=await dijkstraChapter.locator('.coding-drill pre code').allTextContents();
  for(const name of ['dijkstra','dijkstra_visited','network_delay_time','network_delay_time_visited','minimum_effort_path','minimum_effort_path_visited']) {
    assert.ok(dijkstraSnippets.some(code=>code.includes('def '+name+'(')),'Missing rendered Dijkstra template: '+name);
  }
  assert.ok(dijkstraSnippets.some(code=>code.includes('if d > dist[u]:')),'The template must skip stale heap entries');
  assert.ok(dijkstraSnippets.some(code=>code.includes('candidate = max(e, abs(')),'Minimum Effort must relax with max, not +');
  assert.ok(dijkstraSnippets.some(code=>code.includes('if (r, c) in visited:')),'The visited-set variant must settle on first pop');
  const dijkstraDiagrams=dijkstraChapter.locator('pre code.language-text,pre code.language-plaintext');
  assert.ok(await dijkstraDiagrams.count()>=2,'Dijkstra needs the settle-order trace and the relaxation comparison');
  for(const diagram of await dijkstraDiagrams.all())assert.ok(await diagram.isVisible());
  const memoChapter=treeChapter;
  const memoTables=memoChapter.locator('.table-scroll');
  assert.ok(await memoTables.count()>=1,'DFS + Memoization needs its state table');
  for(const table of await memoTables.all())assert.ok(await table.evaluate(el=>el.scrollWidth<=el.clientWidth+1),'DFS + Memoization tables should wrap on desktop');
  const memoSnippets=await memoChapter.locator('.coding-drill pre code').allTextContents();
  for(const name of ['combination_sum4','num_squares','num_squares_table','length_of_lis_take_skip','length_of_lis','word_break','can_partition','can_partition_table','max_product','max_product_loop']) {
    assert.ok(memoSnippets.some(code=>code.includes('def '+name+'(')),'Missing rendered DFS + Memoization example: '+name);
  }
  assert.ok(['combination_sum4','num_squares','length_of_lis_take_skip','length_of_lis','word_break','can_partition','max_product'].every(name=>{const code=memoSnippets.find(code=>code.includes('def '+name+'('));return code&&code.includes('def dfs(')&&/memo\[/.test(code);}),'Every memoized DFS stores its results');
  assert.ok(memoSnippets.some(code=>code.includes('for s in range(target, value - 1, -1):')),'416 must run the sums downward so each number is used once');
  assert.ok(memoSnippets.some(code=>code.includes('for i in range(len(nums))')&&code.includes('def max_product(')),'152 calls dfs in increasing order to stay shallow');
  const overview=await treeChapter.locator('[id="31-one-recursion-three-questions"] ~ .code-block code.language-text').first().textContent();
  for(const piece of ['return combine(node, left, right)','path.pop()','memo[state] = answer'])assert.ok(overview.includes(piece),'3.1 shows all three templates: '+piece);
  assert.ok(memoSnippets.some(code=>code.includes('range(math.isqrt(n), 0, -1)')),'Perfect Squares tries the largest square first to keep the recursion shallow');
  assert.ok(memoSnippets.some(code=>code.includes('memo[index] = False')),'Word Break must cache failures');
  const memoText=await memoChapter.textContent();
  for(const [concept,pattern] of [['the state',/state/i],['states times work per call',/distinct states/i],['ordered versus unordered counting',/518/],['recursion depth',/recursionlimit|1,000 frames/]])assert.match(memoText,pattern,'DFS + Memoization notes must explain '+concept);
  const memoDiagrams=memoChapter.locator('pre code.language-text,pre code.language-plaintext');
  assert.ok(await memoDiagrams.count()>=2,'DFS + Memoization needs the template and the call-tree diagram');
  for(const diagram of await memoDiagrams.all())assert.ok(await diagram.isVisible());
  const snippets=await drills.locator('pre code').allTextContents();
  for(const name of ['binary_search_exact','lower_bound','upper_bound','search_range','min_eating_speed']) {
    assert.ok(snippets.some(code=>code.includes('def '+name+'(')),'Missing rendered template: '+name);
  }
  for(const name of ['Node','DoublyLinkedList','MyCircularQueue','LRUCache']) {
    assert.ok(snippets.some(code=>new RegExp('class '+name+'\\s*[:(]').test(code)),'Missing rendered class: '+name);
  }
  const diagrams=linkedChapter.locator('pre code.language-text,pre code.language-plaintext');
  assert.ok(await diagrams.count()>0,'The linked-list notes need a readable pointer diagram');
  for(const diagram of await diagrams.all())assert.ok(await diagram.isVisible());
  // Verify native code-fold keyboard behavior and the actual clipboard payload.
  for(const drill of [drills.first(),treeChapter.locator('.coding-drill').first(),backtrackingChapter.locator('.coding-drill').first()]) {
    const summary=drill.locator('summary');
    await summary.focus();await summary.press('Enter');
    assert.equal(await drill.evaluate(node=>node.open),false);
    await summary.press('Enter');
    assert.equal(await drill.evaluate(node=>node.open),true);
  }
  await page.evaluate(()=>Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{window.__copiedCode=text;}}}));
  for(const button of await drills.locator('.copy-code').all()) {
    const expected=await button.locator('xpath=ancestor::div[contains(@class,"code-block")]//code').textContent();
    await button.click();
    assert.equal(await page.evaluate(()=>window.__copiedCode),expected);
  }

  const style='html{scroll-behavior:auto!important}.toast{visibility:hidden!important}';
  await drills.first().scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(preview,'leetcode-templates-desktop.png'),style,animations:'disabled'});
  await linkedChapter.locator(':scope > summary').scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(preview,'leetcode-linked-lists-desktop.png'),style,animations:'disabled'});
  await treeChapter.locator(':scope > summary').evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
  await page.screenshot({path:path.join(preview,'leetcode-tree-dfs-desktop.png'),style,animations:'disabled'});
  await deletionDiagram.evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
  await page.screenshot({path:path.join(preview,'leetcode-tree-deletion-desktop.png'),style,animations:'disabled'});
  await backtrackingChapter.locator(':scope > summary').evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
  await page.screenshot({path:path.join(preview,'leetcode-backtracking-desktop.png'),style,animations:'disabled'});
  const originalTheme=await page.locator('html').getAttribute('data-theme');
  await page.locator('[data-theme-toggle]').click();
  assert.notEqual(await page.locator('html').getAttribute('data-theme'),originalTheme);
  await drills.first().scrollIntoViewIfNeeded();
  assert.ok(await drills.first().locator('pre code').isVisible());
  await page.screenshot({path:path.join(preview,'leetcode-templates-dark.png'),style,animations:'disabled'});
  await backtrackingChapter.locator('.coding-drill').first().evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
  assert.ok(await backtrackingChapter.locator('.coding-drill pre code').first().isVisible());
  await page.screenshot({path:path.join(preview,'leetcode-backtracking-dark.png'),style,animations:'disabled'});
  await page.locator('[data-theme-toggle]').click();

  await page.locator('[data-collapse]').click();
  await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));
  await page.emulateMedia({media:'print'});
  assert.equal(await page.locator('article details:not([open])').count(),0);
  for(const code of await drills.locator('pre code').all())assert.ok(await code.isVisible());
  assert.ok(await treeHelper.isVisible());
  for(const table of await backtrackingTables.all())assert.ok(await table.evaluate(el=>el.scrollWidth<=el.clientWidth+1),'Every recursion table should fit the print layout');
  await page.emulateMedia({media:'screen'});
  await page.evaluate(()=>window.dispatchEvent(new Event('afterprint')));
  await page.locator('[data-expand]').click();
  for(const id of ['1-binary-search','2-doubly-linked-lists','3-recursion-tree-dfs-backtracking-memoization','4-dijkstra-shortest-path']) {
    const mark=page.locator('#chapter-'+id+' [data-mark]');
    if(await mark.getAttribute('aria-pressed')!=='true')await mark.click();
  }
  await page.reload();
  for(const id of ['1-binary-search','2-doubly-linked-lists','3-recursion-tree-dfs-backtracking-memoization','4-dijkstra-shortest-path']) {
    assert.equal(await page.locator('#chapter-'+id+' [data-mark]').getAttribute('aria-pressed'),'true');
  }

  await page.setViewportSize({width:390,height:844});
  await page.locator('[data-expand]').click();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  for(const code of await drills.locator('pre code').all())assert.ok(await code.isVisible());
  await drills.first().scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(preview,'leetcode-templates-mobile.png'),style,animations:'disabled'});
  await linkedChapter.locator(':scope > summary').scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(preview,'leetcode-linked-lists-mobile.png'),style,animations:'disabled'});
  await treeChapter.locator(':scope > summary').evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
  await page.screenshot({path:path.join(preview,'leetcode-tree-dfs-mobile.png'),style,animations:'disabled'});
  assert.ok(await contractTable.evaluate(el=>el.scrollWidth<=el.clientWidth+1),'The return-contract overview should wrap on mobile');
  await deletionDiagram.evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
  await page.screenshot({path:path.join(preview,'leetcode-tree-deletion-mobile.png'),style,animations:'disabled'});
  await backtrackingChapter.locator(':scope > summary').evaluate(node=>node.scrollIntoView({block:'start',behavior:'instant'}));
  for(const [index,table] of (await backtrackingTables.all()).entries()) {
    const fits=await table.evaluate(el=>el.scrollWidth<=el.clientWidth+1);
    if(index>=overviewTableCount)assert.ok(fits,'Every recursion table after the overview should wrap on mobile');
    else assert.deepEqual(await table.evaluate(el=>{
      // A panning overview table must not split words: every word of 4+ letters stays on one line.
      const split=[];const walker=document.createTreeWalker(el,NodeFilter.SHOW_TEXT);
      for(let node=walker.nextNode();node;node=walker.nextNode()){let offset=0;for(const word of node.textContent.split(/(\s+)/)){if(/^[A-Za-z]{4,}$/.test(word)){const range=document.createRange();range.setStart(node,offset);range.setEnd(node,offset+word.length);if(new Set([...range.getClientRects()].filter(rect=>rect.width>0).map(rect=>Math.round(rect.top))).size>1)split.push(word);}offset+=word.length;}}
      return split;
    }),[],'Overview table '+index+' pans on mobile instead of splitting words');
  }
  await page.screenshot({path:path.join(preview,'leetcode-backtracking-mobile.png'),style,animations:'disabled'});
  for(const diagram of await backtrackingDiagrams.all())assert.ok(await diagram.isVisible());
  assert.ok(await treeHelper.isVisible());
  for(const diagram of await diagrams.all())assert.ok(await diagram.isVisible());
  await page.setViewportSize({width:1440,height:1000});

  const offline=await context.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
  try {
    await offline.route('**/*',route=>/^https?:/.test(route.request().url())?route.abort():route.continue());
    const staticPage=await offline.newPage();await staticPage.goto(target('06_leetcode_notes.html'));
    assert.equal(await staticPage.locator('[data-bs-lab]').count(),0);
    for(const [id,count] of [['1-binary-search',4],['2-doubly-linked-lists',3],['3-recursion-tree-dfs-backtracking-memoization',26],['4-dijkstra-shortest-path',6]]) {
      const chapter=staticPage.locator('#chapter-'+id);
      if(!await chapter.evaluate(node=>node.open))await chapter.locator(':scope > summary').click();
      assert.equal(await chapter.locator('.coding-drill').count(),count);
      for(const drill of await chapter.locator('.coding-drill').all()) {
        if(!await drill.evaluate(node=>node.open))await drill.locator('summary').click();
        for(const code of await drill.locator('pre code').all())assert.ok(await code.isVisible(),'Templates and diagrams must be readable without JavaScript');
        await checkDrillComplexity(drill);
      }
    }
    assert.ok(await staticPage.locator('#chapter-3-recursion-tree-dfs-backtracking-memoization pre code').filter({hasText:'class TreeNode:'}).isVisible());
    for(const [index,table] of (await staticPage.locator('#chapter-3-recursion-tree-dfs-backtracking-memoization .table-scroll').all()).entries())if(index>=3)assert.ok(await table.evaluate(el=>el.scrollWidth<=el.clientWidth+1),'Every recursion table after the overview should wrap without JavaScript');
    for(const diagram of await staticPage.locator('#chapter-3-recursion-tree-dfs-backtracking-memoization pre code.language-text,#chapter-3-recursion-tree-dfs-backtracking-memoization pre code.language-plaintext').all())assert.ok(await diagram.isVisible());
    assert.equal(await staticPage.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  } finally {await offline.close();}
  console.log('PASS LeetCode binary-search, linked-list, recursion (tree DFS, backtracking and memoization as three numbered sections under one overview) and Dijkstra notes, scoped search, thirty-nine code folds with time/space analyses, legacy chapter and inbox bookmarks, merged-chapter progress migration, diagrams, keyboard, copy, progress, theme, mobile, print and no-JS reading.');
}
