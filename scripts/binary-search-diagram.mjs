// Original teaching traces. Render every case for offline / no-JS / print use.
export const binarySearchCases = [
  {id:'insert',label:'Insert 2',nums:[1,3,5,6],target:2},
  {id:'found',label:'Find 5',nums:[1,3,5,6],target:5},
  {id:'end',label:'After end: 7',nums:[1,3,5,6],target:7},
  {id:'duplicates',label:'Duplicates: 3',nums:[1,3,3,3,5],target:3},
];
export const binarySearchMethods = [
  {id:'half-open',title:'Half-open lower bound',rule:'right = n · while left < right',goal:'First index with value ≥ target, or n.'},
  {id:'converging',title:'Your first version',rule:'right = n − 1 · while left < right',goal:'Keep an answer in [left, right]; guards handle outside positions.'},
  {id:'closed',title:'Your second version',rule:'right = n − 1 · while left ≤ right',goal:'Return an equal index; otherwise return the insertion position.'},
];

export function traceBinarySearch(nums,target,method) {
  if(!binarySearchMethods.some(item=>item.id===method)) throw new Error('Unknown binary-search method');
  const n=nums.length,trace=[];
  let left=0,right=method==='half-open'?n:n-1;
  const record=(mid,action,done=false,result=null)=>trace.push({left,right,mid,action,done,result});
  if(method==='converging'&&(!n||target>nums[n-1]||target<nums[0])) {
    const result=n&&target>nums[n-1]?n:0;
    left=null;right=null;
    record(null,'Check edge guards before initializing the interval.');
    record(null,!n?'Empty input: return 0 (added safety guard).':target>nums[n-1]?`target > last value: return n = ${n}. The loop never starts.`:'target < first value: return 0. The loop never starts.',true,result);
    return trace;
  }
  record(null,method==='half-open'?'Initialize unchecked indices [0, n); n is a legal answer boundary.':'Initialize closed indices [0, n − 1].');
  let safety=0;
  while(method==='closed'?left<=right:left<right) {
    if(++safety>64) throw new Error('Trace failed to make progress');
    const mid=left+Math.floor((right-left)/2),value=nums[mid];
    if(method==='closed'&&value===target) {
      record(mid,`nums[${mid}] = ${value}: equality returns mid immediately.`,true,mid);
      return trace;
    }
    if(value<target) {
      left=mid+1;
      record(mid,`nums[${mid}] = ${value} < ${target}: left = mid + 1 = ${left}.`);
    } else {
      right=method==='closed'?mid-1:mid;
      record(mid,method==='closed'?`nums[${mid}] = ${value} > ${target}: right = mid − 1 = ${right}.`:`nums[${mid}] = ${value} ≥ ${target}: right = mid = ${right}.`);
    }
  }
  record(null,method==='closed'?`left > right: no equal value found. Return left = ${left}.`:`left = right: the boundary has converged. Return ${left}.`,true,left);
  return trace;
}

const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const shown = value => value===null?'—':value;

export function renderBinarySearchLab() {
  return `<figure class="bs-lab" id="binary-search-lab" data-bs-lab aria-labelledby="bs-lab-title">
    <figcaption><strong id="bs-lab-title">Same input. Three different interval contracts.</strong><span>BINARY SEARCH LAB</span></figcaption>
    <p>Choose an example, then step through the three algorithms together. Each table shows the boundaries <strong>after</strong> that action; “mid used” belongs to the comparison that produced it.</p>
    <div class="bs-controls" role="group" aria-label="Choose a binary search example">${binarySearchCases.map((item,i)=>`<button type="button" data-bs-case="${item.id}" aria-pressed="${i===0}" aria-controls="bs-case-${item.id}">${item.label}</button>`).join('')}</div>
    <div class="bs-playback" role="group" aria-label="Step through binary search"><button type="button" data-bs-back>← Previous</button><button type="button" data-bs-next>Next step →</button><button type="button" data-bs-reset>Reset</button><span data-bs-status></span></div>
    ${binarySearchCases.map(item=>{
      const traces=binarySearchMethods.map(method=>({method,steps:traceBinarySearch(item.nums,item.target,method.id)}));
      return `<section class="bs-case" id="bs-case-${item.id}" data-bs-panel="${item.id}" data-bs-max="${Math.max(...traces.map(trace=>trace.steps.length))-1}" aria-labelledby="bs-case-${item.id}-title">
        <p class="bs-case-title" id="bs-case-${item.id}-title"><strong>nums = [${item.nums.join(', ')}]</strong><span>target = ${item.target}</span></p>
        <div class="bs-array" aria-label="Array values and zero-based indices">${item.nums.map((value,index)=>`<span aria-label="Index ${index}, value ${value}"><small>i:${index}</small><b>${value}</b></span>`).join('')}<span class="bs-sentinel"><small>n = ${item.nums.length}</small><b>boundary</b></span></div>
        <p class="bs-note">i labels a zero-based index. The dashed n position is not an array element; none of these algorithms reads nums[n].</p>
        <div class="bs-methods">${traces.map(({method,steps})=>`<section class="bs-method" data-bs-method="${method.id}" data-bs-result="${steps.at(-1).result}"><h4>${method.title}</h4><code>${esc(method.rule)}</code><p>${esc(method.goal)}</p><div class="bs-table-wrap"><table><thead><tr><th scope="col">Step</th><th scope="col">(left, right)<br>after action</th><th scope="col">mid<br>used</th></tr></thead><tbody>${steps.map((step,index)=>`<tr data-bs-step="${index}" data-bs-action="${esc(step.action)}"><th scope="row">${step.done?'Done':index}</th><td>(${shown(step.left)}, ${shown(step.right)})</td><td>${shown(step.mid)}</td></tr>`).join('')}</tbody></table></div><ol class="bs-explanations">${steps.map((step,index)=>`<li data-bs-detail="${index}"><b>${step.done?'Done':`Step ${index}`}:</b> ${esc(step.action)}${step.done?` <strong>Result: ${step.result}</strong>`:''}</li>`).join('')}</ol></section>`).join('')}</div>
        <p class="bs-takeaway">${item.id==='duplicates'?'Different answers are intentional: the boundary searches return index 1, while the equality-return version finds index 2. “Any matching index” is not the same task as “first matching index.”':item.id==='end'?'The first version needs its after-end guard to return n. Half-open lower bound reaches n naturally; the closed search advances left past n − 1.':'Matching outputs do not make the update rules interchangeable. Each version stops when its own interval contract says the answer is known.'}</p>
      </section>`;
    }).join('')}
    <p class="bs-live" data-bs-live aria-live="polite" aria-atomic="true"></p>
  </figure>`;
}
