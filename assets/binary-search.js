/* Enhance the server-rendered trace tables; all cases remain readable without JS. */
(() => {
  document.querySelectorAll('[data-bs-lab]').forEach(lab=>{
    const choices=[...lab.querySelectorAll('[data-bs-case]')];
    const panels=[...lab.querySelectorAll('[data-bs-panel]')];
    const previous=lab.querySelector('[data-bs-back]'),next=lab.querySelector('[data-bs-next]');
    let selected=choices[0].dataset.bsCase,step=0;
    function update(announce=false) {
      const panel=panels.find(item=>item.dataset.bsPanel===selected);
      panels.forEach(item=>item.hidden=item!==panel);
      choices.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.bsCase===selected)));
      const max=Number(panel.dataset.bsMax),messages=[];
      for(const method of panel.querySelectorAll('[data-bs-method]')) {
        const rows=[...method.querySelectorAll('[data-bs-step]')];
        const active=Math.min(step,rows.length-1);
        rows.forEach((row,index)=>{row.classList.toggle('bs-active-step',index===active);if(index===active)row.setAttribute('aria-current','step');else row.removeAttribute('aria-current');});
        method.querySelectorAll('[data-bs-detail]').forEach((detail,index)=>detail.hidden=index!==active);
        messages.push(method.querySelector('h4').textContent+': '+rows[active].dataset.bsAction);
      }
      previous.disabled=step===0;next.disabled=step===max;
      lab.querySelector('[data-bs-status]').textContent=`Step ${step} / ${max}`;
      lab.dataset.bsSelected=selected;lab.dataset.bsCurrentStep=String(step);
      if(announce)lab.querySelector('[data-bs-live]').textContent=messages.join(' ');
    }
    choices.forEach(button=>button.addEventListener('click',()=>{selected=button.dataset.bsCase;step=0;update(true);}));
    previous.addEventListener('click',()=>{step=Math.max(0,step-1);update(true);});
    next.addEventListener('click',()=>{const panel=panels.find(item=>item.dataset.bsPanel===selected);step=Math.min(Number(panel.dataset.bsMax),step+1);update(true);});
    lab.querySelector('[data-bs-reset]').addEventListener('click',()=>{step=0;update(true);});
    lab.dataset.bsReady='true';update();
  });
})();
