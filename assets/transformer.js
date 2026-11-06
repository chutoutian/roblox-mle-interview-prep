/* Offline progressive enhancement: native buttons, no dependencies or requests. */
(() => {
  function initializeTransformerDiagrams() {
    document.querySelectorAll('[data-transformer-diagram]').forEach(figure => {
      if (figure.dataset.transformerReady) return;
      const live = figure.querySelector('[data-transformer-live]');
      const choices = [...figure.querySelectorAll('[data-transformer-choice]')];
      const panels = [...figure.querySelectorAll('[data-transformer-panel]')];
      function select(key, announce) {
        const selected = panels.find(panel => panel.dataset.transformerPanel === key);
        if (!selected) return;
        panels.forEach(panel => { panel.hidden = panel !== selected; });
        choices.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.transformerChoice === key)));
        figure.dataset.transformerState = key;
        if (announce && live) live.textContent = `${selected.querySelector('.tf-panel-title')?.textContent || ''}. ${selected.querySelector('[data-transformer-explanation]')?.textContent || ''}`;
      }
      choices.forEach(button => button.addEventListener('click', () => select(button.dataset.transformerChoice, true)));
      if (choices.length && panels.length) select(choices[0].dataset.transformerChoice, false);

      const queryButtons = [...figure.querySelectorAll('[data-transformer-query-choice]')];
      function selectQuery(row, announce) {
        if (!Number.isInteger(row) || row < 0 || row > 3) return;
        const labels = ['BOS', 'nous', 'regardons', 'video'];
        const crossLabels = ['BOS', 'tu', 'regardes', 'EOS'];
        const readable = labels.slice(0,row + 1);
        const list = readable.length === 1 ? readable[0] : `${readable.slice(0,-1).join(', ')} and ${readable.at(-1)}`;
        const description = `At position ${row}: batch 0 query ${labels[row]} can read ${list} in self-attention. Batch 1 query ${crossLabels[row]} can read you, watch and video in cross-attention, but not PAD.`;
        figure.querySelectorAll('[data-transformer-query]').forEach(tr => tr.classList.toggle('tf-active-query', Number(tr.dataset.transformerQuery) === row));
        queryButtons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.transformerQueryChoice) === row)));
        const summary = figure.querySelector('[data-transformer-query-summary]');
        if (summary) summary.textContent = description;
        figure.dataset.transformerQueryRow = String(row);
        if (announce && live) live.textContent = description;
      }
      queryButtons.forEach(button => button.addEventListener('click', () => selectQuery(Number(button.dataset.transformerQueryChoice), true)));
      if (queryButtons.length) selectQuery(Number(queryButtons.find(button => button.getAttribute('aria-pressed') === 'true')?.dataset.transformerQueryChoice ?? 0), false);
      figure.dataset.transformerReady = 'true';
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initializeTransformerDiagrams, {once: true});
  else initializeTransformerDiagrams();
})();
