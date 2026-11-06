/* Scenario controls for the offline network figures. No libraries or requests. */
(() => {
  function initializeNetworkDiagrams() {
    document.querySelectorAll('[data-network-diagram]').forEach(figure => {
      if (figure.dataset.networkReady) return;
      const buttons = [...figure.querySelectorAll('[data-network-choice]')];
      const panels = [...figure.querySelectorAll('[data-network-panel]')];
      if (!buttons.length || !panels.length) return;
      const live = figure.querySelector('[data-network-live]');
      function select(key, announce) {
        const selected = panels.find(panel => panel.dataset.networkPanel === key);
        if (!selected) return;
        panels.forEach(panel => { panel.hidden = panel !== selected; });
        buttons.forEach(button => {
          button.setAttribute('aria-pressed', String(button.dataset.networkChoice === key));
        });
        figure.querySelectorAll('[data-network-highlight]').forEach(item => {
          item.classList.toggle('is-network-active', item.dataset.networkHighlight === key);
        });
        figure.dataset.networkState = key;
        if (announce && live) {
          live.textContent = `${selected.querySelector('.network-state-title')?.textContent || ''}. ${selected.querySelector('[data-network-explanation]')?.textContent || ''}`;
        }
      }
      buttons.forEach(button => button.addEventListener('click', () => select(button.dataset.networkChoice, true)));
      select(buttons.find(button => button.getAttribute('aria-pressed') === 'true')?.dataset.networkChoice || buttons[0].dataset.networkChoice, false);
      figure.dataset.networkReady = 'true';
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initializeNetworkDiagrams, {once: true});
  else initializeNetworkDiagrams();
})();
