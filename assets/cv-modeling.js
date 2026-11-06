(() => {
  const body = document.body;
  const storage = {
    get(key, fallback) { try { const value = localStorage.getItem(key); return value === null ? fallback : JSON.parse(value); } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; } },
  };
  let theme = storage.get('cv-design-theme', null);
  if (theme !== 'light' && theme !== 'dark') theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.dataset.theme = theme;
  document.querySelector('[data-design-theme]')?.addEventListener('click', () => {
    theme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    storage.set('cv-design-theme', theme);
  });
  document.querySelector('[data-design-print]')?.addEventListener('click', () => window.print());
  const boxes = [...document.querySelectorAll('[data-question]')];
  const count = document.querySelector('.design-review-count');
  let canSave = true;
  function update() {
    if (count) count.textContent = `Reviewed ${boxes.filter(box => box.checked).length} of ${boxes.length} follow-ups${canSave ? ' · saved in this browser' : ' · progress cannot be saved in this browser'}`;
  }
  for (const box of boxes) {
    const key = `cv-design-review:${body.dataset.design}:${box.dataset.question}`;
    box.checked = storage.get(key, false) === true;
    box.closest('li').classList.toggle('reviewed', box.checked);
    box.addEventListener('change', () => {
      canSave = storage.set(key, box.checked);
      box.closest('li').classList.toggle('reviewed', box.checked);
      update();
    });
  }
  update();
})();
