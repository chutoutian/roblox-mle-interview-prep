// A static, printable worked example; book.js adds output-pixel selection.
export function renderBilinearDiagram() {
  const originX = 38, originY = 34, cell = 32;
  const examples = Array.from({length:4}, (_, id) => {
    const oy = Math.floor(id / 2), ox = id % 2;
    const sy = (oy + 0.5) * 4 - 0.5, sx = (ox + 0.5) * 4 - 0.5;
    const iy = Math.floor(sy), ix = Math.floor(sx);
    const neighbors = [[iy,ix], [iy,ix+1], [iy+1,ix], [iy+1,ix+1]];
    const values = neighbors.map(([y,x]) => y * 8 + x);
    return {id, letter:'ABCD'[id], oy, ox, sy, sx, iy, ix, neighbors, values,
      value:values.reduce((a,b) => a+b, 0) / 4};
  });
  const description = e => `Output ${e.letter}, row ${e.oy}, column ${e.ox}: source position (${e.sy}, ${e.sx}); rows ${e.iy} and ${e.iy+1}, columns ${e.ix} and ${e.ix+1}. Values ${e.values.join(', ')} each have weight 0.25, giving ${e.value}.`;
  const cells = Array.from({length:64}, (_, value) => {
    const row = Math.floor(value / 8), col = value % 8;
    const group = examples.find(e => e.neighbors.some(([y,x]) => row===y && col===x));
    return `<g class="bilinear-source-cell${group?` bilinear-color-${group.id}`:''}${group?.id===0?' is-active':''}" data-source-row="${row}" data-source-col="${col}"><rect x="${originX+col*cell}" y="${originY+row*cell}" width="${cell}" height="${cell}"/><text x="${originX+(col+0.5)*cell}" y="${originY+(row+0.5)*cell+4}">${value}</text></g>`;
  }).join('');
  const axes = Array.from({length:8}, (_, i) => `<text x="${originX+(i+0.5)*cell}" y="24">${i}</text><text x="26" y="${originY+(i+0.5)*cell+4}">${i}</text>`).join('');
  const markers = examples.map(e => {
    const cx = originX + (e.sx+0.5)*cell, cy = originY + (e.sy+0.5)*cell;
    return `<g class="bilinear-source-group bilinear-color-${e.id}${e.id===0?' is-active':''}" data-bilinear-group="${e.id}"><rect x="${originX+e.ix*cell}" y="${originY+e.iy*cell}" width="64" height="64" rx="3"/><circle cx="${cx}" cy="${cy}" r="4"/><text x="${cx+9}" y="${cy+4}">${e.letter}</text></g>`;
  }).join('');
  const buttons = examples.map(e => `<button type="button" class="bilinear-output-cell bilinear-color-${e.id}" data-bilinear-pixel="${e.id}" aria-pressed="${e.id===0}" aria-controls="bilinear-panel-${e.id}" aria-label="Inspect output ${e.letter}, row ${e.oy}, column ${e.ox}, value ${e.value}"><span>${e.letter} · (${e.oy},${e.ox})</span><strong>${e.value}</strong></button>`).join('');
  const panels = examples.map(e => `<div class="bilinear-panel bilinear-color-${e.id}" id="bilinear-panel-${e.id}" data-bilinear-panel="${e.id}" role="region" aria-label="${description(e)}"${e.id===0?'':' hidden'}>
    <p class="bilinear-panel-title">Output ${e.letter} <code>out[${e.oy},${e.ox}]</code> <span>→ source (${e.sy}, ${e.sx})</span></p>
    <div class="bilinear-formulas"><code>sy = (${e.oy} + 0.5) × 8/2 − 0.5 = ${e.sy}</code><code>sx = (${e.ox} + 0.5) × 8/2 − 0.5 = ${e.sx}</code><code>iy = ${e.iy}, ix = ${e.ix}; wy = wx = 0.5</code></div>
    <div class="bilinear-contributions">${e.neighbors.map(([y,x],i) => `<div><small>img[${y},${x}]</small><span><strong>${e.values[i]}</strong> × 0.25</span></div>`).join('')}</div>
    <p class="bilinear-result"><code>(${e.values.join(' + ')}) / 4 = <strong>${e.value}</strong></code></p>
  </div>`).join('');
  return `<figure class="bilinear-demo" id="bilinear-8-to-2" data-bilinear-diagram aria-labelledby="bilinear-diagram-title">
    <figcaption><strong id="bilinear-diagram-title">From 8 × 8 to 2 × 2</strong><span>Half-pixel bilinear · no antialiasing</span></figcaption>
    <p class="bilinear-intro">One channel, with <code>img[y,x] = 8y + x</code>. Coordinates are zero-based <code>(row, col)</code>. Select an output pixel to trace its four source samples.</p>
    <div class="bilinear-layout">
      <div class="bilinear-source" data-bilinear-source><p class="bilinear-grid-label">INPUT <strong>8 × 8</strong></p>
        <svg viewBox="0 0 304 318" role="img" aria-labelledby="bilinear-source-title bilinear-source-description"><title id="bilinear-source-title">8 by 8 source pixel grid, containing values 0 through 63</title><desc id="bilinear-source-description" data-bilinear-description>${description(examples[0])}</desc><g class="bilinear-axis">${axes}<text x="166" y="312">column x →</text><text transform="translate(12 162) rotate(-90)">row y →</text></g>${cells}${markers}</svg>
      </div>
      <div class="bilinear-arrow" aria-hidden="true">→</div>
      <div class="bilinear-output" data-bilinear-output><p class="bilinear-grid-label">OUTPUT <strong>2 × 2</strong></p><div class="bilinear-output-grid" role="group" aria-label="Choose an output pixel">${buttons}</div><p class="bilinear-output-hint">A–D match the source markers.<br>Click, or Tab to focus.<br>Enter / Space to select.</p></div>
    </div>
    <div class="bilinear-legend"><span>● Mapped sampling point</span><span>Outlined 2 × 2 = four contributing pixels</span></div>
    <div aria-live="polite" aria-atomic="true">${panels}</div>
    <p class="bilinear-warning"><strong>Four neighbors, not a 4 × 4 area average.</strong> The toy code only uses the four samples around each mapped point. Each gets 25% here because <code>wx = wy = 0.5</code>; other positions have different weights. Strong downscaling needs low-pass antialiasing.</p>
    <noscript><p>JavaScript is off. The grids show all four results; the worked calculation below the grids shows output A.</p></noscript>
  </figure>`;
}
