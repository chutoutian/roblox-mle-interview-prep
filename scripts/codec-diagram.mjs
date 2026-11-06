// The hybrid encoder of video book §3.1 as one static figure: every module is labelled with its
// step number, every arrow with the symbol that step's "input ⟶ output" line uses, and each box
// links to its step. Rendered at build time (KaTeX inside SVG foreignObject), so it needs no
// JavaScript and prints as it looks.
import katex from 'katex';

const tex = s => katex.renderToString(s, {throwOnError: true, displayMode: false});
// Replace $...$ spans in a label with KaTeX HTML; everything else is plain text.
const rich = s => s.split(/(\$[^$]+\$)/).map(part => part.startsWith('$') ? tex(part.slice(1, -1)) : part.replaceAll('&', '&amp;').replaceAll('<', '&lt;')).join('');

// Each module: [step, x, y, width, height, title, lines under the title, runs in the decoder too?]
export const codecModules = [
  [1, 12, 32, 140, 52, '1 · Color 4:2:0', ['RGB $\\to Y, C^b, C^r$'], false],
  [2, 12, 120, 140, 52, '2 · Partition', ['quadtree $\\{B_k\\}$'], false],
  [6, 264, 120, 120, 52, '6 · Transform', ['$C = T E T^{\\top}$'], false],
  [7, 416, 120, 120, 52, '7 · Quantize', ['step $\\Delta$ from QP'], false],
  [8, 624, 120, 160, 52, '8 · Entropy coding', ['CABAC'], true],
  [12, 401, 32, 150, 52, '12 · Rate control', ['$R_{\\text{target}}$, buffer'], false],
  [9, 486, 214, 160, 66, '9 · Reconstruct', ['$\\hat C = \\Delta\\, \\ell$', '$\\hat E = T^{\\top} \\hat C\\, T$'], true],
  [10, 184, 300, 176, 56, '10 · Mode decision', ['$k^{*} = \\arg\\min_k D_k + \\lambda R_k$'], false],
  [3, 280, 410, 184, 52, '3 · Intra prediction', ['DC, planar, angular'], true],
  [11, 478, 480, 176, 56, '11 · In-loop filters', ['deblock, SAO, ALF'], true],
  [4, 280, 562, 184, 70, '4 · Inter prediction', ['ME: search $\\mathbf v^{*}$ (SAD)', 'MC: $P_{\\mathbf v^{*}}$ from $\\tilde F_{t-1}$'], true],
];

export function renderCodecDiagram() {
  const W = 824, H = 676;
  const box = ([step, x, y, w, h, title, lines, decoder]) => `<a href="#codec-step-${step}" class="codec-node${decoder ? ' codec-decoder' : ''}" aria-label="Step ${step}: ${title.replace(/^\d+ · /, '')}">
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7"/>
    <foreignObject x="${x}" y="${y}" width="${w}" height="${h}"><div xmlns="http://www.w3.org/1999/xhtml" class="codec-box"><b>${rich(title)}</b>${lines.map(s => `<span>${rich(s)}</span>`).join('')}</div></foreignObject></a>`;
  // A label is anchored at (x, y): its left edge, centre or right edge, vertically centred on y.
  const label = (x, y, s, cls = '', anchor = 'start') => {
    const w = 240, left = anchor === 'middle' ? x - w / 2 : anchor === 'end' ? x - w : x;
    return `<foreignObject x="${left}" y="${y - 18}" width="${w}" height="36"><div xmlns="http://www.w3.org/1999/xhtml" class="codec-label codec-${anchor} ${cls}"><span>${rich(s)}</span></div></foreignObject>`;
  };
  const line = (points, cls = '') => `<polyline class="codec-edge ${cls}" points="${points.map(p => p.join(',')).join(' ')}" marker-end="url(#${cls === 'codec-side' ? 'codec-arrow-side' : 'codec-arrow'})"/>`;
  const sum = (x, y, sign) => `<circle class="codec-sum" cx="${x}" cy="${y}" r="13"/><text class="codec-sign" x="${x}" y="${y + 5.5}" text-anchor="middle">${sign}</text>`;
  const dot = (x, y) => `<circle class="codec-dot" cx="${x}" cy="${y}" r="3.5"/>`;
  const buffer = [[524, 584], [517, 577], [510, 570]].map(([x, y]) => `<rect x="${x}" y="${y}" width="112" height="56" rx="5"/>`).join('');
  const svg = `<svg class="codec-svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="codec-diagram-title codec-diagram-desc">
  <title id="codec-diagram-title">Hybrid block-based video encoder, steps 1 to 12</title>
  <desc id="codec-diagram-desc">Step 1 converts the RGB frame to Y, Cb, Cr; step 2 partitions F t into blocks B. Step 5 subtracts the prediction P to give the residual E, step 6 transforms it to C, step 7 quantizes C to the levels ell with the step size from QP, and step 8 entropy codes ell, the mode m and the motion-vector difference v star minus v p into bits, with R bits spent. Step 12 turns R into the QP of step 7. Step 9 dequantizes and inverse transforms ell to C hat and E hat, and adds P back: B hat equals clip of P plus round of E hat. The reconstructed frame F t raw passes the in-loop filters of step 11 and enters the reference buffer as F tilde t. Step 3 predicts from the reconstructed neighbours t and l; step 4 searches F tilde t minus 1 for v star and fetches P inter. Step 10 compares the candidates by D k plus lambda R k and sends the winner P to steps 5 and 9.</desc>
  <defs>
    <marker id="codec-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z"/></marker>
    <marker id="codec-arrow-side" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z"/></marker>
  </defs>
  ${line([[82, 2], [82, 30]])}${label(90, 15, 'frame $t$')}
  ${line([[82, 84], [82, 118]])}${label(90, 101, '$F_t$')}
  ${line([[152, 146], [203, 146]])}${dot(168, 146)}${label(176, 129, '$B$')}
  ${sum(218, 146, '−')}${label(218, 101, '5 · $E = B - P$', 'codec-step', 'middle')}
  ${line([[231, 146], [262, 146]])}${label(238, 129, '$E$')}
  ${line([[384, 146], [414, 146]])}${label(391, 129, '$C$')}
  ${line([[536, 146], [622, 146]])}${dot(566, 146)}${label(545, 129, '$\\ell$')}
  ${line([[784, 146], [822, 146]])}${label(788, 129, 'bits')}
  ${line([[704, 120], [704, 58], [553, 58]])}${label(711, 89, '$R$')}
  ${line([[476, 84], [476, 118]])}${label(483, 101, 'QP')}
  ${line([[566, 146], [566, 212]])}
  ${line([[566, 280], [566, 313]])}${label(574, 297, '$\\hat E$')}
  ${sum(566, 328, '+')}${label(586, 328, '$\\hat B = \\operatorname{clip}(P + \\operatorname{round}(\\hat E))$', 'codec-step')}
  ${line([[566, 341], [566, 478]])}${dot(566, 436)}${label(574, 458, '$\\tilde F_t^{\\text{raw}}$')}
  ${line([[566, 436], [466, 436]])}${label(516, 422, '$\\mathbf t, \\mathbf l$', '', 'middle')}
  ${line([[566, 536], [566, 568]])}${label(574, 552, '$\\tilde F_t$')}
  <g class="codec-buffer">${buffer}</g>${label(566, 587, 'reference', 'codec-buffer-text', 'middle')}${label(566, 607, 'buffer', 'codec-buffer-text', 'middle')}
  ${line([[510, 597], [466, 597]])}${label(488, 583, '$\\tilde F_{t-1}$', '', 'middle')}
  ${line([[168, 146], [168, 612], [278, 612]])}${label(196, 598, '$B$')}
  ${line([[280, 580], [240, 580], [240, 358]])}${label(247, 520, '$P^{\\text{inter}}$')}
  ${line([[320, 410], [320, 358]])}${label(328, 384, '$P^{\\text{intra}}_m$')}
  ${line([[218, 300], [218, 161]])}${label(225, 232, '$P$')}
  ${line([[360, 328], [551, 328]])}${label(456, 314, '$P$', '', 'middle')}
  ${line([[486, 247], [330, 247], [330, 298]], 'codec-trial')}${label(408, 234, '$D_k, R_k$', 'codec-trial-text', 'middle')}
  ${line([[270, 300], [270, 192], [744, 192], [744, 174]], 'codec-side')}${label(277, 222, 'mode $m$', 'codec-side-text')}
  ${line([[372, 632], [372, 664], [776, 664], [776, 174]], 'codec-side')}${label(690, 650, '$\\mathbf v^{*} - \\mathbf v^{p}$', 'codec-side-text', 'middle')}
  ${codecModules.map(box).join('\n  ')}
</svg>`;
  return `<figure class="codec-diagram" id="codec-pipeline-figure">
  <figcaption>The encode pipeline at a glance</figcaption>
  <div class="codec-diagram-scroll" tabindex="0" role="region" aria-label="Encoder diagram, scrollable on narrow screens">${svg}</div>
  <p class="codec-legend"><span class="codec-legend-item"><span class="codec-key codec-key-data"></span>samples</span> <span class="codec-legend-item"><span class="codec-key codec-key-side"></span>side information in the bitstream</span> <span class="codec-legend-item"><span class="codec-key codec-key-trial"></span>trial costs of each candidate</span> <span class="codec-legend-item"><span class="codec-key codec-key-decoder"></span>also run by the decoder</span> <span class="codec-legend-item"><span class="codec-key codec-key-encoder"></span>encoder only</span> <span class="codec-legend-item">Each box links to its step.</span></p>
</figure>`;
}
