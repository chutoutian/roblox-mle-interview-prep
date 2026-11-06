export const jpegStages = [
  {
    title: 'RGB → YCbCr',
    href: '#22-color-conversion-and-chroma-subsampling',
    detail: 'Separate brightness from color.',
    output: 'RGB pixels → Y, Cb, Cr planes',
  },
  {
    title: 'Chroma subsampling',
    href: '#22-color-conversion-and-chroma-subsampling',
    detail: 'Optional: low-pass filter Cb and Cr, then reduce their resolution. Keep Y unchanged.',
    output: '4:2:0: 2×2 chroma samples → 1',
    tag: 'Optional · lossy',
  },
  {
    title: '8×8 blocks + level shift',
    href: '#23-blocks-and-level-shift',
    detail: 'Split each plane into blocks; pad edges as needed. Subtract 128 from each 8-bit sample.',
    output: '[0, 255] → [−128, 127]',
  },
  {
    title: 'DCT',
    href: '#24-dct',
    detail: 'Represent each block by spatial frequencies: one DC coefficient and 63 AC coefficients.',
    output: '64 samples → 64 coefficients',
  },
  {
    title: 'Quantization',
    href: '#25-quantization',
    detail: 'Divide each coefficient by its quantization-table entry, then round. Small coefficients often become zero.',
    output: 'q[u,v] = round(c[u,v] / Q[u,v])',
    tag: 'Lossy',
  },
  {
    title: 'Zigzag scan',
    href: '#26-zigzag-and-entropy-coding',
    detail: 'Reorder the quantized block from low to high spatial frequency, grouping zeros near the end.',
    output: '8×8 → [DC, AC₁, …, AC₆₃]',
  },
  {
    title: 'DC difference / AC zero runs',
    href: '#26-zigzag-and-entropy-coding',
    branches: [
      { label: 'DC', text: 'Difference from the previous DC of the same component.', code: 'ΔDC → size + amplitude bits' },
      { label: 'AC', text: 'Count zeros before each nonzero value; use EOB for trailing zeros.', code: 'AC → (run, size) + amplitude bits' },
    ],
  },
  {
    title: 'Huffman coding + amplitude bits',
    href: '#26-zigzag-and-entropy-coding',
    detail: 'Huffman-code each DC size or AC symbol, then append its amplitude bits. EOB and ZRL have no amplitude bits.',
    output: 'symbols + values → compressed bits',
  },
  {
    title: 'JPEG file assembly',
    href: '#27-jpeg-file-and-decoding',
    detail: 'Write the image information, quantization and Huffman tables, then the encoded scan and end marker.',
    output: 'SOI → headers / tables → SOS → scan → EOI',
  },
];

const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

export function renderJpegDiagram() {
  const steps = jpegStages.map((stage, index) => `<li class="jpeg-step">
    <span class="jpeg-step-number" aria-hidden="true">${index + 1}</span>
    <div class="jpeg-step-card${stage.tag ? ' jpeg-step-lossy' : ''}">
      <div class="jpeg-step-heading"><a href="${stage.href}">${escapeHtml(stage.title)}</a>${stage.tag ? `<span class="jpeg-step-tag">${escapeHtml(stage.tag)}</span>` : ''}</div>
      ${stage.branches ? `<div class="jpeg-branches">${stage.branches.map(branch => `<div class="jpeg-branch"><strong>${branch.label}</strong><p>${escapeHtml(branch.text)}</p><code>${escapeHtml(branch.code)}</code></div>`).join('')}</div>` : `<div class="jpeg-step-body"><p>${escapeHtml(stage.detail)}</p><code>${escapeHtml(stage.output)}</code></div>`}
    </div>
  </li>`).join('');

  return `<figure class="jpeg-diagram" id="jpeg-encoding-pipeline" data-jpeg-diagram aria-labelledby="jpeg-pipeline-title" aria-describedby="jpeg-pipeline-scope">
    <figcaption id="jpeg-pipeline-title">JPEG: from pixels to a file</figcaption>
    <p class="jpeg-scope" id="jpeg-pipeline-scope">Common 8-bit baseline color JPEG. Chroma subsampling is optional; 4:2:0 is one choice. Follow the arrows from top to bottom; select a step for details.</p>
    <ol class="jpeg-steps">${steps}</ol>
    <p class="jpeg-takeaway">Quantization and optional chroma subsampling discard information. Zigzag and entropy coding preserve the quantized coefficients.</p>
  </figure>`;
}
