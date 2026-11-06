// Original, code-native teaching figures. All content survives without JS.
export const transformerDiagramIds = Object.freeze(['overview', 'heads', 'masks', 'decoding']);

const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const shape = text => `<code class="tf-shape">${escape(text)}</code>`;
const tokens = (words, extra = '') => `<div class="tf-tokens ${extra}">${words.map(word => `<span>${escape(word)}</span>`).join('')}</div>`;
const down = '<div class="tf-down" aria-hidden="true">↓</div>';
const figure = (id, title, intro, body, note) => `<figure class="transformer-diagram" id="tf-${id}" data-transformer-diagram="${id}" aria-labelledby="tf-${id}-title">
  <figcaption><strong id="tf-${id}-title">${title}</strong><span>TENSOR LAB · ${id.toUpperCase()}</span></figcaption>
  <p class="tf-intro">${intro}</p>${body}<p class="tf-note">${note}</p>
  <p class="tf-sr-only" data-transformer-live aria-live="polite" aria-atomic="true"></p>
</figure>`;

function stage(title, tensor, body, kind = '') {
  return `<section class="tf-stage ${kind}"><div class="tf-stage-title"><strong>${title}</strong>${shape(tensor)}</div>${body}</section>`;
}

function sublayer(title, formula, body, tensor, kind = '') {
  return stage(title, tensor, `<p>${body}</p><div class="tf-residual"><span>↳ residual: keep the input</span><code>${escape(formula)}</code><span>Add first → LayerNorm over the 8 features of each token</span></div>`, kind);
}

function overview() {
  const encoder = `<section class="tf-lane tf-encoder"><p class="tf-lane-label">ENCODER · SOURCE LENGTH S = 5</p>
    ${stage('Source token IDs', '[B,S] = [2,5]', '<p>Two examples; five positions each. IDs are integers, not feature vectors.</p>')}${down}
    ${stage('Embedding + position', 'E0: [2,5,8]', '<p>Lookup → scale by √8 → add sinusoidal position vectors.</p><p><code>[2,5,8] + [1,5,8]</code> broadcasts across the batch.</p>', 'tf-source')}${down}
    ${sublayer('1 · Multi-head self-attention', 'E1 = LN(E0 + MHA(E0, E0, E0))', 'Q, K and V come from E0. Each source token can read every non-padding source key.', 'E1: [2,5,8]', 'tf-source')}${down}
    ${sublayer('2 · Position-wise feed-forward', 'M = LN(E1 + FFN(E1))', 'Linear 8 → 32 · ReLU · Linear 32 → 8. The same FFN processes each token independently.', 'M: [2,5,8]', 'tf-source')}${down}
    ${stage('Encoder memory M', '[2,5,8]', '<p>One contextual vector per source position. This is a sequence, not one pooled vector.</p><div class="tf-memory-link"><strong>M → decoder cross-attention</strong><span>Supplies both K and V; does not replace decoder Q.</span></div>', 'tf-memory')}
  </section>`;
  const decoder = `<section class="tf-lane tf-decoder"><p class="tf-lane-label">DECODER · TARGET LENGTH T = 4</p>
    ${stage('Shifted target token IDs', '[B,T] = [2,4]', '<p>Training input starts with BOS. The label at each position is the next token.</p>')}${down}
    ${stage('Embedding + position', 'Z0: [2,4,8]', '<p>Separate target embedding → scale by √8 → add target position vectors.</p><p><code>[2,4,8] + [1,4,8]</code></p>', 'tf-target')}${down}
    ${sublayer('1 · Masked multi-head self-attention', 'Z1 = LN(Z0 + MHA(Z0, Z0, Z0))', 'Q, K and V come from Z0. A target position can read itself and earlier non-padding positions only.', 'Z1: [2,4,8]', 'tf-target')}${down}
    ${sublayer('2 · Multi-head cross-attention', 'Z2 = LN(Z1 + MHA(Z1, M, M))', '<strong>Q ← Z1; K, V ← encoder memory M.</strong> Each target position can read the non-padding source positions.', 'Z2: [2,4,8]', 'tf-memory')}${down}
    ${sublayer('3 · Position-wise feed-forward', 'Z3 = LN(Z2 + FFN(Z2))', 'Linear 8 → 32 · ReLU · Linear 32 → 8. Restore width 8 before adding the residual.', 'Z3: [2,4,8]', 'tf-target')}${down}
    ${stage('Vocabulary projection', 'logits: [2,4,8]', '<p>Linear D = 8 → V_tgt = 8. The last axis now means vocabulary scores, not hidden features.</p><p>Training: cross-entropy with labels <code>[2,4]</code>. Inference: choose from the last position.</p>')}
  </section>`;
  return figure('overview', 'One encoder block → one decoder block', 'Read each lane downward. B = 2 examples, S = 5 source positions, T = 4 target positions, D = 8 features, H = 2 heads, and 4 features per head.', `<div class="tf-architecture">${encoder}${decoder}</div><div class="tf-summary-strip"><strong>Same external width; different jobs</strong><span>Attention mixes tokens · FFN transforms features · LayerNorm normalizes features · residuals preserve an input path.</span></div>`, 'Original post-norm layout: LayerNorm(x + sublayer(x)). Dropout is omitted in this teaching model. All shapes include the batch dimension; source and target lengths need not match.');
}

function scenarios(id, states) {
  return `<div class="tf-controls" role="group" aria-label="Choose ${escape(id)} example">${states.map((state, index) => `<button type="button" data-transformer-choice="${state.key}" aria-pressed="${index === 0}" aria-controls="tf-${id}-${state.key}">${state.label}</button>`).join('')}</div>
    <div class="tf-panels">${states.map(state => `<section class="tf-panel" data-transformer-panel="${state.key}" id="tf-${id}-${state.key}" aria-labelledby="tf-${id}-${state.key}-title"><p class="tf-panel-title" id="tf-${id}-${state.key}-title">${state.title}</p>${state.body}<p class="tf-explanation" data-transformer-explanation>${state.explanation}</p></section>`).join('')}</div>`;
}

function heads() {
  const states = [
    {key: 'encoder-self', label: 'Encoder self', title: 'Source reads source', q: 'E0', kv: 'E0', lq: 5, lk: 5, mask: 'Source padding mask', explanation: 'All three inputs are the same source sequence, but Q, K and V have different learned projections. Each head produces a 5 × 5 table: one row per source query, one column per source key.'},
    {key: 'decoder-self', label: 'Decoder self', title: 'Target prefix reads itself', q: 'Z0', kv: 'Z0', lq: 4, lk: 4, mask: 'Causal AND target-key padding mask', explanation: 'Each head has a 4 × 4 score table. Mask future keys before softmax; the causal rule is what prevents the input at a later target position from leaking into an earlier prediction.'},
    {key: 'cross', label: 'Cross-attention', title: 'Target queries read source memory', q: 'Z1', kv: 'M', lq: 4, lk: 5, mask: 'Source padding mask; no target-to-source causal triangle', explanation: 'The 4 × 5 table is not square: four target queries attend to five source keys. Output length follows Q, not K or V, so the result returns to [2,4,8].'},
  ].map(state => ({...state, body: `<div class="tf-qkv-origins"><div><strong>Query input</strong><span>Q input ← ${state.q}</span>${shape(`[2,${state.lq},8]`)}</div><div><strong>Key + value input</strong><span>K input, V input ← ${state.kv}</span>${shape(`[2,${state.lk},8]`)}</div></div>
    <ol class="tf-shape-flow">
      <li><div><strong>1 · Project Q, K, V</strong><p>Three independent Linear(8, 8) layers; not three copies of one weight matrix.</p></div><div>${shape(`Q [2,${state.lq},8]`)}${shape(`K, V [2,${state.lk},8]`)}</div></li>
      <li><div><strong>2 · Split features into heads</strong><p>Reshape <code>[B,L,8] → [B,L,2,4]</code>, then transpose to <code>[B,2,L,4]</code>. Split features, not tokens.</p></div><div>${shape(`Q [2,2,${state.lq},4]`)}${shape(`K, V [2,2,${state.lk},4]`)}</div></li>
      <li><div><strong>3 · Score every query–key pair</strong><p><code>Q @ K.transpose(-2,-1) / √4</code>. Scaling divisor = 2, not √8.</p></div><div>${shape(`scores [2,2,${state.lq},${state.lk}]`)}</div></li>
      <li><div><strong>4 · Mask → softmax over keys</strong><p>${state.mask}. Blocked scores become −∞, so their attention weights become 0.</p></div><div>${shape(`weights [2,2,${state.lq},${state.lk}]`)}<small>Last axis sums to 1 for each query/head.</small></div></li>
      <li><div><strong>5 · Weighted sum of values</strong><p><code>weights @ V</code>: reduce over ${state.lk} key/value positions; keep ${state.lq} query positions.</p></div><div>${shape(`context [2,2,${state.lq},4]`)}</div></li>
      <li><div><strong>6 · Rejoin heads + output projection</strong><p>Transpose → <code>[2,${state.lq},2,4]</code>; concatenate the feature axis → <code>[2,${state.lq},8]</code>; apply Linear(8, 8).</p></div><div>${shape(`output [2,${state.lq},8]`)}</div></li>
    </ol>`}));
  return figure('heads', 'The same attention calculation, three different inputs', 'Choose an attention site. The operation is shared; the origin and lengths of Q, K and V change. Each of the three sites has its own learned parameters.', scenarios('heads', states), 'B = 2, H = 2, D_head = 4. Attention is independent across batch examples and heads; only the final output projection mixes concatenated head features. Every valid query must have at least one allowed key.');
}

function maskTable(kind, title, columns, allow) {
  const queries = kind === 'causal' ? ['BOS', 'nous', 'regardons', 'video'] : ['BOS', 'tu', 'regardes', 'EOS'];
  return `<section class="tf-mask-example"><p class="tf-panel-title">${title}</p><div class="tf-table-scroll" tabindex="0" role="region" aria-label="${escape(title)}; query rows and key columns"><table class="tf-mask-table" data-transformer-mask="${kind}"><caption>Rows = target queries · columns = ${kind === 'causal' ? 'target' : 'source'} keys</caption><thead><tr><th scope="col">Query ↓ / Key →</th>${columns.map(token => `<th scope="col">${token}</th>`).join('')}</tr></thead><tbody>${queries.map((query, row) => `<tr data-transformer-query="${row}"><th scope="row"><span>${row}</span> ${query}</th>${columns.map((_, col) => `<td class="${allow(row, col) ? 'tf-allowed' : 'tf-blocked'}" data-allowed="${allow(row, col)}" aria-label="${allow(row, col) ? 'Allowed' : 'Blocked'}">${allow(row, col) ? '✓' : '—'}</td>`).join('')}</tr>`).join('')}</tbody></table></div></section>`;
}

function masks() {
  const controls = `<div class="tf-controls" role="group" aria-label="Highlight the same target query position in both examples">${[0,1,2,3].map(i => `<button type="button" data-transformer-query-choice="${i}" aria-pressed="${i === 2}">Query position ${i}</button>`).join('')}</div>`;
  return figure('masks', 'A mask removes keys before softmax', '✓ = allowed (True); — = blocked (False) in this custom implementation. A blocked score becomes −∞, not 0. Select a row position in both examples; the tables show different batch examples.', `${controls}<div class="tf-mask-pair">${maskTable('causal', 'Decoder self-attention · batch example 0', ['BOS', 'nous', 'regardons', 'video'], (r,c) => c <= r)}${maskTable('padding', 'Cross-attention · batch example 1', ['you', 'watch', 'video', 'PAD', 'PAD'], (_,c) => c < 3)}</div>
    <div class="tf-mask-shapes"><p><strong>Causal mask:</strong> <code>[1,1,4,4]</code> broadcasts over B = 2 and H = 2.</p><p><strong>Source-key mask:</strong> <code>[2,1,1,5]</code> broadcasts over heads and target queries. The table shows batch example 1: <code>[you, watch, video, PAD, PAD]</code>.</p><p><strong>Decoder self mask:</strong> causal <code>[1,1,T,T]</code> AND target-key padding <code>[B,1,1,T]</code> → <code>[B,1,T,T]</code>.</p></div>
    <p class="tf-query-summary" data-transformer-query-summary>At position 2: batch 0 query regardons can read BOS, nous and regardons in self-attention. Batch 1 query regardes can read you, watch and video in cross-attention, but not PAD.</p>`, 'The decoder input has no PAD in either illustrated row; the second row’s final label is PAD and is ignored in the loss. Masking padding keys does not force outputs at padded query positions to zero. Boolean mask polarity differs across library APIs; True = allowed is specific to this chapter’s custom attention.');
}

function decoding() {
  const target = ['BOS', 'nous', 'regardons', 'video'];
  const labels = ['nous', 'regardons', 'video', 'EOS'];
  const training = {key: 'training', label: 'Training', title: 'Teacher forcing: all target positions in one forward pass', body: `<div class="tf-training-rows"><div><strong>Decoder input</strong>${tokens(target)}${shape('[2,4]')}</div><div><strong>Labels to predict</strong>${tokens(labels, 'tf-label-tokens')}${shape('[2,4]')}</div></div><div class="tf-decode-steps"><span>Embed + position</span><b>↓</b><span>Masked self → cross → FFN</span><b>↓</b><span>Vocabulary projection ${shape('[2,4,8]')}</span><b>↓</b><span>Cross-entropy against shifted labels</span></div>`, explanation: 'Ground-truth target prefixes supply every input position. The causal mask allows parallel training without revealing future target tokens to an earlier query. The batch has two examples; only example 0 is illustrated. Here V_tgt = 8.'};
  const inference = target.map((_, i) => ({key: `step-${i+1}`, label: `Generate ${i+1}`, title: `Inference step ${i+1}: prefix length ${i+1}`, body: `<div class="tf-fixed-memory"><strong>Reuse encoder memory M</strong>${shape('[2,5,8]')}<span>Encode the source once; keep the same memory at every generation step.</span></div><div class="tf-prefix"><strong>Current prefix · example 0</strong>${tokens(target.slice(0,i+1))}${shape(`[2,${i+1}]`)}</div><div class="tf-decode-steps"><span>Decoder block + vocabulary projection</span><b>↓</b><span>All prefix logits ${shape(`[2,${i+1},8]`)}</span><b>↓</b><span>Use only <code>logits[:, -1, :]</code> ${shape('[2,8]')}</span><b>↓</b><span>Illustrative chosen token: <strong>${labels[i]}</strong>${i < 3 ? ' → append to the prefix' : ' → end this example'}</span></div>`, explanation: i < 3 ? `The next call receives the longer prefix [${target.slice(0,i+1).concat(labels[i]).join(', ')}] in example 0. This simple version recomputes the decoder prefix; it has no KV cache.` : 'EOS marks the end of this illustrated sequence. Real batched generation tracks finished examples separately and also needs a maximum generation length.'}));
  return figure('decoding', 'Training shifts once; inference grows the prefix', 'Switch between a full training pass and four generation steps. Same encoder and decoder; different availability of target tokens.', scenarios('decoding', [training, ...inference]), 'Token choices here are a teaching illustration, not predictions produced by the untrained demo. A randomly initialized model does not translate; running a forward pass or one optimizer step does not establish translation quality.');
}

const renderers = {overview, heads, masks, decoding};

export function renderTransformerDiagram(id) {
  if (!Object.hasOwn(renderers, id)) throw new Error(`Unknown Transformer diagram: ${id}`);
  return renderers[id]();
}
