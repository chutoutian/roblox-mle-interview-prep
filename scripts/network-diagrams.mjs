// Native, offline teaching figures. Every scenario is present in the HTML so
// these remain useful without JavaScript and when printed.
export const networkDiagramIds = Object.freeze([
  'packet-path', 'tcp-handshake', 'tcp-udp-loss', 'http-path', 'dash-flow',
  'low-latency', 'abr-buffer', 'webrtc-connect', 'realtime-budget', 'recovery', 'cdn-flow',
]);

export const networkModels = Object.freeze({
  packetPath: {
    request: 'GET /hello HTTP/1.1\r\nHost: example.test\r\n\r\n', requestBytes: 43,
    tcpSequence: 1001, nextAck: 1044, tcpHeaderBytes: 20, ipHeaderBytes: 20,
    ethernetHeaderBytes: 14, ethernetFcsBytes: 4, senderTtl: 64, receiverTtl: 63,
  },
  handshake: {clientInitialSequence: 100, serverInitialSequence: 500, synSequenceSpace: 1},
  abr: {segmentSeconds: 4, representationsMbps: [2, 4, 8], safetyFactor: 0.8,
    throughputScenariosMbps: [2, 5, 10], startingBuffersSeconds: [6, 2]},
  lowLatency: {segmentSeconds: 2, partSeconds: 0.5, exampleStartupBufferSeconds: 1},
  realtime: [
    {label: 'Capture', milliseconds: 16}, {label: 'Encode', milliseconds: 20},
    {label: 'Pacing + network', milliseconds: 35}, {label: 'Jitter buffer', milliseconds: 50},
    {label: 'Decode + render', milliseconds: 29},
  ],
  recovery: {retryRoundTripMilliseconds: 40, remainingBudgetsMilliseconds: [100, 20]},
});

export function calculateAbrScenario(throughputMbps, bufferSeconds) {
  if (!(throughputMbps > 0) || !(bufferSeconds >= 0)) throw new RangeError('Positive throughput and nonnegative buffer required.');
  const model = networkModels.abr;
  const safeBudgetMbps = throughputMbps * model.safetyFactor;
  const fitting = model.representationsMbps.filter(rate => rate <= safeBudgetMbps);
  const representationMbps = fitting.at(-1) ?? model.representationsMbps[0];
  const segmentMegabits = representationMbps * model.segmentSeconds;
  const downloadSeconds = segmentMegabits / throughputMbps;
  return {throughputMbps, bufferSeconds, safeBudgetMbps, representationMbps,
    lowestRungFallback: fitting.length === 0, segmentMegabits, downloadSeconds,
    stallSeconds: Math.max(0, downloadSeconds - bufferSeconds),
    newBufferSeconds: Math.max(0, bufferSeconds - downloadSeconds) + model.segmentSeconds};
}

const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const number = value => Number(value.toFixed(2)).toString();
const arrow = '<span class="network-arrow" aria-hidden="true">→</span>';
const flow = (steps, extra = '') => `<ol class="network-flow ${extra}">${steps.map(([title, body]) => `<li><strong>${title}</strong><span>${body}</span></li>`).join('')}</ol>`;
const chip = (text, kind = '') => `<span class="network-packet ${kind ? `network-${kind}` : ''}">${text}</span>`;
const note = text => `<p class="network-note">${text}</p>`;
const figure = (id, title, intro, body, footnote = '') => `<figure class="network-diagram" id="net-${id}" data-network-diagram="${id}" aria-labelledby="net-${id}-title">
  <figcaption><strong id="net-${id}-title">${title}</strong><span>NETWORK LAB · ${id.toUpperCase().replaceAll('-', ' ')}</span></figcaption>
  <p class="network-intro">${intro}</p>${body}${footnote ? note(footnote) : ''}
</figure>`;

function scenarios(id, label, states, before = '') {
  const controls = `<div class="network-controls" role="group" aria-label="${escape(label)}">${states.map((state, index) => `<button type="button" data-network-choice="${state.key}" aria-pressed="${index === 0}" aria-controls="net-${id}-${state.key}">${state.label}</button>`).join('')}</div>`;
  const panels = states.map(state => `<section class="network-state" id="net-${id}-${state.key}" data-network-panel="${state.key}" aria-labelledby="net-${id}-${state.key}-title"><p class="network-state-title" id="net-${id}-${state.key}-title">${state.title}</p>${state.body}<p class="network-explanation" data-network-explanation>${state.explanation}</p></section>`).join('');
  return `${controls}${before}<div class="network-states">${panels}</div><p class="network-live network-sr-only" data-network-live aria-live="polite" aria-atomic="true"></p>`;
}

function packetPath() {
  const model = networkModels.packetPath;
  const segmentBytes = model.tcpHeaderBytes + model.requestBytes;
  const packetBytes = model.ipHeaderBytes + segmentBytes;
  const frameBytes = model.ethernetHeaderBytes + packetBytes + model.ethernetFcsBytes;
  const layers = [
    ['application', '5 · Application', 'HTTP · message'],
    ['transport', '4 · Transport', 'TCP · segment'],
    ['network', '3 · Network', 'IPv4 · packet'],
    ['link', '2 · Link', 'Ethernet · frame'],
    ['physical', '1 · Physical', 'Encoded signals'],
  ];
  const stack = (kind, title, description) => `<section class="network-protocol-stack" data-protocol-stack="${kind}"><div class="network-stack-heading"><strong>${title}</strong><span>${description}</span></div>${layers.map(([id, label, protocol], index) => kind === 'router' && index < 2
    ? `<div class="network-peer-line"><span>${index === 0 ? 'HTTP peers' : 'TCP peers'}</span><small>logical relationship</small></div>`
    : `<div class="network-stack-cell network-layer-${id}" data-stack-layer="${id}"><strong>${label}</strong><span>${kind === 'router' ? {network:'Choose next hop',link:'Replace link envelope',physical:'Receive / transmit'}[id] : protocol}</span></div>`).join('')}</section>`;
  const block = (kind, label, size) => `<span class="network-byte-piece network-layer-${kind}"><b>${label}</b><small>${size} B</small></span>`;
  const message = block('application', 'M · HTTP bytes', model.requestBytes);
  const tcp = block('transport', 'Hₜ · TCP', model.tcpHeaderBytes);
  const ip = block('network', 'Hₙ · IPv4', model.ipHeaderBytes);
  const eth = block('link', 'Hₗ · Ethernet', model.ethernetHeaderBytes);
  const fcs = block('link', 'FCS', model.ethernetFcsBytes);
  const rows = [
    ['application', '5 · Application', 'HTTP message · 43 B', message, '<code>GET /hello HTTP/1.1\\r\\nHost: example.test\\r\\n\\r\\n</code><br>The entire request above is M. HTTP headers are part of M; they are not the TCP header.'],
    ['transport', '4 · Transport', `TCP segment · ${segmentBytes} B`, tcp + message, `<code>src port=51514 · dst port=80 · seq=${model.tcpSequence}</code><br>Hₜ also carries ACK, flags, receive window and checksum. After these ${model.requestBytes} bytes arrive contiguously, the server can return <code>ACK=${model.nextAck}</code>. TCP treats M as bytes, not as an HTTP command.`],
    ['network', '3 · Network', `IPv4 packet · ${packetBytes} B`, ip + tcp + message, '<code>src=192.0.2.10 · dst=198.51.100.20 · TTL=64 · protocol=6 (TCP)</code><br>Hₙ supplies the destination used for routing. The complete TCP segment is its payload.'],
    ['link', '2 · Link', `Ethernet frame · ${frameBytes} B`, eth + ip + tcp + message + fcs, '<code>src MAC=Client · dst MAC=Router-in · EtherType=0x0800 (IPv4)</code><br>Hₗ addresses this link; the FCS trailer detects frame corruption. The server is not the first frame’s MAC destination.'],
    ['physical', '1 · Physical', 'Bits carried as signals', '<span class="network-signal-chain"><b>Frame bits</b><span>→ encode →</span><b>Electrical signal</b><span>→ recover →</span><b>Frame bits</b></span>', '<code>One byte inside M: G = 0x47 = 01000111</code><br>This is a byte representation, not an Ethernet waveform or wire-bit ordering. Each interface encodes frame bits into signals for its link; the coding depends on the physical standard. There is no universal extra header.'],
  ];
  return figure('packet-path', 'Five layers: one HTTP request, nested envelopes', 'Plain HTTP/1.1 over an established TCP connection, IPv4, and two Ethernet links. Read the sender downward; read the receiver upward.', `
    <div class="network-stack-map">
      ${stack('sender', 'CLIENT · SEND ↓', '192.0.2.10:51514')}
      ${stack('router', 'ROUTER · FORWARD', 'Network / Link / Physical')}
      ${stack('receiver', 'SERVER · RECEIVE ↑', '198.51.100.20:80')}
    </div>
    <p class="network-stack-legend">Dashed lines connect logical peers; HTTP and TCP do not travel directly across them. For this transit packet, the router does not pass the payload up to TCP or HTTP.</p>
    <div class="network-actual-route"><strong>Actual data path</strong><span>Client: 5 ↓ 1</span>${arrow}<span>Link A → Router: 1 ↑ 3 ↓ 1</span>${arrow}<span>Link B → Server: 1 ↑ 5</span></div>
    <p class="network-subfigure-title">What the sender hands to each lower layer</p>
    <p class="network-stack-legend">M = message bytes · H = header · FCS = frame check sequence. Colors identify layers; block widths are not byte-scaled.</p>
    <div class="network-encapsulation">${rows.map(([id,label,unit,bytes,detail]) => `<section class="network-encapsulation-row" data-encapsulation-layer="${id}"><div class="network-layer-label"><strong>${label}</strong><span>${unit}</span></div><div class="network-layer-content"><div class="network-layer-bytes">${bytes}</div><p>${detail}</p></div></section>`).join('')}</div>
    <div class="network-hop-comparison">
      <section><strong>Link A · Client → Router</strong><p>Ethernet: <b>Client MAC → Router-in MAC</b></p><p>IP: <b>192.0.2.10 → 198.51.100.20</b><br>TTL = ${model.senderTtl}</p></section>
      <section><strong>Link B · Router → Server</strong><p>Ethernet: <b>Router-out MAC → Server MAC</b></p><p>IP: <b>192.0.2.10 → 198.51.100.20</b><br>TTL = ${model.receiverTtl}</p></section>
    </div>
    <p class="network-stack-legend">At the router: new Ethernet header and FCS; updated TTL and IPv4 header checksum. The TCP segment and HTTP bytes stay intact. MAC names above are labels for four distinct interface addresses.</p>`,
    'Teaching assumptions: DNS and ARP already resolved; one request fits one TCP segment; no NAT, fragmentation, TLS, VLAN tags, or IP/TCP options. Sizes exclude Ethernet preamble/SFD and the inter-frame gap. A large request need not fit one segment, and TCP does not preserve application-message boundaries.');
}

function handshake() {
  const rows = [
    {key: 'syn', direction: 'right', name: '1 · SYN', detail: 'seq=100', left: 'SYN-SENT', right: 'SYN-RECEIVED'},
    {key: 'synack', direction: 'left', name: '2 · SYN + ACK', detail: 'seq=500 · ack=101', left: 'Receives SYN + ACK', right: 'Sends SYN + ACK'},
    {key: 'ack', direction: 'right', name: '3 · ACK', detail: 'seq=101 · ack=501', left: 'ESTABLISHED', right: 'ESTABLISHED after ACK'},
  ];
  const diagram = `<div class="network-sequence" aria-label="TCP client and server handshake; read downward in time"><div class="network-sequence-heading"><b>CLIENT</b><span>time ↓</span><b>SERVER</b></div>${rows.map(row => `<div class="network-sequence-row" data-network-highlight="${row.key}"><span>${row.left}</span><div class="network-sequence-message network-direction-${row.direction}"><strong>${row.name}</strong><code>${row.detail}</code><span class="network-message-line" aria-hidden="true">${row.direction === 'left' ? '←' : '→'}</span></div><span>${row.right}</span></div>`).join('')}</div>`;
  return figure('tcp-handshake', 'A TCP connection starts with sequence numbers', 'Select each exchange. Assume no application data is carried in these three segments.', scenarios('tcp-handshake', 'Inspect a handshake exchange', [
    {key: 'syn', label: '1 · SYN', title: 'Client proposes its initial sequence number', body: `<div class="network-equation"><code>100 + 1 SYN = 101</code></div>`, explanation: 'The SYN consumes one sequence number even though it carries no application byte. The server next expects client sequence number 101.'},
    {key: 'synack', label: '2 · SYN + ACK', title: 'Server acknowledges and proposes its own sequence number', body: `<div class="network-equation"><code>ack=101</code><span>acknowledges the client SYN</span><code>500 + 1 SYN = 501</code></div>`, explanation: 'The two directions have independent sequence spaces. Server ack=101 acknowledges the client SYN; the server SYN starts its own sequence space at 500.'},
    {key: 'ack', label: '3 · ACK', title: 'Client acknowledges the server SYN', body: `<div class="network-equation"><code>seq=101 · ack=501</code></div>`, explanation: 'An ACK without payload, SYN or FIN consumes no sequence number. The next client data can start at 101; the next server data can start at 501.'},
  ], diagram), 'ACK means “next sequence number expected.” This is the ordinary three-way handshake, without TCP Fast Open or simultaneous open.');
}

function loss() {
  const lanes = (stage) => {
    const gap = stage !== 'sent', recovered = stage === 'recovered';
    return `<div class="network-loss-layout">${['TCP byte stream', 'UDP datagrams'].map((protocol, index) => `<section class="network-loss-lane"><p class="network-lane-title">${protocol}</p><div class="network-labeled-row"><b>Send</b><div class="network-packets">${chip('A')}${chip('B', gap ? 'lost' : '')}${chip('C')}</div></div><div class="network-wire-label">${gap ? 'B is lost on the path ×' : 'Three chunks leave the sender'}<span aria-hidden="true">↓</span></div><div class="network-labeled-row"><b>Receive</b><div class="network-packets">${chip('A', 'good')}${gap ? chip(recovered && index === 0 ? 'B retry' : 'B missing', recovered && index === 0 ? 'good' : 'lost') : chip('B')}${chip('C', index === 0 && gap && !recovered ? 'waiting' : 'good')}</div></div><div class="network-labeled-row"><b>App sees</b><div class="network-packets">${chip('A', 'good')}${!gap || (recovered && index === 0) ? chip('B', 'good') : ''}${!gap || index === 1 || recovered ? chip('C', 'good') : chip('C waits', 'waiting')}</div></div><p>${index === 0 ? (gap && !recovered ? 'C can be buffered, but cannot pass the missing bytes of B in this stream.' : recovered ? 'Retransmitted B closes the gap. The application receives ordered bytes A, B, C.' : 'TCP exposes an ordered byte stream; these chunk labels are teaching units.') : (gap ? 'C can be delivered despite missing B. UDP itself does not request B again.' : 'Each datagram is independent. These are datagram boundaries, unlike TCP.')}</p></section>`).join('')}</div>`;
  };
  return figure('tcp-udp-loss', 'The same loss, different delivery rules', 'A, B and C denote consecutive application chunks. For simplicity, each chunk fits one packet or datagram here.', scenarios('tcp-udp-loss', 'Inspect a loss scenario', [
    {key: 'sent', label: '1 · No loss', title: 'Baseline: all three arrive', body: lanes('sent'), explanation: 'Both applications receive all data in this illustrative arrival order. UDP does not promise this order or successful delivery.'},
    {key: 'gap', label: '2 · B is lost', title: 'C arrives while B is missing', body: lanes('gap'), explanation: 'TCP preserves order within the connection byte stream, so the gap blocks later bytes from application delivery. UDP provides no built-in ordered delivery or recovery.'},
    {key: 'recovered', label: '3 · Recovery', title: 'TCP retransmits B', body: lanes('recovered'), explanation: 'Once B arrives, TCP can release the contiguous bytes. A UDP application needs its own recovery or loss-tolerance mechanism if missing B matters.'},
  ]), 'QUIC runs over UDP and adds reliable, ordered streams. Loss in one QUIC stream need not block delivery in another, although congestion control and shared network capacity still couple their performance.');
}

function httpPath() {
  return figure('http-path', 'Before the first response byte', 'A conceptual path for an HTTPS request when there is no usable browser-cache entry and no reusable connection.', flow([
    ['1 · Resolve', 'DNS: name → server address. A cached DNS result can shorten this phase.'],
    ['2 · Connect + secure', 'HTTP/1.1 or HTTP/2: TCP + TLS. HTTP/3: QUIC, with TLS 1.3 integrated.'],
    ['3 · Request', 'Send HTTP method, target and headers; include a request body if needed.'],
    ['4 · Serve', 'A CDN/cache or origin handles the request and begins the response.'],
    ['5 · Receive + use', 'Receive headers/body; cache if policy permits; parse, decode or render.'],
  ]) + `<div class="network-bracket"><span>DNS / connection setup</span>${arrow}<span>Request → first byte</span>${arrow}<span>Body transfer → useful output</span></div>`, 'This is a dependency map, not a fixed timing formula. Reused connections, cached results, cache revalidation, retries and protocol choices change the critical path. HTTP semantics remain separate from the video codec.');
}

function dash() {
  return figure('dash-flow', 'DASH: choose bytes for the next interval', 'The MPD manifest describes the presentation, representations and segment addressing. In this simplified switching set, all representations share aligned 4-second intervals.', flow([
    ['Read MPD', 'Discover representations and segment locations.'],
    ['Choose a rung', 'Use throughput, buffer and device constraints.'],
    ['Fetch segment', 'An HTTP request downloads the chosen interval.'],
    ['Buffer → decode', 'Queue media time, then present it on schedule.'],
  ]) + `<div class="network-scroll" tabindex="0" role="region" aria-label="Aligned DASH representation timeline; scroll horizontally on small screens"><table class="network-timeline-table"><caption>One possible selection across four media intervals</caption><thead><tr><th scope="col">Representation</th>${['0–4 s', '4–8 s', '8–12 s', '12–16 s'].map(t => `<th scope="col">${t}</th>`).join('')}</tr></thead><tbody>${[8,4,2].map(rate => `<tr><th scope="row">${rate} Mb/s</th>${[0,1,2,3].map(index => `<td class="${[2,4,8,4][index] === rate ? 'network-selected-cell' : ''}">${[2,4,8,4][index] === rate ? '✓ FETCH' : 'available'}<small>segment ${index + 1}</small></td>`).join('')}</tr>`).join('')}</tbody></table></div><p class="network-axis-label">Media time → · each column describes the same content interval</p>`, 'Alignment alone is not every switching requirement: usable random-access points, decoder configuration and manifest constraints also matter. This ladder and selection are illustrative, not a mandated DASH algorithm.');
}

function lowLatency() {
  const states = [1,2,3,4].map(count => {
    const elapsed = count * .5;
    const parts = Array.from({length:4}, (_, index) => `<div class="network-part ${index < count ? 'network-good' : 'network-future'}"><strong>P${index+1}</strong><span>${index < count ? 'available' : 'producing later'}</span><small>${index*.5}–${(index+1)*.5} s</small></div>`).join('');
    return {key: `part${count}`, label: `${elapsed.toFixed(1)} s produced`, title: `Production time: ${elapsed.toFixed(1)} seconds`, body: `<div class="network-availability"><p class="network-lane-title">COMPLETE-SEGMENT DELIVERY</p><div class="network-whole-segment ${count === 4 ? 'network-good' : 'network-waiting'}"><strong>One 2 s segment</strong><span>${count === 4 ? 'Available now' : `Wait until 2.0 s · ${elapsed.toFixed(1)} s produced`}</span></div><p class="network-lane-title">PART / CHUNK DELIVERY</p><div class="network-parts">${parts}</div><p class="network-startup">Example startup buffer: <strong>1 s = two parts</strong>. ${count < 2 ? 'One part does not meet that target yet.' : 'Enough media has been produced for that target; transfer and player processing still take time.'}</p></div>`, explanation: 'Making partial media available during production removes the wait for the whole segment. It does not remove transport time, decoding time or the player’s safety buffer.'};
  });
  return figure('low-latency', 'Publish useful media before the segment ends', 'A deliberately simplified comparison: one 2-second segment contains four 0.5-second parts. The clock starts when production of that segment begins.', scenarios('low-latency', 'Choose how much media has been produced', states), '“Part” is a teaching label here: LL-HLS parts and low-latency DASH/CMAF chunks have different signaling and delivery details. Availability assumes each unit is published promptly; startup does not happen at zero buffer.');
}

function abr() {
  const states = networkModels.abr.startingBuffersSeconds.flatMap(buffer => networkModels.abr.throughputScenariosMbps.map(throughput => {
    const result = calculateAbrScenario(throughput, buffer);
    return {key: `t${throughput}-b${buffer}`, label: `${throughput} Mb/s · ${buffer} s buffer`, title: `${throughput} Mb/s throughput; ${buffer} seconds initially buffered`, body: `<div class="network-abr-ladder" aria-label="Available bitrates and selected representation">${networkModels.abr.representationsMbps.map(rate => `<span class="network-rung ${result.representationMbps === rate ? 'network-chosen' : ''}">${rate} Mb/s${result.representationMbps === rate ? '<b>CHOSEN</b>' : ''}</span>`).join('')}</div><div class="network-calculation"><p><span>80% budget</span><code>0.8 × ${throughput} = ${number(result.safeBudgetMbps)} Mb/s</code></p><p><span>Segment size</span><code>${result.representationMbps} Mb/s × 4 s = ${result.segmentMegabits} Mb</code></p><p><span>Download time</span><code>${result.segmentMegabits} Mb ÷ ${throughput} Mb/s = ${number(result.downloadSeconds)} s</code></p></div><div class="network-buffer-equation"><div><span>Before request</span><strong>${buffer} s</strong></div><span aria-hidden="true">−</span><div><span>Playback while downloading</span><strong>${number(Math.min(buffer,result.downloadSeconds))} s</strong></div><span aria-hidden="true">+</span><div><span>New media</span><strong>4 s</strong></div><span aria-hidden="true">=</span><div class="network-good"><span>New buffer</span><strong>${number(result.newBufferSeconds)} s</strong></div></div><p class="network-stall ${result.stallSeconds > 0 ? 'network-warning' : 'network-good'}">${result.stallSeconds > 0 ? `STALL: download takes ${number(result.downloadSeconds)} s, but only ${buffer} s was buffered. Playback pauses for ${number(result.stallSeconds)} s.` : 'NO STALL: the download finishes before the existing buffer runs out.'}</p>`, explanation: `${result.lowestRungFallback ? 'No rung fits the 80% budget, so this toy player falls back to the 2 Mb/s minimum. That fallback has no throughput headroom. ' : `The largest rung within the 80% budget is ${result.representationMbps} Mb/s. `}Buffer after download = max(0, B − download time) + 4 s. An adequate long-term bitrate does not guarantee a stall-free download when the starting buffer is thin.`};
  }));
  return figure('abr-buffer', 'ABR trades quality against buffer survival', 'Select a measured throughput and starting buffer. This toy policy chooses the highest of 2 / 4 / 8 Mb/s within 80% of throughput, then requests a 4-second segment.', scenarios('abr-buffer', 'Choose throughput and initial buffer', states), 'Model assumptions: constant throughput during one download; media size = bitrate × duration; playback consumes 1 second of buffer per second until empty. No request, header, RTT, decode or scheduling overhead. Whole segments become usable only after download. Real ABR also responds to uncertainty, content, latency and device limits.');
}

function webrtc() {
  return figure('webrtc-connect', 'Find a working path, then secure the media', 'The application coordinates the peers. ICE discovers and checks candidate paths; media then follows the selected path.', `
    <div class="network-signaling"><strong>Application signaling service</strong><div class="network-peer-exchange"><b>Peer A</b><span>↔ SDP + ICE candidates ↔</span><b>Peer B</b></div><p>Exchange session descriptions and candidates over an application-defined channel. Candidate gathering may use STUN and TURN.</p></div>
    <div class="network-down" aria-hidden="true">↓</div>
    ${flow([['ICE connectivity checks', 'Test candidate pairs and select a working pair.'], ['DTLS handshake', 'Authenticate the peer transport and establish keying material.'], ['SRTP / SRTCP media', 'Send protected RTP media and RTCP control on the selected path.']])}
    <div class="network-route-options"><section><p class="network-lane-title">IF A DIRECT PAIR WORKS</p><div class="network-peer-exchange"><b>Peer A</b><span>← protected media →</span><b>Peer B</b></div></section><section><p class="network-lane-title">IF A RELAY PAIR IS SELECTED</p><div class="network-relay-path"><b>Peer A</b>${arrow}<strong>TURN relay</strong>${arrow}<b>Peer B</b></div><p>TURN relays transport traffic. It does not need to decode the media.</p></section></div>`, 'Signaling is not the media path. Candidate exchange can overlap gathering (trickle ICE). TURN allocation enables relay candidates before they can be checked; a direct path is not guaranteed. This figure focuses on WebRTC media, not data channels.');
}

function realtimeBudget() {
  const stages = networkModels.realtime;
  const total = stages.reduce((sum, stage) => sum + stage.milliseconds, 0);
  let cumulative = 0;
  return figure('realtime-budget', 'Where a 150 ms example budget goes', 'An illustrative additive capture-to-display budget. These numbers are chosen for arithmetic, not performance promises or universal targets.', `<div class="network-budget-bar" role="img" aria-label="Capture 16 milliseconds, encode 20, pacing and network 35, jitter buffer 50, decode and render 29; total 150 milliseconds">${stages.map((stage,index) => `<span class="network-budget-piece network-color-${index}" style="flex:${stage.milliseconds}">${index+1}</span>`).join('')}</div><ol class="network-budget-stages">${stages.map((stage,index) => { const start = cumulative; cumulative += stage.milliseconds; return `<li><span class="network-stage-number network-color-${index}">${index+1}</span><div><strong>${stage.label}</strong><span>${stage.milliseconds} ms · cumulative ${start} → ${cumulative} ms</span></div><div class="network-stage-meter"><span class="network-color-${index}" style="width:${stage.milliseconds / 50 * 100}%"></span></div></li>`; }).join('')}</ol><div class="network-equation"><code>${stages.map(stage => stage.milliseconds).join(' + ')} = ${total} ms</code></div>`, 'A larger jitter buffer can absorb arrival variation but adds delay. Real implementations pipeline work and measure stage boundaries differently; do not blindly add overlapping measurements. Network variation changes the budget from frame to frame.');
}

function recovery() {
  const retry = networkModels.recovery.retryRoundTripMilliseconds;
  const states = networkModels.recovery.remainingBudgetsMilliseconds.map(budget => {
    const fits = retry < budget;
    return {key: `deadline${budget}`, label: `${budget} ms left`, title: `${budget} ms remain until this packet’s playout deadline`, body: `<div class="network-recovery-timeline"><div class="network-recovery-row"><b>Time remaining</b><div class="network-time-track"><span class="network-time-window" style="width:${budget}%">${budget} ms</span></div><span>deadline</span></div><div class="network-recovery-row"><b>NACK + retry</b><div class="network-time-track"><span class="network-time-retry ${fits ? 'network-good' : 'network-warning'}" style="width:${retry}%">40 ms</span></div><span>arrival</span></div><p class="network-axis-label">0 ms: loss detected / NACK sent → same 0–100 ms scale</p></div><div class="network-recovery-result ${fits ? 'network-good' : 'network-warning'}"><strong>${fits ? 'Retry can fit the deadline' : 'Retry misses the deadline'}</strong><code>${retry} ms ${fits ? '<' : '>'} ${budget} ms</code><span>${fits ? 'Illustrative remaining margin: 60 ms.' : 'The retransmission would arrive 20 ms too late for this playout deadline.'}</span></div>${!fits ? `<div class="network-recovery-alternatives">${flow([['Enough FEC already received?', 'Recover locally if the code and available parity cover this loss.'], ['Otherwise', 'Conceal or drop late data; manage damaged references if needed.']])}</div>` : ''}`, explanation: fits ? 'A retransmission is useful only if it arrives before the data is needed. This model treats 40 ms as the complete NACK-to-retransmission arrival time.' : 'FEC can avoid waiting for a feedback round trip when sufficient repair data is available in time. It costs bandwidth and may add waiting; it is not a guarantee of on-time recovery.'};
  });
  return figure('recovery', 'Recovery is a race against playout', 'Assume the loss has just been detected and a NACK-triggered retransmission takes 40 ms to return. Compare two remaining deadlines.', scenarios('recovery', 'Choose a remaining playout budget', states), 'Illustrative timing only: real recovery includes loss detection, feedback scheduling, sender retention, pacing and retransmission travel time. An RTT estimate alone is not a guaranteed repair arrival time.');
}

function cdn() {
  return figure('cdn-flow', 'A cache hit shortens the HTTP serving path', 'A video segment is an HTTP object to the CDN. The cache uses its configured key and freshness rules, not the codec’s picture-prediction structure.', `
    <div class="network-cdn-lane"><p class="network-lane-title">EDGE HIT</p><div class="network-cdn-flow"><strong>Client</strong><span>request →<br>← cached bytes</span><strong class="network-good">Edge cache · HIT</strong></div></div>
    <div class="network-cdn-lane"><p class="network-lane-title">EDGE MISS · SHIELD ALSO MISSES IN THIS EXAMPLE</p>${flow([['Client', 'Request segment URL'], ['Edge cache', 'Miss → request upstream'], ['Shield cache', 'Miss → request origin'], ['Origin', 'Return the object']])}<div class="network-cache-return"><strong>Client</strong><span>← bytes</span><strong>Edge stores*</strong><span>← bytes</span><strong>Shield stores*</strong><span>← bytes</span><strong>Origin</strong></div></div>`, '*Stored only when cache policy permits. A shield hit would avoid the origin request. An actual response may be streamed downstream while it is being fetched and cached; this diagram shows logical ownership, not a requirement to wait for the complete object.');
}

const renderers = {'packet-path': packetPath, 'tcp-handshake': handshake, 'tcp-udp-loss': loss,
  'http-path': httpPath, 'dash-flow': dash, 'low-latency': lowLatency, 'abr-buffer': abr,
  'webrtc-connect': webrtc, 'realtime-budget': realtimeBudget, recovery, 'cdn-flow': cdn};

export function renderNetworkDiagram(id) {
  if (!Object.hasOwn(renderers, id)) throw new Error(`Unknown network diagram: ${id}`);
  return renderers[id]();
}
