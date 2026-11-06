# Part V — Networking & Streaming

> A beginner-friendly path from packets and transport protocols to HTTP video, adaptive playback, WebRTC, CDNs, and production debugging. Codec internals remain in
> [01_video_domain.md](01_video_domain.md); ML systems remain in
> [02_ml_domain.md](02_ml_domain.md) and [02_ml_domain.md#6-generative-models--learned-compression](02_ml_domain.md#6-generative-models--learned-compression).

**How to use these notes**

- Read Chapters 1–3 in order if TCP, DNS, TLS, and HTTP are new to you.
- Chapters 4–5 follow one on-demand segment from a manifest into the playback buffer.
- Chapters 6–7 contrast deadline-driven real-time media with cache-friendly HTTP delivery.
- Chapter 8 turns metrics and traces into a repeatable debugging method.
- Numbers are worked examples, not universal production targets; measure the actual path and product.

**Contents**

| § | Topic | What you should be able to explain |
|---|-------|------------------------------------|
| 1 | Network foundations | Packets, addresses, queues, latency, loss, and bandwidth-delay product |
| 2 | UDP, TCP, and QUIC | Reliability, ordering, handshakes, congestion control, and trade-offs |
| 3 | HTTP and the request path | DNS → connection → TLS → HTTP → cache/origin response |
| 4 | HLS, DASH, and CMAF | Manifests, segments, timelines, switching, and low-latency delivery |
| 5 | Adaptive bitrate and QoE | Throughput, buffer, MPC/RL approaches, and player outcomes |
| 6 | Real-time media and WebRTC | ICE, RTP/RTCP, latency budgets, recovery, and congestion control |
| 7 | Delivery at scale | CDN hierarchy, cache keys, origin protection, and live-event resilience |
| 8 | Debugging | Layered triage, measurements, symptoms, and incident reasoning |
| 9 | Rapid-fire drill | Networking and streaming oral self-test |
| 10 | Coding reverse index | Prompt → knowledge point → implementation status |

---

# 1. Network Foundations

## 1.1 From an application message to packets

**TCP belongs to the transport layer; it is not the whole network stack.** We use the five-layer
Internet teaching model below, separating the physical layer from the link layer. These are five
different jobs applied to the same application data, not five alternative ways to send it.

| Layer | Question it answers | Protocol / technology in our example | Data at this layer |
|---|---|---|---|
| **5. Application** | What does the application want to say? | HTTP/1.1 | A request message: “get `/hello`” |
| **4. Transport** | Which application connection, and how are its bytes delivered? | TCP | A segment: TCP header + application bytes |
| **3. Network** | Which destination host, across which networks? | IPv4 | A packet: IP header + TCP segment |
| **2. Link** | Which next interface on this local link? | Ethernet | A frame: Ethernet header + IP packet + error-check trailer |
| **1. Physical** | How do the bits cross this medium? | An Ethernet physical interface over copper | Encoded electrical signals representing the frame bits |

**One concrete example:** a client at `192.0.2.10:51514` requests `/hello` from a server at
`198.51.100.20:80`, through one router. These are documentation addresses. DNS, next-hop MAC
resolution (ARP), and the TCP handshake have already finished. We use **plain HTTP**, two Ethernet
links, and no NAT so the envelopes remain visible.

The redraw below follows the five-layer and encapsulation teaching structure in Romit Roy
Choudhury's UIUC CS/ECE 438
[overview slides, PDF pages 58–59](https://courses.grainger.illinois.edu/cs438/fa2020/slides/1-overview-annotated.pdf#page=58).
The concrete request, header fields, and byte counts are added here; this is not a screenshot of the slides.

<!-- network-diagram: packet-path -->

**Walk through the five layers:**

1. **Application — create the message.** The HTTP client creates
   `GET /hello HTTP/1.1\r\nHost: example.test\r\n\r\n`. Here `\r\n` represents two bytes
   (carriage return and line feed), not four printable characters. The complete request is **43 ASCII
   bytes**. It contains an HTTP request line and HTTP headers; those are all *payload* from TCP's
   perspective. The client writes these bytes to a socket.

2. **Transport — TCP adds connection and delivery information.** The example segment has source
   port `51514`, destination port `80`, and sequence number `1001`. The ports, endpoint IPs, and
   transport protocol identify the connection. TCP also carries acknowledgment, flags, receive-window,
   and checksum fields. If these 43 bytes arrive contiguously with no earlier gap, the server can
   acknowledge **`1001 + 43 = 1044`**, the next byte it expects. TCP handles loss recovery, ordered
   delivery, flow control, and congestion control; it does not understand `/hello`. A TCP ACK confirms
   byte receipt, **not** that the HTTP request was successfully processed.

3. **Network — IP adds the end-to-end addresses.** The IPv4 packet carries source `192.0.2.10`,
   destination `198.51.100.20`, a TTL, and `protocol=6` to identify TCP as its payload. A router uses
   the destination IP to select a next hop. IP itself does not promise delivery, ordering, or
   retransmission; those TCP behaviors are implemented at the endpoints.

4. **Link — Ethernet adds the local-hop addresses.** On the first link, the destination MAC is the
   **router's incoming interface**, not the remote server. The frame also carries `EtherType=0x0800`
   to identify IPv4 and ends with a four-byte FCS error check. At the router, that link envelope ends:
   the outgoing frame uses **router-out MAC → server MAC**. In this no-NAT example, the IP endpoints
   and TCP segment stay the same, but TTL changes from `64` to `63`, the IPv4 header checksum is
   updated, and a new Ethernet header/FCS is generated.

5. **Physical — send a signal, not a printed string of 0s and 1s.** The interface encodes the frame's
   bits into electrical signaling on the copper link. The next interface recovers bits from that
   signal. The router transmits a new signal on its outgoing link. Different physical technologies
   encode bits differently; physical transmission does not add a universal “physical header.”

**At the receiver, run the process upward:** recover bits → validate and remove the Ethernet
envelope → process the IPv4 header → let TCP validate, order, and deliver stream bytes → let the HTTP
server parse the request. TCP may deliver the request through several socket reads: **one HTTP
message, one TCP segment, and one `read()` are not generally the same unit**. Our one-segment request
is deliberately small. See [TCP's specification](https://www.rfc-editor.org/rfc/rfc9293.html) for
byte-stream semantics and [IPv4 router forwarding](https://www.rfc-editor.org/rfc/rfc1812.html#section-5.2.1)
for the per-hop processing rules.

**Why the byte counts grow:** with 20-byte TCP and IPv4 headers and no options, the segment is
`20 + 43 = 63 B`, the IP packet is `20 + 63 = 83 B`, and the Ethernet frame is
`14 + 83 + 4 = 101 B`. The last count includes FCS but excludes preamble/SFD and the inter-frame gap;
it is not the full physical-link time cost. No Ethernet padding is needed for this example.

> **HTTPS caveat:** the example is plain HTTP so you can see the bytes. With HTTPS over HTTP/1.1 or
> HTTP/2, TLS protects the HTTP content before TCP carries it, so TCP payload is TLS record bytes,
> not readable `GET` text. HTTP/3 uses QUIC over UDP instead. Encapsulation still applies, but the
> protocol stack and byte counts differ.

**Names, addresses, and endpoints:**

- A domain name such as `video.example.com` is stable application-facing identity.
- DNS resolves that name to one or more addresses. A cached record normally expires after its TTL;
  [RFC 1034](https://www.rfc-editor.org/rfc/rfc1034.html) defines this cache lifetime model.
- An IP address locates an interface or service route at the network layer.
- A port identifies a transport endpoint on a host.
- A socket is the operating-system object an application reads from and writes to.
- A flow is commonly identified by protocol plus source/destination address and port (the five-tuple).

> **Interview pitfall:** DNS does not “connect to the server,” and an IP address does not identify a
> process. DNS supplies candidate addresses; transport establishment and ports come afterward.

## 1.2 Latency, bandwidth, throughput, jitter, and loss

These quantities are related but not interchangeable:

| Term | Meaning | Typical measurement |
|------|---------|---------------------|
| **One-way delay** | send time to receive time | requires synchronized clocks |
| **RTT** | request/packet out plus acknowledgment back | ping, transport telemetry |
| **Bandwidth/capacity** | maximum rate a path can sustain | bottleneck property |
| **Throughput** | bytes delivered at a stated measurement layer per unit time | response or transport bytes / time |
| **Goodput** | unique useful application payload rate, excluding headers/retries | compressed media payload bytes / time |
| **Jitter** | variation in packet arrival spacing or delay | RTP/RTCP or arrival trace |
| **Loss** | sent data not delivered before the observer gives up | sequence gaps, retransmission data |

Propagation delay is bounded by distance and medium. Transmission delay is packet bits divided by
link rate. Queueing delay appears when arrivals temporarily exceed service rate. Processing delay
comes from hosts and network devices. A useful decomposition is:

```
end-to-end delay = propagation + transmission + queueing + processing
```

**Worked example:** a 1,500-byte packet is 12,000 bits. Serialization takes `12,000 / 10 Mb/s =
1.2 ms` on a 10 Mb/s bottleneck, but only `0.012 ms` on a 1 Gb/s link. If the path also has 30 ms
of propagation and 40 ms of queueing, upgrading the gigabit access link cannot remove the other
70 ms.

> **Interview pitfall:** a speed test's “100 Mb/s” is a rate, not a latency. A high-bandwidth path can
> still have a large RTT, and a low-RTT path can still be too narrow for the selected rendition.

## 1.3 Queues, buffers, and the bandwidth-delay product

A sender can have data “in flight” until acknowledgments return. The amount required to fill a path
is approximately its **bandwidth-delay product**:

```
BDP (bits) = bottleneck bandwidth (bits/s) × RTT (s)
```

**Worked example:** on a `20 Mb/s`, `80 ms` path, `BDP = 20,000,000 × 0.08 = 1,600,000 bits =
200 KB`. A reliable sender limited to a 50 KB receive/congestion window can use at most about
`50 KB / 0.08 s = 625 KB/s ≈ 5 Mb/s`, even though the path could carry 20 Mb/s.

Buffers absorb bursts, but oversized unmanaged queues create **bufferbloat**: packets are not lost,
yet RTT rises by hundreds of milliseconds. That may look acceptable to a bulk download and terrible
to a call. Streaming therefore needs both rate and delay telemetry.

## 1.4 MTU, fragmentation, and packetization

Each link has a maximum transmission unit (MTU). If an IP packet is too large for the path, it may be
fragmented where supported or dropped so the sender can learn a smaller path MTU. Losing one IP
fragment loses the whole original packet, so media transports normally packetize below the path MTU.
[RFC 8085](https://www.rfc-editor.org/rfc/rfc8085.html) explicitly advises UDP applications to avoid
IP fragmentation because it reduces reliability and efficiency.

**Worked example:** with a 1,200-byte safe UDP payload budget and a 48 KB encoded frame, the sender
needs roughly `48,000 / 1,200 = 40` packets before transport extensions. If independent packet-loss
probability were 0.5%, the chance that at least one of those 40 packets is lost would be
`1 − 0.995^40 ≈ 18.2%`. Frame assembly and codec recovery determine whether that becomes visible.

> **Interview pitfall:** “one frame equals one packet” is usually false. A video frame is commonly packetized across many datagrams, and one datagram may also contain only part of a codec unit.

---

# 2. UDP, TCP, and QUIC

Transport protocols answer different application needs. The handshake diagram first shows what TCP
must establish before ordinary reliable byte delivery begins.

<!-- network-diagram: tcp-handshake -->

The next comparison follows one lost packet. TCP hides the gap from the application by retransmitting
and preserving byte order; UDP exposes whatever datagrams arrive; QUIC reliably orders bytes within
each stream while allowing unrelated streams to keep progressing.

<!-- network-diagram: tcp-udp-loss -->

These are abstractions, not statements that one protocol is always faster. The application workload,
implementation, path, connection reuse, and deadline determine the outcome.

## 2.1 UDP: messages with minimal transport machinery

UDP preserves message boundaries: one send creates a datagram, and a receive obtains one datagram
(or a truncated one if the supplied buffer is too small). UDP itself does not establish a connection,
retransmit, order, pace, or estimate congestion.

That makes UDP a useful substrate when an application needs custom deadlines or recovery:

- late real-time video may be less useful than a fresh frame;
- the application can add sequence numbers, timestamps, FEC, NACK, and pacing selectively;
- multicast is possible where the network supports it;
- a single lost datagram need not stall later datagrams.

But “UDP is unreliable” does **not** mean “send without restraint.”
[RFC 8085](https://www.rfc-editor.org/rfc/rfc8085.html) requires UDP applications to provide suitable
congestion control for Internet use and to consider message size, checksums, middleboxes, and path
behavior.

> **Interview pitfall:** UDP does not guarantee lower end-to-end latency. An unpaced UDP sender can
> fill queues, cause its own loss, and make latency worse than a congestion-controlled transport.

## 2.2 TCP: a reliable ordered byte stream

TCP presents an ordered byte stream, not application messages. It assigns sequence numbers, uses
cumulative acknowledgments, retransmits missing data, rejects corrupt segments, and regulates how
much data is in flight. [RFC 9293](https://www.rfc-editor.org/rfc/rfc9293.html) is the current core
TCP specification.

The three-way handshake synchronizes both peers' initial sequence numbers:

```
Client                                         Server
  | --- SYN, seq=x --------------------------> |
  | <--- SYN+ACK, seq=y, ack=x+1 ------------- |
  | --- ACK, ack=y+1 ------------------------> |
  |             connection established         |
```

The client can then write `5 KB`, while TCP may transmit that as several segments and the receiver
may return it through one or many reads. Applications therefore need framing such as HTTP message
syntax or an explicit length prefix.


TCP has two independent windows:

- **Receive window (`rwnd`)** protects the receiver from an application that consumes bytes slowly.
- **Congestion window (`cwnd`)** protects the network by limiting unacknowledged data.
- Usable flight size is approximately `min(rwnd, cwnd)`.

When a segment is missing, later bytes can arrive at the host but are normally not delivered past the
gap to the application. This transport-level head-of-line blocking is useful for exact files but can
turn one loss into a deadline miss for live media.

> **Interview pitfall:** TCP does not send “packets in order.” IP packets may reorder. TCP reconstructs
> an ordered byte stream before exposing it to the application.

## 2.3 TCP startup, recovery, and congestion control

A new connection does not instantly transmit at path capacity. Its congestion window begins finite
and grows as acknowledgments demonstrate delivery. Loss or explicit congestion signals cause the
sender to reduce flight. Modern stacks use retransmission timers and acknowledgment patterns to infer
missing data; the exact congestion-control algorithm is an implementation choice, not fixed by
RFC 9293.

For segmented video, reusing a warm connection avoids repeated handshakes and startup behavior;
long idle gaps can make a later burst behave differently from a continuous transfer.

**Worked example:** a 2 MB segment delivered in 1.6 s has measured application throughput of
`2 × 8 / 1.6 = 10 Mb/s`. That estimate includes all startup, loss recovery, and sharing that occurred;
it is not proof that bottleneck capacity equals 10 Mb/s.

## 2.4 QUIC: secure multiplexed streams over UDP

QUIC builds reliable streams, connection migration, loss recovery, congestion control, and TLS-based
security in user space over UDP. QUIC version 1 integrates TLS 1.3 and multiplexes streams inside one
connection; see [RFC 9000](https://www.rfc-editor.org/rfc/rfc9000.html).

The important ordering boundary is the **stream**:

- bytes within one stream are delivered reliably and in order;
- loss affecting stream A need not stop application progress on stream B;
- packets and congestion control are still shared at the connection/path level;
- a lost packet can therefore reduce sending rate for the whole connection even when stream delivery
  is independent.

QUIC can resume a known server quickly and can support 0-RTT application data. However, early data can
be replayed; [TLS 1.3 §8](https://www.rfc-editor.org/rfc/rfc8446.html#section-8) requires applications
to use it only for operations safe under replay.

> **Interview pitfall:** QUIC does not make packet loss free and is not “unreliable UDP.” It implements
> reliable streams and congestion control above UDP, while avoiding cross-stream delivery blocking.

---

# 3. HTTP and the Request Path

A video segment request is the visible end of several dependent systems. The diagram separates name
resolution, route/edge selection, connection security, cache lookup, and origin work so failures can
be localized rather than called “the network.”

<!-- network-diagram: http-path -->

A cache hit can end the request at the edge. A miss may traverse an origin shield and object store;
those extra stages affect time to first byte even when the client-to-edge RTT is excellent.

## 3.1 One HTTPS request, step by step

For a cold `https://video.example.com/seg-104.m4s` request:

1. The client checks local and recursive DNS caches, then resolves the host if needed.
2. Routing takes packets toward a selected address, often an anycast or DNS-selected CDN edge.
3. The client establishes TCP then TLS, or establishes QUIC with integrated TLS.
4. It sends an HTTP request with method, target, and fields such as `Host`/`:authority` and `Range`.
5. The edge constructs a cache key and checks freshness.
6. On a hit, the edge returns stored bytes; on a miss, it forwards upstream and may store the result.
7. The player appends media bytes, demuxes, decodes, and advances its buffer.

[TLS 1.3](https://www.rfc-editor.org/rfc/rfc8446.html) provides peer authentication, confidentiality,
and integrity after its handshake. It does not conceal all metadata, packet sizes, or timing.

## 3.2 HTTP/1.1, HTTP/2, and HTTP/3

HTTP defines request/response semantics; its versions map those semantics to transports differently:

| Version | Transport shape | Loss consequence |
|---------|-----------------|------------------|
| HTTP/1.1 | usually one in-flight request per connection unless pipelined | TCP gap blocks that connection |
| HTTP/2 | many HTTP streams multiplexed over one TCP connection | TCP gap temporarily blocks all streams' bytes |
| HTTP/3 | HTTP streams mapped to QUIC streams | one blocked stream need not block other streams |

[RFC 9114](https://www.rfc-editor.org/rfc/rfc9114.html) defines HTTP/3 over QUIC, including HEADERS and
DATA frames and QPACK field compression. HTTP/3 improves cross-stream independence; it does not erase
RTT, server time, radio loss, or application buffering.

**Worked example:** suppose requests A and B share a connection and a packet carrying A is lost. With
HTTP/2 over TCP, bytes for B behind the TCP gap wait for recovery. With HTTP/3, B can be delivered if
its QUIC stream's data arrived, although both streams may slow because congestion control observed loss.

## 3.3 Cache semantics for media

HTTP caching reuses a stored response only when the cache key and reuse conditions match. Freshness,
validation, `Age`, `Cache-Control`, `ETag`, and `Vary` are protocol semantics, not merely CDN settings;
see [RFC 9111](https://www.rfc-editor.org/rfc/rfc9111.html).

Practical media policy:

- Immutable VOD segments can use content-versioned URLs and long freshness.
- A live manifest changes frequently and needs short freshness or carefully controlled revalidation.
- Completed live segments can often become immutable even while the playlist remains dynamic.
- Signed query parameters can fragment the cache unless the CDN deliberately normalizes its key.
- `Range` requests help seeking and partial retrieval, but cache support for partial objects must be
  tested rather than assumed.
- CORS controls whether browser JavaScript may read a cross-origin response; it does not make a video
  object cacheable or encrypt it.

> **Interview pitfall:** “a CDN caches GETs” is incomplete. Method, status, freshness, authorization,
> `Vary`, range behavior, query handling, and product configuration all affect reuse.

## 3.4 A cold-start latency budget

If DNS costs 25 ms, client-edge RTT is 40 ms, new TCP and TLS 1.3 establishment each cost roughly one
RTT, and a sequential HTTP request/first-byte exchange costs one RTT plus 5 ms of edge work, the teaching estimate is
`25 + 40 + 40 + 40 + 5 = 150 ms`. Operations may overlap; resumption changes the cost; HTTP/3 combines
transport and cryptographic establishment; and an origin miss adds another path. Draw dependencies
before summing numbers.

---

# 4. HLS, DASH, and CMAF

## 4.1 HLS, DASH, and CMAF

HLS and MPEG-DASH are **segment-based adaptive media delivery systems commonly carried over HTTP**.
The client fetches a manifest, then requests media objects while choosing among aligned representations.

| | HLS | MPEG-DASH |
|---|-----|-----------|
| Manifest | `.m3u8`, line-oriented text | `.mpd`, XML |
| Organization | multivariant/master + media playlists | Period → AdaptationSet → Representation |
| Common media | MPEG-2 TS or fragmented MP4 | commonly fragmented MP4; other profiles exist |
| Standards home | Apple/IETF publication | ISO/IEC, with DASH-IF interoperability guidance |
| Player decision | client chooses variant/rendition | client chooses representation |

[RFC 8216](https://www.rfc-editor.org/rfc/rfc8216.html) describes HLS playlists, segments, variant
streams, and client responsibilities. DASH is broader and profile-driven; the current
[DASH-IF IOP v5](https://dashif.org/guidelines/iop-v5/) constrains DASH delivery to interoperable CMAF
media profiles and separates guidance for on-demand, live, and low-latency live services.

**CMAF** defines interoperable fragmented MP4 media structures. It can let compatible HLS and DASH
presentations reference the same encoded media objects. It does **not** automatically make any HLS
ladder interchangeable with any DASH ladder: codec, encryption, segment boundaries, timestamps, and
profile constraints must align.

> **Correction to a common answer:** HLS does not inherently mean FairPlay, and DASH does not
> inherently mean Widevine or PlayReady. Manifest ecosystems often pair with those DRMs, but delivery
> format, encryption scheme, key signaling, and license system are separate design dimensions.

## 4.2 Manifests, segments, and switching

The diagram follows a DASH-style flow: an MPD advertises representations, the player requests an
initialization object and media segments, measures downloads, then chooses the next representation.
The HLS flow is analogous even though playlist syntax and terminology differ.

<!-- network-diagram: dash-flow -->

For safe switching, representations in an adaptation set need aligned timing and compatible random
access points. A player cannot decode arbitrary bytes from a higher rung merely because the URL exists;
it needs the correct initialization state and a decodable boundary.

**Small HLS media-playlist example:**

```m3u8
#EXTM3U
#EXT-X-VERSION:6
#EXT-X-TARGETDURATION:4
#EXT-X-MEDIA-SEQUENCE:104
#EXT-X-MAP:URI="init.mp4"
#EXTINF:4.000,
seg-104.m4s
#EXTINF:4.000,
seg-105.m4s
```

`BANDWIDTH` is a declared requirement used for selection; actual segment size can vary. The player's
measured sample is `downloaded bits / transfer time`, so variable-size chunks create noisy estimates.

**Segment-duration trade-off:**

- longer segments reduce request overhead and permit longer compression structures;
- shorter segments expose new choices sooner and can reduce live latency;
- very short independently decodable segments may require more frequent random-access frames;
- segment duration alone does not set latency: hold-back, encoder pipeline, publication timing,
  network transfer, player buffer, and decoder scheduling all contribute.

> **Interview pitfall:** “the player switches resolution when bandwidth changes” skips the hard parts:
> estimate future delivery, protect the buffer, switch only at valid aligned boundaries, and avoid
> oscillation when measurements hover near a rung.

## 4.3 Low-latency streaming

Classic live HTTP playback often waits for complete segments and keeps several segments of safety.
Low-latency designs publish smaller independently addressable or progressively available units so the
player can stay closer to the live edge.

The diagram shows why latency is a pipeline, not simply `segment duration × N`. Media must be captured,
encoded, packaged, published, transferred, buffered, decoded, and rendered.

<!-- network-diagram: low-latency -->

| Approach | Typical product order | Main mechanism |
|----------|-----------------------|----------------|
| conventional HLS/DASH | seconds to tens of seconds | complete segments plus safety buffer |
| tuned short-segment HLS/DASH | several seconds | shorter segments and smaller hold-back |
| LL-HLS | low single-digit seconds when well operated | partial segments, blocking reload, preload hints, delta updates |
| low-latency DASH | low single-digit seconds when well operated | CMAF chunks available before a full segment completes |
| WebRTC | sub-second is a common goal | deadline-driven RTP media and a small jitter buffer |
| MOQT (work in progress) | designed for low latency at scale | publish/subscribe objects over QUIC/WebTransport and relays |

Apple's [Low-Latency HLS documentation](https://developer.apple.com/documentation/http-live-streaming/enabling-low-latency-http-live-streaming-hls)
defines Partial Segments, blocking playlist reloads, preload hints, rendition reports, and delta
updates. These features reduce polling and publication delay; they do not guarantee one fixed latency.

For low-latency DASH, CMAF chunks can be made available progressively; HTTP/2 and HTTP/3 do not use HTTP/1.1 “chunked transfer encoding.” The [DASH-IF live guideline](https://dashif.org/guidelines/iop-v5/) is the interoperability reference. The IETF's active [Media over QUIC work](https://datatracker.ietf.org/doc/draft-ietf-moq-transport/) targets scalable low-latency publication through relays, but remains an Internet-Draft rather than an RFC.

**The fundamental tension:** a smaller buffer leaves less time to absorb throughput variation. On a
variable path, lowering latency without changing anything else raises rebuffer risk or forces more
conservative quality. The product chooses a point among latency, quality, continuity, and cost.

---

# 5. Adaptive Bitrate and QoE

## 5.1 ABR algorithms

The client picks a bitrate per segment. Four families:

**1. Rate-based (throughput prediction).** Estimate future throughput from past segment download rates (a harmonic mean is common and dampens optimistic high samples, though one low sample pulls it down sharply), then pick the highest bitrate below the estimate × a safety factor.
- Fails on: throughput estimation bias (a low-bitrate segment downloads fast and *underestimates* capacity), and on/off cycling interacting with TCP.

**2. Buffer-based.** Make buffer occupancy the primary signal and map buffer → bitrate.
- **BBA** (Huang et al.): a piecewise-linear reservoir/cushion map. Buffer level directly measures playback safety; the published deployment also used capacity information during startup.
- **BOLA** (Spiteri et al.): formulates utility maximization as Lyapunov optimization, giving a principled buffer→bitrate rule with provable bounds; an updated form was incorporated into dash.js.

**3. Control-theoretic / hybrid.**
- **MPC / RobustMPC** (Yin et al.): at each step, solve a QoE maximization over a horizon of the next ~5 chunks, using a throughput prediction and the buffer model, then apply the first decision (receding horizon). RobustMPC uses a conservative throughput estimate to hedge prediction error.
- Strong research results; cost is a short-horizon optimization for each choice, often made tractable by the small discrete ladder.

**4. Learning-based.**
- **Pensieve** (Mao et al., SIGCOMM '17): A3C reinforcement learning. State = past throughput measurements, download times, buffer level, remaining chunks, chunk sizes; action = next bitrate; reward = the QoE function. Learns a policy without a hand-tuned model.
- Later: Comyco (imitation learning from an oracle, much better sample efficiency), Karma (causal sequence modeling / decision-transformer style), Genet (curriculum over network traces).
- **The standard critique** — and you should raise it before they do: a learned policy can fail outside its training-trace distribution and is harder to debug or constrain. Published trace/simulation gains do not by themselves prove a broad production advantage; the real deployment mix is mostly proprietary. Your TMM paper (lower-layer information exposure + online tuning) is a response to that generalization problem — frame it that way.

The original work is worth reading directly: [BBA](https://web.stanford.edu/class/cs244/papers/sigcomm2014-video.pdf),
[BOLA](https://arxiv.org/abs/1601.06748),
[MPC](https://web.stanford.edu/class/cs244/papers/videostreaming-sigcomm2015.pdf), and
[Pensieve](https://web.mit.edu/pensieve/).

**A canonical QoE objective** used to evaluate or tune these approaches is:

```
QoE = Σ q(R_n)  −  μ · Σ T_rebuf,n  −  Σ |q(R_{n+1}) − q(R_n)|
       ↑                ↑                    ↑
   quality of        rebuffering        smoothness
   each chunk        penalty            (switching penalty)
```
`q(·)` maps bitrate to a quality utility—linear, log, or a measured-quality mapping such as VMAF where appropriate. **Choosing `q` and `μ` is the actual product decision**, and interviewers like candidates who notice that the "algorithm" question is downstream of the "what do users care about" question.

The buffer diagram connects the variables: a download adds media seconds, playback drains one second
per second, and a stall occurs when the buffer reaches zero before the next usable segment arrives.

<!-- network-diagram: abr-buffer -->

**Fixed-rate worked example:** a four-second, 2 MB segment is encoded at 4 Mb/s. At 8 Mb/s goodput it takes two seconds
to download, so the buffer gains roughly `4 − 2 = 2 s`; at 2 Mb/s it takes eight seconds, so any starting buffer below eight seconds drains first. An interactive ABR would instead consider a 2 Mb/s rung whose four-second segment is about 1 MB and takes four seconds at 2 Mb/s, before overhead and variance.

#### Coding: throughput + buffer ABR selector `[TODO]`

**Task:** Take the harmonic mean of the latest `K` throughput samples, multiply it by a safety factor, then combine it with a low-buffer guardrail to select the highest ladder rung within budget.  
**Complexity:** `O(K + L)` time for `K` samples and `L` bitrate rungs, with `O(1)` auxiliary space.  
**Pitfalls:** Zero/invalid throughput, units, segment-size bias, hysteresis for upgrades versus fast downgrades, and avoiding oscillation between adjacent rungs.

```python
def select_representation(throughputs, buffer_s, ladder_bps):
    """TODO: harmonic estimate + safety margin + buffer guardrail.

    Tests: sudden bandwidth collapse, gradual recovery, empty history,
    low-buffer emergency downgrade, and no oscillation near a boundary.
    """
    raise NotImplementedError
```

Full implementation: TODO (suggested file: `streaming.py`).

## 5.2 QoE metrics in production

What streaming companies actually watch on dashboards:

- **Play delay / startup time**: request-to-first-render or play-to-first-frame, with an exact event definition.
- **Rebuffer ratio** = rebuffer time / eligible watch time.
- **Rebuffer frequency** and time to first rebuffer.
- **Average delivered bitrate or quality score**, weighted by watched duration.
- **Bitrate/quality switches**, including direction and magnitude.
- **Playback failure rate**, error code, and failed pipeline stage.
- **Video start failure** and exit-before-video-start.
- **Live latency** and drift from the target live edge.

Ordering matters, but no universal metric is always “the most important.” For many entertainment
products, a stall receives a larger QoE penalty than a modest quality reduction; for an interactive
auction, latency may dominate; for offline download, continuity is trivial. Define the product outcome,
then validate metric weights against experiments or observed engagement.

**Worked aggregation:** ten sessions contain 1,000 total watched seconds and 20 rebuffer seconds, so
the global rebuffer ratio is 2%. Averaging per-session ratios can give a different result because a
10-second session and a 300-second session receive equal weight. State which aggregation you use.

> **Interview pitfall:** telemetry is not ground truth if events are dropped when playback fails.
> Monitor event delivery, join keys, clock consistency, and censoring alongside the QoE metric.

---

# 6. Real-Time Media and WebRTC

## 6.1 WebRTC and the RTP stack

WebRTC combines several protocols rather than replacing them with one:

```
Application signaling (your HTTPS/WebSocket design; not standardized by WebRTC)
   ↓ exchanges session descriptions and ICE candidates
ICE + STUN + TURN                 ← finds a viable network path
DTLS                              ← authenticates peers and derives media keys
SRTP / SRTCP                      ← protects RTP media and RTCP control
RTP media + RTCP feedback         ← timing, sequence, reports, repair requests
UDP in the common case            ← other ICE transports/fallbacks may be attempted
```

[RFC 3550](https://www.rfc-editor.org/rfc/rfc3550.html) defines RTP sequence numbers, timestamps,
payload identification, marker bit, SSRCs, and RTCP reports. Sequence numbers reveal gaps/reordering;
video payloads commonly use a 90 kHz media clock and payload-defined marker semantics, not wall time.

Useful RTCP information includes:

- Sender/Receiver Reports: packet counts, loss fields, interarrival jitter, and timing relationships.
- NACK feedback: identifies missing packets worth retransmitting.
- PLI/FIR feedback: asks an encoder for picture refresh when decoder state cannot recover.
- transport-wide arrival feedback in supporting profiles: input to sender bandwidth estimation.

SDP offer/answer negotiates media descriptions, codecs, and extensions. It does not transport media,
discover a NAT path, or prescribe the application's signaling channel.

## 6.2 ICE, STUN, and TURN connection establishment

Browsers often sit behind NATs and firewalls, so a private address is not enough. ICE gathers candidate
addresses, exchanges them through application signaling, tests candidate pairs, and nominates a working
pair. [RFC 8445](https://www.rfc-editor.org/rfc/rfc8445.html) defines ICE's use of STUN and TURN.

The diagram separates discovery from selection. STUN can reveal a server-reflexive mapping; TURN
allocates a relay address when direct connectivity is unavailable or undesirable.

<!-- network-diagram: webrtc-connect -->

Typical candidate types:

- **host**: an address on a local interface (often privacy-obscured in browsers);
- **server reflexive**: a public mapping observed through STUN;
- **peer reflexive**: a mapping discovered during connectivity checks;
- **relay**: an address allocated on a TURN server.

TURN is not “signaling.” It relays media/data and therefore consumes bandwidth, adds a path, and needs
capacity planning. For a visual protocol walkthrough, use the implementer-maintained
[WebRTC for the Curious — Connecting](https://webrtcforthecurious.com/docs/03-connecting/).

> **Interview pitfall:** STUN cannot force every NAT pair to communicate directly. TURN exists because
> some mappings, firewalls, and enterprise policies require relay.

## 6.3 Capture-to-playout latency budget

The diagram is a **capture-to-display** media budget; an interactive input-to-visible-response budget
also includes uplink and application/render response before the captured output frame exists.

<!-- network-diagram: realtime-budget -->

One illustrative path aligned with the diagram is:

```
capture + frame assembly       16 ms
encode + sender work           20 ms
network                        35 ms
jitter buffer                  50 ms
decode + render/display        29 ms
-----------------------------------
capture-to-display total      150 ms
```

**Worked example:** at 30 fps, waiting for the next complete frame can add anywhere from near zero to
33.3 ms, averaging about 16.7 ms if arrival phase is uniform. A measured “encoder time” of 5 ms does
not include that capture wait unless instrumentation explicitly starts before it.

The receiver's jitter buffer is a central trade-off: more queued media absorbs delay variation and
late repair, but every extra millisecond is playout latency. Keep a latency budget per stage and report
percentiles, not only an average.

For motion-to-photon constraints, reprojection, and VR-specific rendering, continue to
[Research & resume §4, MARS](04_resume_drill.md#4-mars--motion-aware-abr-for-real-time-vr).

## 6.4 Loss recovery and error resilience

The correct recovery strategy depends on whether repaired media can arrive before its playout deadline.
The diagram contrasts waiting one RTT for retransmission, spending proactive FEC bytes, and concealing
damage locally.

<!-- network-diagram: recovery -->

| Technique | Main cost | Good fit |
|-----------|-----------|----------|
| **NACK + retransmission** | roughly one RTT plus scheduling | repair deadline comfortably exceeds RTT |
| **FEC** | steady bandwidth and computation | tight deadline or paths where selected losses are recoverable from parity |
| **intra refresh** | distributed codec overhead | limit long error propagation without one huge keyframe |
| **keyframe request (PLI/FIR)** | bitrate/latency spike | decoder state cannot otherwise recover |
| **packet-loss concealment** | local quality degradation | packet is already too late to repair |
| **independent slices/tiles** | compression/packetization overhead | contain damage spatially where codec/profile supports it |

NACK and FEC are not mutually exclusive. A system may protect important packets, retransmit when RTT
permits, and conceal whatever still misses deadline. “20% FEC” also does not guarantee recovery from
every 20% loss pattern; burst structure and code design matter.

Inter prediction can propagate damage when later pictures reference corrupted state. Refresh cadence,
reference structure, decoder error handling, and feedback policy therefore interact.

> **AquaScope connection:** an extremely lossy, high-latency acoustic link can make RTT-based repair
> miss the deadline and heavy channel FEC expensive. A representation whose independently useful units
> degrade gracefully is **source-level resilience**, complementary to channel-level repair.

## 6.5 Congestion control for real-time media

A real-time sender must match the path while keeping queues short. Loss is one congestion signal, but
delay trends or explicit congestion markings can reveal growing queues earlier. The control loop is:

```
packet pacing → path queues/loss → receiver arrival feedback
      ↑                                  |
      └── target rate / probe / backoff ─┘
```

Chromium-derived WebRTC stacks commonly use Google Congestion Control concepts, combining delay and
loss-related signals. The exact estimator has evolved and is not mandated by the WebRTC API. TWCC is
feedback data, not the rate-control algorithm itself; REMB is an older receiver-estimated bitrate signal.
The [WebRTC for the Curious media chapter](https://webrtcforthecurious.com/docs/06-media-communication/)
walks through RTP/RTCP feedback and bandwidth estimation.

Rate adaptation can change encoder bitrate, resolution, frame rate, layer selection, or pacing. These
act on different timescales. A queue can grow during a slow encoder reconfiguration, so pacing and a
small transport queue still matter.

> **Interview pitfall:** “UDP avoids congestion control” is unsafe and incorrect. UDP supplies none,
> so the real-time application must implement it; see [RFC 8085](https://www.rfc-editor.org/rfc/rfc8085.html).

## 6.6 Interactive media pipeline

```
client input → uplink → application/game server → render or compose
  → GPU/media capture → low-delay encode → pace + transmit
  → jitter buffer → decode → render → display/audio playout
```

Design points that generalize beyond one product:

- **Avoid copies:** keep rendered frames in GPU-compatible surfaces through hardware encode when possible.
- **Pipeline work:** encode/transmit available slices or chunks when the codec and transport support it.
- **Bound queues:** stale inputs or frames may be less valuable than fresh state.
- **Coordinate controllers:** application simulation, encoder rate control, and network congestion
  control should not independently build hidden queues.
- **Choose recovery by deadline:** late retransmissions consume capacity without improving the frame.
- **Measure interaction:** input-to-visible-response and frame-time variance can matter more than average bitrate.

Cloud gaming often disables B-frame reordering and long lookahead, uses a small rate-control buffer,
and relies on hardware encode. Those are product choices for latency, not universal WebRTC requirements.
VR adds head-pose prediction and reprojection; those specialized constraints remain in
[Research & resume §5, Trinity](04_resume_drill.md#5-trinity--qoe-for-cloud-vr-gaming).

---

# 7. Delivery at Scale

## 7.1 CDN and scale

A CDN places reusable content near viewers and protects a smaller origin tier. The diagram follows a
request through edge, shield/mid-tier, and origin, then shows how the response fills caches on its way
back.

<!-- network-diagram: cdn-flow -->

- **Cache hierarchy:** edge → shield/mid-tier → origin. Each layer trades storage for fewer upstream bytes.
- **Cache hit ratio:** requests or bytes served without reaching the next tier; byte hit ratio is often
  more informative when object sizes vary greatly.
- **Origin shielding:** many edges consolidate misses through fewer upstream caches/connections.
- **Request collapsing:** compatible concurrent misses can share one upstream fetch; HTTP caching permits
  this pattern under the reuse rules in [RFC 9111](https://www.rfc-editor.org/rfc/rfc9111.html).
- **Long-tail catalog:** popular objects produce high reuse; rare objects create storage and fill trade-offs.
- **Live:** completed segments are requested by many viewers almost together, which helps reuse but creates
  a synchronized miss burst when publication or cache keys are wrong.

Netflix Open Connect is a purpose-built example, not a generic CDN template. Netflix documents embedded
Open Connect Appliances plus fill/update paths in its
[Open Connect overview](https://openconnect.netflix.com/Open-Connect-Overview.pdf). Catalog predictability
allows proactive placement; user-generated live content has different demand and freshness behavior.

## 7.2 Cache keys, manifests, and origin load

Suppose one million viewers request the same 4 MB segment. With perfect edge reuse, the upstream tier
may fetch roughly one copy per edge rather than `4 MB × 1,000,000 = 4 TB`. But accidental cache-key
variation—unique query strings, cookies, headers selected by `Vary`, or inconsistent hostnames—can turn
that object into many misses.

Manifest and segment policies should differ:

- a live manifest is small and mutable, so freshness is short and request rate is high;
- a uniquely named completed segment is large and immutable, so freshness can be long;
- a VOD manifest may be versioned and long-lived unless personalization changes it;
- personalized ad manifests need intentional cache partitioning and privacy review.

> **Interview pitfall:** cache hit ratio alone can look healthy while expensive large objects miss.
> Track request hit ratio, byte hit ratio, origin egress, shield effectiveness, and status/latency by tier.

## 7.3 Multi-CDN and live-event resilience

Multi-CDN routing can improve reach and failure isolation, but only if switching uses independent health
signals and content is consistently available. DNS steering acts at DNS-cache timescales; application
or manifest steering can react differently; existing connections do not necessarily move when DNS changes.

Plan capacity on every failover path; keep object naming, TLS, CORS, range, and cache behavior
consistent; protect the origin from cold failover; bound player retries; label telemetry by delivery
path; and test traffic draining before the event.

**Worked failure:** if CDN A carries 70% of a 1 Tb/s event and fails, B and C must absorb 700 Gb/s in
addition to their existing 300 Gb/s. Having two configured vendors is not resilience if spare capacity,
cache warmth, or origin capacity cannot survive that step.

---

# 8. Debugging Streaming Systems

## 8.1 Debug by layer and by timeline

Start with one affected session and establish **what worked last**:

1. Did DNS return usable addresses?
2. Did transport and TLS establish? Which protocol/version and edge?
3. Did the manifest request succeed, and was the content current?
4. Did media requests succeed with expected status, size, range, and timing?
5. Could the demuxer parse initialization and media timestamps?
6. Did the decoder accept frames?
7. Did the player buffer/render, or deliberately wait/drop?
8. Did telemetry itself arrive?

Build one timeline from monotonic timestamps. Wall clocks across devices can be skewed, while durations
measured on one host's monotonic clock remain meaningful.

## 8.2 Minimum useful measurements

For each request capture:

- URL class (manifest, init, audio, video, key/license), with sensitive tokens redacted;
- DNS, connect, TLS, request queue, TTFB, transfer time, bytes, status, redirects, and protocol;
- cache status, `Age`, range/content length, CDN/edge identity, and retry number;
- representation, media sequence/segment number, decode and presentation timestamps;
- buffer before/after, selected quality, estimated throughput, and switch reason.

For WebRTC capture:

- selected ICE candidate pair and whether TURN is used;
- RTT, available outgoing bitrate estimate, send/receive rate, packet loss, jitter, and NACK/FEC counts;
- sender queue/pacer delay, jitter-buffer delay, frames encoded/decoded/dropped, QP/quality proxy;
- PLI/FIR count, freezes, resolution/frame-rate changes, and audio concealment.

Tools answer different layers: `dig`/`nslookup` for DNS, `curl -v` for HTTP/TLS timing and headers,
browser network panels for player requests, `ffprobe` for container/timestamps, packet capture for wire
events where permitted, and `webrtc-internals`/`getStats()` for WebRTC. No single tool is the truth for
the entire pipeline.

## 8.3 Symptom → hypothesis, not symptom → verdict

| Symptom | Plausible hypotheses | Discriminating evidence |
|---------|----------------------|-------------------------|
| high startup time | cold DNS/TLS, origin miss, slow first segment, decoder wait | stage timing + cache status + first-frame event |
| periodic stalls | rung too high, radio cycle, segment outlier, timestamp gap | buffer trace + bytes/time + media timeline |
| low quality, no stalls | conservative ABR, bad throughput samples, weak ladder | selection reason + capacity + encoded quality |
| live latency drifts upward | downloads slightly slower than playback, catch-up disabled | live-edge distance + buffer slope |
| call freezes after network switch | stale ICE path, failed restart, TURN unreachable | candidate-pair changes + ICE states |
| audio works, video fails | codec/profile mismatch, bandwidth, decoder, video SSRC/path | negotiation + inbound RTP + decoder counters |
| huge RTT, little loss | bufferbloat or oversized sender queue | queue/pacer delay + RTT vs baseline |
| CDN 5xx burst | origin overload, bad deployment, shield failure | cache status + tier logs + object distribution |

Say “hypothesis” until a measurement rules alternatives in or out. Correlation—for example, stalls and
bitrate increases occurring together—does not prove the switch caused the stall; both may follow a bad
capacity estimate.

---

# 9. Rapid-Fire Drill — Networking

Cover the answer and give a crisp 30-second version before expanding.

**Basic networking**

1. Walk one HTTPS request through DNS, routing, transport, TLS, HTTP, cache, and origin.
2. Bandwidth vs throughput vs goodput vs latency: define each and give a counterexample.
3. What creates queueing delay, and why can “no packet loss” still feel bad?
4. Compute the BDP of a 50 Mb/s, 100 ms path. Why does the sender window matter?
5. Why should a UDP media sender avoid IP fragmentation?
6. What does a port identify? What does a five-tuple identify?
7. Explain TCP's three-way handshake and why an ACK does not consume byte sequence space.
8. Receive flow control vs congestion control: whom does each protect?
9. What is TCP head-of-line blocking? What exactly does QUIC improve?
10. Why is “UDP is faster” not a valid transport decision?
11. What security risk makes TLS/QUIC 0-RTT unsuitable for arbitrary writes?
12. On a cache miss, which additional stages can increase TTFB?

**Streaming and delivery — migrated from the Video drill**

13. HLS vs DASH vs CMAF; why did CMAF win?
14. How does LL-HLS get to 2 s?
15. Walk the WebRTC stack; what does RTCP feed back?
16. NACK vs FEC — how do you decide?
17. Why can loss-only congestion control build delay, and how do delay and loss signals complement each other?
18. Compare rate-based / buffer-based / MPC / RL ABR. Why hasn't RL taken over?
19. Write the standard QoE objective and explain each term.
20. Which QoE metric matters most in production, and why?

21. Why is a segment 4 s and not 0.5 s or 30 s?

**Interactive media and operations — new**

22. Build a generic capture-to-playout budget. Which queues are hidden most often?
23. STUN vs TURN vs ICE: what problem does each solve?
24. Why can a TURN relay be necessary, and what must you capacity-plan?
25. Why can a live segment cache well while the live manifest must stay fresh?
26. Request hit ratio vs byte hit ratio: when do they tell different stories?
27. Why might DNS failover not move an already-playing client immediately?
28. A player's live latency grows with no hard stalls. What traces do you inspect first?

---

# 10. Coding Reverse Index — Networking

Coding prompts live beside the concepts they exercise. `[TODO]` is an interface and test contract; it
does not claim an implementation exists.

| Priority | Practice task | Review topic | Status / implementation |
|---|---|---|---|
| P2 | Throughput + buffer ABR | [§5.1](#51-abr-algorithms) | `[TODO]` `streaming.py` |

---

*Return to [01_video_domain.md](01_video_domain.md), continue to [02_ml_domain.md](02_ml_domain.md), or connect the two in [02_ml_domain.md#6-generative-models--learned-compression](02_ml_domain.md#6-generative-models--learned-compression).*
