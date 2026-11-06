# Interview Prep — Video, Networking & ML Domain Knowledge

> **Interactive reading version:** [Open the study homepage](index.html) · [Video](01_video_domain.html) · [Networking & Streaming](05_networking_streaming.html) · [ML & LLMs](02_ml_domain.html) · [Video × ML](02_ml_domain.html#6-generative-models--learned-compression) · [Research](04_resume_drill.html) · [LeetCode Notes](06_leetcode_notes.html) · [Company Interviews](08_company_interviews.html).
> Open `index.html` directly in a browser to search offline, collapse sections, copy code, and save reading progress.
> After editing Markdown, run `npm run build` to update the HTML. Run `npm install` before the first build. Browser check: `npm run check` (uses the locally installed Chrome).

**Current interview focus:** Nuro Senior Software Engineer, Video Streaming. Start with
[System Design](07_system_design.html) for complete Nuro-topic design walkthroughs.

Prep hub for **Lingzhi Zhao** (UIUC CS PhD, video systems + generative/multimodal models).

**Target**: PhD-level research / applied-scientist / research-engineer roles at the intersection of
**video systems** and **multimodal ML** — big tech, mid-size video infra, and well-funded labs alike.
See [§ Role archetypes](#role-archetypes--where-they-live) for the actual landscape.

---

## The files

| File | What's in it | When to read |
|------|-------------|--------------|
| **[01_video_domain.md](01_video_domain.md)** | **3 chapters**: color → image codecs → video codecs, with rate control and the VBV model → **15 inline Video coding drills** | Video domain round |
| **[02_ml_domain.md](02_ml_domain.md)** | NN & math → **Transformer/LLM interview Q&A** → **Transformer LM** (incl. GPT-2) → **VLM** → LoRA post-training → inference → scaling/GPU → precision/parallelism → inline coding drills | ML breadth/depth round |
| **[02_ml_domain.md#6-generative-models--learned-compression](02_ml_domain.md#6-generative-models--learned-compression)** | **Generative models & compression**: autoencoders and VAEs → VQ-VAE, VQGAN, FSQ, LFQ, RVQ, TiTok → GANs and diffusion → learned compression → the Cosmos tokenizer, each with notation and loss → inline coding drills | GenStream and AquaScope deep dives |
| **[04_resume_drill.md](04_resume_drill.md)** | Per-project attack list: what they'll ask about each paper, the answer to lead with, the weakness to volunteer | Job talk + paper deep-dive prep |
| **[05_networking_streaming.md](05_networking_streaming.md)** | **10 chapters**: network basics → TCP/UDP/QUIC → HTTP streaming → ABR → WebRTC/real-time media → CDN/operations → **1 inline coding drill** | Networking, streaming, and RTC round |
| **[06_leetcode_notes.md](06_leetcode_notes.md)** | Binary search, doubly linked lists, recursion (tree DFS, backtracking and DFS + memoization in one chapter), and Dijkstra shortest-path templates with representative examples | Algorithm practice and mistake review |
| **[07_system_design.md](07_system_design.md)** | Nuro-reported Webhook and vehicle-metrics topics: basics → components → complete design; optional video reference | System Design round |
| **[08_company_interviews.md](08_company_interviews.md)** | Company interview questions from 1point3acres. **Roblox MLE**: LeetCode (8 tested solutions), ML questions, system design and behavioral questions, each with the candidate's level, the source link and an answer | Before a company's loop |

Primary local practice code: `image_codec.py`, `video_encoding_pipeline.py`,
`intra_prediction_mode.py`, `arithmetic_coding.py`, `rotation.py`, `ml_coding.py`,
`ml_coding_torch.py`, and `learned_compression.py`. Each relevant knowledge point now embeds its
coding prompt, a readable Python snippet, complexity / pitfalls, and a link to the fuller local
implementation. Blocks are labeled `READY`, `PARTIAL`, `TOY`, or `TODO`;
`video_codec_components.py`, `motion_vector.py`, `sampling.py`, and `interview_practice.py` remain
scratch material until the listed issues are fixed.

162 oral self-test questions across the four domain files (Video 41 / Networking 28 / ML 52 / Video×ML 41), plus inline coding drills
under the matching concepts. The end sections are only reverse indexes / backlogs:
[05 §10](05_networking_streaming.md#10-coding-reverse-index--networking),
and [02 §14](02_ml_domain.md#14-coding-reverse-index--ml).

---

# How PhD-level loops actually run

This is **not** the SWE loop. For research scientist / applied scientist / research engineer, the
structure is fairly consistent across companies:

```
Recruiter screen
   ↓
Technical or research phone screen  (30–60 min)
   often blended: a bit of coding + project deep-dive + foundational ML
   ↓
Virtual onsite loop  (4–6 rounds)
   ├─ RESEARCH PRESENTATION / JOB TALK       30–45 min + Q&A   ← often the highest-signal round
   ├─ Paper deep-dive × 2–3                   different interviewers, different subdomains
   ├─ ML breadth                              rapid-fire fundamentals across the field
   ├─ ML depth                                your specialty, pushed to the edge
   ├─ Coding                                  LeetCode-medium, or an ML/domain implementation
   ├─ ML / system design                      "design a video quality system", "design transcoding"
   └─ Behavioral / bar-raiser                 weighted heavily at Amazon, Meta, Netflix
   ↓
Hiring committee → team match
```

**Track distinctions worth knowing**, because they change what's tested:

| Track | Weight |
|-------|--------|
| **Research Scientist** | Job talk + paper depth dominate. Publication record matters. Coding is a floor, not a differentiator. |
| **Research Engineer** | Balanced. Strong coding + can you turn research into a running system. |
| **Applied Scientist** | ML breadth + system design + business framing. Amazon's pillars: ML Breadth, ML Depth, Live Coding, Science Application/Design, Bar Raiser. |
| **ML/Video Engineer** | Coding and systems dominate; research is context. |

**Apply to the track that matches your strength.** Yours is unusually broad — you have both papers
and shipped systems — so you can credibly go for any of the first three. Research Engineer and
Applied Scientist are probably the highest-conversion given the systems work.

## The job talk is the round people under-prepare

30–45 minutes on your own work, to a room of researchers who will interrupt. Standard advice that
actually matters:

- **Pick one project, not five.** Depth beats survey. A talk covering everything signals nothing.
- **Motivation for the first 5 minutes.** Half the room isn't in your subfield. If they don't
  understand why the problem is hard before you show a method, the rest is wasted.
- **Say your limitations out loud.** The single strongest credibility move. If you don't name them,
  someone will find one and it'll look like you didn't know.
- **Expect hostile-sounding questions.** That's engagement, not rejection. Being able to say
  "that's a real weakness, here's what it costs and here's what I'd do" scores higher than a defense.
- **Have a backup slide for every number** on your headline slide.

**Your best job-talk candidates**, in order: **Trinity** (cleanest non-obvious insight, real user
study, systems + perception), **MARS** (same, plus production integration), or the **TikTok VQA work**
if the team is ML-leaning. GenStream/AquaScope if the team is codec/generative-model focused.

## Paper deep-dive rounds

2–3 separate interviewers, deliberately probing **different subdomains** to test whether your depth is
narrow or broad. Prep:

- Read **5–10 recent papers** in your area deeply enough to critique, not summarize. For you: DCVC-FM,
  Cosmos/MAGVIT-v2/TiTok tokenizers, Q-Align + the NTIRE UGC VQA line, BasicVSR++, recent ABR work.
- For each of your own papers: what's the baseline, why is it the right baseline, what breaks it,
  what would you do with 10× the compute, what did reviewers object to.
- Know the **related work you didn't cite** and why.

---

# Role archetypes — where they live

Stop thinking "which five companies" and start thinking "which of these seven jobs." Your resume
qualifies you for at least five of them, and the prep differs.

### A. Video codec / encoding research
Compression efficiency, rate control, encoding pipelines, next-gen standards.
> Netflix (Encoding Technologies) · Amazon Prime Video (encoding team) · Google/YouTube · Meta video
> infra · Apple · Qualcomm (has explicit *AI for video compression* research roles) · Nvidia · Bilibili ·
> Tencent · Kuaishou · **mid-size**: Bitmovin, V-Nova, Visionular, Beamr, Harmonic, Ateme, MainConcept, NETINT

**Prep**: [01](01_video_domain.md) §2–3 cold. BD-rate methodology. Be ready to defend neural codecs
*against* a skeptic — these teams have thought hard about why they don't ship.

### B. Video quality & metrics
Perceptual quality modeling, VQA, encoding decisions driven by quality.
> Netflix (VMAF) · TikTok/ByteDance · Kuaishou · Meta · YouTube · IMAX (ex-SSIMWAVE) · Dolby ·
> academic-adjacent industrial labs

**Prep**: [04](04_resume_drill.md) §6. **This is your single most
transferable project.** Subjective-study design rigor is rarer than you'd think — lean on it.

### C. Real-time communication / cloud rendering
RTC, congestion control, cloud gaming, low-latency streaming.
> Zoom · Meta RTC · Google Meet · Apple · Nvidia (GeForce Now) · Microsoft (xCloud) · Amazon (Luna) ·
> Tencent Start · **mid-size**: Agora, LiveKit, Daily, 100ms, Twilio, Cloudflare Stream

**Prep**: [05](05_networking_streaming.md) §§4–6 for delivery and real-time transport, plus
[04](04_resume_drill.md) §4–5 for VR streaming. MARS and Trinity are directly on-target here.

### D. Multimodal / video foundation-model research
Video understanding, video-language models, embeddings, agents over video.
> Google DeepMind · Meta FAIR/GenAI · Apple AIML · Nvidia · Microsoft Research · Amazon (Prime Video
> GenAI applied science) · ByteDance Seed · Qwen/Alibaba · **startups**: Twelve Labs (Marengo embeddings,
> Pegasus VLM), Hive, Coactive, Clarifai

**Prep**: [02](02_ml_domain.md) §2–5. Your AquaVLM + Qwen3-VL embedding
work fits; expect heavy post-training questioning.

### E. On-device / edge multimodal
Quantization, latency, thermal, memory budgets for models on phones and wearables.
> Apple · Qualcomm · Samsung · Google (Pixel) · Meta (Ray-Ban) · MediaTek · Ambarella · Arm

**Prep**: [02](02_ml_domain.md) §10. **Your iOS 4-bit MobileVLM deployment is exactly this job's
problem statement** — lead with it, and be precise about *how you measured* (sustained load, not a cold run).

### F. Video generation
Text-to-video, video editing, generative tokenizers.
> Runway (~$5.3B) · Luma (~$4B) · Pika · Higgsfield · Kling (Kuaishou) · MiniMax · OpenAI (Sora) ·
> Google (Veo) · Meta (Movie Gen) · Adobe (Firefly) · Nvidia (Cosmos)

**Prep**: [02](02_ml_domain.md#6-generative-models--learned-compression) §6. **GenStream's use of the Cosmos
Tokenizer is a real point of contact here** — generative tokenizers are the shared substrate between
video generation and generative compression.

### G. XR / spatial computing systems
> Meta Reality Labs · Apple (Vision Pro) · Nvidia · Snap AR · Varjo · Niantic

**Prep**: [04](04_resume_drill.md) §4–5. Trinity + MARS + 360LiveCast are a coherent three-paper story
for these teams.

---

# Open roles — job boards & live postings

> **Read this first.** Individual req links go stale in weeks; the **durable search links** below don't.
> Specific postings are dated to when I found them (2026-08-30) and I have *not* individually verified
> each is still open. Treat the specific list as "these teams were hiring for this profile in Aug 2026,"
> which is the more useful signal anyway.
>
> **Timing check**: you defend **Dec 2026**. New-grad programs bucket by graduation year and are strict
> about it — "2027 Start (PhD)" reqs usually require a 2027 graduation date, so those are a *mismatch*
> despite looking like a fit. Filter for **2026 grad / immediate start** or apply to experienced-track
> reqs, where a finishing PhD with your publication record and two industry internships is competitive.

### Durable search links (bookmark these)

| Archetype | Where to search |
|-----------|----------------|
| **All research roles** | [CVPR careers board](https://cvpr.thecvf.com/careers/) — the highest-density source for this exact profile; refreshed continuously |
| A · codec research | [Netflix](https://explore.jobs.netflix.net/careers) · [Qualcomm](https://careers.qualcomm.com/careers) · [Amazon "video encoding"](https://www.amazon.jobs/en/search?base_query=video+encoding) · [Bitmovin](https://bitmovin.com/careers/) · [Dolby](https://jobs.dolby.com/careers) |
| B · quality & metrics | [Netflix](https://explore.jobs.netflix.net/careers) · [ByteDance](https://joinbytedance.com/) · [Dolby](https://jobs.dolby.com/careers) |
| C · RTC / cloud rendering | [LiveKit](https://livekit.com/careers) · [Nvidia](https://jobs.nvidia.com/careers) · [Zoom](https://careers.zoom.us/) · [Agora](https://www.agora.io/en/careers/) |
| D · multimodal / video FM | [Meta Research](https://www.metacareers.com/careerprograms/research/) · [Nvidia](https://jobs.nvidia.com/careers) · [Apple ML&AI: CV](https://jobs.apple.com/en-us/search?team=computer-vision-MLAI-CV) · [Twelve Labs](https://jobs.techstars.com/companies/twelve-labs) · [ByteDance Seed](https://joinbytedance.com/) |
| E · on-device multimodal | [Apple ML&AI (software)](https://jobs.apple.com/en-us/search?team=machine-learning-and-ai-SFTWR-MCHLN) · [Apple ML&AI (hardware)](https://jobs.apple.com/en-us/search?team=machine-learning-and-ai-HRDWR-MCHLN) · [Qualcomm](https://careers.qualcomm.com/careers) |
| F · video generation | [Nvidia](https://jobs.nvidia.com/careers) · [Runway](https://runwayml.com/careers) · [Luma](https://lumalabs.ai/careers) · [Pika](https://pika.art/careers) |
| G · XR / spatial | [Meta Reality Labs](https://www.metacareers.com/jobs) · [Apple](https://jobs.apple.com/en-us/search) · [Nvidia](https://jobs.nvidia.com/careers) |

### Specific postings found (Aug 2026 — verify before applying)

**Strongest single match on this list** — it reads like it was written from your resume:
- 🎯 [**Qualcomm — AI for Video Compression, Research Scientist**](https://careers.qualcomm.com/careers/job/446703677037-artificial-intelligence-for-video-compression-research-scientist-san-diego-california-united-states-of-america) (San Diego, ~$158–238k). Multimedia R&D & Standards group: neural-network-based video compression, standards contributions, PyTorch + C/C++. Neural codecs *and* classical coding *and* AR/VR — that's three of your four areas in one req.
- [Qualcomm — ML for Video Compression, Principal Scientist](https://jobs.anitab.org/companies/qualcomm/jobs/46132284-machine-learning-for-video-compression-principal-scientist) (senior sibling of the above)

**Archetype A/B — codec & quality**
- [Amazon — Research Engineer, Prime Video Encoding Team](https://www.amazon.jobs/en/jobs/10402367/research-engineer-prime-video-encoding-team)
- [Amazon — Applied Scientist, Prime Video GenAI](https://www.amazon.jobs/en/jobs/10441113/applied-scientist-prime-video-generative-ai)
- [Amazon — Sr. Applied Scientist, Prime Video Personalization & Discovery](https://www.amazon.jobs/en/jobs/10406954/sr-applied-scientist-prime-video-personalization-and-discovery-science)
- [Netflix — Video Algorithms Intern, Video Coding (Gaussian Splatting), Fall 2026](https://explore.jobs.netflix.net/careers/job/790315673635-video-algorithms-intern-video-coding-gaussian-splatting-fall-2026-los-gatos-california-united-states-of-america) — *intern req, wrong stage for you*, but it tells you what Encoding Technologies is investing in (GS as a streaming format). Watch their board for the full-time equivalent, and it's a good thing to have read before talking to them.

**Archetype D/F — multimodal & generation**
- [Nvidia — Research Scientist, Generative AI for Physical AI, **PhD New College Grad 2026**](https://jobs.nvidia.com/careers/job/893394590339) — this is the **Cosmos team**. You use their tokenizer in GenStream. Timing bucket likely fits.
- [Nvidia — Research Scientist, Fundamental Generative AI, New College Grad 2026](https://jobs.nvidia.com/careers/job/893393154096)
- [Nvidia — Senior Research Scientist, Generative AI](https://jobs.nvidia.com/careers/job/893392079137)
- [Twelve Labs — ML Research Scientist, Video Embedding Model](https://jobs.techstars.com/companies/twelve-labs/jobs/11851324-ml-research-scientist-video-embedding-model) — maps directly onto your Qwen3-VL embedding work
- [Twelve Labs — Research Scientist/Engineer, Video Cognition System](https://jobs.techstars.com/companies/twelve-labs/jobs/63641995-research-scientist-or-engineer-video-cognition-system)
- [Twelve Labs — Senior ML Research Scientist, Perception Models](https://www.indexventures.com/startup-jobs/twelve-labs/senior-ml-research-scientist-perception-models/)
- [ByteDance — Research Scientist, Generative AI (Intelligent Creation), 2027 Start](https://joinbytedance.com/search/7629855646167091509) ⚠️ *2027 grad requirement*
- [ByteDance — Research Scientist, Multimodal Interaction & World Model, Seed, 2027 Start](https://joinbytedance.com/search/7623545710681016581) ⚠️ *same*

**Archetype E — on-device**
- [Apple — Machine Learning Video Engineer](https://jobs.apple.com/en-us/details/200618357-0836/machine-learning-video-engineer)
- [Apple — ML Engineer, Video Engineering](https://jobs.apple.com/en-us/details/200665577-3956/machine-learning-engineer-video-engineering)
- [Apple — ML Video Processing Engineer](https://jobs.apple.com/en-us/details/200622819-0836/machine-learning-video-processing-engineer)

**Archetype A/C — mid-size**
- [Bitmovin — Senior Video Streaming Engineer](https://bitmovin.com/careers/7877394002/)
- [Dolby — Staff Data Research Engineer](https://jobs.dolby.com/careers/job/23985650) (AI/ML platform for A/V research)

### Where to actually spend your applications

1. **Qualcomm's video-compression research group** — the single highest-fit req found. Apply first.
2. **Nvidia new-grad research** — Cosmos connection is a real, unusual differentiator. Reference GenStream in the application.
3. **Twelve Labs** — startup, faster loop, and your embedding + video-understanding work is on-target.
4. **Amazon Prime Video** — two distinct doors (encoding research, GenAI applied science); you fit both, which is rare.
5. **Mid-size video infra** (Bitmovin, LiveKit, Dolby, V-Nova, Visionular) — shortest loops, depth valued over publication count, and your systems background counts more here than anywhere else. **Under-applied-to relative to their quality.**
6. **Netflix / Apple / Meta / Google** — apply, but treat as long-cycle. Their full-time research reqs open irregularly; a referral matters more than the application.

---

# Company-flavored notes

Not exhaustive — these are the flavors that change your answers. Map any company you interview at onto
an archetype above, then check whether one of these notes applies.

**Streaming platforms** (Netflix, Prime Video, YouTube, Meta, TikTok, Disney/Hulu, Roku, Bilibili)
The recurring question is a **cost model**: encode once, deliver N times. That single relation decides
how much compute a video deserves, which codecs to support, how many ladder rungs, and whether per-shot
optimization pays. Netflix specifically: VMAF internals and its failure modes, Dynamic Optimizer /
convex hull, AV1 at ~30% of streams, film grain synthesis, subjective-study design. Expect direct
technical debate — push back with evidence.

**Silicon & systems** (Nvidia, Qualcomm, Apple, Intel, AMD, Samsung, Ambarella)
Hardware/software boundary reasoning. Why fixed-function blocks exist, what they cost you in
flexibility, zero-copy pipelines, roofline/bandwidth reasoning — all at the **pipeline level**, which
is where your experience actually is. **Nvidia**: they built the Cosmos Tokenizer you use, so have
opinions about it. If kernel-level CUDA comes up, the honest answer is the right one — "I work at the
pipeline and model level; I've optimized NVDEC→inference→NVENC dataflow, not written kernels" — and
then redirect to what you *have* optimized. Screen for roles where that's a gate before investing time.
**Apple**: on-device constraints, ANE op coverage, CoreML conversion, color management,
Low-Latency HLS, MV-HEVC for spatial video.

**At-scale infra** (Google/YouTube, Meta, Amazon)
Custom silicon exists because transcoding is a top-line datacenter cost — Google's Argos VCU
(20–33× compute efficiency, gen-2 does AV1), Meta's MSVP (~9× vs libx264). Expect "design transcoding
for 500 hours uploaded per minute" and "which videos get the expensive encode." Amazon adds the
Bar Raiser and Leadership Principles; Meta weights behavioral heavily.

**Mid-size video infra** (Bitmovin, Mux, Agora, LiveKit, V-Nova, Visionular, Dolby)
Smaller loops, less ritual, far more depth on the actual product. You'll talk to people who ship
encoders or SFUs for a living, so vague answers get caught immediately — but a real systems background
counts for more here than a publication list. Often the highest-leverage applications for someone with
your profile, and usually faster.

**Foundation-model labs** (DeepMind, FAIR, OpenAI, Runway, Luma, Twelve Labs, ByteDance Seed)
Publication record and research taste dominate. The job talk is the round. Expect to defend
methodology hard and to be asked what you'd do with 100× compute.

---

## Cross-cutting: what every one of them probes

1. **Can you reason about a tradeoff you weren't handed?** Every strong answer here is "it depends on
   X, and here's the curve." Latency vs quality, rate vs distortion vs perception, FLOPs vs bandwidth,
   fidelity vs realism.
2. **Do you know *why*, not just *what*?** "HEVC is 50% better" is weak. "Larger flexible partitioning,
   better MV prediction, 8-tap interpolation, and SAO — and the partitioning is why the gain is bigger
   at 4K" is strong.
3. **Have you shipped and measured, or only read?** Say how you measured, what confounded it, and what
   you'd trust.
4. **Do you volunteer the limitation?** Naming your method's weakness before they find it is the
   single strongest credibility move available.

---

## Weak spots to shore up

- **Classical codec internals at source level** — you're strong at the systems/ML layer. Be ready for
  "how does CABAC context selection work" or "walk me through x264's mode decision."
- **Coding fluency under time pressure** — the local `.py` files are good domain practice; also drill
  plain LeetCode-medium, since it's a *floor* in every loop.
- **A crisp, non-defensive story for why neural codecs aren't deployed** — you'll be tested on this
  wherever traditional codecs pay the bills.
- **The job talk itself** — 30–45 min, rehearsed out loud, timed, with backup slides. Most PhD
  candidates under-invest here relative to how much it counts.
- **Behavioral stories** — 5–6 STAR narratives ([04](04_resume_drill.md) §8).

---

## Four-week plan

**Two parallel tracks, every day.** Content alone doesn't get you hired at this level; the talk does a
disproportionate amount of the work. Running them in parallel also means the talk gets *four weeks of
iteration* instead of being crammed at the end — which is the actual difference between a good talk and
a rehearsed-once talk.

| | Track A — Domain content | Track B — Talk & stories |
|---|---|---|
| Time/day | ~1.5–2 h | ~30–45 min |
| Output | 162 drill questions answerable cold | One talk you can give in your sleep + 7 project stories |

**Daily ritual** (10 min, before Track A): pull 10 random drill questions you've already covered and
answer them **out loud** from memory. Spaced repetition is the only thing that makes 162 questions stick.

---

### Week 1 — Video & networking foundations · pick and skeleton the talk

| Day | Track A | Track B |
|-----|---------|---------|
| 1 | [01](01_video_domain.md) §1–2 color, transform, quantization, JPEG entropy coding · drill Q1–8 | **Pick the project.** Write your thesis as ONE sentence. If it takes two, you haven't found it. |
| 2 | [01](01_video_domain.md) §2–3 image codecs, video codecs, arithmetic coding/CABAC · drill Q9–20 | Write the motivation as **prose, no slides** — why is this hard, for someone outside your subfield |
| 3 | [01](01_video_domain.md) §3 second pass · drill Q21–31 | Outline: **8–12 slide titles only.** No content yet. |
| 4 | [05](05_networking_streaming.md) §§1–4 network foundations through HTTP streaming · network drill Q1–14 | Build the motivation slides (3–4). This is where you'll spend the most time and it's worth it. |
| 5 | [05](05_networking_streaming.md) §§5–8 ABR, real-time media, CDN, debugging + [04](04_resume_drill.md) §4–5 VR streaming · network drill Q15–28 | Build the method slides |
| 6 | [01](01_video_domain.md#32-rate-control-cbr-vbr-cqp-and-buffers) §3.2 rate control and buffers, then **reread 01 §1–3 cold, timed**; mark every miss and reread only those sections | Build results slides + start the backup-slide inventory |
| 7 | Light / catch-up | **Run-through #1, alone, timed, recorded.** Watch it. It will be bad. That's the point. |

### Week 2 — ML domain · cut the talk down

| Day | Track A | Track B |
|-----|---------|---------|
| 8 | [02](02_ml_domain.md) §1.1–1.3 fundamentals, losses, regularization · drill Q1–11 | Watch the recording. **Cut 30%.** Everyone's first version is too long and too detailed. |
| 9 | [02](02_ml_domain.md) §1.3.2–1.3.3, §1.4 optimization, normalization, CNN · drill Q12–15 | Rebuild the trimmed version |
| 10 | [02](02_ml_domain.md) §3, §6 transformer, generative models · drill Q16–31 | **Q&A bank**: predict 20 questions the audience will ask |
| 11 | [02](02_ml_domain.md) §2–4 LLM internals, VLM/MLLM · drill Q32–36 | Answer 10 of them out loud, each in under 60 seconds |
| 12 | [02](02_ml_domain.md) §5, §10–11 post-training, efficiency/deployment, parallelism · drill Q37–48 | Answer the other 10. Any you can't → that's a backup slide. |
| 13 | [02](02_ml_domain.md) §12 + **full 02 drill cold, timed** | Backup slides for **every number** on your headline slide |
| 14 | Light / catch-up | **Run-through #2, recorded, timed.** Compare to #1. |

### Week 3 — Intersection · first live audiences

| Day | Track A | Track B |
|-----|---------|---------|
| 15 | [02](02_ml_domain.md#64-learned--generative-compression) §6.4 learned & generative compression | Read 2 of your external papers (DCVC-FM, a tokenizer paper) |
| 16 | [02](02_ml_domain.md#61-autoencoders--vaes) §6.1–6.2 VAEs and tokenizers | Read 2 more (Q-Align / NTIRE UGC VQA line) |
| 17 | [02](02_ml_domain.md#63-gans--diffusion) §6.3 GANs and diffusion | Read 2 more (BasicVSR++, recent ABR) |
| 18 | [02](02_ml_domain.md#cosmos-tokenizer) §6.5 the Cosmos tokenizer, step by step | Rehearse the **transitions** — the seams between sections are where talks fall apart |
| 19 | [04](04_resume_drill.md) §1–6 project drills, [02](02_ml_domain.md#13-rapid-fire-drill--ml) §13 rapid-fire | Dry run of just the Q&A: have someone fire your 20 questions at you, no talk |
| 20 | **Full 03 drill cold** | **LIVE TALK #1** — a labmate. Brief them: *interrupt me*. Write down every question you fumbled. |
| 21 | Light / catch-up | Fix the talk from those questions. This iteration is worth more than the previous six days of slide work. |

### Week 4 — Pressure, breadth, and the rest of the loop

| Day | Track A | Track B |
|-----|---------|---------|
| 22 | Coding: the inline 01 drills P0 (`image_codec`, `arithmetic_coding`, `intra_prediction_mode`, pipeline motion search) + the networking/ABR drill in [05](05_networking_streaming.md) | [04](04_resume_drill.md) projects 1–3: **90-second AND 15-minute version of each, out loud, timed** |
| 23 | Coding: LeetCode medium × 4, timed | [04](04_resume_drill.md) projects 4–7 + the earlier-work one-liners |
| 24 | ML implementation drill: [02 §14](02_ml_domain.md#ml-coding-companion) Transformer ladder; then PSNR/SSIM, SAD motion search, and the top-k / Annex-B placeholders | Behavioral: write out 6 STAR stories ([04](04_resume_drill.md) §9) |
| 25 | **System design mock #1**: design a transcoding pipeline at scale. Whiteboard, out loud, 45 min. | Behavioral: say all 6 out loud. Cut each to 2 minutes. |
| 26 | **Mock #2**: design a UGC video quality system. **Mock #3**: design a low-latency cloud-VR stack. | **LIVE TALK #2** — different audience, ideally someone *outside* your subfield |
| 27 | Re-drill every question you've ever missed (keep the list from day 1) | Final talk polish. Rehearse the opening 2 minutes until it's automatic. |
| 28 | Target-specific: read the team's recent papers/blog posts. Prepare your questions for them ([04](04_resume_drill.md) §9). | Full dress rehearsal: talk + Q&A + one project deep-dive, back to back |

---

### Job talk build checklist

Track B produces this. Check it before any onsite.

- [ ] **One-sentence thesis** you can say without notes
- [ ] **First 5 minutes are motivation**, comprehensible to someone outside your subfield
- [ ] **8–12 content slides.** More than 15 means you're surveying, not going deep.
- [ ] **One project, not five.** A talk covering everything signals nothing.
- [ ] **Limitations stated out loud by you**, before anyone asks
- [ ] **A backup slide for every number** on your headline slide
- [ ] **20-question Q&A bank**, each answerable in under 60 seconds
- [ ] Rehearsed: alone (recorded) → labmate (interrupting) → outside-subfield listener
- [ ] Timed to **land 5 minutes short** — you will run long when interrupted
- [ ] Opening 2 minutes memorized, so nerves can't touch it

### Scaling the plan

- **Two weeks instead of four**: keep every Track B day, halve Track A by skipping the "full drill cold"
  days and reading §-summaries instead of full sections. Content is recoverable in an interview; a bad
  talk isn't.
- **Six-plus weeks**: add a second job talk on a different project (Trinity *and* the TikTok VQA work),
  so you can match the talk to the team. Add a weekly full-drill rotation across all three files.
- **Interview already scheduled inside two weeks**: skip to Week 4, then backfill Track A in whatever
  order matches the team's archetype.

---

## Sources

**Interview process**
- [Meta Research Scientist interview guide 2026](https://davidfosterhq.medium.com/meta-research-scientist-interview-guide-get-hired-in-2026-a26c5e0c77ca) ·
  [Google DeepMind interview process 2026](https://www.techinterview.org/post/3233474918/deepmind-interview-process-2026/) ·
  [Amazon Applied Scientist guide](https://dataford.io/interview-guides/amazon/applied-scientist) ·
  [Microsoft Applied Scientist guide](https://www.tryexponent.com/guides/microsoft-applied-scientist-as-interview) ·
  [Nvidia Research Scientist prep](https://www.cleverprep.com/companies/nvidia/research-scientist)

**Roles / landscape**
- [Amazon Prime Video encoding research engineer](https://www.amazon.jobs/en/jobs/10402367/research-engineer-prime-video-encoding-team) ·
  [Prime Video GenAI applied scientist](https://www.amazon.jobs/en/jobs/10441113/applied-scientist-prime-video-generative-ai) ·
  [Twelve Labs — video embedding research scientist](https://jobs.techstars.com/companies/twelve-labs/jobs/11851324-ml-research-scientist-video-embedding-model) ·
  [CVPR 2026 careers board](https://cvpr.thecvf.com/careers/) ·
  [Top video startups 2026](https://www.failory.com/startups/video)

**Domain**
- [State of Streaming Codecs 2026](https://www.streamingmedia.com/Articles/Editorial/Featured-Articles/The-State-of-Streaming-Codecs-2026-173838.aspx) ·
  [AV1 now 30% of Netflix streaming](https://netflixtechblog.com/av1-now-powering-30-of-netflix-streaming-02f592242d80) ·
  [AOMedia releases AV2](http://aomedia.org/press%20releases/Alliance-for-Open-Media-Releases-AV2-Codec/) ·
  [VMAF: The Journey Continues](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12) ·
  [Best practices for VMAF](https://streaminglearningcenter.com/encoding/best-practices-for-netflixs-vmaf-metric.html) ·
  [The Bjøntegaard Bible](https://arxiv.org/pdf/2304.12852) ·
  [Pensieve (MIT)](https://web.mit.edu/pensieve/content/pensieve-tech-report.pdf) ·
  [Google Argos VCU](https://www.datacenterdynamics.com/en/news/google-develops-custom-argos-video-transcoding-chips-for-youtube-processing/) ·
  [Meta AV1 on Reels](https://engineering.fb.com/2023/02/21/video-engineering/av1-codec-facebook-instagram-reels/) ·
  [Q-Bench-Video](https://arxiv.org/pdf/2409.20063) ·
  [KVQ short-form UGC VQA](https://arxiv.org/pdf/2402.07220) ·
  [NTIRE 2025 short-form UGC VQA](https://arxiv.org/pdf/2504.13131) ·
  [Post-training: PPO/DPO/GRPO guide](https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms)

**ML systems**
- [How to Scale Your Model (DeepMind / JAX scaling book)](https://jax-ml.github.io/scaling-book/) — the reference for
  roofline reasoning, arithmetic intensity, transformer FLOP/memory accounting, sharding, and the prefill-vs-decode
  math behind [02](02_ml_domain.md) §7–9 and §10.3. Read Part 1 (rooflines) and the inference chapters before any
  efficiency/deployment round; do the "how many FLOPs / bytes does this layer move" exercises — they get asked verbatim.
