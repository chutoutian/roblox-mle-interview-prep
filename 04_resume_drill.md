# Part IV — Resume Drill

For every project: the **90-second pitch**, the questions they *will* ask, the answer to lead with,
and the weakness to name before they find it.

**The universal rule of a project round**: lead with the *non-obvious insight*, not the method.
"We used a VLM" is forgettable. "Latency tolerance isn't uniform across action types, so you should
spend your latency budget asymmetrically" is what they remember and repeat in the debrief.

---

## The five questions asked about every project

Have an answer for these on all of them, before you go in:

1. **Why is this hard?** (If the answer sounds easy, you've mis-framed it.)
2. **What was the alternative, and why is yours better?** (Name a real baseline you actually ran.)
3. **What's the one insight someone else wouldn't have had?**
4. **Where does it break?** (Volunteer this. It's the strongest credibility move available.)
5. **What would you do differently / next?**

---

# 1. GenStream — Generative Codec Real-Time Video Streaming

**Spoken version, about two minutes.**

> This project started from a question: how do you stream video over an extreme link, a few hundred kilobits per second with packets getting lost? In this case H264,H265 fell apart due to its compression efficiency and its vulnerability to packet loss.
>
> Therefore, we are investigating the generative models for the video compression. The idea is to stop compressing pixels. Instead, a learned encoder turns each region of the frame into a latent vector and quantizes it to the nearest entry of a codebook that is shared between sender and receiver. So we send only a grid of integers, the token indices, and the receiver's decoder rebuilds the frame from them.
>
> That buys two things. The tokens are a semantic description of the image, so the payload is much smaller at the same visual quality. And tokens are fixed-length and independent, so a lost packet costs a few regions, which the decoder fills in, instead of breaking the frame.
>
> The next question was adapting to the network. In a token codec the rate knob is not QP but how many tokens you send, and we turn it by downscaling the frame before tokenization to meet the target bitrate from the transport layer. The rate then matches the target exactly, because the token count is fixed by the resolution, while a traditional encoder still fluctuates with scene complexity and with I- versus P-frames.
>
> We evaluated in emulation: WebRTC ran between a sender and a receiver over Mahimahi, which replayed real FCC broadband traces as the link capacity, with bursty Gilbert–Elliott loss averaging about 8%. At matched bitrate and loss, we saved 50 to 78% of bandwidth against H.264 and H.265, and improved LPIPS by 38 to 59% over the neural-codec and FEC baselines, all on a single RTX 4090.
>
> The takeaway: when the channel is the bottleneck, sending tokens instead of pixels gives a codec whose quality degrades with the network instead of collapsing, and whose rate you can control.

**Implementation details.** Fill the bracketed slots from your code and logs.

| Part | What we used |
|---|---|
| Codec | NVIDIA Cosmos tokenizer, [variant, e.g. discrete DV8×16×16]; its structure, token counts and delay are worked through in [02 §6.5](02_ml_domain.md#cosmos-tokenizer). |
| Hardware | One RTX 4090 for tokenizing and detokenizing. |
| Rate adaptation | Content-aware token adaptation: the frame is downscaled before tokenization, and the scale is chosen so that the token rate meets the sending rate set by the congestion controller. |
| Network emulation | Mahimahi between sender and receiver. WebRTC runs over it, and its GCC congestion controller sets the sending rate; the network traces it produces are then replayed for every scheme. |
| Bandwidth traces | FCC Measuring Broadband America raw data (released January 2023): real bandwidth time series. |
| Loss model | Gilbert–Elliott two-state Markov model: good → bad 0.068, bad → good 0.825, loss 0.04 in the good state and 0.525 in the bad state. The channel is bad 7.6% of the time, bursts last 1.2 packets on average, and the mean loss is about 7.7%. |
| Baselines | Grace (loss-resilient neural codec, NVC-based), Reparo (VQGAN image tokenizer, one frame at a time), Tambur (streaming-code FEC on VP9, run on Microsoft's Ringmaster with the authors' hyperparameters, on two Ubuntu 20.04 servers), plus H.264 and H.265 ([encoder, preset, rate-control mode]). |
| Data | Training: UCF101 (13,320 action videos). Evaluation: the 62 clips of 10–30 s from Grace's dataset. |
| Metric | LPIPS measured end to end, after the network, on every decoded frame. |
| Results | At about 0.24 Mbps, mean LPIPS 0.0404: 38% better than Reparo, 45% than Tambur and 59% than Grace. Quality is also steadier: LPIPS standard deviation 0.0295, against 0.0569 for Grace and 0.0604 for Tambur. 50–78% bandwidth savings over H.264/H.265 and prior neural codecs at comparable quality. |

**Questions asked in interviews.** Apple, ML Video Codec Engineer, 2026-10-07.

1. **Q: Extending this to video, how many frames does one token represent?** A: The group is fixed by the tokenizer. With the Cosmos DV8×16×16 variant [confirm the variant you shipped], one token stands for 8 frames of one 16×16 pixel patch, so at 30 fps a token covers 267 ms of one region; a 720p clip gives 45×80 = 3,600 tokens per group, 16 bits each, about 216 kbit/s. The first frame is tokenized on its own, which is how an image is a one-frame video. The 4-frame variant halves the wait to 133 ms, but at 8×8 patches it emits 4 times more tokens per group, 8 times the bitrate (1.72 Mbit/s). In our earlier version with ElasticTok a token covered one patch across a 4-frame block. The span is the trade-off: a longer group costs fewer bits and more latency, and a lost token blanks its region for the whole group. Because the tokenizer is causal, a group's tokens depend only on frames up to its last one, so they can be sent as soon as the eighth frame arrives.

**Questions asked in interviews.** ByteDance.

1. **Q: If the compression ratio is that high, why isn't it used in production?** A: Because bitrate is only one cost. First, compute and energy: we encoded and decoded on an RTX 4090, while phones decode H.264 and H.265 in dedicated hardware at a tiny fraction of that power, and we never showed 30 fps on a phone. Second, delay that faster hardware cannot remove: the tokenizer groups frames, so with the 8-frame variant the oldest frame waits about 267 ms at 30 fps, above the roughly 150 ms one-way target for calls; the 4-frame variant halves that at eight times the bitrate. Third, fidelity: the decoder fills in plausible but possibly wrong text, faces or small objects, the perception–distortion trade-off, and the frame still looks clean, so errors are silent, worst on content unlike our UCF101 training data. Fourth, it is a low-rate win: we evaluated at about 0.24 Mbps with about 8% bursty loss, where H.264 and H.265 fall apart; with more bits they approach transparent quality, while the tokenizer is capped by its own reconstruction, about 25 dB PSNR on DAVIS for 8×16×16. Fifth, the ecosystem: no deployed standard for learned video, both ends need the same model and codebook, so every update is a new format, and the weights must ship to every device. So today it fits bandwidth-bound links where we control both ends, like satellite or disaster networks, and underwater images in AquaScope; wider use needs decoders that fit phone neural accelerators, and a standard, as JPEG AI now is for images. *In the interview I said compute and latency; I would keep both and add the built-in delay, fidelity, the low-rate-only gain and the ecosystem.*

**Questions asked in interviews.** [company].

1. **Q: If the server generates the content itself, for example a digital human shown in XR at stereo 4K, how would you transmit it efficiently, and can you exploit the fact that it is generated?** A: I would not send video at all. The server generates the avatar, so it already has the exact driving signals for every frame: the speech, and the expression and pose parameters. The receiver only needs those plus the same avatar model. The 3D avatar, for example a Gaussian-splatting model of roughly 15 to 25 MB before compression, is downloaded once while normal video plays. After that each frame costs a few dozen to a few hundred numbers, so speech plus parameters is tens of kilobits per second, against tens of megabits per second for stereo 4K at 90 fps, or about 11.7 Mbit/s even with GenStream's tokens. The headset renders each eye locally, so the bitrate no longer depends on resolution or on the number of views, head motion never waits for the network, and the two eyes stay consistent. This extends GenStream: there the rate knob was how many tokens to send; here it is which layer of the generation pipeline to send, text, parameters, tokens or pixels, chosen from the bandwidth and the device, with server-rendered video as the fallback for weak devices. The parameter decoder must give identical results on both sides, so I would implement it in integers and resend a full state periodically. The idea goes back to model-based coding and MPEG-4 face animation in the nineties. It failed because the parameters had to be estimated from a camera and the rendering looked fake. With generated content the parameters are exact, and avatars now render in real time on headsets: TaoAvatar renders 2K stereo at 90 fps on a Vision Pro, and NVIDIA's keypoint talking heads match H.264 quality with 6.5 to 10 times less bandwidth.

**Questions asked in interviews.** ByteDance, second round.

1. **Q: What exactly is the LPIPS, and the PSNR if you have it?** A: The number to say is 0.0404: the mean LPIPS of GenStream, measured end to end after the network on every decoded frame, at a target bitrate of about 0.24 Mbps, on 256×256 video at 30 fps, with bursty Gilbert–Elliott loss averaging about 8%. LPIPS is a distance, so lower is better. At the same rate and loss that is 38% better than Reparo, 45% better than Tambur and 59% better than Grace. Quality is also steadier: the standard deviation of LPIPS is 0.0295, against 0.0569 for Grace and 0.0604 for Tambur, because FEC has a cliff while a token codec degrades gradually. We did not report PSNR, on purpose. A generative decoder synthesizes plausible texture instead of reproducing pixels, so PSNR penalizes it even when the frame looks right; that is the perception–distortion trade-off, which is why we fine-tuned on a weighted sum of LPIPS and MSE and evaluated with LPIPS. If an interviewer wants PSNR, I would compute it on the same decoded frames and expect a generative codec to look worse on PSNR than on LPIPS.
   *In the interview I said* I did not remember the exact numbers. Say 0.0404 at 0.24 Mbps, the 38–59% gains, and the steadier quality.
2. **Q: How do you get from the tokens back to an image? Is it a CNN, and what is the backbone?** A: It is a learned decoder trained jointly with the encoder, not an ImageNet backbone such as ResNet, and its type depends on the tokenizer. In the version behind our paper's numbers we used ElasticTok, whose encoder and decoder are ViTs. The received indices are looked up in the FSQ grid (64,000 codes, so 16 bits per token, 640 to 3,840 tokens per 4-frame block), lost tokens are left as gaps, and the transformer decoder, which we fine-tuned jointly with the encoder under simulated loss up to 20% on LPIPS plus MSE, attends over the whole block and outputs pixel patches. The Cosmos tokenizer, which the resume names, is convolutional with attention only at the bottleneck. Each index becomes an FSQ code, a 1×1×1 convolution lifts it to 16 channels, and residual blocks of causal factorized convolutions follow: a 1×3×3 spatial kernel and a 3×1×1 temporal kernel, with LayerNorm and Swish. Spatial and causal temporal self-attention run at the smallest grid, upsampling layers grow it back, and an inverse two-level Haar wavelet returns pixels; the decoder has about 66M parameters.

**Numbers to keep consistent with the resume.** The 38–59% LPIPS gains are against Reparo (38%), Tambur (45%) and Grace (59%) at about 0.24 Mbps, not against H.264/H.265; the 50–78% bandwidth savings are against H.264/H.265 and prior neural codecs at comparable quality.

---

# 2. AquaScope — Acoustic Underwater Image Transmission

**Spoken version, about two minutes.**

> This project started from a question I had while scuba diving: how could I share a photo with my dive buddy, using only our phones? Underwater, radio does not work, so phones talk through sound, from speaker to microphone. That channel is extremely narrow, about 1.5 to 3.5 kilohertz, roughly 300 bits per second, and noisy, with bit error rates in the tens of percent. A small JPEG would take over a minute, and after a few bit errors JPEG or PNG cannot be decoded at all.
>
> So instead of compressing pixels, we used a generative tokenizer. It turns the image into a short sequence of tokens, each an index into a codebook that both phones have, so a 256 by 256 color image fits in under one kilobit. A wrong index does not break the stream; it changes part of the content, so quality degrades gracefully.
>
> An off-the-shelf tokenizer did not fit this environment, and that is where our contributions are. First, on underwater images more than half of the codebook is never used, so we cluster it with usage-weighted k-means into fewer entries and fewer bits per index. Second, the tokenizer assumes tokens arrive intact, so we fine-tuned the decoder on randomly corrupted tokens, up to a quarter of them, so it reconstructs without knowing which tokens are wrong.
>
> We deployed it on Android phones in waterproof pouches and tested it in a lake, including two people swimming. AquaScope delivered images reliably up to 20 meters in about 9 seconds each, cut the bit error rate from 19% to under 2% against the state of the art, and improved LPIPS by about 37 to 40% over the same codec on earlier physical layers.

**Implementation details.**

| Part | What we used |
|---|---|
| Codec | TiTok tokenizer (transformer-based), initialized from a pre-trained TiTok: 64 tokens with 12-bit indices per 256×256 image, 768 bits before channel coding. A VQGAN version (256 indices of 6 bits after clustering) was also built on Android; TiTok was kept because it matched or beat VQGAN with fewer tokens. 48, 64 and 80 tokens were compared; 64 balanced robustness against compute and airtime. |
| Token clustering | k-means over the codebook embeddings; each merged entry is the usage-weighted sum of its members. Example: 1,024 → 64 entries, so 10 → 6 bits per index, about 40% less data. Measured on VQGAN, where 519 of the 1,024 entries were never used on underwater images. |
| Training | 78,420 public underwater images + 135,000 ImageNet images; 112 epochs of domain fine-tuning, then 60 epochs of error-resilient fine-tuning (replace m random tokens, m uniform in [1, p], curriculum on p up to 25% of the tokens); pixel + perceptual + GAN loss; 4× A100, 11 h + 14 h. |
| Channel coding | Hamming (7,4), diagonal interleaving, Gray coding. |
| Modulation | Chirp spread spectrum in 1.5–3.5 kHz: SF 5, 2 kHz bandwidth, 312 bps, 16 ms symbols sampled at 48 kHz. |
| Receiver | AquaApp's preamble detector; groups of 1 training symbol + 3 data symbols (64 ms, within the coherence time); per-group MMSE time-domain equalizer (240 taps); training-symbol timing by cross-correlation, smoothed with a moving average and allowed to change by at most 40 samples per group; linear interpolation against Doppler. |
| Phone | Samsung Galaxy S21 in a soft waterproof pouch; PyTorch models converted with PyTorch Mobile, under 1 GB, running on the CPU; built on AquaApp's open-source Android code. |
| Evaluation | Lake field tests: dock, bridge (up to 20 m, 3 m deep), lake bank, pedal boat, two swimmers. Test images: 1,216 frames from five diving videos. Baselines: AquaApp+ (OFDM) and CSS+ (standard chirps), both carrying the same codec, plus PNG, JPEG and a learned codec (Cheng et al. 2020) shown error-free. Metrics: LPIPS, index error rate, BER. |
| Cost on the phone | 9.2 s end to end per image (AquaApp+ 10.9 s), airtime dominating; about 55% CPU to encode and 46% to decode; 2.0–2.5 GB memory; 20% battery in 50 minutes. |

**Questions asked in interviews.** Apple, ML Video Codec Engineer, 2026-10-07.

1. **Q: Taking VQGAN as the example, how is the codebook size, that is the bits per token, decided? How is that parameter usually set?** A: It is a hyperparameter set by three things that pull against each other. Rate: each index costs log2 K bits, 1,024 codes are 10 bits and 16,384 are 14; doubling K adds one bit per token, while halving the patch size quadruples the number of tokens, so K sets the precision per token and the grid sets the resolution. Quality: a larger codebook lowers the quantization error with diminishing returns, because the encoder adapts to whatever grid it is given. Utilization: nearest-code training leaves many entries dead, so the nominal K overstates the effective one, and you measure it with the usage histogram and its perplexity. VQGAN ships with 1,024 and 16,384 entries at d = 256. For generation K is also the vocabulary of the prior; for transmission it is literally the bits per index, so you pick it on the target data from the rate–quality curve. On underwater images 519 of VQGAN's 1,024 codes were never used, so we merged the codebook to 64 entries with usage-weighted k-means, 10 to 6 bits per index, and LPIPS barely moved; our TiTok setting is 64 tokens of 12 bits, 768 bits per image.
2. **Q: You cluster the codebook offline with k-means. Would end-to-end training do better?** A: For quality at a fixed bit budget, probably yes: end-to-end training lets the encoder, the codebook and the decoder co-adapt to the domain and to the small K, whereas k-means can only merge embeddings that already exist, and the decoder then sees merged codes it was never trained on, which is why we fine-tune after clustering. We chose clustering for cost. It is training-free: minutes from a usage histogram, no labels, and K can be chosen from the measured curve without retraining once per K. It keeps the pretrained codebook and decoder, and with them the generalization of ImageNet-scale pretraining; our 78k underwater images are far too few to train a tokenizer from scratch, and a 64-entry codebook trained with VQ losses is prone to collapse. What we shipped is the middle path: cluster first, then fine-tune, 112 epochs of domain adaptation and 60 of error-resilient fine-tuning on four A100s, about 25 hours in total. If I did it end to end today I would replace VQ by FSQ with levels chosen to give 6 bits, which removes the collapse risk and lets the encoder learn where those 6 bits should go.

**Numbers to keep consistent with the resume.** The resume's "40% bandwidth efficiency" is the clustering example, 10 → 6 bits per index. The paper's headline is 36.7% better visual quality with a 256×256 image in under 1k bits. "BER from 19% to below 2%" is against state-of-the-art phone systems.

---

# 3. AquaVLM — On-Device VLM for Underwater Communication

**Spoken version, about two minutes.**

> In this project, we built an end-to-end perception and communication system powered by vision language models. Our starting point was this: in daily life we express ourselves by speaking, but sometimes that is not possible. Some people cannot speak, and in extreme environments, such as underwater or in firefighting, people wear masks, so they cannot talk to each other. How can they still express their intent and share information? That requires understanding the surrounding environment and the person's own status and turning them into language, which is exactly what a vision language model does, so we built the system around one.
>
> We use scuba diving as the concrete example. A mobile VLM runs on the diver's phone and analyzes the images the diver captures, together with sensor readings such as depth, temperature and heart rate, to assess the current diving situation and propose suitable messages. The diver selects one, and it is sent through acoustic signals to nearby dive partners. On the receiving end, the VLM processes the incoming message along with the receiver's own context and generates candidate replies.
>
> To get good underwater conversations, we instruction-tuned an existing mobile VLM, MobileVLM V2 with 3 billion parameters, on a custom multimodal dataset. We used GPT-4o with few-shot prompts on frames from real dive videos to generate diverse conversations, organized into three subtasks, message generation, message recovery and reply generation, about 85 thousand examples. We also put the purpose, such as a safety alert or an environment description, into the prompt: the diver picks a purpose first, so the model generates two options instead of eight.
>
> To evaluate it, we built a VR platform where users explore an underwater world, run into events and talk with virtual divers through AquaVLM, plus a working iOS prototype with the model quantized to 4 bits. With 20 participants, messages matched the chosen purpose 70 to 80% of the time, and in a lake test they kept about 90% semantic similarity up to 15 meters. The same approach carries over to firefighting and to autonomous systems.

**Implementation details.**

| Part | What we used |
|---|---|
| Base model | MobileVLM V2, 3B parameters; a 1.7B version was also measured for cost. |
| Fine-tuning | LoRA, rank 64, on the language model only, vision encoder frozen; AdamW, learning rate 4e-5 with cosine decay and 3% warm-up; batch 16 per GPU, 1 epoch, 2× RTX 4090, about 4 h. |
| Data | 1,368 keyframes from five public diving videos; synthesized nine-parameter sensor traces for each dive phase (descent, exploration, ascent, safety stop, completion); GPT-4o in two steps, messages then replies, with guidelines and few-shot examples; 85k instruction pairs: 16k message generation, 21k reply generation, 48k message recovery. The dataset is released. |
| Error-resilient data | Text sent as 5-bit characters with numbers spelled as words; independent bit flips at BER 0–20% in 1% steps; the target is the original message, trained with next-token cross-entropy. |
| Hierarchy | Purpose first: safety, navigation, environment or equipment for messages; acknowledge, refuse, assist or resend for replies. Two candidates instead of eight; a threshold filter keeps only abnormal sensor readings in the prompt, such as water below 15 °C or tank below 700 psi. |
| On-device | 4-bit quantization with llama.cpp to GGUF, run through LLMFarm on the iPhone GPU via Metal; app in Swift and SwiftUI; iPhone 12 Pro + Apple Watch Ultra, the watch sending its readings to the phone acoustically within 1 m. |
| Acoustic link | (7,4) Hamming, chirp spread spectrum (SF 5, 2 kHz, 312 bps), a training symbol every 3 data symbols, preamble detection: the same physical layer family as AquaScope. |
| VR simulator | Unity on Meta Quest 3, rendered on a server and streamed over AirLink; four random events (shark, tank leak, shipwreck, kelp forest); the server runs the same model and replays real phone latency traces. |
| Evaluation | VR study with 20 participants; lake test at 1–20 m with 100 conversation rounds; baselines AquaApp+ (OFDM) and CSS+ carrying the same model; semantic similarity as the cosine of all-MiniLM-L6-v2 embeddings, with a human-calibrated threshold of 0.92. |
| Cost on the phone | 3B: about 95% CPU, 3.48 GB GPU memory, 27% battery per hour. 1.7B: 1.81 GB, 21% per hour, about half the inference latency. |

**Questions asked in interviews.** ByteDance.

1. **Q: Which model did you use, and what is MobileVLM's own design?** A: We used MobileVLM V2 3B, and measured the 1.7B for cost. It has LLaVA's three parts, but only two are built for phones. The vision encoder is the standard CLIP ViT-L/14 at 336×336, giving 576 patch tokens. Version one's lightweight downsample projector, LDP, used pointwise and depthwise convolutions, a stride-2 depthwise one cutting 576 tokens to 144. V2's LDPv2 is simpler: two pointwise convolutions map CLIP features to the LLM width, a 2×2 average pool does the same 576 to 144, and a 3×3 depthwise convolution with a skip connection, a PEG, enhances positional information. At the 1.7B scale that is 6.3M parameters instead of 18.9M, no bigger than a plain two-layer MLP, which it matched on average with a quarter of the tokens. The language model is MobileLLaMA, 1.4B or 2.7B, a LLaMA-style decoder with RMSNorm, SwiGLU and RoPE, trained from scratch on 1.3T RedPajama tokens with the LLaMA2 tokenizer and a 2k context. V2 also changed the training: 1.2M ShareGPT4V captions, then 2.4M multi-task samples, training the projector and LLM in both stages with CLIP frozen. It suited us for three reasons. The phone prefills 144 image tokens instead of 576 before the first word. llama.cpp converts V2 natively and LLMFarm runs it, so the language model went to 4-bit GGUF on the iPhone 12 Pro GPU. And the weights are open, with the 3B beating many 7B models on the paper's benchmarks. We tuned only the language model, LoRA rank 64, vision encoder frozen.

**Questions asked in interviews.** ByteDance, second round.

1. **Q: Which inference engine runs the model on the phone?** A: llama.cpp. After fine-tuning, the model is converted to GGUF, the language model is quantized to 4 bits, and it runs through LLMFarm on the iPhone 12 Pro GPU via Metal. We chose it because it supports MobileVLM V2's projector natively and has a Metal backend, so no custom kernels were needed. On the phone the 3B model uses about 95% CPU, 3.48 GB of GPU memory and 27% battery per hour; the 1.7B model uses 1.81 GB and 21% per hour, at about half the inference latency. Next I would compare MLC-LLM or a Core ML build that can use the Neural Engine.\
   **The whole pipeline.** Who does what, from top to bottom: our Swift app; LLMFarm, which wraps llama.cpp for iOS and handles model loading, the chat template and sampling settings; llama.cpp, which holds the model graph, the quantized weights, the tokenizer and the KV cache; Metal, Apple's GPU programming interface and the counterpart of CUDA, which runs the kernels; and the A14 GPU, which shares the phone's 6 GB of memory with the CPU. From the trained model to a reply:
   1. **Fine-tune.** LoRA on the language model in PyTorch, then merge the adapters into the base weights [confirm], so the checkpoint is a plain model.
   2. **Convert.** llama.cpp's scripts, which support MobileVLM V2's LDPv2 projector, write two GGUF files: the language model, and an mmproj file with the CLIP encoder and the projector. GGUF stores the weights with the architecture, hyperparameters and tokenizer, so each file is self-contained.
   3. **Quantize.** llama-quantize rewrites the language model to 4 bits [confirm the type, e.g. Q4_K_S]: each small block of weights keeps 4-bit integers plus its own scale, about 4.5 bits per weight, so 2.7B parameters take about 1.5 GB. The mmproj stays FP16 by default.
   4. **Package.** LLMFarm compiles llama.cpp in through its Swift library and loads both files; our Swift/SwiftUI code adds the purpose menu, the watch's sensor readings and the acoustic link [confirm whether we modified LLMFarm or embedded its library in our own app].
   5. **Load.** llama.cpp memory-maps the model file, and because memory is unified, Metal can read the weights in place, with no copy to separate video memory.
   6. **Run.** The image encoder turns the photo into 144 embeddings, spliced into the prompt with the purpose and the abnormal sensor readings. llama.cpp builds each forward pass as a graph of operations, and its Metal backend runs each one as a hand-written GPU kernel: matrix multiplications that unpack the 4-bit weights on the fly, attention, RMSNorm, RoPE. Prefill pushes all prompt tokens through at once and fills the KV cache; decoding then emits one token at a time, sampled on the CPU, until both candidates are written. Each decoded token reads every weight once, so decoding speed is set by memory bandwidth, which is why 4-bit weights decode faster.
2. **Q: Besides the purpose-alignment rate, which metrics did you use to evaluate the model?** A: Three groups. Subjective, from the 20-person VR study: the purpose-align rate, 70–80%, plus 1-to-5 opinion scores for responsiveness, interaction, immersiveness and overall experience from a nine-question questionnaire; responsiveness scored lowest, because a messaging round takes time. Objective, for communication: BER, and semantic similarity between the sent and the recovered message, the cosine of all-MiniLM-L6-v2 sentence embeddings. We calibrated its threshold with a second human study on 200 message pairs: agreement that the messages meant the same dropped below 0.92, so 0.92 counts as success. In the lake, BER stayed under 3% within 20 m and similarity above 90% within 15 m, and BER stayed below 2% while the phones moved. System: the one-round latency split into generation, recovery, reply and transmission, and the phone's CPU, memory and battery. Ablations add one more: fine-tuning keeps messages meaningful up to 6% BER, while the original model fails at 1.5%.
3. **Q: Why use people to score, and not a bigger model as the judge?** A: Because the bigger model is where our training data came from. GPT-4o wrote the 85k training conversations, so judging with GPT-4o would measure how well the small model imitates its teacher, and LLM judges tend to favor text that resembles their own. Second, purpose alignment is defined by the diver's own intent: each participant chose the purpose and then reviewed their own session, so only they can say whether a message matched. Third, part of the evaluation is about the experience itself, responsiveness and immersion, which a model cannot judge. Where an automatic metric was enough we did use one: semantic similarity for transmission, with its threshold calibrated on human judgments. A larger model would be a cheap complement for message quality and safety, once it agrees with the human labels on a subset.
   *In the interview I said* we used a large model to make the ground-truth data. That is the reason not to use it as the judge as well: make the circularity point explicit.
4. **Q: Briefly, what is the overall structure of the VR system?** A: It is the platform for our user study. Real dives are dangerous, costly and hard to repeat, and 15 of our 20 participants had never dived, so we built a controllable, repeatable underwater world where people use AquaVLM to handle events. It is one headset and one server. The headset, a Meta Quest 3, only displays and takes input: a virtual phone screen in the view is AquaVLM's interface, and a gesture recognizer lets users swim with their hands instead of using controllers. The server renders the Unity scene, with photorealistic assets, lighting that changes with depth and obstacles, and real underwater sound; it randomly triggers four events, a shark approaching, a tank leak, a shipwreck and a kelp forest, and streams the frames to the headset over Air Link, because rendering on the headset gives a low frame rate and makes people dizzy. The same server runs the same 4-bit model as the phone. When a user takes a photo and picks a purpose, the frame and simulated sensor readings go to the server, and the candidates come back after a delay replayed from inference and transmission latencies measured on phones, so the experience matches the real device. One of us played the dive partner. Every session is logged; afterwards participants reviewed their own images, messages and replies on a web page and marked whether each message matched the purpose they had chosen, which gives the 70–80% purpose-align rate. The split is deliberate: the VR study measures whether the messages are good and usable, while the iOS prototype and the lake test measure whether they arrive intact and what they cost the phone. VR also exposed corner cases early: participants who were lost and wanted to go back got poor answers, which suggests a backtrace purpose, and their feedback led to two features, "Go back" and "Generate more candidates".
5. **Q: What experience or insights do you have on model deployment?** A: My experience is mostly on-device and real-time systems. In AquaVLM we ran a 3B VLM on an iPhone 12 Pro. We measured first: a messaging round is dominated by model inference and the 312 bps acoustic link, while encoding and modulation are negligible, so the goal was to generate less and send less. Then we cut cost. On a phone, latency is roughly the prefill of the input tokens plus the number of output tokens times the time per token, and we reduced all three. Fewer input tokens: MobileVLM V2's projector pools 576 image tokens to 144, and keeping only abnormal sensor readings in the prompt, such as water below 15 °C, improved the messages and shortened the prompt. Fewer output tokens: the diver picks a purpose first, so the model writes two candidates instead of eight, which also suits a small screen. Faster tokens: llama.cpp and LLMFarm, 4-bit GGUF, on the GPU through Metal; the 3B model used 3.48 GB and 27% battery per hour, enough for a one-hour dive, and the 1.7B model needs 1.81 GB and halves the latency. Finally we validated on the device itself: in the lake test everything ran on the phones, and semantic similarity stayed above 90% within 15 m. Three lessons. The cheapest computation is the one you skip; at TikTok, adaptive frame sampling made SAM3 surface detection 8.5× faster. Train for the deployment environment: we fine-tuned on the bit-error range measured underwater, and the model keeps the meaning at 6% BER, where the original fails at 1.5%. Accuracy is not the only metric: memory, battery and perceived latency have to be measured on the device, and responsiveness scored lowest in our study, because a messaging round takes time and many people expected a phone call. Elsewhere, AquaScope ran its tokenizer with PyTorch Mobile on an Android CPU, and Trinity split rendering between headset and server by latency sensitivity. All of this is prototypes, not large-scale serving: for serving I know the levers, such as continuous batching, paged KV cache as in vLLM, FP8 or INT8 weights and speculative decoding, but have not run them in production.
   *In the interview I said* two things: cutting cost at the algorithm level, such as the hierarchical generation, and quantization. Keep that order, but add the numbers and the toolchain, start from what we measured, and end with a lesson: "quantization" alone is what every candidate says.

**Numbers to keep consistent with the resume.** The resume says ">80% user-intent alignment"; the IMWUT camera-ready says 70–80%, so quote the paper. The resume's "60–70% lower inference latency" comes from generating two candidates instead of eight; check it against the paper's latency figure before quoting.

---

# 4. MARS — Motion-Aware ABR for Real-Time VR

**Spoken version, about two minutes.**

> This project is about cutting the bandwidth of real-time VR streaming without users noticing. Streaming VR over Wi-Fi at 4K and 72 frames per second can need hundreds of megabits per second. Most prior work saves bandwidth spatially, by sending only the viewport or by foveation, which depends on accurate viewport or gaze prediction. We looked at the temporal side instead: when your head moves fast, your dynamic visual acuity drops and you cannot see fine detail, so during those moments we can lower the bitrate for free.
>
> The idea is simple, adjust the bitrate to the head motion, but two things make it hard. The first is timing. Motion-to-photon latency in our system is at least 70 milliseconds, so a frame encoded now is displayed later. If we react to the current speed, we waste bits at the start of a turn and show a low-quality frame after the head has already stopped. So we built a kinematics-based state machine with four phases, slow, speed-up, fast and decelerate, driven by speed and acceleration thresholds, with minimum and maximum dwell times profiled from thousands of real head-motion sessions. When it enters speed-up or decelerate, it extrapolates the speed one motion-to-photon latency ahead, so the bitrate change lands when the frame is actually shown.
>
> The second is how much to cut. We used the just-noticeable-difference principle through a perceptual metric, PSPNR. Offline, we encoded 8K VR videos at a ladder of bitrates and built one quality-versus-bitrate curve per head-speed range. At runtime, for the anticipated speed, we pick the lowest bitrate whose perceived quality still matches what a static viewer would get at the target bitrate.
>
> We built it into ALVR with an HEVC encoder and evaluated it two ways. In trace-driven emulation it saved 10 to 19% of bandwidth compared with a fixed bitrate. In a user study with 31 participants on a PICO headset, 21 could not tell it apart from the fixed bitrate, and nobody reported a significant difference.

**Implementation details.**

| Part | What we used |
|---|---|
| Streaming system | ALVR (open source), extended in three modules: TrackingManager (head speed and acceleration), StatisticsManager (measured motion-to-photon latency as the prediction horizon) and BitrateManager (applies the bitrate). |
| Encoder | HEVC in CBR mode through NVIDIA NvCodec; the bitrate changes per frame through the Reconfigure API. |
| Testbed | Laptop server with an RTX 4080, i7 and 32 GB; PICO Neo3 headset over Wi-Fi 6 (802.11ax); 4K, 72 fps, 60 Mbps target. |
| State machine | v_slow = 20°/s, v_fast = 60°/s, a_up = +10°/s², a_down = −10°/s²; minimum dwell times such as 74.1 ms for slow and 25.5 ms for decelerate, maximum 708 ms for speed-up and 773 ms for decelerate, profiled from 3,791 sessions over 88 360° videos; anticipation v = v_now + a·MTP. No learned model. |
| Perceptual map | PSPNR (JND-based, from Pano); 17 8K YouTube VR videos encoded at 10–60 Mbps in 5 Mbps steps, 5 head trajectories each; speed bins of 10°/s up to ≥100°/s. Rule: the lowest bitrate whose PSPNR at the anticipated speed equals the static PSPNR at the target. |
| Emulation | E3PO + NvCodec: a 2K viewport video cut from 8K 60 fps panoramas; head traces from the 360VidStr dataset (10 Hz, yaw and pitch); 150 network traces (Lumos 4G and Wi-Fi, Lumos5G). Baselines: Fixed and Pano- (a speed-only JND map). |
| User study | 31 participants. Study 1, 14 people: blind, randomized MARS vs Fixed. Study 2, 17 people: MARS vs Fixed or Fixed vs Fixed, as a control for response bias. Apps: Secret Shop (Valve) and the Unreal Engine VR template. |
| Overhead | Per-frame encode time unchanged when switching between 45 and 60 Mbps every 30 frames. |

**Questions asked in interviews.** ByteDance, second round.

1. **Q: Did this feature ship?** A: No. It is a research prototype built into ALVR, the open-source VR streaming stack, and evaluated in trace-driven emulation, where it saved 10–19% of bandwidth against a fixed bitrate, and in a 31-person user study on a PICO headset. It was designed to be easy to deploy: a runtime-agnostic bitrate controller that needs only head-motion telemetry and the encoder's rate-control interface, with no game-engine changes and no learned model, and switching the bitrate per frame through NvCodec's Reconfigure API did not change the encode time. To ship it, I would integrate it into a production stack such as Steam Link or Air Link behind an A/B test, check how it interacts with congestion control, which already moves the bitrate, and measure bandwidth, stalls and quality complaints; then personalize the speed thresholds per user.

**Numbers to keep consistent with the resume.** The resume says 15%; the paper says 10–19%, depending on the network and the amount of motion. The 10–19% comes from emulation and the 21 of 31 from the user study; do not attribute the savings to the user study. The paper file is the ACM MM 2026 version while the resume lists a submission to MMSys 2027: say whichever is current.

---

# 5. Trinity — QoE for Cloud VR Gaming

**Spoken version, about two minutes.**

> In Trinity we looked at cloud VR gaming, where the game runs on a remote GPU and is streamed to a standalone headset. The hard requirement is motion-to-photon latency: to feel natural, the scene has to respond within about 25 milliseconds, but a network round trip plus rendering and encoding easily takes over 100. We targeted a first-person shooter, which is about as latency-sensitive as games get.
>
> Our intuition was that not every action needs 25 milliseconds. We split inputs into three kinds of motion: head motion, hand motion like aiming and firing the gun, and body motion like walking. Head rotation is the most sensitive, so we handle it on the headset: the server renders a partial panorama, about 140 by 130 degrees, and when the head turns the headset re-projects a new viewport from it, which keeps head latency under 25 milliseconds. Body motion and the rest of the world are rendered on the server and take the full round trip, around 130 milliseconds.
>
> The problem, common to earlier systems, is the gap this creates: the view moves instantly but the gun and the world respond late, and players notice the mismatch. Our key finding was that the mismatch matters more than any single latency. So we render the gun and the crosshair locally on the headset, and instead of showing their update immediately, we delay it on purpose, by an amount that follows the measured round trip and puts hand latency between head and body, around 90 milliseconds in our runs. That way the gun's recoil and the explosion it causes feel like one consistent response.
>
> We built it on Unity's open-source FPS sample, with a PICO headset and an RTX 4080 server, and ran a user study with 31 participants. With a round trip of about 130 milliseconds, Trinity scored an overall MOS of 4.0, against 3.9 for ALVR on a local Wi-Fi network, and compared with cubemap streaming it used about 40% less GPU and 60% less bandwidth.

**Implementation details.**

| Part | What we used |
|---|---|
| Game | Unity FPSSample, an open-source multiplayer first-person shooter, adapted to VR: headset pitch, yaw and roll drive the camera, the left joystick moves, the right joystick aims, the triggers fire and throw. |
| Server | Laptop with an RTX 4080, i7 and 32 GB. A three-camera rig renders a 140×130° partial panorama, which covers over 99% of head movement at a 130 ms round trip according to prior work (Freedom). It is packed with VAM (from Freedom) and AADT foveated downsampling (from Oculus Link). The gun and crosshair are removed from the server render. |
| Encode and transport | YUV420 with the motion index embedded in the frame, NVIDIA hardware H.265, TCP; the pipeline is packaged as DLLs inside Unity. |
| Client | PICO Neo3 (Android) over Wi-Fi 6; MediaCodec H.265 decoding; frames drawn on a skybox shader; head re-projection on the headset; 72 Hz display. |
| Hand delay | A queue of the gun's past positions and rotations. Hand latency equals head latency while body latency is under 70 ms, equals ¼ head + ¾ body between 70 and 160 ms, and equals body latency from 160 ms on. Thresholds come from manual calibration at 50–160 ms. |
| Latency measurement | Each input becomes an integer motion index that travels to the server and back inside the frame, so the client measures the round trip. |
| Network | A Linux gateway emulating delay; only stable networks were tested. |
| User study | 31 participants aged 20–65, 20 of them new to VR; local ALVR, remote ALVR and Trinity, 5 minutes each, randomized and blind; MOS for visual quality, responsiveness, artifacts, sickness and overall. Ablations: hand at head latency (Trinity-L) and hand at body latency (Trinity-R). |
| Resources | Measured on the Viking Village scene, H.265 at QP 26: about 40% less GPU and 60% less bandwidth than cubemap streaming, slightly more than viewport-only streaming; about 10% battery in 20 minutes. |

**Questions asked in interviews.** None recorded yet; add them after each interview, with the answer you gave and the answer you wish you had given.

**Numbers to keep consistent with the resume.** "60% less bandwidth and 40% less GPU" are versus cubemap streaming, not versus viewport streaming. "QoE comparable to local streaming" is the overall MOS of 4.0 against 3.9. The paper's intro calls the 130 ms "network delay"; it is the full round trip, including rendering and encoding.

---

# 6. TikTok — UGC Short-Video Quality Assessment via MLLM

**Spoken version, about two minutes.**

> Video quality assessment sits at the center of a short-video platform: every upload is re-encoded into a ladder of resolutions and bitrates, and quality scores guide those transcoding choices, adaptive delivery and A/B tests. Traditional approaches, hand-crafted metrics or neural regressors, work on the data they were designed for but break on new distributions. Multimodal large language models generalize better because of their pretrained semantic knowledge, and they are now the state of the art.
>
> But existing MLLM scorers are context-agnostic: they give one score per video. In transcoding, what matters is the loss relative to the uploaded source, as seen on a specific phone. So we first built a benchmark called TSVQ. We took 690 sources from TikTok uploads, transcoded them into 12,600 versions, and collected about 377 thousand ratings from 48 raters on 25 phones, showing each transcode side by side with its upload and logging the device for every rating. The data showed two gaps. First of all, the quality degradation itself is largely invisible to these models, and there is a divergence between human scores and MLLM scores. Second, these models have never been taught to place the perceptual loss in its context.
>
> Guided by that, we built MOCA-VQA with two parts. The first, a motion-aware quality scorer, rests on two ideas. One is to show the model low-level patches cropped at native resolution, not only a downscaled frame, so the compression artifacts are actually visible. The other is to fuse each patch with the motion at the same location, so the model can tell whether an artifact sits in fast motion, where people barely notice it, or in a still region, where they do. Then a small context-aware rectifier adjusts the scorer's output using content features plus the source resolution, the encoding settings and the screen resolution, all of which come from metadata.
>
> The scorer improved SROCC on TSVQ from 0.850 for the best baseline to 0.884, and reached 0.913 on the public KVQ dataset, with statistically significant gains. The rectifier improved screen-tier accuracy for all six base scorers we tried.

**Implementation details.**

| Part | What we used |
|---|---|
| Benchmark (TSVQ) | 690 sources selected from about 120,000 TikTok uploads, 22 genres, five native resolutions (540p to 4K), balanced on spatial and temporal complexity; transcoded at 480p to 4K, never above the source, CRF 22–34, with different codecs: 12,600 transcodes. DSIS per ITU-R BT.500, side by side on 25 phones in three screen tiers; 48 raters, 377,050 ratings, turned into DMOS. |
| Scorer (MQS) | Initialized from DeQA (the DeQA-Mix3 checkpoint), a 7–8B-class MLLM; LoRA on the LLM decoder, with the vision encoder and projector trained jointly. Motion encoder: SlowFast. Patches on a FAST-VQA-style grid at native resolution. Same-location fusion with learned gating, then pooling. Score = expected value over excellent, good, fair, poor, bad (5 to 1). |
| Training | 3 epochs on 4 GPUs, AdamW, learning rate 2e-5 with cosine decay, effective batch 128; about 6.5 h on TSVQ. |
| Rectifier (CSR) | A small MLP on PCA-compressed MQS visual tokens and the base score, plus source resolution, transcoded resolution, CRF and screen resolution; corrections are centered across screen tiers; loss = correction fit + within-tier ranking + a penalty on large corrections; trained with MQS frozen. |
| Baselines | Q-Align, DeQA, CP-LLM, LMM-VQA, VQA2, Q-Insight, VQ-Insight and FAST-VQA, all retrained on each dataset; for context, the device-aware adapter of Safonov and Vatolin. |
| Datasets and metrics | TSVQ, KVQ and KoNViD-1k; SROCC, PLCC, KROCC and RMSE, with 95% confidence intervals for the main table. |

**Questions asked in interviews.** ByteDance.

1. **Q: Why not use a closed-source model, such as GPT-4o or Gemini, to replace the human raters?** A: Because in TSVQ the human ratings are the measurement, not a cost to cut. The benchmark exists to capture how people perceive transcoding loss, so human DMOS is the ground truth any scorer, a closed MLLM included, is validated against; if GPT-4o wrote the labels, the benchmark would only measure agreement with GPT-4o, which is circular. MOCA-VQA replaces raters at serving time, but is trained and judged on them. Second, general MLLMs are weak at this signal: they see sparse, resized frames, Gemini by default about one per second, and on Q-Bench-Video even GPT-4o and Gemini clearly trailed humans. In our data, under heavier compression, even DeQA, an MLLM tuned for quality, scores high-motion clips lower while human ratings stay largely independent of motion, consistent with motion masking; that is why our scorer fuses native-resolution patches with motion. Third, ratings depend on context: each transcode was rated side by side with its upload, DSIS per BT.500, on 25 phones in three screen tiers, and display resolution had the strongest device effect. A model can be given the screen resolution, as our rectifier is, but that effect must be measured on people. Practically, vendors retire model versions, so labels could not be regenerated for a benchmark we plan to release, and user uploads should not go to a third-party API [confirm the policy]. Budget is the weakest reason: one API pass over 12,600 clips costs far less than 377 thousand human ratings. Closed models can help around the humans: tagging genres for source selection, or pseudo-labeling beyond the human set, calibrated on a human-rated subset.
   *In the interview I said* we had no budget for it and humans are better anyway; keep "humans are better" as the conclusion, but lead with circularity and the DeQA motion evidence, and do not lead with budget: we hired a vendor annotation team for 377 thousand ratings, so an interviewer can turn that around.

**Questions asked in interviews.** ByteDance, second round.

1. **Q: Why use an MLLM instead of an objective metric such as PSNR?** A: PSNR measures pixel fidelity to a reference, and our problem breaks both of its assumptions. First, perception depends on context that pixel error ignores: an artifact in fast motion is barely noticed, and in our data the display changed how much a resolution step helped, with the largest gain when the video reached the screen's resolution. Second, in UGC the reference is not pristine: the upload is already compressed, so fidelity to it does not say whether the result looks good, and viewers rated a 720p transcode more strictly when its source was 1080p or 2K. Third, at serving time we wanted a no-reference scorer: the rectifier needs only the source metadata that production already logs, not the source frames. An MLLM brings semantic priors that generalize across content, which hand-crafted and regression scorers lack, and our scorer feeds it native-resolution patches and motion so it can actually see the artifacts. The fair comparison would be full-reference perceptual metrics such as VMAF. We did not report them in the paper, so I would add them as baselines.
2. **Q: In a multimodal model, what does one token represent? Follow-up: how is a patch size such as 32×32 chosen?** A: For an image, a token is the embedding of one patch: the encoder cuts the image into P×P patches, projects each to the model width d and adds a position embedding; after the encoder each token also carries context from the whole image. An MLLM's projector maps these tokens into the language model's embedding space, often merging them first: MobileVLM V2 pools 2×2, and Qwen2-VL merges each 2×2 group of 14×14 patches, so one LLM token covers 28×28 pixels; in video a token can span 2 frames. A token need not be a patch: TiTok, which we used in AquaScope, describes a whole image with 64 tokens. The patch size is a hyperparameter that trades compute against detail. The token count is (H/P)×(W/P): at 224 pixels, P = 16 gives 196 tokens and P = 32 gives 49; attention grows with its square, and in an MLLM every visual token also lengthens the prefill. Smaller patches see finer detail: ViT-B/16 is clearly more accurate than ViT-B/32 at about four times the compute, and OCR or small objects need small patches or high resolution. In practice P comes with the pretrained encoder, 14 for CLIP ViT-L/14 and SigLIP-So400m, and the knobs we actually turn are the input resolution and how tokens are merged. Width is set by model size, not by P: ViT-B/16 and ViT-B/32 both use d = 768, so a 32×32 patch, 3,072 values, is compressed 4× by the first layer, a cost of large patches rather than a rule. Content-adaptive sizes exist but are not standard: like HEVC splitting 64×64 blocks only where there is detail, ElasticTok, which we used in GenStream, gives complex content more tokens. Finally, the task decides the granularity. Compression artifacts are tiny and resizing erases them, so our VQA scorer uses two granularities: the resized frame for semantics, plus native-resolution patches on a FAST-VQA-style grid [fill in the patch size and count]. Adding the patches raised SROCC from 0.850 to 0.871; patches alone fell to 0.801.
   *In the interview I said* a token is a patch, and that the patch size depends on the content, larger patches when there is more redundancy, and on the embedding width, larger patches needing more dimensions. Both are refinements, not the main reason: lead with compute against detail, say that P usually comes with the pretrained encoder, then bring in adaptive tokenization and the compression in the first layer.

**Numbers to keep consistent with the resume.** The resume says "state-of-the-art on the new benchmark and three public datasets"; the paper evaluates two public datasets, KVQ and KoNViD-1k, and on KoNViD-1k the scorer ranks fourth. Say: significant gains on TSVQ and KVQ, competitive on KoNViD-1k.

---

# 7. TikTok — Scalable Ad Video Insertion

*In preparation. Similarity matching → insertable surface detection (SAM3) → composition. Adaptive frame sampling 8.5× faster; SFT'd Qwen3-VL embedding for brand-conflict and context-similarity checks.*

**90-second pitch**
> Automatically placing an ad into a UGC video is three problems: find videos where a placement is plausible, find a surface in the frame that can hold it, and composite it so it doesn't look pasted on. The expensive part is segmentation, so the systems contribution is adaptive frame sampling — run the detector only when the scene has actually changed enough to matter, and propagate in between, which gave 8.5× over the vanilla per-frame approach. The quality guardrail is an embedding model fine-tuned to catch brand conflicts and context mismatches, because a wrong placement is far more costly than a missed one.

**They will ask:**

| Question | Lead with |
|----------|-----------|
| How does adaptive sampling decide when to re-run? | The change signal and threshold, plus the bound: a maximum gap between forced full detections, so a missed fast change can't propagate indefinitely. |
| Could you use compressed-domain signals? | **Yes — motion vectors and residual energy are already in the bitstream, free.** Even if you didn't do it, saying it shows the two halves of your background connecting. ([03](02_ml_domain.md#6-generative-models--learned-compression) §5.3) |
| What breaks the 8.5× claim? | High-motion content where every frame is a change. Say what the worst case degrades to. |
| How did you fine-tune the embedding model? | InfoNCE, and **how you mined hard negatives** — that's the real lever. ([02](02_ml_domain.md) §1.2.5.5) |
| How do you evaluate a *blocking* decision, not retrieval? | Not Recall@k. Precision at the operating threshold, with the false-positive/false-negative cost asymmetry stated. This reframing is the good answer. |
| How do you know the composite looks right? | Human eval, and the temporal-consistency question — does the inserted surface flicker or drift? |

**Best fit**: archetype D (multimodal / video understanding) — video-search and video-intelligence teams (Twelve Labs, Hive, Coactive), platform ads/integrity teams, and creative-tools companies (Adobe, CapCut). The pitch here is *pipeline engineering under a cost constraint*, not model novelty — frame it that way and it's strong; frame it as research and it's thin.

---

# 8. Behavioral Bank

Meta and Netflix weight these heavily; everyone asks some. Write out 5–6 STAR stories and **say them out loud**:

1. **Conflict / disagreement** — you and a collaborator or advisor disagreed on direction. What was the evidence, how did it resolve? (Co-first-author papers are a natural source.)
2. **Failure** — something that didn't work. The best version is a research direction you killed, with the data that killed it.
3. **Changed your mind on data** — you believed X, measured, and switched. Interviewers love this; it's the hardest to fake.
4. **Shipped under a deadline** — the internship deliverables, a submission deadline, what you cut and why.
5. **Ambiguous problem you scoped yourself** — you were handed a vague area and defined the problem. (Trinity's "what does latency tolerance actually depend on?" is ideal.)
6. **Cross-team / cross-discipline** — user studies, ByteDance production systems, working with people who didn't share your background.

**Netflix specifically**: they probe judgment and independence — "when did you make a call without asking permission, and was it right?"
**Meta specifically**: impact and speed — quantify, and be clear about what *you* did versus the team.

---

# 9. Questions to Ask Them

Signals more than your answers do. Pick from:

- What fraction of the team's time goes to codec/encoding work vs ML? Is that changing?
- Where do you see neural tools entering the production pipeline first — pre/post-processing, or inside the codec?
- How do you decide a quality metric is trustworthy enough to make ship decisions on?
- What's the current bottleneck — encoding compute, delivery cost, or quality?
- (Netflix) How do you handle a case where VMAF and human raters disagree?
- (Nvidia) How much of the Cosmos tokenizer roadmap is aimed at compression vs generation?
- (Apple) How do you decide what's worth the silicon vs what stays in software?
- (Meta/Google) How do you decide which videos get the expensive encode?

---

*Back to [readme.md](readme.md).*
