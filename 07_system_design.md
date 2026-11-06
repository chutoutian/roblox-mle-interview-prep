# Part VII — System Design

For the Video Streaming role, start with [CPU/GPU Video Processing](#4-nuro-example-cpugpu-video-processing), then work through [Low-Latency Vehicle Video](#5-worked-example-low-latency-vehicle-video). The first has interview-report support; the second is role-inspired practice. Webhook Delivery and Vehicle Metrics Aggregation provide other reported-topic examples. Each walkthrough starts with the user's goal and builds the design one component at a time. The optional reference is not a prerequisite. For a codec or ML role, rehearse [An ML-Based Video Codec](#7-apple-example-an-ml-based-video-codec), the Apple prompt, last.

**Evidence:** a [report posted March 10, 2026](https://www.1point3acres.com/bbs/thread-1167866-1-1.html), describing a **2025 Q1** interview experience, names **Webhook** and **vehicle metrics aggregation** as System Design topics. The user supplied its text. It combines phone and onsite rounds, so these are reported Nuro topics, **not verified phone-screen prompts**. The requirements, numbers, architectures, and explanations below are our worked examples, not the original questions or Nuro's production design.

**Video-processing evidence:** a separate [report posted October 1, 2024](https://www.1point3acres.com/bbs/thread-1089077-1-1.html) describes a **2024 Q3 technical phone interview**. Its [public preview](https://www.1point3acres.com/bbs/tag/nuro-2709-3.html) names CPU preprocessing, GPU processing, CPU postprocessing, one million videos within a week, and efficient resource use. This is a reported **batch-processing** question, not a verified live-streaming question or a confirmed question for this particular team. Section 4 separates those reported constraints from our practice assumptions.

<a id="1-how-to-answer-a-system-design-question"></a>

# 1. The Answer Shape

You do not need to start by learning a catalog of distributed systems. For each example, answer: **what goes in, what comes out, where it waits, what survives a crash, and what happens when we retry?**

```text
Clarify the request → Set useful limits → Define APIs and records
                                             ↓
Summarize trade-offs ← Test failures ← Connect the complete path
```

For a **45-minute interview**, leave about 5 minutes for introductions and 5 minutes for questions. Use the remaining **35 technical minutes** like this; the examples below follow the same order.

| Technical minutes | What to do | What you should have explained |
|---|---|---|
| 0–5 | Clarify scope | Who uses it? Which operations? What can wait? |
| 5–10 | Set limits and interfaces | A useful size estimate; inputs, outputs, identifiers |
| 10–20 | Build the complete path | Where one event goes and when it is safely accepted |
| 20–30 | Deep-dive into two risks | Usually retries, failures, ordering, or overload |
| 30–35 | Check and summarize | Failure behavior, measurements, and the main trade-off |

These are flexible allocations, not a script to recite. Follow the interviewer's priorities. This progression also follows the [Hello Interview delivery framework](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery): establish requirements and a complete initial design before targeted deep dives.

# 2. Nuro Example: Webhook Delivery

## 2.1 Start with the smallest example

Suppose a trip finishes. A partner wants its own server notified. Instead of asking our server every second, the partner registers a URL and **our server sends an HTTP request when the event happens**. That callback over HTTP is a webhook.

```text
Our service                         Partner's server
trip finishes
     │
     └── POST https://partner.example/events ──→ receives event
         {"id":"evt-17", "type":"trip.completed"}
     ←────────────────── HTTP 200 ───────────── accepts it
```

**Endpoint** means the receiving URL. **Payload** means the request's data, here JSON. **2xx** means a successful HTTP response; it does not automatically prove every downstream business action is finished. These basic delivery mechanics match the [Stripe webhook documentation](https://docs.stripe.com/webhooks).

For this design, the receiver promises: “I return 2xx after storing the event durably, and process it afterward.” **Durable** means the accepted record survives the failures covered by its storage contract, rather than existing only in a process's memory.

## 2.2 Clarify the problem — technical minutes 0–5

Ask: “Are we building the service that sends webhooks, or the application receiving them?” Here we build the **sender platform** and define the receiver's contract.

Our chosen requirements:

1. A customer registers an HTTPS endpoint and selected event types.
2. An internal service submits an event; we deliver it to matching subscriptions.
3. Temporary failures are retried. Customers can inspect delivery status.

Assume no ordering guarantee initially. Retries may produce duplicates. We keep trying within a **24-hour delivery window**, then expose an explicit failed status. A permanently unavailable endpoint cannot be guaranteed delivery. Exclude vehicle control commands and arbitrary user-code execution.

**Say:** “The central requirement is that an accepted event is not silently forgotten. Receivers can be slow or unavailable, so I would separate accepting an event from delivering it.”

## 2.3 Establish the scale and contract — minutes 5–10

Assume 1,000 events/s, an average of three matching endpoints, and a 1 KB payload. These are illustrative discussion numbers, not company measurements.

```text
Initial delivery attempts: 1,000 × 3 = 3,000/s
Outbound payload:         3,000 × 1 KB ≈ 3 MB/s, before retries
If one attempt averages 0.2 s: about 600 requests are in progress
```

The last estimate explains why a worker cannot send every request sequentially. Use bounded concurrent requests; a timeout increase also increases the capacity required.

| Interface / record | Meaning |
|---|---|
| `POST /subscriptions` | Authenticated customer supplies URL and event types; returns subscription ID |
| `POST /events` | Internal producer submits stable event ID, type, time, and payload |
| `GET /deliveries/{id}` | Returns pending, attempting, delivered, or failed and the latest error |
| Event | `event_id`, `customer_id`, `type`, `created_at`, immutable payload |
| Delivery | `event_id`, `subscription_id`, state, attempt count, next attempt time, lease token/expiry |

An **event** is one thing that happened. A **delivery** is the work of notifying one subscriber about it: one event can create three deliveries. Enforce a unique `(event_id, subscription_id)` pair. A producer retry keeps the same event ID; reuse with different payload is rejected. In this example, subscriptions are selected when the event is accepted, not reevaluated on every retry.

## 2.4 Add a durable queue — minutes 10–15

Start with a database-backed work queue. It is simply a table of deliveries that still need work; it need not be a separate message broker on the first diagram.

```text
Producer → Event API → ONE database transaction
                          ├─ store event evt-17
                          ├─ delivery to subscription A: pending
                          └─ delivery to subscription B: pending
                                │
                   commit succeeds → acknowledge producer
                                │
                 Workers claim due deliveries → send HTTPS POST
```

Why one **transaction**? It makes these database changes commit together or not at all. A crash must not leave a stored event with its delivery work missing. The API acknowledges only after the configured durable commit succeeds. The table needs an index for due work and bounded retention; benchmark database throughput before deciding to shard it.

A worker briefly claims a row, commits the claim, and performs HTTP **outside the database transaction**. Claiming must be atomic: a conditional database update or row lock lets only one worker change an eligible row to owned. Merely reading a pending row and updating it later is a race.

Its claim has a **lease**, a time-limited ownership marker. If the worker dies, another can retry after the lease expires. Completion updates must match the current lease token, so an old worker cannot overwrite a newer attempt's state. Leases protect queue state; they cannot undo an HTTP request already sent, so duplicates still need handling.

```text
PENDING → ATTEMPTING ── 2xx ───────────────→ DELIVERED
              │
              ├─ temporary failure → RETRY_WAIT → ATTEMPTING
              ├─ worker dies → lease expires → eligible again
              └─ permanent error / budget used → FAILED
```

If the originating business update lives in a different service, that producer also needs to avoid “trip committed, event never submitted.” It can record the business change and an **outbox row** in its own transaction, then have a background sender retry `POST /events`. An outbox is a durable list of notifications still to send; this addresses the dual-write failure described in [AWS's transactional outbox pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html).

## 2.5 Complete the path — minutes 15–20

```text
Customer ── register URL ──→ Subscription API ──→ Subscription records
                                                      │ matching
Internal producer ─ event ─→ Event API ──→ Events + Delivery queue
                                              │        ↑ status/retry
                                              ↓        │
                                         Delivery workers
                                              │ signed HTTPS POST
                                              ↓
                                      Customer's endpoint
                                              │
                                  Durable inbox → Business worker

Customer ── inspect delivery ──→ Status API ──→ Delivery records
```

Walk `evt-17` through it: authenticate the producer, validate the event, select subscriptions, commit the event and delivery rows, return acceptance, claim one due row, send a signed request, and record its result. The receiver stores `evt-17` before returning 2xx; its own worker performs the business action later.

**Say:** “A slow customer does not block event creation. Pending work is durable and visible. I would measure queue age so that a growing backlog is not hidden behind successful API responses.”

## 2.6 Deep dive: timeout does not mean nothing happened — minutes 20–25

```text
Sender                         Receiver
POST evt-17 ─────────────────→ stores evt-17 in durable inbox
            X ←── HTTP 200      response is lost
timeout
retry SAME evt-17 ────────────→ unique ID already exists: no second inbox item
             ←── HTTP 200
mark delivery successful
```

**Idempotency** means retrying the same logical operation does not create another effect. Duplicate requests are expected, not an exceptional bug. The inbox uses a database uniqueness constraint, not a race-prone “check then insert” in memory.

For the receiver's background processing, mark the inbox item processed and apply its local database changes **in the same transaction**. Marking it processed first risks losing the action after a crash; applying the action first risks repeating it. An external effect, such as calling another service, additionally needs that service's idempotency key or another durable handoff. An inbox alone cannot make arbitrary external side effects exactly once.

Keep duplicate-detection records for at least the allowed retry/replay horizon plus operational margin. A manual replay after that horizon needs an explicit policy, not an unlimited exactly-once claim.

## 2.7 Deep dive: isolate failures and authenticate delivery — minutes 25–30

**Retries.** Treat timeouts, 429, and selected 5xx responses as retryable. Choose exponential delays with random jitter, an attempt cap, and the 24-hour age limit; respect a valid `Retry-After` within those limits. Most other 4xx responses require customer correction. A failed delivery remains inspectable; replay preserves its logical event identity.

**Isolation.** Bound concurrent requests and request duration per endpoint and globally. A slow endpoint gets its own delayed backlog rather than occupying all worker slots. If accepted work arrives faster than storage or delivery can sustain, enforce customer quotas and reject new submissions before acknowledging them; never silently drop accepted work.

**Security.** Use HTTPS and a per-endpoint secret to authenticate a signature over the raw payload and a fresh attempt timestamp. The receiver checks both signature and timestamp. Rotate secrets safely. Validate endpoint destinations and enforce outbound network restrictions on every connection; do not follow redirects into internal networks. URL validation alone is insufficient against DNS changes.

Stripe's documented delivery behavior includes retries, duplicate events, no guaranteed event order, and timestamped signatures. Those are useful real-world constraints; our retry limits and architecture above are separate design choices. [Stripe delivery and security guidance](https://docs.stripe.com/webhooks#event-delivery-behaviors)

## 2.8 Close the design — minutes 30–35

| Failure to walk through | Expected result | Measurement |
|---|---|---|
| Worker dies after sending | Lease expiry permits retry; duplicate is safe | Expired leases, duplicate attempts |
| One endpoint stalls | Its concurrency is capped; healthy endpoints continue | Queue age and success rate per endpoint |
| Event database is unavailable | Do not acknowledge new events; producer retries | Acceptance errors, backlog age |
| Receiver is down beyond 24 hours | Visible failed delivery, not endless retries | Exhausted delivery count |

**Spoken recap:** “I would accept events into durable delivery records and let bounded workers send them asynchronously. The hard part is an uncertain HTTP outcome: a timeout may follow successful receipt, so retries preserve event IDs and receivers must deduplicate. Leases recover abandoned work, while endpoint-specific limits stop one customer from blocking others. I would validate queue age, delivery success, and failed attempts. This trades immediate delivery and ordering guarantees for isolated, recoverable delivery with explicit limits.”

If ordering is requested, first ask **ordering per what**: subscription, vehicle, or trip? Serializing one such key and using sequence numbers is a possible extension, but a failed item then delays later items for that key.

## 2.9 Interview script: what to say at each step

This script is for a **45-minute interview**: about 5 minutes of introductions, the 35 technical minutes below (following 2.2–2.8), and 5 for questions. Speak these in your own words. Every block ends where the diagram should stand at that moment, so an interruption never leaves you without something drawn. The spoken recap in 2.8 remains the 30-second version.

**Minutes 0–5 — clarify before drawing**

> “Before I draw anything: are we building the platform that sends webhooks, or an application that receives them? I will assume the sender, and I will define what we ask receivers to do. Do you need ordering, and are duplicates acceptable? Unless you say otherwise, I assume no ordering guarantee, duplicates are allowed, and we retry for up to 24 hours before exposing an explicit failed status. The central requirement is that an accepted event is never silently forgotten. Receivers can be slow or unavailable, so I will separate accepting an event from delivering it.”

**Minutes 5–10 — scale and interfaces**

> “Let me size it with illustrative numbers: 1,000 events per second, three matching endpoints each, 1 KB payloads. That is 3,000 delivery attempts per second and about 3 MB/s outbound before retries. If one attempt averages 200 ms, roughly 600 requests are in flight at any moment, so workers need bounded concurrency rather than sequential sends.
>
> The interfaces are `POST /subscriptions`, where a customer registers an HTTPS URL and event types; `POST /events`, where an internal producer submits an event with a stable event ID; and `GET /deliveries/{id}` for status. An event is one thing that happened. A delivery is the work of notifying one subscriber, so one event fans out into several delivery records, unique on `(event_id, subscription_id)`.”

**Minutes 10–15 — the durable queue**

> “The simplest version has the event API POST synchronously to each customer. That couples our producer to the slowest customer and loses work on a crash. So the event API does one database transaction: store the event and create one pending delivery row per matching subscription, then acknowledge the producer. I start with a database-backed work table rather than a broker, because I need to find due rows and change their state, and this load does not establish a broker requirement.
>
> Workers claim a due row with an atomic conditional update, commit the claim, and send the HTTP request outside the transaction. The claim carries a lease: if the worker dies, the lease expires and another worker retries. Completion must match the lease token so a stale worker cannot overwrite a newer attempt. A lease protects queue state; it cannot undo a request already sent, so duplicates still need handling on the receiving side.”

**Minutes 15–20 — complete the path and check in**

> “Here is the complete path. [Draw the 2.5 diagram.] Walking `evt-17` through it: the producer is authenticated, the event validated, subscriptions selected, event and delivery rows committed, and the producer gets a 202 before any HTTP is sent. A worker claims one due row, sends a signed request, and records the result. The receiver stores `evt-17` before returning 2xx and processes it later.
>
> Two properties matter: a slow customer never blocks event creation, and pending work is durable and inspectable. I would measure queue age so a growing backlog is not hidden behind successful API responses.”

Then hand the choice to the interviewer: “That is the complete path. I would like to go deeper on two risks: what a timeout actually means, and how one bad endpoint could hurt everyone else. Which would you like first?”

**Minutes 20–25 — deep dive: a timeout is not “nothing happened”**

> “A timeout does not tell me whether the receiver got the event; it may have stored it and only the response was lost. So I retry with the same event ID, and the receiver contract is: insert into a durable inbox with a uniqueness constraint on `event_id`, return 2xx, and process in the background. Deduplication must be a database constraint, not a check-then-insert in memory. For the receiver's own processing, marking the inbox item processed and applying its local changes belong in one transaction. I am not claiming exactly-once for arbitrary external side effects; those need their own idempotency keys, and the dedup records must outlive the retry horizon.”

**Minutes 25–30 — deep dive: isolation, retries, and security**

> “Retries: timeouts, 429, and selected 5xx are retryable with exponential backoff and jitter, an attempt cap, and the 24-hour age limit; I honor a valid `Retry-After`. Most other 4xx mean the customer must fix something, so I stop and make the failure visible.
>
> Isolation: I bound concurrent requests per endpoint and globally, so a stalled endpoint accumulates its own delayed backlog instead of occupying every worker slot. If submissions outrun capacity, I enforce quotas and reject before acknowledging; accepted work is never dropped silently.
>
> Security: HTTPS, a per-endpoint secret that signs the raw payload plus a fresh timestamp, safe rotation, and outbound network restrictions so a customer-supplied URL cannot reach our internal network. Validating the URL string alone is not enough, because DNS can change.”

**Minutes 30–35 — close**

Give the spoken recap from 2.8, then name the trade-off: immediate delivery and ordering are traded for isolated, recoverable delivery with explicit limits. If ordering comes up: “Ordering per what: subscription, vehicle, or trip? Serializing one key with sequence numbers works, but a failed item then delays everything behind it, so I would only add it for a stated need.”

**Check-in lines to reuse**

- “Does that match the scope you had in mind, or is the receiving side more interesting to you?”
- “I can go deeper on the lease and claim mechanics or on the receiver's inbox; which is more useful?”
- “Before I add anything else: is there a constraint you would like to tighten?”

# 3. Nuro Example: Vehicle Metrics Aggregation

## 3.1 Start with four measurements

A **metric** is a named measurement, such as CPU utilization or camera frame rate. A **sample** is one measurement at a particular time. **Aggregation** combines samples into a summary.

```text
Vehicle v7 / cpu_utilization_percent / minute 12:00

12:00:05  20 ─┐
12:00:15  40  │
12:00:25  30  ├──→ sum = 140, count = 4 → mean = 35%
12:00:35  50 ─┘

Raw samples → one per-minute record → dashboard point
```

Here the mean is **sample-weighted**. If samples arrive at irregular intervals, a time-weighted mean is a different calculation; confirm which the user wants. We start with regularly sampled gauges, not cumulative counters or arbitrary percentiles.

## 3.2 Clarify the request — technical minutes 0–5

Our worked prompt: “Collect numeric measurements from a vehicle fleet and let engineers query one-minute summaries by vehicle and metric.” Ask which metrics matter, how fresh the dashboard must be, how long vehicles can be offline, and whether historical completeness matters more than immediate results.

Choose: connected-vehicle data normally becomes visible within 10 seconds; vehicles can spool up to one hour offline; raw data is retained seven days and minute summaries 30 days. Recent summaries may be revised when delayed samples arrive. Show unavailable/stale data explicitly instead of treating missing samples as zero. Exclude video payloads, vehicle actuation, and safety-critical alerting.

**Say:** “I would separate fresh dashboard results from complete historical results. Cellular outages mean a current dashboard cannot always include every vehicle.”

## 3.3 Set the scale and records — minutes 5–10

Assume 1,000 vehicles, 20 metrics each, one sample per second, and an illustrative 100 serialized bytes per sample. Use decimal units.

```text
Samples:      1,000 × 20 × 1/s = 20,000 samples/s
Payload:      20,000 × 100 B   = 2 MB/s ≈ 173 GB/day
One vehicle: 20 × 100 B × 3,600 s = 7.2 MB for one offline hour
```

Storage needs additional space for indexes, replication, and encoding overhead. Batch multiple samples per upload to reduce request overhead. A **batch** is only a transport envelope; its samples still have separate identities.

| Record / API | Important fields or behavior |
|---|---|
| Sample | `vehicle_id`, `boot_id`, `sequence`, `event_time`, metric, value, unit, permitted labels |
| `POST /metrics/batches` | Authenticated vehicle submits samples; returns accepted batch ID after durable storage |
| Minute summary | Key: vehicle, metric, approved label set, minute; values: sum, count, min, max, version |
| `GET /metrics?vehicle=v7&metric=cpu&start=...&end=...` | Returns authorized, bounded time-series results with freshness and coverage |

`boot_id` changes on restart, while `sequence` is unique across samples within that boot. Together with vehicle identity, they form a stable sample ID. Retry exactly the same records; never assign fresh IDs merely because an upload timed out.

## 3.4 Module A: get a sample safely off the vehicle — minutes 10–13

```text
Measurement → Local durable spool → Upload batch → Cloud ingest
                    ↑                  │                │
                    └── keep on error ─┘        durable append succeeds
                                                       │
                              delete acknowledged batch ← ACK
```

A **spool** is data waiting on local storage. The vehicle deletes a batch only after acknowledgment that the cloud has durably accepted it, not after the network send completes. The cloud validates identity, size, timestamp, and schema before accepting the entire batch; a timeout may require resending it.

The server's durable log is an append-only sequence that workers can read and replay. An **offset** identifies progress in a log partition. An ACK means stored, **not yet aggregated or displayed**. Define replication and acknowledgment settings to survive the failures you promise; merely naming a log product does not establish durability.

Set disk and upload limits. If an outage exceeds the spool budget, the product must choose which data may be discarded or downsampled and report gaps. Pause low-priority catch-up traffic when it interferes with live operational traffic.

## 3.5 Module B: turn samples into a correct summary — minutes 13–17

Group by `(vehicle, metric, labels, minute)`. The minute comes from the sample's event timestamp: `12:00:59` belongs to `[12:00, 12:01)`. Keep sum and count rather than only the average.

```text
Partial A: [20, 40] → sum 60, count 2, mean 30
Partial B: [90]     → sum 90, count 1, mean 90

Combined mean = (60 + 90) / (2 + 1) = 50
WRONG: (30 + 90) / 2 = 60
```

A **partition** assigns different subsets of records to workers. Start with a stable hash of vehicle ID so one vehicle's data goes to one processing partition; repartition by a finer key if measured imbalance requires it. Parallelism does not remove the need to coordinate worker ownership and recovery.

For a straightforward correctness-first design, a worker transaction inserts each unseen sample ID into a deduplication table and updates the corresponding aggregate. The unique sample ID ensures one effect even on replay. Advance the log offset only after the transaction commits. If the worker crashes before advancing it, replay is harmless. Batch these transactions; benchmark this database write load before assuming it meets the target.

Store raw records for later repair as a separate log consumer. Duplicate raw deliveries retain the stable sample ID so a historical rebuild can deduplicate them too. “We use a log” by itself is not an end-to-end exactly-once guarantee.

## 3.6 Connect the full path — minutes 17–20

```text
Vehicle measurements → Local spool → Batched upload
                                         │
                                         ↓
                                  Auth + Ingest API
                                         │ durable append, then ACK
                                         ↓
                                  Partitioned durable log
                                    │                 │
                                    ↓                 ↓
                            Aggregation workers   Raw-data writer
                                    │                 │
                             Transactional DB     Object storage
                            dedup + minute rows   replay / repair
                                    ↑
Engineer → Dashboard → Query API ────┘
```

The query API reads compact minute rows, not every raw sample for every refresh. It can combine their sums and counts for a larger range. Return coverage, latest sample time, and summary version so a partial result does not look complete. Use authorization on both vehicle uploads and engineer queries.

**Say:** “The log separates receiving data from processing it. The summary database serves fast dashboard queries, while raw storage lets us repair or recompute results later. They have different jobs.”

## 3.7 Deep dive: event time and late data — minutes 20–25

**Event time** is when the measurement happened. **Processing time** is when a server handles it. These differ when a vehicle is offline.

```text
12:00:30  vehicle measures value 50; network unavailable
12:01:10  dashboard shows the 12:00 minute from samples received so far
12:03:00  delayed value 50 arrives
          update the 12:00 minute, NOT the 12:03 minute
```

Choose a policy: publish provisional summaries promptly and accept historical corrections within a **two-hour event-time horizon**, covering the assumed one-hour outage plus processing margin. Reject implausible future timestamps; record clock quality because a wrong vehicle clock puts data in the wrong minute.

A **watermark** estimates event-time progress: roughly, “we expect most data older than this point to have arrived.” It is not proof that a disconnected vehicle has nothing left to send. If a stream processor uses watermarks, configure lateness and idle-source handling explicitly. These distinctions are explained in the [Apache Beam guide to watermarks and late data](https://beam.apache.org/documentation/programming-guide/#watermarks-and-late-data).

Our simple database design does not require implementing a watermark engine. It can update eligible historical rows directly. Samples outside the two-hour horizon go to raw storage and a repair path; they do not silently reopen expired online state. A window being closed means “closed under our policy,” not “all vehicles definitely reported.”

## 3.8 Deep dive: retries, restarts, and catch-up — minutes 25–30

```text
Worker reads sample v7 / boot9 / seq42
      │
      ├─ transaction: insert ID + add value to minute → COMMIT
      X crashes before saving log progress
      │
Replacement rereads seq42 → ID exists → do not add it again
                         → commit progress
```

Retain deduplication IDs for at least the supported online late/retry/replay horizon plus margin. With our two-hour online acceptance policy, keeping IDs for 24 hours is an illustrative bounded choice; older replays are routed to a separate deduplicated rebuild, not blindly reapplied after IDs expire. Cleanup must be coordinated with recovery and timestamp validation.

Check that this bound is affordable: `20,000 samples/s × 86,400 s = 1.728 billion IDs` per day, before index and replication overhead. That estimate may justify partitioning storage or changing the supported replay/state policy; do not assume one database node can sustain the proposed retention and write load.

If aggregation is down, the durable log buffers within a **finite retention budget**. Track the oldest unprocessed record and provision catch-up capacity before that budget expires. Vehicles also need rate-limited, jittered reconnects: 1,000 vehicles each sending an hour of history at once can overwhelm the normal ingest rate. Reserve capacity for new data while draining old batches.

Keep metric names and labels controlled. A label is a dimension such as camera ID or software version; allowing arbitrary request IDs as labels creates a new time series per request and can explode storage. This is the cardinality pitfall described in [Prometheus's metric and label guidance](https://prometheus.io/docs/practices/naming/#labels).

## 3.9 Close the design — minutes 30–35

| Failure to walk through | Expected result | Measurement |
|---|---|---|
| Upload ACK is lost | Retry stable sample IDs; no duplicate aggregate contribution | Retry and deduplication counts |
| Vehicle stays offline | Bounded local spool; dashboard marks missing coverage | Last-seen age, disk runway, gaps |
| Aggregator crashes after commit | Replay skips IDs already applied | Consumer lag, recovery time |
| Very old or bad-clock sample arrives | Quarantine/repair policy, not silent current-minute insertion | Late and invalid-timestamp counts |

**Spoken recap:** “I would collect timestamped samples into a bounded vehicle spool and acknowledge uploads only after durable cloud acceptance. Partitioned workers create minute summaries using sum and count, with stable IDs preventing double counting on retries. Event time determines the correct minute; delayed data revises provisional history within an explicit horizon. Raw storage supports later repair, and the dashboard exposes freshness and missing coverage. My main trade-off is fresh partial results versus waiting for completeness, which intermittent connectivity cannot guarantee immediately.”

If asked for fleet-wide averages, clarify **sample-weighted versus equal-weight-per-vehicle** before merging. If asked for p95, sum/count is insufficient: introduce mergeable histograms or sketches with an explicit accuracy contract only then.

# 4. Nuro Example: CPU/GPU Video Processing

**Reported prompt, paraphrased:** design a system that takes one million videos through CPU preprocessing, GPU processing, and CPU postprocessing within one week, using resources efficiently. [Original phone-screen report](https://www.1point3acres.com/bbs/thread-1089077-1-1.html) · [Public prompt preview](https://www.1point3acres.com/bbs/tag/nuro-2709-3.html)

There is no single official or universally correct answer. The following is an **interview-ready reference design**, built like the webhook example: start small, explain each component, then discuss the two hardest issues. The timings, file sizes, algorithms, APIs, and failure policies below are **practice assumptions**, not additional facts from the report.

<a id="41-start-with-one-video-and-six-basic-concepts"></a>

## 4.1 Start with the user's goal and the big picture

Imagine an engineer who already has one million recorded videos saved in shared storage. The engineer wants to **choose the videos, start processing, check progress, and get the results**. They should not have to pick a machine or manually start each processing step.

For an illustrative application, suppose the result is object detections for each video. The processing itself is:

```text
Stored video
    ↓
CPU: decode / sample / resize frames
    ↓
GPU: run the chosen model
    ↓
CPU: combine detections and write results
```

But a **processing algorithm** is not yet the whole **system**. We also need a way to request the work and find out what happened:

```text
Engineer / submitting program
    │ "Process these videos"
    ▼
VIDEO-PROCESSING SYSTEM
    ├─ remembers what was requested and its progress
    ├─ assigns work to CPU → GPU → CPU workers
    └─ saves the finished results
    │
    └─ progress and result locations → Engineer / program
```

The recorded videos are already stored; submitting a request does **not** mean sending all their bytes again. Workers read the files when they need them.

For now, keep just three terms: a **job** is the work for one video, a **stage** is one processing step, and a **worker** is a program doing that step. A **worker pool** is a bounded group of those programs, possibly running on several machines. The remaining components will be introduced when a concrete need appears.

One video's stages must run in order. Different videos can move through the system together: CPU workers prepare video B while a GPU worker processes video A. Your Python scheduler's **waiting work + available workers** is a useful starting point.

## 4.2 The 60-minute interview plan

These are **wall-clock minutes from the start of the meeting**, including introductions and your questions. Adjust if the interviewer gives you a full hour of technical discussion. Draw a working end-to-end design before spending time on optimizations.

| Minutes | What you do | What you say or leave on the board |
|---|---|---|
| 0–3 | Introduce yourself and restate the problem | “I will clarify the workload, estimate capacity, then design the pipeline and failure recovery.” |
| 3–10 | Ask the essential requirements questions | Batch versus live, deadline, inputs/outputs, workload distribution, resource limits, failure policy |
| 10–17 | Do the capacity calculation | Videos/s; CPU cores and GPUs from measured service demand; storage/network estimate |
| 17–22 | Explain the high-level user journey | Choose videos → submit → receive a tracking ID → check progress → get results |
| 22–28 | Add components only when needed | Request handler, progress records, waiting work, CPU/GPU workers, shared file storage |
| 28–33 | Name the interfaces and records | Translate the working picture into API calls, batches, manifests, and a simple data model |
| 33–43 | Deep dive 1: failures and retries | Lease, crash recovery, duplicate execution, one accepted stage result |
| 43–51 | Deep dive 2: GPU efficiency and overload | Bottlenecks, batching, bounded prefetch, backpressure, deadline-aware scaling |
| 51–55 | Validate and summarize | Failure tests, metrics, security, cost and the key trade-off |
| 55–60 | Answer remaining follow-ups and ask your questions | “Which constraint would you like to tighten?” Then ask about the team's workloads |

This is a discussion, not a 60-minute monologue. After the first diagram, ask: **“Would you like me to go deeper on recovery or resource utilization first?”** The follow-ups in 4.13 are a menu, not a list to recite.

If the actual interview is **45 minutes**, preserve scope, a complete diagram, the capacity calculation, and one failure walkthrough. Shorten the API details and choose one deep dive; leave about five minutes for questions. The structure is consistent with the [Hello Interview delivery framework](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery), but these allocations are our practice plan.

## 4.3 Clarify the requirements — minutes 3–10

Ask questions whose answers change the architecture:

| Ask the interviewer | Why it matters | Assumption if no answer is supplied |
|---|---|---|
| “Are these stored videos or live streams?” | A weekly completion target differs from a per-frame latency target | Stored videos; offline batch processing |
| “Are all one million available initially, or arriving throughout the week?” | A batch deadline requires finishing the last jobs, not just matching an average arrival rate | A fixed collection is available at the start; inputs are already in shared file storage |
| “What are typical and maximum duration, resolution, size, and stage runtimes?” | One long 4K video may cost far more than one short 720p clip | Use measured representative service demand; numbers below are illustrative |
| “What does the GPU step do, and what output is required?” | Inference, transcoding, and temporal models have different batching/chunking rules | A fixed inference pipeline; saved detection results for each video |
| “Is the hardware budget fixed? May we add GPUs?” | Determines whether the deadline is feasible | Resources may scale within an agreed cost cap |
| “Must every valid video succeed? What about corrupt input?” | A terminal failure is not a successful result | Every valid video needs a result; invalid inputs need explicit diagnosis and agreement |

**Functional requirements** describe what the user can do: choose a collection of videos to process, start processing, inspect progress/errors, and retrieve results.

**Non-functional requirements:** meet the batch deadline under agreed workload/resource assumptions, recover from worker failures without silently losing accepted work, and control compute/storage cost. Do not promise a deadline for arbitrary input and unlimited failures.

**Out of scope initially:** live playback, CDN delivery, vehicle control, arbitrary user-supplied code, and operating the processing system across multiple regions. Start in one region; discuss stronger disaster-recovery requirements if asked.

**Say:** “I am treating this as a throughput-oriented batch pipeline. I will preserve each video's stage order, but I do not need to preserve order across different videos.”

## 4.4 Estimate capacity and check feasibility — minutes 10–17

The reported volume gives a useful lower-bound average completion rate:

```text
One week = 7 × 24 × 3,600 = 604,800 seconds
Required average = 1,000,000 / 604,800 ≈ 1.65 videos/second
```

That rate alone does **not** tell us how many GPUs we need. Ask for measurements on representative inputs and the intended hardware/model/batch configuration.

For a worked example, assume these average service demands:

| Stage | Assumed work per video | Capacity at a 70% planning load |
|---|---|---|
| CPU preprocessing | 20 CPU-core-seconds | `ceil(1.6534 × 20 / 0.70) = 48 CPU cores` |
| GPU processing | 60 GPU-device-seconds | `ceil(1.6534 × 60 / 0.70) = 142 GPUs` |
| CPU postprocessing | 10 CPU-core-seconds | `ceil(1.6534 × 10 / 0.70) = 24 CPU cores` |

```text
Resource units ≈ required videos/s × resource-seconds/video
                 ─────────────────────────────────────────
                            planning load
```

**Units matter.** Twenty CPU-core-seconds could mean one core for 20 seconds or four cores for five seconds; it does not mean 20 servers. GPU-device-seconds must reflect the selected batching configuration: a batch of ten videos occupying one GPU for 60 seconds costs six GPU-device-seconds per video. Benchmark, rather than infer performance from a GPU model name.

The 70% factor is **chosen planning headroom**, not a promised hardware-utilization measurement. The figures are a first capacity estimate, not a guaranteed deadline. Include startup, retries, transfers, long-video tails, and the final pipeline drain in an end-to-end benchmark and forecast.

If hardware is fixed, the ideal bottleneck rate is:

```text
min(preprocess cores / 20, GPUs / 60, postprocess cores / 10)
```

For example, **20 GPUs** at the assumed service demand can process at most **201,600 videos in a week**, even before overhead. More CPU workers cannot fix that. Negotiate more capacity, cheaper processing, or a longer deadline; do not silently change output quality.

**Storage and bandwidth:** with an assumed 100 MB input per video, the input corpus is **100 TB**, and reading each input once over a week averages roughly **165 MB/s**. The inputs are already uploaded in this scenario; upload capacity is an additional concern if that assumption changes. Intermediate writes, intermediate reads, results, retries, replication, and retention all add to this budget. Decoded frames can be much larger than compressed inputs.

**Say:** “I need about 1.65 completed videos per second on average. I would convert measured stage costs into CPU/GPU capacity and check storage throughput before deciding whether the deadline is achievable.”

**Transition to the design:** “This estimate tells me how much processing capacity I need. Now I need to explain how the engineer submits work, how workers know what to process, and how we track progress and results.”

<a id="45-define-the-interfaces-and-durable-records--minutes-1722"></a>

## 4.5 Explain the high-level user journey — minutes 17–22

Start with the engineer's experience, not HTTP syntax. Think of these as operations a Python program might offer:

```text
1. start_processing(list_of_videos) → tracking_id
2. check_progress(tracking_id)      → "300 finished; others waiting/running"
3. get_results(tracking_id)         → locations of completed result files
```

These are **conceptual operations**, not an implementation or a required API format.

Processing a large collection takes time. The first operation should return a tracking ID after the request is safely recorded, **not wait until all videos finish**. The engineer's program can continue, then check later. This is **asynchronous processing**: accepting the request and finishing the work are separate events.

```text
Engineer uses a script                     Processing system
          │                                       │
          ├── "Process videos A, B, C" ───────────→│
          │←── "Accepted; tracking ID = 42" ───────┤
          │                                       │ CPU → GPU → CPU
          │ script can do something else          │ work continues
          ├── "How is request 42 doing?" ────────→│
          │←── "A finished; B and C still running" ┤
          ├── "Where is A's result?" ────────────→│
          │←── "Read this saved result file" ──────┤
```

**Who is the client?** In a diagram, the **client** is that submitting script or application. It is not necessarily a browser, a vehicle, or a separate machine. An engineer uses the client to ask the processing system to do work.

This immediately tells us that the system must (1) accept requests, (2) remember progress, (3) execute work, and (4) save results. Those needs explain the boxes in the next diagram.

**Say:** “The user submits the list of video locations and receives a tracking ID. Background workers do the processing, while the user can check progress and retrieve completed results.”

<a id="46-draw-the-complete-architecture--minutes-2228"></a>

## 4.6 Build the architecture one need at a time — minutes 22–28

### A. Someone must receive the request

Add a **request handler**: a program that receives “process these videos” or “show my progress.” It checks the request, records it, and replies. It does not run hours of GPU computation inside the request.

For a small internal tool, this could be part of a command-line program. For several remote callers, we can expose it as a network service. **A separate public web service is a design choice, not a requirement implied by video processing.**

### B. We must remember what was requested and what finished

If progress lives only in the handler's memory, restarting it loses that information. Save small progress records in a **database**, a storage system for structured records:

```text
Video A | preprocessing finished | GPU waiting | no final result yet
Video B | all stages finished    | done        | results/video-B.json
```

These records are called **metadata**: information *about* the work and files, not the video bytes themselves. **Durable** means the saved record survives the failures covered by the storage system's contract; it is not just a variable in one process.

### C. More videos are waiting than workers can process at once

Add a **queue**, or waiting list, for each stage. A free worker takes suitable work from its waiting list. After preprocessing finishes, GPU processing for that video becomes eligible; the next CPU stage waits until the GPU stage finishes.

```text
Preprocessing waiting list → CPU workers
                                  │ completion makes GPU work ready
                                  ▼
GPU waiting list ──────────→ GPU workers
                                  │ completion makes final CPU work ready
                                  ▼
Postprocessing waiting list → CPU workers → finished result
```

This resembles your Python `queue.Queue` and worker pool, except an in-memory queue disappears when its process dies. For this design, save waiting tasks in the database too. This is a **database-backed queue**. “Queue” describes its job; it does not automatically mean another product or server.

The three queues can be groups of rows in one table, selected by stage and status. The three worker pools can be processes on one or many machines; they are not three mandatory microservices.

### D. Workers need access to the actual files

Keep videos and saved results in **shared file storage** that workers can read and write. **Object storage** is one common implementation: save a file under a key, then retrieve it by its address.

Workers take a small task record, such as “process video A at this address,” and read the actual file from storage. They save intermediate outputs there and make the next task refer to that output. The request handler and queues do not need to forward the large video bytes.

### E. Put the pieces together

```text
Submitting script / application
          │ start work / check progress
          ▼
    Request handler
          │ save requests / read progress
          ▼
    Database: progress records + waiting tasks
          ↕ workers take tasks and update progress
    CPU preprocessing → GPU processing → CPU postprocessing
          ↕                 ↕                  ↕
    Shared file storage: videos, intermediate files, results

Check progress: script → handler → database → progress reply
Read results:   script → saved result location → result bytes
```

The CPU → GPU → CPU arrows represent the **order of work**, not direct transfers of the entire video between those boxes. The lower arrows show where file bytes move. We have two kinds of traffic: **small instructions/progress messages** and **large file reads/writes**.

Walk video A through it: record the request, put preprocessing on the waiting list, let a CPU worker prepare A, save its output, make GPU work ready, run the model, save that output, run postprocessing, and record the final result location. Another video can occupy a different stage at the same time.

**Say:** “The handler receives requests, the database remembers pending work and progress, and the workers do the processing. Large files live separately in shared storage. Now that the responsibilities are clear, I can define the precise interface.”

<a id="47-walk-video-17-through-the-system--minutes-2833"></a>

## 4.7 Translate the picture into APIs and records — minutes 28–33

### A. What is a batch?

A **submission batch** is the collection the user wants to process and track together. Batch 42 could contain videos A, B, and C; each video has its own job.

```text
Batch 42: one submitted collection
    ├─ Job 17: video A → CPU → GPU → CPU → result
    ├─ Job 18: video B → CPU → GPU → CPU → result
    └─ Job 19: video C → CPU → GPU → CPU → result
```

This is different from a **GPU batch**, a temporary group of model inputs computed together for efficiency. A million-video submission does not mean loading a million videos onto a GPU together.

### B. Why introduce a manifest?

For three videos, the caller can send their locations directly. For a million, a giant request is inconvenient. Save the list itself as a file:

```text
File: submissions/run-42.csv
video_id, input_location
A, inputs/video-A.mp4
B, inputs/video-B.mp4
C, inputs/video-C.mp4
... one row per input video
```

That list file is called a **manifest**. A **URI** is an address identifying a resource; here it identifies a stored file. The **manifest URI** is the address of the list, not a video and not the list's contents.

Instead of sending a million entries, the caller says: “Process the videos listed in this saved file.” A small background **loader** reads that list in chunks and registers each video's job and first waiting task. It remembers its position so it can resume after a restart.

The videos and list must be readable by the processing system. For this version, they stay **immutable**, meaning their contents do not change during the run.

### C. API, POST, and GET are names for the operations we already chose

An **API** is an agreed way for one program to ask another to do something. Our three conceptual operations already form an interface. **HTTP** is a protocol programs use to send requests and receive responses over a network. If the caller and handler use HTTP, we can express the operations like this:

| What the caller wants | Example HTTP request | Meaning |
|---|---|---|
| “Start processing this collection” | `POST /batches` | Submit a request to create a processing run |
| “How is collection 42 doing?” | `GET /batches/42` | Read its progress without starting new work |
| “List the videos and finished results in collection 42” | `GET /batches/42/jobs` | Return job IDs, statuses, and available result locations in small pages |
| “Where is video A's result?” | `GET /jobs/17` | Read that video's job status and, when ready, its result location |

The collection listing tells the caller that video A has job ID 17; the caller does not have to guess it. **Pagination** means returning a limited portion and a way to request the next portion, rather than a million records in one response.

**POST** is the HTTP method we choose here to submit new work. **GET** reads information. `/batches` and `/jobs/17` are endpoint paths we choose, like names for remotely callable operations. They are not database tables or folders of video files.

Return `202 Accepted` with a tracking ID only after saving the request durably. It means “the work was accepted,” **not** “the videos finished.” The caller can use GET later. These HTTP choices are one possible interface, not new processing stages.

### D. Translate the original diagram

```text
Client ── manifest URI ──→ Batch API ──→ Batch + Job metadata
```

It means:

> “The engineer's program sends the address of its saved video list to our request handler. The system records the collection and, as the loader reads the list, records the individual videos and their progress.”

“Batch API” is simply the request handler's batch-submission/status interface. **The API initially records the batch; the loader creates the jobs afterward.** The short arrow is not a claim that one request synchronously creates a million jobs.

### E. Decide what to remember

Start with the questions a record must answer:

| Record | Plain-English questions | Initial fields |
|---|---|---|
| Batch | Which collection is this? Where is its list? How far has registration progressed? | `batch_id, manifest_uri, loader_cursor, state` |
| Job | Which video? Which processing step? Where is its result? | `job_id, batch_id, input_uri, current_stage, state, result_uri` |
| Stage task | What work can a worker take, and where is its input? | `job_id, stage, state, input_uri, output_uri` |

`loader_cursor` means the loader's saved position in the list. IDs let the caller and workers refer to the same records unambiguously. The state tells us whether work is waiting, running, successful, or failed.

These are the core records, not the final production schema. Retry counters, lease tokens, request IDs, and processing-version fields will be added in the next section **because failures give us a reason to need them**.

**Say:** “POST and GET just express submit and read operations. A batch groups the user's videos, a manifest holds their locations, and metadata records their progress. None of those replaces the CPU/GPU processing path.”

## 4.8 Deep dive: failures, leases, and safe retries — minutes 33–43

We now have a complete working path. The next question is: **what happens if a request is repeated or a worker stops halfway through?** This is why we add the following bookkeeping, rather than putting every field on the first diagram.

A **database transaction** commits a group of record changes together, or commits none of them. The loader uses one transaction to create a chunk of jobs and their first tasks **and** advance its saved position. A crash cannot commit only the position and skip the work.

For repeated submissions, keep a stable `request_id`: the same request returns the same batch instead of starting a duplicate run. Reject reuse of the ID with different parameters. Unique `(batch_id, video_id)` job records make rereading a list chunk safe. Also save a `pipeline_version`, identifying the model and processing configuration, so retries do not silently switch to a different algorithm.

Each logical `(job_id, pipeline_version, stage)` has one task record. Add an attempt count, next retry time, latest error, and the ownership fields explained below. Several attempts can execute that one task.

A **lease** is temporary ownership, not a lock held for the whole computation. Workers send **heartbeats**, periodic progress messages, to renew it while making progress, subject to a task-runtime limit. An expired task becomes retryable. Use the database's time for lease decisions.

A **lease token** identifies the current attempt. Heartbeats and completion use a conditional database update requiring `state = RUNNING`, the expected token, and an unexpired lease according to database time. An expired attempt cannot revive its lease. Publish the output and create the next task only when that guarded completion succeeds, in the same transaction. This is fencing: an old worker cannot change the accepted state after a newer attempt takes ownership.

Claim work in a short database transaction, then process the video **outside** that transaction. PostgreSQL's `FOR UPDATE SKIP LOCKED` is one possible way for workers to avoid claiming the same currently available row; it is an implementation detail, not a box needed in the high-level diagram. [PostgreSQL queue-consumer guidance](https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE)

| Failure | What the design does |
|---|---|
| Worker crashes before uploading an output | Lease expires; retry that stage from its last published input |
| Upload succeeds, worker crashes before the DB commit | The uploaded object is unreferenced; retry is safe, and later cleanup can remove the orphan |
| Completion commits but the reply is lost | Reread stage state; the accepted output and next task already exist |
| Old worker resumes after its lease expires | Its completion is rejected; its immutable attempt output cannot overwrite the winner |
| Database is temporarily unavailable | Pause claims/publication; do not acknowledge new batches or declare uncommitted results successful |
| Input is corrupt or repeatedly fails | Record a visible terminal error after a bounded policy; do not retry forever |

**The key invariant:** publishing the accepted output URI, completing the current stage, and creating the next stage are one metadata transaction. Otherwise, “stage succeeded, next task never created” can silently strand a video.

The file upload itself is **not** in that database transaction. Upload first to an immutable attempt path, then commit its reference. This allows unused files after crashes but avoids publishing a pointer before the file exists. Garbage collection must exclude live attempts and published references and use a safe grace period.

Execution is **at least once**, not guaranteed exactly once. Duplicate attempts can compute twice; conditional publication and uniqueness select one accepted result. This assumes stages only produce versioned artifacts. Any additional external side effects need their own idempotency contract.

Use exponential retry delays with jitter, a retry budget, and an absolute task deadline. Classify errors: a temporary storage failure may recover; an unsupported codec or invalid file will not improve through identical retries. Only the former should be automatically retried unchanged.

If asked to add a broker, keep the database authoritative. Write an **outbox record in the same transaction** as stage completion, then relay it to the broker. Relay retries can duplicate messages, so consumers still claim the logical task conditionally. Do not update the database and publish a message as unrelated writes. [Transactional outbox pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html)

**Say:** “I do not promise that computation runs exactly once. I promise that attempts cannot publish conflicting accepted results or lose the next stage after a successful metadata commit.”

## 4.9 Deep dive: keep GPUs useful without unbounded queues — minutes 43–51

**Find the bottleneck first.** If GPUs are waiting for decoded inputs, add or optimize preprocessing and transfers. If the GPU queue grows while GPUs are busy, more CPU preprocessing is not the solution. Check CPU, GPU, storage, network, and memory together.

**Batch compatible work.** For our inference assumption, group compatible model versions and input shapes. Measure throughput as batch size increases and cap batch size by GPU memory and acceptable wait time. Stateless inputs may be combined; temporal/stateful models may require sequence-aware scheduling. Triton's dynamic batcher illustrates both the throughput motivation and bounded batch-wait configuration for stateless models. [NVIDIA batching documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)

**Keep workers warm.** Reuse GPU processes and loaded models rather than starting a new GPU process and loading weights for every video. Bound CPU decoding, RAM, pinned buffers, and input prefetch. GPU concurrency is limited by device resources, not by how many threads we can create.

**Control intermediate storage.** Let only a bounded window of videos advance into expensive preprocessing. Track outstanding preprocessed **bytes and estimated GPU work**, not just item count. When the GPU backlog reaches a high watermark, stop claiming new preprocessing tasks; resume below a lower watermark. Keep the source videos and durable task records until work can proceed. Accepted work is delayed, not silently dropped.

**Avoid moving giant raw-frame files unnecessarily.** A simple baseline persists bounded, packed intermediate artifacts for easy recovery. Measure their size. If transfers dominate, consider preprocessing near GPU workers, streaming bounded chunks through local buffers, or persisting compact intermediates. These alternatives trade some independent scaling or recovery simplicity for lower I/O. A lost local intermediate must be reproducible from a durable input.

**Scale with the deadline in mind.** For each stage, estimate:

```text
remaining resource work ≈ sum of predicted costs of unfinished stages
rough resource requirement ≈ remaining resource work
                             / usable seconds before deadline
                             / planning load
```

Include **upstream jobs that have not reached that stage yet**. A currently empty GPU queue does not mean no GPU work remains. Reserve time for downstream completion and long-tail jobs; validate the whole pipeline forecast, not only independent stage averages. Use queue age, backlog work, a warm minimum, cooldowns, and resource quotas to guide scaling.

For a fixed cluster, report the projected completion time. If it will miss the deadline, improve a measured bottleneck or explicitly negotiate capacity/scope. A queue stores overload; it does not create processing capacity.

**Say:** “I would scale the stage that limits completion, not every stage equally. I would keep GPUs fed with bounded prefetch and compatible batches, while backpressure prevents cheap CPU preprocessing from filling storage faster than GPUs can drain it.”

## 4.10 Validate the design — minutes 51–55

| Check | What you measure or test |
|---|---|
| Deadline and throughput | Completed valid videos/s, remaining estimated work, projected finish time, late jobs |
| Stage health | Queue age, p50/p95 runtime by input class, retries, expired leases, terminal failures |
| Resource efficiency | GPU utilization and memory, CPU saturation, storage/network throughput, cost per completed video |
| Correctness | Kill workers before/after upload and before/after metadata commit; verify no stranded next stage or overwritten accepted result |
| Skew and bursts | Mixed video lengths/resolutions, initial million-job registration, GPU loss, slow storage, OOM |
| Security and isolation | Authenticated submitters, authorization on input/result access, least-privilege worker credentials, encrypted transport/storage, sandboxed media decoding |

Track success and terminal failure separately. For a fixed manifest, account for every registered job; also track the loader's unfinished portion. A batch with corrupt videos may be **finished with errors**, but it has not met a requirement that every valid input produce a result unless those failures are resolved or explicitly excluded.

Cost is not just GPU count: include CPU, object retention, intermediate transfers, and wasted retries. Keep raw inputs and intermediate outputs under an explicit retention policy; remove intermediates only when they are no longer needed for retries, consumers, or audit.

## 4.11 Close and ask questions — minutes 55–60

Give the recap in 4.12, state the main trade-off, then let the interviewer choose any remaining deep dive.

Useful questions to ask:

- “In the team's actual workloads, is the dominant constraint latency, GPU capacity, or moving video data?”
- “How much of the workload is offline processing versus live vehicle streaming?”
- “How do you measure reliability and cost for these pipelines?”

These are questions about the role, not assumptions about Nuro's internal implementation.

## 4.12 Interview-ready reference answer

Use this as a compact answer skeleton after clarifying the requirements, not a speech to memorize word for word:

> “I would first distinguish stored-video processing from live streaming. For this batch problem, I assume the videos are already in durable storage and that each valid video must complete three ordered stages within a week.
>
> One million videos per week requires roughly 1.65 completed videos per second on average. I would benchmark CPU and GPU service demand on representative inputs, size each stage independently with headroom, and verify that storage bandwidth and long-running videos do not invalidate the deadline.
>
> The user starts a processing run and receives a tracking ID, then checks progress and retrieves saved results. A request handler receives those requests, a database remembers waiting work and progress, and separate CPU/GPU/CPU worker pools do the processing. Large files live in shared object storage; tasks carry their locations, not the video bytes. For a million-video submission, a saved input list and a restartable loader let us register jobs in chunks.
>
> A worker claims a task with a lease, processes outside the database transaction, and uploads an immutable attempt output. A conditional transaction then publishes the output reference, completes the stage, and creates the next task. Crashed tasks are retried after lease expiry; stale attempts cannot overwrite the selected result.
>
> To use GPUs efficiently, I would keep models warm, batch compatible work, bound prefetch, and apply backpressure when downstream processing falls behind. I would monitor completion rate, queue age, projected finish time, resource utilization, and cost per successful video.
>
> The main trade-off is that durable stage boundaries simplify recovery and independent scaling but add I/O. I would optimize data locality if measurements show those transfers are the bottleneck.”

## 4.13 Follow-up questions and concise answers

These are **plausible interviewer follow-ups**, not additional questions confirmed by the report.

### Capacity and scheduling

**“Why not one thread per video?”** Threads do not create GPU capacity and a million threads would waste resources. Keep durable task records and a bounded pool sized to CPU, GPU memory, and I/O limits. Unlike the toy scheduler, these workers may be distributed processes.

**“Why separate CPU and GPU workers?”** Their capacities and costs differ. Independent pools avoid reserving GPUs during CPU-only phases. Co-location is an alternative if moving intermediates costs more than the benefit of separation.

**“Why a work table instead of Kafka?”** The initial workload does not establish a broker requirement. A work table gives atomic state transitions and next-stage creation. Add a broker for a demonstrated dispatch/scale need, using an outbox and duplicate-safe consumers.

**“We only have 20 GPUs. Can you still finish?”** Not at the assumed 60 GPU-seconds/video: even the ideal weekly limit is 201,600 videos. Benchmark optimizations, obtain capacity, or renegotiate the deadline/output contract.

**“What if video durations vary widely?”** Estimate work from duration, resolution, model, and measurements; do not scale from job count alone. Use fair/aged priorities and start expensive jobs early enough to avoid a long final tail.

### Failures and correctness

**“The GPU worker dies after doing most of the work. Do you restart everything?”** Restart the failed stage from its last durable boundary, not completed upstream stages. For unusually expensive stages, add meaningful checkpoints or independently retryable chunks if the algorithm supports them.

**“Can two workers execute the same task?”** Yes, a lease can expire while an old worker is still computing. Tokens fence publication, not physical computation. Immutable attempt outputs and conditional metadata updates select one accepted result.

**“What if stage completion succeeds but creating the next task fails?”** They are one transaction in the baseline, so they commit together. If a broker is added, commit an outbox entry with completion and relay it afterward.

**“Do you guarantee exactly-once processing?”** No: duplicate attempts are allowed. The guarantee is one accepted stage result under the defined artifact-publication contract. Arbitrary external effects require separate deduplication/idempotency.

**“Can a client cancel a job or change the model?”** Treat these as extensions. Cancellation prevents further dispatch/publication through a conditional state check but may not immediately interrupt a GPU kernel. A new model version creates a new versioned processing run; do not overwrite results from the old version.

### GPU and video details

**“GPU utilization is low. Should we buy more GPUs?”** First check whether the GPUs are starved by CPU decoding, storage, transfers, small batches, or repeated model loading. Add GPUs only when GPU processing is the measured bottleneck.

**“The GPU runs out of memory.”** Reduce a compatible batch size and bound concurrent work. If one input still cannot fit, use supported chunks, a larger-memory worker, or an explicit failure. Do not endlessly retry the same oversized request.

**“Can we split a long video into independent chunks?”** Only if decoding and algorithm semantics permit it. Respect codec dependencies: decode from an appropriate preceding keyframe and discard overlap as needed. Stateful temporal models may require state transfer or overlapping context. Track chunk order and publish the final manifest only after all required chunks succeed.

**“Can we skip storing intermediate files?”** Yes, local or streamed intermediates reduce I/O, but failure may require recomputation. Keep durable source inputs and explicit retry boundaries, and never assume a local buffer survives machine loss.

**“CPU preprocessing is much faster than GPU processing.”** Pause or limit preprocessing based on downstream bytes/work backlog. Durable pending tasks can wait cheaply; unbounded decoded output cannot.

### Role-specific extension: live streaming

**“Now make it a live video-streaming system.”** Re-clarify the latency, quality, camera-count, network, and loss requirements. Weekly throughput is no longer the main target. Bound queue age and use an explicit codec-aware frame-drop/adaptation policy rather than processing stale frames indefinitely. Add capture, encoding, transport, receiver buffering, and playback; do not reuse the durable batch path unchanged.

This last question is a **role-inspired extension**, not a verified Nuro streaming prompt. Work through [Low-Latency Vehicle Video](#5-worked-example-low-latency-vehicle-video) next; use [Networking & Streaming](05_networking_streaming.md) for protocol details.

## 4.14 Final interview checklist

- State batch versus live and agree on inputs, outputs, deadline, and resource limits.
- Show the 1.65 videos/s calculation and explain what measurements determine CPU/GPU capacity.
- Draw a complete path with durable work, object references, and independent worker pools.
- Trace one video's three stages and identify the acknowledgment/publication points.
- Walk through a worker crash and a duplicate attempt without claiming exactly-once computation.
- Explain one utilization improvement and one backpressure policy.
- Check deadline feasibility, long-tail work, storage/network cost, security, and observability.
- Finish with the chosen trade-off and leave time for questions.

## 4.15 Interview script: what to say at each step

This script is cut for a **45-minute interview**: about 3 minutes of introductions, 37 technical minutes, and 5 for questions. It follows the 45-minute rule in 4.2: keep the scope, a complete diagram, the capacity calculation, and one failure walkthrough; shorten the API details; make failures the full deep dive and compress GPU efficiency. Speak these in your own words. The compact answer in 4.12 is the recap.

**Minutes 0–3 — introduce and restate**

> “I will treat this as an offline batch pipeline: one million stored videos, each through CPU preprocessing, GPU processing, and CPU postprocessing, finished within a week. I will clarify the workload, estimate capacity, design the pipeline, then go deep on failure recovery.”

**Minutes 3–8 — clarify what changes the architecture**

> “Three questions change the design: are these stored files or live streams, and are all million available on day one? What does the GPU step do, and what output is required? Is the hardware budget fixed, and must every valid video succeed? Unless you tell me otherwise, I assume a fixed collection already in shared storage, a fixed inference pipeline that saves a result per video, resources that can scale within a cost cap, and that corrupt inputs need an explicit diagnosis rather than a guarantee. I preserve stage order within a video, not across videos.”

**Minutes 8–13 — capacity and feasibility**

> “One week is 604,800 seconds, so I need about 1.65 completed videos per second on average. That rate alone does not give me a GPU count; I need measured service demand per stage. With illustrative numbers — 20 CPU-core-seconds of preprocessing, 60 GPU-seconds, 10 CPU-core-seconds of postprocessing — at a 70% planning load that is about 48 cores, 142 GPUs, and 24 cores: videos per second times resource-seconds per video, divided by planning load. If hardware is fixed, the bottleneck stage sets the rate: 20 GPUs cap us near 200,000 videos a week, and more CPU cannot fix that. At 100 MB per input the corpus is 100 TB, about 165 MB/s just to read once; decoded frames are far larger than compressed input, so preprocessing must produce compact artifacts.”

Transition: “That tells me how much capacity I need. Now I need how work is submitted, how workers find it, and how progress is tracked.”

**Minutes 13–23 — the user journey and the complete path**

> “The engineer starts a run with a list of videos and gets a tracking ID immediately, checks progress later, and retrieves result locations. Accepted is not finished. So the system must accept requests, remember progress, execute work, and save results — those are the boxes.
>
> A request handler records requests and answers progress; it never runs GPU work inside a request. Progress and waiting tasks live in a database so a restart loses nothing; a task table split by stage acts as three queues. Separate CPU, GPU, and CPU worker pools claim tasks for their stage. Large files live in shared object storage; a task carries a location, not the bytes. The stage-to-stage arrows are not transfers: finishing a stage creates the next stage's task row, and the output is a path in storage that the next worker reads. Small records go through the database, large files through storage, and workers never hand bytes to each other.
>
> Tracing video A: the handler records the batch; a task row for preprocessing appears; a CPU worker reads A, writes compact frames, and completes, which creates the GPU task; a GPU worker reads those frames, writes detections, and creates the postprocessing task; a CPU worker writes the final result and marks the job done. Video B can be in a different stage at the same time.”

**Minutes 23–26 — interfaces and records, briefly**

> “For a million videos the caller saves a manifest file and posts its URI; the handler records the batch and returns 202 with a tracking ID. A restartable loader registers jobs in chunks and advances a saved cursor in the same transaction. `GET /batches/{id}` reports progress; `GET /jobs/{id}` returns a result location. The records are a batch, a job per video, and a stage task per job and stage; retry counters and lease fields come next, because failures will give us a reason for them.”

Check in: “That is the complete working path. I would like to spend the deep dive on what happens when a worker dies or a request repeats — unless you would rather look at GPU utilization?”

**Minutes 26–34 — deep dive: failures, leases, and safe retries**

> “A worker claims a task with a lease and heartbeats to renew it; processing happens outside the database transaction. It uploads to an immutable attempt path first, then one conditional transaction — requiring the current lease token and an unexpired lease — publishes the output URI, marks the stage done, and creates the next task. Those three changes commit together, or a video could be stranded with a finished stage and no next task.
>
> A crashed task is retried after lease expiry from its last published input. If the old worker wakes up, its completion is rejected by the token check, and its file is an orphan for garbage collection. So I promise at-least-once execution and exactly one accepted result, not exactly-once computation. Repeated submissions deduplicate on a request ID; re-reading a manifest chunk deduplicates on batch and video ID. Corrupt inputs get a bounded retry policy and a visible terminal error.”

**Minutes 34–37 — GPU efficiency, compressed**

> “On utilization, I would find the measured bottleneck first: if GPUs are starved, fix decoding, transfers, or batching; if the GPU queue grows while GPUs are busy, more CPU will not help. I keep models warm, batch compatible inputs, and bound prefetch. Backpressure: when outstanding preprocessed bytes pass a high watermark, preprocessing stops claiming tasks until they drop below a low watermark — durable pending tasks wait cheaply, decoded output does not.”

Expand this block from 4.9 if the interviewer steers here; otherwise keep it at this length.

**Minutes 37–40 — validate and recap**

> “I would track completed videos per second, projected finish time, queue age per stage, expired leases, GPU utilization, and cost per successful video, and I would kill workers before and after upload and before and after commit to prove there is never a stranded next stage or an overwritten accepted result.”

Then give the reference answer in 4.12 and name the trade-off: durable stage boundaries buy recovery and independent scaling at the cost of I/O.

**Minutes 40–45 — questions**

Ask the questions in 4.11. If “why not one thread per video” comes up, answer from 4.13: threads do not create GPU capacity; keep durable task records and a bounded pool sized to the hardware.

**Check-in lines to reuse**

- “Is the deadline the constraint you care about most, or resource cost?”
- “I can trace one video through the three stages now, or go straight to what happens when a worker dies — which is more useful?”
- “Would you like me to tighten any assumption, such as the GPU cost per video?”

# 5. Worked Example: Low-Latency Vehicle Video

**Evidence and scope:** this is **role-inspired practice, not a verified Nuro interview question**. The prompt, numbers, thresholds, and architecture below are our design assumptions, not Nuro's production requirements. We design observation and remote assistance, **not safety-critical remote driving or vehicle actuation**.

## 5.1 Start with one camera and one viewer

**Practice prompt:** “An authorized remote operator needs to see live video from a vehicle over a changing cellular connection. Design the system so the view stays useful, and so the operator can tell when it is no longer live.”

Start with the user's experience: the operator selects a vehicle and camera, clicks **Connect**, and sees recent camera images. The camera keeps producing frames; the viewer does not request each frame separately.

```text
Vehicle                                             Operator
Camera → Compress frames → Cellular network → Decode → Display
              ↑                                  │
              └── Feedback: delivery is slowing ──┘
```

For the earlier batch pipeline, an old unfinished video still needs processing. Here, displaying every frame from ten seconds ago defeats the purpose. **We prefer a recent usable view over a complete history of late frames.** If historical recording is required, make that a separate path with its own reliability policy.

**Say:** “I would first confirm that this is a live observation system. My primary goals are bounded frame age, usable image quality, and an unmistakable indication when the image is stale.”

## 5.2 Learn the few terms as they become necessary

| Term | Meaning in this example |
|---|---|
| Encoder / decoder | Compress camera frames into a stream / reconstruct pictures at the viewer |
| Bitrate | Encoded bits produced per second; it must fit the available path capacity |
| Glass-to-glass latency | Time from camera capture to display; network delay is only one part |
| Jitter buffer | A short receiver-side wait to handle uneven packet arrival; more waiting adds latency |
| Signaling | Small setup messages: who may connect, which codecs to use, and how to reach each other |
| Media path | The continuous flow of encoded video packets; separate from setup messages |
| Keyframe / refresh frame | A point that can restore decoding without relying on missing earlier pictures; an H.264 IDR is one example |

Do not introduce a database, relay, or API until you can explain what problem it solves. The first drawing needs only capture, encode, send, decode, and display.

## 5.3 Clarify the requirements

| Ask the interviewer | Chosen assumption for this walkthrough |
|---|---|
| Observation, or remote driving commands? | Observation / assistance only; no actuation |
| How many cameras and viewers? | Start with one; extend to four cameras and one operator per vehicle |
| Which client? | Browser viewer; a native vehicle agent can access cameras and hardware encoders |
| What latency and quality are useful? | Target p95 capture-to-display below 300 ms during an agreed connected operating envelope |
| What happens outside that envelope? | Degrade quality, explicitly mark stale/unavailable video; no guarantee during an outage |
| Can we sacrifice some views or frame rate? | Operator-selected primary camera has priority; lower secondary-view quality is permitted |
| Do we need recording or rewind? | Not in the live path; optional recording is separate |
| What scale? | Illustratively, 100 simultaneous vehicle-viewer sessions |

The **operating envelope** means conditions such as usable bandwidth, RTT, loss, and supported devices under which we evaluate the target. Measure outage time and failed connections separately; do not hide them by reporting latency only for frames that arrived.

## 5.4 A 60-minute interview plan

| Interview minutes | What to explain |
|---|---|
| 0–5 | Introductions and clarify observation versus control |
| 5–12 | Cameras, viewer, quality, latency, network constraints, and scale |
| 12–18 | Allocate a latency budget and estimate bandwidth |
| 18–28 | Build the media path and explain how a viewer connects |
| 28–40 | Work through a bandwidth drop: feedback, adaptation, bounded queues, frame dropping |
| 40–50 | Work through a disconnect and a frozen camera; explain fresh recovery |
| 50–55 | Failure tests, measurements, access control, and the main trade-off |
| 55–60 | Concise recap and questions |

For 45 minutes, spend roughly 5 minutes on introductions, 7 on requirements/budgets, 10 on the architecture, 15 on bandwidth and failure behavior, 3 on validation, and 5 on recap/questions. These are flexible allocations, not a required speech order.

## 5.5 Allocate latency before choosing infrastructure

Latency accumulates across the whole path. An illustrative allocation for the 300 ms target is:

| Part of the path | Planning budget |
|---|---:|
| Capture / waiting for the next frame | 33 ms |
| Preprocessing and low-delay encoding | 35 ms |
| Sender queue and packet pacing | 20 ms |
| Network, including any relay | 80 ms |
| Receiver jitter / packet assembly | 40 ms |
| Decode and display | 42 ms |
| Margin | 50 ms |
| **Total** | **300 ms** |

These are allocations to validate, not measured guarantees. End-to-end p95 must be measured directly; adding independently measured stage p95 values does not calculate end-to-end p95.

At 30 fps, frames are roughly 33 ms apart. A queue of 30 frames can add about one second before network delay is even considered. Faster encoding cannot compensate for an unbounded queue elsewhere.

For bandwidth, four cameras at an illustrative 3 Mb/s each produce **12 Mb/s of video payload**. With an illustrative 25% allowance on top for protocol/repair traffic and planning margin, budget about **15 Mb/s**; measure actual overhead, which varies with loss and packetization. Do not assume every cellular path can provide this.

**Say:** “I would instrument every stage, because a 20 ms network does not help if frames spend a second waiting for the encoder.”

## 5.6 Build the architecture one need at a time

### A. Turn camera frames into a low-delay stream

The **vehicle agent** is a program on the vehicle. It reads camera frames, preserves their source identity and capture time, and feeds a hardware encoder when available. Start with a mutually supported codec such as H.264, a low-delay configuration without B-frame reordering, and bounded encoder lookahead. Confirm the selected profile and decoder support during setup.

Reducing reordering can cost compression efficiency. Encoder capacity and heat/power limits also constrain how many simultaneous camera streams we can produce.

### B. Choose a real-time media transport

Use a mature **WebRTC** implementation on the vehicle and in the browser. It supplies media transport, encryption, congestion feedback, and connectivity mechanisms; we configure and measure it rather than writing a new transport stack.

The baseline media path is encrypted **SRTP over UDP**. UDP does not by itself provide reliability or congestion control; the media stack handles deadline-aware loss recovery and rate control. TCP/TLS relay fallback may improve connectivity when UDP is blocked, but it can worsen delay and must still meet the freshness policy. [WebRTC transport standard](https://www.rfc-editor.org/rfc/rfc8835.html)

### C. Help the two endpoints find a usable route

The vehicle is commonly behind a cellular **NAT**, a gateway translating private addresses. The browser may be behind another NAT. Knowing the vehicle ID is not enough to send media directly to it.

- **ICE** tests candidate network paths and chooses a working one.
- **STUN** helps discover an externally visible address; it does not relay video.
- **TURN** relays packets when a usable direct path is unavailable or relay-only policy is required.

Use direct media where policy permits and it works; provision regional TURN capacity for the fallback. Region selection should minimize the actual vehicle-to-relay-to-viewer path, not merely choose a region near the viewer. [WebRTC peer-connection setup](https://webrtc.org/getting-started/peer-connections)

### D. Add a small setup and authorization service

The operator must be authorized to view that vehicle. A **session service** checks this permission, chooses connection settings, issues short-lived credentials, and exchanges signaling messages between the browser and vehicle agent. The agent maintains an authenticated outbound connection to this service; we do not require an inbound public vehicle address.

Now the complete drawing has two different paths:

```text
SETUP AND AUTHORIZATION — small messages

Browser ←────→ Session / signaling service ←────→ Vehicle agent
                  │
             Permissions + session records
                  └── connection settings / relay credentials

LIVE MEDIA — continuous video, not database writes

Camera → Encoder → Pacer → Direct path OR TURN relay
                                            │
                                            ▼
Viewer ← Display ← Decoder ← Short jitter buffer
                       │
                       └── Receiver feedback → Sender adaptation
```

A **pacer** spreads outgoing packets over time instead of emitting every encoded frame as a large burst. It also needs a bounded queue. The session service does not process every frame, and no durable message queue sits between the camera and viewer.

## 5.7 Walk through connection setup and one frame

1. The operator selects vehicle V7 and its front camera. The session service authenticates the operator and authorizes that camera.
2. The browser and agent exchange codec capabilities, transport settings, and candidate addresses through signaling. ICE selects a direct or relayed path, and the media stack establishes encryption.
3. The agent starts the stream and sends the configuration and a decoder refresh frame. The UI remains **CONNECTING** until a usable fresh frame is displayed.
4. Camera frame 812 is captured, identified, encoded, and divided into packets. A frame can span many packets; one packet is not one frame.
5. The receiver briefly buffers/reorders packets, reconstructs the encoded frame, decodes it, and presents it. Expired or unusable data does not wait forever.
6. Feedback on delivery timing and loss lets the sender adjust its offered bitrate. The viewer also monitors whether the displayed source frame is actually advancing.

There is no HTTP request per frame. The viewer's connect/disconnect or camera-selection messages are occasional control operations; media and feedback flow continuously after setup.

**Say:** “Signaling arranges the session. Media then travels directly or through a relay. I would trace one frame through capture, encoding, transport, buffering, decoding, and presentation.”

## 5.8 Work through a bandwidth drop: 12 Mb/s to a 4 Mb/s path

Assume the four streams were producing 12 Mb/s of payload, but the usable path suddenly falls to 4 Mb/s. Continuing at the old rate produces at least **8 megabits of excess data per second**, before overhead. After one second, that excess alone takes about **two seconds** to transmit over the new path.

**A bigger queue cannot solve this.** It turns congestion into an increasingly old picture.

```text
Receiver observes rising delay / loss
                 │ feedback
                 ▼
Sender estimates available transport rate
                 │
                 ▼
Camera budget allocator → Encoder targets → Bounded pacer
        │
        └── prioritize primary view; reduce other views first
```

Use the transport stack's congestion controller. It combines delivery timing, loss, and feedback; loss alone does not distinguish congestion from every wireless impairment. Configure encoder targets below the estimated total capacity and reserve room for headers, repair traffic, and bursts. Congestion control is required even on UDP. [Real-time congestion-control requirements](https://www.rfc-editor.org/rfc/rfc8836.html)

For a **4 Mb/s estimated total budget**, an illustrative allocation is **3 Mb/s of video**: 1.8 Mb/s for the selected camera and 0.4 Mb/s for each of three secondary views. The remaining 1 Mb/s is an allowance, not a fixed protocol overhead. If estimates fall further, lower resolution/frame rate or pause secondary views according to the agreed policy. If the primary view cannot remain useful, show unavailable rather than silently claiming success.

All cameras share the same uplink: coordinate their total budget instead of letting four controllers each assume they own the entire path. Reduce the target promptly when delivery worsens, and increase it cautiously after sustained recovery. Pace large refresh frames and cap their burst impact.

**Say:** “I would reduce what the vehicle sends, not just make the viewer drop more. Receiver-side dropping cannot recover cellular bandwidth already consumed.”

## 5.9 Bound buffers and drop frames where it is safe

Use both **queue age** and byte/count limits. A one-frame queue is not sufficient if that frame is already old. Check the oldest source capture time at each stage that can expose it.

| Location | Baseline policy | Reason |
|---|---|---|
| Before encoding | Keep a small bounded set of recent raw frames; discard older waiting frames | Frames not yet encoded have not become prediction references |
| Encoder and sender pacer | Low-delay configuration, bounded waiting, adapt bitrate early | Encoded pictures may depend on earlier pictures |
| Receiver jitter buffer | Small adaptive buffer constrained by the latency target | More buffering hides jitter but makes the view older |
| Decode / presentation | Present recent decodable pictures; do not retain an unbounded display backlog | A frame can be needed for decoding even if it is not shown |

Discarding a queued raw frame must not overwrite memory still owned by the camera, encoder, or another consumer. Buffer ownership and device completion remain necessary.

### Why not drop any old encoded frame?

```text
IDR ─→ P1 ─→ P2 ─→ P3
        X      └──── may depend on the missing P1
```

Dropping P1 but sending its dependent pictures can leave the receiver unable to reconstruct them correctly. Prefer dropping **before encoding**, using explicitly discardable layers when supported, or coordinating a decoder refresh. If we must abandon an encoded backlog, discard its dependent chain and recover from a fresh IDR/appropriate random-access point with codec configuration. Rate-limit refresh requests so recovery itself does not create repeated bitrate spikes.

Some expired pictures may still be decoded to preserve references but not displayed. Do not promise that a browser application can directly tune every internal WebRTC queue; choose implementations with suitable controls and verify actual delay and drop metrics.

## 5.10 Handle packet loss without waiting forever

Packets may be lost even when the connection remains usable. A **NACK** asks for a missing packet; a **PLI** asks the sender to refresh a damaged picture stream. These are recovery mechanisms, not a requirement to recover every historical frame. WebRTC RTP transport includes feedback and retransmission mechanisms. [WebRTC media transport](https://www.rfc-editor.org/rfc/rfc8834.html)

Retransmit only when the repair can plausibly arrive before that frame's deadline. For example, with about 80 ms of useful time left and an estimated 30 ms repair round trip, repair may help; with 10 ms left, waiting for that same repair would usually be too late. Include processing and uncertainty, not just RTT.

Optional **forward error correction** sends redundant data that can recover some losses without a round trip. It consumes bandwidth, so account for it inside the total transport budget. If references are no longer recoverable, request a fresh decoding point instead of letting a damaged chain persist.

The trade-off is not “reliable versus unreliable.” It is **useful repair before the deadline versus delivery too late to be useful**.

## 5.11 Recover from disconnects at the live edge

Unlike batch processing, we do not checkpoint the live frame queue and replay it after a crash. We preserve session authorization and configuration, then return to **new camera frames**.

```text
LIVE → frames stop → STALE / RECONNECTING
                            │
                  restore signaling / ICE path
                            │
                  codec config + fresh refresh frame
                            │
                   fresh frame is displayed
                            ▼
                           LIVE
```

### Cellular handover or address change

Mark staleness from frame progress immediately according to the policy; do not wait for a long transport timeout. On sustained path failure, initiate an **ICE restart** to discover a usable path. Brief disconnection can be transient, so avoid recreating sessions on every jitter spike. Use bounded reconnect attempts with backoff and jitter, and maintain the unavailable indication while reconnecting.

### Agent, relay, or viewer restart

The restarted endpoint reauthenticates, restores the requested cameras/configuration, and establishes a new transport or replacement session. If a relay fails, select another eligible relay; in-flight media is not recovered from the failed server. When the decoder has lost its references or the old chain was abandoned, send configuration and a fresh refresh frame. An ICE path change alone does not inherently require an encoder restart.

Assign a **stream generation** whenever the source stream/decoder context is recreated. Match frame identity and capture metadata to that generation, and ignore old callbacks/media from superseded tracks or sessions. Return to LIVE only after fresh frames are presented, not merely after a socket connects. Drain no old network backlog into the recovered live view.

If signaling fails while the media path remains healthy, existing media can continue for its authorized lifetime. New sessions and path renegotiation may fail. Define token expiry/revocation behavior explicitly rather than assuming an established encrypted connection remains authorized forever. The WebRTC API provides connection states and an ICE-restart operation. [WebRTC connection API](https://www.w3.org/TR/webrtc/)

## 5.12 Detect stale video even when the connection is healthy

**Connected does not mean live.** Packets, heartbeats, or decoded frames can continue while the camera image is old. Monitor source progress and presentation, per camera.

At capture, attach a source-generated **frame ID, capture timestamp, and stream generation**. Keep the mapping through encoding and presentation. Do not replace the source capture time with the time a relay forwarded the frame, and do not assign a new capture identity when repeating an old image.

Track two different signals:

1. **No new source frame has been presented recently.** An independent viewer watchdog checks elapsed monotonic time since the last advancing source frame ID. It must still run when there are no video callbacks.
2. **The frame being displayed was captured too long ago.** Estimate capture-to-presentation age using source-to-viewer clock mapping and its uncertainty. Do not subtract unrelated device clocks or interpret an RTP timestamp directly as wall-clock time.

```text
New frame arrives now, but was captured 2 seconds ago
    Transport: receiving packets
    Picture:   changing
    Freshness: STALE — arrival time is not capture time
```

WebRTC-capable browser frame callbacks may expose an **estimated captureTime**, reception time, and presentation timing; availability varies. They observe compositor submission/expected display, not a direct measurement of photons on the display. Select a sender/client combination exposing the required frame metadata, validate the clock mapping, and include uncertainty. If reliable capture timing is unavailable, still detect stalled frame progress but label absolute freshness **unknown** rather than inventing an age. [Video-frame callback timing](https://web.dev/articles/requestvideoframecallback-rvfc)

An illustrative UI policy: below the 300 ms age target is LIVE; 300–500 ms is DELAYED; over 500 ms or no new source frame for 500 ms is STALE. Use a conservative age bound that accounts for clock error. These thresholds need product validation; they are not a safety guarantee. A fresh-looking stream with untrusted timing should show freshness unknown.

On stale video, show an unmistakable overlay such as **“VIDEO STALE — last fresh frame 1.2 s ago”**, dim/blank the image or clearly label a frozen last frame, and expose camera identity. Do not leave an unmarked still image that looks live. On page visibility/resume, reevaluate freshness before presenting the old frame as live. A static scene is not evidence of a frozen camera; source sequence/timing and capture-device health are better signals than pixel equality. A faulty camera that fabricates fresh metadata requires additional device diagnostics and cannot be conclusively detected from transport alone.

**Say:** “I would independently check whether new source frames are being displayed and whether those frames were captured recently. Network connectivity alone proves neither.”

## 5.13 Scale the design and protect access

For 100 sessions at the earlier 12 Mb/s payload per vehicle, the fleet sends **1.2 Gb/s of video payload**. If every session uses TURN, the relay tier handles roughly 1.2 Gb/s ingress **and** 1.2 Gb/s egress. Applying the same illustrative 25% planning allowance gives about 1.5 Gb/s each way, or 3 Gb/s combined traffic. Also plan for packet rate, concurrent sessions, encryption/relay CPU, memory, and failover headroom; not just nominal link speed.

Keep relay capacity in more than one failure domain and avoid placing all sessions on one machine. Session/signaling services can scale separately because they do not handle every media packet. New connection bursts and relay failure must not exhaust their capacity.

For **one operator**, start without a forwarding media server beyond TURN. If several viewers must watch one vehicle, uploading separate copies consumes extra vehicle bandwidth. A **selective forwarding unit (SFU)** can receive a stream once and forward it to multiple viewers, optionally selecting available layers. Unlike TURN's packet relay role, it manages media streams/subscriptions; it normally does not transcode. This is an extension, with additional service cost and a different encryption/trust boundary.

Keep setup records small: `session_id`, vehicle/camera IDs, authorized viewer, expiry, chosen region, and configuration/generation. Do not store every live frame in the session database. Use authenticated devices, per-vehicle/camera authorization, encrypted media, expiring TURN credentials, viewer limits, and auditable access. Define whether media is endpoint-encrypted through TURN or terminates at a trusted SFU; “encrypted” does not automatically mean the server cannot access content.

## 5.14 Test failures and measure what the operator experiences

| Inject this failure | Expected behavior | Measure |
|---|---|---|
| Uplink falls from 12+ Mb/s to 4 Mb/s | Reduce total encoder targets, prioritize primary view, bound queues | Time to adapt; displayed frame age; usable primary-view quality |
| High RTT, jitter, or packet loss | Bounded jitter buffering and timely repair; refresh if needed | End-to-end age, freeze duration, late drops, NACK/refresh rate |
| Encoder becomes slower than capture | Drop old waiting raw frames; reduce workload | Capture-to-encode queue age, CPU/GPU load, encoder time |
| Camera freezes but heartbeat continues | Source-progress watchdog marks that camera stale | Time from source stall to visible warning |
| Old frames arrive continuously | Capture-age check marks them delayed/stale | Age estimate and clock uncertainty, not arrival rate alone |
| Cellular outage or relay crash | Immediate freshness policy; reconnect; recover at live edge | Stale duration, reconnect success, time to first fresh frame |
| Signaling service fails | Existing authorized media may continue; setup/restart visibly fails | Active media health versus new-session success |
| Viewer tab sleeps or decoder stalls | Recheck freshness on resume; never label cached frame live | Presentation progress, decoder time, resume-to-fresh-frame time |

Collect capture/encode/send/receive/decode/presentation timestamps where available, queue age and bytes, actual encoder bitrate, estimated path capacity, RTT/loss, dropped frames, camera availability, and stale-time fraction. Track p50/p95/p99 end-to-end latency **alongside** connection failures and missing/stale-video time so dropping nearly every frame cannot make the service look healthy. WebRTC statistics expose transport and video counters, but do not alone establish true camera-to-display freshness. [WebRTC statistics specification](https://www.w3.org/TR/webrtc-stats/)

Validate software timing against a camera/display test such as filming a changing time display and comparing capture and rendered output. Repeat under shaped bandwidth, loss, handovers, device load, and restarts. State the trade-off explicitly: **bounded latency can require lower quality or missing frames; no transport guarantees a fresh view during an outage**.

## 5.15 Interview-ready reference answer

> “I would first confirm this is a live observation system, not a vehicle-control loop. I would agree on camera count, useful image quality, the connected-network envelope, and a capture-to-display latency target.
>
> I would use a vehicle agent with bounded capture queues and low-delay hardware encoding, sending WebRTC media to a browser. A separate session service handles authorization and signaling. ICE selects a usable path, with regional TURN relays when direct connectivity is unavailable.
>
> I would budget latency across capture, encoding, sender queues, transport, receiver buffering, decoding, and display. Receiver feedback drives a shared uplink budget across cameras. If bandwidth falls, I reduce bitrate and secondary-view quality instead of allowing old frames to accumulate. Frame dropping must respect codec dependencies; damaged reference chains recover at a fresh decoding point.
>
> After disconnection, I recover at the live edge rather than replay queued video. The viewer checks both source-frame progress and estimated capture age, with clock uncertainty, and clearly marks stale or unknown freshness even if the connection remains alive.
>
> I would validate bandwidth drops, frozen cameras, encoder overload, relay loss, and reconnects. My main trade-off is freshness versus completeness and quality: I can reduce quality or drop frames, but I cannot promise live video when the network is unavailable.”

## 5.16 Follow-up questions and concise answers

These are plausible practice follow-ups, **not additional reported Nuro questions**.

| Interviewer asks | Start with this answer |
|---|---|
| “Why not reuse the batch queue and retries?” | Live frames expire. Keep session/configuration state, but do not durably queue old frames for later live display. |
| “Why WebRTC instead of HLS/DASH?” | Our target is interactive freshness. WebRTC provides an interactive media stack; segmented HTTP delivery is a different latency/fan-out trade-off. Reconsider if many viewers and seconds of delay are acceptable. |
| “Why not TCP for every video packet?” | Ordered recovery can hold fresh data behind lost old data. Prefer UDP-based media with congestion control and deadline-aware repair; permit TCP/TLS fallback when connectivity requires it. |
| “Can I just increase the buffer?” | It can absorb jitter but adds age. Bound waiting and reduce the offered rate if the path cannot keep up. |
| “What if four cameras compete for 4 Mb/s?” | Allocate one total transport budget, reserve overhead, prioritize the agreed primary view, and lower or pause secondary views. |
| “Can we drop all old P-frames?” | Not arbitrarily: later frames may depend on them. Drop before encoding, discard safe layers, or recover the dependent chain with a fresh decoder refresh. |
| “Why is a heartbeat not enough?” | It proves the process/path is alive, not that a new camera frame was captured and displayed. |
| “Do we resume at the last frame after a crash?” | No. Reestablish the session and decoding context, discard obsolete work, and display a fresh frame. Recording/replay is a separate feature. |
| “What if clocks are not synchronized?” | Use bounded clock mapping when available; otherwise mark absolute age unknown while still detecting lack of source-frame progress. |
| “Now support several viewers per vehicle.” | Consider an SFU to avoid uploading one full copy per viewer. Evaluate viewer-specific quality, layers, capacity, and the trust boundary. |

**Final check:** draw both setup and media paths, explain the 12-to-4 Mb/s event, show where dropping is safe, and walk through reconnect plus a camera freeze. Those four explanations form a complete first-pass answer.

## 5.17 Interview script: what to say at each step

This script is cut for a **45-minute interview** using the allocation in 5.4: about 5 minutes of introductions, 7 on requirements and budgets, 10 on the architecture, 15 on bandwidth and failure behavior, 3 on validation, and 5 for recap and questions. Speak these in your own words. The compact answer in 5.15 is the recap.

**Minutes 0–5 — introduce; observation, not control**

> “First I will confirm this is live observation and remote assistance, not a control loop that drives the vehicle. My goals are bounded frame age, usable image quality, and an unmistakable indication when the picture is no longer live. I would rather show a recent usable view than a complete history of late frames; recording, if needed, is a separate path.”

**Minutes 5–12 — requirements, latency budget, and bandwidth**

> “Questions: how many cameras and viewers; browser or native viewer; what latency and quality are useful; what happens outside the network envelope; can I sacrifice secondary views; what scale? My assumptions: one operator and four cameras with a browser viewer and a native vehicle agent; p95 capture-to-display under 300 ms inside an agreed operating envelope; degrade and mark stale outside it; the primary camera has priority; illustratively 100 concurrent sessions. I measure outage time and failed connections separately, so latency reported only for frames that arrived cannot hide them.
>
> Latency accumulates, so I allocate it before choosing infrastructure: capture 33 ms, encoding 35, sender queue 20, network 80, jitter buffer 40, decode and display 42, margin 50 — 300 ms, to be validated end to end. At 30 fps a queue of 30 frames alone adds a second, so bounded queues matter more than a faster encoder. Four cameras at 3 Mb/s is 12 Mb/s of payload; I budget about 15 Mb/s with overhead and would not assume every cellular path provides it.”

**Minutes 12–22 — build the media path and connect one viewer**

> “A vehicle agent reads cameras, preserves frame identity and capture time, and feeds a hardware encoder in a low-delay configuration without B-frame reordering. Transport is a mature WebRTC stack: encrypted SRTP over UDP with congestion feedback, and a TCP or TLS relay fallback only when UDP is blocked. The vehicle sits behind cellular NAT, so ICE finds a path, STUN discovers an external address, and regional TURN relays are the fallback. A small session service authorizes the operator, issues short-lived credentials, and exchanges signaling; the agent keeps an authenticated outbound connection, so the vehicle needs no public address.
>
> That gives two paths on the board: small setup messages through the session service, and continuous media directly or through TURN — no database in the media path. [Draw the 5.6 diagram.] Tracing one session: the operator selects vehicle V7's front camera; the session service authorizes; the endpoints exchange capabilities and candidates; ICE picks a path; the agent sends configuration and a refresh frame; the UI stays CONNECTING until a fresh frame is displayed. Frame 812 is captured, encoded, packetized, briefly jitter-buffered, decoded, and presented, and receiver feedback adjusts the sender's bitrate. There is no HTTP request per frame.”

Check in: “That is the working path. The two hardest events are a bandwidth drop and a disconnect or frozen camera. Which would you like first?”

**Minutes 22–30 — the 12-to-4 Mb/s bandwidth drop**

> “Say the usable path falls from 12 to 4 Mb/s. Continuing at the old rate produces 8 megabits of excess every second; a bigger queue only turns congestion into an older picture. Receiver feedback drives the congestion controller's rate estimate, and one allocator splits a single shared uplink budget across cameras — for a 4 Mb/s estimate, roughly 3 Mb/s of video: 1.8 for the primary view and 0.4 for each secondary, leaving room for headers and repair. Cut promptly, recover cautiously, pace refresh frames. I reduce what the vehicle sends; receiver-side dropping cannot recover bandwidth already consumed.
>
> Dropping is safe before encoding, because raw frames are not yet references. Encoded P-frames may be needed by later frames, so if I must abandon an encoded backlog I discard the dependent chain and recover from a fresh IDR, rate-limited. For packet loss, I retransmit only when the repair can arrive before the frame's deadline and request a refresh for a damaged chain. The trade-off is useful repair before the deadline versus delivery too late to matter.”

**Minutes 30–37 — disconnect and a frozen camera**

> “On a disconnect I do not checkpoint and replay the frame queue. I keep session authorization and configuration and return to new frames: mark stale from frame progress immediately, ICE-restart on sustained failure, reconnect with bounded backoff, assign a new stream generation so old media is ignored, and return to LIVE only after a fresh frame is presented — not when a socket connects.
>
> Connected does not mean live. I check two independent signals per camera: whether a new source frame ID has been presented recently, and how old the displayed frame's capture time is, with clock uncertainty. Under 300 ms is LIVE, 300 to 500 ms is DELAYED, over 500 ms or no progress is STALE, with an overlay that says how long ago the last fresh frame was. If capture timing is not trustworthy, I show freshness unknown rather than inventing an age. A heartbeat proves the process is alive, not that the camera captured anything.”

**Minutes 37–40 — validate, scale, and protect**

> “I would inject a 12-to-4 drop, high loss, a slow encoder, a frozen camera with a live heartbeat, a relay crash, and a sleeping viewer tab, and measure time to adapt, displayed frame age, and time to first fresh frame — reporting p95 and p99 age alongside stale-time fraction and failed connections, so dropping every frame cannot look healthy. At 100 sessions the fleet sends 1.2 Gb/s, about 1.5 Gb/s each way through TURN, so relays sit in more than one failure domain. Access: authenticated devices, per-camera authorization, expiring TURN credentials, and an explicit statement of where encryption terminates.”

**Minutes 40–45 — recap and questions**

Give the reference answer in 5.15 and name the trade-off: bounded latency can cost quality or frames, and no transport guarantees a fresh view during an outage. If asked why not reuse the batch queue: live frames expire, so keep session state durable but never queue old frames for later live display. Then ask about the team's real latency, camera, and network constraints.

**Check-in lines to reuse**

- “Is the operator's priority the primary camera, or equal quality across all four?”
- “I can go deeper on the codec dependency rules for dropping, or on the freshness detector — which is more useful?”
- “Should I treat multiple viewers per vehicle as in scope?”

<a id="4-optional-video-reference"></a>

<a id="5-optional-video-reference"></a>

# 6. Optional Video Reference

The three reported-topic cases and the role-inspired live-video walkthrough above are the main examples. These short references preserve useful earlier material; they are **not additional reported Nuro questions** and are not prerequisites.

<a id="2-building-blocks-you-actually-need"></a>

<a id="41-a-few-terms-to-look-up-when-needed"></a>

<a id="51-a-few-terms-to-look-up-when-needed"></a>

## 6.1 A few terms to look up when needed

**Latency** is time for one operation; **throughput** is work completed per second. **p95** is a value at or below which 95% of observations fall. **Backpressure** asks producers to slow down; if they cannot, a bounded queue needs an explicit reject, drop, or spill policy. **Replication** keeps copies; **sharding** divides ownership for capacity. A cache is not automatically durable storage.

<a id="3-practice-live-video-for-a-remote-operator"></a>

<a id="42-live-vehicle-video"></a>

<a id="52-live-vehicle-video"></a>

## 6.2 Live vehicle video

For the complete beginner-first design, use [Low-Latency Vehicle Video](#5-worked-example-low-latency-vehicle-video). This is only the compact reference.

```text
Operator → Session authorization ↔ Vehicle agent
Camera → Encoder → Variable cellular path / relay → Viewer
             ↑           receiver feedback           │
             └────────────────────────────────────────┘
```

For four cameras at 3 Mb/s each, one session needs 12 Mb/s of video payload. At 100 sessions, that is 1.2 Gb/s before overhead; a fully relayed service handles ingress and egress separately. If one vehicle's capacity falls to 4 Mb/s, prioritize views and reduce offered bitrate instead of accumulating stale frames. Bound queue age; use codec-aware recovery and mark frozen video clearly. This is observation/remote assistance, not a cloud driving-control safety loop. [WebRTC peer connections](https://webrtc.org/getting-started/peer-connections) explains connectivity setup and relay use.

<a id="4-practice-an-on-vehicle-camera-processing-pipeline"></a>

<a id="43-on-vehicle-camera-buffers"></a>

<a id="53-on-vehicle-camera-buffers"></a>

## 6.3 On-vehicle camera buffers

```text
Camera → Shared immutable buffer → Encoder queue → release
                                → Recorder queue → release
                                → Preview slot → release
Buffer: FREE → WRITING → READERS → FREE only when safe
```

A 1080p 8-bit YUV420 frame takes `1920 × 1080 × 1.5 = 3,110,400` bytes. Four 30-fps cameras produce about 373 MB/s of raw payload. Passing handles can reduce copies, but every reader still needs explicit ownership. Do not overwrite a frame being read. A process timeout is not proof its mapping or device access is finished; coordinate process teardown and GPU/DMA completion. Give slow optional consumers bounded claims so they cannot pin all buffers.

<a id="5-practice-resumable-video-upload-and-replay"></a>

<a id="44-resumable-video-upload"></a>

<a id="54-resumable-video-upload"></a>

## 6.4 Resumable video upload

```text
Encoded chunks → Local spool → Upload API → Immutable objects
                                      └→ Verified manifest + time index
Engineer → Authorized time query → Manifest → Chunk reads
```

Stable recording/chunk IDs and checksums make retries safe. Publish a complete manifest only after verifying expected chunks; retain codec configuration and keyframe metadata for decoding and seeking. Six 4-Mb/s cameras produce 3 MB/s, or 86.4 GB over eight hours. A 200-GB local budget provides about 18.5 recording hours without uploads, not unlimited outage tolerance. Delete local data only after remote durability confirmation, and keep uploads lower priority than live traffic.

For details already covered elsewhere, use [Networking & Streaming](05_networking_streaming.md) and [Video Domain](01_video_domain.md).

---

# 7. Apple Example: An ML-Based Video Codec

**Evidence and scope:** this prompt, "how would you develop an ML-based video codec", was asked in your own Apple system-design interview. The numbers, budgets and architecture below are our design assumptions for a rehearsal, not Apple's products or requirements. The codec internals are covered in [ML §6.4](02_ml_domain.md#64-learned--generative-compression) and the classical pipeline in [Video Domain §3](01_video_domain.md#3-video-codecs); this chapter is about turning them into a system that can ship.

## 7.1 Start with one frame and one bit budget

A codec turns pictures into bits and bits back into pictures. A 1080p frame has 1,920 × 1,080 = 2.07 million pixels, about 3 MB raw at 8 bits per channel. A video call sends it at roughly 2 Mb/s at 30 frames per second, which is 0.03 bits per pixel: the encoder must throw away 99.9% of the raw bits and keep the ones a viewer notices.

```text
Camera frame ─→ ENCODER ─→ 0.03 bits per pixel ─→ network / disk ─→ DECODER ─→ shown frame
                 (choose what to keep)                              (rebuild the picture)
```

Every codec since H.261 fills those two boxes the same way: predict each frame from already-coded frames (motion), transform and quantize what prediction missed, entropy-code the result. "ML-based" can mean three different things, and the first job in the interview is to find out which one the interviewer means:

| Where the ML goes | What changes | Ships with |
|---|---|---|
| Around the codec | ML pre-processing and post-processing; the bitstream is still HEVC or AV1 | every phone's hardware decoder |
| Inside the codec | a learned in-loop filter, intra predictor or rate control; the bitstream format changes | a new standard, hardware later |
| Instead of the codec | an end-to-end learned encoder and decoder; the decoder is a network | a neural accelerator on every receiver |

## 7.2 The 45-minute plan

| Minutes | Do | Have said by the end |
|---|---|---|
| 0–7 | Clarify which "ML-based" and which product | the target, the baseline, and the hard constraints |
| 7–14 | Estimate the compute and bit budgets | why the decoder budget decides the architecture |
| 14–19 | High-level path and the roadmap | ship ML around and inside the standard first; build the learned codec behind it |
| 19–27 | Build the learned codec one module at a time | I-frame model, P-frame model, entropy coding, rate control, bitstream |
| 27–33 | Deep dive: bit-exact decoding and drift | what makes a learned codec unusable in production |
| 33–39 | Deep dive: training and on-device deployment | data, model size, energy, versioning, fallback |
| 39–42 | Validate | BD-rate, subjective tests, device metrics, fuzzing |
| 42–45 | Questions | |

## 7.3 Clarify the requirements — minutes 0–7

Three questions change the whole design:

1. **Which ML?** "Do you want a learned codec that replaces HEVC, or machine learning that improves a standard codec? I would propose both, in that order of shipping." A learned codec needs a neural decoder on every receiver; an ML encoder tool needs nothing on the receiver at all.
2. **Which product and which side is constrained?** Real-time two-way calls need a symmetric budget on two phones. Camera capture needs a real-time encoder and a decoder that runs on every device that will ever open the file. Cloud storage can afford an expensive encoder once and needs a cheap decoder many times. The asymmetry decides where ML is affordable.
3. **What must it beat, and how is that measured?** Bitrate savings at equal quality against the hardware HEVC or AV1 encoder of the same device, measured with VMAF and PSNR on standard sets and confirmed by subjective tests; not against a slow software reference encoder that no product uses.

Our chosen requirements, stated back to the interviewer:

- Target: real-time 1080p at 30 frames per second, encode and decode on the phone, video calls first, capture and storage later.
- Quality: at least 25% lower bitrate than the device's hardware HEVC encoder at equal VMAF, in the 1–6 Mb/s range where calls live; never worse than HEVC on any content class, including text and screen content.
- Latency: encode plus decode under 33 ms per frame, low-delay structure for calls.
- Power: comparable to the hardware codec path within a small factor, measured in mJ per frame and sustained without thermal throttling.
- Interoperability: the decoder must be bit-exact across chips and OS versions, the model version must be part of the bitstream, and every path must fall back to HEVC.
- Privacy: training uses licensed and synthetic video; no user content leaves the device; on-device metrics only.

**Say:** "The receiver decides everything. If the receiver can run a network at 30 frames per second within the power budget, I can replace the codec; if not, ML lives at the encoder and in post-processing, and the bitstream stays standard."

## 7.4 Estimate the budgets and check feasibility — minutes 7–14

```text
Pixels:      1,920 × 1,080 × 30/s ≈ 62 million pixels per second
Bits:        2 Mb/s ÷ 62 Mpixel/s ≈ 0.03 bits per pixel   (a call)
             6 Mb/s                ≈ 0.10 bits per pixel   (a stored recording)

Decoder cost at c multiply-adds per pixel:
   c = 1,000      →   62 GMAC/s   fine for a phone accelerator
   c = 10,000     →  620 GMAC/s   feasible, power is the question
   c = 100,000    →  6.2 TMAC/s   not sustainable on a phone; server class
```

Published learned video codecs that beat the VVC reference encoder in PSNR spend on the order of 10^5 to 10^6 multiply-adds per pixel in the decoder; a hardware HEVC decoder is a fixed-function block that costs milliwatts. That gap is the design problem. A phone's neural accelerator advertises tens of TOPS at peak, but sustained throughput under a power budget is a fraction of that, so the on-device decoder must stay in the low thousands of multiply-adds per pixel. The encoder may be heavier for storage, where it runs once, but not for calls, where it runs on the same phone.

Memory bandwidth is the second limit: one 64-channel feature map at 1080p in 16-bit floats is 2.07 M × 64 × 2 B = 265 MB. A decoder that materializes ten such maps per frame moves gigabytes per second, so the network must downsample early, process at 1/4 or 1/16 resolution, and upsample at the end, or work in tiles.

Bits: at 0.03 bits per pixel, a 1080p frame is 62 kbit. The hyperprior side information of a learned codec costs 1–5% of that, which is affordable; a model checkpoint of tens of megabytes is not something to put in a stream, so the bitstream carries a model identifier and the decoder must already have that exact model.

**Say:** "Before any architecture, I would fix a decoder budget in multiply-adds per pixel and in bytes moved per frame, and treat every model as failing if it exceeds it, whatever its BD-rate."

## 7.5 Explain the high-level path — minutes 14–19

```text
Roadmap, ordered by what can ship

1. ML around the codec         camera → ML denoise / adaptive downscale → HW HEVC encoder
   (bitstream unchanged)        HW HEVC decoder → ML post-filter (deblock, super-resolve) → display

2. ML inside the encoder       ML rate control, mode decision, per-shot parameter choice
   (bitstream unchanged)        the decoder does not change at all

3. ML inside the standard      learned in-loop filter or intra predictor, signalled in a new profile
   (new bitstream, needs        both ends update; hardware years later
    a standard)

4. Learned codec               neural encoder and decoder; the model version is the format
   (new bitstream, neural       calls between updated devices first; fallback to HEVC otherwise
    decoder required)
```

Stages 1 and 2 ship on hardware that already exists and are measured with the same metrics, so they are the first year of the project and the baseline the learned codec must beat. Stage 4 is the research track: it wins clearly at low bitrates, where blocking and blurring of a standard codec are worst, and it is the only stage that can adapt to a content domain end to end. Stage 3 is what the standards bodies are exploring; it matters if the product must interoperate with other vendors.

**Say:** "I would run stages 1 and 2 as the shipping track and stage 4 as the research track, with one shared evaluation harness, so that the learned codec is always compared against the improved standard pipeline and not against a static baseline."

## 7.6 Build the learned codec one module at a time — minutes 19–27

The end-to-end codec keeps the shape of a classical codec, predict then code the rest, with every module a network trained on the rate-distortion loss `L = D + λ R` from [ML §6.4.1](02_ml_domain.md#641-the-core-idea).

### A. The I-frame model

```text
x ─→ analysis g_a ─→ y ─→ quantize ─→ ŷ ─→ arithmetic coder ─→ bits
                     │
                     └→ hyper-analysis h_a ─→ z ─→ ẑ ─→ bits (side information, 1–5%)
bits ─→ ẑ ─→ hyper-synthesis h_s ─→ (μ, σ) for every element of ŷ ─→ decode ŷ ─→ synthesis g_s ─→ x̂
```

The hyperprior transmits a small description of the local statistics so that each latent element is coded under a tight distribution. This is where the bits are won; the transforms matter less.

### B. The P-frame model

```text
reference x̂_{t-1}, input x_t
   ↓ motion network                 flow v_t
   ↓ motion codec                   v̂_t as bits  (its own analysis, quantizer, entropy model)
   ↓ warp the reference features    context C_t
   ↓ conditional coder              y_t = g_a(x_t | C_t),  decoder g_s(ŷ_t | C_t)
```

Conditioning on the warped context instead of subtracting a predicted frame lets the network decide how to use the reference; this is the DCVC family of designs. For calls, use a low-delay structure: every frame predicts from the previous decoded frame, with a long-term reference frame kept for loss recovery. For storage, hierarchical B-frames buy 20–30% more compression at the cost of latency.

### C. Entropy coding that can run on a phone

An autoregressive context model conditions every element on its decoded neighbours and serializes decoding; it is the wrong choice for a real-time decoder. Use the hyperprior plus a checkerboard or channel-wise context: decode half of the latent in parallel, then the other half conditioned on the first, in a handful of passes. The probabilities must be computed in integer arithmetic on both ends; see the deep dive.

### D. Rate control

A learned codec trained for one λ produces one rate; a product needs a target bitrate. Train one variable-rate model conditioned on a quality index, and let the encoder pick the index per frame with a controller that tracks a buffer, exactly as a classical rate controller picks a quantization parameter. Signal the index in the frame header.

### E. The bitstream

```text
stream header:  model id + version hash, resolution, frame rate, colour format, quality range
frame header:   frame type (I/P), reference ids, quality index, latent sizes, checksum
payload:        arithmetic-coded ẑ, then ŷ (motion latents first for a P-frame)
```

The model identifier is the format: two decoders with different weights are two different codecs. Decoders ship with every model version they support; a stream whose version is unknown is rejected at the header, and the sender falls back to HEVC.

### F. Put the pieces together

```text
Encoder (phone A)                                  Decoder (phone B)
frame → [I or P model] → latents → int entropy coder → bits → header check → int entropy decoder
       ↑ decoded reference kept                                        → [I or P model] → frame
       ↑ rate controller picks the quality index                       ↑ decoded reference kept
       ↑ content check: text / screen → switch to HEVC for this GOP
```

**Say:** "Encoder and decoder keep the same decoded references, the entropy coder is integer on both sides, and the model version is in the header. Those three sentences are the difference between a paper codec and a product."

## 7.7 Deep dive: bit-exact decoding and drift — minutes 27–33

Arithmetic coding needs the encoder and the decoder to compute the same probability table, bit for bit. Two chips that differ in the last floating-point bit of one activation produce a different table, the decoder reads the wrong symbol, and every following symbol of the frame is garbage. That is not a quality loss; it is a broken stream.

- **Integer entropy model.** Quantize the network that produces `(μ, σ)` to integer arithmetic with fixed rounding, tabulate the cumulative distributions, and test equality of the tables across every chip generation and OS build as a release gate. The transforms `g_a`, `g_s` may stay in floating point: a small difference in the reconstruction is tolerable, a difference in the probabilities is not.
- **Drift.** The decoder's reference frame is the decoded frame, not the input, so the encoder must run its own decoder in the loop and predict from the same reconstruction; train the P-frame model on decoded references as well. Insert an intra frame or a refresh region periodically so that any residual mismatch cannot accumulate forever.
- **Packet loss on calls.** A lost P-frame breaks every frame that references it. Keep a long-term reference on both ends and let the receiver report loss so that the encoder predicts the next frame from the last reference both sides still share; combine with forward error correction from [Networking & Streaming](05_networking_streaming.md).
- **Content outside the training distribution.** Sharp text, screen recordings and synthetic graphics are where learned codecs smear. The encoder classifies content per group of pictures and switches to HEVC when the classifier or the measured quality says so; the fallback is part of the design, not an admission of failure.

## 7.8 Deep dive: training and on-device deployment — minutes 33–39

```text
Data      licensed and synthetic video, every content class, no user data
Stages    1. image model on frames   2. motion model with the image model frozen
          3. everything jointly with decoded references   4. quality-index conditioning
          5. quantization-aware fine-tuning of the decoder and the entropy model
Ship      compile for the neural accelerator, tile 4K, pin the version hash, stage the rollout
```

- **Training.** Train the I-frame model first, then the motion codec, then the whole chain end to end with several consecutive frames so that errors in the decoded reference are seen during training. The quality-index conditioning gives one model for the whole rate range.
- **Making the decoder fit.** Distil the decoder into the multiply-add budget of 7.4: fewer channels, early downsampling, no autoregressive context. Then quantization-aware fine-tuning to 8-bit weights and activations for the accelerator, with the entropy model fully integer.
- **Deployment.** Compile the decoder for the neural accelerator, measure latency and energy per frame on every supported chip, and keep the model under a size that can ship inside an OS update. Roll out to calls between updated devices first, where the sender knows the receiver's version from the call setup; anything else negotiates down to HEVC.
- **Versioning.** A new model is a new format. Old streams stay decodable because decoders keep the old weights for as long as stored content can reference them; calls only ever use the newest version both ends share.
- **Measuring in production without seeing content.** Log decode time, dropped frames, fallback rate and the quality index chosen, aggregated on the device; never the frames.

Your own work is the evidence to bring here: the on-device quantization of a VLM for iOS, and GenStream's lesson that a generative codec that decodes too slowly does not ship.

## 7.9 Validate the design — minutes 39–42

| Check | How | Gate |
|---|---|---|
| Compression | BD-rate in PSNR and VMAF against hardware HEVC and AV1 on standard sets and on device recordings | at least 25% savings in the 1–6 Mb/s range, no class worse than HEVC |
| Perceived quality | subjective tests on the hardest content: faces, text, motion, low light | no significant loss against HEVC at equal bitrate |
| Real time | frames per second and ms per frame per chip, at sustained temperature | 30 fps for 1080p, no throttling in a 30-minute call |
| Energy | mJ per frame against the hardware codec path | within the agreed factor |
| Bit exactness | decode the same streams on every chip, compare hashes | identical |
| Robustness | fuzzed and truncated streams, packet loss traces, out-of-distribution content | no crash, fallback engages |

## 7.10 Interview-ready reference answer

> "First I would separate three meanings of ML-based: ML around a standard codec, ML inside it, and a learned codec that replaces it. The receiver decides which are possible: a learned codec needs a neural decoder on every device that will play the stream, within the power budget of a phone.
>
> For 1080p at 30 frames per second I would fix a decoder budget of a few thousand multiply-adds per pixel and a memory budget per frame, because published learned codecs spend a hundred times more than a phone can sustain. I would ship ML pre- and post-processing and ML encoder decisions first, since they need no decoder change, and build the learned codec as the research track with the same evaluation harness.
>
> The learned codec keeps the classical shape: an I-frame model with a hyperprior entropy model, a P-frame model that codes motion and then codes the frame conditioned on the warped reference, a parallel checkerboard context instead of an autoregressive one, and a variable-rate model driven by a rate controller. The bitstream carries the model version, the frame type and the quality index.
>
> Two things make it a product rather than a paper: the entropy model runs in integer arithmetic so decoding is bit-exact across chips, and encoder and decoder keep the same decoded references with periodic refresh so mismatch cannot accumulate. Content the model handles badly, such as text, switches to HEVC per group of pictures, and any version mismatch falls back to HEVC.
>
> I would train in stages, distil and quantize the decoder to the budget, roll out to calls between updated devices first, and gate releases on BD-rate against the hardware encoder, subjective tests, sustained frame rate and energy per frame, and hash-equal decoding on every chip."

## 7.11 Follow-up questions and concise answers

These are plausible follow-ups, not questions confirmed from the interview.

- **Why not just use AV1?** For general content at normal bitrates, use it; the hardware exists and the ecosystem is standard. The learned codec earns its place at low bitrates, in a controlled two-ended product such as calls, and wherever the codec can be specialized to a content domain.
- **How do you hit an exact bitrate?** One variable-rate model conditioned on a quality index, plus a buffer-based controller that picks the index per frame; the index plays the role of the quantization parameter of a classical codec.
- **What breaks first in production?** Non-deterministic probabilities across chips. The fix is an integer entropy model and a cross-chip hash test as a release gate, before any quality work.
- **Why does the decoder budget dominate?** The encoder runs once per frame on one device; the decoder runs on every receiver, on the same phone for calls, and in hardware terms it competes with a fixed-function block that costs milliwatts.
- **How do you make decoding parallel?** Drop the autoregressive context; use the hyperprior with a checkerboard or channel-wise context so the latent decodes in a few parallel passes.
- **What about 4K?** Tile the frame or run the network at reduced resolution with a learned upsampler; memory bandwidth, not multiply-adds, is usually the first wall.
- **Where does a generative decoder fit?** At very low bitrates a generative decoder produces convincing but not faithful pictures; that is acceptable for a call and unacceptable for a document or a medical recording, so it is a product decision per use case, the rate-distortion-perception trade-off of [ML §6.4.5](02_ml_domain.md#645-the-rate-distortion-perception-tradeoff).
- **Two phones with different model versions?** The call setup negotiates the newest version both sides have and otherwise uses HEVC; stored content carries its version and decoders keep old weights.
- **How do you know the training data is enough?** Evaluate by content class, not by a global average; the classes where a learned codec fails are the sharp synthetic ones, and those are where the fallback is measured.
- **What would you build first?** The evaluation harness and the bit-exact decoder path. Without the first there is no comparison against the hardware codec; without the second there is no product.

## 7.12 Interview script: what to say at each step

**Minutes 0–3 — restate**

> "I will treat this as a codec that has to ship on phones, not a research prototype. I will clarify what ML-based means here and which product it serves, set a compute and bit budget, lay out a roadmap where ML tools around a standard codec ship first, then design the learned codec and go deep on what makes it deployable."

**Minutes 3–7 — clarify**

> "Three questions change the design. Do you want ML that improves HEVC or a learned codec that replaces it? Is the target real-time calls, capture, or storage, because that decides whether the encoder and decoder share one phone's budget? And what is the baseline and the metric: I would measure bitrate savings at equal VMAF against the device's hardware encoder. Unless you say otherwise, I assume real-time 1080p calls, a 25% bitrate target, bit-exact decoding across devices, and a fallback to HEVC."

**Minutes 7–14 — budgets**

> "1080p at 30 frames per second is 62 million pixels a second. At a thousand multiply-adds per pixel that is 62 giga-MACs a second, fine for a phone accelerator; at a hundred thousand it is 6 tera-MACs, which is where published learned codecs are and which a phone cannot sustain. So the decoder budget is a few thousand per pixel and a bounded number of bytes moved per frame. On bits, a 2 Mb/s call is 0.03 bits per pixel, so a frame is about 60 kilobits; the entropy model's side information at a few percent is affordable, the model weights are not, so the stream carries a version identifier."

Transition: "That budget decides the architecture, so let me give the roadmap before the internals."

**Minutes 14–19 — roadmap**

> "Four stages by what can ship: ML pre- and post-processing around HEVC, ML decisions inside the encoder, learned tools inside a new standard, and a fully learned codec. The first two need no decoder change and ship this year; the learned codec is the research track, evaluated against the improved standard pipeline with the same harness."

**Minutes 19–27 — the learned codec**

> "It keeps the classical shape. An I-frame model: analysis transform, quantizer, hyperprior entropy model, synthesis transform. A P-frame model: a motion network, a small codec for the motion, warping of the reference features, and a frame coder conditioned on that context rather than subtracting a prediction. Entropy coding uses the hyperprior with a checkerboard context so it decodes in parallel. Rate control is one variable-rate model with a quality index chosen by a buffer controller. The stream header carries the model version, the frame type and that index."

Check in: "That is the complete path. I would like to spend the deep dive on bit-exact decoding and drift, unless you would rather look at training and deployment?"

**Minutes 27–33 — deep dive: bit-exact and drift**

> "Arithmetic coding needs identical probabilities on both ends; a last-bit floating-point difference between two chips corrupts the whole frame. So the entropy model runs in integer arithmetic with tabulated distributions, and hash-equal decoding on every chip is a release gate. Encoder and decoder predict from the same decoded reference, the model is trained on decoded references, and periodic refresh bounds any mismatch. For calls, a long-term reference and loss feedback let the encoder predict from a frame both sides still have. Text and screen content switch to HEVC per group of pictures."

**Minutes 33–39 — deep dive: training and deployment**

> "Train the image model, then the motion model, then everything jointly on decoded references with a quality index, then quantization-aware fine-tuning. Distil the decoder into the budget, compile it for the neural accelerator, measure milliseconds and millijoules per frame on every chip, and pin the version hash. Roll out to calls between updated devices, negotiate down to HEVC otherwise, and log only on-device metrics."

**Minutes 39–42 — validate and recap**

> "The gates are BD-rate against the hardware encoder on standard and device content, subjective tests on faces, text and motion, sustained 30 frames per second without throttling, energy per frame, hash-equal decoding on every chip, and fuzzed streams that never crash."

Then give the reference answer in 7.10 and name the trade-off: the learned codec wins compression at low bitrates and in controlled products, and pays for it with a neural decoder, a versioned format and a determinism problem that the standard codec does not have.

**Minutes 42–45 — questions**

Ask which stage the team is in today, and what fraction of the codec work is encoder-only versus new-format work; the answers say how much of this design is on their roadmap.

**Check-in lines to reuse**

- "Is the receiver a phone in the same call, or a server that can afford a big decoder?"
- "Should I go deeper on the entropy model, or on how this rolls out across device generations?"
- "Do you want the learned codec compared with the hardware encoder or with the reference software encoder?"
