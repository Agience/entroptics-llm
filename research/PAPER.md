# Selection for language-model context, read from the data

**Using entroptics to decide how many retrieved items to keep, which to keep, and where foreign
material sits**

Ikailo Inc. — 2026-09-04

---

## Abstract

A language model's context is assembled from several sources at once: a retrieval-augmented
prompt, an agent scratchpad, a long conversation. The decisions made about it — how many
retrieved items to keep, which of them to trust, whether an answer stayed on its evidence — are
conventionally made with one number per item against one summary of the whole: a cosine, a
centroid, a fixed top-$k$.

This work reads the context as the $K$ directions it resolves above its own noise floor, using
the `entroptics` instrument, and makes each of those decisions from that span. $K$ is counted
from the data, so nothing is trained, no threshold is set, and no constant is fitted per corpus.

**Where relevance spreads across several facets, the derived cut holds $F_1$ at 1.000 while a
cosine cut falls to 0.000.** On locating a real prompt injection planted in real documents, it
names the attack as the first item to inspect three times as often as a centroid does at five
sources, and leads a k-nearest-neighbour detector outright at one and two sources. On deciding
whether a continuation belongs to its context, it leads a centroid by a margin that grows from
one source to five.

One statement covers all of it. **These reads beat a single summary statistic exactly when the
deciding distribution is multi-modal, and the margin grows with how multi-modal it is.** Where
the deciding distribution has one mode they reduce to the statistic they are measured against
and tie, which four BEIR corpora and an end-to-end HotpotQA run confirm.

---

## 1. The claim

> **A read taken over the $K$ directions a context resolves decides better than one taken
> against a single summary of it, exactly when the deciding distribution is multi-modal, and
> the advantage grows with how multi-modal it is.**

Three measurements carry it, on three different axes of modality: facets of a query's relevance
(§4), sources in a context (§5), and topics spanned by a conversation (§6). Two real-corpus
measurements confirm the tie where a distribution has one mode (§7).

## 2. A span in place of a point

Take a context $C = \{x_1, \dots, x_n\}$ of direction-normalised embeddings drawn from $M$
distinct sources. The centroid

$$\bar{x} = \frac{1}{n}\sum_i x_i \Big/ \Big\lVert \frac{1}{n}\sum_i x_i \Big\rVert$$

is a single point on the sphere, and ranking by cosine to it is identical to ranking by average
similarity to the rest of the set — the standard summary an item is usually scored against.
Where the sources occupy $M$ well-separated directions, $\bar{x}$ is their weighted mean and for
$M > 1$ it points into the gap between them. The cosine of a legitimate item from source $m$
then falls roughly as $\cos\theta$, with $\theta$ that item's angle from the consensus
direction — a quantity set by **how spread the context is**, carrying no information about the
item.

A subspace read puts a span where that point was. Let $X \in \mathbb{R}^{n \times d}$ stack the
context, let $X = U\Sigma V^\top$, and take

$$V_K = [v_1, \dots, v_K], \qquad K = \#\{\,\sigma_j : \sigma_j \text{ resolves above the noise floor}\,\}.$$

Score an item by the energy it retains under projection:

$$\phi_{\text{in}}(y) = \frac{\lVert V_K^\top y \rVert^2}{\lVert y \rVert^2} \in [0, 1].$$

Each resolved source direction lies in $\operatorname{span}(V_K)$, so a legitimate item from
*any* source scores high: its own direction is represented whether or not the others are. The
span admits every source at once, and the margin over a single-point summary widens as the
sources multiply, because the point moves further from all of them while the span keeps holding
each.

## 3. Choosing $K$ without choosing anything

$K$ is the one quantity that could be tuned, and it is read rather than set. It comes from the
singular spectrum via the `entroptics` instrument [1], which counts the modes standing above the
noise floor derived from the block's own shape. For an $n \times d$ block of pure noise the
singular values follow a Marchenko-Pastur law with a Tracy-Widom edge; modes above that edge are
the ones the block resolves. The count is a property of the data and moves with the context: in
§5 it reads 5.6 modes at one source and 12.5 at five, without being told how many sources there
were.

The block is read **as recorded**, column means included. This read counts modes and a constant
level is a mode: in the optics the library speaks it is the zero-order beam, physically present
and passed by the aperture like any other. For a set of embeddings that matters directly,
because **the shared mean direction is the topic the set is about** — it is the leading coherent
mode, and it carries signal.

Absence is handled by masking. A row or column that was never measured is dropped before the
decomposition and never reaches the denominator. A channel of measured zeros is an observation
of no power and counts as a mode. The two are distinct inputs and give distinct values of $K$,
which is what a caller with sparse features needs to know.

## 4. How many to keep, when relevance has several facets

The largest margin in this work is on selection. A query whose relevance is spread across $R$
independent facets is the multi-modal case at its cleanest, and $R$ is a controlled variable:
the facets are planted, so the axis is exact rather than estimated.

Both sides run the same parameter-free lock over the same head screen; the only difference is
whether the screen precedes the lock (`research/benchmarks/bench_facets.py`, 60 items, 12 gold,
40 draws).

| facets $R$ | cosine $F_1$ | **entroptics** $F_1$ | edge | 95% CI |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | $+0.000$ | [$+0.000$, $+0.000$] |
| 2 | 1.000 | 1.000 | $+0.000$ | [$+0.000$, $+0.000$] |
| 3 | 1.000 | 1.000 | $+0.000$ | [$+0.000$, $+0.000$] |
| 4 | 0.806 | **1.000** | $\mathbf{+0.194}$ | [$+0.099$, $+0.301$] |
| 5 | 0.000 | **1.000** | $\mathbf{+1.000}$ | [$+1.000$, $+1.000$] |
| 6 | 0.000 | **1.000** | $\mathbf{+1.000}$ | [$+1.000$, $+1.000$] |

**Entroptics holds $F_1$ at 1.000 at every facet count.** The cosine cut matches it while the
relevance has one to three facets and falls to 0.000 by five. The reason is §2's, transposed
from a context to a query: one scalar carries one facet's worth of evidence, and each gold
item's total weakens toward the distractor level as the facets multiply, where an $R$-mode
subspace holds all of them at once.

The count kept is derived per query throughout. Nothing is swept, and the same lock produces
both columns.

## 5. Which item is foreign, when a context spans several sources

The second axis of modality is the number of sources a context is assembled from. Legitimate
content is whole wikitext-103 articles, so each source is a genuine topic; the attacks are the
`deepset/prompt-injections` corpus, strings written to attack real systems. One attack is
appended to a context built from $M$ sources and the read ranks it. Encoder is
`bge-small-en-v1.5`, 150 trials per row, intervals paired per trial
(`research/benchmarks/bench_injection.py`).

Two baselines. The **centroid** is the standard single summary of §2. **k-NN** scores an item by
its similarity to its five nearest neighbours — a local read, so it carries no penalty for a
context that spans topics, and it is the stronger of the two.

| sources | modes | items | **entroptics** | centroid | k-NN | vs centroid | vs k-NN |
|---|---|---|---|---|---|---|---|
| 1 | 5.6 | 39 | **0.990** | 0.983 | 0.983 | $+0.007$ [$+0.003$, $+0.012$] | $\mathbf{+0.007}$ [$+0.003$, $+0.011$] |
| 2 | 7.5 | 78 | **0.970** | 0.901 | 0.960 | $+0.069$ [$+0.052$, $+0.088$] | $\mathbf{+0.010}$ [$+0.004$, $+0.017$] |
| 3 | 9.4 | 115 | **0.941** | 0.811 | 0.940 | $+0.131$ [$+0.105$, $+0.157$] | $+0.001$ [$-0.007$, $+0.010$] |
| 5 | 12.5 | 192 | 0.907 | 0.733 | 0.927 | $+0.174$ [$+0.147$, $+0.201$] | $-0.020$ [$-0.031$, $-0.010$] |

**Against the standard summary the margin grows from $+0.007$ to $+0.174$**, every interval
excluding zero, exactly as §2 predicts: the centroid falls from 0.983 to 0.733 as the sources
multiply while entroptics holds from 0.990 to 0.907.

**Against a local detector entroptics leads where the context is tightest** — $+0.007$ and
$+0.010$ at one and two sources, both intervals excluding zero — and the two converge as the
context widens.

The operator's question is which item to open first, and there entroptics leads at every
source count:

| sources | **entroptics** | centroid | k-NN |
|---|---|---|---|
| 1 | **0.787** | 0.740 | 0.707 |
| 2 | **0.467** | 0.253 | 0.380 |
| 3 | **0.287** | 0.140 | 0.267 |
| 5 | **0.160** | 0.053 | **0.160** |

At five sources it names the attack first three times as often as the centroid does, from a pool
of 192 items, and matches the local detector. Each item is scored against the subspace of the
set without it; Appendix A gives the reason.

## 6. Whether a continuation belongs to its context

The third axis is a conversation's topic spread, and the question is non-adversarial: **is this
continuation on-topic?** Held-out sentences from the context's own sources — on-topic, and held
out of the context — are scored against sentences from a different article. Same encoder, 150
trials per row, six held-out sentences per source so the target count resolves every effect
reported (`research/benchmarks/bench_drift.py`).

| sources | modes | context | **entroptics** | centroid | k-NN | vs centroid |
|---|---|---|---|---|---|---|
| 1 | 2.0 | 12 | 0.955 | 0.954 | 0.960 | $+0.001$ [$-0.002$, $+0.004$] |
| 2 | 2.8 | 24 | **0.931** | 0.893 | 0.945 | $\mathbf{+0.038}$ [$+0.031$, $+0.046$] |
| 3 | 3.8 | 36 | **0.922** | 0.838 | 0.942 | $\mathbf{+0.084}$ [$+0.074$, $+0.095$] |
| 5 | 5.8 | 60 | **0.872** | 0.717 | 0.909 | $\mathbf{+0.155}$ [$+0.141$, $+0.169$] |

**The same curve, from an independent task.** The edge over the standard summary opens at two
sources and grows to $+0.155$ at five, every interval from two upward excluding zero, with the
centroid falling 0.954 → 0.717 while entroptics holds 0.955 → 0.872. §5 appends foreign text and
this withholds native text; they were measured separately and agree on where the crossover sits
and on how the margin grows.

A local neighbour read leads on this task throughout, by $0.005$ to $0.038$. Ranking a target
against its nearest context sentences is a close fit to the question as posed, and the value
entroptics adds here is the count it derives rather than the ordering it produces — §4 and §7
measure that directly.

## 7. Real corpora, where relevance has one mode

The law predicts a tie wherever the deciding distribution has one mode, because the read
reduces to the statistic it is measured against. Two real-corpus measurements confirm it.

**Four BEIR corpora** — nfcorpus, scifact, arguana, fiqa, 2,421 queries — against the identical
`top_break` lock run on plain cosine salience, so the screen is the only difference
(`research/benchmarks/bench_beir.py`):

| dataset | cosine + lock | **entroptics** | edge | 95% CI |
|---|---|---|---|---|
| nfcorpus | 0.295 | 0.292 | $-0.003$ | [$-0.017$, $+0.010$] |
| scifact | 0.571 | 0.557 | $-0.013$ | [$-0.029$, $+0.001$] |
| arguana | 0.257 | 0.255 | $-0.002$ | [$-0.008$, $+0.004$] |
| fiqa | 0.344 | 0.342 | $-0.002$ | [$-0.012$, $+0.007$] |

Four ties, every interval containing zero. At the point the cut sees them, 12.2% of the 3,265
pooled queries carry four or more relevant documents in their horizon, so the deciding
distribution has one mode and §4's condition is absent.

**HotpotQA end-to-end**, 150 questions, a language model answering from the selected context
(`research/benchmarks/bench_hotpot_e2e.py`). The decisive arm is cosine's top-$m$, where $m$ is
the count entroptics derived *for that question*, so the only difference is which items were
chosen:

| model | fixed $k=5$ | cosine top-$m$ | **entroptics** | entroptics − top-$m$ |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.487 | 0.436 | 0.436 | $+0.000$ [$+0.000$, $+0.000$] |
| Qwen2.5-7B | 0.676 | 0.545 | 0.543 | $-0.001$ [$-0.020$, $+0.019$] |

The selection matches cosine's at both model sizes, as the law requires on unimodal relevance.

**What the derived count buys where the reads tie is the count itself.** The best single
constant $k$ gives up as much as $0.086$ $F_1$ against a corpus's own tuned $k$; reading the
count per query gives up at most $0.019$. That is the value where there is no sweep to run — a
service answering against more than one corpus, or a corpus whose content moves.

**Size the budget to the task.** On HotpotQA the derived count spends 36% of a fixed top-5's
context tokens, keeping 2.1 paragraphs where a multi-hop answer needs two gold paragraphs.
Against that fixed top-5 the difference is $-0.133$ [$-0.202$, $-0.067$] at Qwen2.5-7B, and it
is the count rather than the ranking: the matched-budget arm above shows the ranking already
matches the baseline. Where a task needs several passages, set a floor on the count.

## 8. What this costs

One thin SVD of the context block, $O(n d \min(n,d))$, on embeddings the pipeline has already
computed for retrieval. No additional model call, no second encoder, no index. Scoring an item
against the basis is a $d \times K$ matrix-vector product. For the context sizes here — tens to
a few hundred items — the read is well under the cost of the retrieval that produced them.

The anomaly read of §5 scores each item against the subspace of the set without it, which is
$n$ decompositions rather than one. Appendix A gives the reason and the alternative.

## 9. Scope

**Coherence reads topic.** An answer that stays on its sources' subject scores high whether or
not it is correct: on the 150 HotpotQA questions, coherence separated right answers from wrong
ones by $+0.008$ at 1.5B and $+0.010$ at 7B. It is a monitor for off-source and lost-thread
behaviour.

**Scores and rankings.** The reads order items within one context. "Is there an anomaly in this
set at all" is a question about a reference distribution, which a caller supplies;
`anomaly_z` takes one.

**Measured on one encoder.** All embedding numbers are `bge-small-en-v1.5`, and the end-to-end
tables are Qwen2.5-1.5B and Qwen2.5-7B. The mechanism in §2 is a statement about
direction-normalised embeddings generally; the magnitudes are measured on these models, and $K$
depends on how a model spreads a topic.

## 10. Reproducing

```bash
pip install -e ".[bench]"
python -m research.benchmarks.bench_facets       # §4
python -m research.benchmarks.bench_injection    # §5
python -m research.benchmarks.bench_drift        # §6
python -m research.benchmarks.bench_beir         # §7
```

The end-to-end table in §7 needs a language model as well:

```bash
pip install -e ".[bench,bench-llm]"
python -m research.benchmarks.bench_hotpot_e2e --model Qwen/Qwen2.5-7B-Instruct --n 150
```

`bench_facets` runs on CPU in seconds and needs no corpus. The others download their data from
Hugging Face. Every number in this paper carries the conditions it was measured under.

---

## Appendix A. Two measurements the API's shape rests on

**A.1 — Each item is scored against the subspace of the set without it.** `anomaly` leaves the
item out before building the span. Scoring an item against a span it helped define gives the
item its own coherent mode once it is foreign enough to clear the noise floor, at which point it
reads as maximally normal. One injected item in a clean multi-topic set, 15 draws, top-1
localisation:

| injected item is | read | 1 topic | 3 topics | 5 topics |
|---|---|---|---|---|
| 70% off the topics | either | 1.00 | 1.00 | 1.00 |
| 90% off | in-set | 1.00 | **0.80** | 1.00 |
| 90% off | leave-one-out | 1.00 | 1.00 | 1.00 |
| fully orthogonal | in-set | 1.00 | **0.07** | 1.00 |
| fully orthogonal | leave-one-out | 1.00 | 1.00 | 1.00 |

The effect moves with whether the resolved rank reaches the outlier's own mode, which a caller
cannot predict from anything visible to them, so leave-one-out is the default: it scored 1.00 in
every cell. The in-set read is available for latency and is sound where the set holds no hard
outlier, which is the ordinary case for text — real injections land inside the embedder's cone
and a partial outlier does not earn a mode.

**A.2 — A subspace overlap is read against its null.** In high dimension two subspaces that are
not specifically aligned are near-orthogonal, so a raw `subspace_overlap` reports "most of the
structure lies outside" for a random matrix as readily as for a real one, measuring the ambient
dimension. The analytic null is $k_b/d$ — a random $k_b$-dimensional subspace covers that
fraction of anything — and `chance_overlap` returns it. Report the ratio of measured overlap to
chance.

---

## References

[1] **entroptics** — the spectral instrument. Resolved-mode counting against a
Marchenko-Pastur/Tracy-Widom noise floor, and the occupancy read used here to set $K$.
Apache-2.0. <https://pypi.org/project/entroptics/>

[2] **wikitext-103** — Merity et al., *Pointer Sentinel Mixture Models*, ICLR 2017.

[3] **deepset/prompt-injections** — a corpus of real prompt-injection strings.
<https://huggingface.co/datasets/deepset/prompt-injections>

[4] **BEIR** — Thakur et al., *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of
Information Retrieval Models*, NeurIPS 2021.

[5] **HotpotQA** — Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
Answering*, EMNLP 2018.

## Citation

```bibtex
@techreport{ikailo2026entropticsllm,
  title  = {Selection for language-model context, read from the data},
  author = {{Ikailo Inc.}},
  year   = {2026},
  note   = {https://github.com/Agience/entroptics-llm}
}
```
