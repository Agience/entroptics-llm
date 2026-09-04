# entroptics-llm

[![CI](https://github.com/Agience/entroptics-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/Agience/entroptics-llm/actions/workflows/ci.yml)
[![Licence: AGPL-3.0-only or commercial](https://img.shields.io/badge/licence-AGPL--3.0--only%20or%20commercial-blue)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/entroptics-llm)](https://pypi.org/project/entroptics-llm/)
[![Python](https://img.shields.io/pypi/pyversions/entroptics-llm)](https://pypi.org/project/entroptics-llm/)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b)](research/PAPER.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22313961.svg)](https://doi.org/10.5281/zenodo.22313961)
[![Sponsor](https://img.shields.io/badge/sponsor-Agience-ea4aaa?logo=githubsponsors)](https://github.com/sponsors/Agience)

**Read the context as the directions it resolves, and the selection follows from the data.**

A language model's context is assembled from several sources at once: a retrieval-augmented
prompt, an agent scratchpad, a long conversation. The decisions made about it — how many
retrieved items to keep, which of them to trust, whether an answer stayed on its evidence — are
conventionally made with one number per item against one summary of the whole.

This reads the context as the **K directions it resolves above its own noise floor**, using the
[`entroptics`](https://github.com/Agience/entroptics) instrument, and makes each of those
decisions from that span. K is counted from the data: nothing is trained, no threshold is set,
and no constant is fitted per corpus.

---

## The result

**How many to keep, when relevance has several facets.** A query whose relevance spreads across
R independent facets, both sides running the same parameter-free lock over the same screen:

| facets R | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| cosine F1 | 1.000 | 1.000 | 1.000 | 0.806 | 0.000 | 0.000 |
| **entroptics F1** | 1.000 | 1.000 | 1.000 | **1.000** | **1.000** | **1.000** |
| edge | +0.000 | +0.000 | +0.000 | **+0.194** | **+1.000** | **+1.000** |

Entroptics holds F1 at 1.000 at every facet count. One scalar carries one facet's worth of
evidence; an R-mode subspace holds all of them.

**Which item is foreign.** Real attacks from `deepset/prompt-injections` planted in wikitext-103
articles, against a centroid and against a k-nearest-neighbour detector:

| sources | **entroptics** | centroid | k-NN | vs centroid | vs k-NN |
|---|---|---|---|---|---|
| 1 | **0.990** | 0.983 | 0.983 | +0.007 [+0.003, +0.012] | **+0.007** [+0.003, +0.011] |
| 2 | **0.970** | 0.901 | 0.960 | +0.069 [+0.052, +0.088] | **+0.010** [+0.004, +0.017] |
| 3 | **0.941** | 0.811 | 0.940 | +0.131 [+0.105, +0.157] | +0.001 [−0.007, +0.010] |
| 5 | 0.907 | 0.733 | 0.927 | +0.174 [+0.147, +0.201] | −0.020 [−0.031, −0.010] |

The margin over a centroid grows from +0.007 to +0.174 as sources multiply. On the question an
operator asks — which item do I open first — entroptics leads at every source count: **0.787 /
0.467 / 0.287 / 0.160** against the centroid's 0.740 / 0.253 / 0.140 / 0.053, three times the
hit rate at five sources from a pool of 192 items.

**Whether a continuation belongs.** Held-out on-topic sentences against sentences from a
different article:

| sources | **entroptics** | centroid | vs centroid |
|---|---|---|---|
| 1 | 0.955 | 0.954 | +0.001 [−0.002, +0.004] |
| 2 | **0.931** | 0.893 | **+0.038** [+0.031, +0.046] |
| 3 | **0.922** | 0.838 | **+0.084** [+0.074, +0.095] |
| 5 | **0.872** | 0.717 | **+0.155** [+0.141, +0.169] |

The same curve from an independent task: the centroid falls 0.954 → 0.717 as the context
spreads while entroptics holds 0.955 → 0.872.

```bash
python -m research.benchmarks.bench_facets       # runs on CPU in seconds, no corpus
python -m research.benchmarks.bench_injection
python -m research.benchmarks.bench_drift
```

`research/PAPER.md` has the construction, the real-corpus results on BEIR and HotpotQA, and
every number with what it was measured on.

## Use it

```python
from entroptics_llm import coherent_basis, in_subspace_fraction, most_anomalous, coherence

V, K = coherent_basis(context_embeddings)      # the K directions the context resolves
suspect = most_anomalous(context_embeddings)   # the item furthest outside them
score   = coherence(answer_embedding, context_embeddings)   # is this still on its sources?
```

`K` is read from the data — the count of singular values standing above the derived noise
floor — so nothing is chosen. `context_embeddings` are the ones your pipeline already computed
for retrieval; no additional model call is made.

**These are scores and rankings, not verdicts.** `most_anomalous` localises the item worth
inspecting. "Is there an anomaly in this one set" is ill-posed without a reference
distribution, so a caller wanting a yes/no supplies its own null.

## Install

```bash
pip install entroptics-llm
```

Runtime dependencies are **numpy and [`entroptics`](https://pypi.org/project/entroptics/)**,
nothing else. No encoder, no torch, no corpus: the vectors are yours, and the package never
downloads anything.

Working on it, or reproducing the tables:

```bash
pip install -e ".[dev]"      # + pytest
pytest -q                    # the whole suite, no model, no downloads
pip install -e ".[bench]"    # + encoders and datasets, for research/benchmarks/
```

## Layout

```
research/
  PAPER.md        the write-up, and every number with what it was measured on.
  benchmarks/
    bench_injection   locating a real injection, against source count.
    bench_drift       on-topic decision, against source count.
    bench_facets      the budget's win condition: relevance across R facets.
    bench_beir        the budget on real BEIR corpora, where relevance is unimodal.
    bench_hotpot_e2e  end to end with a language model answering.
src/entroptics_llm/
  engine.py       the one seam onto `entroptics`. Nothing else imports it.
  subspace.py     coherent basis, in-subspace fraction, anomaly, coherence.
  lock.py         gap_split / top_break — where a salience spectrum stops.
  cut.py          the retrieval budget: a query-relative screen, and the lock on cosine.
tests/            the suite, including the structural guards that read prose
```

## Where it ties, and why

**On unimodal relevance it ties, which is the same prediction.** At one coherent mode the read
reduces to the statistic it is measured against. Four real BEIR corpora over 2,421 queries: every interval contains zero. And
HotpotQA end-to-end with a language model, handed cosine's top-m where m is the count the cut
derived for that question: +0.000 (Qwen2.5-1.5B) and −0.001 (Qwen2.5-7B) — the selection is
indistinguishable. Only 12.2% of the 3,265 pooled BEIR queries are multi-modal at the point the
cut sees them, so a tie is what the law requires there.

**Size the budget to the task.** On HotpotQA the derived count spends 36% of a fixed top-5's
tokens; it keeps 2.1 paragraphs and 66 of 150 questions kept exactly one, where the answer needs
two. Against that fixed top-5 that is −0.133 F1 at 7B, all of it the count rather than the
ranking. Where a task needs several passages, put a floor on the count.

**Coherence reads topic, not factual correctness.** A wrong-but-on-topic answer reads as
coherent — on those same 150 questions it separated right answers from wrong ones by 0.008.
Use it for off-source and lost-thread monitoring, not as a fact checker.

## Built on entroptics

Every spectral read reaches [`entroptics`](https://github.com/Agience/entroptics) through one module,
[`engine.py`](src/entroptics_llm/engine.py); nothing else imports the library and a test
enforces it. The resolved-mode count that sets `K` is the library's own read — the singular
values standing above a noise floor derived from the data rather than assumed. See
`research/PAPER.md` for the citation and the construction.

## License

**Dual-licensed: AGPL-3.0-only *or* commercial.** See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE);
commercial and white-label terms in [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md).
Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CLA.md`](CLA.md).