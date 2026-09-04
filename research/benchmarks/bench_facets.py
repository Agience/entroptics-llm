"""Where the retrieval cut beats a fair cosine, and by how much.

The claim under test is narrow and it is the whole thing: the cut beats the standard
summary statistic **exactly when relevance is multi-modal**, the margin grows with how
multi-modal it is, and it ties at one mode.

The contest is made fair by construction. The **same screen** ``W`` is fed to both
sides — cosine is its row sum, the cut is its coherent-mode energy — and the **same
parameter-free lock** (``top_break``) is applied to both. So the only thing under test
is whether the screen precedes the lock. An earlier version of this result did not do
that: the cosine side ran a crippled lock, it kept ~1.2 items where the cut kept ~3.6,
and the reported "+0.038 win" was the lock rather than the screen. That is the single
most important methodological note in this repository.

The construction. A gold row is strong on the heads of ONE facet and empty elsewhere;
a distractor is weakly present on every head. As the facet count R grows a gold row's
TOTAL falls — it covers 1/R of the query's heads — toward the distractor total, which
is the failure a single scalar cannot avoid, while its facet PATTERN stays intact and
gold rows sharing a facet remain a coherent mode.

Run::

    python -m benchmarks.bench_facets
    python -m benchmarks.bench_facets --draws 100 --heads 32
"""
from __future__ import annotations

import argparse

import numpy as np

from entroptics_llm.cut import signal_power
from entroptics_llm.lock import top_break


def facet_screen(rng, n, H, R, n_gold, distractor=0.22, noise=0.04):
    per = max(1, H // R)
    facet = [rng.permutation(H)[:per] for _ in range(R)]
    W = np.full((n, H), float(distractor))
    gold = np.arange(n_gold)
    for i in gold:
        W[i, :] = 0.0
        W[i, facet[i % R]] = 1.0
    W += noise * np.abs(rng.standard_normal((n, H)))
    return np.clip(W, 0.0, None), gold


def f1(pred, true, n):
    p = np.zeros(n, bool)
    p[list(pred)] = True
    t = np.zeros(n, bool)
    t[list(true)] = True
    tp = (p & t).sum()
    if tp == 0:
        return 0.0
    prec, rec = tp / p.sum(), tp / t.sum()
    return 2 * prec * rec / (prec + rec)


def sweep(R, draws, n, H, n_gold, seed):
    cos, hc = [], []
    for s in range(draws):
        rng = np.random.default_rng(seed + s)
        W, gold = facet_screen(rng, n, H, R, n_gold)
        cos.append(f1(np.where(top_break(W.sum(1))[0])[0], gold, n))
        hc.append(f1(np.where(top_break(signal_power(W))[0])[0], gold, n))
    return np.array(cos), np.array(hc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws", type=int, default=40)
    p.add_argument("--items", type=int, default=60)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--gold", type=int, default=12)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--facets", default="1,2,3,4,5,6")
    args = p.parse_args()

    print(f"\nretrieval F1 by relevance-facet count  |  {args.items} items, "
          f"{args.gold} gold, {args.heads} heads, {args.draws} draws")
    print("same screen and the same parameter-free lock on both sides\n")
    print(f"{'facets R':>9}{'cosine F1':>12}{'entroptics':>12}{'edge':>9}{'95% CI':>18}"
          f"{'won':>7}")
    for R in [int(x) for x in args.facets.split(",")]:
        cos, hc = sweep(R, args.draws, args.items, args.heads, args.gold, args.seed)
        d = hc - cos
        # Percentile bootstrap over draws — the edge is a mean over a small sample and
        # quoting it without a spread is how "+0.038" got believed the first time.
        boot = np.array([np.random.default_rng(9000 + i).choice(d, len(d)).mean()
                         for i in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{R:>9}{cos.mean():>12.3f}{hc.mean():>12.3f}{d.mean():>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{f'{(d > 0).mean():.0%}':>7}")
    print("\nA CI that crosses zero is a tie, and a tie at one mode is the correct "
          "result:\nat one coherent mode the subspace IS the centroid.\n")


if __name__ == "__main__":
    main()
