"""Reading a context that spans several sources as a subspace rather than a centre.

A context assembled from several documents — a retrieval-augmented prompt, an agent
scratchpad, a long conversation — has no single direction. Its centroid sits between its
sources, in a direction none of them occupies, and every guard built on that centroid gets
worse as the context spreads. The subspace the context resolves above its own noise floor
spans the sources instead, so legitimate material lies inside it however many topics it
covers and foreign material does not. Nothing is trained, no threshold is set, and no
constant is fitted per corpus.

Quick start
-----------
::

    from entroptics_llm import coherent_basis, most_anomalous, coherence

    V, K = coherent_basis(context_embeddings)    # the K directions the context resolves
    suspect = most_anomalous(context_embeddings) # the item furthest outside them
    score = coherence(answer_embedding, context_embeddings)

When it beats a centroid
------------------------
Exactly when the context spans **more than one source**, and the margin grows with the
number of sources. On real prompt injections the edge over a centroid is +0.069, +0.131 and
+0.174 AUROC at 2, 3 and 5 sources; on the on-topic decision, +0.038, +0.084 and +0.155. The
largest margin is on selection: where relevance spreads across several facets the derived cut
holds F1 at 1.000 while a cosine cut falls to 0.000. research/PAPER.md has the construction
and what every number was measured on.

``coherence`` reads topic coherence rather than factual correctness, and the anomaly reads
rank items within one set rather than returning a verdict about it.

Layout
------
``engine``     the one seam onto `entroptics`. Nothing else imports it.
``subspace``   coherent basis, in-subspace fraction, anomaly, coherence.
``lock``       ``gap_split`` / ``top_break`` — where a salience spectrum stops.
``cut``        the retrieval budget: a query-relative screen, and the lock on cosine.
               A separate and weaker result; see the scope note in README.md.
"""
from __future__ import annotations

from . import cut, engine, lock, subspace
from .cut import (cosine_salience, derive_heads, head_screen, select,
                  select_by_cosine, select_with_info, signal_power)
from .engine import CutUnavailable, focus, occupancy, resolved_rank
from .lock import gap_split, top_break
from .subspace import (
    anomaly,
    anomaly_rank,
    anomaly_z,
    chance_overlap,
    coherence,
    coherent_basis,
    grounding,
    in_subspace_fraction,
    most_anomalous,
    novelty_score,
    subspace_overlap,
)

__version__ = "0.1.0"

__all__ = [
    # modules
    "engine", "lock", "cut", "subspace",
    # engine
    "CutUnavailable", "resolved_rank", "occupancy", "focus",
    # the lock
    "gap_split", "top_break",
    # the retrieval cut
    "select", "select_by_cosine", "cosine_salience",
    "select_with_info", "head_screen", "signal_power", "derive_heads",
    # set geometry
    "coherent_basis", "in_subspace_fraction", "subspace_overlap", "chance_overlap",
    "coherence", "grounding", "anomaly", "anomaly_rank", "most_anomalous", "anomaly_z",
    "novelty_score",
]
