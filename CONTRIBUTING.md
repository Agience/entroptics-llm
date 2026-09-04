# Contributing to entroptics-llm

Reading a language-model context as the K directions it resolves, and taking the selection
from that span. **entroptics is the engine, not a dependency.**

## Rule zero: check entroptics first

Read the installed package - `batch.py`, `entropy.py`, `projection.py`, `reads.py`,
`null_providers.py`, `aperture.py` - or the source at
<https://github.com/Agience/entroptics>. Grep it **before** writing any spectral, statistical
or scaling primitive, not after a test fails. If a library read genuinely does not fit,
**measure** that and show the number. Two of this repo's five closed design questions closed
*against* the hypothesis that raised them, and only because they were measured.

## One seam

Every spectral read goes through `engine.py`, which is the only name in
`tests/test_no_shadow.py::ALLOWED`. `entroptics` is imported nowhere else.

```bash
python -m pytest -q
PYTHONPATH=/path/to/agience-mantle/src pytest tests/test_conformance_mantle.py
```

`test_no_shadow.py` asserts that boundary. `test_conformance_mantle.py` measures agreement with
`mantle.search.beacon` - the other implementation of these reads. **A second implementation does not
have to be wrong the day it is written; it only has to drift.**

## Measurement discipline

- **No tuned constants in the decision path.** Anything still typed names the error or geometry it
  comes from.
- **Measure, do not fit.** Nothing here trains.
- Every number quoted anywhere carries what it was measured on, changed in the same commit.
- Every comparison carries its baselines. The paper measures against a centroid and against
  a k-nearest-neighbour detector, and reports both wherever they disagree.
- **Prose is checked too.** `test_prose_does_not_drift.py` refuses any document that describes
  a practice this code no longer has. State a removed one in the past tense with its date.

## On the licence split

`entroptics` is Apache-2.0 because it is the instrument and is meant to be depended on. This repo
*applies* it, which puts it on the product side. Do not invert that direction, and do not copy
entroptics source here to work around a missing export - add the export upstream.

## Contributing

**Sign the CLA** - this project is AGPL-3.0-only **or** commercially licensed
([`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md)), so the project must hold the right to relicense
every line it ships. The bot checks on PR open and links [`CLA.md`](CLA.md).

Fork, branch from `main`, sign off every commit (`git commit -s`), open a PR. Commit format:
`fix:` / `feat(scope):` / `docs:` / `test:` / `chore:`.

**Security vulnerabilities: do not open a public issue** - email **connect@agience.ai**.

## License

**Dual-licensed: AGPL-3.0-only or commercial.** See [`LICENSE`](LICENSE),
[`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) and [`NOTICE`](NOTICE).
