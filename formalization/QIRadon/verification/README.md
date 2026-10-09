# Verification — 2026-10-09

The reorganized library passed `lake build`. Before the first rebuild, the
project's entire `.lake/build` directory was removed; dependency-package caches
were retained. The final integrated build also passed after the proof
simplifications. A final style pass removed unused section instances, redundant simplifier
arguments, and dead tactic steps. The final build passes without warnings;
no linter was disabled to obtain this result.

`lake env lean Audit.lean` passed for **71 principal and supporting declarations**.
Every reported dependency is one of the standard foundational axioms:
`propext`, `Classical.choice`, and `Quot.sound`. Output is in [axioms.txt](axioms.txt).

`python3 scripts/check_structure.py` verifies the configured library root,
requires source for every local import, checks cycles and full reachability,
rejects archive imports, and scans active source for proof shortcuts.
[structure.json](structure.json) records:

- 143 active source modules, all reachable from `QIRadon.lean`.
- 13 direct public-entry imports, reduced from 54 without losing the dependency closure.
- No missing source imports, unreachable active files, or archive dependencies.

The cleanup moved three unused source files and 37 historical reports into
`archive/`, removed obsolete generated logs, and grouped 48 module paths by
mathematical responsibility. Declaration namespaces were preserved. The map in
`archive/module_moves.json` records the path changes.

Two proof simplifications followed the reorganization:

1. The kernel-specific Poisson formula now invokes the general scaled Poisson
   theorem rather than repeating its scaling proof.
2. Constrained and unconstrained per-spoke allocation share the exponential
   budget's first-order convexity inequality.

All prior theorem statements in these two edited proof modules were compared
with the saved baseline and are unchanged. Assumptions, bounds, and convergence
exponents were not weakened. The new shared allocation lemma is included in
the axiom audit. Historical independent reviews are preserved in the archive;
no subagents were used for this cleanup.

The argument and theorem map are the current explanation. Archive files are
historical or unfinished and carry no claim of current compilation. Generated
build logs stay local under this directory and are gitignored. No commit or push
was made.

The release includes the PDF organized around four main theorems, its LaTeX
source, and reproducible experiments. The corrected tanh contour argument and
its use after the existing entire sample preparation are human proofs in the
note, not new Lean declarations. No Lean source changed during this editorial
and empirical revision; the build and axiom evidence above refer to the same
source. The sharper benchmark-specific angular estimates are also proved in
the note without a claim of Lean coverage.
The release script packages only active source, current documentation, pinned
dependency metadata, empirical sources/results/figures, and verification evidence. It checks ZIP integrity
and records SHA-256 checksums in release_manifest.json.
Archived development material and build caches are excluded.
