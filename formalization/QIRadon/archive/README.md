# Archive — excluded from the verified proof

The Lean library is rooted at `QIRadon/`, and its public entry is `QIRadon.lean`.
Nothing in this archive is imported by that entry or by `Audit.lean`.
`python3 scripts/check_structure.py` checks the complete local import closure,
requires every active module to be reachable, and rejects archive imports.

- `history/reports/`: dated implementation reports, old audits, and progress notes.
- `history/README.md`, `history/STATUS.md`: the pre-consolidation overview and status.
- `unused/`: three former library files absent from the verified dependency graph.
- `scratch/`: local probes, unfinished numerical/tanh work, and the inspected external library. This directory is gitignored and is not verified.
- `module_moves.json`: old-to-new module paths for this cleanup. Public declaration namespaces were preserved.

Historical reports describe their state at the time they were written; their
relative paths and completion claims may be stale. The current argument and
scope are in `docs/`, with build evidence in `verification/`.
