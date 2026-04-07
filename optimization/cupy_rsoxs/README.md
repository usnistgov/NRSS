# `cupy-rsoxs` Optimization Index

Open this file first for any `cupy-rsoxs` optimization task.

Do not open the archive by default. The archive exists only to reconstruct a
specific historical experiment that is not captured in the compact docs below.

## Routing Table

| Need | Open First | Usually Do Not Open |
| --- | --- | --- |
| Current maintained optimized state | `accepted_state.md` | archive, old root stubs |
| Remaining work that has not been cleanly closed | `remaining_untried_ideas.md` | minimal history docs unless an idea needs provenance |
| `tensor_coeff` optimization history | `tensor_coeff_minimal_history.md` | direct-path history, archive |
| `direct_polarization` optimization history | `direct_polarization_minimal_history.md` | tensor history, archive |
| Benchmark surface, commands, acceptance gates | `benchmarking_guide.md` | archive harness README |
| Maintained validation/path-matrix/program status | `validation_and_status.md` | archive repo plan |
| `z_collapse_mode`, half-input, or other approximation tradeoffs | `fast_approximations.md` | archive proposal unless redesigning the approximation |
| Backend contract or runtime semantics unrelated to optimization ranking | `backend_spec.md` | optimization archive |

## Reading Rules

1. Start with exactly one target document from the table above.
2. Open a second document only if the first one explicitly points you there.
3. Do not open anything under `archive/` unless:
   - a compact history doc says detail is missing,
   - or you must reconstruct a specific historical measurement or artifact path.
4. Treat the old root-level files as compatibility stubs only.

## Document Map

- `accepted_state.md`
  - maintained defaults
  - current accepted optimization state
  - current caveats that still matter
- `remaining_untried_ideas.md`
  - short list of still-open ideas
  - excludes already-landed work
  - marks partially tried items separately from truly untried ones
- `tensor_coeff_minimal_history.md`
  - compact accepted / rejected / still-open record for `tensor_coeff`
- `direct_polarization_minimal_history.md`
  - compact accepted / rejected / still-open record for `direct_polarization`
- `benchmarking_guide.md`
  - authoritative timing lanes
  - required comparison commands
  - acceptance gates
- `validation_and_status.md`
  - compact maintained validation/program status
  - path-matrix and residency summary
- `fast_approximations.md`
  - expert-only approximation modes
  - `z_collapse_mode`
  - half-input / mixed-precision caveats
- `archive/`
  - preserved long-form historical documents
  - consult only when the compact docs are insufficient
