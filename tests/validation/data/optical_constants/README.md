Vendored optical-constant files used by validation helpers and archived
development diagnostics.

Files here are test assets, not pytest modules, so they stay under
`tests/validation/data/` rather than at the top level of `tests/validation/`.

Current contents:
- `PEOlig2018.txt`: optical constants referenced by legacy sphere/circle
  validation helpers through `tests/validation/lib/generateConstants.py`.
