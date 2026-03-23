Legacy validation workflows moved out of `tests/validation/` so they do not
participate in normal pytest discovery.

Files parked here:
- `test_CoreShell.py`
- `test_proj_sphere.py`
- `circle-lattice-test.py`

These are historical development-era validation scripts, not maintained pytest
modules. They still use legacy file-writing/manual-run patterns and should be
treated as source material for future migration work rather than active tests.
