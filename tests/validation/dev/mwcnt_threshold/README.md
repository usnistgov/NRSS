This directory is for non-official MWCNT threshold/falsification development work.

Nothing here is part of the maintained validation test lane.

Current purpose:
- generate nearby, deterministic alternate MWCNT geometries from the same
  tutorial-compatible RSA logic,
- compare those falsified scenarios to the maintained experimental validation
  thresholds,
- support decisions about whether the maintained thresholds should tighten.

Primary script:
- `run_mwcnt_threshold_probe.py`

Outputs are written under:
- `test-reports/mwcnt-threshold-dev/`
