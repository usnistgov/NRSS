This directory is for non-official MWCNT periodic-field/windowing development work.

Nothing here is part of the maintained validation test lane.

Current purpose:
- compare the legacy field-construction path against the new periodic-field path,
- evaluate whether the periodic-field morphology is a defensible basis for
  `WindowingType=0`,
- keep the comparison tied to the fixed-seed, tutorial-compatible MWCNT
  validation geometry used elsewhere in dev studies.

Primary script:
- `run_mwcnt_windowing_compare.py`

Outputs are written under:
- `test-reports/mwcnt-windowing-dev/`
