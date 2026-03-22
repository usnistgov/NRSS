This directory is for non-official MWCNT EAngleRotation development work.

Nothing here is part of the maintained validation test lane.

Current purpose:
- reuse the dynamic seed-12345 MWCNT geometry from dev validation work,
- run single-seed validation panels for multiple `EAngleRotation` step sizes,
- compare fit quality and runtime while avoiding duplicated 0/360 sampling.

Primary script:
- `run_mwcnt_eangle_step_sweep.py`

Outputs are written under:
- `test-reports/mwcnt-eangle-dev/`
