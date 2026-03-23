This directory is for non-official MWCNT RSA development work.

Nothing here is part of the maintained validation test lane.

Current purpose:
- benchmark alternate straight-CNT RSA geometry-generation strategies against the
  tutorial MWCNT parameter set,
- keep the candidate stream fixed so different collision-detection approaches can
  be compared fairly,
- identify a faster geometry-generation path before considering any transplant
  into the maintained MWCNT validation helper.

Primary script:
- `benchmark_mwcnt_rsa.py`

Outputs are written under:
- `test-reports/mwcnt-rsa-dev/`
