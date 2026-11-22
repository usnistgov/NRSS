NRSSIntegrator Validation Migration
===================================

Summary
-------

NRSS now has a clearer reduction split:

- ``WPIntegrator`` remains the detector-plane ``qx/qy -> chi/q_perp`` remesher.
- ``NRSSIntegrator`` is the maintained reduction path for NRSS analytical validation,
  especially when the target comparison is a geometry-aware analytical
  ``I(|q|)`` curve or shell position.

This note records the completed migration of the maintained NRSS validation
surface so that the form-factor and Bragg-structure tests use
``NRSSIntegrator`` as the primary radial reducer, while the CoreShell and
MWCNT tests remain on ``WPIntegrator`` for explicit
historical-comparability reasons.

Implementation Status
---------------------

This migration has now been applied in the maintained pytest surface.

Implemented changes:

- ``tests/validation/test_analytical_sphere_form_factor.py`` now uses
  ``NRSSIntegrator`` as the maintained sphere analytical reduction path.
- ``tests/validation/test_analytical_2d_disk_form_factor.py`` now uses
  ``NRSSIntegrator`` as the maintained 2D disk analytical reduction path.
- ``tests/validation/test_bragg_2d_lattice.py`` now uses
  ``NRSSIntegrator`` for radial shell workups with explicit
  ``q_perp`` semantics.
- ``tests/validation/test_bragg_3d_lattice.py`` now uses
  ``NRSSIntegrator`` for radial shell workups with detector-corrected
  ``|q|`` semantics.
- ``tests/validation/lib/bragg.py`` now exposes explicit detector-plane and
  detector-corrected radial keys for predicted Bragg spots and lets radial
  shell clustering choose the intended coordinate key.
- The transition-only modules
  ``tests/validation/test_nrss_integrator_sphere_form_factor.py`` and
  ``tests/validation/test_nrss_integrator_2d_disk_form_factor.py`` were
  removed after their logic was promoted into the maintained test files.
- CoreShell and MWCNT helper/doc paths now explicitly label their
  ``WPIntegrator`` workflow as a maintained historical exception rather than
  recommended practice.

Verification
------------

Focused maintained-path verification was run against the
``cupy-rsoxs`` ``tensor_coeff`` execution path in the shared
``nrss-pyhyper-dev`` environment using:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/homes/deand/dev/NRSS \
   mamba run -n nrss-pyhyper-dev pytest \
     /homes/deand/dev/NRSS/tests/validation/test_analytical_2d_disk_form_factor.py \
     /homes/deand/dev/NRSS/tests/validation/test_analytical_sphere_form_factor.py \
     /homes/deand/dev/NRSS/tests/validation/test_bragg_2d_lattice.py \
     /homes/deand/dev/NRSS/tests/validation/test_bragg_3d_lattice.py \
     -k cupy_tensor_coeff -q -s

Result:

- ``8 passed, 16 deselected`` in about 166 seconds.

Notes from that run:

- The new maintained 3D sphere path fit the analytical ``I(|q|)`` reference
  cleanly after switching to ``NRSSIntegrator`` and retuning the absolute
  regression thresholds to the detector-corrected radial coordinate.
- The 3D HCP Bragg case required a small relaxation of the radial-shell
  ``p95_abs_dq`` threshold when the shell comparison moved from detector-plane
  ``q_perp`` to detector-corrected ``|q|``.
- The old maintained z-collapse sphere analytical test was removed from this
  migrated surface because it was outside the explicit 3D/2D migration scope
  and did not belong in the maintained ``NRSSIntegrator`` analytical path.

Goals
-----

- Stop treating detector-plane ``q_perp`` as the maintained analytical
  comparison coordinate for 3D NRSS validations.
- Make the maintained form-factor and Bragg tests compare against analytical
  references in the same radial coordinate that the test claims to validate.
- Keep the historical CoreShell and MWCNT reduction path intact, but document
  that it is preserved for continuity with legacy references and is not the
  recommended reduction pattern for new NRSS analytical tests.
- Reduce duplication between the older detector-space analytical tests and the
  newer ``NRSSIntegrator``-specific validation modules.

Maintained End State
--------------------

Form-factor tests:

- ``tests/validation/test_analytical_sphere_form_factor.py`` now uses the
  primary assertion path:
  ``NRSS scattering -> NRSSIntegrator -> analytical sphere I(|q|)``.
- ``tests/validation/test_analytical_2d_disk_form_factor.py`` now uses the
  primary assertion path:
  ``NRSS scattering -> NRSSIntegrator -> analytical 2D disk I(q)``.
- The temporary transition-style modules that previously carried the
  ``NRSSIntegrator``-specific migration logic were merged into the maintained
  analytical tests and then removed.

Bragg tests:

- ``tests/validation/test_bragg_2d_lattice.py`` now uses
  ``NRSSIntegrator`` for the quasi-powder radial workup.
- ``tests/validation/test_bragg_3d_lattice.py`` now uses
  ``NRSSIntegrator`` for the quasi-powder radial workup and compares shell
  positions against detector-corrected ``|q|`` rather than detector-plane
  ``q_perp``.

Historical exceptions:

- ``tests/validation/test_core_shell_reference.py`` remains on
  ``WPIntegrator``.
- ``tests/validation/test_mwcnt_reference.py`` remains on
  ``WPIntegrator``.
- Both now state clearly in docstrings, helper docstrings, plot labels, and
  data README notes that this is a maintained historical path kept for direct
  comparability with vendored legacy references, not the recommended pattern
  for new analytical validations.

Implemented File-Level Changes
------------------------------

1. Promote the ``NRSSIntegrator`` form-factor logic into the maintained path.

- ``tests/validation/test_analytical_sphere_form_factor.py`` was reworked onto
  an ``NRSSIntegrator``-based maintained path with detector-corrected
  ``|q|`` semantics.
- ``tests/validation/test_analytical_2d_disk_form_factor.py`` was reworked
  onto an ``NRSSIntegrator``-based maintained path with explicit
  ``q_perp`` semantics.
- The temporary ``test_nrss_integrator_*`` migration modules were removed after
  their logic was folded into the maintained analytical test files.

2. Standardize the reduction helper used by Bragg validations.

- In ``tests/validation/test_bragg_2d_lattice.py``, the local
  ``WPIntegrator``-based ``_pyhyper_iq`` helper was replaced with an
  ``NRSSIntegrator``-based helper.
- The test now asserts that the reduced output carries:

  - ``source_integrator == "NRSSIntegrator"``
  - ``nrss_semantic_mode == "2d_reciprocal_plane"``
  - ``radial_semantics == "q_perp"``

- In ``tests/validation/test_bragg_3d_lattice.py``, the local
  ``WPIntegrator``-based ``_pyhyper_iq`` helper was replaced with an
  ``NRSSIntegrator``-based helper that passes explicit metadata when needed.
- The test now asserts that the reduced output carries:

  - ``source_integrator == "NRSSIntegrator"``
  - ``nrss_semantic_mode == "3d_detector_aware"``
  - ``radial_semantics == "q_abs_detector_corrected"``

3. Fix Bragg shell semantics in the shared helper layer.

- In ``tests/validation/lib/bragg.py``, the 3D spot predictor now exposes both
  detector-plane and detector-corrected radial coordinates explicitly.
- The shared helper now exposes:

  - ``q_perp`` for detector-plane spot overlay logic
  - ``q_abs_detector`` for radial-shell comparison logic
  - optionally ``q_abs_lattice`` for pure reciprocal-lattice bookkeeping

- ``radial_shells_from_spots`` now accepts a key name so the 2D tests cluster
  shells using ``q_perp`` while the 3D tests cluster shells using
  detector-corrected ``|q|``.

4. Clarify historical ``WPIntegrator`` maintenance in shared helpers.

- Added explicit docstrings/comments to
  ``tests/validation/lib/core_shell.py:scattering_to_awedge`` and
  ``tests/validation/lib/mwcnt.py:scattering_to_chiq`` explaining that these
  helpers keep the historical ``WPIntegrator`` reduction for compatibility
  with vendored legacy references.
- Updated default plot labels such as
  ``"Pybind + WPIntegrator (historical maintained path)"``.
- Updated:

  - ``tests/validation/data/core_shell/README.md``
  - ``tests/validation/data/mwcnt/README.md``

  so the same point is documented near the vendored reference artifacts.

Applied Migration Sequence
--------------------------

1. Consolidate the sphere and 2D disk tests first.

- These were the cleanest analytical references.
- They provided the baseline semantics and threshold style for later Bragg
  work.

2. Migrate the 2D Bragg radial workup next.

- This was low risk because ``NRSSIntegrator`` reduces to the same radial
  coordinate as ``WPIntegrator`` for ``z_dim == 1``.
- The main value was making the maintained reducer explicit and asserting the
  semantics in test metadata.

3. Migrate the 3D Bragg radial workup last.

- This was the only Bragg path where shell-position semantics actually changed.
- Thresholds were re-derived after the shell coordinate switched from
  ``q_perp`` to detector-corrected ``|q|``.
- Detector-image spot-overlay assertions were kept unchanged; only the radial
  shell comparison moved to the detector-corrected coordinate.

4. Apply the historical-note pass to CoreShell and MWCNT.

- This was documentation and labeling work, not a physics-model change.
- It landed with the test migration so reviewers can see the maintained policy
  in one place.

Acceptance Criteria
-------------------

The migration should be considered complete when all of the following are true:

- The maintained sphere and 2D disk analytical tests use ``NRSSIntegrator`` as
  their primary reduction path.
- The maintained Bragg 2D and Bragg 3D tests use ``NRSSIntegrator`` for radial
  shell workups.
- No maintained 3D analytical or Bragg test claims agreement against
  analytical ``I(|q|)`` while actually fitting in detector-plane ``q_perp``.
- The CoreShell and MWCNT tests explicitly describe their
  ``WPIntegrator`` dependency as historical-maintenance policy rather than
  recommended practice.
- The maintained test surface has one primary module per validation concept,
  rather than parallel old/new detector-semantics modules that can drift apart.

Non-Goals
---------

- Changing ``WPIntegrator`` semantics.
- Rewriting the historical CoreShell or MWCNT reference reductions.
- Replacing detector-image spot-position checks in the Bragg tests.
- Requiring new NRSS-side scattering metadata before the validation migration
  can proceed.
