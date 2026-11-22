CuPy GPU memory and material count
==================================

For full-depth float32 simulations the CuPy backend normally uses the compact
tensor Fourier basis for one or more energies. It builds a basis from the
material fields once and keeps the Fourier samples required by the detector.
The same basis also serves the electric-field directions specified by
``EAngleRotation`` in ``[StartAngle, IncrementAngle, EndAngle]`` order.

Use ``SFieldMode.ISOTROPIC`` for a fully isotropic material. CuPy stages only
its ``Vfrac`` field and uses one basis field. A material with orientation
fields uses five basis fields for a single axis-aligned polarization and
seven when the tensor requires both in-plane polarizations, as for an
oblique direction or a sweep covering both axes. The anisotropic basis
contains the volume fraction, the volume fraction weighted by the order
parameter, and the required orientation products.

Estimating the retained basis
-----------------------------

Let ``I`` and ``A`` count the isotropic and anisotropic materials, ``C``
be the number of tensor components (three or five), and ``F`` be the
number of basis fields, ``I + (2 + C) * A``. Let ``U`` be the number of
distinct Fourier grid points sampled by the detector across all energies.
A single plane uses one grid point per valid detector pixel. A volume uses
two grid points per valid pixel per energy and shares points where the
energies map to the same endpoints. Each additional anisotropic material
therefore costs about ``8 * (2 + C) * U`` bytes in retained Fourier
samples and an isotropic material costs about ``8 * U`` bytes. Including
the shared index array and the mixing coefficients for ``E`` energy entries,
the retained basis occupies ``8 * U * (F + 1) + 8 * E * C * F`` bytes.

Peak GPU memory also includes staged material volumes, FFT buffers and
workspace, detector geometry and panels, and any device-resident results.
Windowed compact tensor runs retain Hann factors with shapes
``(NumZ, NumY, 1)`` and ``(1, 1, NumX)`` during basis construction.
The compact projector reads the retained Fourier samples directly, and the
dense tensor path stages the same material fields and reuses its transform
workspace across materials. In a thin model where the detector samples
most of the Fourier grid, the retained basis can outweigh the compact
transform workspace savings. The crossover depends on the fraction of
sampled grid points, the material mix, and the workspace selected by cuFFT.
Increasing the in-plane dimensions with the same depth and detector
coverage increases the basis and volume sizes together. Adding depth
generally lowers the fraction of the volume that the detector samples.

Measured cases
--------------

These runs used float32 host inputs, host detector output, and a single
37-degree polarization on an RTX 8000. Peak VRAM is the sampled increase
over the idle GPU, in MiB. Single-energy cases used 285 eV. The 20-energy
scans covered 280 to 289.5 eV in 0.5 eV steps.

.. list-table::
   :header-rows: 1

   * - Shape and voxel size
     - Materials
     - Energies
     - Dense peak
     - Compact peak
   * - 32 × 256 × 256 at 2.5 nm
     - 6 anisotropic
     - 1
     - 614
     - 488
   * - 32 × 256 × 256 at 2.5 nm
     - 8 anisotropic
     - 1
     - 694
     - 580
   * - 32 × 256 × 256 at 2.5 nm
     - 2 anisotropic
     - 20
     - 454
     - 304
   * - 32 × 256 × 256 at 2.5 nm
     - 4 anisotropic
     - 20
     - 534
     - 398
   * - 32 × 256 × 256 at 2.5 nm
     - 6 anisotropic
     - 20
     - 614
     - 494
   * - 32 × 256 × 256 at 2.5 nm
     - 8 anisotropic
     - 20
     - 694
     - 588
   * - 1 × 1024 × 1024 at 5 nm
     - 1 anisotropic
     - 1
     - 362
     - 324
   * - 1 × 1024 × 1024 at 5 nm
     - 8 anisotropic
     - 1
     - 502
     - 856
   * - 1 × 1024 × 1024 at 5 nm
     - 2 anisotropic, 6 isotropic
     - 1
     - 406
     - 472
   * - 1 × 1024 × 1024 at 5 nm
     - 8 isotropic
     - 1
     - 374
     - 344

The one-energy volume sampled 5.7% of its Fourier grid. The 20-energy
volume sampled 6.7%, and the single plane sampled 99.8%. With eight
anisotropic materials, the 20-energy volume's retained basis occupied
61 MiB. The single plane's basis occupied 455 MiB with eight anisotropic
materials and 168 MiB with two anisotropic and six isotropic materials.

For a thin model with broad detector coverage close to the device's memory
limit, measure both paths with the actual shape, material fields, and energy
list. Pass these options to select the dense tensor path when constructing
the ``Morphology``.

.. code-block:: python

   backend_options = {
       "execution_path": "tensor_coeff",
       "energy_basis_mode": None,
   }

Virtual z padding for films
---------------------------

Set ``virtual_z_pad_to`` to the total axial transform depth when constructing
a CuPy ``Morphology``. For example, the options below center a film in a
1024-plane vacuum box. Use the film's original material arrays and voxel pitch.

.. code-block:: python

   backend_options = {
       "execution_path": "tensor_coeff",
       "virtual_z_pad_to": 1024,
       "virtual_z_method": "residues",
   }

The calculation transforms the occupied planes and evaluates the axial bins
used by the detector. The default evaluator, ``"sum"``, supports any target
depth at least as large as the occupied depth. The ``"residues"`` evaluator
requires an integer ratio between the two depths. Setting the target equal to
the occupied depth runs the native transform.

For an extended depth, use the full-depth float32 compact tensor path. Detector
and reduced results report ``occupied_z``, ``transform_z``, ``z_offset``, and
the evaluator in ``virtual_z_method``. For an extended film, ``WindowingType=0``
is the default and leaves all axes unweighted. ``WindowingType=1`` applies
Hann weights in XY; every occupied Z plane has unit axial weight. Native
transforms, including an identity pad, retain the ordinary XYZ Hann behavior
for ``WindowingType=1``.

Set ``virtual_pad_material`` to a material ID registered in the morphology to
fill the added planes with that material. For example, add
``"virtual_pad_material": 3`` to the options above when material 3 is water.
The exterior uses the selected material's isotropic optical response at
each energy (``S=0``). All fractions, order and orientations inside the
supplied crop retain their input values. The option defaults to vacuum
when omitted or set to ``None``.

For a material-filled extension the result also records ``pad_material_id``,
``pad_fill_order``, and the occupied and transform shapes and offsets in ZYX
order. These attributes describe the finite box used for the transform.

Virtual XYZ padding
-------------------

Set ``virtual_3d_pad_to`` to an integer for a cubic logical grid or to
``(Mz, My, Mx)`` for an exact rectangular grid. The transform centers the
supplied crop in that box and returns an ``(My, Mx)`` detector for each energy.
For example, ``virtual_3d_pad_to=(256, 1024, 1024)`` preserves the voxel pitch
and detector sampling of a physically occupied box of that shape.

Use either ``virtual_3d_pad_to`` or ``virtual_z_pad_to``. XYZ padding uses
batched XY transforms of occupied planes and evaluates only the axial Fourier
samples used by the detector. The ``virtual_z_method`` selector applies to
Z-only padding. Both modes support ``virtual_pad_material`` and the same
lateral-only window at the logical coordinates when ``WindowingType=1``.

For example, use material 2 as the exterior around a cropped morphology and
return the native intensity and anisotropy on the padded detector's q grid.

.. code-block:: python

   morph = Morphology(
       len(materials),
       materials=materials,
       PhysSize=voxel_pitch_nm,
       backend="cupy-rsoxs",
       backend_options={
           "execution_path": "tensor_coeff",
           "virtual_3d_pad_to": (256, 1024, 1024),
           "virtual_pad_material": 2,
           "result_layout": "i_a",
       },
   )
   result = morph.run()

If material 2 has Euler orientation fields, the supplied crop uses those
fields and the exterior uses material 2 at ``S=0``. Omit
``virtual_pad_material`` to simulate a vacuum exterior. Set
``result_layout="detector"`` for the complete 1024 × 1024 detector at each
energy, or ``result_layout="i_para_i_perp"`` for the native sector intensities.
