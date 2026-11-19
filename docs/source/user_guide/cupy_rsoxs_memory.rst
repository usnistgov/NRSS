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
