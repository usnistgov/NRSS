``cupy-rsoxs`` Reduced-Result q Grids
=====================================

The ``cupy-rsoxs`` backend can reduce simulated detector panels on the GPU
before results are streamed to host memory or retained on device. Reduced
layouts are selected through ``Morphology(..., backend_options=...)``:

.. code-block:: python

   morph = Morphology(
       ...,
       backend="cupy-rsoxs",
       backend_options={
           "result_layout": "integrated",
           "result_q_grid_mode": "union",
       },
   )
   result = morph.run(return_xarray=True)

The supported reduced layouts are ``"integrated"``, ``"i_only"``,
``"i_para_i_perp"``, and ``"i_a"``. Raw ``"detector"`` output has ``qx`` and
``qy`` coordinates rather than a shared radial coordinate, so it rejects
``result_q_grid_mode="union"``.

Grid modes
----------

For multi-energy 3D results, detector-corrected ``|q|`` varies with energy.
``result_q_grid_mode`` determines how those per-energy radial coordinates are
placed on one shared xarray ``q`` dimension:

``"intersection"``
   The default. The shared q range is restricted to the overlap supported by
   every energy. This preserves the maintained ``NRSSIntegrator`` behavior and
   existing parity workflows.

``"union"``
   The shared q range extends from the smallest finite per-energy bound to the
   largest. Values outside an individual energy's physical q support are
   represented by ``NaN``. This permits a high-energy frame, such as Cu K-alpha
   at 8047.8 eV, to retain q values that are unavailable to soft-X-ray frames
   in the same result stack.

For a single energy, there is no shared-grid interpolation. In 2D
reciprocal-plane mode, the radial coordinate is energy-independent ``q_perp``.
Consequently, union and intersection differ only for multi-energy 3D reduced
results.

Coordinates and metadata
------------------------

The returned xarray object uses ``q`` as the uniform shared coordinate. For a
multi-energy 3D reduction, ``q_abs`` stores the exact detector-corrected
per-energy radial axes before common-grid interpolation. Union output also
contains:

- ``result_q_grid_mode = "union"``
- ``radial_coordinate_mode = "shared_q_grid_union_interpolated"``
- a ``q_axis_note`` describing the union and NaN support mask

The data's finite mask is the authoritative indication that an energy supports
a shared q sample:

.. code-block:: python

   supported = result.notnull()
   hard_energy = result.sel(energy=8047.8)

For ``Dataset`` layouts, apply ``notnull()`` to the desired data variable.

q resolution
------------

The native Cartesian detector-axis increment is

.. math::

   \Delta q_{x,y} = \frac{2\pi}{N\,\mathrm{PhysSize}}
                  = \frac{2\pi}{L_{\mathrm{sim}}}.

The polar remesh has a fixed radial bin count determined by the detector's
pixel-space center-to-corner radius. Both common-grid modes retain that bin
count and construct a uniform shared coordinate over their selected q range:

.. math::

   \Delta q_{\mathrm{common}} =
   \frac{q_{\max}-q_{\min}}{N_q-1}.

Union and intersection therefore generally do not have the same q increment.
When a soft-X-ray energy supports only a small fraction of the numerical
detector range, intersection mode can oversample the available native radial
information because it places the full polar bin count inside the shorter
overlap. Union mode spans the wider range with the same number of bins and can
have a larger common-grid increment. This changes interpolation density, not
the underlying simulated detector resolution.

Compatibility and validation
----------------------------

``"intersection"`` remains the default, so existing workflows and
``NRSSIntegrator`` parity are unchanged unless union mode is requested
explicitly. NRSS validates the option during ``Morphology`` construction and
raises ``BackendOptionError`` for unknown grid modes or for union mode combined
with ``result_layout="detector"``.
