NRSS.fourier
============

Prepare a reusable sampler for integer Fourier modes of a real scalar field.
Both field axes and mode triples are in ZYX order. Each axis of length
``N`` accepts signed modes from ``-floor(N/2)`` through ``ceil(N/2)-1``.
The sampler keeps the order and multiplicity of the requested modes.

.. code-block:: python

    import cupy as cp
    import numpy as np
    from NRSS.fourier import prepare_sampler

    field = cp.ones((32, 40, 48), dtype=cp.float32)
    modes = np.array([(0, 0, 0), (2, -3, 5), (-2, 3, -5)])
    plan = prepare_sampler(
        shape_zyx=field.shape, logical_shape_zyx=(64, 80, 96), modes_zyx=modes,
    )
    amplitudes = cp.empty(len(modes), dtype=cp.complex64)
    plan.capture_into(field, out=amplitudes)

Omit ``logical_shape_zyx`` for a transform on the field's own grid. For
virtual padding the field starts at ``(logical_shape_zyx - shape_zyx) // 2``
on each axis, with zero exterior. The forward transform uses the negative
exponential, with unit voxel weights and no normalization. The default
``dc="raw"`` returns the sum of the field at zero mode. ``dc="nrss"``
replaces the zero mode with the incumbent axial-neighbor mean. On a native
grid this repair requires at least two samples on each lateral axis.

Preparation accepts host integer modes or a CuPy integer array on the
current device. Each capture takes a C-contiguous CuPy ``float32`` field
and a separate C-contiguous ``complex64`` output vector of length
``len(modes)``. The plan and both arrays use the device and stream that
were current at preparation. The method enqueues work, preserves the input
field, and returns ``None``. Reuse the plan with a new field and the same
output vector. If using the output from another stream, synchronize or
establish the dependency with an event. A virtual ``dc="nrss"`` request
that includes zero uses prepared workspace for the private stencil.

For complex fields, capture the real and imaginary parts separately and
combine the two results as ``F(real) + 1j * F(imag)``.

.. autofunction:: NRSS.fourier.prepare_sampler

.. autoclass:: NRSS.fourier.PreparedSampler
   :members: capture_into
