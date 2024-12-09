# """
# NRSS

# The NIST RSoXS Simulation Suite: A collection of Python and C++/CUDA tools for simulating and analyzing Resonant Soft X-ray Scattering (RSoXS).
# """

# from NRSS import writer
# from NRSS import checkH5
# from NRSS import morphology
# from NRSS import reader

# __version__ = "0.1.0"
# __author__ = "Peter Dudenas"
# __credits__ = "National Institute of Standards and Technology"

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"  # Fallback version if the file is missing
