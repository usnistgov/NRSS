import os


# Default test execution to a single visible GPU for reproducibility and to avoid
# known multi-GPU CyRSoXS instability during energy fan-out. Respect any explicit
# user or CI pinning that is already present in the environment.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
