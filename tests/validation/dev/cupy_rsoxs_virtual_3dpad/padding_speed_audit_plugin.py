"""Opt-in pytest entry point for the existing virtual-versus-explicit audit."""
import os
from pathlib import Path


_installed = False


def install_once():
    global _installed
    if _installed:
        return
    from tests.validation.dev.cupy_rsoxs_virtual_zpad.workflow_padding_audit import install

    log_path = Path(os.environ["NRSS_PADDING_SPEED_AUDIT_LOG"]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    install(log_path)
    _installed = True


def pytest_configure(config):
    install_once()
