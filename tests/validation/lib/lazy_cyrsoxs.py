from __future__ import annotations

import importlib


class _LazyCyRSoXS:
    def __init__(self):
        self._module = None

    def _load(self):
        if self._module is not None:
            return self._module

        errors = []
        for name in ("CyRSoXS", "cyrsoxs"):
            try:
                self._module = importlib.import_module(name)
                return self._module
            except Exception as exc:  # pragma: no cover - exercised only when unavailable
                errors.append(f"{name}: {exc.__class__.__name__}({exc})")

        raise RuntimeError(
            "CyRSoXS bindings are unavailable. "
            f"Import attempts failed: {'; '.join(errors)}"
        )

    def __getattr__(self, name):
        return getattr(self._load(), name)


cy = _LazyCyRSoXS()
