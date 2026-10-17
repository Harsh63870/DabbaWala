from __future__ import annotations


try:
    from importlib.metadata import version as _pkg_version
except ImportError:

    def _pkg_version(distribution_name: str) -> str:
        return "unknown"


# Package version
__version__ = _pkg_version("tabpfn-common-utils")
