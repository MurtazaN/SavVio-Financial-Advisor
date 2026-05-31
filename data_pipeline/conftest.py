"""
Root-level conftest.py for data_pipeline (loaded by pytest before any other file).

KEY FIX: We unconditionally overwrite sys.modules['src'] if it is a namespace
package (i.e. __spec__.origin is None).  This silently gets put there by the
savviocore editable-install .pth file, which adds a directory that contains a
src/ folder without __init__.py, causing Python to register 'src' as a
namespace package before pytest even starts conftest.py.  Any subsequent
'from src.ingestion import' then fails with:
    ModuleNotFoundError: No module named 'src.ingestion'; 'src' is not a package
"""

from __future__ import annotations

import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# Absolute paths
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
DAGS_DIR = os.path.join(_HERE, "dags")
SRC_DIR  = os.path.join(DAGS_DIR, "src")
_SRC_INIT = os.path.join(SRC_DIR, "__init__.py")

# ---------------------------------------------------------------------------
# 1. Insert dags/ at front of sys.path (idempotent)
# ---------------------------------------------------------------------------
if DAGS_DIR not in sys.path:
    sys.path.insert(0, DAGS_DIR)

# ---------------------------------------------------------------------------
# 2. Detect whether 'src' in sys.modules is a PROPER regular package.
#    A regular package has __spec__.origin != None (points to __init__.py).
#    A namespace package has __spec__.origin == None  (no __init__.py found).
#
#    The savviocore editable install can place a namespace 'src' in
#    sys.modules before conftest runs.  We must OVERWRITE it.
# ---------------------------------------------------------------------------
def _src_is_bad() -> bool:
    """Return True if 'src' is absent or is a namespace package."""
    mod = sys.modules.get("src")
    if mod is None:
        return True
    spec = getattr(mod, "__spec__", None)
    if spec is None:
        return True
    # Namespace packages have origin=None
    if getattr(spec, "origin", None) is None:
        return True
    return False


if _src_is_bad() and os.path.isfile(_SRC_INIT):
    _spec = importlib.util.spec_from_file_location(
        "src",
        _SRC_INIT,
        submodule_search_locations=[SRC_DIR],
    )
    if _spec is not None and _spec.loader is not None:
        _src_mod = importlib.util.module_from_spec(_spec)
        _src_mod.__path__ = [SRC_DIR]      # type: ignore[assignment]
        _src_mod.__package__ = "src"
        sys.modules["src"] = _src_mod      # overwrite any bad entry
        _spec.loader.exec_module(_src_mod)  # type: ignore[union-attr]
