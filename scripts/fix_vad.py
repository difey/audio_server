#!/usr/bin/env python3
"""Fix webrtcvad compatibility with modern setuptools.

Run once after installing dependencies:
    uv run python scripts/fix_vad.py
"""

import shutil
import sys
from pathlib import Path


def fix_webrtcvad():
    """Patch webrtcvad.py to use importlib.metadata instead of pkg_resources."""
    venv_site = _find_site_packages()
    if not venv_site:
        print("Could not find site-packages in active venv")
        return False

    vad_path = venv_site / "webrtcvad.py"
    if not vad_path.exists():
        print(f"webrtcvad.py not found at {vad_path}")
        return False

    content = vad_path.read_text()
    if "importlib.metadata" in content:
        print("Already patched")
        return True

    # Backup
    shutil.copy2(vad_path, vad_path.with_suffix(".py.bak"))

    # Patch
    new_content = content.replace(
        "import pkg_resources",
        "from importlib.metadata import version as _version",
    ).replace(
        "pkg_resources.get_distribution('webrtcvad').version",
        "_version('webrtcvad')",
    )

    vad_path.write_text(new_content)
    print(f"Patched {vad_path}")
    return True


def _find_site_packages() -> Path | None:
    """Find the site-packages directory of the active venv."""
    import site
    from os.path import abspath

    # Try the current virtual environment
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        for p in site.getsitepackages():
            path = Path(p)
            if path.exists():
                return path

    # Fallback to searching from venv
    venv = Path(sys.prefix)
    for p in (venv / "lib").rglob("site-packages"):
        if p.is_dir():
            return p

    return None


if __name__ == "__main__":
    fix_webrtcvad()
