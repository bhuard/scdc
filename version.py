"""
version.py -- Version of scdc and identification of the source code.

__version__ follows semantic versioning (MAJOR.MINOR.PATCH). Increase MINOR
when an option or an output is added, PATCH for a fix that does not change
the interface, MAJOR when results or the interface change incompatibly.

Since the version number is edited by hand, the report also gives a
fingerprint of the source files (SHA-256 of the *.py of this folder, in
alphabetical order) and, when the folder belongs to a git repository, the
commit and whether the working tree differs from it. Two runs with the same
fingerprint used exactly the same code.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys

__version__ = "1.0.0"


def source_fingerprint(folder=None):
    """SHA-256 (first 12 hex digits) of the *.py files of the package
    folder, names included, in alphabetical order."""
    folder = folder or os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".py"):
            h.update(fn.encode())
            with open(os.path.join(folder, fn), "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:12]


def git_state(folder=None):
    """(commit, dirty) of the repository holding the package, or None."""
    folder = folder or os.path.dirname(os.path.abspath(__file__))
    try:
        c = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                           cwd=folder, capture_output=True, text=True,
                           timeout=5)
        if c.returncode != 0:
            return None
        d = subprocess.run(["git", "status", "--porcelain",
                            "--untracked-files=no", "--", "."],
                           cwd=folder, capture_output=True, text=True,
                           timeout=5)
        return c.stdout.strip(), bool(d.stdout.strip())
    except Exception:
        return None


def version_info():
    """Dictionary describing the program and its environment."""
    import numpy
    import scipy
    info = dict(version=__version__, fingerprint=source_fingerprint(),
                python=platform.python_version(), numpy=numpy.__version__,
                scipy=scipy.__version__, platform=platform.platform())
    g = git_state()
    if g is not None:
        info['git_commit'], info['git_dirty'] = g
    return info


def version_string():
    i = version_info()
    s = f"scdc {i['version']} (sources {i['fingerprint']}"
    if 'git_commit' in i:
        s += f", git {i['git_commit']}" + (" + local changes"
                                           if i['git_dirty'] else "")
    return s + ")"


if __name__ == "__main__":
    print(version_string())
    print(f"Python {platform.python_version()} on {platform.platform()}")
    sys.exit(0)
