"""TABX adapters for MAICC.

The package intentionally has no import-time dependency on h5py, faiss, or
the TABX simulator. Command line entrypoints import those optional
dependencies only when the corresponding feature is used.
"""

from .manifest import TabxManifest, TabxTask

__all__ = ["TabxManifest", "TabxTask"]
