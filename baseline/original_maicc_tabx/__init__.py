"""Original MAICC execution with a TABX-only data adapter.

The package deliberately does not replace the MAICC model, loss, retrieval
formula, Sacred entry point, or sequential training loop.
"""

from .manifest import BaselineManifest, BaselineManifestError

__all__ = ["BaselineManifest", "BaselineManifestError"]
