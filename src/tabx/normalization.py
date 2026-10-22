from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


class NormalizationError(RuntimeError):
    pass


@dataclass
class ObservationNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    clip: float = 10.0
    preserve_zeros: bool = True

    @classmethod
    def identity(cls, obs_dim: int) -> "ObservationNormalizer":
        return cls(
            mean=np.zeros(obs_dim, dtype=np.float32),
            scale=np.ones(obs_dim, dtype=np.float32),
            clip=0.0,
            preserve_zeros=True,
        )

    @classmethod
    def load(
        cls,
        path: Optional[Path],
        obs_dim: int,
        clip: float = 10.0,
        preserve_zeros: bool = True,
        manifest_path: Optional[Path] = None,
    ) -> "ObservationNormalizer":
        if path is None:
            return cls.identity(obs_dim)
        try:
            payload = np.load(path, allow_pickle=False)
            mean = np.asarray(payload["mean"], dtype=np.float32)
            scale = np.asarray(payload["scale"], dtype=np.float32)
            fit_split = str(payload["fit_split"].item()) if "fit_split" in payload else None
            stored_manifest_hash = (
                str(payload["manifest_sha256"].item())
                if "manifest_sha256" in payload
                else None
            )
        except (OSError, KeyError, ValueError) as exc:
            raise NormalizationError(
                f"Cannot load observation normalization from {path}: {exc}"
            ) from exc
        if mean.shape != (obs_dim,) or scale.shape != (obs_dim,):
            raise NormalizationError(
                f"Normalization shape mismatch: mean={mean.shape}, "
                f"scale={scale.shape}, expected {(obs_dim,)}"
            )
        if fit_split not in (None, "train"):
            raise NormalizationError("Baseline normalization must be fitted on split=train")
        if fit_split is None and manifest_path is not None:
            import json
            tasks = json.loads(Path(manifest_path).read_text())["tasks"]
            if any(task.get("split", "train") != "train" for task in tasks):
                raise NormalizationError("Recompute legacy normalization with --split train for a held-out-task manifest")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise NormalizationError("Normalization contains NaN or Inf")
        if np.any(np.abs(mean) > 1e-8):
            raise NormalizationError(
                "TABX normalization must be scale-only; non-zero mean found"
            )
        if np.any(scale <= 0):
            raise NormalizationError("Normalization scale must be positive")
        if manifest_path is not None and stored_manifest_hash is not None:
            actual_hash = hashlib.sha256(
                Path(manifest_path).read_bytes()
            ).hexdigest()
            if actual_hash != stored_manifest_hash:
                raise NormalizationError(
                    "Normalizer was computed from a different TABX manifest"
                )
        return cls(
            mean=mean,
            scale=scale,
            clip=float(clip),
            preserve_zeros=bool(preserve_zeros),
        )

    def apply(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=np.float32)
        zero_mask = source == 0 if self.preserve_zeros else None
        result = (source - self.mean) / self.scale
        if self.clip > 0:
            np.clip(result, -self.clip, self.clip, out=result)
        if zero_mask is not None:
            result[zero_mask] = 0
        return result
