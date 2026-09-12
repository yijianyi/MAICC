"""One checked layout for TABX normalization and observation loss."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObservationLayout:
    max_agents: int
    max_enemies: int
    n_zones: int

    @property
    def other_blocks(self):
        return self.max_agents + self.max_enemies - 1

    @property
    def zone_start(self):
        return 14 + 16 * self.other_blocks

    @property
    def obs_dim(self):
        return self.zone_start + 6 * self.n_zones

    @classmethod
    def resolve(cls, obs_dim, max_agents, n_zones, max_enemies=None):
        remainder = int(obs_dim) - 14 - 6 * int(n_zones)
        if remainder < 0 or remainder % 16:
            raise ValueError(
                f"TABX obs_dim={obs_dim} is incompatible with n_zones={n_zones}; "
                "use slot capacities, not the number of active zones"
            )
        inferred = remainder // 16 + 1 - int(max_agents)
        if inferred < 0 or (max_enemies is not None and inferred != int(max_enemies)):
            raise ValueError("TABX observation layout and unit capacities disagree")
        result = cls(int(max_agents), inferred, int(n_zones))
        if result.max_agents < 1 or result.n_zones < 0:
            raise ValueError("Invalid TABX observation capacities")
        return result

    def check_normalizer(self, path):
        if path is None:
            return
        with np.load(Path(path), allow_pickle=False) as data:
            expected = dict(obs_dim=self.obs_dim, max_agents=self.max_agents,
                            max_enemies=self.max_enemies, n_zones=self.n_zones,
                            other_unit_blocks=self.other_blocks)
            for name, value in expected.items():
                if name in data and int(data[name]) != value:
                    raise ValueError(f"Normalizer {name}={int(data[name])}, model expects {value}")

    def semantic_indices(self):
        indices = [1, 4, 11, 12]
        indices += [14 + 16 * b + i for b in range(self.other_blocks)
                    for i in (1, 4, 8, 11, 12, 13, 14)]
        indices += [self.zone_start + 6 * z + i for z in range(self.n_zones)
                    for i in (0, 5)]
        return indices
