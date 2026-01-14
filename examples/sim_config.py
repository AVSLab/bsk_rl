# sim_config.py
from dataclasses import dataclass

@dataclass
class SimConfig:
    """
    Shared simulation configuration for both training and evaluation.

    You can change the defaults here, or construct SimConfig(...) explicitly
    in each script if you want different settings.
    """
    n_targets: int = 100
    n_targets_ahead: int = 10
    imaging_duration: float = 300.0          # [s]
    extra_time_factor: float = 1.5           # multiplier for total_time
    obs_v: float = 2.0                       # default obs version; overwritten for known policies
    just_imaging: bool = False               # allow huge storage/battery if True

    @property
    def total_time(self) -> float:
        """Episode length (what you currently call total_time)."""
        return self.extra_time_factor * self.n_targets * self.imaging_duration
