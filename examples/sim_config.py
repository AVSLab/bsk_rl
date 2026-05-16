# sim_config.py
from dataclasses import dataclass
from typing import Optional

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
    variable_duration_imaging: bool = True   # True: early-stop on success/window close
    min_pointing_hold_s: float = 10.0        # [s] hold-gate requirement
    hold_mode: str = "cumulative"            # {"cumulative", "continuous"}
    require_illumination_during_hold: bool = False  # AMOS: verify quality on downlink
    hold_illumination_threshold: Optional[float] = None

    reimage_cooldown_orbits: float = 2.0     # cooldown before target is eligible again
    verify_image_quality_on_downlink: bool = True
    hide_pending_targets: bool = True
    image_quality_threshold: Optional[float] = None
    variable_duration_downlink: bool = True   # True: early-stop when storage is empty
    downlink_empty_threshold_bits: float = 1.0

    priority_mode: str = "uniform"           # {"uniform","gaussian","constant"}
    priority_sum: Optional[float] = 100.0    # total target points if rescaling enabled
    rescale_priorities_to_sum: bool = True
    priority_constant: float = 1.0
    priority_uniform_low: float = 0.0
    priority_uniform_high: Optional[float] = None
    priority_gaussian_mean: Optional[float] = None
    priority_gaussian_std: Optional[float] = None
    priority_min: float = 0.0
    priority_max: Optional[float] = None

    dynamic_priority_event_enabled: bool = False
    dynamic_priority_event_time_sec: Optional[float] = None
    dynamic_priority_event_fraction: float = 0.5
    hio_count: int = 5
    hio_priority: float = 5.0
    shio_count: int = 3
    shio_priority: float = 10.0
    dynamic_priority_event_seed: Optional[int] = None

    extra_time_factor: float = 1.5           # multiplier for total_time
    obs_v: float = 2.0                       # default obs version; overwritten for known policies
    just_imaging: bool = False               # allow huge storage/battery if True

    @property
    def total_time(self) -> float:
        """Episode length (what you currently call total_time)."""
        return self.extra_time_factor * self.n_targets * self.imaging_duration
