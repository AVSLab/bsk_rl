"""Lightweight opt-in performance profiling helpers."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator


def env_flag(name: str, default: bool = False) -> bool:
    """Return an environment-backed boolean flag."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off", ""}


def profile_enabled() -> bool:
    """Return True when bsk_rl profiling is enabled."""
    return env_flag("BSK_RL_PROFILE_SIM", False)


@dataclass
class _TimingStats:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    latest_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.latest_s = elapsed_s
        self.max_s = max(self.max_s, elapsed_s)

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0


class PerformanceProfiler:
    """Collect wall-clock timings without changing default behavior."""

    def __init__(self, label: str = "bsk_rl") -> None:
        """Create a profiler for one environment instance."""
        self.label = label
        self.enabled = profile_enabled()
        self.reset_episode()

    def reset_episode(self) -> None:
        """Clear timing state for a new episode."""
        self.timings: dict[str, _TimingStats] = defaultdict(_TimingStats)
        self.step_timings: dict[str, float] = {}
        self.step_index = 0
        self.episode_start_wall_s = time.perf_counter()
        self.last_step_sim_dt_s = 0.0
        self.last_step_sim_time_s = 0.0
        self.last_step_wall_s = 0.0

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time a named section when profiling is enabled."""
        if not self.enabled:
            yield
            return

        start_s = time.perf_counter()
        try:
            yield
        finally:
            elapsed_s = time.perf_counter() - start_s
            self.record(name, elapsed_s)

    def record(self, name: str, elapsed_s: float) -> None:
        """Record a wall-clock duration."""
        if not self.enabled:
            return
        self.timings[name].add(elapsed_s)
        self.step_timings[name] = self.step_timings.get(name, 0.0) + elapsed_s

    def begin_step(self) -> None:
        """Start collecting per-step timing details."""
        if self.enabled:
            self.step_timings = {}

    def finish_step(self, sim_time_s: float, sim_dt_s: float) -> None:
        """Print a compact per-step summary."""
        if not self.enabled:
            return
        self.last_step_sim_time_s = float(sim_time_s)
        self.last_step_sim_dt_s = float(sim_dt_s)
        self.last_step_wall_s = sum(self.step_timings.values())
        top_sections = sorted(
            self.step_timings.items(), key=lambda item: item[1], reverse=True
        )[:5]
        top_text = ", ".join(f"{name}={elapsed:.3f}s" for name, elapsed in top_sections)
        print(
            "[BSK_RL_PROFILE] "
            f"{self.label} step={self.step_index} "
            f"sim_t={self.last_step_sim_time_s:.3f}s "
            f"sim_dt={self.last_step_sim_dt_s:.3f}s "
            f"wall_sum={self.last_step_wall_s:.3f}s "
            f"top=[{top_text}]",
            flush=True,
        )
        self.step_index += 1

    def metrics(self, prefix: str = "profile") -> dict[str, float]:
        """Return scalar metrics suitable for RLlib custom metrics."""
        if not self.enabled:
            return {}
        metrics: dict[str, float] = {
            f"{prefix}/episode_wall_s": time.perf_counter()
            - self.episode_start_wall_s,
            f"{prefix}/last_step_wall_s": self.last_step_wall_s,
            f"{prefix}/last_step_sim_dt_s": self.last_step_sim_dt_s,
            f"{prefix}/last_step_sim_time_s": self.last_step_sim_time_s,
        }
        for name, stats in self.timings.items():
            safe_name = name.replace(".", "/")
            metrics[f"{prefix}/{safe_name}/mean_s"] = stats.mean_s
            metrics[f"{prefix}/{safe_name}/total_s"] = stats.total_s
            metrics[f"{prefix}/{safe_name}/max_s"] = stats.max_s
            metrics[f"{prefix}/{safe_name}/count"] = float(stats.count)
        return metrics

    def print_episode_summary(self, limit: int = 10) -> None:
        """Print the largest accumulated timings for the episode."""
        if not self.enabled:
            return
        top_sections = sorted(
            self.timings.items(),
            key=lambda item: item[1].total_s,
            reverse=True,
        )[:limit]
        top_text = ", ".join(
            (
                f"{name}=total:{stats.total_s:.3f}s "
                f"mean:{stats.mean_s:.3f}s n:{stats.count}"
            )
            for name, stats in top_sections
        )
        print(f"[BSK_RL_PROFILE] {self.label} episode_summary [{top_text}]", flush=True)


def profile_section(owner, name: str):
    """Return a profiling context manager for objects with a profiler."""
    profiler = getattr(owner, "profiler", None)
    if profiler is None or not getattr(profiler, "enabled", False):
        return nullcontext()
    return profiler.section(name)
