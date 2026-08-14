"""Research Focus I architecture-comparison experiment package.

The package intentionally lives beside the historical Polaris scripts so the
study remains a research artifact rather than a public BSK-RL API.
"""

from .config import EnvironmentConfig, StudyConfig, load_study_config

__all__ = ["EnvironmentConfig", "StudyConfig", "load_study_config"]
