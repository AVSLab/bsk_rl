"""Utilities for Vizard visualization."""

import inspect
import logging
from functools import wraps

logger = logging.getLogger(__name__)

VIZARD_PATH = None
VIZINSTANCE = None


def visualize(func):
    """Decorator for functions that enable Vizard."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if VIZARD_PATH is None:
            return

        from Basilisk.simulation import vizInterface
        from Basilisk.utilities import vizSupport

        if not vizSupport.vizFound:
            logger.warning("Vizard not found, disabling visualization")
            return

        sig = inspect.signature(func)

        if "vizInstance" in sig.parameters:
            kwargs["vizInstance"] = VIZINSTANCE
        if "vizSupport" in sig.parameters:
            kwargs["vizSupport"] = vizSupport
        if "vizInterface" in sig.parameters:
            kwargs["vizInterface"] = vizInterface

        return func(*args, **kwargs)

    return wrapper


@visualize
def get_color(index):
    from matplotlib.colors import TABLEAU_COLORS

    n_colors = len(TABLEAU_COLORS)
    color = list(TABLEAU_COLORS.keys())[index % n_colors]
    return color


__doc_title__ = "Vizard"
__all__ = ["visualize", "VIZARD_PATH"]
