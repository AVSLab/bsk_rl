import re
import warnings
from copy import deepcopy
from typing import Any, Callable


def valid_func_name(name: str) -> str:
    """Converts a string into a valid function name.

    Args:
        name: desired function name

    Returns:
        sanitized function name
    """
    # Remove all characters except for letters, digits, and underscores
    name = re.sub(r"\W+", "_", name)
    # If the name starts with a digit, add an underscore to the beginning
    if name[0].isdigit():
        name = "_" + name
    return name


def safe_dict_merge(updates: dict, base: dict) -> dict:
    """Merges a dict with another dict, warning for conflicts

    Args:
        updates: dictionary to be added to base
        base: base dictionary to be modified

    Returns:
        dict: updated base
    """
    # Updates base with a copy of elements in updates
    for k, v in updates.items():
        if k in base and base[k] != v:
            warnings.warn(f"Conflicting values for {k}: overwriting {base[k]} with {v}")
    base.update(deepcopy(updates))
    return base


def default_args(**defaults) -> Callable:
    """Decorator to enumerate default arguments of certain functions so they can be collected"""

    def inner_dec(func) -> Callable:
        def inner(*args, **kwargs) -> Callable:
            return func(*args, **kwargs)

        inner.defaults = dict(**defaults)
        return inner

    return inner_dec


def collect_default_args(object: object) -> dict[str, Any]:
    """Collect all function @default_args in an object

    Args:
        object: object with @default_args decorated functions

    Returns:
        dict: dict of keyword-value pairs of default arguments
    """
    defaults = {}
    for name in dir(object):
        if (
            callable(getattr(object, name))
            and not name.startswith("__")
            and hasattr(getattr(object, name), "defaults")
        ):
            safe_dict_merge(getattr(object, name).defaults, defaults)
    return defaults


def aliveness_checker(func: Callable[..., bool]) -> Callable[..., bool]:
    """Decorator to evaluate func -> bool when checking for satellite aliveness"""

    def inner(*args, **kwargs) -> bool:
        self = args[0]
        alive = func(*args, **kwargs)
        if not alive:
            self.satellite.info.append(
                (self.simulator.sim_time, f"failed {func.__name__} check")
            )
            print(
                f"Satellite with id {self.satellite.id} failed {func.__name__} check!"
            )
        return alive

    inner.is_aliveness_checker = True
    return inner


def check_aliveness_checkers(model: Any) -> bool:
    """Evaluate all functions with @aliveness_checker in a model

    Args:
        model (Any): Model to search for checkers in

    Returns:
        bool: Model aliveness status
    """
    is_alive = True
    for name in dir(model):
        if (
            callable(getattr(model, name))
            and not name.startswith("__")
            and hasattr(getattr(model, name), "is_aliveness_checker")
        ):
            is_alive = is_alive and getattr(model, name)()
    return is_alive
