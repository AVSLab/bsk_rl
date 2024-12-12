"""``bsk_rl.utils.functional``: General utility functions."""

import re
import warnings
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

import numpy as np


def valid_func_name(name: str) -> str:
    """Convert a string into a valid function name.

    Args:
        name: Desired function name.

    Returns:
        Sanitized function name.
    """
    # Remove all characters except for letters, digits, and underscores
    name = re.sub(r"\W+", "_", name)
    # If the name starts with a digit, add an underscore to the beginning
    if name[0].isdigit():
        name = "_" + name
    return name


def safe_dict_merge(updates: dict, base: dict) -> dict:
    """Merge a dict with another dict, warning for conflicts.

    .. code-block:: python

        >>> safe_dict_merge(dict(a=1, b=2), dict(b=2, c=3))
        {'a': 1, 'b': 2, 'c': 3}

        >>> safe_dict_merge(dict(a=1, b=4), dict(b=2, c=3))
        Warning: Conflicting values for b: overwriting 2 with 4
        {'a': 1, 'b': 4, 'c': 3}

    Args:
        updates: Dictionary to be added to base.
        base: Base dictionary to be modified.

    Returns:
        dict: Updated copy of base.
    """
    # Updates base with a copy of elements in updates
    for k, v in updates.items():
        if k in base and base[k] != v:
            warnings.warn(f"Conflicting values for {k}: overwriting {base[k]} with {v}")
    base.update(deepcopy(updates))
    return base


P = ParamSpec("P")
T = TypeVar("T")


def default_args(**defaults) -> Callable:
    """Decorate function to enumerate default arguments for collection."""

    def inner_dec(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        inner.defaults = dict(**defaults)

        return inner

    return inner_dec


def collect_default_args(object: object) -> dict[str, Any]:
    """Collect all function :class:`default_args` in an object.

    Args:
        object: Object with :class:`default_args`-decorated functions.

    Returns:
        dict: Dict of keyword-value pairs of default arguments.
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


def vectorize_nested_dict(dictionary: dict) -> tuple[list[str], np.ndarray]:
    """Flattens a dictionary of dictionaries, arrays, and scalars into a vector."""
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    for i, value in enumerate(values):
        if isinstance(value, np.ndarray):
            values[i] = value.flatten()
            keys[i] = [keys[i] + f"[{j}]" for j in range(len(value.flatten()))]
        elif isinstance(value, list):
            keys[i] = [keys[i] + f"[{j}]" for j in range(len(value))]
        elif isinstance(value, (float, int)):
            values[i] = [value]
            keys[i] = [keys[i]]
        elif isinstance(value, dict):
            prepend = keys[i]
            keys[i], values[i] = vectorize_nested_dict(value)
            keys[i] = [prepend + "." + key for key in keys[i]]

    return list(np.concatenate(keys)), np.concatenate(values)


def aliveness_checker(func: Callable[..., bool]) -> Callable[..., bool]:
    """Decorate function to evaluate when checking for satellite aliveness."""

    @wraps(func)
    def inner(*args, log_failure=False, **kwargs) -> bool:
        self = args[0]
        alive = func(*args, **kwargs)
        if not alive and log_failure:
            self.satellite.logger.warning(f"failed {func.__name__} check")
        return alive

    inner.__doc__ = (
        "*Decorated with* :class:`~bsk_rl.utils.functional.aliveness_checker`\n\n"
        + str(func.__doc__)
    )
    inner.is_aliveness_checker = True
    return inner


def check_aliveness_checkers(model: Any, log_failure=False) -> bool:
    """Evaluate all functions with @aliveness_checker in a model.

    Args:
        model: Model to search for checkers in.
        log_failure: Whether to log to the logger on checker failure.

    Returns:
        bool: Model aliveness status.
    """
    is_alive = True
    for name in dir(model):
        if (
            not name.startswith("__")
            and not is_property(model, name)
            and callable(getattr(model, name))
            and hasattr(getattr(model, name), "is_aliveness_checker")
        ):
            is_alive = is_alive and getattr(model, name)(log_failure=log_failure)
    return is_alive


def is_property(obj: Any, attr_name: str) -> bool:
    """Check if obj has an ``@property`` called ``attr_name`` without calling it."""
    cls = type(obj)
    attribute = getattr(cls, attr_name, None)
    return attribute is not None and isinstance(attribute, property)


class AbstractClassProperty:
    def __init__(self):
        """Assign a class property to act like an abstract field."""
        self.__isabstractmethod__ = True

    def __set_name__(self, owner, name):  # noqa
        self.name = name

    def __get__(self, instance, owner):  # noqa
        if instance is None:
            return self
        raise NotImplementedError(
            f"AbstractClassProperty '{self.name}' must be set in subclass"
        )


class Resetable:
    """Base class for resetable objects.

    This class is used to define objects that are reset at the beginning of each episode.
    These include :class:`~bsk_rl.scene.Scenario`, :class:`~bsk_rl.data.GlobalReward`,
    :class:`~bsk_rl.comm.CommunicationMethod`, and :class:`~bsk_rl.sats.Satellite`.

    The :class:`~bsk_rl.GeneralSatelliteTasking.reset` method takes the following steps:

    #. ``reset_overwrite_previous`` - Overwrite attributes from previous episode.
    #. ``reset_pre_sim_init`` - Reset before simulator initialization.
    #. Initialize a new Basilisk simulator.
    #. ``reset_post_sim_init`` - Reset after simulator initialization.
    """

    def reset_overwrite_previous(self) -> None:
        """Overwrite attributes from previous episode."""
        pass

    def reset_pre_sim_init(self) -> None:
        """Reset before simulator initialization."""
        pass

    def reset_during_sim_init(self) -> None:
        """Reset after simulator models have been created but before the simulator is initialized."""
        pass

    def reset_post_sim_init(self) -> None:
        """Reset after simulator initialization."""
        pass


__doc_title__ = "Functional"
__all__ = [
    "valid_func_name",
    "safe_dict_merge",
    "default_args",
    "collect_default_args",
    "vectorize_nested_dict",
    "aliveness_checker",
    "check_aliveness_checkers",
    "is_property",
    "AbstractClassProperty",
    "Resetable",
]
