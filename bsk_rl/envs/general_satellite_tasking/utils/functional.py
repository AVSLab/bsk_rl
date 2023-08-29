import inspect
import re
import warnings
from copy import deepcopy
from typing import Any, Callable

import numpy as np


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
    """Decorator to enumerate default arguments of certain functions so they can be
    collected"""

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


def vectorize_nested_dict(dictionary: dict) -> np.ndarray:
    """Flattens a dictionary of dictionaries, arrays, and scalars into a single
    vector."""
    values = list(dictionary.values())
    for i, value in enumerate(values):
        if isinstance(value, np.ndarray):
            values[i] = value.flatten()
        elif isinstance(value, (float, int)):
            values[i] = [value]
        elif isinstance(value, dict):
            values[i] = vectorize_nested_dict(value)

    return np.concatenate(values)


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
            not name.startswith("__")
            and not is_property(model, name)
            and callable(getattr(model, name))
            and hasattr(getattr(model, name), "is_aliveness_checker")
        ):
            is_alive = is_alive and getattr(model, name)()
    return is_alive


def is_property(obj: Any, attr_name: str) -> bool:
    """Check if obj has an @property attr_name without calling it"""
    cls = type(obj)
    attribute = getattr(cls, attr_name, None)
    return attribute is not None and isinstance(attribute, property)


def configurable(cls):
    """Class decorator to create new instance of a class with different defaults to
    __init__"""

    @classmethod
    def configure(cls, **config_kwargs):
        class Configurable(cls):
            def __init__(self, *args, **kwargs):
                init_kwargs = deepcopy(config_kwargs)
                for key in init_kwargs.keys():
                    if not (
                        key in inspect.getfullargspec(super().__init__).args
                        or key in inspect.getfullargspec(super().__init__).kwonlyargs
                    ):
                        raise KeyError(
                            f"{key} not a keyword argument for {cls.__name__}"
                        )
                for k, v in kwargs.items():
                    init_kwargs[k] = v
                super().__init__(*args, **init_kwargs)

        configcls = Configurable
        return configcls

    cls.configure = configure
    return cls


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method
