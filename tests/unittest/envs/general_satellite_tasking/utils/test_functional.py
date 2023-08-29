from unittest.mock import MagicMock

import numpy as np
import pytest

from bsk_rl.envs.general_satellite_tasking.utils import functional


@pytest.mark.parametrize("input", ["valid", "123", "money$", "Hello, world!"])
def test_valid_valid_func_name(input):
    assert functional.valid_func_name(input).isidentifier()


@pytest.mark.parametrize(
    "updates,base,expected,warn",
    [
        ({"a": 1, "b": 2}, {}, {"a": 1, "b": 2}, False),
        ({"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 1, "b": 2}, True),
        ({}, {"a": 1, "b": 2}, {"a": 1, "b": 2}, False),
        ({"b": 2}, {"a": 1, "b": 4}, {"a": 1, "b": 2}, True),
        ({"b": 2}, {"a": 1}, {"a": 1, "b": 2}, False),
        ({}, {}, {}, False),
    ],
)
def test_safe_dict_merge(updates, base, expected, warn):
    if warn:
        with pytest.warns():
            updated = functional.safe_dict_merge(updates, base)
    else:
        updated = functional.safe_dict_merge(updates, base)
    assert updated is base
    assert updated == expected


class TestDefaultArgs:
    class C1:
        @functional.default_args(a=1)
        def foo(self, a):
            self.a = a
            return self.a

    class C2:
        @functional.default_args(b=2)
        def bar(self, b):
            self.b = b
            return self.b

    class C3:
        @functional.default_args(a=3)
        def baz(self, a):
            self.a = a
            return self.a

    def test_default_args(self):
        c = self.C1()
        assert functional.collect_default_args(c) == {"a": 1}

    def test_call(self):
        c = self.C1()
        assert c.foo(**functional.collect_default_args(c)) == 1

    def test_default_args_combined(self):
        class C12(self.C1, self.C2):
            pass

        c = C12()
        assert functional.collect_default_args(c) == {"a": 1, "b": 2}

    def test_default_args_overwrite(self):
        class C13(self.C1, self.C3):
            pass

        c = C13()
        with pytest.warns():
            assert functional.collect_default_args(c) == {"a": 1}


@pytest.mark.parametrize(
    "input,output",
    [
        ({"a": np.array([1]), "b": 2, "c": [3]}, np.array([1, 2, 3])),
        ({"a": {"b": 1, "c": 2}, "d": 3}, np.array([1, 2, 3])),
    ],
)
def test_vectorize_nested_dict(input, output):
    assert np.equal(output, functional.vectorize_nested_dict(input)).all()


class TestAlivenessChecker:
    class Alive(MagicMock):
        @functional.aliveness_checker
        def is_alive(self):
            return True

    class Dead(MagicMock):
        @functional.aliveness_checker
        def is_living(self):
            return False

    class Schrodinger(Alive, Dead):
        pass

    def test_alive(self):
        a = self.Alive()
        assert functional.check_aliveness_checkers(a)

    @pytest.mark.parametrize(
        "type",
        [Dead, Schrodinger],
    )
    def test_dead(self, type):
        d = type()
        d.simulator.sim_time = 0
        d.satellite.info = []
        d.satellite.id = "SAT"
        assert functional.check_aliveness_checkers(d) is False
        assert d.satellite.info[0] == (0, "failed is_living check")


@pytest.mark.parametrize("prop_name,expected", [("prop", True), ("not_a_prop", False)])
def test_is_property(prop_name, expected):
    class CalledException(BaseException):
        pass

    class Class:
        @property
        def prop(self):
            # should not be called when checked
            raise CalledException

    c = Class()
    assert functional.is_property(c, prop_name) == expected


class TestConfigurable:
    @functional.configurable
    class Class:
        def __init__(self, *, a=1, b=2):
            self.a = a
            self.b = b

    ClassConfigured = Class.configure(b=4)

    def test_default(self):
        tc = self.Class()
        assert tc.a == 1
        assert tc.b == 2

    def test_configured(self):
        tc = self.ClassConfigured()
        assert tc.a == 1
        assert tc.b == 4

    def test_configured_overwrite(self):
        tc = self.ClassConfigured(a=3)
        assert tc.a == 3
        assert tc.b == 4
        tc = self.ClassConfigured(b=2)
        assert tc.a == 1
        assert tc.b == 2

    ClassNotConfigured = Class.configure(c=0)

    def test_not_configurable(self):
        with pytest.raises(KeyError):
            self.ClassNotConfigured()


def test_bind():
    class Thing:
        def __init__(self, val):
            self.val = val

    something = Thing(21)

    def double(self):
        return 2 * self.val

    functional.bind(something, double)
    assert something.double() == 42
