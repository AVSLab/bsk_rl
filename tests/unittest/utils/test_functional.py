from unittest.mock import MagicMock

import numpy as np
import pytest

from bsk_rl.utils import functional


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
    "input,outkeys,outvec",
    [
        (
            {"alpha": np.array([1]), "b": 2, "c": [3]},
            ["alpha[0]", "b", "c[0]"],
            np.array([1, 2, 3]),
        ),
        (
            {"a": {"b": 1, "charlie": 2}, "d": 3},
            ["a.b", "a.charlie", "d"],
            np.array([1, 2, 3]),
        ),
    ],
)
def test_vectorize_nested_dict(input, outkeys, outvec):
    keys, vec = functional.vectorize_nested_dict(input)
    assert np.equal(outvec, vec).all()
    assert outkeys == keys


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
        d.satellite.id = "SAT"
        d.satellite._is_alive
        assert functional.check_aliveness_checkers(d, log_failure=True) is False
        d.satellite.logger.warning.assert_called_with("failed is_living check")


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
