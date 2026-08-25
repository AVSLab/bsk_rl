from types import SimpleNamespace

import numpy as np
import pytest

from bsk_rl.act.discrete_actions import _select_highest_priority_candidate


def target(target_id, priority):
    return SimpleNamespace(id=target_id, priority=priority)


def test_candidate_priority_selects_highest_priority_from_supplied_set():
    candidates = [target(4, 2.0), target(9, 7.0), target(1, 3.0)]

    assert _select_highest_priority_candidate(candidates).id == 9


def test_candidate_priority_uses_target_id_as_deterministic_tie_breaker():
    candidates = [target(8, 5.0), target(3, 5.0)]

    assert _select_highest_priority_candidate(candidates).id == 3


def test_candidate_priority_ignores_duplicate_padding_and_nonfinite_priority():
    chosen = target(2, 4.0)
    candidates = [target(1, np.nan), chosen, chosen]

    assert _select_highest_priority_candidate(candidates) is chosen


def test_candidate_priority_rejects_an_empty_candidate_set():
    with pytest.raises(RuntimeError, match="No candidate targets"):
        _select_highest_priority_candidate([])
