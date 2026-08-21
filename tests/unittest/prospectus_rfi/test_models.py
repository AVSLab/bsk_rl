import numpy as np
import pytest
import torch

from examples.prospectus_rfi.config import ArchitectureConfig
from examples.prospectus_rfi.environment import zero_padded_target_rows
from examples.prospectus_rfi.models import (
    INVALID_ACTION_LOGIT,
    FixedInputMonolithicActorCritic,
    ObservationLayout,
    TargetSetAttentionActorCritic,
)


@pytest.fixture
def layout():
    return ObservationLayout(
        global_features=3,
        target_features=3,
        target_capacity=4,
        target_mask_index=2,
        non_target_actions=3,
    )


def make_observation(layout, valid_count=2):
    global_values = torch.tensor([[0.2, -0.3, 0.8]])
    targets = torch.tensor(
        [[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    targets[:, valid_count:, :] = 0.0
    return torch.cat([global_values, targets.flatten(start_dim=1)], dim=-1)


def small_architecture(name):
    return ArchitectureConfig(
        name=name,
        hidden_widths=(16,),
        embedding_dim=8,
        attention_heads=2,
        attention_blocks=1,
        feed_forward_width=16,
        dropout=0.0,
    )


def test_padding_helper_zeroes_features_and_writes_mask():
    rows = np.arange(20, dtype=float).reshape(4, 5)
    masked = zero_padded_target_rows(rows, valid_count=2, mask_index=4)

    np.testing.assert_array_equal(masked[2:], np.zeros((2, 5)))
    np.testing.assert_array_equal(masked[:2, 4], np.ones(2))
    np.testing.assert_array_equal(rows[:, 4], np.array([4.0, 9.0, 14.0, 19.0]))


@pytest.mark.parametrize(
    "model_type,name",
    [
        (FixedInputMonolithicActorCritic, "fixed_input_monolithic_mlp"),
        (TargetSetAttentionActorCritic, "target_set_attention"),
    ],
)
def test_invalid_target_logits_are_masked(model_type, name, layout):
    model = model_type(layout, small_architecture(name)).eval()
    logits, value = model(make_observation(layout))

    assert logits.shape == (1, 7)
    assert value.shape == (1,)
    assert torch.all(logits[0, 2:4] <= INVALID_ACTION_LOGIT)
    assert torch.all(torch.isfinite(logits[0, [0, 1, 4, 5, 6]]))


@pytest.mark.parametrize(
    "model_type,name",
    [
        (FixedInputMonolithicActorCritic, "fixed_input_monolithic_mlp"),
        (TargetSetAttentionActorCritic, "target_set_attention"),
    ],
)
def test_padded_values_cannot_change_outputs(model_type, name, layout):
    torch.manual_seed(7)
    model = model_type(layout, small_architecture(name)).eval()
    clean = make_observation(layout)
    contaminated = clean.clone()
    rows = contaminated[:, layout.global_features :].reshape(1, 4, 3)
    rows[:, 2:, :2] = torch.tensor([[[9999.0, -9999.0], [-55.0, 88.0]]])

    with torch.inference_mode():
        clean_logits, clean_value = model(clean)
        dirty_logits, dirty_value = model(contaminated)

    torch.testing.assert_close(clean_logits, dirty_logits)
    torch.testing.assert_close(clean_value, dirty_value)


def test_attention_policy_is_permutation_equivariant(layout):
    torch.manual_seed(11)
    model = TargetSetAttentionActorCritic(
        layout, small_architecture("target_set_attention")
    ).eval()
    observation = make_observation(layout)
    permuted = observation.clone()
    target_rows = permuted[:, layout.global_features :].reshape(1, 4, 3)
    target_rows[:, [0, 1]] = target_rows[:, [1, 0]]

    with torch.inference_mode():
        logits, value = model(observation)
        permuted_logits, permuted_value = model(permuted)

    torch.testing.assert_close(logits[:, [0, 1]], permuted_logits[:, [1, 0]])
    torch.testing.assert_close(logits[:, 4:], permuted_logits[:, 4:])
    torch.testing.assert_close(value, permuted_value)


def test_attention_handles_no_valid_target_without_nan(layout):
    model = TargetSetAttentionActorCritic(
        layout, small_architecture("target_set_attention")
    ).eval()
    observation = make_observation(layout, valid_count=0)
    logits, value = model(observation)

    assert torch.all(logits[:, :4] <= INVALID_ACTION_LOGIT)
    assert torch.all(torch.isfinite(logits[:, 4:]))
    assert torch.all(torch.isfinite(value))


def test_checkpoint_control_layout_extracts_middle_target_rows():
    layout = ObservationLayout(
        global_features=4,
        target_start=2,
        target_features=3,
        target_capacity=2,
        target_mask_index=2,
        non_target_actions=3,
    )
    # Two global prefix values, two target rows, then two global suffix values.
    observation = torch.tensor([[10.0, 11.0, 1.0, 2.0, 1.0, 3.0, 4.0, 0.0, 12.0, 13.0]])

    global_values, physical, valid = layout.split(observation)

    torch.testing.assert_close(global_values, torch.tensor([[10.0, 11.0, 12.0, 13.0]]))
    torch.testing.assert_close(physical, torch.tensor([[[1.0, 2.0], [0.0, 0.0]]]))
    assert valid.tolist() == [[True, False]]
