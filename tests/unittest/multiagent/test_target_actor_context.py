import torch

from bsk_rl.utils.rllib.target_gnn_module import GNNActor
from examples.multiagent_imaging.config import GLOBAL_FEATURES


def actor(*, conditioned):
    torch.manual_seed(4)
    return GNNActor(
        inputs=GLOBAL_FEATURES + 3 * 13,
        obs_sat=GLOBAL_FEATURES,
        n_tgts=3,
        non_imaging_actions=4,
        width_f=16,
        depth_f=2,
        tgt_encoded_dim=16,
        width_g=16,
        depth_g=2,
        attention_dim=16,
        num_heads=2,
        condition_on_spacecraft=conditioned,
    ).eval()


def test_opt_in_spacecraft_context_changes_actor_logits():
    observation = torch.zeros((1, GLOBAL_FEATURES + 3 * 13))
    changed = observation.clone()
    changed[:, :GLOBAL_FEATURES] = 1.0
    with torch.no_grad():
        conditioned = actor(conditioned=True)
        assert not torch.allclose(conditioned(observation), conditioned(changed))
        legacy = actor(conditioned=False)
        assert torch.allclose(legacy(observation), legacy(changed))


def test_target_candidate_permutation_is_equivariant():
    model = actor(conditioned=True)
    observation = torch.randn((1, GLOBAL_FEATURES + 3 * 13))
    global_features = observation[:, :GLOBAL_FEATURES]
    targets = observation[:, GLOBAL_FEATURES:].reshape(1, 3, 13)
    permutation = torch.tensor([2, 0, 1])
    permuted = torch.cat(
        [global_features, targets[:, permutation].reshape(1, -1)], dim=1
    )
    with torch.no_grad():
        original_logits = model(observation)
        permuted_logits = model(permuted)
    assert torch.allclose(original_logits[:, :4], permuted_logits[:, :4], atol=1e-6)
    assert torch.allclose(
        original_logits[:, 4:][:, permutation], permuted_logits[:, 4:], atol=1e-6
    )
