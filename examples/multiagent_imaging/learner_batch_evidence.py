"""Retain the first actual learner input, without changing sampled data or loss."""

import gzip
import json
from pathlib import Path

import numpy as np
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing


def numeric(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


def save_numeric_batch(directory, name, batch):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists():
        return
    arrays = {}
    for key in (
        Columns.OBS,
        Columns.ACTIONS,
        Columns.ACTION_LOGP,
        Columns.ACTION_DIST_INPUTS,
        Columns.REWARDS,
        Columns.TERMINATEDS,
        Columns.TRUNCATEDS,
        Columns.LOSS_MASK,
        Columns.VF_PREDS,
        Postprocessing.ADVANTAGES,
        Postprocessing.VALUE_TARGETS,
    ):
        if key in batch:
            arrays[key] = numeric(batch[key])
    if Columns.INFOS in batch:
        arrays["durations"] = np.array([info["d_ts"] for info in batch[Columns.INFOS]])
    assert all(
        a.dtype.kind != "O" for a in arrays.values()
    ), "Non-numeric learner evidence"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


class CaptureLearnerBatch(ConnectorV2):
    def __init__(self, directory, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory

    def get_ctor_args_and_kwargs(self):
        args, kwargs = super().get_ctor_args_and_kwargs()
        return args, {**kwargs, "directory": self.directory}

    def __call__(self, *, data, **kwargs):
        for module_id, batch in data.items():
            if module_id != "imager":
                continue
            final = Postprocessing.VALUE_TARGETS in batch
            if final:
                actions = numeric(batch[Columns.ACTIONS])
                action_count = numeric(batch[Columns.ACTION_DIST_INPUTS]).shape[-1]
                assert np.all(
                    (actions >= 0) & (actions < action_count)
                ), "Forced NO_ACTION or invalid action reached the learner"
                assert (
                    Columns.LOSS_MASK not in batch
                    or numeric(batch[Columns.LOSS_MASK]).all()
                ), "Unexpected padded learner batch"
            save_numeric_batch(
                self.directory,
                "learner-batch.npz" if final else "bootstrap-batch.npz",
                batch,
            )
        return data


def save_raw_episodes(directory, episodes):
    directory = Path(directory)
    if (directory / "raw-episodes.json.gz").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    records = [
        dict(
            id=episode.id_,
            multi_agent_episode_id=episode.multi_agent_episode_id,
            agent_id=episode.agent_id,
            module_id=episode.module_id,
            observations=numeric(episode.observations.data).tolist(),
            actions=list(episode.actions.data),
            rewards=list(episode.rewards.data),
            infos=list(episode.infos.data),
            terminated=episode.is_terminated,
            truncated=episode.is_truncated,
        )
        for episode in episodes
    ]
    with gzip.open(directory / "raw-episodes.json.gz", "xt") as stream:
        json.dump(records, stream, default=lambda value: numeric(value).tolist())
