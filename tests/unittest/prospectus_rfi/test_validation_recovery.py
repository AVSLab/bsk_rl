from pathlib import Path

from examples.prospectus_rfi.validation_campaign import (
    build_tasks,
    completed_task,
    task_output_relative,
    slurm_array_expression,
    write_task_shards,
)


def _checkpoint_tree(root: Path, architecture: str, candidate_count: int) -> None:
    checkpoint_root = (
        root
        / "training"
        / f"{architecture}_k{candidate_count}_seed10001"
        / "checkpoints"
    )
    (checkpoint_root / "iteration_003").mkdir(parents=True)
    (checkpoint_root / "final").mkdir()


def test_validation_manifest_has_one_task_per_checkpoint_seed_and_catalog(tmp_path):
    for architecture in ("mlp", "attention"):
        for candidate_count in (5, 10, 20):
            _checkpoint_tree(tmp_path, architecture, candidate_count)

    tasks = build_tasks(
        tmp_path, catalog_sizes=(100, 150, 200), seeds=(91001, 91002)
    )

    # 2 architectures * 3 candidate counts * 2 checkpoints * 3 catalog sizes * 2 seeds
    assert len(tasks) == 72
    assert [task["task_id"] for task in tasks] == list(range(72))
    assert len({task["output_name"] for task in tasks}) == 72
    assert tasks[0]["checkpoint_name"] == "iteration_003"
    assert tasks[0]["output_name"].endswith("_n100_seed91001.csv")


def test_completed_task_requires_csv_and_metadata(tmp_path):
    task = {"output_name": "mlp_k5_final_n100_seed91001.csv"}
    output = tmp_path / task_output_relative(task)
    output.parent.mkdir(parents=True)
    output.write_text("value\n1\n")
    assert not completed_task(task, tmp_path)
    output.with_suffix(".metadata.json").write_text("{}\n")
    assert completed_task(task, tmp_path)


def test_task_ids_are_compacted_for_slurm_array_submission():
    assert slurm_array_expression([]) == ""
    assert slurm_array_expression([0, 1, 2, 5, 7, 8, 10]) == "0-2,5,7-8,10"


def test_validation_uses_only_last_five_iterations_plus_final(tmp_path):
    checkpoint_root = tmp_path / "training" / "mlp_k5_seed10001" / "checkpoints"
    for iteration in (3, 6, 9, 12, 15, 18, 21):
        (checkpoint_root / f"iteration_{iteration:03d}").mkdir(parents=True)
    (checkpoint_root / "final").mkdir()
    from examples.prospectus_rfi.validation_campaign import checkpoint_directories

    assert [path.name for path in checkpoint_directories(checkpoint_root.parent)] == [
        "iteration_009",
        "iteration_012",
        "iteration_015",
        "iteration_018",
        "iteration_021",
        "final",
    ]


def test_task_maps_keep_slurm_array_indices_bounded(tmp_path):
    paths = write_task_shards(list(range(540)), tmp_path, shard_size=400)
    assert [len(path.read_text().splitlines()) for path in paths] == [400, 140]
    assert paths[1].read_text().splitlines()[0] == "400"
