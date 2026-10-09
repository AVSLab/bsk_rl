"""Exercise the actual release producer and source gate without sampling."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from examples.multiagent_imaging.cluster.prepare_release import prepare
from examples.multiagent_imaging.cluster.release_manifest import (
    STUDY_CONFIGS, validate_study_manifest,
)
from examples.multiagent_imaging.cluster.study_gate import source_gate


@pytest.fixture
def released_tree(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    current = Path(__file__).resolve().parents[3]
    for name in STUDY_CONFIGS.values():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current / name, dest)
    # Historic evidence must stay outside the executable source overlay.
    history = root / "examples/multiagent_imaging/cluster/evidence/old.json"
    history.parent.mkdir(parents=True)
    history.write_text('{"historical": true}')
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Release fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "fixture"],
                   cwd=root, check=True)
    output = tmp_path / "release"
    prepare(output, root=root)
    manifest = json.loads((output / "release.json").read_text())
    (root / "completion-release.json").write_text(json.dumps(manifest))
    return root, output, manifest


def test_real_producer_passes_real_source_gate(released_tree):
    root, output, manifest = released_tree
    result = source_gate(root)
    assert result["passed"] and result["release"] == manifest["release_id"]
    assert set(result["configs"]) == set(STUDY_CONFIGS)
    with tarfile.open(output / "source-overlay.tar.gz") as archive:
        hashes = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                  for m in archive.getmembers() if m.isfile()}
    assert hashes == manifest["source_sha256"]
    assert all("evidence" not in Path(name).parts for name in hashes)
    for arm, name in STUDY_CONFIGS.items():
        assert manifest["config_sha256"][arm] == hashes[name]


@pytest.mark.parametrize("field", ["release_id", "source_sha256", "config_sha256"])
def test_missing_required_field_fails_before_runtime(released_tree, field):
    root, _, manifest = released_tree
    del manifest[field]
    with pytest.raises(ValueError, match="missing required field: " + field):
        validate_study_manifest(root, manifest)


@pytest.mark.parametrize("fault", ["source", "config", "missing_arm", "extra_arm", "empty_id"])
def test_malformed_or_mutated_release_fails_closed(released_tree, fault):
    root, _, manifest = released_tree
    name = STUDY_CONFIGS["catalog_tx"]
    if fault == "source":
        (root / name).write_text("mutated")
    elif fault == "config":
        manifest["config_sha256"]["catalog_tx"] = "0" * 64
    elif fault == "missing_arm":
        del manifest["config_sha256"]["geometry_only"]
    elif fault == "extra_arm":
        manifest["config_sha256"]["unexpected"] = "0" * 64
    else:
        manifest["release_id"] = ""
    with pytest.raises(ValueError):
        validate_study_manifest(root, manifest)
