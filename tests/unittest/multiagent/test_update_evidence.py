"""Journal compatibility, corruption rejection, and streaming JSON boundaries."""
import io
import json

import pytest

from examples.multiagent_imaging import update_evidence as evidence


def record(iteration):
    return dict(iteration=iteration, episodes=[dict(episode_id=f"{iteration}",
        reward_events=[dict(sensor="sensor_0", time_s=45000., reward=.125)],
        text='quote " slash \\ unicode \u2603')], losses={"total_loss": -.1})


def test_journal_and_legacy_preserve_all_scientific_fields(tmp_path):
    journal = evidence.UpdateJournal(tmp_path / "train")
    expected = [record(i) for i in (1, 2, 3)]
    for item in expected:
        journal.append(item)
    rows = evidence.UpdateRecords(tmp_path / "train")
    assert len(rows) == 3 and rows[-1] == expected[-1] and rows[0] == expected[0]
    assert list(rows) == expected
    legacy = tmp_path / "legacy.json"
    rows.export_legacy_array(legacy)
    assert list(evidence.UpdateRecords(legacy)) == expected
    assert list(rows + evidence.UpdateRecords(legacy)) == expected * 2
    assert len(rows + evidence.UpdateRecords(legacy)) == 6
    assert (rows + evidence.UpdateRecords(legacy))[-1] == expected[-1]
    assert (tmp_path / "train/updates.json").stat().st_size < 200
    with pytest.raises(FileExistsError):
        evidence.UpdateJournal(tmp_path / "train")
    with pytest.raises(ValueError):
        journal.append(record(3))


@pytest.mark.parametrize("change", ["contents", "index", "orphan", "partial", "path"])
def test_corruption_or_interrupted_publication_is_rejected(tmp_path, change):
    journal = evidence.UpdateJournal(tmp_path)
    journal.append(record(1))
    if change == "contents":
        with (journal.root / "update-000001.json").open("a") as stream:
            stream.write(" ")
    elif change == "index":
        with (journal.root / "index.jsonl").open("a") as stream:
            stream.write('{"iteration":2')
    elif change == "orphan":
        (journal.root / "update-000002.json").write_text("{}")
    elif change == "partial":
        (journal.root / "update-000002.partial").write_text("{")
    else:
        index = journal.root / "index.jsonl"
        item = json.loads(index.read_text())
        item["file"] = "../private.json"
        index.write_text(json.dumps(item) + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        list(evidence.UpdateRecords(tmp_path))
    with pytest.raises((ValueError, FileNotFoundError)):
        evidence.UpdateRecords(tmp_path).export_legacy_array(tmp_path / "export.json")
    assert not (tmp_path / "export.json").exists()


def test_failed_serialization_never_publishes_update(tmp_path):
    journal = evidence.UpdateJournal(tmp_path)
    with pytest.raises(ValueError):
        journal.append(dict(iteration=1, reward=float("nan")))
    assert list(evidence.UpdateRecords(tmp_path)) == []
    assert list(journal.root.iterdir()) == [journal.root / "index.jsonl"]


def test_indexed_reads_also_reject_a_mislabeled_update(tmp_path):
    journal = evidence.UpdateJournal(tmp_path)
    journal.append(record(1))
    path = journal.root / "update-000001.json"
    path.write_text(json.dumps(record(2)))
    index = journal.root / "index.jsonl"
    entry = json.loads(index.read_text())
    entry.update(size_bytes=path.stat().st_size, sha256=evidence.file_sha256(path))
    index.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError, match="identity"):
        evidence.UpdateRecords(tmp_path)[0]


@pytest.mark.parametrize("chunk", [1, 2, 7, 64])
def test_streaming_json_crosses_strings_unicode_numbers_and_object_keys(monkeypatch, chunk):
    monkeypatch.setattr(evidence, "CHUNK_BYTES", chunk)
    values = [record(1), -1.254e-123, 1234567, None, True, [], {"nested": [1,2,3]}]
    assert list(evidence.iter_json_array(io.StringIO(json.dumps(values)))) == values
    reader = evidence.JSONStream(io.StringIO(json.dumps(dict(episodes=values, iteration=2))))
    result = {}
    for key in reader.keys():
        result[key] = list(reader.array()) if key == "episodes" else reader.value()
    reader.finish()
    assert result == dict(episodes=values, iteration=2)


@pytest.mark.parametrize("text", ['[1,', '[1,]', '[{"x":1}', '[1] garbage', '[1e]', '["unterminated]'])
def test_malformed_or_partial_legacy_evidence_is_rejected(text):
    with pytest.raises(ValueError):
        list(evidence.iter_json_array(io.StringIO(text)))
