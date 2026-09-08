"""Tests for runtime package identity after RLlib changes ``sys.path``."""

from examples.multiagent_imaging.cluster import audit_runtime


class FakeDistribution:
    """Small importlib.metadata distribution stand-in."""

    def __init__(self, name, version, path):
        self.metadata = {"Name": name}
        self.version = version
        self._path = path


def test_runtime_audit_ignores_ray_vendored_metadata(monkeypatch):
    """Ray's compatibility bundle must not replace the environment pin."""
    distributions = [
        FakeDistribution(
            "psutil",
            "6.0.0",
            "/venv/site-packages/ray/thirdparty_files/psutil-6.0.0.dist-info",
        ),
        FakeDistribution(
            "psutil", "6.1.0", "/venv/site-packages/psutil-6.1.0.dist-info"
        ),
    ]
    monkeypatch.setattr(
        audit_runtime.importlib.metadata,
        "distributions",
        lambda: distributions,
    )

    record = audit_runtime.distribution_record("PSUTIL")

    assert record["selected"]["version"] == "6.1.0"
    assert record["matches"][0]["ray_vendored"] is True
