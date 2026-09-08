"""Compatibility checks for legacy and current Basilisk package metadata."""

from importlib.metadata import PackageNotFoundError
import importlib

import pytest

version_check = importlib.import_module("bsk_rl.check_bsk_version")


def test_current_bsk_distribution_name_is_accepted(monkeypatch):
    """The current Basilisk project publishes metadata under the name ``bsk``."""

    def installed_version(name):
        if name == "bsk":
            return "2.2.1b0"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(version_check, "version", installed_version)
    version_check.check_bsk_version()


def test_legacy_basilisk_distribution_name_remains_supported(monkeypatch):
    """Existing wheel environments still expose the legacy distribution name."""

    def installed_version(name):
        if name == "Basilisk":
            return "2.2.1b0"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(version_check, "version", installed_version)
    version_check.check_bsk_version()


def test_missing_basilisk_metadata_is_rejected(monkeypatch):
    """Do not silently accept a source tree without installed package metadata."""

    def missing(_name):
        raise PackageNotFoundError

    monkeypatch.setattr(version_check, "version", missing)
    with pytest.raises(ImportError, match="Neither the 'bsk' nor legacy 'Basilisk'"):
        version_check.check_bsk_version()
