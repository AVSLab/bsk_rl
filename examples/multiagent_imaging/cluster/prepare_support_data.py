"""Fetch only the mission's physical data and verify the saved preflight hashes.

This performs file I/O and downloads, not a simulation or Ray launch. Run once
before concurrent episodes, with BSK_SUPPORT_DATA_CACHE pointing at the separate
completion checkout's cache, and export that setting to all Slurm jobs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


# Measured from the successful desktop mission preflight's actual resolved files.
# Basilisk does not pin external NAIF files itself; these hashes close that gap.
EXPECTED = {
    "de430.bsp": "6e1b277c5f07135a84950604b83e56b736be696a7f3560bcddb1d4aeb944fca1",
    "naif0012.tls": "678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b",
    "de-403-masses.tpc": "5cd68fcd3f59ddc21ed8bbad3a341126b8e092a29ac8f0ae585db718af8e7468",
    "pck00010.tpc": "59468328349aa730d18bf1f8d7e86efe6e40b75dfb921908f99321b3a7a701d2",
    "GGM03S.txt": "71f8736f98c5d8c972d0f6d56a2389302f72b8ac7517c605d312162ca446a3a1",
}


def prepare(output):
    if not os.environ.get("BSK_SUPPORT_DATA_CACHE"):
        raise ValueError(
            "Set the separate BSK_SUPPORT_DATA_CACHE before importing Basilisk."
        )
    from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path

    entries = [
        DataFile.EphemerisData.de430,
        DataFile.EphemerisData.naif0012,
        DataFile.EphemerisData.de_403_masses,
        DataFile.EphemerisData.pck00010,
        DataFile.LocalGravData.GGM03S,
    ]
    records, errors = {}, []
    for entry in entries:
        path = get_path(entry)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        records[entry.value] = dict(
            path=str(path),
            size_bytes=path.stat().st_size,
            sha256=actual,
            expected_sha256=EXPECTED[entry.value],
        )
        if actual != EXPECTED[entry.value]:
            errors.append(f"{entry.value}: differs from the measured preflight data")
    record = dict(
        passed=not errors,
        errors=errors,
        files=records,
        cache=os.environ["BSK_SUPPORT_DATA_CACHE"],
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            dict(passed=record["passed"], errors=errors, file_count=len(records))
        )
    )
    if errors:
        raise ValueError("Mission support-data validation failed.")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
