"""Bounded update storage, with read compatibility for historical JSON arrays.

One full update is live at a time. Scientific payloads are never summarized or
dropped. The small updates.json descriptor points to an append-only hash index.
No Ray, simulator, or third-party parser is imported by this module.
"""
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil

FORMAT = "bsk-rl-update-journal-v1"
CHUNK_BYTES = 64 * 1024


def peak_rss_bytes():
    """Normalize getrusage's byte units on macOS and KiB units on Linux."""
    import resource
    import sys
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class JSONStream:
    """Incremental stdlib JSON reader; memory is bounded by the current value.

    Container iterators let audit callers process episodes within an update
    without materializing the entire update. Strict separators and trailing-data
    checks reject truncated or malformed evidence rather than accepting a prefix.
    """
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def fill(self, size=CHUNK_BYTES):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        block = self.stream.read(size)
        if isinstance(block, bytes):
            raise TypeError("JSONStream requires a UTF-8 text stream")
        self.buffer += block
        self.eof = not block

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.fill()

    def expect(self, char):
        if self.peek() != char:
            raise ValueError(f"Expected JSON {char!r}, got {self.peek()!r}")
        self.position += 1

    def value(self):
        self.peek()
        read_size = CHUNK_BYTES
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # A number ending at a chunk boundary can still be incomplete.
                if end == len(self.buffer) and not self.eof:
                    self.fill(read_size)
                    read_size *= 2
                    continue
                if end < len(self.buffer) and self.buffer[end] not in " \t\r\n,:]}":
                    raise json.JSONDecodeError("Invalid JSON value delimiter", self.buffer, end)
                self.position = end
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError("Truncated or malformed JSON evidence") from None
                self.fill(read_size)
                read_size *= 2

    def array(self):
        self.expect("[")
        if self.peek() != "]":
            while True:
                yield self.value()
                if self.peek() == "]":
                    break
                self.expect(",")
        self.expect("]")

    def keys(self):
        self.expect("{")
        seen = set()
        if self.peek() != "}":
            while True:
                key = self.value()
                if not isinstance(key, str) or key in seen:
                    raise ValueError("Invalid or repeated JSON object key")
                seen.add(key)
                self.expect(":")
                yield key  # The caller must consume the field's value.
                if self.peek() == "}":
                    break
                self.expect(",")
        self.expect("}")

    def finish(self):
        if self.peek():
            raise ValueError("Trailing data after JSON evidence")


def iter_json_array(stream):
    reader = JSONStream(stream)
    yield from reader.array()
    reader.finish()


class UpdateJournal:
    """Single-writer journal for a fresh training output, published per update."""
    def __init__(self, output):
        self.output = Path(output)
        if (self.output / "updates.json").exists() or (self.output / "update-journal").exists():
            raise FileExistsError("Existing update evidence; use a fresh output directory")
        self.output.mkdir(parents=True, exist_ok=True)
        self.root = self.output / "update-journal"
        self.root.mkdir()
        (self.root / "index.jsonl").touch(exist_ok=False)
        with (self.output / "updates.json").open("x") as stream:
            json.dump(dict(format=FORMAT, index="update-journal/index.jsonl"), stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(self.output)
        self.last_iteration = -1

    def append(self, record):
        iteration = record["iteration"]
        if type(iteration) is not int or iteration <= self.last_iteration or iteration < 0:
            raise ValueError("Update iterations must be strictly increasing nonnegative integers")
        target = self.root / f"update-{iteration:06d}.json"
        partial = target.with_suffix(".partial")
        try:
            with partial.open("x") as stream:
                json.dump(record, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            entry = dict(iteration=iteration, file=target.name,
                         size_bytes=partial.stat().st_size, sha256=file_sha256(partial))
            # Exclusive publication never overwrites an immutable update.
            os.link(partial, target)
            partial.unlink()
            _sync_directory(self.root)
            with (self.root / "index.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.last_iteration = iteration
            return entry
        finally:
            partial.unlink(missing_ok=True)


class UpdateRecords(Sequence):
    """Read-only, disk-backed sequence. Iteration never retains prior updates."""
    def __init__(self, output):
        self.path = Path(output)
        if self.path.is_dir():
            self.path = self.path / "updates.json"
        with self.path.open() as stream:
            first = JSONStream(stream).peek()
        self.journal = first == "{"
        if self.journal:
            if self.path.stat().st_size > CHUNK_BYTES:
                raise ValueError("Oversized update descriptor")
            descriptor = json.loads(self.path.read_text())
            if descriptor != dict(format=FORMAT, index="update-journal/index.jsonl"):
                raise ValueError("Unknown update evidence descriptor")
            self.root = self.path.parent / "update-journal"
        elif first != "[":
            raise ValueError("Unknown update evidence format")

    def entries(self):
        previous, seen = -1, set()
        with (self.root / "index.jsonl").open() as stream:
            while True:
                line = stream.readline(CHUNK_BYTES + 1)
                if not line:
                    break
                if len(line) > CHUNK_BYTES or not line.endswith("\n"):
                    raise ValueError("Oversized or incomplete journal index entry")
                entry = json.loads(line)
                iteration = entry["iteration"]
                if type(iteration) is not int or iteration <= previous or iteration < 0:
                    raise ValueError("Duplicate or unordered update iteration")
                if entry["file"] != f"update-{iteration:06d}.json":
                    raise ValueError("Unsafe or inconsistent update filename")
                source = self.root / entry["file"]
                if source.is_symlink() or source.stat().st_size != entry["size_bytes"]:
                    raise ValueError("Update size or file type mismatch")
                if file_sha256(source) != entry["sha256"]:
                    raise ValueError("Update evidence hash mismatch")
                previous = iteration
                seen.add(entry["file"])
                yield entry
        actual = {p.name for p in self.root.iterdir() if p.name != "index.jsonl"}
        if actual != seen:
            raise ValueError("Unindexed or partial update evidence; collection is incomplete")

    def __iter__(self):
        if self.journal:
            for entry in self.entries():
                record = self._load_entry(entry)
                yield record
                del record
        else:
            with self.path.open() as stream:
                yield from iter_json_array(stream)

    def _load_entry(self, entry):
        with (self.root / entry["file"]).open() as stream:
            record = json.load(stream)
        if record["iteration"] != entry["iteration"]:
            raise ValueError("Update identity differs from index")
        return record

    def __len__(self):
        return sum(1 for _ in self.entries()) if self.journal else sum(1 for _ in self)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("Use streaming iteration instead of materializing an update slice")
        count = len(self)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(index)
        if self.journal:
            entry = next(e for i, e in enumerate(self.entries()) if i == index)
            return self._load_entry(entry)
        return next(record for i, record in enumerate(self) if i == index)

    def __add__(self, other):
        return CombinedUpdates(self, other)

    def export_legacy_array(self, destination):
        """Explicit compatibility export; training never rewrites this history."""
        destination = Path(destination)
        partial = destination.with_suffix(destination.suffix + ".partial")
        if destination.exists():
            raise FileExistsError(destination)
        try:
            with partial.open("xb") as stream:
                if self.journal:
                    stream.write(b"[")
                    for i, entry in enumerate(self.entries()):
                        if i:
                            stream.write(b",")
                        with (self.root / entry["file"]).open("rb") as source:
                            shutil.copyfileobj(source, stream, CHUNK_BYTES)
                    stream.write(b"]\n")
                else:
                    # Validate before publishing an unchanged historical array.
                    for _ in self:
                        pass
                    with self.path.open("rb") as source:
                        shutil.copyfileobj(source, stream, CHUNK_BYTES)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(partial, destination)
            _sync_directory(destination.parent)
        finally:
            partial.unlink(missing_ok=True)


class CombinedUpdates(Sequence):
    """Compatibility for bounded pilot callers that concatenate two tranches."""
    def __init__(self, *groups):
        self.groups = groups

    def __iter__(self):
        for group in self.groups:
            yield from group

    def __len__(self):
        return sum(map(len, self.groups))

    def __getitem__(self, index):
        count = len(self)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(index)
        for group in self.groups:
            if index < len(group):
                return group[index]
            index -= len(group)

    def __add__(self, other):
        return CombinedUpdates(*self.groups, other)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Training output directory")
    parser.add_argument("--export-legacy", type=Path, required=True, help="Fresh destination array file")
    arguments = parser.parse_args()
    UpdateRecords(arguments.output).export_legacy_array(arguments.export_legacy)
