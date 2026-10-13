"""Opt-in host diagnostics; never observe policy data or initialize Ray.

RSS sums can double-count shared pages. PSS, USS and the effective cgroup are
reported separately; no process commands, environments or credentials are read.
"""
import json
import math
import os
from pathlib import Path
import threading
import time


def cgroup_directories(proc=Path("/proc")):
    """Resolve this process's v1/v2 memory cgroup using its mount namespace."""
    membership = (proc / "self/cgroup").read_text().splitlines()
    mounts = (proc / "self/mountinfo").read_text().splitlines()
    directories = []
    for row in membership:
        _, controllers, relative = row.split(":", 2)
        for mount in mounts:
            left, right = mount.split(" - ", 1)
            fields, kind = left.split(), right.split()
            if not ((controllers == "" and kind[0] == "cgroup2") or
                    ("memory" in controllers.split(",") and kind[0] == "cgroup" and
                     "memory" in kind[2].split(","))):
                continue
            base, root = Path(fields[4]), Path(fields[3])
            path = Path(relative)
            if not path.is_relative_to(root):
                continue
            directory = base / path.relative_to(root)
            # A parent job cgroup can impose a tighter limit than the step.
            while directory.is_relative_to(base):
                if directory not in directories:
                    directories.append(directory)
                if directory == base:
                    break
                directory = directory.parent
    return directories


def read_cgroup_memory(proc=Path("/proc")):
    records = []
    for directory in cgroup_directories(proc):
        values = {}
        for name in ("memory.current", "memory.peak", "memory.max", "memory.events",
                     "memory.stat", "memory.usage_in_bytes", "memory.max_usage_in_bytes",
                     "memory.limit_in_bytes", "memory.failcnt"):
            path = directory / name
            if path.is_file():
                values[name] = path.read_text().strip()
        records.append(dict(path=str(directory), values=values))
    return records


def ray_memory_options(environ=None):
    """Declare an absolute object-store cap; keep the old default when unset."""
    environ = os.environ if environ is None else environ
    raw = environ.get("BSK_RL_RAY_OBJECT_STORE_BYTES")
    if raw is None:
        return {}
    size = int(raw)
    if size < 80 * 1024**2:
        raise ValueError("Ray object store must be at least 80 MiB")
    allocation = environ.get("BSK_MEMORY_ALLOCATION_BYTES")
    if allocation is not None and size >= int(allocation):
        raise ValueError("Object store must leave space for worker/learner heaps")
    return dict(object_store_memory=size)


class MemoryProfile:
    """Bounded append-only periodic driver/child and job-cgroup measurement."""
    def __init__(self, output, interval_s=None):
        raw = os.environ.get("BSK_RL_MEMORY_PROFILE_INTERVAL_S")
        self.interval = interval_s if interval_s is not None else (float(raw) if raw else None)
        if self.interval is not None and (not math.isfinite(self.interval) or self.interval < 5):
            raise ValueError("Memory profile interval must be finite and at least five seconds")
        self.path = Path(output) / "memory-profile.jsonl"
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.phase = "before-ray"
        self.started = time.monotonic()
        self.thread = None
        self.stream = None
        self.error = None

    def start(self):
        if self.interval is None:
            return
        self.stream = self.path.open("x")
        self.mark("before-ray")
        self.thread = threading.Thread(target=self._loop, name="memory-profile", daemon=True)
        self.thread.start()

    def _snapshot(self):
        import psutil
        root = psutil.Process()
        processes, errors = [], []
        for process in [root, *root.children(recursive=True)]:
            try:
                memory = process.memory_info()
                row = dict(pid=process.pid, name=process.name(), rss_bytes=memory.rss,
                           shared_bytes=getattr(memory, "shared", None))
                try:
                    full = process.memory_full_info()
                    row.update(pss_bytes=getattr(full, "pss", None), uss_bytes=getattr(full, "uss", None))
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    row.update(pss_bytes=None, uss_bytes=None)
                processes.append(row)
            except (psutil.AccessDenied, psutil.NoSuchProcess) as error:
                errors.append(dict(pid=process.pid, error_type=type(error).__name__))
        cgroups = []
        if Path("/proc/self/cgroup").exists():
            cgroups = read_cgroup_memory()
        return dict(version="memory-profile-v1", phase=self.phase,
                    time_unix_s=time.time(), elapsed_s=time.monotonic()-self.started, processes=processes,
                    rss_sum_bytes=sum(p["rss_bytes"] for p in processes),
                    rss_sum_is_job_memory=False, cgroups=cgroups, errors=errors)

    def mark(self, phase):
        if self.stream is None:
            return
        with self.lock:
            self.phase = phase
            self.stream.write(json.dumps(self._snapshot(), allow_nan=False) + "\n")
            self.stream.flush()

    def _loop(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.mark(self.phase)
        except Exception as error:
            self.error = error

    def close(self):
        if self.stream is None:
            return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self.mark("shutdown")
        finally:
            self.stream.close()
        if self.error is not None:
            raise RuntimeError("Required memory profiling failed") from self.error
