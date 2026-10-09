"""Versioned physical-owner metadata; no task, intent, resource or image bytes."""
from dataclasses import asdict, dataclass
import json
from math import isfinite

PROTOCOL_VERSION = "completed-onboard-v1"
HEADER_BYTES = 64

@dataclass(frozen=True)
class InventoryEntry:
    fact: object
    revision: int
    status: str
    revised_at: float
    @property
    def record_id(self): return self.fact.record_id
    @property
    def version(self): return self.revision
    @property
    def source_sensor(self): return self.fact.source_sensor

@dataclass(frozen=True)
class InventorySnapshot:
    owner: str
    revision: int
    epoch: float
    entries: tuple
    complete: bool
    protocol: str = PROTOCOL_VERSION
    def wire_bytes(self):
        return HEADER_BYTES + len(json.dumps(asdict(self), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode())

class ReceivedInventory:
    """Partial snapshots cannot infer absence; status expires, history does not."""
    def __init__(self):
        self.entries = {}
        self.snapshot_epoch = {}
        self.complete_revision = {}
        self.receipts = {}
    def receive(self, snapshot, now, ttl):
        if (snapshot.protocol != PROTOCOL_VERSION or not isfinite(snapshot.epoch)
                or not isfinite(now) or not isfinite(ttl) or ttl <= 0
                or not isinstance(snapshot.revision,int) or snapshot.revision < 0):
            raise ValueError("Inventory protocol/epoch mismatch")
        if not snapshot.epoch <= now < snapshot.epoch + ttl:
            raise ValueError("Future or expired inventory snapshot")
        ids = [e.record_id for e in snapshot.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated inventory identity")
        changed = 0
        for e in snapshot.entries:
            if (e.source_sensor != snapshot.owner or e.status not in {"onboard", "delivered", "dropped"}
                    or not isinstance(e.revision,int) or not 0 < e.revision <= snapshot.revision or not isfinite(e.revised_at)
                    or e.revised_at > snapshot.epoch):
                raise ValueError("Malformed inventory provenance/revision")
            key = (snapshot.owner, e.record_id)
            old = self.entries.get(key)
            if old and e.revision > old[0].revision and old[0].status in {"delivered", "dropped"} and e.status != old[0].status:
                raise ValueError("Terminal inventory cannot resurrect")
            if old and old[0].revision == e.revision and old[0] != e:
                raise ValueError("Conflicting inventory revision")
            if e.revision < self.complete_revision.get(snapshot.owner, 0) and old is None:
                continue
            if old and e.revision < old[0].revision:
                continue
            if old and e.revision == old[0].revision and snapshot.epoch <= old[1]:
                continue
            self.entries[key] = (e, snapshot.epoch, float(ttl))
            self.receipts.setdefault(e.record_id, float(now))
            changed += 1
        if snapshot.complete and snapshot.revision >= self.complete_revision.get(snapshot.owner, 0):
            # A complete newer snapshot is authoritative only for entries it can
            # supersede. Inference yields unknown/absent, never a delivery fact.
            for key, (e, epoch, life) in list(self.entries.items()):
                if key[0] == snapshot.owner and key[1] not in ids and e.revision <= snapshot.revision and epoch <= snapshot.epoch:
                    del self.entries[key]
            self.complete_revision[snapshot.owner] = snapshot.revision
        self.snapshot_epoch[snapshot.owner] = max(snapshot.epoch,
            self.snapshot_epoch.get(snapshot.owner, float("-inf")))
        return changed
    def status(self, owner, record_id, now):
        row = self.entries.get((owner, record_id))
        if row is None: return "unknown"
        e, epoch, ttl = row
        return e.status if now < epoch + ttl else "unknown"
    def current(self, now, *, target_id=None, owner=None):
        return [e for (who, rid), (e, epoch, ttl) in self.entries.items()
            if (owner is None or who == owner) and (target_id is None or e.fact.target_id == target_id)
            and self.status(who, rid, now) == "onboard"]
