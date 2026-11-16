Ground knowledge catalog
========================

A ground catalog represents estimates operators have actually received. Ground
clients share one :class:`~bsk_rl.data.GroundCatalog` instance per environment.
Each spacecraft keeps a separate local catalog or belief store. Giving a policy
a ground snapshot grants that policy ground knowledge; it does not model a radio
transfer. No component is attached to an environment automatically.

Run the synthetic example from a checkout with BSK-RL installed::

    python examples/ground_catalog.py

The example models information flow only: a spacecraft generates an estimate,
transfers it to ground, a telescope-like producer submits a newer estimate, and
another spacecraft receives it later. It does not simulate telescope optics,
mount motion, orbit determination, or communication physics.

Records and time tags
---------------------

:class:`~bsk_rl.data.CatalogEstimate` contains a target ID, original source and
update IDs, a six-element J2000 state (m, m/s), and a positive-definite 3x3 RTN
**position** covariance (m²). It is not a full 6x6 orbit-estimation covariance.
Supported frame and unit identifiers are validated, not converted.

Every record and packet names its time reference explicitly. Times are seconds
relative to that common origin; mismatched references are rejected. Use an exact
scenario epoch and time scale in production configuration.

* ``solution_time_s`` is the information epoch used to compare estimates.
* ``state_epoch_s`` is the epoch to which the state has been propagated.
* ``CatalogPacket.created_s`` is the packet freeze time.
* ``CatalogEntry.received_s`` is the local successful-delivery time.

Information time ≤ state epoch ≤ packet creation ≤ receipt. Forwarding or
propagation never makes information newer. Delivery calls must follow
nondecreasing receipt times. A delayed packet may contain old information while
still arriving at the current simulation time.

Knowledge and delivery
----------------------

.. code-block:: python

    from bsk_rl.data import GroundCatalog

    ground = GroundCatalog(
        time_reference="seconds since 2026-01-01T00:00:00 UTC",
        initial_estimates=prior,
        initial_received_s=0.0,
    )
    # Called by a transport or local-processing adapter after successful delivery:
    changed_ids = ground.ingest(packet, received_s=sim_time)
    operator_view = ground.snapshot()

Initial estimates are stored as actual knowledge and are restored by ``reset``.
All configured prior states must be available by ``initial_received_s``. An empty
prior produces an empty catalog. Environment code owns reset and registration;
there is no shared global singleton.

Only strictly newer solution times replace a target's entry. Equal-time estimates
retain the first accepted record and its provenance. Simultaneous deliveries
therefore require stable caller-defined ordering; the rule is deterministic for
a given event order, not commutative fusion. A newer estimate need not have a
smaller covariance. This policy is an explicit baseline, not optimal estimation.

Records, packets, and snapshots are immutable copies of the supplied values.
Snapshots retain the original producer and update ID, plus the most recent
sender, packet creation time, receipt time, and local revision. A packet frozen
before a later ground update still delivers its original estimate and provenance.
No truth state is consulted or synthesized by this service.

Transport and prediction adapters
---------------------------------

``make_packet(sender_id=..., created_s=..., since_revision=...)`` selects records
changed after a local catalog revision. That watermark belongs to the sending
catalog; it is not the receiver's independently numbered revision. Transport
advances an acknowledgment ledger only after confirmed delivery, using the frozen
``packet.catalog_revision`` rather than the catalog's current revision. Creating a
packet, losing a link, or receiving a NACK does not update remote knowledge.

The catalog owns neither pass windows nor exchanges in progress. Keep access,
pointing, bitrate, latency, packet loss, retry, ACK/NACK and partial-delivery
behavior in a separate transport component. In particular, the existing
``CommunicationMethod`` data sharing is not automatically made contact-aware by
adding this class.

``snapshot_at(epoch_s, predict=callback)`` explicitly requests predicted copies.
The callback accepts a ``CatalogEstimate`` and epoch and returns a new estimate.
It must propagate both mean and RTN covariance consistently, preserve source,
update ID, information time, frames, units and reference, and return the requested
state epoch. The catalog validates that contract but does not assess the
physical accuracy of the callback. Predictions do not modify stored knowledge.

An ordinary Python adapter can consume completed Basilisk sensor/storage/link
events and call this API at the appropriate simulation time. A native scheduled
estimator may instead need a suitable Basilisk message boundary. No new compiled
message payload is required to use this Python service. A telescope measurement
must first pass through its measurement-processing/estimation model; raw angles
are not silently treated as a Cartesian state estimate.

State and limitations
---------------------

``state_dict`` and ``from_state_dict`` preserve committed entries, delivery time,
revision, and the reset prior. ``CatalogPacket.to_dict`` / ``from_dict`` provide
a versioned JSON-compatible packet representation. Transport must separately
serialize its pending exchanges or explicitly abort them on restart; catalog
serialization alone is not a complete simulator checkpoint.

This foundation supports single- and multi-agent applications without a learned
policy. It does not add sensor physics, full state covariance, correlation-aware
fusion, unknown-object discovery logic, RF modeling, or automatic policy features.

.. toctree::
    :hidden:

    examples/ground_catalog
