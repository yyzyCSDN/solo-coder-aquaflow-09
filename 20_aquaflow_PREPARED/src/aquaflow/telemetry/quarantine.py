"""Telemetry quality tracking and credibility isolation.

Each asset owns a :class:`QualityStream`.  Raw samples are append-only and
immutable; derived quality state (gaps, out-of-order arrivals, credibility)
is recomputed from the raw history.  When credibility falls below the
policy threshold the stream is *quarantined* and downstream decisions must
read an explicit ``unavailable`` marker instead of stale values.

Reviewers confirm the quarantine, backfill missing samples and restore the
stream.  A restore is only accepted when the continuity check passes and
credibility satisfies the policy currently in force.  Every state change is
an immutable event carrying the policy version that was applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEALTHY = 'healthy'
QUARANTINED = 'quarantined'
CONFIRMED = 'confirmed'

_LIVE = 'live'
_BACKFILL = 'backfill'


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TelemetryQualityError(ValueError):
    """Base class for telemetry quality violations."""


class ImmutableSampleError(TelemetryQualityError):
    """Raised when an existing raw sample would be modified."""


class InvalidTransitionError(TelemetryQualityError):
    """Raised when an action is illegal for the current stream state."""


class ContinuityError(TelemetryQualityError):
    """Raised when a restore is attempted with unresolved continuity gaps."""


class TimeReversalError(TelemetryQualityError):
    """Raised when an event is older than the stream clock."""


# ---------------------------------------------------------------------------
# Immutable facts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """An immutable raw telemetry sample."""

    asset: str
    at: datetime
    value: float
    sequence: int
    quality: float = 1.0


@dataclass(frozen=True)
class QualityPolicy:
    """Credibility / continuity policy.

    ``version`` travels with every state change so the audit trail shows
    exactly which policy produced each decision.
    """

    version: str
    expected_interval: timedelta
    gap_tolerance: float = 0.25
    min_credibility: float = 0.8
    gap_penalty: float = 0.2
    out_of_order_penalty: float = 0.2
    max_staleness: timedelta = timedelta(minutes=15)

    def __post_init__(self) -> None:
        if self.expected_interval.total_seconds() <= 0:
            raise TelemetryQualityError('expected_interval must be positive')
        if self.gap_tolerance < 0:
            raise TelemetryQualityError('gap_tolerance must be non-negative')
        if not 0.0 <= self.min_credibility <= 1.0:
            raise TelemetryQualityError('min_credibility must be within [0,1]')
        if self.gap_penalty < 0 or self.out_of_order_penalty < 0:
            raise TelemetryQualityError('penalties must be non-negative')
        if self.max_staleness.total_seconds() <= 0:
            raise TelemetryQualityError('max_staleness must be positive')
        if not self.version:
            raise TelemetryQualityError('policy version is required')

    @property
    def gap_threshold(self) -> timedelta:
        return self.expected_interval * (1.0 + self.gap_tolerance)


@dataclass(frozen=True)
class Arrival:
    """A raw sample on the wire and how it arrived."""

    sample: Sample
    kind: str  # 'live' | 'backfill'
    at: datetime
    reviewer: str | None = None
    out_of_order: bool = False
    accepted: bool = True


@dataclass(frozen=True)
class StateEvent:
    """An immutable state-change record (the audit entry)."""

    at: datetime
    action: str
    from_state: str
    to_state: str
    policy_version: str
    credibility: float
    reviewer: str | None = None
    reason: str = ''


@dataclass(frozen=True)
class ContinuityReport:
    """Result of a continuity check over the raw history."""

    asset: str
    passed: bool
    gaps: tuple[tuple[datetime, datetime], ...]
    expected_samples: int
    observed_samples: int
    completeness: float
    policy_version: str

    @property
    def missing_slots(self) -> int:
        return max(0, self.expected_samples - self.observed_samples)


@dataclass(frozen=True)
class DecisionView:
    """What a downstream control loop is allowed to see.

    ``available`` is the single explicit answer.  When it is ``False`` the
    ``value`` is ``None`` and callers must fail safe rather than reuse any
    previously observed value.
    """

    asset: str
    available: bool
    value: float | None
    sampled_at: datetime | None
    state: str
    reason: str
    evaluated_at: datetime
    policy_version: str


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _utc(at: datetime) -> datetime:
    if at.tzinfo is None:
        return at.replace(tzinfo=timezone.utc)
    return at.astimezone(timezone.utc)


def _ordered_samples(samples: dict[int, Sample]) -> list[Sample]:
    return sorted(samples.values(), key=lambda s: (_utc(s.at), s.sequence))


def detect_gaps(samples: list[Sample], policy: QualityPolicy
                ) -> tuple[tuple[datetime, datetime], ...]:
    """Return the (a, b) adjacent pairs whose spacing exceeds the policy."""
    rows = _ordered_samples({s.sequence: s for s in samples})
    limit = policy.gap_threshold
    found: list[tuple[datetime, datetime]] = []
    for a, b in zip(rows, rows[1:]):
        if _utc(b.at) - _utc(a.at) > limit:
            found.append((a.at, b.at))
    return tuple(found)


def expected_slot_count(samples: list[Sample], policy: QualityPolicy) -> int:
    """Expected slots inside the observed sequence range (inclusive).

    The sequence range defines how many slots *should* exist between the
    oldest and newest observed samples; backfills shrink the difference
    without moving the endpoints.
    """
    rows = _ordered_samples({s.sequence: s for s in samples})
    if not rows:
        return 0
    return rows[-1].sequence - rows[0].sequence + 1


def continuity_report(samples: list[Sample], asset: str,
                      policy: QualityPolicy) -> ContinuityReport:
    """Check the materialised timeline for unresolved gaps."""
    gaps = detect_gaps(samples, policy)
    expected = expected_slot_count(samples, policy)
    observed = len({s.sequence for s in samples})
    completeness = 1.0 if expected == 0 else min(1.0, observed / expected)
    return ContinuityReport(
        asset=asset,
        passed=bool(samples) and not gaps,
        gaps=gaps,
        expected_samples=expected,
        observed_samples=observed,
        completeness=completeness,
        policy_version=policy.version,
    )


def score_credibility(samples: list[Sample], arrivals: list[Arrival],
                      policy: QualityPolicy) -> float:
    """1.0 minus policy penalties for gaps and live out-of-order arrivals.

    Backfilled late data closes gaps without itself being penalised: the
    improvement shows up in the gap count.
    """
    penalty = (len(detect_gaps(samples, policy)) * policy.gap_penalty
               + sum(1 for a in arrivals
                     if a.kind == _LIVE and a.out_of_order)
               * policy.out_of_order_penalty)
    return max(0.0, min(1.0, 1.0 - penalty))


# ---------------------------------------------------------------------------
# Stream aggregate
# ---------------------------------------------------------------------------

@dataclass
class QualityStream:
    """Quality state for one asset, built from an append-only raw history."""

    asset: str
    policy: QualityPolicy
    state: str = HEALTHY
    samples: dict[int, Sample] = field(default_factory=dict)
    arrivals: list[Arrival] = field(default_factory=list)
    events: list[StateEvent] = field(default_factory=list)
    opened_at: datetime | None = None
    updated_at: datetime | None = None

    # -- introspection -----------------------------------------------------

    def raw_samples(self) -> tuple[Sample, ...]:
        """Raw samples in timeline order; never mutated in place."""
        return tuple(_ordered_samples(self.samples))

    def gaps(self) -> tuple[tuple[datetime, datetime], ...]:
        return detect_gaps(list(self.samples.values()), self.policy)

    def out_of_order_arrivals(self) -> tuple[Arrival, ...]:
        """Live arrivals whose sequence regressed.

        Late backfills are never counted: they repair history rather than
        violate the live ordering contract.
        """
        return tuple(a for a in self.arrivals
                     if a.kind == _LIVE and a.out_of_order)

    def credibility(self) -> float:
        return score_credibility(list(self.samples.values()),
                                 self.arrivals, self.policy)

    def report(self) -> ContinuityReport:
        return continuity_report(list(self.samples.values()),
                                 self.asset, self.policy)

    def latest_sample(self) -> Sample | None:
        rows = self.raw_samples()
        return rows[-1] if rows else None

    # -- internal plumbing -------------------------------------------------

    def _tick(self, at: datetime) -> datetime:
        when = _utc(at)
        if self.updated_at is not None and when < self.updated_at:
            raise TimeReversalError('event older than stream clock')
        self.updated_at = when
        return when

    def _transition(self, when: datetime, action: str, to_state: str,
                    reviewer: str | None, reason: str) -> None:
        if to_state == self.state:
            return
        event = StateEvent(
            at=when,
            action=action,
            from_state=self.state,
            to_state=to_state,
            policy_version=self.policy.version,
            credibility=self.credibility(),
            reviewer=reviewer,
            reason=reason,
        )
        self.events.append(event)
        self.state = to_state

    def _store(self, sample: Sample, kind: str, when: datetime,
               reviewer: str | None) -> Arrival:
        existing = self.samples.get(sample.sequence)
        if existing is not None:
            same_value = (
                _utc(existing.at) == _utc(sample.at)
                and existing.value == sample.value
                and existing.asset == sample.asset
                and existing.quality == sample.quality
            )
            if not same_value:
                # Raw history is immutable: a conflicting resend is rejected
                # instead of overwriting the original value.
                raise ImmutableSampleError(
                    f'sample sequence {sample.sequence} already recorded '
                    f'with a different value')
            arrival = Arrival(existing, kind, when, reviewer,
                              out_of_order=False, accepted=False)
            self.arrivals.append(arrival)
            return arrival

        sequences = list(self.samples.keys())
        out_of_order = bool(sequences) and sample.sequence < max(sequences)
        arrival = Arrival(sample, kind, when, reviewer,
                          out_of_order=out_of_order, accepted=True)
        self.samples[sample.sequence] = sample
        self.arrivals.append(arrival)
        return arrival

    def _auto_quarantine(self, when: datetime) -> None:
        if self.state == HEALTHY and self.credibility() < self.policy.min_credibility:
            self._transition(
                when, 'auto_quarantine', QUARANTINED, None,
                f'credibility {self.credibility():.2f} below policy '
                f'{self.policy.version} threshold '
                f'{self.policy.min_credibility:.2f}')

    # -- commands ----------------------------------------------------------

    def ingest(self, sample: Sample, at: datetime) -> Arrival:
        """Accept a live sample; quarantine automatically on policy breach."""
        if sample.asset != self.asset:
            raise TelemetryQualityError('sample belongs to another asset')
        if self.state not in (HEALTHY, QUARANTINED, CONFIRMED):  # pragma: no cover
            raise InvalidTransitionError(f'unknown state {self.state}')
        when = self._tick(at)
        arrival = self._store(sample, _LIVE, when, None)
        self._auto_quarantine(when)
        return arrival

    def confirm_quarantine(self, reviewer: str, at: datetime,
                           note: str = '') -> StateEvent:
        """Reviewer acknowledges that the stream is rightfully isolated."""
        when = self._tick(at)
        if self.state != QUARANTINED:
            raise InvalidTransitionError(
                f'cannot confirm quarantine from state {self.state}')
        self._transition(when, 'confirm_quarantine', CONFIRMED, reviewer, note)
        return self.events[-1]

    def backfill(self, samples: list[Sample], reviewer: str,
                 at: datetime) -> tuple[Arrival, ...]:
        """Reviewer submits missing raw samples (append-only, immutable)."""
        if self.state == HEALTHY:
            raise InvalidTransitionError(
                'backfill is only accepted for isolated streams')
        when = self._tick(at)
        accepted: list[Arrival] = []
        for sample in sorted(samples, key=lambda s: s.sequence):
            if sample.asset != self.asset:
                raise TelemetryQualityError('sample belongs to another asset')
            # Backfilled late data fills history; it is never counted as an
            # out-of-order live arrival.
            accepted.append(self._store(sample, _BACKFILL, when, reviewer))
        # Healthy streams auto-quarantine; healthy is unreachable here, and
        # isolated streams never silently recover -- a reviewer must restore.
        return tuple(accepted)

    def restore(self, reviewer: str, at: datetime,
                note: str = '') -> tuple[ContinuityReport, StateEvent]:
        """Restore the stream after continuity + credibility checks pass."""
        when = self._tick(at)
        if self.state not in (QUARANTINED, CONFIRMED):
            raise InvalidTransitionError(
                f'cannot restore from state {self.state}')
        report = self.report()
        if not report.passed:
            raise ContinuityError(
                f'continuity check failed: {len(report.gaps)} unresolved '
                f'gap(s), {report.missing_slots} missing slot(s)')
        credibility = self.credibility()
        if credibility < self.policy.min_credibility:
            raise ContinuityError(
                f'credibility {credibility:.2f} below policy '
                f'{self.policy.version} threshold '
                f'{self.policy.min_credibility:.2f}')
        self._transition(when, 'restore', HEALTHY, reviewer, note)
        return report, self.events[-1]

    def can_restore(self) -> bool:
        return (self.state in (QUARANTINED, CONFIRMED)
                and self.report().passed
                and self.credibility() >= self.policy.min_credibility)

    def apply_policy(self, policy: QualityPolicy, at: datetime
                     ) -> StateEvent | None:
        """Adopt a new policy version.

        Policy adoption never silently *recovers* a stream -- that still
        requires a reviewer restore that passes the continuity check.  A
        stricter threshold does take effect immediately: a healthy stream
        that no longer qualifies is quarantined under the new version.
        Both the revision and any resulting state change are events.
        """
        when = self._tick(at)
        previous = self.policy
        self.policy = policy
        event = StateEvent(
            at=when,
            action='policy_change',
            from_state=self.state,
            to_state=self.state,
            policy_version=policy.version,
            credibility=self.credibility(),
            reviewer=None,
            reason=f'policy {previous.version} -> {policy.version}',
        )
        self.events.append(event)
        if self.state == HEALTHY and self.credibility() < policy.min_credibility:
            self._transition(
                when, 'auto_quarantine', QUARANTINED, None,
                f'credibility {self.credibility():.2f} below new policy '
                f'{policy.version} threshold '
                f'{policy.min_credibility:.2f}')
        return event


def open_stream(asset: str, policy: QualityPolicy, at: datetime) -> QualityStream:
    """Open a healthy quality stream for an asset."""
    when = _utc(at)
    stream = QualityStream(asset=asset, policy=policy, opened_at=when,
                           updated_at=when)
    stream.events.append(StateEvent(
        at=when,
        action='open',
        from_state='',
        to_state=HEALTHY,
        policy_version=policy.version,
        credibility=1.0,
        reason='stream opened',
    ))
    return stream


# ---------------------------------------------------------------------------
# Downstream gate
# ---------------------------------------------------------------------------

def decision_view(stream: QualityStream, at: datetime) -> DecisionView:
    """Explicit availability gate for downstream control decisions.

    Isolated streams are unavailable.  Healthy streams whose newest sample
    exceeds ``max_staleness`` are also unavailable: stale data must never be
    served as if it were current.
    """
    when = _utc(at)
    latest = stream.latest_sample()

    if stream.state != HEALTHY:
        return DecisionView(
            asset=stream.asset, available=False, value=None,
            sampled_at=latest.at if latest else None, state=stream.state,
            reason=f'stream {stream.state}', evaluated_at=when,
            policy_version=stream.policy.version)

    if latest is None:
        return DecisionView(
            asset=stream.asset, available=False, value=None, sampled_at=None,
            state=stream.state, reason='no samples', evaluated_at=when,
            policy_version=stream.policy.version)

    if when - _utc(latest.at) > stream.policy.max_staleness:
        return DecisionView(
            asset=stream.asset, available=False, value=None,
            sampled_at=latest.at, state=stream.state, reason='stale',
            evaluated_at=when, policy_version=stream.policy.version)

    return DecisionView(
        asset=stream.asset, available=True, value=latest.value,
        sampled_at=latest.at, state=stream.state, reason='fresh',
        evaluated_at=when, policy_version=stream.policy.version)


def require_value(view: DecisionView) -> float:
    """Return the usable value or fail loudly -- never an expired value."""
    if not view.available:
        raise TelemetryQualityError(
            f'telemetry unavailable for {view.asset}: {view.reason}')
    return view.value  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Per-asset registry
# ---------------------------------------------------------------------------

class TelemetryQualityRegistry:
    """Holds one :class:`QualityStream` per asset."""

    def __init__(self, policy: QualityPolicy) -> None:
        self.policy = policy
        self._streams: dict[str, QualityStream] = {}

    def open(self, asset: str, at: datetime) -> QualityStream:
        if asset in self._streams:
            raise TelemetryQualityError(f'stream already open for {asset}')
        stream = open_stream(asset, self.policy, at)
        self._streams[asset] = stream
        return stream

    def get(self, asset: str) -> QualityStream:
        try:
            return self._streams[asset]
        except KeyError:
            raise TelemetryQualityError(f'no stream for asset {asset}') from None

    def ingest(self, sample: Sample, at: datetime) -> Arrival:
        return self.get(sample.asset).ingest(sample, at)

    def decision(self, asset: str, at: datetime) -> DecisionView:
        return decision_view(self.get(asset), at)

    def isolated_assets(self) -> tuple[str, ...]:
        return tuple(a for a, s in self._streams.items()
                     if s.state != HEALTHY)

    def revise_policy(self, policy: QualityPolicy, at: datetime
                      ) -> tuple[StateEvent | None, ...]:
        """Adopt a new policy version across every open stream."""
        self.policy = policy
        return tuple(s.apply_policy(policy, at) for s in self._streams.values())
