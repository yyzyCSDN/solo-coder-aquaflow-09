from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from .pipeline import Reading, gaps

HEALTHY = 'healthy'
QUARANTINED = 'quarantined'
CONFIRMED = 'confirmed'
RECOVERED = 'recovered'

AVAILABLE_STATES = (HEALTHY, RECOVERED)


@dataclass(frozen=True)
class QualityPolicy:
    version: str
    expected_interval: timedelta
    gap_tolerance: float = 0.25
    trust_floor: float = 0.9
    gap_penalty: float = 0.2
    disorder_penalty: float = 0.15

    def __post_init__(self):
        if not self.version:
            raise ValueError('policy version required')
        if self.expected_interval.total_seconds() <= 0:
            raise ValueError('expected_interval must be positive')
        if self.gap_tolerance < 0 or not 0.0 <= self.trust_floor <= 1.0:
            raise ValueError('invalid quality policy')
        if self.gap_penalty < 0 or self.disorder_penalty < 0:
            raise ValueError('penalties must be non-negative')


DEFAULT_POLICY = QualityPolicy(version='v1', expected_interval=timedelta(minutes=1))


@dataclass(frozen=True)
class Sample:
    asset: str
    at: datetime
    value: float
    sequence: int
    received_at: datetime
    quality: float = 1.0
    source: str = 'live'

    def reading(self) -> Reading:
        return Reading(self.asset, self.at, self.value, self.sequence, self.quality)


@dataclass(frozen=True)
class QualityEvent:
    at: datetime
    action: str
    policy_version: str
    from_state: str | None
    to_state: str
    trust: float
    detail: str = ''
    reviewer: str | None = None
    sequence: int | None = None


@dataclass(frozen=True)
class QualityReport:
    asset: str
    trust: float
    gap_count: int
    out_of_order_count: int
    missing: tuple[tuple[datetime, datetime], ...]
    trust_floor: float
    below_floor: bool
    policy_version: str


@dataclass(frozen=True)
class ContinuityReport:
    asset: str
    ok: bool
    remaining_gaps: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class ControlView:
    asset: str
    available: bool
    value: float | None
    at: datetime | None
    state: str
    reason: str | None


@dataclass
class QualityStream:
    asset: str
    policy: QualityPolicy = DEFAULT_POLICY
    state: str = HEALTHY
    trust: float = 1.0
    samples: list[Sample] = field(default_factory=list)
    events: list[QualityEvent] = field(default_factory=list)
    out_of_order_sequences: tuple[int, ...] = ()
    max_sequence: int | None = None
    last_event_at: datetime | None = None


def raw_samples(stream: QualityStream) -> tuple[Sample, ...]:
    return tuple(stream.samples)


def history(stream: QualityStream) -> tuple[QualityEvent, ...]:
    return tuple(stream.events)


def evaluate(stream: QualityStream, policy: QualityPolicy | None = None) -> QualityReport:
    policy = policy or stream.policy
    rows = [sample.reading() for sample in stream.samples]
    missing = tuple(gaps(rows, policy.expected_interval, policy.gap_tolerance).get(stream.asset, []))
    disorder = len(stream.out_of_order_sequences)
    trust = max(0.0, 1.0 - len(missing) * policy.gap_penalty - disorder * policy.disorder_penalty)
    return QualityReport(
        asset=stream.asset,
        trust=trust,
        gap_count=len(missing),
        out_of_order_count=disorder,
        missing=missing,
        trust_floor=policy.trust_floor,
        below_floor=trust < policy.trust_floor,
        policy_version=policy.version,
    )


def continuity_check(stream: QualityStream) -> ContinuityReport:
    report = evaluate(stream)
    return ContinuityReport(stream.asset, report.gap_count == 0, report.missing)


def recovery_blockers(stream: QualityStream, policy: QualityPolicy | None = None) -> list[str]:
    report = evaluate(stream, policy)
    blockers = []
    if report.gap_count:
        blockers.append('gap')
    if report.below_floor:
        blockers.append('trust')
    return blockers


def control_view(stream: QualityStream) -> ControlView:
    if stream.state in AVAILABLE_STATES:
        latest = max(stream.samples, key=lambda s: s.at, default=None)
        return ControlView(stream.asset, True, latest.value if latest else None,
                           latest.at if latest else None, stream.state, None)
    return ControlView(
        stream.asset, False, None, None, stream.state,
        f'telemetry {stream.state}: trust {stream.trust:.2f} below floor {stream.policy.trust_floor:.2f}',
    )


def is_available(stream: QualityStream) -> bool:
    return control_view(stream).available


def is_healthy(stream: QualityStream) -> bool:
    return stream.state == HEALTHY


def is_quarantined(stream: QualityStream) -> bool:
    return stream.state == QUARANTINED


def is_confirmed(stream: QualityStream) -> bool:
    return stream.state == CONFIRMED


def is_recovered(stream: QualityStream) -> bool:
    return stream.state == RECOVERED


def allowed_actions(stream: QualityStream) -> list[str]:
    if stream.state in (HEALTHY, RECOVERED):
        return ['quarantine']
    if stream.state == QUARANTINED:
        return ['backfill', 'confirm']
    return ['backfill', 'recover']


def _check_time(stream: QualityStream, at: datetime):
    if stream.last_event_at is not None and at < stream.last_event_at:
        raise ValueError('time reversal')


def _record(stream: QualityStream, event: QualityEvent, report: QualityReport):
    stream.events.append(event)
    stream.last_event_at = event.at
    stream.trust = report.trust


def ingest(stream: QualityStream, sample: Sample, at: datetime | None = None) -> QualityReport:
    sample = replace(sample, source='live')
    if sample.asset != stream.asset:
        raise ValueError('sample asset does not match stream')
    at = at or sample.received_at
    _check_time(stream, at)
    if any(existing.sequence == sample.sequence for existing in stream.samples):
        raise ValueError('duplicate sequence: raw samples are immutable')

    disordered = stream.max_sequence is not None and sample.sequence < stream.max_sequence
    stream.samples.append(sample)
    stream.max_sequence = sample.sequence if stream.max_sequence is None else max(stream.max_sequence, sample.sequence)
    if disordered:
        stream.out_of_order_sequences += (sample.sequence,)

    report = evaluate(stream)
    if stream.state in AVAILABLE_STATES and report.below_floor:
        _record(stream, QualityEvent(
            at=at, action='quarantine', policy_version=stream.policy.version,
            from_state=stream.state, to_state=QUARANTINED, trust=report.trust,
            detail=_reason(report), sequence=sample.sequence,
        ), report)
        stream.state = QUARANTINED
    else:
        _record(stream, QualityEvent(
            at=at, action='ingest', policy_version=stream.policy.version,
            from_state=stream.state, to_state=stream.state, trust=report.trust,
            detail=_reason(report), sequence=sample.sequence,
        ), report)
    return report


def quarantine(stream: QualityStream, at: datetime) -> QualityReport:
    _check_time(stream, at)
    if stream.state not in AVAILABLE_STATES:
        raise ValueError('invalid transition')
    report = evaluate(stream)
    _record(stream, QualityEvent(
        at=at, action='quarantine', policy_version=stream.policy.version,
        from_state=stream.state, to_state=QUARANTINED, trust=report.trust,
        detail='reviewer quarantine',
    ), report)
    stream.state = QUARANTINED
    return report


def confirm_isolation(stream: QualityStream, reviewer: str, at: datetime) -> QualityStream:
    _check_time(stream, at)
    if stream.state != QUARANTINED:
        raise ValueError('invalid transition')
    report = evaluate(stream)
    _record(stream, QualityEvent(
        at=at, action='confirm', policy_version=stream.policy.version,
        from_state=QUARANTINED, to_state=CONFIRMED, trust=report.trust,
        detail='quarantine acknowledged', reviewer=reviewer,
    ), report)
    stream.state = CONFIRMED
    return stream


def backfill(stream: QualityStream, sample: Sample, reviewer: str, at: datetime) -> QualityReport:
    _check_time(stream, at)
    if stream.state not in (QUARANTINED, CONFIRMED):
        raise ValueError('invalid transition: stream is not isolated')
    sample = replace(sample, source='backfill')
    if sample.asset != stream.asset:
        raise ValueError('sample asset does not match stream')
    if any(existing.sequence == sample.sequence for existing in stream.samples):
        raise ValueError('duplicate sequence: raw samples are immutable')

    stream.samples.append(sample)
    report = evaluate(stream)
    _record(stream, QualityEvent(
        at=at, action='backfill', policy_version=stream.policy.version,
        from_state=stream.state, to_state=stream.state, trust=report.trust,
        detail=_reason(report), reviewer=reviewer, sequence=sample.sequence,
    ), report)
    return report


def apply_policy(stream: QualityStream, policy: QualityPolicy, reviewer: str, at: datetime) -> QualityReport:
    _check_time(stream, at)
    if policy.version == stream.policy.version:
        raise ValueError('policy version unchanged')
    previous = stream.policy
    stream.policy = policy
    report = evaluate(stream)
    _record(stream, QualityEvent(
        at=at, action='policy', policy_version=policy.version,
        from_state=stream.state, to_state=stream.state, trust=report.trust,
        detail=f'policy {previous.version} -> {policy.version}', reviewer=reviewer,
    ), report)
    return report


def recover(stream: QualityStream, reviewer: str, at: datetime) -> QualityStream:
    _check_time(stream, at)
    if stream.state != CONFIRMED:
        raise ValueError('invalid transition')
    blockers = recovery_blockers(stream)
    if blockers:
        raise ValueError('recovery blocked: ' + ', '.join(blockers))
    report = evaluate(stream)
    _record(stream, QualityEvent(
        at=at, action='recover', policy_version=stream.policy.version,
        from_state=CONFIRMED, to_state=RECOVERED, trust=report.trust,
        detail='continuity check passed', reviewer=reviewer,
    ), report)
    stream.state = RECOVERED
    return stream


def _reason(report: QualityReport) -> str:
    return (f'gaps={report.gap_count} out_of_order={report.out_of_order_count} '
            f'trust={report.trust:.2f} floor={report.trust_floor:.2f}')
