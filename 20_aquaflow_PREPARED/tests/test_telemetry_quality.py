import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from aquaflow.telemetry import (
    Sample,
    QualityPolicy,
    open_stream,
    decision_view,
    require_value,
    TelemetryQualityRegistry,
    HEALTHY,
    QUARANTINED,
    CONFIRMED,
    ImmutableSampleError,
    InvalidTransitionError,
    ContinuityError,
    TelemetryQualityError,
)

UTC = timezone.utc
T0 = datetime(2026, 10, 2, 9, 0, 0, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def policy(version='v1', threshold=0.9):
    return QualityPolicy(
        version=version,
        expected_interval=MINUTE,
        gap_tolerance=0.0,
        min_credibility=threshold,
        # A single unresolved gap quarantines; once every slot is补交 the
        # penalty disappears and continuity alone gates recovery.
        gap_penalty=0.2,
        out_of_order_penalty=0.2,
        max_staleness=5 * MINUTE,
    )


def sample(asset, minute, sequence, value=1.0):
    return Sample(asset, T0 + minute * MINUTE, float(value), sequence)


class OutOfOrderTests(unittest.TestCase):
    def test_out_of_order_live_sample_quarantines_and_blocks_downstream(self):
        stream = open_stream('A', policy(), T0)
        stream.ingest(sample('A', 0, 1, 10), T0)
        stream.ingest(sample('A', 1, 3, 11), T0 + MINUTE)
        self.assertEqual(stream.state, HEALTHY)

        # seq 2 arrives after seq 3: the timeline is continuous but the
        # arrival is out of order.
        arrival = stream.ingest(sample('A', 2, 2, 12), T0 + 2 * MINUTE)
        self.assertTrue(arrival.out_of_order)
        self.assertEqual(stream.state, QUARANTINED)
        self.assertAlmostEqual(stream.credibility(), 0.8)
        self.assertEqual(len(stream.out_of_order_arrivals()), 1)

        # The expired raw value is still in the immutable history...
        self.assertIsNotNone(stream.latest_sample())
        # ...but downstream control decisions must see explicit unavailable.
        view = decision_view(stream, T0 + 2 * MINUTE)
        self.assertFalse(view.available)
        self.assertIsNone(view.value)
        self.assertEqual(view.reason, f'stream {QUARANTINED}')
        with self.assertRaises(TelemetryQualityError):
            require_value(view)

        # Reviewer confirms the quarantine; event records the policy version.
        event = stream.confirm_quarantine('rev-1', T0 + 3 * MINUTE)
        self.assertEqual(stream.state, CONFIRMED)
        self.assertEqual(event.from_state, QUARANTINED)
        self.assertEqual(event.to_state, CONFIRMED)
        self.assertEqual(event.policy_version, 'v1')
        self.assertEqual(event.reviewer, 'rev-1')


class BackfillGapTests(unittest.TestCase):
    def test_backfill_still_missing_one_gap_keeps_stream_isolated(self):
        stream = open_stream('B', policy(), T0)
        # seq5 lands 4 minutes after seq1 (slot seq N @ minute N-1).
        stream.ingest(sample('B', 0, 1, 10), T0)
        stream.ingest(sample('B', 4, 5, 50), T0 + 4 * MINUTE)
        self.assertEqual(stream.state, QUARANTINED)
        self.assertEqual(len(stream.gaps()), 1)

        stream.confirm_quarantine('rev-1', T0 + 6 * MINUTE)

        # Reviewer补交了 seq2/seq3，但 seq4 仍缺：02:00 -> 04:00 之间还有一个缺口。
        accepted = stream.backfill([
            sample('B', 1, 2, 20),
            sample('B', 2, 3, 30),
        ], 'rev-1', T0 + 7 * MINUTE)
        self.assertTrue(all(a.accepted for a in accepted))
        self.assertEqual(len(stream.gaps()), 1)
        self.assertFalse(stream.report().passed)
        self.assertEqual(stream.report().missing_slots, 1)
        # Late backfills repair history; they are not counted as OOO arrivals.
        self.assertEqual(len(stream.out_of_order_arrivals()), 0)

        # 连续性检查不通过：恢复被拒绝，数据流保持隔离。
        with self.assertRaises(ContinuityError):
            stream.restore('rev-1', T0 + 8 * MINUTE)
        self.assertEqual(stream.state, CONFIRMED)
        view = decision_view(stream, T0 + 8 * MINUTE)
        self.assertFalse(view.available)
        self.assertIsNone(view.value)
        self.assertFalse(stream.can_restore())

        # Backfilling the final slot (seq4 @ minute 3) closes the gap.
        stream.backfill([sample('B', 3, 4, 40)], 'rev-1', T0 + 9 * MINUTE)
        self.assertTrue(stream.report().passed)
        self.assertEqual(stream.report().missing_slots, 0)


class PolicyThresholdRecoveryTests(unittest.TestCase):
    def test_threshold_change_allows_recovery_after_continuity_passes(self):
        stream = open_stream('C', policy(threshold=0.9), T0)
        stream.ingest(sample('C', 0, 1, 10), T0)
        stream.ingest(sample('C', 1, 3, 11), T0 + MINUTE)
        stream.ingest(sample('C', 2, 2, 12), T0 + 2 * MINUTE)
        self.assertEqual(stream.state, QUARANTINED)
        stream.confirm_quarantine('rev-1', T0 + 3 * MINUTE)

        # Timeline is continuous (no gaps); only the OOO penalty remains.
        self.assertTrue(stream.report().passed)
        self.assertAlmostEqual(stream.credibility(), 0.8)

        # Under v1 the threshold still blocks recovery.
        with self.assertRaises(ContinuityError):
            stream.restore('rev-1', T0 + 4 * MINUTE)

        # Policy version changes: the revision itself is recorded.
        change = stream.apply_policy(policy(version='v2', threshold=0.7),
                                     T0 + 5 * MINUTE)
        self.assertEqual(change.policy_version, 'v2')
        self.assertEqual(change.action, 'policy_change')

        # Continuity check passes and 0.8 clears the relaxed v2 threshold.
        report, event = stream.restore('rev-1', T0 + 6 * MINUTE)
        self.assertTrue(report.passed)
        self.assertEqual(stream.state, HEALTHY)
        self.assertEqual(event.policy_version, 'v2')
        self.assertEqual(event.action, 'restore')

        view = decision_view(stream, T0 + 6 * MINUTE)
        self.assertTrue(view.available)
        self.assertEqual(view.value, 12.0)
        self.assertEqual(view.policy_version, 'v2')
        self.assertEqual(require_value(view), 12.0)

        # Every state change carries the policy version then in force.
        versions = [(e.action, e.policy_version) for e in stream.events]
        self.assertIn(('auto_quarantine', 'v1'), versions)
        self.assertIn(('confirm_quarantine', 'v1'), versions)
        self.assertIn(('policy_change', 'v2'), versions)
        self.assertIn(('restore', 'v2'), versions)

    def test_stricter_threshold_quarantines_healthy_stream_under_new_version(self):
        # A healthy stream carrying one out-of-order arrival (credibility .8)
        # passes v1 (threshold .8) but fails a stricter v2.
        v1 = policy(version='v1', threshold=0.8)
        stream = open_stream('H', v1, T0)
        stream.ingest(sample('H', 0, 1, 10), T0)
        stream.ingest(sample('H', 1, 3, 11), T0 + MINUTE)
        stream.ingest(sample('H', 2, 2, 12), T0 + 2 * MINUTE)
        self.assertEqual(stream.state, HEALTHY)
        self.assertAlmostEqual(stream.credibility(), 0.8)

        stream.apply_policy(policy(version='v2', threshold=0.9),
                            T0 + 3 * MINUTE)
        self.assertEqual(stream.state, QUARANTINED)
        quarantine = stream.events[-1]
        self.assertEqual(quarantine.action, 'auto_quarantine')
        self.assertEqual(quarantine.policy_version, 'v2')
        self.assertFalse(decision_view(stream, T0 + 3 * MINUTE).available)


class ImmutabilityAndGuardsTests(unittest.TestCase):
    def test_raw_samples_are_immutable_and_conflicts_rejected(self):
        stream = open_stream('D', policy(), T0)
        stream.ingest(sample('D', 0, 1, 10), T0)
        original = stream.raw_samples()[0]
        with self.assertRaises(FrozenInstanceError):
            original.value = 99.0  # type: ignore[misc]

        # A conflicting resend for the same sequence cannot overwrite history.
        with self.assertRaises(ImmutableSampleError):
            stream.ingest(sample('D', 0, 1, 99), T0 + MINUTE)
        self.assertEqual(stream.raw_samples()[0].value, 10.0)

        # An identical resend is a duplicate delivery, not a new sample.
        duplicate = stream.ingest(sample('D', 0, 1, 10), T0 + MINUTE)
        self.assertFalse(duplicate.accepted)
        self.assertEqual(len(stream.raw_samples()), 1)

    def test_stale_healthy_stream_is_unavailable(self):
        stream = open_stream('E', policy(), T0)
        stream.ingest(sample('E', 0, 1, 10), T0)
        view = decision_view(stream, T0 + 10 * MINUTE)
        self.assertFalse(view.available)
        self.assertEqual(view.reason, 'stale')

    def test_illegal_transitions_are_rejected(self):
        stream = open_stream('F', policy(), T0)
        stream.ingest(sample('F', 0, 1, 10), T0)
        with self.assertRaises(InvalidTransitionError):
            stream.confirm_quarantine('rev-1', T0 + MINUTE)
        with self.assertRaises(InvalidTransitionError):
            stream.backfill([sample('F', 1, 2, 20)], 'rev-1', T0 + MINUTE)
        with self.assertRaises(InvalidTransitionError):
            stream.restore('rev-1', T0 + MINUTE)

    def test_registry_tracks_isolated_assets(self):
        registry = TelemetryQualityRegistry(policy())
        stream = registry.open('G', T0)
        registry.ingest(sample('G', 0, 1, 10), T0)
        registry.ingest(sample('G', 1, 3, 11), T0 + MINUTE)
        registry.ingest(sample('G', 2, 2, 12), T0 + 2 * MINUTE)
        self.assertEqual(registry.isolated_assets(), ('G',))
        self.assertFalse(registry.decision('G', T0 + 2 * MINUTE).available)
        self.assertIs(stream, registry.get('G'))


if __name__ == '__main__':
    unittest.main()
