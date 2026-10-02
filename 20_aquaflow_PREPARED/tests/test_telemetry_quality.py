import unittest
from datetime import datetime, timedelta, timezone

from aquaflow.telemetry.quality import (
    DEFAULT_POLICY,
    CONFIRMED,
    QUARANTINED,
    RECOVERED,
    QualityPolicy,
    QualityStream,
    Sample,
    allowed_actions,
    apply_policy,
    backfill,
    confirm_isolation,
    continuity_check,
    control_view,
    evaluate,
    history,
    ingest,
    is_available,
    raw_samples,
    recover,
    recovery_blockers,
)

T0 = datetime(2026, 10, 2, 9, 0, 0, tzinfo=timezone.utc)
MINUTE = timedelta(minutes=1)


def sample(seq, at_offset, *, value=None, received_offset=None, source='live'):
    return Sample(
        asset='PUMP-7',
        at=T0 + at_offset * MINUTE,
        value=value if value is not None else 100.0 + seq,
        sequence=seq,
        received_at=T0 + (received_offset if received_offset is not None else at_offset) * MINUTE,
        source=source,
    )


def state_changes(stream):
    return [e for e in history(stream) if e.action != 'ingest']


class OutOfOrderTests(unittest.TestCase):
    def test_out_of_order_packet_quarantines_and_downstream_sees_unavailable(self):
        stream = QualityStream('PUMP-7')
        for seq, offset in ((1, 0), (2, 1), (3, 2), (5, 3)):
            ingest(stream, sample(seq, offset))
        self.assertTrue(is_available(stream))

        # packet 4 is delayed on the wire and arrives after packet 5;
        # the measured timeline stays complete, only ordering is violated
        report = ingest(stream, sample(4, 4, received_offset=5))

        self.assertEqual(report.gap_count, 0)
        self.assertEqual(report.out_of_order_count, 1)
        self.assertAlmostEqual(report.trust, 0.85)
        self.assertLess(report.trust, DEFAULT_POLICY.trust_floor)
        self.assertEqual(stream.state, QUARANTINED)
        self.assertEqual(stream.out_of_order_sequences, (4,))

        view = control_view(stream)
        self.assertFalse(view.available)
        self.assertIsNone(view.value)          # no stale last-good value leaks downstream
        self.assertIsNone(view.at)
        self.assertIn(QUARANTINED, view.reason)

        with self.assertRaises(ValueError):    # reviewer clock is behind the last event
            confirm_isolation(stream, 'r1', T0 + 4 * MINUTE)
        confirm_isolation(stream, 'reviewer-1', T0 + 7 * MINUTE)
        self.assertEqual(stream.state, CONFIRMED)
        self.assertEqual(allowed_actions(stream), ['backfill', 'recover'])

        # continuity is fine, but disorder penalty is permanent under v1
        self.assertTrue(continuity_check(stream).ok)
        self.assertEqual(recovery_blockers(stream), ['trust'])
        with self.assertRaises(ValueError):
            recover(stream, 'reviewer-1', T0 + 8 * MINUTE)
        self.assertFalse(is_available(stream))

        quarantine_event = state_changes(stream)[0]
        self.assertEqual(quarantine_event.action, 'quarantine')
        self.assertEqual(quarantine_event.policy_version, 'v1')
        self.assertEqual(quarantine_event.from_state, 'healthy')
        self.assertEqual(quarantine_event.sequence, 4)
        confirm_event = state_changes(stream)[1]
        self.assertEqual(confirm_event.action, 'confirm')
        self.assertEqual(confirm_event.reviewer, 'reviewer-1')


class BackfillGapTests(unittest.TestCase):
    def test_backfill_still_missing_one_gap_then_full_restoration(self):
        # smaller penalties: with one gap left the gap itself must be the sole
        # recovery blocker (trust can already be back above the floor)
        policy = QualityPolicy(version='v1', expected_interval=MINUTE,
                               trust_floor=0.8, gap_penalty=0.15, disorder_penalty=0.15)
        stream = QualityStream('PUMP-7', policy=policy)
        ingest(stream, sample(1, 0))
        ingest(stream, sample(9, 8))   # gap 1@0 -> 9@8 (trust 0.85, still above floor)
        ingest(stream, sample(13, 12)) # second gap 9@8 -> 13@12 (trust 0.70, isolated)
        self.assertEqual(stream.state, QUARANTINED)
        self.assertEqual(evaluate(stream).gap_count, 2)
        confirm_isolation(stream, 'reviewer-1', T0 + 13 * MINUTE)

        # reviewer backfills packets 2..8: first gap closes, one gap remains
        for seq in range(2, 9):
            report = backfill(stream, sample(seq, seq - 1, source='backfill'),
                              'reviewer-1', T0 + (13 + seq) * MINUTE)
        self.assertEqual(report.gap_count, 1)
        self.assertAlmostEqual(report.trust, 0.85)

        continuity = continuity_check(stream)
        self.assertFalse(continuity.ok)
        self.assertEqual(len(continuity.remaining_gaps), 1)
        self.assertEqual(recovery_blockers(stream), ['gap'])
        with self.assertRaises(ValueError):
            recover(stream, 'reviewer-1', T0 + 30 * MINUTE)
        self.assertFalse(is_available(stream))   # control stays explicitly unavailable

        # completing the backfill closes the remaining time gap
        for seq in (10, 11, 12):
            backfill(stream, sample(seq, seq - 1, source='backfill'),
                     'reviewer-1', T0 + (30 + seq) * MINUTE)

        report = evaluate(stream)
        self.assertEqual(report.gap_count, 0)
        self.assertAlmostEqual(report.trust, 1.0)
        self.assertTrue(continuity_check(stream).ok)
        recover(stream, 'reviewer-1', T0 + 50 * MINUTE)

        self.assertEqual(stream.state, RECOVERED)
        view = control_view(stream)
        self.assertTrue(view.available)
        self.assertEqual(view.value, 113.0)
        self.assertEqual(view.at, T0 + 12 * MINUTE)

        backfill_events = [e for e in history(stream) if e.action == 'backfill']
        self.assertTrue(all(e.policy_version == 'v1' for e in backfill_events))
        self.assertEqual(backfill_events[0].reviewer, 'reviewer-1')


class PolicyThresholdTests(unittest.TestCase):
    def test_recovery_succeeds_after_policy_threshold_change(self):
        policy_v2 = QualityPolicy(
            version='v2',
            expected_interval=MINUTE,
            trust_floor=0.8,
            gap_penalty=0.2,
            disorder_penalty=0.15,
        )

        stream = QualityStream('PUMP-7')
        ingest(stream, sample(1, 0))
        ingest(stream, sample(2, 1))
        ingest(stream, sample(5, 4))     # gap 2@1 -> 5@4: trust 0.80, auto-quarantine
        self.assertEqual(stream.state, QUARANTINED)

        # while isolated, a late packet 4 arrives live and is out of order vs packet 5
        ingest(stream, sample(4, 5))
        self.assertEqual(stream.out_of_order_sequences, (4,))
        self.assertAlmostEqual(evaluate(stream).trust, 0.65)
        confirm_isolation(stream, 'reviewer-1', T0 + 6 * MINUTE)

        #补交数据 closes the time gap (continuity passes) but disorder penalty persists
        backfill(stream, sample(3, 2, source='backfill'), 'reviewer-1', T0 + 7 * MINUTE)
        backfill(stream, sample(6, 3, source='backfill'), 'reviewer-1', T0 + 8 * MINUTE)
        self.assertEqual(evaluate(stream).gap_count, 0)
        self.assertTrue(continuity_check(stream).ok)
        self.assertAlmostEqual(evaluate(stream).trust, 0.85)

        self.assertEqual(recovery_blockers(stream), ['trust'])
        with self.assertRaises(ValueError):
            recover(stream, 'reviewer-1', T0 + 9 * MINUTE)

        # re-applying the same version is rejected; threshold change needs a new version
        with self.assertRaises(ValueError):
            apply_policy(stream, DEFAULT_POLICY, 'reviewer-1', T0 + 10 * MINUTE)
        apply_policy(stream, policy_v2, 'reviewer-1', T0 + 10 * MINUTE)
        self.assertEqual(evaluate(stream).policy_version, 'v2')
        self.assertFalse(evaluate(stream).below_floor)
        self.assertEqual(recovery_blockers(stream), [])

        recover(stream, 'reviewer-1', T0 + 11 * MINUTE)
        self.assertEqual(stream.state, RECOVERED)
        self.assertTrue(is_available(stream))

        events = history(stream)
        self.assertEqual([e.policy_version for e in events if e.action == 'quarantine'], ['v1'])
        policy_event = next(e for e in events if e.action == 'policy')
        self.assertEqual(policy_event.policy_version, 'v2')
        self.assertIn('v1 -> v2', policy_event.detail)
        recover_event = next(e for e in events if e.action == 'recover')
        self.assertEqual(recover_event.policy_version, 'v2')
        self.assertEqual(recover_event.from_state, CONFIRMED)
        self.assertAlmostEqual(recover_event.trust, 0.85)

        # after recovery a fresh gap re-quarantines automatically under the new policy
        ingest(stream, sample(7, 10, received_offset=12))
        self.assertEqual(stream.state, QUARANTINED)
        self.assertEqual(state_changes(stream)[-1].policy_version, 'v2')


class ImmutabilityTests(unittest.TestCase):
    def test_raw_samples_are_immutable_and_duplicates_rejected(self):
        stream = QualityStream('PUMP-7')
        ingest(stream, sample(1, 0))
        snapshot = raw_samples(stream)
        self.assertEqual(len(snapshot), 1)

        # the reviewer holds a tuple snapshot; later ingestion cannot mutate it
        ingest(stream, sample(2, 1))
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(raw_samples(stream)), 2)

        with self.assertRaises(Exception):
            snapshot[0].value = 999.0       # frozen dataclass

        # same sequence with an altered value cannot overwrite the original
        ingest(stream, sample(5, 4))    # opens a gap and isolates the stream
        self.assertEqual(stream.state, QUARANTINED)
        with self.assertRaises(ValueError):
            ingest(stream, sample(1, 9, value=50.0))
        confirm_isolation(stream, 'r1', T0 + 10 * MINUTE)
        with self.assertRaises(ValueError):
            backfill(stream, sample(1, 9, value=50.0), 'r1', T0 + 11 * MINUTE)

        self.assertEqual(raw_samples(stream)[0].value, 101.0)

    def test_backfill_source_is_stamped_and_ordered_late_data_is_not_disorder(self):
        stream = QualityStream('PUMP-7')
        ingest(stream, sample(1, 0))
        ingest(stream, sample(5, 4))
        confirm_isolation(stream, 'r1', T0 + 6 * MINUTE)
        backfill(stream, sample(102, 1), 'r1', T0 + 7 * MINUTE)
        # backfill forces source='backfill' even if the caller marked it live
        backfill(stream, sample(9, 8, source='live'), 'r1', T0 + 8 * MINUTE)
        sources = {s.sequence: s.source for s in raw_samples(stream)}
        self.assertEqual(sources[102], 'backfill')
        self.assertEqual(sources[9], 'backfill')
        self.assertEqual(stream.out_of_order_sequences, ())

    def test_invalid_transitions_and_policies_are_rejected(self):
        stream = QualityStream('PUMP-7')
        ingest(stream, sample(1, 0))
        with self.assertRaises(ValueError):   # cannot recover a healthy stream
            recover(stream, 'r1', T0 + 1 * MINUTE)
        with self.assertRaises(ValueError):   # cannot confirm before quarantine
            confirm_isolation(stream, 'r1', T0 + 1 * MINUTE)

        with self.assertRaises(ValueError):
            QualityPolicy(version='bad', expected_interval=timedelta(0))
        with self.assertRaises(ValueError):
            QualityPolicy(version='bad', expected_interval=MINUTE, trust_floor=1.5)


if __name__ == '__main__':
    unittest.main()
