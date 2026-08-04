"""
Tests for proximity state machine and hysteresis.
"""
import time
import pytest
from services.proximity.state import (
    classify_distance, ProximityPair, PairManager,
    NORMAL, MONITOR, CAUTION, WARNING, STALE,
)
from services.proximity.config import PROXIMITY_DEFAULTS

THRESHOLDS = PROXIMITY_DEFAULTS["thresholds"]


class TestClassifyDistance:
    def test_normal(self):
        assert classify_distance(5000, NORMAL, THRESHOLDS) == NORMAL

    def test_monitor_entry(self):
        assert classify_distance(2999, NORMAL, THRESHOLDS) == MONITOR

    def test_caution_entry(self):
        assert classify_distance(1499, NORMAL, THRESHOLDS) == CAUTION

    def test_warning_entry(self):
        assert classify_distance(499, NORMAL, THRESHOLDS) == WARNING

    def test_direct_escalation_to_warning(self):
        """Aircraft appears directly within WARNING range."""
        assert classify_distance(200, NORMAL, THRESHOLDS) == WARNING

    def test_hysteresis_monitor_no_exit(self):
        """At 3100m (between entry 3000 and exit 3300), MONITOR stays."""
        assert classify_distance(3100, MONITOR, THRESHOLDS) == MONITOR

    def test_hysteresis_monitor_exit(self):
        """At 3400m (above exit 3300), MONITOR exits to NORMAL."""
        assert classify_distance(3400, MONITOR, THRESHOLDS) == NORMAL

    def test_hysteresis_caution_no_exit(self):
        assert classify_distance(1600, CAUTION, THRESHOLDS) == CAUTION

    def test_hysteresis_caution_exit(self):
        assert classify_distance(1900, CAUTION, THRESHOLDS) == MONITOR

    def test_hysteresis_warning_no_exit(self):
        assert classify_distance(600, WARNING, THRESHOLDS) == WARNING

    def test_hysteresis_warning_exit(self):
        assert classify_distance(800, WARNING, THRESHOLDS) == CAUTION

    def test_oscillation_no_flicker(self):
        """20 cycles at boundary ±10m should produce max 1 state change."""
        state = NORMAL
        changes = 0
        for i in range(20):
            d = 3000 + (10 if i % 2 == 0 else -10)
            new_state = classify_distance(d, state, THRESHOLDS)
            if new_state != state:
                changes += 1
                state = new_state
        # Should enter MONITOR once and stay (hysteresis prevents exit at 3010)
        assert changes <= 1


class TestProximityPair:
    def test_initial_state(self):
        p = ProximityPair("D1", "AC1")
        assert p.state == NORMAL
        assert p.pair_id == "D1:AC1"

    def test_update_enters_monitor(self):
        p = ProximityPair("D1", "AC1")
        p.update(2500, THRESHOLDS)
        assert p.state == MONITOR

    def test_mark_stale(self):
        p = ProximityPair("D1", "AC1")
        p.update(1000, THRESHOLDS)
        assert p.state == CAUTION
        p.mark_stale()
        assert p.state == STALE

    def test_stale_grace(self):
        p = ProximityPair("D1", "AC1")
        p.mark_stale()
        assert not p.stale_grace_expired(10000)  # just marked
        p.stale_since = time.time() - 11  # simulate 11s ago
        assert p.stale_grace_expired(10000)


class TestPairManager:
    def test_get_or_create(self):
        pm = PairManager()
        p = pm.get_or_create("D1", "AC1")
        assert p.pair_id == "D1:AC1"
        # Same call returns same instance
        p2 = pm.get_or_create("D1", "AC1")
        assert p is p2

    def test_ranked_excludes_normal(self):
        pm = PairManager()
        p1 = pm.get_or_create("D1", "AC1")
        p1.update(2500, THRESHOLDS)  # MONITOR
        p2 = pm.get_or_create("D1", "AC2")
        p2.update(400, THRESHOLDS)  # WARNING
        p3 = pm.get_or_create("D1", "AC3")
        # p3 stays NORMAL (no update)

        ranked = pm.get_ranked_pairs()
        assert len(ranked) == 2
        assert ranked[0].state == WARNING
        assert ranked[1].state == MONITOR

    def test_cleanup_expired(self):
        pm = PairManager()
        p = pm.get_or_create("D1", "AC1")
        p.mark_stale()
        p.stale_since = time.time() - 20  # 20s ago
        removed = pm.cleanup_expired(10000)
        assert "D1:AC1" in removed
        assert "D1:AC1" not in pm.all_pairs

    def test_remove_absent_marks_stale(self):
        pm = PairManager()
        p = pm.get_or_create("D1", "AC1")
        p.update(2000, THRESHOLDS)  # MONITOR
        pm.remove_absent(set())  # no active pairs
        assert p.state == STALE

    def test_deterministic_ranking(self):
        """Same severity + distance: deterministic by drone then target id."""
        pm = PairManager()
        p1 = pm.get_or_create("D2", "AC1")
        p1.state = MONITOR
        p1.distance_m = 2000
        p2 = pm.get_or_create("D1", "AC1")
        p2.state = MONITOR
        p2.distance_m = 2000

        ranked = pm.get_ranked_pairs()
        assert ranked[0].drone_id == "D1"
        assert ranked[1].drone_id == "D2"
