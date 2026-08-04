"""
Tests for movement trend analysis.
"""
import time
import pytest
from services.proximity.trend import (
    TrendTracker, TrendManager,
    APPROACHING, DIVERGING, STABLE, UNKNOWN,
)


class TestTrendTracker:
    def test_insufficient_samples(self):
        t = TrendTracker()
        t.add_sample(1000, timestamp=0)
        t.add_sample(900, timestamp=5)
        trend, rate = t.get_trend(min_samples=3, min_window_s=10)
        assert trend == UNKNOWN

    def test_insufficient_window(self):
        t = TrendTracker()
        now = time.time()
        t.add_sample(1000, timestamp=now)
        t.add_sample(900, timestamp=now + 3)
        t.add_sample(800, timestamp=now + 6)
        trend, rate = t.get_trend(min_samples=3, min_window_s=10)
        assert trend == UNKNOWN

    def test_approaching(self):
        t = TrendTracker()
        t.add_sample(3000, timestamp=0)
        t.add_sample(2500, timestamp=5)
        t.add_sample(2000, timestamp=10)
        t.add_sample(1500, timestamp=15)
        trend, rate = t.get_trend(min_samples=3, min_window_s=10, deadband_m=50)
        assert trend == APPROACHING
        assert rate < 0  # distance decreasing

    def test_diverging(self):
        t = TrendTracker()
        t.add_sample(1000, timestamp=0)
        t.add_sample(1500, timestamp=5)
        t.add_sample(2000, timestamp=10)
        t.add_sample(2500, timestamp=15)
        trend, rate = t.get_trend(min_samples=3, min_window_s=10, deadband_m=50)
        assert trend == DIVERGING
        assert rate > 0

    def test_stable_within_deadband(self):
        t = TrendTracker()
        t.add_sample(2000, timestamp=0)
        t.add_sample(2010, timestamp=5)
        t.add_sample(1990, timestamp=10)
        t.add_sample(2005, timestamp=15)
        trend, rate = t.get_trend(min_samples=3, min_window_s=10, deadband_m=50)
        assert trend == STABLE

    def test_mixed_direction_unknown(self):
        t = TrendTracker()
        t.add_sample(2000, timestamp=0)
        t.add_sample(1500, timestamp=5)
        t.add_sample(2500, timestamp=10)
        t.add_sample(1800, timestamp=15)
        trend, rate = t.get_trend(min_samples=3, min_window_s=10, deadband_m=50)
        assert trend == UNKNOWN

    def test_reset_clears_history(self):
        t = TrendTracker()
        t.add_sample(1000, timestamp=0)
        t.add_sample(900, timestamp=5)
        t.add_sample(800, timestamp=10)
        t.reset()
        assert t.sample_count == 0

    def test_buffer_limited(self):
        t = TrendTracker(max_entries=4)
        for i in range(10):
            t.add_sample(1000 - i * 100, timestamp=i * 5)
        assert t.sample_count == 4

    def test_speed_heading_not_required(self):
        """Movement trend only needs positions and timestamps, not speed/heading."""
        t = TrendTracker()
        # Simulating samples where only distance is known (no speed/heading needed)
        t.add_sample(5000, timestamp=0)
        t.add_sample(4000, timestamp=5)
        t.add_sample(3000, timestamp=10)
        t.add_sample(2000, timestamp=15)
        trend, rate = t.get_trend(min_samples=3, min_window_s=10, deadband_m=50)
        assert trend == APPROACHING


class TestTrendManager:
    def test_update_and_get(self):
        tm = TrendManager()
        for i in range(4):
            tm.update("D1:AC1", 3000 - i * 500, timestamp=i * 5)
        trend, rate = tm.get_trend("D1:AC1", min_samples=3, min_window_s=10)
        assert trend == APPROACHING

    def test_unknown_for_missing_pair(self):
        tm = TrendManager()
        trend, rate = tm.get_trend("nonexistent")
        assert trend == UNKNOWN

    def test_reset(self):
        tm = TrendManager()
        tm.update("D1:AC1", 1000, timestamp=0)
        tm.reset("D1:AC1")
        trend, rate = tm.get_trend("D1:AC1")
        assert trend == UNKNOWN

    def test_cleanup_removes_inactive(self):
        tm = TrendManager()
        tm.update("D1:AC1", 1000)
        tm.update("D1:AC2", 2000)
        tm.cleanup({"D1:AC1"})
        assert tm.get_trend("D1:AC2") == (UNKNOWN, None)

    def test_remove(self):
        tm = TrendManager()
        tm.update("D1:AC1", 1000)
        tm.remove("D1:AC1")
        assert tm.get_trend("D1:AC1") == (UNKNOWN, None)
