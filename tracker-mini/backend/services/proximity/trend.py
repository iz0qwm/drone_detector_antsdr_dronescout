"""
Movement trend analysis using distance history.
Speed and heading are NOT required — trend is derived from successive distances.
"""
import time
from collections import deque

APPROACHING = "APPROACHING"
DIVERGING = "DIVERGING"
STABLE = "STABLE"
UNKNOWN = "UNKNOWN"

# Display labels for UI
TREND_LABELS = {
    APPROACHING: "APR",
    DIVERGING: "DIV",
    STABLE: "STB",
    UNKNOWN: "\u2014",  # em-dash
}


class TrendEntry:
    __slots__ = ("timestamp", "distance")

    def __init__(self, timestamp, distance):
        self.timestamp = timestamp
        self.distance = distance


class TrendTracker:
    """Per-pair distance history and trend determination."""

    def __init__(self, max_entries=4):
        self._history = deque(maxlen=max_entries)

    def add_sample(self, distance_m, timestamp=None):
        """Add a distance sample. Timestamp in epoch seconds."""
        if timestamp is None:
            timestamp = time.time()
        self._history.append(TrendEntry(timestamp, distance_m))

    def get_trend(self, min_samples=3, min_window_s=10, deadband_m=50):
        """
        Determine movement trend from distance history.

        Returns (trend, rate_m_s):
            trend: APPROACHING | DIVERGING | STABLE | UNKNOWN
            rate_m_s: approximate rate of distance change (m/s), or None
        """
        if len(self._history) < min_samples:
            return UNKNOWN, None

        oldest = self._history[0]
        newest = self._history[-1]
        time_span = newest.timestamp - oldest.timestamp

        if time_span < min_window_s:
            return UNKNOWN, None

        # Check consistency: all deltas in the same direction?
        deltas = []
        for i in range(1, len(self._history)):
            d = self._history[i].distance - self._history[i - 1].distance
            deltas.append(d)

        net_change = newest.distance - oldest.distance

        # Within deadband = STABLE
        if abs(net_change) <= deadband_m:
            return STABLE, 0.0

        # Check consistency (all deltas same sign)
        all_decreasing = all(d <= 0 for d in deltas)
        all_increasing = all(d >= 0 for d in deltas)

        rate = net_change / time_span if time_span > 0 else 0.0

        if all_decreasing and net_change < -deadband_m:
            return APPROACHING, rate

        if all_increasing and net_change > deadband_m:
            return DIVERGING, rate

        # Inconsistent direction
        return UNKNOWN, None

    def reset(self):
        """Clear history (after stale, identity change, or implausible jump)."""
        self._history.clear()

    @property
    def sample_count(self):
        return len(self._history)


class TrendManager:
    """Manages trend trackers for all pairs."""

    def __init__(self, max_entries_per_pair=4):
        self._trackers = {}  # pair_id -> TrendTracker
        self._max_entries = max_entries_per_pair

    def update(self, pair_id, distance_m, timestamp=None):
        """Add a distance sample for the given pair."""
        if pair_id not in self._trackers:
            self._trackers[pair_id] = TrendTracker(self._max_entries)
        self._trackers[pair_id].add_sample(distance_m, timestamp)

    def get_trend(self, pair_id, min_samples=3, min_window_s=10, deadband_m=50):
        """Get current trend for a pair."""
        tracker = self._trackers.get(pair_id)
        if not tracker:
            return UNKNOWN, None
        return tracker.get_trend(min_samples, min_window_s, deadband_m)

    def reset(self, pair_id):
        """Reset history for a pair."""
        if pair_id in self._trackers:
            self._trackers[pair_id].reset()

    def remove(self, pair_id):
        """Remove tracker for a pair."""
        self._trackers.pop(pair_id, None)

    def cleanup(self, active_pair_ids):
        """Remove trackers for pairs no longer active."""
        to_remove = [pid for pid in self._trackers if pid not in active_pair_ids]
        for pid in to_remove:
            del self._trackers[pid]
