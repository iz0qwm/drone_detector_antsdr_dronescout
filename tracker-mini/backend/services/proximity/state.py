"""
Proximity state machine with hysteresis and stale lifecycle.
"""
import time
from collections import OrderedDict


# Proximity states
NORMAL = "NORMAL"
MONITOR = "MONITOR"
CAUTION = "CAUTION"
WARNING = "WARNING"
STALE = "STALE"
UNKNOWN = "UNKNOWN"

# Severity ranking (higher = more severe)
SEVERITY = {
    UNKNOWN: 0,
    NORMAL: 1,
    MONITOR: 2,
    CAUTION: 3,
    WARNING: 4,
    STALE: 0,
}


def classify_distance(distance_m, current_state, thresholds):
    """
    Classify a distance into a proximity state using hysteresis.
    Returns the new state based on entry/exit thresholds.
    """
    entry_w = thresholds.get("warning_entry_m", 500)
    exit_w = thresholds.get("warning_exit_m", 700)
    entry_c = thresholds.get("caution_entry_m", 1500)
    exit_c = thresholds.get("caution_exit_m", 1800)
    entry_m = thresholds.get("monitor_entry_m", 3000)
    exit_m = thresholds.get("monitor_exit_m", 3300)

    # Direct escalation: check from most severe to least
    if distance_m < entry_w:
        return WARNING

    if distance_m < entry_c:
        # Can enter CAUTION, or stay in WARNING if de-escalating
        if current_state == WARNING:
            return WARNING if distance_m < exit_w else CAUTION
        return CAUTION

    if distance_m < entry_m:
        # Can enter MONITOR, or stay in CAUTION/WARNING if de-escalating
        if current_state == WARNING:
            return WARNING if distance_m < exit_w else CAUTION
        if current_state == CAUTION:
            return CAUTION if distance_m < exit_c else MONITOR
        return MONITOR

    # distance >= entry_m
    if current_state == WARNING:
        if distance_m < exit_w:
            return WARNING
        elif distance_m < exit_c:
            return CAUTION
        elif distance_m < exit_m:
            return MONITOR
        else:
            return NORMAL
    if current_state == CAUTION:
        if distance_m < exit_c:
            return CAUTION
        elif distance_m < exit_m:
            return MONITOR
        else:
            return NORMAL
    if current_state == MONITOR:
        return MONITOR if distance_m < exit_m else NORMAL

    return NORMAL


class ProximityPair:
    """Tracks the state of one drone-aircraft proximity pair."""

    def __init__(self, drone_id, target_id):
        self.pair_id = f"{drone_id}:{target_id}"
        self.drone_id = drone_id
        self.target_id = target_id
        self.state = NORMAL
        self.distance_m = None
        self.entered_at = time.time()
        self.stale_since = None

    def update(self, distance_m, thresholds):
        """Update state based on new distance. Returns previous state."""
        prev = self.state
        if self.state == STALE:
            return prev  # stale pairs don't transition on distance

        new_state = classify_distance(distance_m, self.state, thresholds)
        self.distance_m = distance_m

        if new_state != self.state:
            self.state = new_state
            self.entered_at = time.time()

        return prev

    def mark_stale(self):
        """Transition to STALE state."""
        if self.state != STALE:
            self.state = STALE
            self.stale_since = time.time()

    def stale_grace_expired(self, grace_ms):
        """Check if stale grace period has expired."""
        if self.state != STALE or self.stale_since is None:
            return False
        return (time.time() - self.stale_since) * 1000 >= grace_ms


class PairManager:
    """Manages all proximity pairs with lifecycle."""

    def __init__(self):
        self._pairs = OrderedDict()  # pair_id -> ProximityPair

    def get_or_create(self, drone_id, target_id):
        """Get existing pair or create new one."""
        pair_id = f"{drone_id}:{target_id}"
        if pair_id not in self._pairs:
            self._pairs[pair_id] = ProximityPair(drone_id, target_id)
        return self._pairs[pair_id]

    def mark_stale(self, pair_id):
        """Mark a pair as stale."""
        if pair_id in self._pairs:
            self._pairs[pair_id].mark_stale()

    def cleanup_expired(self, grace_ms):
        """Remove pairs whose stale grace has expired."""
        expired = [
            pid for pid, p in self._pairs.items()
            if p.stale_grace_expired(grace_ms)
        ]
        for pid in expired:
            del self._pairs[pid]
        return expired

    def remove_absent(self, active_pair_ids):
        """Mark pairs as stale if they're no longer in active evaluation."""
        for pid, pair in self._pairs.items():
            if pid not in active_pair_ids and pair.state != STALE:
                pair.mark_stale()

    def get_ranked_pairs(self, max_entries=None):
        """
        Return pairs ranked by severity (desc) then distance (asc).
        Deterministic tie-break: drone_id alpha, then target_id alpha.
        Excludes NORMAL and UNKNOWN pairs.
        """
        displayable = [
            p for p in self._pairs.values()
            if p.state not in (NORMAL, UNKNOWN)
        ]

        displayable.sort(key=lambda p: (
            -SEVERITY.get(p.state, 0),
            p.distance_m if p.distance_m is not None else float("inf"),
            p.drone_id,
            p.target_id,
        ))

        if max_entries:
            return displayable[:max_entries]
        return displayable

    @property
    def all_pairs(self):
        return dict(self._pairs)
