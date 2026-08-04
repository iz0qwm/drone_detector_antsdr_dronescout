"""
Managed proximity engine: background worker that evaluates all
drone-aircraft pairs and exposes results via thread-safe snapshot.
"""
import time
import threading
from datetime import datetime, timezone

from services.proximity.config import get_proximity_config, is_adsb_net_enabled
from services.proximity.calc import haversine_meters, is_valid_position, evaluate_pairs
from services.proximity.normalize import (
    normalize_adsbrx, normalize_adsbnet, merge_targets, NormalizedTarget,
)
from services.proximity.state import PairManager, STALE, NORMAL
from services.proximity.trend import TrendManager, TREND_LABELS
from services.logger import log


class SourceHealth:
    """Tracks health of a traffic source."""

    def __init__(self, name):
        self.name = name
        self.state = "DISABLED"
        self.last_successful = None
        self.error = None

    def set_available(self):
        self.state = "AVAILABLE"
        self.last_successful = time.time()
        self.error = None

    def set_offline(self, error=None):
        self.state = "OFFLINE"
        self.error = error

    def set_disabled(self):
        self.state = "DISABLED"
        self.error = None

    def to_dict(self):
        return {
            "state": self.state,
            "last_successful": self.last_successful,
            "error": self.error,
        }


class ADSBNetCache:
    """
    Shared ADSBNet acquisition cache.
    Fetches providers at a separate interval; proximity engine reads the cache.
    """

    def __init__(self, refresh_interval_ms=15000):
        self._refresh_interval_s = refresh_interval_ms / 1000.0
        self._snapshot = []  # list of aircraft dicts
        self._last_fetch = 0
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._started = False

    def _loop(self):
        while not self._stop_event.is_set():
            self._refresh()
            self._stop_event.wait(timeout=self._refresh_interval_s)

    def _refresh(self):
        """Fetch ADSBNet providers. Stores result in cache."""
        if not is_adsb_net_enabled():
            with self._lock:
                self._snapshot = []
            return

        try:
            from services.network import has_internet
            if not has_internet():
                return  # Keep last valid snapshot

            from services.air_network import get_network_aircraft
            # Use wide bounds to get all available aircraft
            bounds = {
                "minLat": -90, "maxLat": 90,
                "minLon": -180, "maxLon": 180,
            }
            result = get_network_aircraft(bounds, show_all=True)
            aircraft = result.get("aircraft", [])

            with self._lock:
                self._snapshot = aircraft
                self._last_fetch = time.time()

        except Exception as e:
            log("PROXIMITY", f"ADSBNet cache refresh error: {e}", level="WARNING")

    def get_snapshot(self):
        """Return last completed result immediately (never blocks on network)."""
        with self._lock:
            return list(self._snapshot)

    @property
    def last_fetch_time(self):
        return self._last_fetch


class ProximitySnapshot:
    """Immutable result of one proximity calculation cycle."""

    def __init__(self, enabled=True, source_health=None, drones_active=0,
                 targets_active=0, pairs=None, calculation_time_ms=0,
                 last_calculated=None):
        self.enabled = enabled
        self.source_health = source_health or {}
        self.drones_active = drones_active
        self.targets_active = targets_active
        self.pairs = pairs or []
        self.calculation_time_ms = calculation_time_ms
        self.last_calculated = last_calculated or time.time()

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "source_health": {k: v.to_dict() if hasattr(v, 'to_dict') else v
                              for k, v in self.source_health.items()},
            "drones_active": self.drones_active,
            "targets_active": self.targets_active,
            "pairs": self.pairs,
            "calculation_time_ms": self.calculation_time_ms,
            "last_calculated": self.last_calculated,
        }


class ProximityEngine:
    """
    Managed background worker that evaluates proximity pairs.
    Runs independently of browser connections.
    Exposes thread-safe snapshot for API and future Meshtastic.
    """

    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._started = False
        self._snapshot = ProximitySnapshot(enabled=False)
        self._snapshot_lock = threading.Lock()

        self._pair_manager = PairManager()
        self._trend_manager = TrendManager()
        self._adsb_net_cache = ADSBNetCache()

        # Source health tracking
        self._health_rx = SourceHealth("ADSBRx")
        self._health_net = SourceHealth("ADSBNet")
        self._health_rid = SourceHealth("RemoteID")

    def start(self):
        """Idempotent start. No-op if already running."""
        if self._started:
            return
        self._started = True
        self._stop_event.clear()

        config = get_proximity_config()
        refresh_ms = config.get("adsb_net_refresh_interval_ms", 15000)
        self._adsb_net_cache = ADSBNetCache(refresh_interval_ms=refresh_ms)
        self._adsb_net_cache.start()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log("PROXIMITY", "Engine started")

    def stop(self):
        """Clean shutdown."""
        self._stop_event.set()
        self._adsb_net_cache.stop()
        if self._thread:
            self._thread.join(timeout=5)
        self._started = False
        log("PROXIMITY", "Engine stopped")

    def _loop(self):
        """Main worker loop."""
        while not self._stop_event.is_set():
            try:
                self._calculate_cycle()
            except Exception as e:
                log("PROXIMITY", f"Cycle error: {e}", level="ERROR")
            config = get_proximity_config()
            interval_s = config.get("calculation_interval_ms", 5000) / 1000.0
            self._stop_event.wait(timeout=interval_s)

    def _calculate_cycle(self):
        """One proximity evaluation cycle."""
        start_time = time.time()
        config = get_proximity_config()

        if not config.get("enabled", True):
            self._update_snapshot(ProximitySnapshot(enabled=False))
            return

        # --- Read sources ---
        drones = self._read_drones()
        adsbrx_aircraft = self._read_adsbrx(config)
        adsbnet_aircraft = self._read_adsbnet(config)

        # --- Normalize and merge ---
        rx_targets = normalize_adsbrx(adsbrx_aircraft)
        net_targets = normalize_adsbnet(adsbnet_aircraft)

        tie_window = config.get("source_precedence_tie_window_s", 3.0)
        all_targets = merge_targets(rx_targets, net_targets, tie_window)

        # --- Filter fresh targets ---
        stale_ms = config.get("aircraft_source_stale_ms", 30000)
        fresh_targets = [t for t in all_targets if t.is_fresh(stale_ms)]

        # --- Evaluate pairs ---
        radius_m = config.get("evaluation_radius_m", 10000)
        thresholds = config.get("thresholds", {})
        drone_stale_ms = config.get("drone_stale_ms", 15000)
        grace_ms = config.get("pair_stale_grace_ms", 10000)
        deadband_m = config.get("movement_deadband_m", 50)
        trend_window_s = config.get("movement_history_window_s", 15)

        # Build target dicts for evaluate_pairs
        target_dicts = [
            {"track_id": t.track_id, "latitude": t.latitude, "longitude": t.longitude}
            for t in fresh_targets
        ]

        # Build drone dicts with freshness check
        now = time.time()
        fresh_drones = []
        for d in drones:
            last_seen = d.get("last_seen")
            if last_seen:
                try:
                    dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                    age_s = now - dt.timestamp()
                except Exception:
                    age_s = 999
            else:
                age_s = 999

            if age_s * 1000 > drone_stale_ms:
                continue
            fresh_drones.append(d)

        raw_pairs = evaluate_pairs(fresh_drones, target_dicts, radius_m)

        # --- Update state for each pair ---
        active_pair_ids = set()
        for rp in raw_pairs:
            pair = self._pair_manager.get_or_create(rp["drone_id"], rp["target_id"])
            prev_state = pair.update(rp["distance_m"], thresholds)
            active_pair_ids.add(pair.pair_id)

            # Update trend
            self._trend_manager.update(pair.pair_id, rp["distance_m"], now)

            # Log state changes
            if pair.state != prev_state and pair.state != NORMAL:
                log("PROXIMITY",
                    f"{pair.drone_id} -> {pair.target_id}: "
                    f"{prev_state} -> {pair.state} ({int(rp['distance_m'])}m)")

        # --- Mark absent pairs as stale ---
        self._pair_manager.remove_absent(active_pair_ids)

        # --- Cleanup expired stale pairs ---
        expired = self._pair_manager.cleanup_expired(grace_ms)
        for pid in expired:
            self._trend_manager.remove(pid)

        # --- Clean up trend for non-active pairs ---
        self._trend_manager.cleanup(active_pair_ids)

        # --- Build ranked result ---
        max_entries = config.get("max_panel_entries", 5)
        ranked = self._pair_manager.get_ranked_pairs(max_entries)

        # Build target index for lookup
        target_index = {t.track_id: t for t in all_targets}
        drone_index = {(d.get("serial") or "unknown"): d for d in drones}

        pairs_result = []
        for pair in ranked:
            trend, rate = self._trend_manager.get_trend(
                pair.pair_id, min_samples=3,
                min_window_s=10, deadband_m=deadband_m
            )
            target = target_index.get(pair.target_id)
            drone = drone_index.get(pair.drone_id)

            pairs_result.append({
                "pair_id": pair.pair_id,
                "drone_id": pair.drone_id,
                "drone_label": (drone.get("model") or drone.get("serial") or pair.drone_id)
                               if drone else pair.drone_id,
                "target_id": pair.target_id,
                "target_label": (target.callsign or target.icao or pair.target_id)
                                if target else pair.target_id,
                "distance_m": int(pair.distance_m) if pair.distance_m else None,
                "state": pair.state,
                "trend": TREND_LABELS.get(trend, "\u2014"),
                "drone_lat": drone.get("lat") if drone else None,
                "drone_lon": drone.get("lon") if drone else None,
                "target_lat": target.latitude if target else None,
                "target_lon": target.longitude if target else None,
                "target_altitude_m": target.altitude if target else None,
                "target_source": target.source_label if target else "?",
                "target_updated_ago_s": int(now - target.updated_at) if target else None,
                "drone_updated_ago_s": self._drone_age_s(drone, now),
            })

        calc_ms = (time.time() - start_time) * 1000

        if calc_ms > 50:
            log("PROXIMITY", f"Cycle took {calc_ms:.0f}ms", level="WARNING")

        snapshot = ProximitySnapshot(
            enabled=True,
            source_health={
                "adsb_rx": self._health_rx,
                "adsb_net": self._health_net,
                "remote_id": self._health_rid,
            },
            drones_active=len(fresh_drones),
            targets_active=len(fresh_targets),
            pairs=pairs_result,
            calculation_time_ms=round(calc_ms, 1),
            last_calculated=now,
        )
        self._update_snapshot(snapshot)

    def _read_drones(self):
        """Read current drones from DS110 service."""
        try:
            from services.ds110 import get_aircraft, running as ds110_running
            if not ds110_running:
                self._health_rid.set_disabled()
                return []
            aircraft = get_aircraft()
            if aircraft:
                self._health_rid.set_available()
            else:
                self._health_rid.set_available()  # empty is OK
            return aircraft
        except Exception as e:
            self._health_rid.set_offline(str(e))
            return []

    def _read_adsbrx(self, config):
        """Read local ADS-B from air_local service."""
        try:
            from services.air_local import get_local_aircraft
            # Wide bounds to get all local aircraft
            bounds = {"minLat": -90, "maxLat": 90, "minLon": -180, "maxLon": 180}
            result = get_local_aircraft(bounds, show_all=True)
            aircraft = result.get("aircraft", [])
            if result.get("success", True):
                self._health_rx.set_available()
            else:
                self._health_rx.set_offline()
            return aircraft
        except Exception as e:
            self._health_rx.set_offline(str(e))
            return []

    def _read_adsbnet(self, config):
        """Read ADSBNet from the shared cache (no network call here)."""
        if not is_adsb_net_enabled():
            self._health_net.set_disabled()
            return []

        aircraft = self._adsb_net_cache.get_snapshot()
        if self._adsb_net_cache.last_fetch_time > 0:
            self._health_net.set_available()
        else:
            self._health_net.set_offline("No fetch completed yet")
        return aircraft

    def _drone_age_s(self, drone, now):
        """Calculate drone age in seconds."""
        if not drone:
            return None
        last_seen = drone.get("last_seen")
        if not last_seen:
            return None
        try:
            dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            return int(now - dt.timestamp())
        except Exception:
            return None

    def _update_snapshot(self, snapshot):
        """Thread-safe snapshot update."""
        with self._snapshot_lock:
            self._snapshot = snapshot

    def get_snapshot(self):
        """
        Thread-safe read of latest proximity results.
        Used by API route and future Meshtastic consumers.
        Does NOT trigger calculation or network requests.
        """
        with self._snapshot_lock:
            return self._snapshot

    @property
    def is_running(self):
        return self._started


# Module-level singleton
engine = ProximityEngine()
