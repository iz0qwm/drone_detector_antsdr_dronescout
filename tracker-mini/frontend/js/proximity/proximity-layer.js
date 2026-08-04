/**
 * Proximity Layer — renders distance lines, labels, and aircraft indicators.
 * All objects live in the traffic-proximity pane.
 */
window.PROXIMITY_LAYER = window.PROXIMITY_LAYER || {};

(function () {
    let _line = null;
    let _label = null;
    let _rings = new Map(); // target_id -> ring object
    let _map = null;

    const STATE_STYLES = {
        WARNING: { color: "#FF3B30", dash: null, text: "WRN", weight: 3 },
        CAUTION: { color: "#FF9500", dash: "8 6", text: "CTN", weight: 2.5 },
        MONITOR: { color: "#007AFF", dash: "8 6", text: "MON", weight: 2 },
        STALE:   { color: "#8E8E93", dash: "4 4", text: "STL", weight: 2 },
    };

    PROXIMITY_LAYER.update = function (pairs, map) {
        _map = map;
        if (!map || !pairs || pairs.length === 0) {
            PROXIMITY_LAYER.clear();
            return;
        }

        // Highest priority pair gets the line
        const topPair = pairs[0];
        _updateLine(topPair, map);
        _updateRings(pairs, map);
    };

    PROXIMITY_LAYER.clear = function () {
        _removeLine();
        _removeAllRings();
    };

    function _updateLine(pair, map) {
        const style = STATE_STYLES[pair.state] || STATE_STYLES.MONITOR;

        if (!pair.drone_lat || !pair.drone_lon || !pair.target_lat || !pair.target_lon) {
            _removeLine();
            return;
        }

        const from = [pair.drone_lat, pair.drone_lon];
        const to = [pair.target_lat, pair.target_lon];

        // Remove old line
        _removeLine();

        // Create new line
        const dashArray = style.dash || undefined;
        _line = L.polyline([from, to], {
            color: style.color,
            weight: style.weight,
            dashArray: dashArray,
            pane: "traffic-proximity",
            interactive: false,
        }).addTo(map);

        // Distance label at midpoint
        const midLat = (from[0] + to[0]) / 2;
        const midLon = (from[1] + to[1]) / 2;
        const distText = _formatDistance(pair.distance_m);

        _label = L.marker([midLat, midLon], {
            icon: L.divIcon({
                className: "proximity-label",
                html: `<span class="prox-label-text prox-${pair.state.toLowerCase()}">${style.text} ${distText}</span>`,
                iconSize: [80, 20],
                iconAnchor: [40, 10],
            }),
            pane: "traffic-proximity",
            interactive: false,
        }).addTo(map);
    }

    function _updateRings(pairs, map) {
        const seenTargets = new Set();
        const targetBestState = new Map(); // target_id -> best state

        // Determine best (highest severity) state per target
        for (const pair of pairs) {
            if (pair.state === "NORMAL") continue;
            const existing = targetBestState.get(pair.target_id);
            if (!existing || _severity(pair.state) > _severity(existing.state)) {
                targetBestState.set(pair.target_id, pair);
            }
        }

        // Render up to 5 rings
        let count = 0;
        for (const [targetId, pair] of targetBestState) {
            if (count >= 5) break;
            if (!pair.target_lat || !pair.target_lon) continue;

            seenTargets.add(targetId);
            const style = STATE_STYLES[pair.state] || STATE_STYLES.MONITOR;

            let ring = _rings.get(targetId);
            if (ring) {
                ring.setLatLng([pair.target_lat, pair.target_lon]);
                ring.setStyle({
                    color: style.color,
                    dashArray: style.dash || undefined,
                });
            } else {
                ring = L.circleMarker([pair.target_lat, pair.target_lon], {
                    radius: 18,
                    color: style.color,
                    weight: 3,
                    fillOpacity: 0,
                    dashArray: style.dash || undefined,
                    pane: "traffic-proximity",
                    interactive: false,
                    className: pair.state === "WARNING" ? "prox-ring-pulse" : "",
                }).addTo(map);
                _rings.set(targetId, ring);
            }
            count++;
        }

        // Remove rings for targets no longer in proximity
        for (const [targetId, ring] of _rings) {
            if (!seenTargets.has(targetId)) {
                map.removeLayer(ring);
                _rings.delete(targetId);
            }
        }
    }

    function _removeLine() {
        if (_line && _map) {
            _map.removeLayer(_line);
            _line = null;
        }
        if (_label && _map) {
            _map.removeLayer(_label);
            _label = null;
        }
    }

    function _removeAllRings() {
        if (_map) {
            for (const [, ring] of _rings) {
                _map.removeLayer(ring);
            }
        }
        _rings.clear();
    }

    function _formatDistance(m) {
        if (m == null) return "?";
        if (m >= 1000) return (m / 1000).toFixed(1) + " km";
        return Math.round(m) + " m";
    }

    function _severity(state) {
        const map = { WARNING: 4, CAUTION: 3, MONITOR: 2, STALE: 1 };
        return map[state] || 0;
    }
})();
