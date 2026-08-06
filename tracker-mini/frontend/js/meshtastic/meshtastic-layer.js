window.MESHTASTIC = window.MESHTASTIC || {};

MESHTASTIC.operatorMarkers = {};
MESHTASTIC.OPERATOR_STALE_MS = 600000;
MESHTASTIC.OPERATOR_RETENTION_MS = 1800000;
MESHTASTIC.OPERATOR_MIN_OPACITY = 0.25;

MESHTASTIC.operatorIcon = L.icon({
    iconUrl: "icons/operator.png",
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16]
});


MESHTASTIC.clearOperators = function() {

    Object.values(
        MESHTASTIC.operatorMarkers
    ).forEach(marker => {

        window.airNodeMap.removeLayer(
            marker
        );

    });

    MESHTASTIC.operatorMarkers = {};

};


function getOperatorAgeMs(op) {

    if (Number.isFinite(op.age_ms)) {
        return Math.max(
            0,
            op.age_ms
        );
    }

    const timestamp =
        Date.parse(
            op.last_seen ||
            op.lastSeen
        );

    if (!Number.isFinite(timestamp)) {
        return 0;
    }

    return Math.max(
        0,
        Date.now() - timestamp
    );

}


function operatorStaleMs(op, freshness) {

    if (Number.isFinite(op.stale_ms)) {
        return op.stale_ms;
    }

    if (Number.isFinite(freshness.stale_ms)) {
        return freshness.stale_ms;
    }

    return MESHTASTIC.OPERATOR_STALE_MS;

}


function operatorRetentionMs(op, freshness) {

    if (Number.isFinite(op.retention_ms)) {
        return op.retention_ms;
    }

    if (Number.isFinite(freshness.retention_ms)) {
        return freshness.retention_ms;
    }

    return MESHTASTIC.OPERATOR_RETENTION_MS;

}


function isOperatorExpired(op, freshness) {

    return getOperatorAgeMs(op) >
        operatorRetentionMs(
            op,
            freshness
        );

}


function computeOperatorOpacity(op, freshness) {

    const ageMs =
        getOperatorAgeMs(op);
    const staleMs =
        operatorStaleMs(
            op,
            freshness
        );
    const retentionMs =
        operatorRetentionMs(
            op,
            freshness
        );

    if (ageMs <= staleMs) {
        return 1;
    }

    if (ageMs >= retentionMs) {
        return 0;
    }

    const fadeWindowMs =
        Math.max(
            1,
            retentionMs - staleMs
        );
    const remainingMs =
        retentionMs - ageMs;
    const ratio =
        remainingMs / fadeWindowMs;

    return MESHTASTIC.OPERATOR_MIN_OPACITY +
        ((1 - MESHTASTIC.OPERATOR_MIN_OPACITY) * ratio);

}


function applyOperatorMarkerStyle(marker, op, freshness) {

    const el =
        marker.getElement();

    if (!el) {
        return;
    }

    const opacity =
        computeOperatorOpacity(
            op,
            freshness
        );
    const stale =
        op.stale === true ||
        getOperatorAgeMs(op) >
            operatorStaleMs(
                op,
                freshness
            );

    el.style.opacity =
        opacity.toString();
    el.style.filter =
        stale ? "grayscale(1)" : "none";
    el.style.transition =
        "opacity 0.5s linear, filter 0.5s linear";

}


MESHTASTIC.updateOperatorsLayer = function(operators, freshness = {}) {

    if (!window.airNodeMap) {
        return;
    }

    const activeIds =
        new Set();

    operators.forEach(op => {

        if (
            op.lat == null ||
            op.lon == null ||
            isOperatorExpired(
                op,
                freshness
            )
        ) {
            return;
        }

        const markerId =
            op.nodeId || op.id;

        activeIds.add(
            markerId
        );

        const popup = `
            <b>${op.longName || op.name || "Operator"}</b><br>
            Short: ${op.shortName ?? "-"}<br>
            Node: ${op.nodeId || op.id || "-"}<br>
            Battery: ${op.battery ?? "-"}<br>
            SNR: ${op.snr ?? "-"} dB<br>
            Last Seen: ${op.last_seen || op.lastSeen || "-"}<br>
            Position:<br>
            ${op.lat.toFixed(6)}, ${op.lon.toFixed(6)}
        `;

        if (
            MESHTASTIC.operatorMarkers[markerId]
        ) {

            MESHTASTIC.operatorMarkers[markerId]
                .setLatLng([
                    op.lat,
                    op.lon
                ]);

            MESHTASTIC.operatorMarkers[markerId]
                .bindPopup(popup);

            applyOperatorMarkerStyle(
                MESHTASTIC.operatorMarkers[markerId],
                op,
                freshness
            );

            return;
        }

        const marker =
            L.marker(
                [
                    op.lat,
                    op.lon
                ],
                {
                    icon:
                        MESHTASTIC.operatorIcon,
                    pane:
                        "traffic-drone"
                }
            ).addTo(
                window.airNodeMap
            );

        marker.bindPopup(
            popup
        );

        applyOperatorMarkerStyle(
            marker,
            op,
            freshness
        );

        MESHTASTIC.operatorMarkers[markerId] =
            marker;

    });

    Object.keys(
        MESHTASTIC.operatorMarkers
    ).forEach(markerId => {

        if (
            !activeIds.has(markerId)
        ) {

            window.airNodeMap.removeLayer(
                MESHTASTIC.operatorMarkers[markerId]
            );

            delete MESHTASTIC.operatorMarkers[markerId];

        }

    });

};
