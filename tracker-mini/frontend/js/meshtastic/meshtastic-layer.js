window.MESHTASTIC = window.MESHTASTIC || {};

MESHTASTIC.operatorMarkers = {};

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


MESHTASTIC.updateOperatorsLayer = function(operators) {

    if (!window.airNodeMap) {
        return;
    }

    const activeIds =
        new Set();

    operators.forEach(op => {

        if (
            op.lat == null ||
            op.lon == null
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
            Last Seen: ${op.lastSeen ?? "-"}<br>
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