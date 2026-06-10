window.DRONES = window.DRONES || {};

DRONES.markers = {};

DRONES.updateDroneLayer = function (
    map,
    aircraft
) {

    const seen = new Set();

    aircraft.forEach(drone => {

        if (!drone.lat || !drone.lon) {
            return;
        }

        const id =
            drone.serial ||
            crypto.randomUUID();

        seen.add(id);

        let marker =
            DRONES.markers[id];

        if (!marker) {

            const icon = L.icon({
                iconUrl:
                    "icons/drone.png",
                iconSize: [32,32],
                iconAnchor: [16,16]
            });

            marker = L.marker(
                [drone.lat, drone.lon],
                {
                    icon,
                    pane: "traffic-drone"
                }
            );

            marker.addTo(map);

            DRONES.markers[id] =
                marker;
        }

        marker.setLatLng([
            drone.lat,
            drone.lon
        ]);

        marker.bindPopup(`
            <b>${drone.model || "Drone"}</b><br>
            Vendor: ${drone.vendor || "-"}<br>
            Serial: ${drone.serial || "-"}<br>
            Source: ${drone.source || "-"}
        `);
    });

    Object.keys(DRONES.markers)
        .forEach(id => {

            if (seen.has(id)) {
                return;
            }

            map.removeLayer(
                DRONES.markers[id]
            );

            delete DRONES.markers[id];
        });
};



DRONES.clearDroneLayer =
function() {

    Object.values(
        DRONES.markers
    ).forEach(marker => {

        marker.remove();

    });

    DRONES.markers = {};
};