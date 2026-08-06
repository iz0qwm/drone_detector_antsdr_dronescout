window.DRONES = window.DRONES || {};

DRONES.markers = {};
const DRONE_STALE_MS = 45000;
const DRONE_RETENTION_MS = 180000;
const DRONE_MIN_OPACITY = 0.25;


function isValidDronePosition(drone) {
    const lat = Number(drone.lat);
    const lon = Number(drone.lon);

    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        return false;
    }

    if (lat === 0 && lon === 0) {
        return false;
    }

    return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
}


function getDroneAgeMs(drone) {
    if (Number.isFinite(drone.age_ms)) {
        return Math.max(0, drone.age_ms);
    }

    const timestamp = Date.parse(drone.last_seen);

    if (!Number.isFinite(timestamp)) {
        return 0;
    }

    return Math.max(0, Date.now() - timestamp);
}


function isDroneExpired(drone) {
    const retentionMs =
        Number.isFinite(drone.retention_ms)
            ? drone.retention_ms
            : DRONE_RETENTION_MS;

    return getDroneAgeMs(drone) > retentionMs;
}


function computeDroneOpacity(drone) {
    const ageMs = getDroneAgeMs(drone);
    const staleMs =
        Number.isFinite(drone.stale_ms)
            ? drone.stale_ms
            : DRONE_STALE_MS;
    const retentionMs =
        Number.isFinite(drone.retention_ms)
            ? drone.retention_ms
            : DRONE_RETENTION_MS;

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

    return DRONE_MIN_OPACITY +
        ((1 - DRONE_MIN_OPACITY) * ratio);
}


function applyDroneMarkerStyle(marker, drone) {
    const el = marker.getElement();

    if (!el) {
        return;
    }

    const opacity = computeDroneOpacity(drone);
    const staleMs =
        Number.isFinite(drone.stale_ms)
            ? drone.stale_ms
            : DRONE_STALE_MS;
    const stale =
        drone.stale === true ||
        getDroneAgeMs(drone) > staleMs;

    el.style.opacity = opacity.toString();
    el.style.filter = stale ? "grayscale(1)" : "none";
    el.style.transition = "opacity 0.5s linear, filter 0.5s linear";
}


function formatMeters(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "N/D";
    }

    return `${Math.round(number)} m`;
}


function formatSpeed(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "N/D";
    }

    return `${Math.round(number * 3.6)} km/h`;
}


function formatHeading(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "N/D";
    }

    return `${Math.round(number)}&deg;`;
}


function formatAge(drone) {
    const ageMs = getDroneAgeMs(drone);
    const ageSeconds = Math.round(ageMs / 1000);

    if (ageSeconds < 60) {
        return `${ageSeconds} s fa`;
    }

    return `${Math.round(ageSeconds / 60)} min fa`;
}


function dronePopup(drone) {
    return `
        <b>${drone.model || "Drone"}</b><br>
        Vendor: ${drone.vendor || "-"}<br>
        Serial: ${drone.serial || "-"}<br>
        Source: ${drone.source || "-"}<br>
        Quota: ${formatMeters(drone.altitude)}<br>
        Altezza: ${formatMeters(drone.height)}<br>
        Velocita: ${formatSpeed(drone.speed)}<br>
        Direzione: ${formatHeading(drone.heading)}<br>
        Ultimo pacchetto: ${formatAge(drone)}
    `;
}

DRONES.updateDroneLayer = function (
    map,
    aircraft
) {

    const seen = new Set();

    aircraft.forEach(drone => {

        if (
            !isValidDronePosition(drone) ||
            isDroneExpired(drone)
        ) {
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

        marker.bindPopup(
            dronePopup(drone)
        );

        applyDroneMarkerStyle(
            marker,
            drone
        );
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
