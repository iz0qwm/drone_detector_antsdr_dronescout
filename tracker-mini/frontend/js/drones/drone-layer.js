window.DRONES = window.DRONES || {};

DRONES.markers = {};
const DRONE_STALE_MS = 45000;
const DRONE_RETENTION_MS = 180000;
const DRONE_MIN_OPACITY = 0.25;
const DRONE_TRAIL_MAX_AGE_MS = 45 * 60 * 1000;
const DRONE_TRAIL_MAX_POINTS = 600;
const DRONE_TRAIL_MIN_DISTANCE_M = 20;

DRONES.trails =
    window.TRACK_HISTORY
        ? window.TRACK_HISTORY.create({
            ...window.TRACK_HISTORY.getCategorySettings(
                "drone"
            ),
            maxAgeMs:
                window.TRACK_HISTORY.getCategorySettings(
                    "drone"
                ).maxAgeMs ||
                DRONE_TRAIL_MAX_AGE_MS,
            maxPoints: DRONE_TRAIL_MAX_POINTS,
            minDistanceMeters: DRONE_TRAIL_MIN_DISTANCE_M,
            color: "#2f80ed",
            weight: 3,
            minOpacity: 0.1,
            dashArray: "8 7",
            pane: "traffic-drone",
            className: "drone-trail"
        })
        : null;


DRONES.applyTrailSettings = function(map) {
    if (!DRONES.trails || !window.TRACK_HISTORY) {
        return;
    }

    const settings =
        window.TRACK_HISTORY.getCategorySettings(
            "drone"
        );

    DRONES.trails.configure(
        settings
    );

    const targetMap =
        map ||
        window.airNodeMap;

    if (!targetMap) {
        return;
    }

    if (settings.enabled) {
        DRONES.trails.prune(
            targetMap
        );
    } else {
        DRONES.trails.clear(
            targetMap
        );
    }
};


DRONES.clearTrails = function(map) {
    if (
        DRONES.trails &&
        (map || window.airNodeMap)
    ) {
        DRONES.trails.clear(
            map ||
            window.airNodeMap
        );
    }
};


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


function getDroneTrackId(drone) {
    return (
        drone.serial ||
        drone.id ||
        drone.uas_id ||
        drone.operator_id ||
        [
            drone.source || "drone",
            drone.vendor || "-",
            drone.model || "-",
            Number(drone.lat).toFixed(5),
            Number(drone.lon).toFixed(5)
        ].join(":")
    );
}


function getDronePositionTimestampMs(drone) {
    if (Number.isFinite(drone.updatedAt)) {
        return drone.updatedAt;
    }

    const lastSeen =
        Date.parse(drone.last_seen);

    if (Number.isFinite(lastSeen)) {
        return lastSeen;
    }

    if (Number.isFinite(drone.age_ms)) {
        return Date.now() -
            Math.max(
                0,
                drone.age_ms
            );
    }

    return Date.now();
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
            getDroneTrackId(
                drone
            );

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

        if (DRONES.trails) {
            DRONES.trails.update(
                map,
                id,
                drone.lat,
                drone.lon,
                getDronePositionTimestampMs(
                    drone
                )
            );
        }
    });

    if (DRONES.trails) {
        DRONES.trails.prune(
            map
        );
    }

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

    if (
        DRONES.trails &&
        window.airNodeMap
    ) {
        DRONES.clearTrails(
            window.airNodeMap
        );
    }
};
