window.TRACK_HISTORY = window.TRACK_HISTORY || {};

(function(namespace) {
    const EARTH_RADIUS_M = 6371000;
    const CATEGORY_DEFAULTS = {
        air: {
            enabledKey: "mapTrails.air.enabled",
            durationKey: "mapTrails.air.durationMs",
            enabled: true,
            maxAgeMs: 90000
        },
        drone: {
            enabledKey: "mapTrails.drone.enabled",
            durationKey: "mapTrails.drone.durationMs",
            enabled: true,
            maxAgeMs: 45 * 60 * 1000
        },
        operator: {
            enabledKey: "mapTrails.operator.enabled",
            durationKey: "mapTrails.operator.durationMs",
            enabled: true,
            maxAgeMs: 2 * 60 * 60 * 1000
        }
    };

    function toNumber(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function toTimestampMs(value, fallback = Date.now()) {
        if (Number.isFinite(value)) {
            return value;
        }

        const parsed = Date.parse(value);

        if (Number.isFinite(parsed)) {
            return parsed;
        }

        return fallback;
    }

    function distanceMeters(a, b) {
        const lat1 = a.lat * Math.PI / 180;
        const lat2 = b.lat * Math.PI / 180;
        const dLat = lat2 - lat1;
        const dLon = (b.lon - a.lon) * Math.PI / 180;

        const h =
            Math.sin(dLat / 2) ** 2 +
            Math.cos(lat1) *
            Math.cos(lat2) *
            Math.sin(dLon / 2) ** 2;

        return EARTH_RADIUS_M *
            2 *
            Math.atan2(
                Math.sqrt(h),
                Math.sqrt(1 - h)
            );
    }

    function removeSegments(container, track) {
        track.segments.forEach(segment => {
            if (segment) {
                container.removeLayer(segment);
            }
        });

        track.segments = [];
    }

    function getCategoryDefaults(category) {
        return CATEGORY_DEFAULTS[category] || null;
    }

    function getCategorySettings(category) {
        const defaults =
            getCategoryDefaults(category);

        if (!defaults) {
            return {
                enabled: true,
                maxAgeMs: 90000
            };
        }

        const enabledValue =
            localStorage.getItem(
                defaults.enabledKey
            );
        const durationValue =
            Number(
                localStorage.getItem(
                    defaults.durationKey
                )
            );

        return {
            enabled:
                enabledValue === null
                    ? defaults.enabled
                    : enabledValue === "true",
            maxAgeMs:
                Number.isFinite(durationValue) &&
                durationValue > 0
                    ? durationValue
                    : defaults.maxAgeMs
        };
    }

    function saveCategorySettings(category, values = {}) {
        const defaults =
            getCategoryDefaults(category);

        if (!defaults) {
            return;
        }

        if (typeof values.enabled === "boolean") {
            localStorage.setItem(
                defaults.enabledKey,
                values.enabled
            );
        }

        if (
            Number.isFinite(values.maxAgeMs) &&
            values.maxAgeMs > 0
        ) {
            localStorage.setItem(
                defaults.durationKey,
                String(values.maxAgeMs)
            );
        }
    }

    function create(options = {}) {
        const settings = {
            enabled: options.enabled !== false,
            maxAgeMs: options.maxAgeMs || 90000,
            maxPoints: options.maxPoints || 120,
            minDistanceMeters: options.minDistanceMeters || 10,
            color: options.color || "#ff3b30",
            weight: options.weight || 3,
            opacity: options.opacity || 0.95,
            minOpacity: options.minOpacity || 0.12,
            dashArray: options.dashArray || null,
            pane: options.pane || undefined,
            className: options.className || ""
        };

        const tracks = new Map();

        function pruneTrack(container, id, now) {
            const track = tracks.get(id);

            if (!track) {
                return;
            }

            track.points = track.points.filter(point =>
                now - point.t <= settings.maxAgeMs
            );

            if (track.points.length < 2) {
                removeSegments(container, track);

                if (track.points.length === 0) {
                    tracks.delete(id);
                }

                return;
            }

            renderTrack(container, track, now);
        }

        function renderTrack(container, track, now) {
            removeSegments(container, track);

            for (let i = 1; i < track.points.length; i++) {
                const previous = track.points[i - 1];
                const current = track.points[i];
                const ageMs = now - current.t;

                if (ageMs > settings.maxAgeMs) {
                    continue;
                }

                const ratio =
                    Math.max(
                        0,
                        1 - (ageMs / settings.maxAgeMs)
                    );
                const opacity =
                    settings.minOpacity +
                    ((settings.opacity - settings.minOpacity) * ratio);

                const segment = L.polyline(
                    [
                        [previous.lat, previous.lon],
                        [current.lat, current.lon]
                    ],
                    {
                        color: settings.color,
                        weight: settings.weight,
                        opacity,
                        dashArray: settings.dashArray,
                        pane: settings.pane,
                        interactive: false,
                        className: settings.className
                    }
                ).addTo(container);

                if (segment.bringToBack) {
                    segment.bringToBack();
                }

                track.segments.push(segment);
            }
        }

        return {
            configure(nextSettings = {}) {
                if (typeof nextSettings.enabled === "boolean") {
                    settings.enabled =
                        nextSettings.enabled;
                }

                if (
                    Number.isFinite(nextSettings.maxAgeMs) &&
                    nextSettings.maxAgeMs > 0
                ) {
                    settings.maxAgeMs =
                        nextSettings.maxAgeMs;
                }
            },

            update(container, id, latValue, lonValue, timestampValue) {
                if (!container || !id) {
                    return;
                }

                if (!settings.enabled) {
                    this.clear(container);
                    return;
                }

                const lat = toNumber(latValue);
                const lon = toNumber(lonValue);

                if (lat === null || lon === null) {
                    return;
                }

                const now = Date.now();
                const timestamp =
                    toTimestampMs(
                        timestampValue,
                        now
                    );

                let track = tracks.get(id);

                if (!track) {
                    track = {
                        points: [],
                        segments: []
                    };
                    tracks.set(id, track);
                }

                track.points = track.points.filter(point =>
                    now - point.t <= settings.maxAgeMs
                );

                const point = {
                    lat,
                    lon,
                    t: timestamp
                };
                const last =
                    track.points[
                        track.points.length - 1
                    ];

                if (!last) {
                    track.points.push(point);
                } else if (timestamp > last.t) {
                    const movedMeters =
                        distanceMeters(
                            last,
                            point
                        );

                    if (
                        movedMeters >=
                        settings.minDistanceMeters
                    ) {
                        track.points.push(point);
                    } else {
                        last.lat = lat;
                        last.lon = lon;
                        last.t = timestamp;
                    }
                }

                while (
                    track.points.length >
                    settings.maxPoints
                ) {
                    track.points.shift();
                }

                pruneTrack(container, id, now);
            },

            prune(container) {
                if (!container) {
                    return;
                }

                if (!settings.enabled) {
                    this.clear(container);
                    return;
                }

                const now = Date.now();

                Array.from(tracks.keys())
                    .forEach(id =>
                        pruneTrack(
                            container,
                            id,
                            now
                        )
                    );
            },

            remove(container, id) {
                const track =
                    tracks.get(id);

                if (!track || !container) {
                    tracks.delete(id);
                    return;
                }

                removeSegments(
                    container,
                    track
                );
                tracks.delete(id);
            },

            clear(container) {
                if (container) {
                    tracks.forEach(track =>
                        removeSegments(
                            container,
                            track
                        )
                    );
                }

                tracks.clear();
            }
        };
    }

    namespace.create = create;
    namespace.toTimestampMs = toTimestampMs;
    namespace.getCategorySettings = getCategorySettings;
    namespace.saveCategorySettings = saveCategorySettings;
    namespace.CATEGORY_DEFAULTS = CATEGORY_DEFAULTS;

})(window.TRACK_HISTORY);
