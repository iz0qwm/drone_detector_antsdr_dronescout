// js/glider/glider-data.js

(function () {

    let _timer = null;
    let _map = null;

    window.GLIDER_DATA = {

        start(map) {

            if (_timer) {
                return;
            }

            _map = map;

            const fetchData = async () => {

                if (!_map) {
                    return;
                }

                const b = _map.getBounds();
                const ne = b.getNorthEast();
                const sw = b.getSouthWest();

                const params =
                    new URLSearchParams({
                        minLat: sw.lat,
                        maxLat: ne.lat,
                        minLon: sw.lng,
                        maxLon: ne.lng
                    });

                try {

                    const res =
                        await fetch(
                            `/api/ogn/network?${params.toString()}`
                        );

                    const json =
                        await res.json();

                    if (!json.objects) {
                        return;
                    }

                    json.objects.forEach(obj => {

                        window.GLIDER_LAYER.upsert({
                            id: obj.id,
                            lat: obj.lat,
                            lon: obj.lon,
                            alt_m: obj.alt_m,
                            ts: obj.last_seen,
                            heading:
                                window.GLIDER_UTILS
                                    ?.normalizeHeading
                                        ? window.GLIDER_UTILS
                                            .normalizeHeading(
                                                obj.heading || 0
                                            )
                                        : obj.heading || 0,
                            source: obj.source
                        });

                    });

                    console.log(
                        "[OGN-NET]",
                        json.count
                    );

                } catch(e) {

                    console.warn(
                        "[OGN-NET] fetch error",
                        e
                    );

                }
            };

            fetchData();

            _timer =
                setInterval(
                    fetchData,
                    10000
                );
        },

        stop() {

            if (_timer) {
                clearInterval(_timer);
            }

            _timer = null;
        }
    };

})();