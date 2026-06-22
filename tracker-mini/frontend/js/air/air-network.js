window.AIR = window.AIR || {};

AIR.fetchNetworkAircraft =
async function(
    bounds,
    options = {}
) {

    const enabled =
    localStorage.getItem(
        "adsbNetworkEnabled"
    );

    if (enabled === "false") {

        return [];

    }

    try {

        const showAll =
            document.getElementById(
                "showHighAltitudeAircraft"
            )?.checked || false;

        const params =
            new URLSearchParams({
                minLat: bounds.minLat,
                maxLat: bounds.maxLat,
                minLon: bounds.minLon,
                maxLon: bounds.maxLon,
                showAll: showAll
            });

        const res =
            await fetch(
                `/api/air/network?${params.toString()}`
            );

        if (!res.ok) {
            console.warn(
                "[AIR-NET] HTTP",
                res.status
            );
            return [];
        }

        const data =
            await res.json();

        console.log(
            "[AIR-NET] showAll=",
            showAll
        );

        return data.aircraft || [];

    } catch(err) {

        console.error(
            "[AIR-NET] error",
            err
        );

        return [];
    }
};


AIR.fetchLocalAircraft =
async function(
    bounds
) {

    const enabled =
        localStorage.getItem(
            "adsbLocalEnabled"
        );

    if (enabled === "false") {
        return [];
    }

    try {

        const showAll =
            document.getElementById(
                "showHighAltitudeAircraft"
            )?.checked || false;

        const params =
            new URLSearchParams({
                minLat: bounds.minLat,
                maxLat: bounds.maxLat,
                minLon: bounds.minLon,
                maxLon: bounds.maxLon,
                showAll: showAll
            });

        const res =
            await fetch(
                `/api/air/local?${params.toString()}`
            );

        if (!res.ok) {
            console.warn(
                "[AIR-LOCAL] HTTP",
                res.status
            );
            return [];
        }

        const data =
            await res.json();

        console.log(
            "[AIR-LOCAL] aircraft=",
            data.aircraft?.length || 0,
            "showAll=",
            showAll
        );

        return data.aircraft || [];

    } catch(err) {

        console.error(
            "[AIR-LOCAL] error",
            err
        );

        return [];
    }
};