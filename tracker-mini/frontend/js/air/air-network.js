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

        const params =
            new URLSearchParams({
                minLat: bounds.minLat,
                maxLat: bounds.maxLat,
                minLon: bounds.minLon,
                maxLon: bounds.maxLon
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
            "[AIR-NET] sources",
            data.sources
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