window.AIR = window.AIR || {};

let airTrafficTimer = null;

AIR.startAirTraffic = function(map, options = {}) {

    if (airTrafficTimer) {
        return;
    }

    AIR.maxAltitudeMeters =
        options.maxAltitudeMeters || 1000;

    const layer =
        AIR.createAirLayer(map);

    layer.addTo(map);

    const update = async () => {

        const b = map.getBounds();
        const ne = b.getNorthEast();
        const sw = b.getSouthWest();

        const bounds = {
            minLat: sw.lat,
            maxLat: ne.lat,
            minLon: sw.lng,
            maxLon: ne.lng
        };

        const networkAircraft =
            await AIR.fetchNetworkAircraft(
                bounds,
                options
            );

        const localAircraft =
            await AIR.fetchLocalAircraft(
                bounds
            );

        const aircraft = [
            ...networkAircraft,
            ...localAircraft
        ];

        console.log(
            "[AIR]",
            "NET:",
            networkAircraft.length,
            "LOCAL:",
            localAircraft.length
        );

        AIR.updateAirLayer(
            aircraft
        );
    };

    update();

    airTrafficTimer =
        setInterval(
            update,
            15000
        );
};

AIR.stopAirTraffic = function() {

    if (airTrafficTimer) {
        clearInterval(
            airTrafficTimer
        );
        airTrafficTimer = null;
    }

    AIR.clearAirLayer();
};