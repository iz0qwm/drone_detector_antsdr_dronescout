window.DRONES = window.DRONES || {};

DRONES.startDroneTraffic =
function(map) {

    async function update() {

        const aircraft =
            await DRONES.fetchRemoteIdAircraft();

        DRONES.updateDroneLayer(
            map,
            aircraft
        );
    }

    update();

    setInterval(
        update,
        5000
    );
};