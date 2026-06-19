window.DRONES = window.DRONES || {};

DRONES.droneTrafficTimer = null;

DRONES.startDroneTraffic = function(map) {

    if (DRONES.droneTrafficTimer) {
        return;
    }

    async function update() {
        const aircraft =
            await DRONES.fetchRemoteIdAircraft();

        DRONES.updateDroneLayer(
            map,
            aircraft
        );
    }

    update();

    DRONES.droneTrafficTimer =
        setInterval(
            update,
            5000
        );
};

DRONES.stopDroneTraffic = function() {
    if (DRONES.droneTrafficTimer) {
        clearInterval(
            DRONES.droneTrafficTimer
        );

        DRONES.droneTrafficTimer = null;
    }

    if (DRONES.clearDroneLayer) {
        DRONES.stopDroneTraffic();
    }
};