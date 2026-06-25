window.MESHTASTIC = window.MESHTASTIC || {};

MESHTASTIC.refreshTimer = null;

MESHTASTIC.refresh = async function() {

    try {

        const data =
            await MESHTASTIC.fetchTeams();

        MESHTASTIC.updateOperatorsLayer(
            data.operators || []
        );

    } catch(err) {

        console.error(
            "Meshtastic layer error",
            err
        );

    }

};


MESHTASTIC.start = function(map) {

    window.airNodeMap = map;

    MESHTASTIC.refresh();

    if (
        MESHTASTIC.refreshTimer
    ) {
        clearInterval(
            MESHTASTIC.refreshTimer
        );
    }

    MESHTASTIC.refreshTimer =
        setInterval(
            MESHTASTIC.refresh,
            5000
        );

};


MESHTASTIC.stop = function() {

    if (
        MESHTASTIC.refreshTimer
    ) {

        clearInterval(
            MESHTASTIC.refreshTimer
        );

        MESHTASTIC.refreshTimer = null;

    }

    MESHTASTIC.clearOperators();

};