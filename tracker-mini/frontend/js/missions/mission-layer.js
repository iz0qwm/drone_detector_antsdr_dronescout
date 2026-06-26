window.MISSION = window.MISSION || {};

MISSION.layers = {};

MISSION.layer = {

    show(id, layer) {

        MISSION.layers[id] = layer;

        layer.addTo(
            window.airNodeMap
        );

    },

    hide(id) {

        if (!MISSION.layers[id]) {
            return;
        }

        window.airNodeMap.removeLayer(
            MISSION.layers[id]
        );

        delete MISSION.layers[id];

    }

};