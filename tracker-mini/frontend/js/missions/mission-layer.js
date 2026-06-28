window.MISSION = window.MISSION || {};

MISSION.layers = {};

MISSION.layer = {

    buildLeafletLayer(layer) {

        if (!layer.geojson) {
            return null;
        }

        const feature = layer.geojson;
        
        if (
            feature?.properties?.leafletType ===
            "Circle"
        ) {
            return L.circle(
                [
                    feature.geometry.coordinates[1],
                    feature.geometry.coordinates[0]
                ],
                {
                    radius:
                        feature.properties.radius,
                    ...(layer.style || {})
                }
            );
        }

        return L.geoJSON(
            layer.geojson,
            {
                style: layer.style || {},
                onEachFeature(feature, leafletLayer) {

                    leafletLayer.bindPopup(
                        layer.name || "Mission Layer"
                    );

                }
            }
        );

    },

    show(layer) {

        if (!layer || !layer.id) {
            return;
        }

        if (MISSION.layers[layer.id]) {
            return;
        }

        const leafletLayer =
            this.buildLeafletLayer(
                layer
            );

        if (!leafletLayer) {
            return;
        }

        MISSION.layers[layer.id] =
            leafletLayer;

        leafletLayer.addTo(
            window.airNodeMap
        );

        try {

            window.airNodeMap.fitBounds(
                leafletLayer.getBounds()
            );

        } catch(err) {

            console.warn(
                "[MISSION] Unable to fit layer bounds",
                err
            );

        }

    },

    hide(layerId) {

        if (!MISSION.layers[layerId]) {
            return;
        }

        window.airNodeMap.removeLayer(
            MISSION.layers[layerId]
        );

        delete MISSION.layers[layerId];

    },

    toggle(layer) {

        if (
            MISSION.layers[layer.id]
        ) {

            this.hide(
                layer.id
            );

            return false;

        }

        this.show(
            layer
        );

        return true;

    },
    isVisible(layerId) {

        return !!MISSION.layers[layerId];

    },
    refresh(layer) {

        this.hide(
            layer.id
        );

        this.show(
            layer
        );

    },
    clearAll() {

        Object.keys(
            MISSION.layers
        ).forEach(layerId => {

            this.hide(
                layerId
            );

        });

    }

};