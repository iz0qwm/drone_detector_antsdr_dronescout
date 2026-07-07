window.MISSION = window.MISSION || {};

MISSION.layers = {};

MISSION.layer = {

    getDscStyle(feature) {

        const lower =
            Number(
                feature?.properties?.lowerLimit ?? -1
            );

        if (lower === 0) {

            return {
                color: "rgba(0,0,0,0.35)",
                weight: 0.9,
                fillColor: "#ff2a2a",
                fillOpacity: 0.19
            }

        }

        if (lower === 25) {

            return {
                color: "#222",
                weight: 0.7,
                fillColor: "#ff9800",
                fillOpacity: 0.19
            }

        }

        if (lower === 45) {

            return {
                color: "#222",
                weight: 0.7,
                fillColor: "#ffeb3b",
                fillOpacity: 0.19
            }

        }

        if (lower === 60) {

            return {
                color: "#222",
                weight: 0.5,
                fillColor: "#29b6f6",
                fillOpacity: 0.18
            }

        }

        if (lower === 120) {

            return {
                color: "#222",
                weight: 0.7,
                fillOpacity: 0
            };

        }

        return {

            color: "#222",

            weight: 0.7,

            fillColor: "#9e9e9e",

            fillOpacity: 0.19

        }

    },

    buildLeafletLayer(layer) {

        if (!layer.geojson) {
            return null;
        }

        const feature = layer.geojson;

        if (
            feature?.properties?.leafletType ===
            "Circle"
        ) {

            const circle = L.circle(
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

            this.decorate(
                layer,
                circle,
                feature
            );

            return circle;

        }

        return L.geoJSON(
            feature,
            {
                style: feature => {

                    if (
                        layer.properties?.metadata?.source ===
                        "dsc_zones"
                    ) {

                        return this.getDscStyle(feature);

                    }

                    return layer.style || {};

                },

                onEachFeature: (
                    feature,
                    leafletLayer
                ) => {

                    this.decorate(
                        layer,
                        leafletLayer,
                        feature
                    );

                }

            }
        );

    },
    addLabel(layer, leafletLayer, feature) {

        if (
            layer.properties?.metadata?.source ===
            "dsc_zones"
        ) {
            return;
        }

        if (layer.properties?.showLabel === false) {
            return;
        }

        const text =
            this.buildLabel(
                layer,
                feature
            );

        if (!text) {
            return;
        }

        leafletLayer.bindTooltip(
            text,
            {
                permanent: true,
                direction: "center",
                className: "mission-label"
            }
        );

    },
    buildLabel(layer, feature) {

        let lines = [];

        if (layer.name) {
            lines.push(layer.name);
        }

        if (
            layer.properties?.showMeasurements
        ) {

            //
            // Circle
            //
            if (
                feature?.properties?.leafletType ===
                "Circle"
            ) {

                const radius =
                    feature.properties.radius;

                const area =
                    Math.PI * radius * radius;

                if (radius) {

                    lines.push(
                        `📏 R = ${Math.round(radius)} m`
                    );

                    lines.push(
                        `⬜ ${(area / 1000000).toFixed(2)} km²`
                    );

                }

            }

            //
            // Polygon
            //
            else if (
                feature.geometry.type ===
                "Polygon"
            ) {

                const area =
                    turf.area(feature);

                lines.push(
                    `⬜ ${(area / 1000000).toFixed(2)} km²`
                );

            }

        }

        return lines.join("<br>");

    },
    addPopup(layer, leafletLayer, feature) {
        if (
            layer.properties?.metadata?.source ===
            "dsc_zones"
        ) {
            leafletLayer.bindPopup(`
                <b>${feature.properties.name}</b>
                <br>
                ${feature.properties.type}
                <br>
                Lower:
                ${feature.properties.lowerLimit} m
                <br>
                Upper:
                ${feature.properties.upperLimit} m
            `);
            return;
        }
        leafletLayer.bindPopup(`
            <b>${layer.name || "Mission Object"}</b>
            <br>
            Type:
            ${layer.type || "-"}
            <br>
            ${layer.properties?.description || ""}
        `);
    },
    decorate(layer, leafletLayer, feature) {

        this.addLabel(
            layer,
            leafletLayer,
            feature
        );

        this.addPopup(
            layer,
            leafletLayer,
            feature
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