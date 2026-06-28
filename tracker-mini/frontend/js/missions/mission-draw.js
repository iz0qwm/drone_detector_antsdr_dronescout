window.MISSION = window.MISSION || {};

MISSION.draw = {

    mode: null,
    editingMissionId: null,
    editingLayer: null,
    leafletLayer: null,
    originalGeoJson: null,
    temporaryGeoJson: null,
    drawLayer: null,
    drawnLayer: null,

    startRectangle() {

        if (this.mode !== "new-layer") {
            return;
        }

        this.disableGeomanModes();

        window.airNodeMap.pm.enableDraw(
            "Rectangle"
        );

        console.log(
            "[DRAW] Rectangle mode"
        );

    },

    startPolygon() {

        if (this.mode !== "new-layer") {
            return;
        }

        this.disableGeomanModes();

        window.airNodeMap.pm.enableDraw(
            "Polygon"
        );

        console.log(
            "[DRAW] Polygon mode"
        );

    },

    startCircle() {

        if (this.mode !== "new-layer") {
            return;
        }

        this.disableGeomanModes();

        window.airNodeMap.pm.enableDraw(
            "Circle"
        );

        console.log(
            "[DRAW] Circle mode"
        );

    },

    startMarker() {

        if (this.mode !== "new-layer") {
            return;
        }

        this.disableGeomanModes();

        window.airNodeMap.pm.enableDraw(
            "Marker"
        );

        console.log(
            "[DRAW] Marker mode"
        );

    },
    enableVertexEditing() {

        if (this.mode !== "edit-layer") {
            return;
        }

        if (!this.leafletLayer) {
            return;
        }

        this.disableGeomanModes();

        if (this.leafletLayer.eachLayer) {

            this.leafletLayer.eachLayer(
                layer => {

                    if (layer.pm) {

                        layer.pm.enable({
                            allowSelfIntersection: false
                        });

                    }

                }
            );

        }
        else if (this.leafletLayer.pm) {

            this.leafletLayer.pm.enable();

        }

        console.log(
            "[DRAW] Enable vertex editing"
        );

    },

    startEditLayer(
        missionId,
        layer
    ) {

        this.mode = "edit-layer";

        this.editingMissionId = missionId;

        this.editingLayer = layer;

        this.originalGeoJson =
            structuredClone(
                layer.geojson
            );

        this.leafletLayer =
            MISSION.layers[layer.id];

        console.log(
            "[DRAW] Edit layer",
            missionId,
            layer.id
        );

    },

    startNewLayer(missionId) {

        this.originalGeoJson = null;
        this.leafletLayer = null;
        this.temporaryGeoJson = null;

        this.mode =
            "new-layer";

        this.editingMissionId =
            missionId;

        this.editingLayer =
            null;

        console.log(
            "[DRAW] New layer",
            missionId
        );

        window.airNodeMap.off(
            "pm:create"
        );

        window.airNodeMap.on(
            "pm:create",
            e => {

                if (this.mode !== "new-layer") {
                    return;
                }

                if (this.drawnLayer) {

                    window.airNodeMap.removeLayer(
                        this.drawnLayer
                    );

                }

                this.drawnLayer =
                    e.layer;

                this.temporaryGeoJson =
                    e.layer.toGeoJSON();

                this.temporaryGeoJson.properties =
                    this.temporaryGeoJson.properties || {};

                if (e.shape === "Circle") {
                    this.temporaryGeoJson.properties.radius =
                        e.layer.getRadius();
                }

                this.temporaryGeoJson.properties.leafletType =
                    e.shape;

                console.log(
                    "[DRAW] Created geometry",
                    this.temporaryGeoJson
                );

            }
        );
    },
    buildGeoJson() {

        //
        // Nuovo layer
        //
        if (this.mode === "new-layer") {

            if (!this.temporaryGeoJson) {
                return null;
            }

            return structuredClone(
                this.temporaryGeoJson
            );

        }

        //
        // Modifica layer
        //
        if (this.mode === "edit-layer") {

            if (!this.leafletLayer) {
                return null;
            }

            const geojson =
                this.leafletLayer.toGeoJSON();

            geojson.properties =
                geojson.properties || {};

            if (
                this.leafletLayer instanceof L.Circle
            ) {

                geojson.properties.leafletType =
                    "Circle";

                geojson.properties.radius =
                    this.leafletLayer.getRadius();

            }

            return geojson;

        }

        return null;

    },

    save() {
        MISSION.layerProperties.open(
            this.mode,
            this.editingMissionId,
            this.editingLayer
        );
        console.log("[DRAW] Save");
    },

    revert() {
        console.log("[DRAW] Revert");

        if (
            this.mode !== "edit-layer"
        ) {
            return;
        }

    },

    cancel() {

        console.log("[DRAW] Cancel");

        this.disableGeomanModes();

        if (this.drawnLayer) {

            window.airNodeMap.removeLayer(
                this.drawnLayer
            );

        }

        this.drawnLayer = null;
        this.temporaryGeoJson = null;

        MISSION.toolbar.hide();

        this.resetSession();
    },

    resetSession() {

        this.mode = null;

        this.editingMissionId = null;

        this.editingLayer = null;

        this.leafletLayer = null;

        this.originalGeoJson = null;

        this.temporaryGeoJson = null;

        this.drawnLayer = null;

    }, 

    disableGeomanModes() {

        if (!window.airNodeMap?.pm) {
            return;
        }

        window.airNodeMap.pm.disableDraw();

        if (this.leafletLayer?.pm) {
            this.leafletLayer.pm.disable();
        }

    },

    


};


