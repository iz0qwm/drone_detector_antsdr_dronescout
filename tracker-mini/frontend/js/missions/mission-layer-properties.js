window.MISSION = window.MISSION || {};

MISSION.layerProperties = {

    currentLayer: null,
    currentMode: null,
    currentMissionId: null,

    open(
        mode,
        missionId,
        layer
    ) {

        this.currentMode = mode;
        this.currentMissionId = missionId;
        this.currentLayer = layer;

        const isDscLayer =
            layer?.properties?.metadata?.source ===
            "dsc_zones";
            
        document
            .getElementById(
                "layerName"
            )
            .value =
                layer?.name ?? "";

        document
            .getElementById(
                "layerCategory"
            )
            .value =
                layer?.type ?? "generic";

        document
            .getElementById(
                "layerColor"
            )
            .value =
                layer?.style?.color ?? "#ff0000";

        document
            .getElementById(
                "layerDescription"
            )
            .value =
                layer?.properties?.description ?? "";

        document
            .getElementById(
                "layerShowLabel"
            )
            .checked =
                isDscLayer
                    ? false
                    : (layer?.properties?.showLabel ?? true);

        document
            .getElementById(
                "layerShowMeasurements"
            )
            .checked =
            isDscLayer
                ? false
                : (layer?.properties?.showMeasurements ?? true);

        document.getElementById(
            "layerShowLabel"
        ).disabled = isDscLayer;

        document.getElementById(
            "layerShowMeasurements"
        ).disabled = isDscLayer;


        document
            .getElementById(
                "layerPropertiesModal"
            )
            .classList
            .add("open");

    },

    close() {

        document
            .getElementById(
                "layerPropertiesModal"
            )
            .classList
            .remove("open");

    },

    async save() {

        const layer = {

            ...(this.currentLayer || {}),

            name:
                document
                    .getElementById(
                        "layerName"
                    )
                    .value,

            type:
                document
                    .getElementById(
                        "layerCategory"
                    )
                    .value,

            style: {

                ...(this.currentLayer?.style || {}),

                color:
                    document
                        .getElementById(
                            "layerColor"
                        )
                        .value,

                fillColor:
                    document
                        .getElementById(
                            "layerColor"
                        )
                        .value,

                fillOpacity: 0.25,

                weight: 2

            },

            properties: {

                ...(this.currentLayer?.properties || {}),

                description:
                    document
                        .getElementById(
                            "layerDescription"
                        )
                        .value,

                showLabel:
                    document
                        .getElementById(
                            "layerShowLabel"
                        )
                        .checked,

                showMeasurements:
                    document
                        .getElementById(
                            "layerShowMeasurements"
                        )
                        .checked

            },

            visible: true,
            locked: false

        };

        layer.geojson = MISSION.draw.buildGeoJson();

        if (!layer.geojson) {

            alert(
                "Unable to read geometry"
            );

            return;

        }

        layer.geometry =
            layer.geojson.geometry?.type || "unknown";


        if (
            this.currentMode === "new-layer"
        ) {

            const result =
                await MISSION.api.createLayer(
                    this.currentMissionId,
                    layer
                );

            if (
                !this.checkResult(
                    result,
                    "Unable to create layer"
                )
            ) {
                return;
            }

            MISSION.layer.show(
                result.layer
            );

        } else {

            if (!layer.id) {
                alert(
                    "Layer id missing"
                );
                return;
            }

            const result =
                await MISSION.api.updateLayer(
                    this.currentMissionId,
                    layer
                );

            if (
                !this.checkResult(
                    result,
                    "Unable to update layer"
                )
            ) {
                return;
            }

            MISSION.layer.refresh(
                result.layer
            );

        }

        this.close();

        MISSION.draw.cancel();

        if (
            MISSION.planning.currentMission
        ) {

            await MISSION.planning.loadMissionLayers(
                MISSION.planning.currentMission.id
            );

        }

    },
    checkResult(result, message) {
        if (result.success) {
            return true;
        }
        alert(message);
        return false;
    },
    init() {

        document
            .getElementById(
                "closeLayerPropertiesBtn"
            )
            ?.addEventListener(
                "click",
                () => this.close()
            );

        document
            .getElementById(
                "cancelLayerPropertiesBtn"
            )
            ?.addEventListener(
                "click",
                () => this.close()
            );

        document
            .getElementById(
                "saveLayerPropertiesBtn"
            )
            ?.addEventListener(
                "click",
                () => this.save()
            );
    }


};