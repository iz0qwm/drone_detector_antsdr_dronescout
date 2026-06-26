window.MISSION = window.MISSION || {};

MISSION.planning = {

    currentLayerType: null,
    currentMission: null,
    async open() {

        await this.refresh();

    },

    async refresh() {

        await Promise.all([
            this.loadCurrentMission(),
            this.loadMissionList()
        ]);

    },

    async loadCurrentMission() {

        const box =
            document.getElementById(
                "currentMissionInfo"
            );

        box.innerHTML = "Loading...";

        try {

            this.currentMission =
                await MISSION.api.currentMission();

            const mission =
                this.currentMission;

            if (!mission) {

                box.innerHTML =
                    "No active mission";

                return;

            }

            box.innerHTML = `

            <div class="mission-current-card">

                <h3>${mission.name}</h3>

                <div class="mission-description">

                    ${mission.description || ""}

                </div>

                <hr>

                <button
                    id="importGeoJsonBtn"
                    class="btn-mission">

                    Import Layer

                </button>
                <input
                    id="geojsonFileInput"
                    type="file"
                    accept=".geojson,.json"
                    style="display:none">
                <button
                    id="newLayerBtn"
                    class="btn-mission">

                    New Layer

                </button>

                <hr>

                <h3>Mission Layers</h3>

                <div id="missionLayers">

                    Loading...

                </div>

            </div>

            `;

            document
                .getElementById(
                    "importGeoJsonBtn"
                )
                ?.addEventListener(
                    "click",
                    () => {

                        document
                            .getElementById(
                                "geojsonFileInput"
                            )
                            ?.click();

                    }
                );

            document
                .getElementById(
                    "geojsonFileInput"
                )
                ?.addEventListener(

                    "change",

                    async e => {

                        const file =
                            e.target.files[0];

                        if (!file) {
                            return;
                        }

                        await MISSION.api.importGeoJSON(

                            mission.id,

                            file

                        );

                        await this.loadMissionLayers(
                            mission.id
                        );

                    }

                );

                
            document
                .getElementById(
                    "newLayerBtn"
                )
                ?.addEventListener(
                    "click",
                    () => {

                        console.log(
                            "[MISSION] New Layer"
                        );

                    }
                );
                
        }

        catch(err) {

            console.error(err);

            box.innerHTML =
                "Unable to load mission";

        }

        if (this.currentMission) {

            await this.loadMissionLayers(
                this.currentMission.id
            );

        }
    },

    async loadMissionList() {

        const container =
            document.getElementById(
                "missionsList"
            );

        container.innerHTML =
            "Loading...";

        try {

            const missions =
                await MISSION.api.missions();

            container.innerHTML = "";

            missions.forEach(mission => {

                const card =
                    document.createElement("div");

                card.className =
                    "mission-card";

                card.innerHTML = `

                    <div class="mission-card-title">

                        ${mission.name}

                    </div>

                    <div class="mission-card-description">

                        ${mission.description || ""}

                    </div>

                    <div class="mission-card-status">

                        ${mission.status}

                    </div>

                `;

                card.addEventListener(
                    "click",
                    async () => {

                        await this.selectMission(
                            mission.id
                        );

                    }
                );

                container.appendChild(card);

            });

        }

        catch(err) {

            console.error(err);

            container.innerHTML =
                "Unable to load missions";

        }

    },

    async selectMission(
        missionId
    ) {

        const result =
            await MISSION.api.selectMission(
                missionId
            );

        if (!result.success) {

            alert(
                "Unable to select mission"
            );

            return;

        }

        await this.refresh();

    },

    async loadMissionLayers(
        missionId
    ) {

        const container =
            document.getElementById(
                "missionLayers"
            );

        if (!container) {
            return;
        }

        container.innerHTML =
            "Loading...";

        try {

            const layers =
                await MISSION.api.layers(
                    missionId
                );

            if (!layers.length) {

                container.innerHTML =
                    "No layers";

                return;
            }

            container.innerHTML = "";

            layers.forEach(layer => {

                const row =
                    document.createElement(
                        "div"
                    );

                row.className =
                    "mission-layer-card";

                row.innerHTML = `

                    <div class="mission-layer-header">

                        <b>${layer.name}</b>

                    </div>

                    <div class="mission-layer-info">

                        ${layer.type}

                    </div>

                    <div class="mission-layer-actions">

                        <button
                            class="btn-small"
                            data-layer="${layer.id}">

                            👁

                        </button>

                        <button
                            class="btn-small"
                            data-layer="${layer.id}">

                            ✏

                        </button>

                        <button
                            class="btn-small"
                            data-layer="${layer.id}">

                            🗑

                        </button>

                    </div>

                `;

                container.appendChild(
                    row
                );

            });

        }

        catch(err) {

            console.error(err);

            container.innerHTML =
                "Unable to load layers";

        }

    },


};