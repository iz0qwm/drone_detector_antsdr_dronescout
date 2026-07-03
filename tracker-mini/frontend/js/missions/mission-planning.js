window.MISSION = window.MISSION || {};

MISSION.planning = {

    currentLayerType: null,
    currentMission: null,
    menuHandlerInitialized: false,
    showAllHandlerInitialized: false,
    async open() {

        MISSION.layer.clearAll();

        const showAll =
            document.getElementById(
                "showAllMissionObjects"
            );

        if (showAll) {
            showAll.checked = false;
        }

        if (
            !this.menuHandlerInitialized
        ) {

            document.addEventListener(
                "click",
                () => {

                    document
                        .getElementById(
                            "missionMenu"
                        )
                        ?.classList
                        .remove("open");

                }
            );

            this.menuHandlerInitialized =
                true;

        }

        await this.refresh();
        if (
            !this.showAllHandlerInitialized
        ) {

            document
                .getElementById(
                    "showAllMissionObjects"
                )
                ?.addEventListener(
                    "change",
                    e => {

                        this.setAllObjectsVisibility(
                            e.target.checked
                        );

                    }
                );

            this.showAllHandlerInitialized =
                true;

        }
            
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

                <div class="mission-header">

                    <div>

                        <h3>${mission.name}</h3>

                        <div class="mission-description">

                            ${mission.description || ""}

                        </div>

                    </div>

                
                    <div class="mission-menu-container">

                        <button
                            id="missionMenuBtn"
                            class="mission-menu-button">
                            ☰
                        </button>
                        <div id="missionMenu" class="mission-menu">
                            <button id="renameMissionBtn">
                                ✏ Rename mission
                            </button>
                            <hr>
                            <button id="importGeoJsonBtn">
                                📁 Import Layer
                            </button>
                            <button id="newLayerBtn">
                                ➕ New Object
                            </button>
                            <hr>
                            <button
                                id="deleteMissionBtn"
                                class="danger">
                                🗑 Delete mission
                            </button>
                        </div>
                    </div> 
                </div>       
                <hr>
                <input
                    id="geojsonFileInput"
                    type="file"
                    accept=".geojson,.json"
                    style="display:none">
                <h4>Mission Objects</h4>
                <div id="missionLayers">
                    Loading...
                </div>

            </div>

            `;

            this.bindMissionMenu();

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

                        MISSION.toolbar.setNewLayerMode();
                        MISSION.toolbar.show();

                        document
                            .getElementById(
                                "missionPlanningModal"
                            )
                            ?.classList
                            .remove("open");

                        MISSION.draw.startNewLayer(
                            mission.id
                        );

                    }
                );
                

            document
                .getElementById(
                    "renameMissionBtn"
                )
                ?.addEventListener(
                    "click",
                    () => {

                        this.renameMission();

                    }
                );

            document
                .getElementById(
                    "deleteMissionBtn"
                )
                ?.addEventListener(
                    "click",
                    () => {

                        this.deleteMission();

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

    bindMissionMenu() {

        const menuBtn =
            document.getElementById(
                "missionMenuBtn"
            );

        const menu =
            document.getElementById(
                "missionMenu"
            );

        if (!menuBtn || !menu) {
            return;
        }

        menuBtn.onclick =
            e => {

                e.stopPropagation();

                menu.classList.toggle(
                    "open"
                );

            };

        menu.onclick =
            e => {

                e.stopPropagation();

            };

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

                const visible =
                    MISSION.layer.isVisible(
                        layer.id
                    );

                const row =
                    document.createElement(
                        "div"
                    );

                row.className =
                    "mission-layer-card";

                row.dataset.layerId =
                    layer.id;

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
                            data-action="view"
                            data-layer="${layer.id}">

                            ${visible ? "🙈 Hide" : "👁 Show"}

                        </button>

                        <button
                            class="btn-small"
                            data-action="edit"
                            data-layer="${layer.id}">

                            ✏ Rename

                        </button>

                        <button
                            class="btn-small"
                            data-action="delete"
                            data-layer="${layer.id}">

                            🗑 Delete

                        </button>
                        <button
                            class="btn-small"
                            data-action="shape"
                            data-layer="${layer.id}">

                            🧭 Shape

                        </button>
                    </div>

                `;

                row
                    .querySelector(
                        '[data-action="view"]'
                    )
                    ?.addEventListener(
                        "click",
                        async () => {

                            await this.toggleLayerVisibility(
                                missionId,
                                layer.id
                            );

                        }
                    );

                row
                    .querySelector(
                        '[data-action="edit"]'
                    )
                    ?.addEventListener(
                        "click",
                        async () => {

                            await this.editLayer(
                                missionId,
                                layer
                            );

                        }
                    );

                row
                    .querySelector(
                        '[data-action="delete"]'
                    )
                    ?.addEventListener(
                        "click",
                        async () => {

                            await this.deleteLayer(
                                missionId,
                                layer
                            );

                        }
                    );

                container.appendChild(
                    row
                );

                row
                .querySelector(
                    '[data-action="shape"]'
                )
                ?.addEventListener(
                    "click",
                    async () => {

                        await this.editLayerShape(
                            missionId,
                            layer.id
                        );

                    }
                );
            });
            this.updateShowAllCheckbox();
        }

        catch(err) {

            console.error(err);

            container.innerHTML =
                "Unable to load layers";

        }

    },
    async setAllObjectsVisibility(
        visible
    ) {

        if (!this.currentMission) {
            return;
        }

        const layers =
            await MISSION.api.layers(
                this.currentMission.id
            );

        for (const layer of layers) {

            const isVisible =
                MISSION.layer.isVisible(
                    layer.id
                );

            if (visible && !isVisible) {

                MISSION.layer.show(layer);

            }

            if (!visible && isVisible) {

                MISSION.layer.hide(layer.id);

            }

        }
        this.updateShowAllCheckbox();

    },
    async toggleLayerVisibility(
        missionId,
        layerId
    ) {

        const layer =
            await MISSION.api.layer(
                missionId,
                layerId
            );

        const visible =
            MISSION.layer.toggle(
                layer
            );

        const button =
            document.querySelector(

                `[data-action="view"][data-layer="${layerId}"]`

            );

        if (button) {

            button.textContent =
                visible
                    ? "🙈 Hide"
                    : "👁 Show";

        }

        this.updateShowAllCheckbox();

    },
    updateShowAllCheckbox() {

        const checkbox =
            document.getElementById(
                "showAllMissionObjects"
            );

        if (!checkbox) {
            return;
        }

        const layers =
            document.querySelectorAll(
                ".mission-layer-card"
            );

        if (!layers.length) {

            checkbox.checked = false;
            checkbox.indeterminate = false;

            return;

        }

        let visible = 0;

        layers.forEach(row => {

            const layerId =
                row.dataset.layerId;

            if (
                MISSION.layer.isVisible(
                    layerId
                )
            ) {
                visible++;
            }

        });

        checkbox.indeterminate =
            visible > 0 &&
            visible < layers.length;

        checkbox.checked =
            visible === layers.length;

    },
    async editLayer(
        missionId,
        layer
    ) {

        const name =
            prompt(
                "Layer name",
                layer.name || ""
            );

        if (!name) {
            return;
        }

        layer.name = name;

        const result =
            await MISSION.api.updateLayer(
                missionId,
                layer
            );

        if (!result.success) {

            alert(
                "Unable to update layer"
            );

            return;

        }

        await this.loadMissionLayers(
            missionId
        );

    },

    async deleteLayer(
        missionId,
        layer
    ) {

        if (
            !confirm(
                `Delete layer "${layer.name}"?`
            )
        ) {
            return;
        }

        MISSION.layer.hide(
            layer.id
        );

        const result =
            await MISSION.api.deleteLayer(
                missionId,
                layer.id
            );

        if (!result.success) {

            alert(
                "Unable to delete layer"
            );

            return;

        }

        await this.loadMissionLayers(
            missionId
        );

    },

    async renameMission() {

        if (!this.currentMission) {
            return;
        }

        const name =
            prompt(
                "Mission name",
                this.currentMission.name
            );

        if (!name) {
            return;
        }

        const description =
            prompt(
                "Mission description",
                this.currentMission.description || ""
            );

        if (description === null) {
            return;
        }

        const result =
            await MISSION.api.updateMission(

                this.currentMission.id,

                {
                    name,
                    description
                }

            );

        if (!result.success) {

            alert(
                "Unable to update mission"
            );

            return;

        }

        this.currentMission =
            result.mission;

        await this.refresh();

    },

    async deleteMission() {

        if (!this.currentMission) {
            return;
        }

        if (
            !confirm(

                `Delete mission "${this.currentMission.name}"?`

            )
        ) {
            return;
        }

        const result =
            await MISSION.api.deleteMission(
                this.currentMission.id
            );

        if (!result.success) {

            alert(
                "Unable to delete mission"
            );

            return;

        }

        this.currentMission = null;

        MISSION.layer.clearAll();

        document.getElementById(
            "currentMissionInfo"
        ).innerHTML =
            "No active mission";

        await this.refresh();

    },

    async editLayerShape(
        missionId,
        layerId
    ) {

        const layer =
            await MISSION.api.layer(
                missionId,
                layerId
            );

        MISSION.layer.show(layer);
        MISSION.toolbar.setEditLayerMode();
        MISSION.toolbar.show();

        document
            .getElementById(
                "missionPlanningModal"
            )
            ?.classList
            .remove("open");

        MISSION.draw.startEditLayer(
            missionId,
            layer
        );

    }

};



