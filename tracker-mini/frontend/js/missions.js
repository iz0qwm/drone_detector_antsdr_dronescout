
const createMissionModal =
    document.getElementById(
        "createMissionModal"
    );


const missionPlanningModal =
    document.getElementById(
        "missionPlanningModal"
    );

const missionGeoJsonLayers = {};

document.addEventListener(
    "DOMContentLoaded",
    () => {

        console.log("MISSIONS DOM READY");
        
        const createBtn =
            document.getElementById(
                "createMissionBtn"
            );

        const closeBtn =
            document.getElementById(
                "closeCreateMissionModal"
            );

        const saveBtn =
            document.getElementById(
                "saveMissionBtn"
            );

        if (createBtn) {
            createBtn.addEventListener(
                "click",
                () => {
                    createMissionModal
                        .classList
                        .add("open");
                }
            );
        }

        if (closeBtn) {
            closeBtn.addEventListener(
                "click",
                () => {
                    createMissionModal
                        .classList
                        .remove("open");
                }
            );
        }

        if (saveBtn) {
            saveBtn.addEventListener(
                "click",
                createMission
            );
        }



        const planningBtn =
            document.getElementById(
                "missionPlanningBtn"
            );

        const closePlanningBtn =
            document.getElementById(
                "closeMissionPlanningModal"
            );

        if (planningBtn) {
            planningBtn.addEventListener(
                "click",
                openMissionPlanning
            );
        }

        if (closePlanningBtn) {
            closePlanningBtn.addEventListener(
                "click",
                () => {
                    missionPlanningModal
                        .classList
                        .remove("open");
                }
            );
        }


    }
);

async function createMission() {
    const name =
        document
            .getElementById(
                "missionName"
            )
            .value
            .trim();

    const description =
        document
            .getElementById(
                "missionDescription"
            )
            .value
            .trim();

    if (!name) {
        alert(
            "Mission name required"
        );
        return;
    }

    try {
        const res =
            await fetch(
                "/api/missions/create",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        name,
                        description
                    })
                }
            );

        const data =
            await res.json();

        if (!data.success) {

            alert(
                data.message ||
                "Error"
            );

            return;

        }

        alert(
            `Mission created: ${name}`
        );

        createMissionModal
            .classList
            .remove("open");

        document.getElementById(
            "missionName"
        ).value = "";

        document.getElementById(
            "missionDescription"
        ).value = "";

    } catch(err) {

        console.error(err);

        alert(
            "Unable to create mission"
        );

    }

}



async function openMissionPlanning() {
    missionPlanningModal
        .classList
        .add("open");
    await loadCurrentMission();
    await loadMissions();
}



async function loadMissions() {

    const container =
        document.getElementById(
            "missionsList"
        );

    container.innerHTML =
        "Loading...";

    try {
        const res =
            await fetch(
                "/api/missions"
            );

        const missions =
            await res.json();

        container.innerHTML = "";

        for (const mission of missions) {
            const card =
                document.createElement(
                    "div"
                );

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

                <div
                    class="mission-actions">
                    <button
                        class="btn-mission"
                        onclick="selectMission('${mission.id}')">
                        Select
                    </button>
                    <button
                        class="btn-delete-mission"
                        onclick="deleteMission('${mission.id}')">
                        Delete
                    </button>
                </div>

            `;

            container.appendChild(
                card
            );

        }

    } catch(err) {
        console.error(err);
        container.innerHTML =
            "Unable to load missions";

    }

}


async function selectMission(missionId) {
    try {
        const res =
            await fetch(
                "/api/missions/select",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        mission_id: missionId
                    })
                }
            );

        const data =
            await res.json();
        if (!data.success) {
            alert(
                "Unable to select mission"
            );
            return;
        }
        await loadCurrentMission();
        await loadMissions();

    } catch(err) {
        console.error(err);
        alert(
            "Mission selection error"
        );
    }
}


async function loadCurrentMission() {
    const box =
        document.getElementById(
            "currentMissionInfo"
        );
    try {
        const res =
            await fetch(
                "/api/missions/current"
            );
        const mission =
            await res.json();
        if (!mission) {
            box.innerHTML =
                "No active mission";
            return;
        }

        box.innerHTML = `
            <div class="mission-current-card">
                <b>Current Mission:</b><br>
                ${mission.name}<br>
                <span>
                    ${mission.description || ""}
                </span>
                <button
                    id="importGeoJsonBtn"
                    class="btn-mission">

                    Import GeoJSON

                </button>

                <input
                    type="file"
                    id="geojsonFileInput"
                    accept=".geojson,.json"
                    style="display:none">
                <hr>

                <h3>Imported Layers</h3>

                <div id="missionLayers">

                    Loading...

                </div>
            </div>
        `;

        const importBtn =
            document.getElementById(
                "importGeoJsonBtn"
            );

        if (importBtn) {

            importBtn.addEventListener(
                "click",
                () => {

                    document
                        .getElementById(
                            "geojsonFileInput"
                        )
                        .click();

                }
            );

        }

        const fileInput =
            document.getElementById(
                "geojsonFileInput"
            );

        if (fileInput) {

            fileInput.addEventListener(
                "change",
                async (event) => {

                    const file =
                        event.target.files[0];

                    if (!file) {
                        return;
                    }

                    await uploadGeoJson(
                        mission.id,
                        file
                    );

                    await loadMissionLayers(
                        mission.id
                    );

                }
            );

        }

        await loadMissionLayers(
            mission.id
        );

    } catch(err) {
        console.error(err);
        box.innerHTML =
            "Unable to load current mission";
    }
}

async function deleteMission(
    missionId
) {
    if (
        !confirm(
            "Delete mission?"
        )
    ) {
        return;
    }

    try {
        const res =
            await fetch(
                `/api/missions/${missionId}`,
                {
                    method: "DELETE"
                }
            );

        const data =
            await res.json();

        if (!data.success) {
            alert(
                "Unable to delete mission"
            );
            return;
        }

        await loadCurrentMission();
        await loadMissions();

    } catch(err) {

        console.error(err);
        alert(
            "Delete failed"
        );
    }

}


async function uploadGeoJson(
    missionId,
    file
) {
    try {

        const formData =
            new FormData();
        formData.append(
            "mission_id",
            missionId
        );
        formData.append(
            "file",
            file
        );

        const res =
            await fetch(
                "/api/missions/import-geojson",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await res.json();

        if (!data.success) {
            alert(
                "Import failed"
            );
            return;
        }

        alert(
            `Imported ${data.file.filename}`
        );

    } catch(err) {
        console.error(err);
        alert(
            "Upload error"
        );
    }
}


async function loadMissionLayers(
    missionId
) {

    const container =
        document.getElementById(
            "missionLayers"
        );

    if (!container) {
        return;
    }
    try {
        const res =
            await fetch(
                `/api/missions/${missionId}/layers`
            );

        const layers =
            await res.json();

        if (!layers.length) {

            container.innerHTML =
                "No imported layers";
            return;
        }

        container.innerHTML = "";

        layers.forEach(layer => {
            const row =
                document.createElement(
                    "div"
                );

            row.className =
                "layer-item";

            row.innerHTML = `

                <span>
                    📄 ${layer}
                </span>

                <button
                    class="btn-mission"
                    onclick="
                        toggleLayer(
                            '${missionId}',
                            '${layer}'
                        )
                    ">

                    ${
                        missionGeoJsonLayers[
                            `${missionId}_${layer}`
                        ]
                            ? "Hide"
                            : "Show"
                    }

                </button>

            `;

            container.appendChild(
                row
            );

        });

    } catch(err) {
        console.error(err);
        container.innerHTML =
            "Unable to load layers";
    }
}


async function toggleLayer(
    missionId,
    filename
) {
    const key =
        `${missionId}_${filename}`;
    if (
        missionGeoJsonLayers[key]
    ) {
        window.airNodeMap.removeLayer(
            missionGeoJsonLayers[key]
        );
        delete missionGeoJsonLayers[key];
        await loadMissionLayers(
            missionId
        );
        return;
    }

    try {
        const res =
            await fetch(
                `/api/missions/${missionId}/layers/${filename}`
            );

        const geojson =
            await res.json();

        const layer =
            L.geoJSON(
                geojson,
                {
                    style: {
                        color: "#ff0000",
                        weight: 3,
                        fillOpacity: 0.15
                    }
                }
            ).addTo(
                window.airNodeMap
            );

        missionGeoJsonLayers[key] =
            layer;

        try {
            window.airNodeMap.fitBounds(
                layer.getBounds()
            );
        } catch(e) {}
        await loadMissionLayers(
            missionId
        );

    } catch(err) {
        console.error(err);
        alert(
            "Unable to load GeoJSON"
        );
    }
}