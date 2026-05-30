async function loadMaps() {

    try {
        const storageRes =
            await fetch("/api/maps/storage");
        const storage =
            await storageRes.json();
        let color = "#4caf50";
        if (storage.free_gb < 10)
            color = "#ff9800";
        if (storage.free_gb < 5)
            color = "#f44336";

        const usedPercent =
            Math.round(
                (storage.used_gb / storage.total_gb) * 100
            );

        document.getElementById("mapsStorageInfo").innerHTML = `
        <div class="storage-card">
            <div class="storage-header">
                <span>Storage</span>
                <span>${storage.free_gb} GB Free</span>
            </div>
            <div class="storage-bar">
                <div
                    class="storage-bar-fill"
                    style="
                        width:${usedPercent}%;
                        background:${color};
                    ">
                </div>
            </div>
            <div class="storage-footer">
                Used:
                ${storage.used_gb} GB
                (${usedPercent}%)
            </div>
        </div>
        `;

        const res =
            await fetch("/api/maps");
        const maps =
            await res.json();
        const container =
            document.getElementById("mapsList");
        container.innerHTML = "";
        maps.forEach(map => {
            const row =
                document.createElement("div");
            row.className = "wifi-row";
            row.innerHTML = `
            <div class="map-card">

                <div class="map-title">
                    🗺 ${map.name}
                </div>

                <div class="map-description">
                    ${map.description || "No description"}
                </div>

                <div class="map-meta">

                    ${map.size_mb} MB

                    <br>

                    Radius:
                    ${map.radius_km || "-" } km

                    <br>

                    Zoom:
                    ${map.min_zoom || "-"}
                    -
                    ${map.max_zoom || "-"}

                </div>

                <div class="map-actions">
                    <label>

                        <input
                            type="checkbox"
                            class="active-map-checkbox"
                            data-map="${map.name}"
                            ${map.active ? "checked" : ""}

                            ${map.protected
                                ? "disabled checked"
                                : ""
                            }
                        >

                        Active

                    </label>
                    <button
                        class="show-map-btn"
                        ${!map.active ? "disabled" : ""}
                        data-lat="${map.center_lat}"
                        data-lon="${map.center_lon}"
                    >
                        📍 Show
                    </button>
                    <button
                        class="edit-map-btn"
                        data-map="${map.name}">
                        ✏ Edit
                    </button>

                    ${
                        map.protected
                        ? `<span class="saved-network">
                            System Map
                        </span>`
                        : `<button
                            class="delete-map-btn"
                            data-map="${map.name}">
                            🗑 Delete
                        </button>`
                    }

                </div>

            </div>
            `;
            container.appendChild(row);
            const btn =
                row.querySelector(".delete-map-btn");
            if (btn) {
                btn.addEventListener("click", async () => {
                    if (
                        !confirm(
                            `Delete ${map.name}?`
                        )
                    ) {
                        return;
                    }
                    await fetch(
                        `/api/maps/${encodeURIComponent(map.name)}`,
                        {
                            method: "DELETE"
                        }
                    );
                    loadMaps();
                });
            }

            const editBtn =
                row.querySelector(
                    ".edit-map-btn"
                );

            if (editBtn) {

                editBtn.addEventListener(
                    "click",
                    async () => {

                        const newDescription =
                            prompt(
                                "Map description",
                                map.description || ""
                            );

                        if (
                            newDescription === null
                        ) {
                            return;
                        }

                        if (
                            newDescription.trim() ===
                            (map.description || "")
                        ) {
                            return;
                        }
                        await fetch(
                            "/api/maps/update-description",
                            {
                                method: "POST",
                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },
                                body: JSON.stringify({
                                    name: map.name,
                                    description:
                                        newDescription
                                })
                            }
                        );

                        loadMaps();

                    }
                );

            }

            const activeCheckbox =
                row.querySelector(
                    ".active-map-checkbox"
                );

            if (activeCheckbox) {

                activeCheckbox.addEventListener(
                    "change",
                    async () => {

                        await fetch(
                            "/api/maps/set-active",
                            {
                                method: "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                body: JSON.stringify({
                                    name: map.name,
                                    active:
                                        activeCheckbox.checked
                                })
                            }
                        );

                        loadMaps();
                    }
                );
            }

            const showBtn =
                row.querySelector(
                    ".show-map-btn"
                );

            if (showBtn) {

                showBtn.addEventListener(
                    "click",
                    () => {

                        mapsModal.classList.remove(
                            "open"
                        );

                        window.airNodeMap.setView(
                            [
                                parseFloat(
                                    map.center_lat
                                ),
                                parseFloat(
                                    map.center_lon
                                )
                            ],
                            parseInt(
                                map.max_zoom
                            )
                        );

                    }
                );
            }
        });

        loadDownloads();
    } catch(err) {
        console.error(err);
    }
}


function estimateMapSize() {

    const radius =
        parseFloat(
            document.getElementById(
                "downloadRadius"
            ).value
        );

    let estimate;

    if (radius <= 5)
        estimate = 15;

    else if (radius <= 10)
        estimate = 40;

    else if (radius <= 25)
        estimate = 120;

    else if (radius <= 50)
        estimate = 350;

    else if (radius <= 100)
        estimate = 900;

    else
        estimate = 1800;

    document.getElementById(
        "downloadEstimate"
    ).innerHTML =
        `Estimated size: ~${estimate} MB`;
}

function useMapCenter() {
    if (!window.airNodeMap) {
        alert(
            "Map not available"
        );

        return;
    }

    const center =
        window.airNodeMap.getCenter();

    document.getElementById(
        "downloadLat"
    ).value =
        center.lat.toFixed(6);

    document.getElementById(
        "downloadLon"
    ).value =
        center.lng.toFixed(6);
}

let downloadCircle = null;
let downloadMarker = null;
let downloadPreviewMap = null;

function initDownloadMapPreview() {
    if (downloadPreviewMap) {
        downloadPreviewMap.invalidateSize();
        return;
    }

    downloadPreviewMap = L.map(
        "downloadMapPreview"
    );

    downloadPreviewMap.setView(
        [41.9028, 12.4964],
        8
    );

    L.tileLayer(
        'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
    ).addTo(downloadPreviewMap);

    downloadMarker = L.marker(
        downloadPreviewMap.getCenter()
    ).addTo(downloadPreviewMap);

    downloadCircle = L.circle(
        downloadPreviewMap.getCenter(),
        {
            radius: 20000,
            color: "#00ff00",
            fillOpacity: 0.15
        }
    ).addTo(downloadPreviewMap);

    downloadPreviewMap.on(
        "move",
        updateDownloadArea
    );

    downloadPreviewMap.on(
        "move",
        updateCenterInfo
    );
    updateCenterInfo();
}


function updateDownloadArea() {

    if (!downloadPreviewMap)
        return;

    const center =
        downloadPreviewMap.getCenter();

    const radiusKm =
        parseFloat(
            document.getElementById(
                "downloadRadius"
            ).value
        ) || 20;

    downloadMarker.setLatLng(center);

    downloadCircle.setLatLng(center);

    downloadCircle.setRadius(
        radiusKm * 1000
    );
    updateCenterInfo();
}

function updateDownloadMode() {
    const mode =
        document.querySelector(
            'input[name="downloadMode"]:checked'
        ).value;

    const mapPanel =
        document.getElementById(
            "mapModePanel"
        );

    const coordsPanel =
        document.getElementById(
            "coordsModePanel"
        );

    const info =
        document.getElementById(
            "downloadModeInfo"
        );
    if (mode === "map") {
        mapPanel.style.display = "block";
        coordsPanel.style.display = "none";
        info.textContent = "Download center defined by map position";
        if (downloadPreviewMap) {
            downloadPreviewMap.dragging.enable();
            downloadPreviewMap.scrollWheelZoom.enable();
            downloadPreviewMap.doubleClickZoom.enable();
            downloadPreviewMap.touchZoom.enable();
        }
        if (downloadPreviewMap) {
            setTimeout(
                () => downloadPreviewMap.invalidateSize(),
                100
            );
        }

    } else {
        if (downloadPreviewMap) {
            downloadPreviewMap.dragging.disable();
            downloadPreviewMap.scrollWheelZoom.disable();
            downloadPreviewMap.doubleClickZoom.disable();
            downloadPreviewMap.touchZoom.disable();
        }
        mapPanel.style.display = "none";
        coordsPanel.style.display = "block";
        info.textContent = "Download center defined by coordinates";
    }

}

function parseCoordinate(coord) {

    coord = coord.trim().toUpperCase();

    const decimal = parseFloat(coord);

    if (!isNaN(decimal) &&
        /^[-+]?\d+(\.\d+)?$/.test(coord)) {
        return decimal;
    }

    const cleaned = coord
        .replace(/°/g, " ")
        .replace(/'/g, " ")
        .replace(/"/g, " ")
        .replace(/,/g, ".");

    const parts = cleaned
        .split(/\s+/)
        .filter(Boolean);

    let sign = 1;

    if (
        coord.includes("S") ||
        coord.includes("W")
    ) {
        sign = -1;
    }

    const nums = parts
        .filter(p =>
            !["N","S","E","W"].includes(p)
        )
        .map(Number);

    if (nums.length === 3) {

        return sign * (
            nums[0] +
            nums[1] / 60 +
            nums[2] / 3600
        );

    }

    if (nums.length === 2) {

        return sign * (
            nums[0] +
            nums[1] / 60
        );

    }

    return NaN;
}

function gotoCoordinates() {
    const lat = parseCoordinate(
        document.getElementById(
            "downloadLat"
        ).value
    );

    const lon = parseCoordinate(
        document.getElementById(
            "downloadLon"
        ).value
    );
    if (
        isNaN(lat) ||
        isNaN(lon)
    ) {
        alert(
            "Invalid coordinates"
        );
        return;
    }
    if (!downloadPreviewMap)
        return;

    downloadPreviewMap.setView(
        [lat, lon],
        12
    );

    updateDownloadArea();
}

function updateCenterInfo() {

    if (!downloadPreviewMap)
        return;

    const center =
        downloadPreviewMap.getCenter();

    document.getElementById(
        "downloadCenterInfo"
    ).innerHTML = `

        Lat:
        ${center.lat.toFixed(6)}
        <br>

        Lon:
        ${center.lng.toFixed(6)}

    `;

}

function showDownloadSummary() {

    const mode =
        document.querySelector(
            'input[name="downloadMode"]:checked'
        ).value;

    let lat;
    let lon;

    if (mode === "coords") {
        lat =
            document.getElementById(
                "downloadLat"
            ).value;

        lon =
            document.getElementById(
                "downloadLon"
            ).value;
    } else {
        const center =
            downloadPreviewMap.getCenter();

        lat =
            center.lat.toFixed(6);

        lon =
            center.lng.toFixed(6);
    }


    const radius =
        document.getElementById(
            "downloadRadius"
        ).value;

    const zoomRange =
        getZoomRange(
            parseFloat(radius)
        );

    const estimate =
        document.getElementById(
            "downloadEstimate"
        ).textContent;

    document.getElementById(
        "downloadSummaryContent"
    ).innerHTML = `

        <b>Center</b><br>

        Lat:
        ${lat}<br>

        Lon:
        ${lon}<br><br>

        <b>Radius</b><br>

        ${radius} km<br><br>

        <b>Zoom Range</b><br>

        ${zoomRange}<br><br>

        <b>${estimate}</b>

    `;

    document.getElementById(
        "downloadSummaryModal"
    ).classList.add("open");

}

async function startMapDownload() {
    const mode =
        document.querySelector(
            'input[name="downloadMode"]:checked'
        ).value;
    let lat;
    let lon;
    if (mode === "coords") {
        lat = parseCoordinate(
            document.getElementById(
                "downloadLat"
            ).value
        );
        lon = parseCoordinate(
            document.getElementById(
                "downloadLon"
            ).value
        );
    } else {
        const center =
            downloadPreviewMap.getCenter();
        lat = center.lat;
        lon = center.lng;
    }
    const radius =
        parseFloat(
            document.getElementById(
                "downloadRadius"
            ).value
        );
    const description =
        document.getElementById(
            "downloadDescription"
        ).value.trim();
    try {
        const res =
            await fetch(
                "/api/maps/download",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        lat,
                        lon,
                        radius,
                        description
                    })
                }
            );

        const data =
            await res.json();
        alert(
            "Download started"
        );
        pollDownload(
            data.job_id
        );
    } catch(err) {
        console.error(err);
        alert(
            "Unable to start download"
        );
    }
}

async function pollDownload(jobId) {
    const timer =
        setInterval(
            async () => {
                try {
                    const res =
                        await fetch(
                            `/api/maps/download-status/${jobId}`
                        );
                    const data =
                        await res.json();

                    console.log(data);
                    if (
                        data.status === "completed"
                    ) {
                        clearInterval(timer);
                        alert(
                           "Download completed"
                        );
                        loadMaps();
                    }
                    if (
                        data.status === "error"
                    ) {
                        clearInterval(timer);
                        alert(
                            data.message ||
                            "Download error"
                        );
                    }
                } catch(err) {
                    console.error(err);
                }
            },
            2000
        );
}



async function loadMapProvider() {

    const res =
        await fetch(
            "/api/maps/provider"
        );

    const data =
        await res.json();

    const input =
        document.getElementById(
            "thunderforestApiKey"
        );

    if (data.configured) {

        input.value =
            "Configured";

    } else {

        input.value = "";

    }

    input.readOnly = true;
}


async function saveMapProvider() {

    const apiKey =
        document.getElementById(
            "thunderforestApiKey"
        ).value;

    await fetch(
        "/api/maps/provider",
        {
            method: "POST",

            headers: {
                "Content-Type":
                "application/json"
            },

            body: JSON.stringify({
                provider:
                    "thunderforest",

                api_key:
                    apiKey
            })
        }
    );

    document
        .getElementById(
            "thunderforestApiKey"
        )
        .readOnly = true;

    alert(
        "API Key saved"
    );

    loadMapProvider();
}


function getZoomRange(radiusKm) {

    if (radiusKm <= 5)
        return "12-16";

    if (radiusKm <= 10)
        return "11-15";

    if (radiusKm <= 25)
        return "10-14";

    if (radiusKm <= 50)
        return "9-13";

    if (radiusKm <= 100)
        return "8-12";

    return "7-11";
}



document
.getElementById(
    "editApiKeyBtn"
)
.addEventListener(
    "click",
    () => {

        const input =
            document.getElementById(
                "thunderforestApiKey"
            );

        input.readOnly = false;

        input.value = "";

        input.focus();
    }
);

document
.getElementById(
    "saveApiKeyBtn"
)
.addEventListener(
    "click",
    saveMapProvider
);


async function loadDownloads() {

    try {

        const res =
            await fetch(
                "/api/maps/downloads"
            );

        const downloads =
            await res.json();

        const container =
            document.getElementById(
                "downloadsList"
            );

        container.innerHTML = "";

        const ids =
            Object.keys(downloads);

        let visibleJobs = 0;

        ids.forEach(id => {

            const job =
                downloads[id];

            if (
                job.status === "completed"
            ) {

                if (!completedJobs[id]) {

                    completedJobs[id] =
                        Date.now();

                }

                const age =
                    Date.now() -
                    completedJobs[id];

                if (age > 5000) {

                    return;

                }

            }

            visibleJobs++;

            const progress =
                job.progress || 0;

            const card =
                document.createElement("div");

            card.className =
                "download-card";

            card.innerHTML = `

                <div>
                    <b>Status:</b>
                    ${job.status}
                </div>

                <div>
                    ${job.message || ""}
                </div>

                <div class="download-bar">

                    <div
                        class="download-bar-fill"
                        style="
                            width:${progress}%;
                        ">
                    </div>

                </div>

                <div>

                    ${progress}%<br>

                    Tile:
                    ${job.current_tile || 0}
                    /
                    ${job.total_tiles || 0}

                </div>

            `;

            container.appendChild(card);

        });

        if (visibleJobs === 0) {

            container.innerHTML =
                "No active downloads";

        }

    } catch(err) {

        console.error(err);

    }

}