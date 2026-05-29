document.addEventListener("DOMContentLoaded", () => {
    const drawer = document.getElementById("drawer");
    const toggle = document.getElementById("drawerToggle");

    toggle.addEventListener("click", () => {
        drawer.classList.toggle("open");
    });

    const scanBtn = document.getElementById("scanWifiBtn");
    const disconnectBtn = document.getElementById("disconnectWifiBtn");

    if (scanBtn) {
        scanBtn.addEventListener("click", scanWifiNetworks);
    }

    if (disconnectBtn) {
        disconnectBtn.addEventListener("click", disconnectWifi);
    }

    const saveLanBtn = document.getElementById("saveLanBtn");

    if (saveLanBtn) {
        saveLanBtn.addEventListener("click", saveLanConfig);
    }

    const networkSettingsToggle =
    document.getElementById("networkSettingsToggle");

    const networkSettingsPanel =
        document.getElementById("networkSettingsPanel");

    if (networkSettingsToggle && networkSettingsPanel) {

        networkSettingsToggle.addEventListener("click", () => {

            networkSettingsPanel.classList.toggle("open");

        });

    }


    const mapsModal =
        document.getElementById("mapsModal");

    const openMapsManager =
        document.getElementById("openMapsManager");

    const closeMapsModal =
        document.getElementById("closeMapsModal");


    if (openMapsManager && mapsModal) {
        openMapsManager.addEventListener("click", () => {
            mapsModal.classList.add("open");
            loadMaps();
        });
    }

    if (closeMapsModal && mapsModal) {
        closeMapsModal.addEventListener("click", () => {
            mapsModal.classList.remove("open");
        });
    }

    const downloadMapBtn =
        document.getElementById("downloadMapBtn");

    const downloadMapModal =
        document.getElementById("downloadMapModal");

    const closeDownloadMapModal =
        document.getElementById("closeDownloadMapModal");


    if (downloadMapBtn) {
        downloadMapBtn.addEventListener("click", () => {
            downloadMapModal.classList.add("open");
            setTimeout(initDownloadMapPreview, 100);
        });
    }

    if (closeDownloadMapModal) {
        closeDownloadMapModal.addEventListener("click", () => {
            downloadMapModal.classList.remove("open");
        });
    }

    const estimateMapBtn =
        document.getElementById("estimateMapBtn");
    if (estimateMapBtn) {
        estimateMapBtn.addEventListener("click", estimateMapSize);
    }

    const useMapCenterBtn =
        document.getElementById(
            "useMapCenterBtn"
        );

    if (useMapCenterBtn) {

        useMapCenterBtn.addEventListener(
            "click",
            useMapCenter
        );

    }

    const radiusInput =
        document.getElementById(
            "downloadRadius"
        );

    if (radiusInput) {

        radiusInput.addEventListener(
            "input",
            updateDownloadArea
        );

    }

    const modeRadios =
        document.querySelectorAll(
            'input[name="downloadMode"]'
        );

    modeRadios.forEach(radio => {

        radio.addEventListener(
            "change",
            updateDownloadMode
        );

    });

    const gotoCoordinatesBtn =
        document.getElementById(
            "gotoCoordinatesBtn"
        );

    if (gotoCoordinatesBtn) {
        gotoCoordinatesBtn.addEventListener(
            "click",
            gotoCoordinates
        );

    }

    const startDownloadBtn =
    document.getElementById(
        "startDownloadBtn"
    );

    const summaryModal =
        document.getElementById(
            "downloadSummaryModal"
        );

    const closeSummaryModal =
        document.getElementById(
            "closeSummaryModal"
        );

    if (startDownloadBtn) {
        startDownloadBtn.addEventListener(
            "click",
            showDownloadSummary
        );
    }

    if (closeSummaryModal) {
        closeSummaryModal.addEventListener(
            "click",
            () => {
                summaryModal.classList.remove(
                    "open"
                );
            }
        );
    }


    loadLanConfig();
});

async function scanWifiNetworks() {
    const box = document.getElementById("wifiNetworks");
    box.innerHTML = "";
    box.textContent = "Scansione reti WiFi...";

    try {
        const res = await fetch("/api/wifi-scan");
        const networks = await res.json();

        if (!Array.isArray(networks)) {
            box.innerHTML = `<span class="error-text">${networks.error || "Errore scansione WiFi"}</span>`;
            return;
        }

        if (networks.length === 0) {
            box.innerHTML = "Nessuna rete trovata.";
            return;
        }

        box.innerHTML = "";

        networks.forEach(net => {
            const row = document.createElement("div");
            row.className = "wifi-row";

            const secure = net.security && net.security.trim() !== "";

            const savedLabel = net.saved
            ? `<span class="saved-network">Saved</span>`
            : "";

            row.innerHTML = `
                <div class="wifi-info">
                    <b>${escapeHtml(net.ssid)}</b> ${savedLabel}<br>
                    Segnale: ${net.signal}% — ${secure ? net.security : "Open"}
                </div>
                <button class="wifi-connect-btn">Connetti</button>
            `;

            row.querySelector("button").addEventListener("click", () => {
                connectWifi(net.ssid, secure, net.saved);
            });

            box.appendChild(row);
        });

    } catch (err) {
        console.error(err);
        box.innerHTML = `<span class="error-text">Errore durante la scansione WiFi</span>`;
    }
}

async function connectWifi(ssid, secure, saved) {
    const buttons = document.querySelectorAll(".wifi-connect-btn");
    buttons.forEach(btn => btn.disabled = true);

    let password = "";

    if (secure && !saved) {
        password = prompt(`Password per "${ssid}"`);
        if (password === null) return;
    }

    try {
        const res = await fetch("/api/wifi/connect", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                ssid,
                password
            })
        });

        let data;

        try {
            data = await res.json();
        } catch {
            throw new Error("Invalid API response");
        }

        alert(data.message || (data.success ? "Connessione avviata" : "Errore connessione"));

        if (typeof loadNetworkStatus === "function") {
            setTimeout(() => {
                loadNetworkStatus();
            }, 3000);
        }

    } catch (err) {
        console.error(err);
        alert("Errore durante la connessione WiFi");
    } finally {
        const buttons = document.querySelectorAll(".wifi-connect-btn");
        buttons.forEach(btn => btn.disabled = false);
    }
}

async function disconnectWifi() {
    try {
        const res = await fetch("/api/wifi/disconnect", {
            method: "POST"
        });

        let data;

        try {
            data = await res.json();
        } catch {
            throw new Error("Invalid API response");
        }

        alert(data.message || (data.success ? "WiFi disconnesso" : "Errore disconnessione"));

        if (typeof loadNetworkStatus === "function") {
            setTimeout(() => {
                loadNetworkStatus();
            }, 3000);
        }

    } catch (err) {
        console.error(err);
        alert("Errore durante la disconnessione WiFi");
    }
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


async function saveLanConfig() {

    const ip = document.getElementById("lanIp").value;
    const mask = document.getElementById("lanMask").value;
    const gateway = document.getElementById("lanGateway").value;

    try {

        const res = await fetch("/api/lan/config", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                ip,
                mask,
                gateway
            })
        });

        const data = await res.json();

        alert(data.message);

    } catch(err) {
        console.error(err);
        alert("LAN configuration error");
    }
}

async function loadLanConfig() {

    try {

        const res = await fetch("/api/lan/config");
        const data = await res.json();

        if (!data.success) {
            return;
        }

        document.getElementById("lanIp").value =
            data.ip || "";

        document.getElementById("lanMask").value =
            data.mask || "255.255.255.0";

        document.getElementById("lanGateway").value =
            data.gateway || "";

    } catch(err) {

        console.error("LAN config load error", err);

    }

}


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
                <b>${map.name}</b><br>
                Size:
                ${map.size_mb} MB<br><br>
                ${
                    map.protected
                    ? "<span class='saved-network'>System Map</span>"
                    : `<button
                        class="delete-map-btn"
                        data-map="${map.name}">
                        Delete
                    </button>`
                }
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

        });
    } catch(err) {
        console.error(err);
    }
}


function estimateMapSize() {

    const radius =
        parseFloat(
            document.getElementById("downloadRadius").value
        );
    const zoom =
        parseInt(
            document.getElementById("downloadZoom").value
        );
    let estimate =
        radius * radius * (zoom - 8);
    estimate =
        Math.round(estimate);

    document.getElementById(
        "downloadEstimate"
    ).innerHTML = `
        Estimated size:
        ${estimate} MB
    `;
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


function gotoCoordinates() {
    const lat =
        parseFloat(
            document.getElementById(
                "downloadLat"
            ).value
        );
    const lon =
        parseFloat(
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

    const zoom =
        document.getElementById(
            "downloadZoom"
        ).value;

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

        <b>Zoom</b><br>

        ${zoom}<br><br>

        <b>${estimate}</b>

    `;

    document.getElementById(
        "downloadSummaryModal"
    ).classList.add("open");

}

