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

        openMapsManager.addEventListener(
            "click",
            async () => {

                try {

                    const res =
                        await fetch("/api/network");

                    const net =
                        await res.json();

                    if (!net.internet) {

                        alert(
                            "To use the map download function Internet access is required.\n\nGo to Network Settings."
                        );

                        return;
                    }

                    mapsModal.classList.add("open");

                    loadMaps();
                    loadMapProvider();
                    loadDownloads();

                    if (
                        window.downloadRefreshTimer
                    ) {

                        clearInterval(
                            window.downloadRefreshTimer
                        );

                    }

                    window.downloadRefreshTimer =
                        setInterval(
                            loadDownloads,
                            2000
                        );

                } catch (err) {

                    alert(
                        "Unable to verify Internet connectivity."
                    );

                }

            }
        );

    }

    if (closeMapsModal && mapsModal) {
        closeMapsModal.addEventListener("click", () => {
            mapsModal.classList.remove("open");

            if (
                window.downloadRefreshTimer
            ) {

                clearInterval(
                    window.downloadRefreshTimer
                );

            }
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

    const confirmDownloadBtn =
        document.getElementById(
            "confirmDownloadBtn"
        );

    if (confirmDownloadBtn) {

        confirmDownloadBtn.addEventListener(
            "click",
            startMapDownload
        );

    }


    // Drawer group toggle logic
    document
        .querySelectorAll(".drawer-group-btn")
        .forEach(btn => {
            btn.addEventListener("click", () => {
                const targetId =
                    btn.dataset.target;
                const target =
                    document.getElementById(
                        targetId
                    );
                const isOpen =
                    target.classList.contains(
                        "open"
                    );
                document
                    .querySelectorAll(
                        ".drawer-group-panel"
                    )
                    .forEach(panel => {

                        panel.classList.remove(
                            "open"
                        );

                    });

                document
                    .querySelectorAll(
                        ".drawer-group-btn"
                    )
                    .forEach(button => {

                        button.classList.remove(
                            "open"
                        );

                    });
                if (!isOpen) {
                    target.classList.add(
                        "open"
                    );
                    btn.classList.add(
                        "open"
                    );
                }
            });
        });


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


