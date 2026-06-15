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

    const systemSettingsBtn =
        document.getElementById(
            "systemSettingsBtn"
        );

    const systemSettingsPanel =
        document.getElementById(
            "systemSettingsPanel"
        );

    if (
        systemSettingsBtn &&
        systemSettingsPanel
    ) {

        systemSettingsBtn.addEventListener(
            "click",
            () => {

                systemSettingsPanel
                    .classList
                    .toggle("open");

                loadHardwareStatus();

            }
        );
    }

    const openLogsBtn =
        document.getElementById(
            "openLogsBtn"
        );

    const logsModal =
        document.getElementById(
            "logsModal"
        );

    const closeLogsModal =
        document.getElementById(
            "closeLogsModal"
        );

    if (openLogsBtn) {
        openLogsBtn.addEventListener(
            "click",
            () => {
                logsModal.classList.add(
                    "open"
                );
                loadLogs();
                window.logsRefreshTimer =
                    setInterval(
                        loadLogs,
                        2000
                    );
            }
        );

    }

    if (closeLogsModal) {
        closeLogsModal.addEventListener(
            "click",
            () => {
                logsModal.classList.remove(
                    "open"
                );
                clearInterval(
                    window.logsRefreshTimer
                );
            }
        );

    }


    const updateModal =
    document.getElementById(
        "updateModal"
    );

    const openUpdateBtn =
        document.getElementById(
            "systemUpdateBtn"
        );

    const closeUpdateBtn =
        document.getElementById(
            "closeUpdateModal"
        );

    if (
        openUpdateBtn &&
        updateModal
    ) {

        openUpdateBtn.addEventListener(
            "click",
            () => {

                updateModal.classList.add(
                    "open"
                );
                loadCurrentVersion();
            }
        );
    }

    if (
        closeUpdateBtn &&
        updateModal
    ) {

        closeUpdateBtn.addEventListener(
            "click",
            () => {

                updateModal.classList.remove(
                    "open"
                );

            }
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

    // Initial load log count
    document
        .getElementById(
            "clearLogsBtn"
        )
        ?.addEventListener(
            "click",
            async () => {

                await fetch(
                    "/api/logs/clear",
                    {
                        method: "POST"
                    }
                );

                loadLogs();

            }
        );

    document
        .getElementById("copyLogsBtn")
        ?.addEventListener(
            "click",
            () => {

                const text =
                    document.getElementById(
                        "logsContainer"
                    ).innerText;

                const textarea =
                    document.createElement(
                        "textarea"
                    );

                textarea.value = text;

                textarea.style.position =
                    "fixed";

                textarea.style.left =
                    "-9999px";

                document.body.appendChild(
                    textarea
                );

                textarea.focus();
                textarea.select();

                const success =
                    document.execCommand(
                        "copy"
                    );

                document.body.removeChild(
                    textarea
                );

                if (success) {

                    alert(
                        "Logs copied"
                    );

                } else {

                    alert(
                        "Copy failed"
                    );

                }

            }
        );

    loadLanConfig();
    initDscSettings();
    loadDs110Settings();
    loadSerialPorts();

    document
        .getElementById("saveDs110Btn")
        ?.addEventListener(
            "click",
            saveDs110Settings
        );
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


async function loadHardwareStatus() {

    const box =
        document.getElementById(
            "hardwareStatus"
        );

    if (!box) {
        return;
    }

    try {

        const res =
            await fetch(
                "/api/hardware"
            );

        const data =
            await res.json();

        box.innerHTML = `
            <b>WiFi Client Adapter</b><br>
            ${
                data.wifi_client
                    ? "🟢 Detected"
                    : "🔴 Missing"
            }

            <br><br>

            <b>DS110 RID Receiver</b><br>
            ${
                data.ds110
                    ? "🟢 Connected"
                    : "🔴 Missing"
            }

            <br><br>

            <b>DS110 RID Heartbeat</b><br>
            ${
                data.ds110_alive
                    ? "🟢 Active"
                    : "🔴 No Data"
            }
        `;

    } catch (err) {

        box.innerHTML =
            "Hardware status unavailable";

    }
}


async function loadLogs() {

    try {

        const res =
            await fetch(
                "/api/logs"
            );

        const logs =
            await res.json();

        const box =
            document.getElementById(
                "logsContainer"
            );

        box.innerHTML = logs
            .slice()
            .reverse()
            .map(log =>
                `<div class="log-line">[${log.time}] [${log.level}] [${log.component}] ${log.message}</div>`
            )
            .join("");

    } catch(err) {

        console.error(err);

    }

}


function initDscSettings() {

    const nodeName =
        document.getElementById(
            "dscNodeName"
        );

    const lat =
        document.getElementById(
            "dscLat"
        );

    const lon =
        document.getElementById(
            "dscLon"
        );

    if (!nodeName) {
        return;
    }

    loadDscSettings();

    document
        .getElementById(
            "dscSaveBtn"
        )
        ?.addEventListener(
            "click",
            saveDscSettings
        );

    document
        .getElementById(
            "dscSelectPositionBtn"
        )
        ?.addEventListener(
            "click",
            selectTrackerPosition
        );


}

async function loadDscSettings() {

    try {

        const res =
            await fetch(
                "/api/dsc/settings"
            );

        const data =
            await res.json();

        document.getElementById(
            "dscNodeId"
        ).value =
            data.node_id || "";


        const warning =
            document.getElementById(
                "dscNodeWarning"
            );

        if (warning) {
            warning.style.display = "none";
            if (!data.node_id) {
                warning.style.display = "block";
                warning.innerHTML =
                    "⚠ Node ID not configured";
            } else if (
                !data.node_id.startsWith(
                    "dsc-node"
                )
            ) {
                warning.style.display = "block";
                warning.innerHTML =
                    `⚠ Unusual Node ID: ${data.node_id}`;
            }

        }

        document.getElementById(
            "dscNodeName"
        ).value =
            data.node_name || "DSC Node";

        document.getElementById(
            "dscLat"
        ).value =
            data.lat || "";

        document.getElementById(
            "dscLon"
        ).value =
            data.lon || "";

        document
            .querySelector(
                `input[name="dscPositionSource"][value="${data.position_source || "manual"}"]`
            )
            ?.setAttribute(
                "checked",
                true
            );

        if (
            data.lat &&
            data.lon
        ) {

            setTimeout(() => {

                updateTrackerMarker(
                    parseFloat(data.lat),
                    parseFloat(data.lon),
                    data.node_name
                );

            }, 1000);

        }

    } catch(err) {

        console.error(
            "DSC settings load error",
            err
        );

    }
}


async function saveDscSettings() {

    try {

        const payload = {

            node_name:
                document.getElementById(
                    "dscNodeName"
                ).value,

            lat:
                parseFloat(
                    document.getElementById(
                        "dscLat"
                    ).value
                ),

            lon:
                parseFloat(
                    document.getElementById(
                        "dscLon"
                    ).value
                ),

            position_source:
                document.querySelector(
                    'input[name="dscPositionSource"]:checked'
                ).value
        };

        const res =
            await fetch(
                "/api/dsc/settings",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify(
                        payload
                    )
                }
            );

        if (!res.ok) {

            throw new Error(
                "Save failed"
            );

        }

        alert(
            "DSC settings saved"
        );

    } catch(err) {

        console.error(err);

        alert(
            "Unable to save DSC settings"
        );

    }
}

function selectTrackerPosition() {

    trackerSelectionMode = true;

    alert(
        "Click on the map to select tracker position"
    );

}

async function loadDs110Settings() {

    try {

        const res =
            await fetch(
                "/api/ds110/settings"
            );

        const data =
            await res.json();

        document.getElementById(
            "ds110Interface"
        ).value =
            data.interface || "usb";

        await loadSerialPorts();

        document.getElementById(
            "ds110Device"
        ).value =
            data.device || "/dev/ttyACM0";

        document.getElementById(
            "ds110Baudrate"
        ).value =
            data.baudrate || 115200;

    } catch(err) {

        console.error(
            "DS110 settings load error",
            err
        );

    }

}

async function saveDs110Settings() {

    try {

        const payload = {

            interface:
                document.getElementById(
                    "ds110Interface"
                ).value,

            device:
                document.getElementById(
                    "ds110Device"
                ).value,

            baudrate:
                parseInt(
                    document.getElementById(
                        "ds110Baudrate"
                    ).value
                )
        };

        const res =
            await fetch(
                "/api/ds110/settings",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify(
                        payload
                    )
                }
            );

        if (!res.ok) {
            throw new Error();
        }

        alert(
            "DS110 settings saved"
        );

    } catch(err) {

        console.error(err);

        alert(
            "Unable to save DS110 settings"
        );

    }

}

async function loadSerialPorts() {

    const select =
        document.getElementById(
            "ds110Device"
        );

    if (!select) {
        return;
    }

    const res =
        await fetch(
            "/api/serial/ports"
        );

    const data = await res.json();

    select.innerHTML = "";

    data.ports.forEach(port => {

        const option =
            document.createElement(
                "option"
            );

        option.value = port;
        option.textContent = port;

        if (
            port === data.current
        ) {
            option.selected = true;
        }

        select.appendChild(
            option
        );

    });

}