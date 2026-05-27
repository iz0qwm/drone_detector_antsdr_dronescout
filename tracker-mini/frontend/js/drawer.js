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

    const startApBtn = document.getElementById("startApBtn");
    const stopApBtn = document.getElementById("stopApBtn");

    if (startApBtn) {
        startApBtn.addEventListener("click", startHotspot);
    }

    if (stopApBtn) {
        stopApBtn.addEventListener("click", stopHotspot);
    }


    const clientModeBtn = document.getElementById("clientModeBtn");
    const fieldModeBtn = document.getElementById("fieldModeBtn");

    if (clientModeBtn) {
        clientModeBtn.addEventListener("click", setClientMode);
    }

    if (fieldModeBtn) {
        fieldModeBtn.addEventListener("click", setFieldMode);
    }

    loadApStatus();
    loadModeStatus();
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

            row.innerHTML = `
                <div class="wifi-info">
                    <b>${escapeHtml(net.ssid)}</b><br>
                    Segnale: ${net.signal}% — ${secure ? net.security : "Open"}
                </div>
                <button class="wifi-connect-btn">Connetti</button>
            `;

            row.querySelector("button").addEventListener("click", () => {
                connectWifi(net.ssid, secure);
            });

            box.appendChild(row);
        });

    } catch (err) {
        console.error(err);
        box.innerHTML = `<span class="error-text">Errore durante la scansione WiFi</span>`;
    }
}

async function connectWifi(ssid, secure) {
    const buttons = document.querySelectorAll(".wifi-connect-btn");
    buttons.forEach(btn => btn.disabled = true);

    let password = "";

    if (secure) {
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

async function loadApStatus() {
    const box = document.getElementById("apStatus");

    try {
        const res = await fetch("/api/ap/status");
        const data = await res.json();

        if (data.active) {
            box.innerHTML = `
                <b>Status:</b> ACTIVE<br>
                <b>SSID:</b> Portable-Air-Node<br>
                <b>Password:</b> tracker123
            `;
        } else {
            box.innerHTML = `
                <b>Status:</b> OFF
            `;
        }

    } catch (err) {
        console.error(err);
        box.innerHTML = "AP status unavailable";
    }
}

async function startHotspot() {
    try {
        const res = await fetch("/api/ap/start", {
            method: "POST"
        });

        const data = await res.json();

        alert(data.message);

        setTimeout(() => {
            loadApStatus();

            if (typeof loadNetworkStatus === "function") {
                loadNetworkStatus();
            }
        }, 3000);

    } catch (err) {
        console.error(err);
        alert("Hotspot start failed");
    }
}

async function stopHotspot() {
    try {
        const res = await fetch("/api/ap/stop", {
            method: "POST"
        });

        const data = await res.json();

        alert(data.message);

        setTimeout(() => {
            loadApStatus();

            if (typeof loadNetworkStatus === "function") {
                loadNetworkStatus();
            }
        }, 3000);

    } catch (err) {
        console.error(err);
        alert("Hotspot stop failed");
    }
}



async function loadModeStatus() {
    const box = document.getElementById("modeStatus");

    try {
        const res = await fetch("/api/mode/status");
        const data = await res.json();

        if (data.mode === "FIELD") {
            box.innerHTML = `
                <b>Mode:</b> FIELD<br>
                Tactical standalone mode
            `;
        }
        else if (data.mode === "CLIENT") {
            box.innerHTML = `
                <b>Mode:</b> INFRASTRUCTURE<br>
                WiFi / Ethernet uplink mode
            `;
        }
        else if (data.mode === "TRANSITION") {
            box.innerHTML = `
                <b>Mode:</b> TRANSITION<br>
                Waiting for infrastructure network...
            `;
        }
        else {
            box.innerHTML = `
                <b>Mode:</b> UNKNOWN
            `;
        }

    } catch (err) {
        console.error(err);
        box.innerHTML = "Mode unavailable";
    }
}

async function setFieldMode() {
    if (!confirm("Switch to FIELD mode? WiFi uplink will be disconnected.")) {
        return;
    }

    try {
        const res = await fetch("/api/mode/field", {
            method: "POST"
        });

        const data = await res.json();

        alert(data.message);

        setTimeout(() => {
            loadModeStatus();
            loadApStatus();

            if (typeof loadNetworkStatus === "function") {
                loadNetworkStatus();
            }
        }, 3000);

    } catch (err) {
        console.error(err);
        alert("Failed to switch mode");
    }
}

async function setClientMode() {
    if (!confirm("Switch to INFRASTRUCTURE mode? Hotspot will remain active until another network becomes available.")) {
        return;
    }

    try {
        const res = await fetch("/api/mode/client", {
            method: "POST"
        });

        const data = await res.json();

        alert(data.message);

        setTimeout(() => {
            loadModeStatus();
            loadApStatus();

            if (typeof loadNetworkStatus === "function") {
                loadNetworkStatus();
            }
        }, 3000);

    } catch (err) {
        console.error(err);
        alert("Failed to switch mode");
    }
}

