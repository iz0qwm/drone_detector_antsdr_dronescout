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