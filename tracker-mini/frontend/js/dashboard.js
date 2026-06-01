async function loadStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        document.getElementById('hostname').textContent = data.hostname;
        document.getElementById('cpu').textContent = `CPU ${data.cpu}%`;
        document.getElementById('ram').textContent = `RAM ${data.ram}%`;
        document.getElementById('disk').textContent = `DISK ${data.disk}%`;

    } catch (err) {
        console.error(err);
    }
}


// -----------------------------
// Intelligent Map Source Selection
// -----------------------------
let map = null;
let currentTileLayer = null;

async function getSelectedMapSource() {

    const saved =
        localStorage.getItem(
            "mapSource"
        );

    return saved || "auto";
}

async function resolveMapSource() {

    const selected =
        await getSelectedMapSource();

    if (selected !== "auto") {
        return selected;
    }

    try {

        const res =
            await fetch("/api/network");

        const net =
            await res.json();

        if (net.internet) {
            return "online_topo";
        }

        return "offline";

    } catch(err) {

        return "offline";

    }
}

function getTileLayerConfig(source) {

    if (source === "online_topo") {

        return {
            url: "https://tile.opentopomap.org/{z}/{x}/{y}.png",
            options: {
                attribution: "&copy; OpenTopoMap contributors",
                maxZoom: 17
            },
            label: "Online Topo"
        };

    }

    return {
        url: "/tiles/{z}/{x}/{y}.png",
        options: {
            attribution: "&copy; Offline MBTiles"
        },
        label: "Offline Maps"
    };
}

async function applyMapSource() {

    if (!map) {
        return;
    }

    const source =
        await resolveMapSource();

    const config =
        getTileLayerConfig(source);

    if (currentTileLayer) {
        map.removeLayer(
            currentTileLayer
        );
    }

    currentTileLayer =
        L.tileLayer(
            config.url,
            config.options
        ).addTo(map);

    const status =
        document.getElementById(
            "mapSourceStatus"
        );

    if (status) {
        status.textContent =
            `Current source: ${config.label}`;
    }
}

async function initMap() {

    try {

        const res =
            await fetch('/api/settings');

        const settings =
            await res.json();

        const mapConfig =
            settings.map;

        if (typeof L === "undefined") {
            console.error("Leaflet not loaded");
            return;
        }

        map = L.map('map').setView(
            [
                mapConfig.default_lat,
                mapConfig.default_lon
            ],
            mapConfig.default_zoom
        );

        window.airNodeMap = map;

        await applyMapSource();

    } catch(err) {

        console.error("Map init error", err);

    }
}
// -----------------------------



async function loadNetworkStatus() {
    try {
        const res = await fetch('/api/network');
        const data = await res.json();

        const adminLan = data.admin_lan;
        const userLan = data.user_lan;
        const wifi = data.wifi;

        let wifiLabel = "Disconnected";
        let wifiExtra = "";

        if (wifi.connected) {

            if (wifi.ssid === "Portable-Air-Node") {

                wifiLabel = "Access Point ACTIVE";
                wifiExtra = `
                    <b>SSID:</b> Portable-Air-Node<br>
                `;

            } else {

                wifiLabel = "Connected";

                wifiExtra = `
                    <b>SSID:</b> ${wifi.ssid || "---"}<br>
                `;
            }
        }

        document.getElementById('networkStatus').innerHTML = `

            <b>Admin LAN:</b>
            ${adminLan.connected ? 'Connected' : 'Disconnected'}<br>

            <b>IP:</b>
            192.168.1.115<br><br>

            <b>User LAN:</b>
            ${userLan.connected ? 'Connected' : 'Disconnected'}<br>

            <b>IP:</b>
            ${userLan.ip || '---'}<br><br>

            <b>WiFi:</b>
            ${wifiLabel}<br>

            ${wifiExtra}

            <b>IP:</b>
            ${wifi.ip || '---'}<br><br>

            <b>Internet:</b>
            ${data.internet ? 'YES' : 'NO'}

        `;

    } catch (err) {
        console.error(err);

        document.getElementById('networkStatus').innerHTML =
            '<span style="color:red;">Network info unavailable</span>';
    }
}


loadStatus();
loadNetworkStatus();
initMap();

// Handle map source selection UI
document.addEventListener(
    "DOMContentLoaded",
    () => {

        const select =
            document.getElementById(
                "mapSourceSelect"
            );

        if (!select) {
            return;
        }

        select.value =
            localStorage.getItem(
                "mapSource"
            ) || "auto";

        select.addEventListener(
            "change",
            async () => {

                localStorage.setItem(
                    "mapSource",
                    select.value
                );

                await applyMapSource();

            }
        );

    }
);



// Periodic refresh of status and network info every 5 seconds
setInterval(async () => {

    loadStatus();
    loadNetworkStatus();

    const selected =
        localStorage.getItem(
            "mapSource"
        ) || "auto";

    if (selected === "auto") {
        await applyMapSource();
    }

}, 5000);