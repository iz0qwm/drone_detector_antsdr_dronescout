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

let map = null;

async function initMap() {

    try {

        const res = await fetch('/api/settings');
        const settings = await res.json();

        const mapConfig = settings.map;

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

        L.tileLayer(
            '/tiles/{z}/{x}/{y}.png',
            {
                attribution: '&copy; OpenStreetMap'
            }
        ).addTo(map);

        // Expose map globally for Drawer and other modules
        window.airNodeMap = map;

    } catch(err) {

        console.error("Map init error", err);

    }
}


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

setInterval(() => {
    loadStatus();
    loadNetworkStatus();
}, 5000);