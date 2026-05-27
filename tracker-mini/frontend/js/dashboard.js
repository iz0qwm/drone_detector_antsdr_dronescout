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

if (typeof L !== "undefined") {
    map = L.map('map').setView([41.9028, 12.4964], 10);

    L.tileLayer(
        'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        {
            attribution: '&copy; OpenStreetMap'
        }
    ).addTo(map);
}
else {
    console.error("Leaflet not loaded");
}


async function loadNetworkStatus() {
    try {
        const res = await fetch('/api/network');
        const data = await res.json();

        const eth = data.ethernet;
        const wifi = data.wifi;

        document.getElementById('networkStatus').innerHTML = `
            <b>Ethernet:</b> ${eth.connected ? 'Connected' : 'Disconnected'}<br>
            <b>IP:</b> ${eth.ip || '---'}<br><br>

            <b>WiFi:</b> ${wifi.connected ? 'Connected' : 'Disconnected'}<br>
            <b>SSID:</b> ${wifi.ssid || '---'}<br>
            <b>IP:</b> ${wifi.ip || '---'}<br><br>

            <b>Internet:</b> ${data.internet ? 'YES' : 'NO'}<br>
        `;

    } catch (err) {
        console.error(err);

        document.getElementById('networkStatus').innerHTML =
            '<span style="color:red;">Network info unavailable</span>';
    }
}


loadStatus();
loadNetworkStatus();

setInterval(() => {
    loadStatus();
    loadNetworkStatus();
}, 5000);