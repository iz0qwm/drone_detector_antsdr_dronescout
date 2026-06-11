async function loadStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        
        const systemBox =
            document.getElementById(
                "systemStatus"
            );

        if (systemBox) {

            systemBox.innerHTML = `
                <b>Hostname:</b>
                ${data.hostname}<br>

                <b>CPU:</b>
                ${data.cpu}%<br>

                <b>RAM:</b>
                ${data.ram}%<br>

                <b>DISK:</b>
                ${data.disk}%<br>
            `;
        }


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

        map.createPane(
            "traffic-air"
        );

        map.getPane(
            "traffic-air"
        ).style.zIndex = 650;

        map.createPane(
            "traffic-glider"
        );

        map.getPane(
            "traffic-glider"
        ).style.zIndex = 655;

        map.createPane(
            "traffic-drone"
        );

        map.getPane(
            "traffic-drone"
        ).style.zIndex = 660;


        window.airNodeMap = map;

        // gestione click su mappa per posizione Marker tracker
        map.on("click", (e) => {

            if (!trackerSelectionMode) {
                return;
            }

            const lat = e.latlng.lat;
            const lon = e.latlng.lng;

            document.getElementById(
                "dscLat"
            ).value = lat.toFixed(6);

            document.getElementById(
                "dscLon"
            ).value = lon.toFixed(6);

            updateTrackerMarker(
                lat,
                lon,
                document.getElementById(
                    "dscNodeName"
                ).value
            );

            trackerSelectionMode = false;

            alert(
                "Tracker position selected"
            );

        });

        if (
            window.AIR &&
            AIR.startAirTraffic
        ) {
            AIR.startAirTraffic(
                map,
                {
                    maxAltitudeMeters: 1000
                }
            );
        }

        if (
            window.DRONES &&
            DRONES.startDroneTraffic &&
            localStorage.getItem(
                "droneNetworkEnabled"
            ) !== "false"
        ) {
            DRONES.startDroneTraffic(
                map
            );
        }

        if (
            window.GLIDER &&
            localStorage.getItem(
                "ognNetworkEnabled"
            ) !== "false"
        ) {
            window.GLIDER.start(
                map
            );
        }

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
        const wifiAp = data.wifi_ap;
        const wifiClient = data.wifi_client;

        
        document.getElementById('networkStatus').innerHTML = `

            <b>Admin LAN:</b>
            ${adminLan.connected ? 'Connected' : 'Disconnected'}<br>

            <b>IP:</b>
            192.168.1.115<br><br>

            <b>User LAN:</b>
            ${userLan.connected ? 'Connected' : 'Disconnected'}<br>

            <b>IP:</b>
            ${userLan.ip || '---'}<br><br>

            <b>Access Point:</b>
            ${wifiAp.connected ? 'ACTIVE' : 'OFF'}<br>

            <b>SSID:</b>
            ${wifiAp.ssid || '---'}<br>

            <b>IP:</b>
            ${wifiAp.ip || '---'}<br><br>

            <b>WiFi Client:</b>
            ${wifiClient.connected ? 'Connected' : 'Disconnected'}<br>

            <b>SSID:</b>
            ${wifiClient.ssid || '---'}<br>

            <b>IP:</b>
            ${wifiClient.ip || '---'}<br><br>

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
initTrafficSettings();
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

function setLed(id, state) {

    const led =
        document.getElementById(id);

    if (!led) {
        return;
    }

    led.className = "led";

    switch(state) {

        case "green":
            led.classList.add(
                "led-green"
            );
            break;

        case "red":
            led.classList.add(
                "led-red"
            );
            break;

        case "orange":
            led.classList.add(
                "led-orange"
            );
            break;

        default:
            led.classList.add(
                "led-off"
            );
    }
}


async function loadServices() {

    try {

        const res =
            await fetch(
                "/api/services"
            );

        const data =
            await res.json();

        setLed(
            "ledNet",
            data.internet
                ? "green"
                : "red"
        );

        setLed(
            "ledAdsLocal",
            data.ads_local
                ? "green"
                : "off"
        );

        const adsbEnabled =
            localStorage.getItem(
                "adsbNetworkEnabled"
            ) !== "false";

        setLed(
            "ledAdsNet",
            (
                data.ads_network &&
                adsbEnabled
            )
                ? "green"
                : "red"
        );

        const droneEnabled =
            localStorage.getItem(
                "droneNetworkEnabled"
            ) !== "false";

        setLed(
            "ledRid",
            (
                data.remote_id &&
                droneEnabled
            )
                ? "green"
                : "red"
        );

        const ognEnabled =
            localStorage.getItem(
                "ognNetworkEnabled"
            ) !== "false";

        setLed(
            "ledOgn",
            (
                data.ogn &&
                ognEnabled
            )
                ? "green"
                : "red"
        );

        setLed(
            "ledDsc",
            data.dsc
                ? "green"
                : "off"
        );

        const servicesBox =
            document.getElementById(
                "servicesStatus"
            );

        if (servicesBox) {
            servicesBox.innerHTML = `

                <div class="services-grid">

                    <div class="service-item">
                        <span class="mini-led ${
                            data.internet
                                ? "mini-led-green"
                                : "mini-led-red"
                        }"></span>
                        NET
                    </div>

                    <div class="service-item">
                        <span class="mini-led ${
                            data.ads_local
                                ? "mini-led-green"
                                : "mini-led-red"
                        }"></span>
                        ADSB Rx
                    </div>

                    <div class="service-item">
                        <span class="mini-led ${
                            (
                                data.ads_network &&
                                adsbEnabled
                            )
                                ? "mini-led-green"
                                : "mini-led-red"
                        }"></span>
                        ADSB Net
                    </div>

                    <div class="service-item">
                        <span class="mini-led ${
                            data.remote_id
                                ? "mini-led-green"
                                : "mini-led-red"
                        }"></span>
                        RID
                    </div>

                    <div class="service-item">
                        <span class="mini-led ${
                            (
                                data.ogn &&
                                ognEnabled
                            )
                                ? "mini-led-green"
                                : "mini-led-red"
                        }"></span>
                        OGN
                    </div>

                    <div class="service-item">
                        <span class="mini-led ${
                            data.dsc
                                ? "mini-led-green"
                                : "mini-led-red"
                        }"></span>
                        DSC
                    </div>

                </div>
            `;
        }

    } catch(err) {

        console.error(
            "Services error",
            err
        );

    }
}


function initTrafficSettings() {

    // ADS-B Network Toggle
    const checkbox =
        document.getElementById(
            "adsbNetworkEnabled"
        );

    if (!checkbox) {
        return;
    }

    const saved =
        localStorage.getItem(
            "adsbNetworkEnabled"
        );

    checkbox.checked =
        saved !== "false";

    checkbox.addEventListener(
        "change",
        () => {

            localStorage.setItem(
                "adsbNetworkEnabled",
                checkbox.checked
            );

            if (
                !checkbox.checked &&
                window.AIR &&
                AIR.clearAirLayer
            ) {

                AIR.clearAirLayer();

            }

            loadServices();

            console.log(
                "[TRAFFIC]",
                "ADS-B Network:",
                checkbox.checked
            );

        }
    );

    // OGN Network Toggle
    const ognCheckbox =
        document.getElementById(
            "ognNetworkEnabled"
        );

    if (ognCheckbox) {

        const savedOgn =
            localStorage.getItem(
                "ognNetworkEnabled"
            );

        ognCheckbox.checked =
            savedOgn !== "false";

        ognCheckbox.addEventListener(
            "change",
            () => {

                localStorage.setItem(
                    "ognNetworkEnabled",
                    ognCheckbox.checked
                );

                if (ognCheckbox.checked) {

                    if (
                        window.GLIDER &&
                        window.airNodeMap
                    ) {
                        window.GLIDER.start(
                            window.airNodeMap
                        );
                    }

                } else {

                    if (window.GLIDER) {
                        window.GLIDER.stop();
                    }

                }

                loadServices();

            }
        );
    }


    // Drone Network Toggle

    const droneCheckbox =
        document.getElementById(
            "droneNetworkEnabled"
        );

    if (droneCheckbox) {

        const savedDrone =
            localStorage.getItem(
                "droneNetworkEnabled"
            );

        droneCheckbox.checked =
            savedDrone !== "false";

        droneCheckbox.addEventListener(
            "change",
            () => {

                localStorage.setItem(
                    "droneNetworkEnabled",
                    droneCheckbox.checked
                );

                if (!droneCheckbox.checked) {

                    if (
                        window.DRONES &&
                        DRONES.clearDroneLayer
                    ) {
                        DRONES.clearDroneLayer();
                    }

                } else {

                    if (
                        window.DRONES &&
                        window.airNodeMap
                    ) {
                        DRONES.startDroneTraffic(
                            window.airNodeMap
                        );
                    }

                }

                loadServices();

            }
        );
    }


}


let trackerMarker = null;
let trackerSelectionMode = false;

function updateTrackerMarker(lat, lon, name) {

    if (!window.airNodeMap) {
        return;
    }

    const icon = L.icon({
        iconUrl: "icons/receiver.png",
        iconSize: [32, 32],
        iconAnchor: [16, 16]
    });

    if (!trackerMarker) {

        trackerMarker = L.marker(
            [lat, lon],
            { icon }
        ).addTo(window.airNodeMap);

    } else {

        trackerMarker.setLatLng(
            [lat, lon]
        );

    }

    trackerMarker.bindPopup(`
        <b>${name || "DSC Node"}</b><br>
        DSC Tracker
    `);
}


// Periodic refresh of status and network info every 5 seconds
setInterval(async () => {

    loadStatus();
    loadServices();
    loadNetworkStatus();


    const selected =
        localStorage.getItem(
            "mapSource"
        ) || "auto";

    if (selected === "auto") {
        await applyMapSource();
    }

}, 5000);


