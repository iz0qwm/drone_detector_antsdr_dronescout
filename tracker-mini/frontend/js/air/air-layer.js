// js/air/air-layer.js

window.AIR = window.AIR || {};
const AIRCRAFT_TTL_MS = 30000; // 30 secondi
const MAX_TRAIL_POINTS = 10;
const trailsByIcao = new Map();
const MAX_MISSES = 2;
const STALE_GRACE_MS = 60000; // 60 secondi (radar-style)
const MAX_ALT_DEFAULT = 1000;
const MIN_AIRCRAFT_SPEED_MS = 5 / 3.6; // 5 km/h → m/s
const FADE_WINDOW_MS = 10000; // ultimi 10 secondi
const MOVE_THRESHOLD = 0.00002; // ~10 metri
const HEADING_THRESHOLD = 2;   // gradi

let airLayer = null;
const markersByIcao = new Map();

function computeOpacity(ac) {
  if (!ac.expiresAt) return 1;

  const now = Date.now();
  const remaining = ac.expiresAt - now;

  if (remaining <= 0) return 0.25;

  if (remaining > FADE_WINDOW_MS) return 1;

  const ratio = remaining / FADE_WINDOW_MS;

  // fade da 1 → 0.25
  return 0.25 + (0.75 * ratio);
}

AIR.createAirLayer = function (map) {
  if (airLayer) return airLayer;
  airLayer = L.layerGroup();
  return airLayer;
};

AIR.clearAirLayer = function () {
  if (!airLayer) return;
  airLayer.clearLayers();
  markersByIcao.clear();
};

AIR.updateAirLayer = function (aircraftList) {
    
  if (!airLayer) return;

  const seen = new Set();
  const seenInFeed = new Set();


  aircraftList.forEach(ac => {
    seenInFeed.add(ac.icao);
    if (isValidAircraft(ac)) {
        if (computeOpacity(ac) < 0.3) return;
        seen.add(ac.icao);

        let marker = markersByIcao.get(ac.icao);
        if (!marker) {
            marker = createMarker(ac);
            marker._misses = 0;
            markersByIcao.set(ac.icao, marker);
            marker.addTo(airLayer);
            } else {
                const oldLatLng = marker.getLatLng();

                if (
                  Math.abs(oldLatLng.lat - ac.lat) > MOVE_THRESHOLD ||
                  Math.abs(oldLatLng.lng - ac.lon) > MOVE_THRESHOLD
                ) {
                  marker.setLatLng([ac.lat, ac.lon]);
                }
                const el = marker.getElement();

                if (el) {
                    const opacity = computeOpacity(ac);

                    el.style.opacity = opacity.toString();
                    el.style.filter = opacity < 0.5 ? "grayscale(0.5)" : "none";

                    const newHeading =
                        ac.heading ?? 0;

                    if (
                        marker._heading === undefined ||
                        Math.abs(
                            marker._heading -
                            newHeading
                        ) > HEADING_THRESHOLD
                    ) {

                        marker.setRotationAngle(
                            newHeading
                        );

                        marker._heading =
                            newHeading;
                    }
                }
                marker.setPopupContent(popup(ac));
                marker._misses = 0;
                marker._stale = false;

            }


        // ===== TRAIL (semplice stile drone) =====
        let trail = trailsByIcao.get(ac.icao);
        if (!trail) {
          trail = {
            points: [],
            segments: []
          };
          trailsByIcao.set(ac.icao, trail);
        }

        const coords = [ac.lat, ac.lon];

        // aggiungi sempre (come nel tuo sistema funzionante)
        trail.points.push(coords);

        if (trail.points.length > MAX_TRAIL_POINTS) {
          trail.points.shift();
        }

        // rimuovi vecchi segmenti
        trail.segments.forEach(seg => {
          if (seg && seg.polyline) {
            airLayer.removeLayer(seg.polyline);
          }
        });
        trail.segments = [];

        // ridisegna tutto con fade semplice
        for (let i = 1; i < trail.points.length; i++) {
          const opacity = i / trail.points.length;

          const segment = L.polyline(
            [trail.points[i - 1], trail.points[i]],
            {
              color: "#ff3b30",
              weight: 3,
              opacity: opacity * 1,
              pane: 'traffic-air'
            }
          ).addTo(airLayer);

          trail.segments.push(segment);
        }
        if (trail.points.length >= MAX_TRAIL_POINTS) {
          trail.points.shift();
        }
        

    }
  });

    // ===== RIMOZIONE COME FACEVI PRIMA =====
    for (const [icao, marker] of markersByIcao.entries()) {
        if (!seenInFeed.has(icao)) {
            marker._misses = (marker._misses || 0) + 1;

            /* console.warn(
            "[ADS-B] miss",
            icao,
            marker._misses + "/" + MAX_MISSES
            ); */

            if (marker._misses >= MAX_MISSES) {
              marker._stale = true;
              marker._staleSince = Date.now();

              // 👇 RIMUOVI SUBITO IL TRAIL
              const trail = trailsByIcao.get(icao);
              if (trail) {
                trail.segments.forEach(seg => {
                  if (seg) airLayer.removeLayer(seg);
                });
                trailsByIcao.delete(icao);
              }

              // effetto visivo marker
              const el = marker.getElement();
              if (el) {
                el.style.opacity = "0.25";
                el.style.filter = "grayscale(1)";
                el.style.transition = "opacity 0.5s linear";
              }
            }
            if (marker._stale) {
                const age = Date.now() - (marker._staleSince || 0);

                if (age > STALE_GRACE_MS) {
                    airLayer.removeLayer(marker);
                    markersByIcao.delete(icao);

                    const trail = trailsByIcao.get(icao);
                    if (trail) {
                      trail.segments.forEach(seg => {
                        if (seg) airLayer.removeLayer(seg);
                      });
                      trailsByIcao.delete(icao);
                    }
                }
            }

            continue;
        }

    }
}

    /* console.warn(
        "[ADS-B] layer size:",
        airLayer.getLayers().length
    ); */



function isValidAircraft(ac) {

  if (!Number.isFinite(ac.lat) || !Number.isFinite(ac.lon)) {
    return false;
  }

  // elicotteri: SEMPRE
  if (ac.isHelicopter) return true;

  // ==============================
  // FILTRO AEREI FERMI (PISTE)
  // ==============================
  if (Number.isFinite(ac.speed)) {
    if (ac.speed <= MIN_AIRCRAFT_SPEED_MS) {
      console.warn(
        "[ADS-B] filtered ground aircraft",
        ac.icao,
        Math.round(ac.speed * 3.6),
        "km/h"
      );
      return false;
    }
  }

  const showAllAircraft =
      document.getElementById(
          "showHighAltitudeAircraft"
      )?.checked === true;

  if (showAllAircraft) {
      return true;
  }

  const maxAlt = Number.isFinite(
      AIR.maxAltitudeMeters
  )
      ? AIR.maxAltitudeMeters
      : MAX_ALT_DEFAULT;

  // quota mancante → accetti
  if (
      ac.altitude == null ||
      !Number.isFinite(ac.altitude)
  ) {
      return true;
  }

  const ok = ac.altitude <= maxAlt;

  if (!ok) {
      console.warn(
          "[ADS-B] filtered by altitude",
          ac.icao,
          ac.altitude,
          ">",
          maxAlt
      );
  }

  return ok;
}



function getAdsbIcon(ac) {

  if (ac.isHelicopter) {
    return "/icons/helicopter.png";
  }

  switch (ac.category) {

    case "A1":
      return "/icons/plane_light.png";

    case "A2":
      return "/icons/plane_light_1.png";

    case "A3":
      return "/icons/plane_medium.png";

    case "A5":
      return "/icons/plane_heavy.png";

    default:
      return "/icons/plane_unknown.png";
  }
}

function createMarker(ac) {

  const size = ac.isHelicopter ? 32 : 32;

  const icon = L.icon({
    iconUrl: getAdsbIcon(ac),
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -10]
  });

  const marker = L.marker(
      [ac.lat, ac.lon],
      {
        icon,
        pane: "traffic-air",
        rotationAngle: ac.heading || 0
      }
  );

  marker.bindPopup(
    popup(ac)
  );

  return marker;
}



function getSourceLabel(source) {

  switch (source) {

    case "LOCAL_ADSB":
      return "📡 RTL-SDR";

    case "SOLARMONITOR":
      return "🌐 SolarMonitor";

    case "OPENSKY":
      return "🌐 OpenSky";

    case "OGN":
      return "🪂 OGN";

    default:
      return source || "Unknown";
  }
}


function popup(ac) {

  const source =
  getSourceLabel(ac.source);


  return `
    <b>${ac.callsign}</b><br>
    ICAO: ${ac.icao}<br>
    Fonte: ${source}<br>
    Quota: ${Math.round(ac.altitude)} m<br>
    Velocità raw: ${ac.speed}<br>
    Velocità: ${
      Number.isFinite(ac.speed)
        ? Math.round(ac.speed * 3.6) + " km/h"
        : "N/D"
    }
  `;
}
