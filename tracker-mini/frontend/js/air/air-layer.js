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
                    const svg = el.querySelector("svg");
                    const opacity = computeOpacity(ac);

                    el.style.opacity = opacity.toString();
                    el.style.filter = opacity < 0.5 ? "grayscale(0.5)" : "none";

                    if (svg) {
                        svg.style.transition = "transform 0.5s linear";
                        const newHeading = ac.heading ?? 0;

                        if (
                          marker._heading === undefined ||
                          Math.abs(marker._heading - newHeading) > HEADING_THRESHOLD
                        ) {
                          svg.style.transform = `rotate(${newHeading}deg)`;
                          marker._heading = newHeading;
                        }
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

  const maxAlt = Number.isFinite(AIR.maxAltitudeMeters)
    ? AIR.maxAltitudeMeters
    : MAX_ALT_DEFAULT;

  // quota mancante → accetti
  if (ac.altitude == null || !Number.isFinite(ac.altitude)) {
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



function createAircraftIcon(ac) {
  const isHelicopter = ac.isHelicopter === true;

  const size = isHelicopter ? 28 : 30;
  const color = isHelicopter ? "#FFFF00" : "#FF00FF";
  const heading = ac.heading ?? 0;

  const pathHelicopter = `
    M 0 -10 L 2 -5 L 4 0 L 4 5 L 2 5 L 0 5
    L -2 5 L -4 5 L -4 0 L -2 -5 Z
    M -11 -11 L -10 -12 L 11 6 L 10 7 Z
    M 9 -12 L 10 -11 L -10 7 L -11 6 Z
    M -2 5 L -2 14 L 2 14 L 2 16
    L 3 16 L 3 12 L 2 12 L 2 5
  `;

  const pathAirplane = `
    M -10.0035 -1.9825 L -10.0052 -2.9825
    L -0.007 -4 L -0.0157 -9
    L 0.9808 -11.0017 L 3.9808 -11.007
    L 4.9843 -9.0087 L 4.993 -4.0087
    L 14.9947 -3.0262 L 14.9965 -2.0262
    L 5 -0.0087 L 5.007 3.9913
    L 8.0105 5.986 L 8.014 7.986
    L 5.0105 5.9913 L 5.0174 9.9913
    L 0.0175 10 L 0.0105 6
    L -2.986 8.0052 L -2.9895 6.0052
    L 0.007 4 L 0 0 Z
  `;

  const svg = `
    <svg width="${size}" height="${size}"
         viewBox="-16 -16 32 32"
         style="
           transform: rotate(${heading}deg);
           transform-origin: 50% 50%;
         ">
      <path d="${isHelicopter ? pathHelicopter : pathAirplane}"
            fill="${color}"
            stroke="white"
            stroke-width="1"/>
    </svg>
  `;

  return L.divIcon({
    html: svg,
    className: "adsb-aircraft-icon",
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2]
  });
}


function createMarker(ac) {

  const size = ac.isHelicopter ? 28 : 30;
  const half = size / 2;

  const svgPathHeli = `
    M 0 -10 L 2 -5 L 4 0 L 4 5 L 2 5 L 0 5
    L -2 5 L -4 5 L -4 0 L -2 -5 Z
    M -11 -11 L -10 -12 L 11 6 L 10 7 Z
    M 9 -12 L 10 -11 L -10 7 L -11 6 Z
    M -2 5 L -2 14 L 2 14 L 2 16
    L 3 16 L 3 12 L 2 12 L 2 5
  `;

  const svgPathPlane = `
    M -10.0035 -1.9825 L -10.0052 -2.9825
    L -0.007 -4 L -0.0157 -9
    L 0.9808 -11.0017 L 3.9808 -11.007
    L 4.9843 -9.0087 L 4.993 -4.0087
    L 14.9947 -3.0262 L 14.9965 -2.0262
    L 5 -0.0087 L 5.007 3.9913
    L 8.0105 5.986 L 8.014 7.986
    L 5.0105 5.9913 L 5.0174 9.9913
    L 0.0175 10 L 0.0105 6
    L -2.986 8.0052 L -2.9895 6.0052
    L 0.007 4 L 0 0 Z
  `;

  const path = ac.isHelicopter ? svgPathHeli : svgPathPlane;
  const fill = ac.isHelicopter ? "#ffff00" : "#ff00ff";

  const icon = L.divIcon({
    className: "adsb-svg-marker",
    iconSize: [size, size],
    iconAnchor: [half, half],
    html: `
      <svg
        width="${size}"
        height="${size}"
        viewBox="-16 -16 32 32"
        style="
          transform: rotate(${ac.heading || 0}deg);
          transform-origin: 50% 50%;
        "
      >
        <path d="${path}"
          fill="${fill}"
          stroke="white"
          stroke-width="1.2"
        />
      </svg>
    `
  });

  const marker = L.marker([ac.lat, ac.lon], {
    icon,
    pane: 'traffic-air'
  });

  marker.bindPopup(popup(ac));
  return marker;
}



function popup(ac) {
  return `
    <b>${ac.callsign}</b><br>
    ICAO: ${ac.icao}<br>
    Quota: ${Math.round(ac.altitude)} m<br>
    Velocità raw: ${ac.speed}<br>
    Velocità: ${
      Number.isFinite(ac.speed)
        ? Math.round(ac.speed * 3.6) + " km/h"
        : "N/D"
    }

  `;
}
