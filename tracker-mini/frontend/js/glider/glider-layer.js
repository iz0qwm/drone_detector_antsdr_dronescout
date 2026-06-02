// js/glider/glider-layer.js
(function () {

  let _map = null;
  let _layer = L.layerGroup();
  let _markers = new Map();
 
  window.GLIDER_LAYER = {

    enable(map) {
      _map = map;
      _layer.addTo(map);
    },

    disable() {
      if (_map) _map.removeLayer(_layer);
      _markers.clear();
    },

    upsert(glider) {
      const { id, lat, lon, heading, source } = glider;
      const src = (source || "").toUpperCase();

      let color =
        src === "SAFESKY"    ? "#FFD600" :
        src === "FLARM"      ? "#00E5FF" :
        src === "FREEFLIGHT" ? "#00FF88" :
        src === "FANET"      ? "#FF6D00" :
        "#FFFFFF";

      let icon;

      if (src === "FLARM") {
        icon = window.GLIDER_ICONS.getIcon(heading, color); // ✈️ nuova
      } else if (src === "SAFESKY") {
        icon = window.GLIDER_ICONS.getAircraftIcon(heading, color); // già ok
      } else {
        icon = window.GLIDER_ICONS.getFreeFlightIcon(heading, color); // 🪂
      }

      let marker = _markers.get(id);

      if (!marker) {
        marker = L.marker([lat, lon], {
          icon: icon,
          pane: 'traffic-glider'
        });
        marker.bindPopup(gliderPopup(glider));
        marker.addTo(_layer);
        _markers.set(id, marker);
      } else {
        marker.setLatLng([lat, lon]);
        // 🔥 aggiorna icona con nuovo heading
        marker.setIcon(icon);
        marker.setPopupContent(gliderPopup(glider));
      }
    },

    remove(id) {
      const marker = _markers.get(id);
      if (!marker) return;

      _layer.removeLayer(marker);
      _markers.delete(id);
    }
  };

  function gliderPopup(g) {
    const ageSec = Math.max(0, Math.round(Date.now() / 1000 - g.ts));

    return `
        <b>Traffico ${g.source}</b><br>
        ID: ${g.id}<br>
        Quota: ${g.alt_m != null ? Math.round(g.alt_m) + " m" : "n/d"}<br>
        Aggiornato: ${ageSec}s fa
    `;
    }

})();
