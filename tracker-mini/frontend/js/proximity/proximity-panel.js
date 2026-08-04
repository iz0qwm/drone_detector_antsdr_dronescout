/**
 * Nearby Traffic Panel — shows ranked proximity pairs.
 * Renders backend-provided data without independent calculation.
 */
window.PROXIMITY_PANEL = window.PROXIMITY_PANEL || {};

(function () {
    let _container = null;

    function _getContainer() {
        if (_container) return _container;
        _container = document.getElementById("proximityPanel");
        return _container;
    }

    PROXIMITY_PANEL.update = function (pairs, data) {
        const container = _getContainer();
        if (!container) return;

        // Filter to non-NORMAL pairs only
        const displayPairs = pairs.filter(p => p.state !== "NORMAL");

        if (displayPairs.length === 0) {
            PROXIMITY_PANEL.hide();
            return;
        }

        let html = '<div class="prox-panel-header">NEARBY TRAFFIC</div>';
        html += '<div class="prox-panel-list">';

        for (const pair of displayPairs) {
            const stateClass = `prox-state-${pair.state.toLowerCase()}`;
            const dist = _formatDistance(pair.distance_m);
            const trend = pair.trend || "\u2014";
            const trendClass = trend === "APR" ? "prox-trend-apr" :
                               trend === "DIV" ? "prox-trend-div" : "prox-trend-neutral";
            const src = pair.target_source || "?";
            const age = pair.target_updated_ago_s;
            const ageText = (age != null && age > 10) ? ` ${age}s` : "";

            html += `<div class="prox-panel-entry" data-lat="${pair.target_lat}" data-lon="${pair.target_lon}">`;
            html += `<span class="prox-drone-label">${_truncate(pair.drone_label, 10)}</span>`;
            html += `<span class="prox-arrow">\u2192</span>`;
            html += `<span class="prox-target-label">${_truncate(pair.target_label, 8)}</span>`;
            html += `<span class="prox-distance">${dist}</span>`;
            html += `<span class="prox-state-badge ${stateClass}">${pair.state.substring(0, 3)}</span>`;
            html += `<span class="prox-trend ${trendClass}">${trend}</span>`;
            html += `<span class="prox-source">${src}${ageText}</span>`;
            html += `</div>`;
        }

        html += '</div>';
        html += '<div class="prox-panel-footer">Informational only \u2022 Not certified</div>';

        container.innerHTML = html;
        container.style.display = "block";

        // Click to center map on aircraft
        container.querySelectorAll(".prox-panel-entry").forEach(el => {
            el.addEventListener("click", () => {
                const lat = parseFloat(el.dataset.lat);
                const lon = parseFloat(el.dataset.lon);
                if (window.airNodeMap && isFinite(lat) && isFinite(lon)) {
                    window.airNodeMap.setView([lat, lon], window.airNodeMap.getZoom());
                }
            });
        });
    };

    PROXIMITY_PANEL.hide = function () {
        const container = _getContainer();
        if (container) {
            container.style.display = "none";
            container.innerHTML = "";
        }
    };

    function _formatDistance(m) {
        if (m == null) return "?";
        if (m >= 1000) return (m / 1000).toFixed(1) + " km";
        return Math.round(m) + " m";
    }

    function _truncate(str, maxLen) {
        if (!str) return "?";
        return str.length > maxLen ? str.substring(0, maxLen) + "\u2026" : str;
    }
})();
