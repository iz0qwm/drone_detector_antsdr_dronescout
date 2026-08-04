/**
 * Proximity Controller — polls backend API for proximity state.
 * Does NOT calculate distances or states independently.
 */
window.PROXIMITY = window.PROXIMITY || {};

(function () {
    let _timer = null;
    let _map = null;
    let _pollInterval = 5000;
    let _lastData = null;
    let _lastSuccessTime = 0;
    let _maxStaleDisplayMs = 30000;
    let _pendingRequest = false;

    PROXIMITY.start = function (map) {
        if (_timer) return;
        _map = map;

        // Create proximity pane
        if (!map.getPane("traffic-proximity")) {
            map.createPane("traffic-proximity");
            map.getPane("traffic-proximity").style.zIndex = 670;
        }

        console.log("[PROXIMITY] Controller started");
        _poll();
        _timer = setInterval(_poll, _pollInterval);
    };

    PROXIMITY.stop = function () {
        if (_timer) {
            clearInterval(_timer);
            _timer = null;
        }
        _pendingRequest = false;
        PROXIMITY_LAYER.clear();
        PROXIMITY_PANEL.hide();
        _lastData = null;
        console.log("[PROXIMITY] Controller stopped");
    };

    async function _poll() {
        if (_pendingRequest) return;
        _pendingRequest = true;

        try {
            const res = await fetch("/api/proximity/status");
            if (!res.ok) {
                console.warn("[PROXIMITY] API error", res.status);
                _checkStaleDisplay();
                return;
            }

            const data = await res.json();
            _lastData = data;
            _lastSuccessTime = Date.now();

            if (!data.enabled) {
                PROXIMITY_LAYER.clear();
                PROXIMITY_PANEL.hide();
                return;
            }

            const pairs = data.pairs || [];

            if (pairs.length === 0 || pairs.every(p => p.state === "NORMAL")) {
                PROXIMITY_LAYER.clear();
                PROXIMITY_PANEL.hide();
            } else {
                PROXIMITY_LAYER.update(pairs, _map);
                PROXIMITY_PANEL.update(pairs, data);
            }

        } catch (err) {
            console.warn("[PROXIMITY] Poll error:", err.message);
            _checkStaleDisplay();
        } finally {
            _pendingRequest = false;
        }
    }

    function _checkStaleDisplay() {
        if (_lastSuccessTime && (Date.now() - _lastSuccessTime) > _maxStaleDisplayMs) {
            PROXIMITY_LAYER.clear();
            PROXIMITY_PANEL.hide();
            _lastData = null;
        }
    }
})();
