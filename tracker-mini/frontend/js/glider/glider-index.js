// js/glider/glider-index.js
(function () {

  window.GLIDER = {

    start(map) {
      if (!window.GLIDER_LAYER || !window.GLIDER_DATA) {
        console.warn("[GLIDER] layer or data not ready");
        return;
      }

      window.GLIDER_LAYER.enable(map);
      window.GLIDER_DATA.start(map);
    },

    stop() {
      window.GLIDER_DATA?.stop();
      window.GLIDER_LAYER?.disable();
    }
  };

})();
