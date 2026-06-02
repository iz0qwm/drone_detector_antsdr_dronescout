// js/glider/glider-utils.js
(function () {

  window.GLIDER_UTILS = {
    normalizeHeading(h) {
      return ((h % 360) + 360) % 360;
    }
  };

})();
