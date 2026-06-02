// js/glider/glider-icons.js
window.GLIDER_ICONS = {
  // colore acceso tipo ADS-B
  COLOR: "#00E5FF",   // cyan neon (cambia se vuoi: "#FF00FF" ecc.)
  SIZE: 30,

  getIcon(heading = 0, colorOverride = null) {
    const size = window.GLIDER_ICONS.SIZE;
    const color = colorOverride || window.GLIDER_ICONS.COLOR;

    const svg = `
    <svg
      width="${size}"
      height="${size}"
      viewBox="0 0 512 512"
      style="
        transform: rotate(${heading || 0}deg);
        transform-origin: 50% 50%;
        filter: drop-shadow(0 0 4px ${color});
      "
      xmlns="http://www.w3.org/2000/svg"
    >
      <path fill="${color}" d="M247.989 307.923l.88.88-118.47 118.42c-22.74 22.79-76.09 54.47-76.09 54.47a17.21 17.21 0 0 1-22.18-26.16l181.72-181.71zm231.86-275.77a17.21 17.21 0 0 0-24.33 0l-181.72 181.72 34.1 34.1.88.88 118.42-118.43c22.74-22.74 54.47-76.09 54.47-76.09a17.21 17.21 0 0 0-1.82-22.18zm-52.44 319.24a32.78 32.78 0 0 0-23.25 9.62l-43.17 43.17a32.89 32.89 0 0 0 0 46.51l6 6 89.69-89.68-6-6a32.78 32.78 0 0 0-23.27-9.62zm-46.8 10.55l-18.69 18.69c-40.87-40.64-64.22-62-102.66-84l-39.27-39.32c-64-64-65.14-86.41-57.12-94.44 1.91-1.91 4.76-3.29 9-3.29 12.64 0 37.47 12.43 85.46 60.41l39.29 39.29c21.95 38.47 43.37 61.8 83.99 102.66zm-156.89-162.82c-34.3-29.84-49.85-33.11-53.79-29.17-2.7 2.7-1.91 8.38 2.33 16.9 4.91 9.84 13.88 22.21 26.79 36.94z"/>
    </svg>
    `;

    return L.divIcon({
      html: svg,
      className: "glider-svg-marker",
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2]
    });
  },

  getFreeFlightIcon(heading = 0, colorOverride = null) {
    const size = window.GLIDER_ICONS.SIZE;
    const color = colorOverride || window.GLIDER_ICONS.COLOR;

    // SVG: usiamo il tuo, ma rendiamo stroke "neon" e ruotiamo come AIR
    const svg = `
      <svg
        width="${size}"
        height="${size}"
        viewBox="0 0 60 60"
        style="
          transform: rotate(${heading || 0}deg);
          transform-origin: 50% 50%;
          filter: drop-shadow(0 0 2px rgba(0,0,0,.6));
        "
        xmlns="http://www.w3.org/2000/svg"
      >

        <line stroke="${color}" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" x1="33.9" y1="31.7" x2="36.7" y2="39.5"/>
        <line stroke="${color}" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" x1="25.3" y1="32" x2="22.6" y2="39.5"/>
        <polyline stroke="${color}" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" points="27.7,31.1 6.2,37.1 6.2,30.7 29.5,14.6 52.9,30.7 52.9,37.1 31.8,31.1 "/>
        <ellipse stroke="${color}" fill="none" stroke-width="2.2" cx="29.5" cy="24.6" rx="1.4" ry="1.5"/>
        <path stroke="${color}" fill="none" stroke-width="2.2" d="M29.5,28.8c-2,0-2.1,4.1-1.9,6.4c0.1,1.3,0.8,3.8,1.9,3.8s1.7-2.4,1.9-3.8C31.6,32.9,31.5,28.8,29.5,28.8"/>
        <line stroke="${color}" fill="none" stroke-width="2.2" x1="47.2" y1="26.8" x2="47.2" y2="35.5"/>
        <line stroke="${color}" fill="none" stroke-width="2.2" x1="11.8" y1="27.2" x2="11.8" y2="35.5"/>

      </svg>
    `;

        return L.divIcon({
          html: svg,
          className: "glider-svg-marker",
          iconSize: [size, size],
          iconAnchor: [size / 2, size / 2]
        });
  },


  getAircraftIcon(heading = 0, colorOverride = null) {
    const size = window.GLIDER_ICONS.SIZE;
    const color = colorOverride || "#FFD600";

    const svg = `
    <svg
      width="${size}"
      height="${size}"
      viewBox="0 0 471.098 471.098"
      style="
        transform: rotate(${heading || 0}deg);
        transform-origin: 50% 50%;
        filter: drop-shadow(0 0 4px ${color});
      "
      xmlns="http://www.w3.org/2000/svg"
    >

      <g fill="${color}">
        <path d="M282.65,200.222c0,0-23.548-71.306-23.548-114.482c0-26.128-8.772-34.009-15.701-36.191
        c-1.723-0.543-3.416-0.739-4.839-0.777c1.36,0.024,3.023,0.205,4.839,0.777V19.662c0-4.33-3.521-7.848-7.853-7.848
        c-4.33,0-7.849,3.518-7.849,7.848v29.974c-7.005,2.292-15.701,10.291-15.701,36.105c0,43.176-23.552,114.482-23.552,114.482
        S10.181,217.234,0.367,259.098c-9.812,41.869,180.229,27.808,188.08,41.869v71.962l18.314,9.16c0,0,20.938-56.26,20.938,22.242
        h-94.207l-2.613,47.103L227.7,437.02v15.564c0,3.703,2.998,6.7,6.703,6.7h2.292c3.707,0,6.707-2.997,6.707-6.7v-15.793
        l97.593,14.643l-2.617-47.103h-94.202c0-78.502,20.546-22.242,20.546-22.242l17.93-9.16v-71.962
        c7.851-14.069,197.895,0,188.08-41.869C460.917,217.234,282.65,200.222,282.65,200.222z"/>
        
        <path d="M211.999,43.213c4.33,0,7.851-3.519,7.851-7.851c0-4.33-3.521-7.851-7.851-7.851h-39.252
        c-4.332,0-7.851,3.521-7.851,7.851c0,4.332,3.519,7.851,7.851,7.851H211.999z"/>
        
        <path d="M259.103,43.213h39.249c4.332,0,7.851-3.519,7.851-7.851c0-4.33-3.519-7.851-7.851-7.851h-39.249
        c-4.332,0-7.851,3.521-7.851,7.851C251.252,39.695,254.771,43.213,259.103,43.213z"/>
      </g>

    </svg>
    `;

    return L.divIcon({
      html: svg,
      className: "glider-svg-marker",
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2]
    });
  }

};
