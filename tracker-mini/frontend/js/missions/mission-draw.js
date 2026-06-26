window.MISSION = window.MISSION || {};

MISSION.draw = {

    mode: null,

    startRectangle() {

        this.stop();
        this.mode = "rectangle";

        console.log(
            "[DRAW] Rectangle mode"
        );

    },

    startPolygon() {

        this.stop();
        this.mode = "polygon";

        console.log(
            "[DRAW] Polygon mode"
        );

    },

    startCircle() {

        this.stop();
        this.mode = "circle";

        console.log(
            "[DRAW] Circle mode"
        );

    },

    startMarker() {

        this.stop();
        this.mode = "marker";

        console.log(
            "[DRAW] Marker mode"
        );

    },

    startEdit() {

        this.stop();
        this.mode = "edit";

        console.log(
            "[DRAW] Edit mode"
        );

    },

    startDelete() {

        this.stop();
        this.mode = "delete";

        console.log(
            "[DRAW] Delete mode"
        );

    },

    stop() {

        this.mode = null;

        console.log(
            "[DRAW] Stop"
        );

    }

};