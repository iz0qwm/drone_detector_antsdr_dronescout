window.MISSION = window.MISSION || {};

MISSION.toolbar = {

    show() {

        document
            .getElementById(
                "missionToolbar"
            )
            .classList
            .add("open");

    },

    hide() {

        document
            .getElementById(
                "missionToolbar"
            )
            .classList
            .remove("open");

    },

    init() {

        document
            .getElementById(
                "drawRectangleBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.startRectangle();

                }
            );

        document
            .getElementById(
                "drawPolygonBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.startPolygon();

                }
            );

        document
            .getElementById(
                "drawCircleBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.startCircle();

                }
            );

        document
            .getElementById(
                "drawMarkerBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.startMarker();

                }
            );

        document
            .getElementById(
                "editGeometryBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.startEdit();

                }
            );

        document
            .getElementById(
                "deleteGeometryBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.startDelete();

                }
            );
    }

};