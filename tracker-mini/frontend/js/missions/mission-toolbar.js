window.MISSION = window.MISSION || {};

MISSION.toolbar = {

    mode: null,
    showButton(id) {

        document
            .getElementById(id)
            ?.classList.remove("hidden");

    },

    hideButton(id) {

        document
            .getElementById(id)
            ?.classList.add("hidden");

    },
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

    setNewLayerMode() {

        this.mode = "new-layer";

        this.showButton("drawRectangleBtn");
        this.showButton("drawPolygonBtn");
        this.showButton("drawCircleBtn");
        this.showButton("drawMarkerBtn");

        this.hideButton("editVerticesBtn");

        this.showButton("saveGeometryBtn");
        this.hideButton("revertGeometryBtn");
        this.showButton("cancelGeometryBtn");

    },

    setEditLayerMode() {

        this.mode = "edit-layer";

        this.hideButton("drawRectangleBtn");
        this.hideButton("drawPolygonBtn");
        this.hideButton("drawCircleBtn");
        this.hideButton("drawMarkerBtn");

        this.showButton("editVerticesBtn");

        this.showButton("saveGeometryBtn");
        this.showButton("revertGeometryBtn");
        this.showButton("cancelGeometryBtn");

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
                "editVerticesBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.enableVertexEditing();

                }
            );

        document
            .getElementById(
                "saveGeometryBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.save();

                }
            );

        document
            .getElementById(
                "revertGeometryBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.revert();

                }
            );

        document
            .getElementById(
                "cancelGeometryBtn"
            )
            ?.addEventListener(
                "click",
                () => {

                    MISSION.draw.cancel();

                }
            );
            
        }

};