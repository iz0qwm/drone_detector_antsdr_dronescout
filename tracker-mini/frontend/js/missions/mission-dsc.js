window.MISSION = window.MISSION || {};

MISSION.dsc = {

    currentMissionId: null,
    previewMap: null,
    previewRectangle: null,
    syncingPreview: false,
    syncingMain: false,
    openImportDialog(missionId) {

        this.currentMissionId = missionId;

        document
            .getElementById(
                "importDscModal"
            )
            ?.classList
            .add("open");

        this.updateStatus();
        this.initPreviewMap();

        setTimeout(
            () => {
                this.previewMap.invalidateSize();
                this.updatePreview();
            },
            100
        );

        window.airNodeMap.off(
            "moveend",
            this.updatePreview
        );

        this.previewMap.off(
            "moveend"
        );

        this.previewMap.on(
            "moveend",
            () => {

                if (this.syncingMain) {
                    return;
                }
                this.syncingPreview = true;
                window.airNodeMap.setView(
                    this.previewMap.getCenter(),
                    this.previewMap.getZoom(),
                    {
                        animate: false
                    }
                );
                this.updatePreviewRectangle();
                this.syncingPreview = false;
            }
        );

        window.airNodeMap.on(
            "moveend",
            () => this.updatePreview()
        );

    },

    initPreviewMap() {

        if (this.previewMap) {
            return;
        }

        this.previewMap = L.map(
            "dscImportPreview",
            {
                zoomControl: true,
                attributionControl: false,
                dragging: true,
                doubleClickZoom: true,
                scrollWheelZoom: true,
                touchZoom: true,
                boxZoom: true,
                keyboard: false
            }
        );

        L.tileLayer(

            "https://tile.openstreetmap.org/{z}/{x}/{y}.png",

            {
                maxZoom: 19
            }

        ).addTo(this.previewMap);

    },

    updatePreview() {
        if (!this.previewMap) {
            return;
        }
        if (this.syncingPreview) {
            return;
        }
        this.syncingMain = true;
        const center =
            window.airNodeMap.getCenter();
        const zoom =
            window.airNodeMap.getZoom();
        this.previewMap.setView(
            center,
            zoom,
            {
                animate: false
            }
        );
        this.updatePreviewRectangle();
        this.syncingMain = false;
    },

    updatePreviewRectangle() {
        if (!this.previewMap) {
            return;
        }
        const bounds =
            this.previewMap.getBounds();
        if (this.previewRectangle) {
            this.previewMap.removeLayer(
                this.previewRectangle
            );
        }
        this.previewRectangle =
            L.rectangle(
                bounds,
                {
                    color: "#ff0000",
                    weight: 2,
                    fillOpacity: 0.12
                }
            ).addTo(
                this.previewMap
            );
    },

    close() {

        const progress =
            document.getElementById(
                "dscImportProgress"
            );

        if (progress) {
            progress.style.display = "none";
        }

        const button =
            document.getElementById(
                "startImportDscBtn"
            );

        if (button) {
            button.disabled = false;
        }

        document
            .getElementById(
                "importDscModal"
            )
            ?.classList
            .remove("open");

        if (this.previewMap) {
            this.previewMap.off("moveend");
        }
    },

    updateStatus() {

        const box =
            document.getElementById(
                "dscImportStatus"
            );

        const importBtn =
            document.getElementById(
                "startImportDscBtn"
            );

        if (!box || !importBtn) {
            return;
        }

        const dscAvailable =
            window.services?.dsc === true;

        if (dscAvailable) {

            box.innerHTML = `
                🟢 <b>Drone Sky Check Cloud available</b><br>
                Aeronautical areas can be downloaded into the current mission.
            `;

            importBtn.disabled = false;

        } else {

            box.innerHTML = `
                🔴 <b>Drone Sky Check unavailable</b><br>
                Internet connection required to download aeronautical areas.
            `;

            importBtn.disabled = true;

        }

    },
    
    async import() {
        if (!this.currentMissionId) {
            return;
        }

        this.setProgress(
            "☁ Connecting to Drone Sky Check..."
        );

        document.getElementById(
            "startImportDscBtn"
        ).disabled = true;

        const bounds =
            this.previewMap.getBounds();
        const bbox = [
            bounds.getSouth(),
            bounds.getWest(),
            bounds.getNorth(),
            bounds.getEast()
        ];

        this.setProgress(
            "⬇ Downloading aeronautical areas..."
        );

        const result =
            await MISSION.api.importDscZones(
                this.currentMissionId,
                {
                    bbox,
                    simplify:
                        document.getElementById(
                            "dscSimplify"
                        ).checked,
                    limit: 1000
                }
            );

        if (!result.success) {
            alert(
                result.message ||
                "Unable to import DSC zones"
            );
            return;
        }

        this.setProgress(
            "🗺 Updating mission..."
        );
        document.getElementById(
            "startImportDscBtn"
        ).disabled = false;

        document.getElementById(
            "dscImportProgress"
        ).style.display = "none";

        this.setProgress(
            "✅ Import completed."
        );

        this.close();
        await MISSION.planning.loadMissionLayers(
            this.currentMissionId
        );
    },

    init() {
        document
            .getElementById(
                "closeImportDscModal"
            )
            ?.addEventListener(
                "click",
                () => this.close()
            );

        document
            .getElementById(
                "cancelImportDscBtn"
            )
            ?.addEventListener(
                "click",
                () => this.close()
            );

        document
            .getElementById(
                "startImportDscBtn"
            )
            ?.addEventListener(
                "click",
                () => this.import()
            );

    },
    setProgress(text) {

        const box =
            document.getElementById(
                "dscImportProgress"
            );

        box.style.display = "block";
        const label =
            document.getElementById(
                "dscImportProgressText"
            );

        if (label) {

            label.textContent = text;

        }

        box.style.display = "flex";

    }

};

