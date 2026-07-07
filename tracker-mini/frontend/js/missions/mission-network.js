window.MISSION = window.MISSION || {};

MISSION.api = {

    async missions() {

        const res =
            await fetch(
                "/api/missions"
            );

        return await res.json();

    },

    async currentMission() {

        const res =
            await fetch(
                "/api/missions/current"
            );

        return await res.json();

    },

    async selectMission(missionId) {

        const res = await fetch(
            "/api/missions/select",
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify({
                    mission_id: missionId
                })
            }
        );

        return await res.json();

    },

    async layers(missionId) {

        const res = await fetch(
            `/api/missions/${missionId}/layers`
        );

        return await res.json();

    },

    async layer(
        missionId,
        layerId
    ) {

        const res = await fetch(
            `/api/missions/${missionId}/layers/${layerId}`
        );

        return await res.json();

    },

    async createLayer(
        missionId,
        layer
    ) {

        const res = await fetch(

            `/api/missions/${missionId}/layers`,

            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify(
                    layer
                )

            }

        );

        return await res.json();

    },

    async updateLayer(
        missionId,
        layer
    ) {

        const res = await fetch(

            `/api/missions/${missionId}/layers/${layer.id}`,

            {
                method: "PUT",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify(
                    layer
                )

            }

        );

        return await res.json();

    },

    async deleteLayer(
        missionId,
        layerId
    ) {

        const res = await fetch(

            `/api/missions/${missionId}/layers/${layerId}`,

            {
                method: "DELETE"
            }

        );

        return await res.json();

    },

    async importGeoJSON(
        missionId,
        file
    ) {

        const form =
            new FormData();

        form.append(
            "mission_id",
            missionId
        );

        form.append(
            "file",
            file
        );

        const res =
            await fetch(

                "/api/missions/import-geojson",

                {
                    method: "POST",
                    body: form
                }

            );

        return await res.json();

    },

    async updateMission(
        missionId,
        mission
    ) {

        const res = await fetch(

            `/api/missions/${missionId}`,

            {
                method: "PUT",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify(
                    mission
                )

            }

        );

        return await res.json();

    },


    async deleteMission(
            missionId
        ) {

            const res = await fetch(

                `/api/missions/${missionId}`,

                {
                    method: "DELETE"
                }

            );

            return await res.json();

        },

        async importDscZones(
        missionId,
        options
    ) {

        const res =
            await fetch(

                `/api/missions/${missionId}/import-dsc-zones`,

                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify(
                        options
                    )

                }

            );

        return await res.json();

    }


};


