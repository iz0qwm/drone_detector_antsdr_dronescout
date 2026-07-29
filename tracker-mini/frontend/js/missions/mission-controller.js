window.MISSION = window.MISSION || {};

(function () {

    function init() {

        console.log("[MISSION] Init");

        bindButtons();

        MISSION.toolbar.init();
        MISSION.layerProperties.init();
        MISSION.dsc.init();
    }

    function bindButtons() {

        bindButton(
            "createMissionBtn",
            openCreateMission
        );

        bindButton(
            "missionPlanningBtn",
            openMissionPlanning
        );

        bindButton(
            "closeCreateMissionModal",
            closeCreateMission
        );

        bindButton(
            "saveMissionBtn",
            createMission
        );

        bindButton(
            "closeMissionPlanningModal",
            closeMissionPlanning
        );

        bindButton(
            "closeMissionTeamsModal",
            closeMissionTeams
        );

        bindButton(
            "missionTeamsBtn",
            openMissionTeams
        );


        bindButton(
            "closeMessageModal",
            closeMessage
        );

        bindButton(
            "cancelMessageBtn",
            closeMessage
        );

        bindButton(
            "sendMessageBtn",
            sendCurrentMessage
        );
    }

    function bindButton(id, handler) {

        document
            .getElementById(id)
            ?.addEventListener(
                "click",
                handler
            );

    }

    function openModal(id) {

        document
            .getElementById(id)
            ?.classList
            .add("open");

    }

    function closeModal(id) {

        document
            .getElementById(id)
            ?.classList
            .remove("open");

    }

    function openCreateMission() {

        openModal(
            "createMissionModal"
        );

        document
            .getElementById(
                "missionName"
            )
            ?.focus();

    }

    function closeCreateMission() {

        closeModal(
            "createMissionModal"
        );

    }

    async function createMission() {

        const nameInput =
            document.getElementById(
                "missionName"
            );

        const descriptionInput =
            document.getElementById(
                "missionDescription"
            );

        const name =
            (
                nameInput?.value ||
                ""
            ).trim();

        const description =
            (
                descriptionInput?.value ||
                ""
            ).trim();

        if (!name) {

            alert(
                "Mission name required"
            );

            nameInput?.focus();

            return;

        }

        try {

            const result =
                await MISSION.api.createMission({
                    name,
                    description
                });

            if (!result.success) {

                alert(
                    result.message ||
                    "Unable to create mission"
                );

                return;

            }

            if (nameInput) {
                nameInput.value = "";
            }

            if (descriptionInput) {
                descriptionInput.value = "";
            }

            closeCreateMission();

            await MISSION.api.selectMission(
                result.mission.id
            );

            if (
                document
                    .getElementById(
                        "missionPlanningModal"
                    )
                    ?.classList
                    .contains("open")
            ) {

                await MISSION.planning.refresh();

            }

        } catch (err) {

            console.error(err);

            alert(
                "Unable to create mission"
            );

        }

    }

    async function openMissionPlanning() {

        openModal(
            "missionPlanningModal"
        );

        MISSION.toolbar.show();

        await MISSION.planning.open();

    }

    function closeMissionPlanning() {

        MISSION.draw.cancel();

        MISSION.toolbar.hide();

        closeModal(
            "missionPlanningModal"
        );

    }

    async function openMissionTeams() {

        openModal(
            "missionTeamsModal"
        );

        await MISSION.teams.open();

    }

    function closeMissionTeams() {

        closeModal(
            "missionTeamsModal"
        );

    }

    function closeMessage() {

        closeModal(
            "messageModal"
        );

    }

    async function sendCurrentMessage() {

        await MISSION.teams.sendCurrentMessage();

    }

    MISSION.init = init;

})();
