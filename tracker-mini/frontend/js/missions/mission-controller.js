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

    }

    function closeCreateMission() {

        closeModal(
            "createMissionModal"
        );

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