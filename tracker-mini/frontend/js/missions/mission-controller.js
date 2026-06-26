window.MISSION = window.MISSION || {};

(function () {

    function init() {

        console.log("[MISSION] Init");

        bindButtons();

        MISSION.toolbar.init();

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

        MISSION.draw.stop();

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

    MISSION.init = init;

})();