window.MESHTASTIC = window.MESHTASTIC || {};

MESHTASTIC.fetchTeams = async function() {

    const res =
        await fetch(
            "/api/teams"
        );

    const data =
        await res.json();

    return data;

};