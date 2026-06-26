window.MISSION = window.MISSION || {};

MISSION.teams = {};

MISSION.teams.open = async function () {

    missionTeamsModal
        .classList
        .add("open");

    const container =
        document.getElementById(
            "teamsContainer"
        );

    container.innerHTML =
        "Loading...";

    try {

        const res =
            await fetch(
                "/api/teams"
            );

        const data =
            await res.json();

        const team =
            await MISSION.teams.loadConfig();

        let html = `
        <div class="mission-current-card">

        <h3>Gateway</h3>

        <div class="team-card">

            🟢 <b>${data.gateway_node?.name ?? "Unknown"}</b>

            <br>

            ${data.gateway_node?.hwModel ?? ""}

            <br>

            Region:
            ${data.gateway?.lora?.region ?? "-"}

            <br>

            TX Power:
            ${data.gateway?.lora?.tx_power ?? "-"}

            dBm

            <br>

            Hop Limit:
            ${data.gateway?.lora?.hop_limit ?? "-"}
            <br>

            Nodes:

            ${data.operators.length + data.external_nodes.length + 1}
            <br>

            Channel Utilization:
            ${
                data.gateway_node &&
                data.gateway_node.channelUtilization != null
                    ? data.gateway_node.channelUtilization.toFixed(1)
                    : "-"
            }
            %

        </div>

                <hr>

                <h3>Mission Operators</h3>

                <button
                    class="btn-mission"
                    onclick="MISSION.teams.addOperator()">

                    + Add operator

                </button>

                <br><br>
        `;

                data.operators.forEach(op => {

                    html += `

                        <div class="team-card">

                            ${op.online ? "🟢" : "🔴"}
                            <b>${op.longName || op.name}</b>

                            <br>

                            Short Name:
                            ${op.shortName ?? "-"}

                            <br>

                            Node:
                            ${op.nodeId || op.id || "-"}

                            <br>

                            Battery:
                            ${op.battery ?? "-"}

                            <br>

                            SNR:
                            ${op.snr ?? "-"} dB

                            <br>

                            Position:
                            ${
                                op.lat && op.lon
                                    ? `${op.lat.toFixed(6)}, ${op.lon.toFixed(6)}`
                                    : "-"
                            }

                            <br>

                            Last Seen:
                            ${op.lastSeen ?? "-"}

                            <br><br>

                            <button
                                class="btn-mission"
                                onclick="MISSION.teams.editOperator(${op.id})">

                                Edit

                            </button>

                            <button
                                class="btn-delete-mission"
                                onclick="MISSION.teams.deleteOperator(${op.id})">

                                Delete

                            </button>

                        </div>

                    `;

                });

        html += `
            <hr>

            <details>

                <summary>
                    External Nodes (${data.external_nodes.length})
                </summary>
        `;

        data.external_nodes.forEach(node => {

            html += `
                <div class="team-card">
                    📡 <b>${node.name}</b>
                    <br>
                    Short Name:
                    ${node.shortName ?? "-"}
                    <br>
                    Node:
                    ${node.id ?? "-"}
                    <br>
                    SNR:
                    ${node.snr ?? "-"} dB
                </div>
            `;

        });

        html += `
            </details>
        `;
        html += `
            <hr>
            <h3>Messages</h3>
        `;

        data.messages.forEach(msg => {
            html += `
                <div class="team-card">
                    <b>${msg.from}</b>
                    <br>
                    ${msg.text}
                </div>
            `;
        });
        html += "</div>";
        container.innerHTML = html;
    }

    catch(err){
        container.innerHTML =
            "Unable to load Teams";
    }

}


MISSION.teams.loadConfig = async function () {

    const res =
        await fetch(
            "/api/teams/config"
        );

    return await res.json();

}


MISSION.teams.addOperator = async function () {

    const longName =
        prompt("Operator long name");

    if (!longName) {
        return;
    }

    const shortName =
        prompt("Operator short name");

    if (!shortName) {
        return;
    }

    await fetch(
        "/api/teams/operator",
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({

                longName,
                shortName

            })
        }
    );

    openMissionTeams();

}

MISSION.teams.deleteOperator = async function () {

    if (
        !confirm(
            "Delete operator?"
        )
    ) {
        return;
    }

    await fetch(

        `/api/teams/operator/${id}`,

        {
            method: "DELETE"
        }

    );

    openMissionTeams();

}

MISSION.teams.editOperator = async function () {

    const team =
        await MISSION.teams.loadConfig();

    const operator =
        team.operators.find(
            o => o.id === id
        );

    if (!operator) {
        return;
    }

    const longName =
        prompt(
            "Operator long name",
            operator.longName
        );

    if (!longName) {
        return;
    }

    const shortName =
        prompt(
            "Operator short name",
            operator.shortName
        );

    if (!shortName) {
        return;
    }

    await fetch(
        `/api/teams/operator/${id}`,
        {
            method: "PUT",
            headers: {
                "Content-Type":
                    "application/json"
            },
            body: JSON.stringify({
                longName,
                shortName
            })
        }
    );

    openMissionTeams();

}
