window.MISSION = window.MISSION || {};

MISSION.teams = {};
MISSION.teams.selectedOperator = null;
MISSION.teams.messageTarget = "operator";

MISSION.teams.open = async function () {

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

        const msgRes =
            await fetch(
                "/api/notifications"
            );

        const notifications =
            await msgRes.json();
            
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
                <button
                    class="btn-mission"
                    onclick="MISSION.teams.sendMessageToAll()">

                    💬 Send message to all

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
                                class="btn-mission"
                                onclick="MISSION.teams.sendMessage(${op.id})">

                                Message

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
                <button
                    class="btn-delete-mission"
                    onclick="MISSION.teams.resetExternalNodes()">

                    Clear radio NodeDB

                </button>
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
                <button
                    class="btn-delete-mission"
                    onclick="MISSION.teams.removeExternalNode('${node.id}')">

                    Remove from radio

                </button>
            `;

        });

        html += `
            </details>
        `;
        html += `
            <hr>
            <h3>Messages</h3>
        `;

        notifications.messages.forEach(msg => {
            html += `
                <div class="team-card">
                    <b>${msg.source ?? "-"}</b>
                    <br>
                    ${msg.text}
                </div>
            `;
        });
        html += "</div>";
        //console.log(html);
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

    await MISSION.teams.open();

}

MISSION.teams.deleteOperator = async function (id) {

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

    await MISSION.teams.open();

}

MISSION.teams.editOperator = async function (id) {

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

    await MISSION.teams.open();

}


MISSION.teams.resetExternalNodes = async function () {

    if (
        !confirm(
            "Clear Meshtastic NodeDB on the radio?"
        )
    ) {
        return;
    }

    await fetch(
        "/api/meshtastic/nodes/reset",
        {
            method: "POST"
        }
    );

    await MISSION.teams.open();

};


MISSION.teams.removeExternalNode = async function (nodeId) {

    if (
        !confirm(
            "Remove this node from the Meshtastic radio?"
        )
    ) {
        return;
    }

    await fetch(
        `/api/meshtastic/nodes/${encodeURIComponent(nodeId)}`,
        {
            method: "DELETE"
        }
    );

    await MISSION.teams.open();

};


MISSION.teams.sendMessage = async function (id) {

    const team =
        await MISSION.teams.loadConfig();

    const operator =
        team.operators.find(
            o => o.id === id
        );

    if (!operator) {
        return;
    }

    MISSION.teams.selectedOperator = operator;

    MISSION.teams.messageTarget = "operator";

    document.getElementById(
        "messageModalTitle"
    ).textContent =
        `Message to ${operator.longName}`;

    document.getElementById(
        "messageText"
    ).value = "";

    document
        .getElementById(
            "messageModal"
        )
        .classList
        .add("open");

};


MISSION.teams.sendMessageToAll = async function () {

    MISSION.teams.selectedOperator = null;

    MISSION.teams.messageTarget = "all";

    document.getElementById(
        "messageModalTitle"
    ).textContent =
        "Message to all operators";

    document.getElementById(
        "messageText"
    ).value = "";

    document
        .getElementById("messageModal")
        .classList
        .add("open");

};


MISSION.teams.sendCurrentMessage = async function () {

    const text =
        document
            .getElementById("messageText")
            .value
            .trim();

    if (!text) {
        alert("Insert a message");
        return;
    }

    let url;
    let body;

    if (
        MISSION.teams.messageTarget === "operator"
    ) {

        url =
            "/api/notifications/operator";

        body = {
            node_id:
                MISSION.teams.selectedOperator.nodeId,
            text
        };

    }
    else {

        url =
            "/api/notifications/all";

        body = {
            text
        };

    }

    const res =
        await fetch(
            url,
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify(body)
            }
        );

    const data =
        await res.json();

    if (!data.ok) {

        alert("Unable to send message");

        return;

    }

    document
        .getElementById("messageModal")
        .classList
        .remove("open");

    document
        .getElementById("messageText")
        .value = "";

    await MISSION.teams.open();

};

MISSION.teams.insertTemplate = function(type) {

    const textarea =
        document.getElementById(
            "messageText"
        );

    switch(type){

        case "drone":
            textarea.value =
                "Drone operating nearby.";
            break;

        case "aircraft":
            textarea.value =
                "Aircraft approaching your area.";
            break;

        case "helicopter":
            textarea.value =
                "Helicopter operating nearby.";
            break;

        case "warning":
            textarea.value =
                "Warning.";
            break;

        case "info":
            textarea.value =
                "Information.";
            break;
    }

};