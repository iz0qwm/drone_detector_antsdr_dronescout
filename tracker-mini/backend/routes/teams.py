from flask import Blueprint
from flask import jsonify
from flask import request

from services.teams import (
    get_team_status,
    load_team,
    save_team
)

teams_bp = Blueprint(
    "teams",
    __name__
)


@teams_bp.route(
    "/api/teams",
    methods=["GET"]
)
def api_teams():

    return jsonify(
        get_team_status()
    )

@teams_bp.route(
    "/api/teams/config",
    methods=["GET"]
)
def api_team_config():

    return jsonify(
        load_team()
    )

@teams_bp.route(
    "/api/teams/operator",
    methods=["POST"]
)
def api_add_operator():

    data = request.json

    team = load_team()

    operators = team["operators"]

    new_id = (
        max(
            [o["id"] for o in operators],
            default=0
        ) + 1
    )

    operators.append({

        "id": new_id,

        "longName": data["longName"],

        "shortName": data["shortName"],

        "nodeId": None,

        "lastSeen": None,

        "online": False

    })

    save_team(team)

    return jsonify({
        "success": True
    })


@teams_bp.route(
    "/api/teams/operator/<int:operator_id>",
    methods=["DELETE"]
)
def api_delete_operator(operator_id):

    team = load_team()

    team["operators"] = [

        op

        for op in team["operators"]

        if op["id"] != operator_id

    ]

    save_team(team)

    return jsonify({
        "success": True
    })


@teams_bp.route(
    "/api/teams/operator/<int:operator_id>",
    methods=["PUT"]
)
def api_update_operator(operator_id):

    data = request.json

    team = load_team()

    for op in team["operators"]:

        if op["id"] == operator_id:

            op["longName"] = data["longName"]
            op["shortName"] = data["shortName"]

            break

    save_team(team)

    return jsonify({
        "success": True
    })


