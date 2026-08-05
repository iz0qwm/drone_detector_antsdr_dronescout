import importlib
import sys
import types


def import_notification_service(monkeypatch, team):
    sent = []

    fake_meshtastic = types.SimpleNamespace(
        send_direct_message=lambda node_id, text: sent.append(
            (node_id, text)
        ),
    )

    fake_teams = types.SimpleNamespace(
        load_team=lambda: team,
    )

    fake_logger = types.SimpleNamespace(
        log=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(
        sys.modules,
        "services.meshtastic_service",
        fake_meshtastic,
    )
    monkeypatch.setitem(
        sys.modules,
        "services.teams",
        fake_teams,
    )
    monkeypatch.setitem(
        sys.modules,
        "services.logger",
        fake_logger,
    )
    sys.modules.pop(
        "services.notification_service",
        None,
    )

    module = importlib.import_module(
        "services.notification_service"
    )
    module.clear_notifications()

    return module, sent


def test_send_to_operator_records_gateway_target_and_direction(
    monkeypatch,
):
    team = {
        "operators": [
            {
                "id": 1,
                "longName": "Operator One",
                "shortName": "OP1",
                "nodeId": "!operator",
                "online": True,
            }
        ]
    }

    service, sent = import_notification_service(
        monkeypatch,
        team,
    )

    notification = service.send_to_operator(
        "!operator",
        "Helicopter operating nearby",
    )

    assert sent == [
        (
            "!operator",
            "Helicopter operating nearby",
        )
    ]
    assert notification["source"] == "Gateway"
    assert notification["target_label"] == "Operator One"
    assert notification["direction"] == "outgoing"
    assert notification["status"] == "sent"


def test_send_to_all_operators_uses_only_online_node_ids(
    monkeypatch,
):
    team = {
        "operators": [
            {
                "id": 1,
                "longName": "Operator One",
                "nodeId": "!operator",
                "online": True,
            },
            {
                "id": 2,
                "longName": "Offline Operator",
                "nodeId": "!offline",
                "online": False,
            },
            {
                "id": 3,
                "longName": "Missing Node",
                "online": True,
            },
        ]
    }

    service, sent = import_notification_service(
        monkeypatch,
        team,
    )

    notifications = service.send_to_all_operators(
        "Check radio",
    )

    assert sent == [
        (
            "!operator",
            "Check radio",
        )
    ]
    assert len(notifications) == 1
    assert notifications[0]["target_node_id"] == "!operator"
    assert notifications[0]["target_label"] == "Operator One"
