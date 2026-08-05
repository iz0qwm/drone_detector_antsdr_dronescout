import importlib
import sys
import types


def import_meshtastic_service(monkeypatch):
    fake_config = types.SimpleNamespace(
        SETTINGS={
            "meshtastic": {
                "node_id": "!435bfc80",
            },
            "traffic": {
                "meshtastic_enabled": True,
            },
        }
    )

    fake_logger = types.SimpleNamespace(
        log=lambda *args, **kwargs: None,
    )

    fake_gps = types.SimpleNamespace(
        get_gps_status=lambda: {
            "available": False,
            "fix": False,
        },
    )

    fake_serial_module = types.ModuleType(
        "meshtastic.serial_interface"
    )
    fake_serial_module.SerialInterface = object

    fake_meshtastic_package = types.ModuleType(
        "meshtastic"
    )
    fake_pubsub = types.ModuleType(
        "pubsub"
    )
    fake_pubsub.pub = types.SimpleNamespace(
        subscribe=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(
        sys.modules,
        "config",
        fake_config,
    )
    monkeypatch.setitem(
        sys.modules,
        "services.logger",
        fake_logger,
    )
    monkeypatch.setitem(
        sys.modules,
        "services.gps",
        fake_gps,
    )
    monkeypatch.setitem(
        sys.modules,
        "meshtastic",
        fake_meshtastic_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "meshtastic.serial_interface",
        fake_serial_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "pubsub",
        fake_pubsub,
    )
    sys.modules.pop(
        "services.meshtastic_service",
        None,
    )

    return importlib.import_module(
        "services.meshtastic_service"
    )


def test_incoming_text_packet_is_recorded(monkeypatch):
    recorded = []

    fake_notification_service = types.SimpleNamespace(
        record_incoming_text=lambda **kwargs: recorded.append(
            kwargs
        ),
    )

    monkeypatch.setitem(
        sys.modules,
        "services.notification_service",
        fake_notification_service,
    )

    service = import_meshtastic_service(monkeypatch)

    active_interface = types.SimpleNamespace(
        nodes={
            "!operator": {
                "user": {
                    "longName": "Operator One",
                    "shortName": "OP1",
                }
            }
        },
        localNode=types.SimpleNamespace(
            nodeNum=int("435bfc80", 16),
        ),
    )

    packet = {
        "fromId": "!operator",
        "toId": "!435bfc80",
        "decoded": {
            "portnum": "TEXT_MESSAGE_APP",
            "text": "Arrived at checkpoint",
        },
    }

    service.on_receive(
        packet,
        active_interface,
    )

    assert recorded == [
        {
            "source_node_id": "!operator",
            "source_label": "Operator One",
            "target_node_id": "!435bfc80",
            "target_label": "Gateway",
            "text": "Arrived at checkpoint",
        }
    ]


def test_local_text_packet_is_not_recorded_as_incoming(monkeypatch):
    recorded = []

    fake_notification_service = types.SimpleNamespace(
        record_incoming_text=lambda **kwargs: recorded.append(
            kwargs
        ),
    )

    monkeypatch.setitem(
        sys.modules,
        "services.notification_service",
        fake_notification_service,
    )

    service = import_meshtastic_service(monkeypatch)

    active_interface = types.SimpleNamespace(
        nodes={},
        localNode=types.SimpleNamespace(
            nodeNum=int("435bfc80", 16),
        ),
    )

    packet = {
        "fromId": "!435bfc80",
        "toId": "!operator",
        "decoded": {
            "portnum": "TEXT_MESSAGE_APP",
            "text": "Outbound echo",
        },
    }

    service.on_receive(
        packet,
        active_interface,
    )

    assert recorded == []
