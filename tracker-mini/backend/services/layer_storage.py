from pathlib import Path
import json
import uuid

MISSIONS_DIR = Path(
    "/home/pi/tracker-mini/missions"
)

def mission_layers_dir(
    mission_id
):

    return (
        MISSIONS_DIR /
        mission_id /
        "layers"
    )

def list_layers(
    mission_id
):

    directory = mission_layers_dir(
        mission_id
    )

    if not directory.exists():
        return []

    layers = []

    for file in sorted(
        directory.glob("*.json")
    ):

        with open(file) as f:

            layers.append(
                json.load(f)
            )

    return layers


def save_layer(
    mission_id,
    layer
):

    directory = mission_layers_dir(
        mission_id
    )

    directory.mkdir(
        exist_ok=True
    )

    if "id" not in layer:

        layer["id"] = (
            uuid.uuid4().hex
        )

    with open(
        directory /
        f"{layer['id']}.json",
        "w"
    ) as f:

        json.dump(
            layer,
            f,
            indent=2
        )

    return layer


def get_layer(
    mission_id,
    layer_id
):

    file = (
        mission_layers_dir(
            mission_id
        ) /
        f"{layer_id}.json"
    )

    if not file.exists():

        return None

    with open(file) as f:

        return json.load(f)
    

def delete_layer(
    mission_id,
    layer_id
):

    file = (
        mission_layers_dir(
            mission_id
        ) /
        f"{layer_id}.json"
    )

    if not file.exists():

        return False

    file.unlink()

    return True