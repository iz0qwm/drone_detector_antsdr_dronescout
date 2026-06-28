from pathlib import Path
import json
from datetime import datetime
import shutil
from services.logger import log

from services.mission_storage import (
    load_index,
    save_index,
    next_mission_id,
    get_current_mission_id,
    set_current_mission_id
)

from services.layer_storage import (
    list_layers,
    save_layer,
    get_layer,
    delete_layer
)

MISSIONS_DIR = Path(
    "/home/pi/tracker-mini/missions"
)


def list_missions():

    return load_index()


def create_mission(
    name,
    description=""
):

    mission_id = next_mission_id()

    mission_dir = (
        MISSIONS_DIR /
        mission_id
    )

    mission_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    layers_dir = (
        mission_dir /
        "layers"
    )

    layers_dir.mkdir(
        exist_ok=True
    )

    mission = {
        "id": mission_id,
        "name": name,
        "description": description,
        "status": "Planning",
        "created": (
            datetime.utcnow()
            .isoformat()
        )
    }

    with open(
        mission_dir / "mission.json",
        "w"
    ) as f:

        json.dump(
            mission,
            f,
            indent=2
        )



    index = load_index()
    index.append({
        "id": mission_id,
        "name": name,
        "description": description,
        "status": "Planning"
    })
    save_index(index)

    log(
        "MISSION",
        "Created",
        mission_id,
        name
    )

    return mission



def get_mission(mission_id):
    mission_file = (
        MISSIONS_DIR /
        mission_id /
        "mission.json"
    )
    if not mission_file.exists():
        return None
    with open(mission_file, "r") as f:
        return json.load(f)


def update_mission(
    mission_id,
    data
):

    mission = get_mission(
        mission_id
    )

    if mission is None:
        return None

    mission["name"] = data.get(
        "name",
        mission["name"]
    )

    mission["description"] = data.get(
        "description",
        mission.get(
            "description",
            ""
        )
    )

    mission["status"] = data.get(
        "status",
        mission.get(
            "status",
            "Planning"
        )
    )

    mission_file = (
        MISSIONS_DIR /
        mission_id /
        "mission.json"
    )

    with open(
        mission_file,
        "w"
    ) as f:

        json.dump(
            mission,
            f,
            indent=2
        )

    index = load_index()

    for item in index:

        if item["id"] == mission_id:

            item["name"] = mission["name"]
            item["description"] = mission["description"]
            item["status"] = mission["status"]

            break

    save_index(
        index
    )

    log(
        "MISSION",
        "Updated",
        mission_id,
        mission["name"]
    )

    return mission

def set_current_mission(
    mission_id
):
    mission = get_mission(
        mission_id
    )
    if mission is None:
        return False
    set_current_mission_id(
        mission_id
    )

    log(
        "MISSION",
        "Selected",
        mission_id
    )

    return True


def get_current_mission():
    mission_id = get_current_mission_id()

    if not mission_id:
        return None

    return get_mission(
        mission_id
    )

def delete_mission(mission_id):

    mission_dir = (
        MISSIONS_DIR /
        mission_id
    )
    if not mission_dir.exists():
        return False
    shutil.rmtree(
        mission_dir
    )
    index = load_index()

    index = [
        m for m in index
        if m["id"] != mission_id
    ]

    save_index(index)

    current_id = get_current_mission_id()

    if current_id == mission_id:

        set_current_mission_id(
            None
        )

    log(
        "MISSION",
        "Deleted",
        mission_id
    )

    return True


def import_geojson(
    mission_id,
    uploaded_file
):

    mission_dir = (
        MISSIONS_DIR /
        mission_id
    )

    if not mission_dir.exists():
        return None

    geojson = json.load(
        uploaded_file.stream
    )

    layer = {
        "name": uploaded_file.filename,
        "type": "geojson",
        "geometry": "feature_collection",
        "visible": True,
        "locked": False,
        "style": {},
        "properties": {
            "source": "import",
            "filename": uploaded_file.filename
        },
        "geojson": geojson,
        "order": 0,
        "created": datetime.utcnow().isoformat(),
    }

    saved = save_layer(
        mission_id,
        layer
    )

    log(
        "MISSION",
        "Imported GeoJSON layer",
        mission_id,
        saved["id"],
        uploaded_file.filename
    )

    return saved


