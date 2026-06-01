from pathlib import Path
import json

MISSIONS_DIR = Path("/home/pi/tracker-mini/missions")
INDEX_FILE = MISSIONS_DIR / "mission_index.json"
CURRENT_FILE = (
    MISSIONS_DIR /
    "current_mission.json"
)

def load_index():

    if not INDEX_FILE.exists():
        return []

    with open(INDEX_FILE, "r") as f:
        return json.load(f)


def save_index(data):

    with open(INDEX_FILE, "w") as f:
        json.dump(data, f, indent=2)


def next_mission_id():
    missions = load_index()
    if not missions:
        return "mission_001"
    ids = []
    for m in missions:
        try:
            ids.append(
                int(
                    m["id"]
                    .replace("mission_", "")
                )
            )
        except:
            pass
    n = max(ids) + 1
    return f"mission_{n:03d}"

def get_current_mission_id():
    if not CURRENT_FILE.exists():
        return None
    with open(CURRENT_FILE, "r") as f:
        data = json.load(f)
    return data.get("mission_id")

def set_current_mission_id(
    mission_id
):
    with open(
        CURRENT_FILE,
        "w"
    ) as f:
        json.dump(
            {
                "mission_id":
                    mission_id
            },
            f,
            indent=2
        )

