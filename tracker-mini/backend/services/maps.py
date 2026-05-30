import sqlite3
import os
from pathlib import Path
from config import SETTINGS
import shutil
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MAPS_DIR = BASE_DIR / "maps"
CATALOG_FILE = MAPS_DIR / "maps_catalog.json"

MAP_PROVIDER_FILE = (
    BASE_DIR /
    "config" /
    "map_provider.json"
)

def load_map_provider():

    if not MAP_PROVIDER_FILE.exists():

        return {
            "provider": "thunderforest",
            "api_key": ""
        }

    with open(
        MAP_PROVIDER_FILE,
        "r"
    ) as f:

        return json.load(f)


def save_map_provider(
    provider,
    api_key
):

    with open(
        MAP_PROVIDER_FILE,
        "w"
    ) as f:

        json.dump(
            {
                "provider": provider,
                "api_key": api_key
            },
            f,
            indent=2
        )

    return True


def read_mbtiles_metadata(
    mbtiles_file
):

    result = {}

    try:

        conn = sqlite3.connect(
            mbtiles_file
        )

        cur = conn.cursor()

        cur.execute(
            """
            SELECT name,value
            FROM metadata
            """
        )

        for name, value in cur.fetchall():

            result[name] = value

        conn.close()

    except Exception:

        pass

    return result




def load_catalog():

    if not CATALOG_FILE.exists():
        return {}

    try:

        with open(CATALOG_FILE, "r") as f:
            return json.load(f)

    except Exception:
        return {}

        
def get_tile(z, x, y):

    try:

        tms_y = (2 ** z - 1) - y

        catalog = load_catalog()

        active_maps = []

        for file_name, meta in catalog.items():

            if meta.get("active", False):
                active_maps.append(file_name)

        base_map = SETTINGS["map"]["base_map"]

        if base_map not in active_maps:
            active_maps.insert(
                0,
                base_map
            )

        for map_name in active_maps:

            mbtiles_file = MAPS_DIR / map_name

            if not mbtiles_file.exists():
                continue

            conn = sqlite3.connect(mbtiles_file)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT tile_data
                FROM tiles
                WHERE zoom_level=?
                AND tile_column=?
                AND tile_row=?
                """,
                (z, x, tms_y)
            )

            row = cursor.fetchone()

            conn.close()

            if row:
                return row[0]

        return None

    except Exception as e:

        print(f"Tile error: {e}")
        return None




def list_maps():

    maps = []

    try:

        base_map = SETTINGS["map"]["base_map"]
        catalog = load_catalog()

        for file in MAPS_DIR.glob("*.mbtiles"):

            mbtiles_meta = (
                read_mbtiles_metadata(file)
            )

            size_mb = round(
                file.stat().st_size / (1024 * 1024),
                2
            )

            meta = catalog.get(file.name, {})

            maps.append({
                "name": file.name,
                "description": meta.get(
                    "description",
                    ""
                ),
                "created": meta.get(
                    "created",
                    ""
                ),
                "source": meta.get(
                    "source",
                    ""
                ),
                "center_lat":
                    mbtiles_meta.get(
                        "center_lat"
                    ),

                "center_lon":
                    mbtiles_meta.get(
                        "center_lon"
                    ),

                "radius_km":
                    mbtiles_meta.get(
                        "radius_km"
                    ),

                "min_zoom":
                    mbtiles_meta.get(
                        "min_zoom"
                    ),

                "max_zoom":
                    mbtiles_meta.get(
                        "max_zoom"
                    ),
                "size_mb": size_mb,
                "active":
                    meta.get(
                        "active",
                        False
                    ),
                "protected": file.name == base_map
            })

        maps.sort(key=lambda x: x["name"].lower())

        return maps

    except Exception as e:

        print(f"Map list error: {e}")
        return []



def get_storage_info():

    total, used, free = shutil.disk_usage(MAPS_DIR)

    return {
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2)
    }


def delete_map(map_name):
    base_map = SETTINGS["map"]["base_map"]
    if map_name == base_map:
        return False, "Protected map"
    file_path = MAPS_DIR / map_name
    if not file_path.exists():
        return False, "Map not found"
    file_path.unlink()
    return True, "Map deleted"


def set_map_active(
    map_name,
    active
):

    catalog = load_catalog()

    if map_name not in catalog:

        catalog[map_name] = {
            "description": "",
            "active": False
        }

    catalog[map_name]["active"] = active

    save_catalog(catalog)

    return True


def save_catalog(catalog):
    with open(CATALOG_FILE, "w") as f:
        json.dump(
            catalog,
            f,
            indent=2
        )


def update_map_description(
    map_name,
    description
):

    catalog = load_catalog()

    if map_name not in catalog:

        catalog[map_name] = {
            "description": "",
            "created": "",
            "source": ""
        }

    catalog[map_name]["description"] = description

    save_catalog(catalog)

    return True

