import sqlite3
import os
from pathlib import Path
from config import SETTINGS
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent.parent

MAPS_DIR = BASE_DIR / "maps"


def get_tile(z, x, y):

    try:

        tms_y = (2 ** z - 1) - y

        active_maps = SETTINGS["map"]["active_maps"]

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

        active_maps = SETTINGS["map"]["active_maps"]
        base_map = SETTINGS["map"]["base_map"]

        for file in MAPS_DIR.glob("*.mbtiles"):

            size_mb = round(
                file.stat().st_size / (1024 * 1024),
                2
            )

            maps.append({
                "name": file.name,
                "size_mb": size_mb,
                "active": file.name in active_maps,
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