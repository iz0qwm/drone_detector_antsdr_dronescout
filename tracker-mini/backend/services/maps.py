import sqlite3
from pathlib import Path

from config import SETTINGS

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