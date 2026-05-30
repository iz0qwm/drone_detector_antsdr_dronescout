import threading
import time
import uuid
from pathlib import Path
import math
import requests
from services.maps import load_map_provider
import sqlite3
import json


BASE_DIR = Path(__file__).resolve().parent.parent.parent

MAPS_DIR = BASE_DIR / "maps"
CATALOG_FILE = MAPS_DIR / "maps_catalog.json"

_download_jobs = {}
_download_lock = threading.Lock()

def load_catalog():

    if not CATALOG_FILE.exists():
        return {}

    try:

        with open(
            CATALOG_FILE,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def save_catalog(catalog):

    with open(
        CATALOG_FILE,
        "w"
    ) as f:

        json.dump(
            catalog,
            f,
            indent=2
        )


def create_download_job():

    job_id = str(uuid.uuid4())

    with _download_lock:

        _download_jobs[job_id] = {

            "status": "queued",
            "progress": 0,
            "current_tile": 0,
            "total_tiles": 0,
            "message": "Waiting",
            "created": time.time()
        }

    return job_id

def update_download_job(
    job_id,
    **kwargs
):

    with _download_lock:

        if job_id not in _download_jobs:
            return

        _download_jobs[job_id].update(kwargs)

def get_download_status(job_id):

    with _download_lock:

        return _download_jobs.get(job_id)


def list_downloads():

    with _download_lock:

        return _download_jobs



def remove_download_job(job_id):

    with _download_lock:

        if job_id in _download_jobs:
            del _download_jobs[job_id]



def deg2tile(lat_deg, lon_deg, zoom):

    lat_rad = math.radians(lat_deg)

    n = 2.0 ** zoom

    xtile = int(
        (lon_deg + 180.0) / 360.0 * n
    )

    ytile = int(
        (
            1.0 -
            math.asinh(
                math.tan(lat_rad)
            ) / math.pi
        ) / 2.0 * n
    )

    return xtile, ytile


# This is a simplified version that only calculates the tiles and simulates the download process.
def calculate_bbox(
    lat,
    lon,
    radius_km
):

    lat_delta = radius_km / 111.32

    lon_delta = (
        radius_km /
        (
            111.32 *
            math.cos(
                math.radians(lat)
            )
        )
    )

    return {

        "north":
            lat + lat_delta,

        "south":
            lat - lat_delta,

        "east":
            lon + lon_delta,

        "west":
            lon - lon_delta

    }


# Map provider configuration
def calculate_tiles(
    lat,
    lon,
    radius_km,
    zoom
):

    bbox = calculate_bbox(
        lat,
        lon,
        radius_km
    )

    x_min, y_max = deg2tile(
        bbox["north"],
        bbox["west"],
        zoom
    )

    x_max, y_min = deg2tile(
        bbox["south"],
        bbox["east"],
        zoom
    )

    tiles = []

    for x in range(
        min(x_min, x_max),
        max(x_min, x_max) + 1
    ):

        for y in range(
            min(y_min, y_max),
            max(y_min, y_max) + 1
        ):

            tiles.append(
                (
                    zoom,
                    x,
                    y
                )
            )

    return tiles

# This is a simplified version that only calculates the tiles and simulates the download process.
def build_tile_url(
    z,
    x,
    y
):
    provider = load_map_provider()

    api_key = provider.get(
            "api_key",
            ""
        )

    if not api_key:
        raise RuntimeError(
            "Thunderforest API key not configured"
        )
    return (
        "https://api.thunderforest.com/"
        f"outdoors/{z}/{x}/{y}.png"
        f"?apikey={api_key}"
    )


def create_mbtiles(filename):

    conn = sqlite3.connect(filename)

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE metadata (
            name TEXT,
            value TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE tiles (
            zoom_level INTEGER,
            tile_column INTEGER,
            tile_row INTEGER,
            tile_data BLOB
        )
    """)

    cur.execute("""
        CREATE UNIQUE INDEX tile_index
        ON tiles (
            zoom_level,
            tile_column,
            tile_row
        )
    """)

    conn.commit()

    return conn



def insert_tile(
    conn,
    z,
    x,
    y,
    tile_data
):

    tms_y = (
        (2 ** z - 1) - y
    )

    conn.execute(
        """
        INSERT OR REPLACE
        INTO tiles
        (
            zoom_level,
            tile_column,
            tile_row,
            tile_data
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            z,
            x,
            tms_y,
            tile_data
        )
    )





def download_tile(
    z,
    x,
    y
):

    url = build_tile_url(
        z,
        x,
        y
    )

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    return response.content


def get_zoom_range(radius_km):

    if radius_km <= 5:
        return 12, 16

    if radius_km <= 10:
        return 11, 15

    if radius_km <= 25:
        return 10, 14

    if radius_km <= 50:
        return 9, 13

    if radius_km <= 100:
        return 8, 12

    return 7, 11


def calculate_multi_zoom_tiles(
    lat,
    lon,
    radius_km
):

    min_zoom, max_zoom = (
        get_zoom_range(radius_km)
    )

    tiles = []

    for zoom in range(
        min_zoom,
        max_zoom + 1
    ):

        tiles.extend(
            calculate_tiles(
                lat,
                lon,
                radius_km,
                zoom
            )
        )

    return tiles


def create_map_filename(description):

    if not description:
        description = "map"

    safe_name = (
        description
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        f"{safe_name}_{timestamp}.mbtiles"
    )



def write_metadata(
    conn,
    lat,
    lon,
    radius_km,
    min_zoom,
    max_zoom,
    tile_count,
    description=""
):

    metadata = {

        "name":
            description or "Downloaded Map",

        "type":
            "baselayer",

        "format":
            "png",

        "center_lat":
            str(lat),

        "center_lon":
            str(lon),

        "radius_km":
            str(radius_km),

        "min_zoom":
            str(min_zoom),

        "max_zoom":
            str(max_zoom),

        "provider":
            "thunderforest_outdoors",

        "tile_count":
            str(tile_count),

        "created":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    for key, value in metadata.items():

        conn.execute(
            """
            INSERT INTO metadata
            (
                name,
                value
            )
            VALUES (?, ?)
            """,
            (
                key,
                value
            )
        )

    conn.commit()



def create_offline_map(
    lat,
    lon,
    radius_km,
    description="",
    job_id=None
):

    tiles = calculate_multi_zoom_tiles(
        lat,
        lon,
        radius_km
    )

    if job_id:

        update_download_job(
            job_id,
            total_tiles=len(tiles),
            current_tile=0,
            progress=0
        )

    print(
        f"Downloading {len(tiles)} tiles"
    )

    min_zoom, max_zoom = (
        get_zoom_range(radius_km)
    )

    filename = create_map_filename(
        description
    )

    catalog = load_catalog()

    catalog[filename] = {

        "description":
            description,

        "created":
            time.strftime(
                "%Y-%m-%d"
            ),

        "source":
            "thunderforest",

        "notes":
            "",

        "active":
            False
    }

    save_catalog(catalog)


    mbtiles_file = (
        MAPS_DIR /
        filename
    )

    conn = create_mbtiles(
        mbtiles_file
    )

    write_metadata(
        conn,
        lat,
        lon,
        radius_km,
        min_zoom,
        max_zoom,
        len(tiles),
        description
    )

    for idx, (z, x, y) in enumerate(
        tiles
    ):

        tile_data = download_tile(
            z,
            x,
            y
        )

        insert_tile(
            conn,
            z,
            x,
            y,
            tile_data
        )

        if idx % 25 == 0:

            print(
                f"{idx+1}/{len(tiles)} "
                f"{z}/{x}/{y}"
            )

        if idx % 100 == 0:

            conn.commit()

        if job_id:

            progress = int(
                ((idx + 1) / len(tiles)) * 100
            )

            update_download_job(
                job_id,
                current_tile=idx + 1,
                total_tiles=len(tiles),
                progress=progress
            )

    conn.commit()

    conn.close()

    return {
        "filename": filename,
        "tiles": len(tiles)
    }


def test_area():

    result = create_offline_map(
        lat=46.07,
        lon=11.12,
        radius_km=25,
        description="Trento Test"
    )

    print(result)




if __name__ == "__main__":
    test_area()