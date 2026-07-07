import requests
from datetime import datetime
from config import SETTINGS
from services.network import get_network_status
from services.logger import log

DEFAULT_ZONES_URL = (
    "https://zones-32dg4v266a-uc.a.run.app/"
)


def dsc_config():
    return SETTINGS.get(
        "dsc_zones",
        {}
    )


def is_dsc_configured():
    cfg = dsc_config()

    return bool(
        cfg.get("enabled", False)
        and cfg.get("api_key")
    )


def check_dsc_ready():
    if not is_dsc_configured():
        return False, "DSC API key not configured"

    network = get_network_status()

    if not network.get("internet"):
        return False, "Internet connection not available"

    return True, None


def fetch_zones(
    bbox,
    simplify=True,
    limit=500,
    zone_type=None
):
    ok, message = check_dsc_ready()

    if not ok:
        return {
            "success": False,
            "message": message
        }

    cfg = dsc_config()

    url = cfg.get(
        "zones_url",
        DEFAULT_ZONES_URL
    )

    params = {
        "bbox": ",".join(
            str(x) for x in bbox
        ),
        "simplify": (
            "true" if simplify else "false"
        ),
        "limit": limit
    }

    if zone_type:
        params["type"] = zone_type

    headers = {
        "x-api-key": cfg["api_key"]
    }

    try:

        log(
            "DSC_ZONES",
            "Download requested",
            f"bbox={bbox}",
            f"limit={limit}",
            f"simplify={simplify}"
        )
                
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20
        )

        log(
            "DSC_ZONES",
            f"HTTP {response.status_code}"
        )
        
        if response.status_code == 401:
            return {
                "success": False,
                "message": "Invalid DSC API key"
            }

        if response.status_code == 429:
            return {
                "success": False,
                "message": "DSC API rate limit exceeded"
            }

        response.raise_for_status()

        geojson = response.json()

        features = len(
            geojson.get(
                "features",
                []
            )
        )

        log(
            "DSC_ZONES",
            "Downloaded",
            f"{features} features"
        )

        return {
            "success": True,
            "geojson": geojson,
            "metadata": {
                "source": "dsc_zones",
                "dataset": "zones",
                "downloaded": (
                    datetime.utcnow()
                    .isoformat()
                ),
                "bbox": bbox,
                "simplify": simplify,
                "limit": limit,
                "zone_type": zone_type,
                "count": geojson.get("count"),
                "truncated": geojson.get("truncated")
            }
        }

    except requests.Timeout:
        return {
            "success": False,
            "message": "DSC request timeout"
        }

    except Exception as e:

        log(
            "DSC_ZONES",
            str(e),
            level="ERROR"
        )
        
        return {
            "success": False,
            "message": f"DSC request failed: {e}"
        }