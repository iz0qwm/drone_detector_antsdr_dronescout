from flask import Flask, send_from_directory
from pathlib import Path
from routes.status import status_bp
from routes.network import network_bp
from routes.network_manager import network_manager_bp
from routes.settings import settings_bp
from routes.maps import maps_bp
from routes.missions import missions_bp
from routes.services import (services_bp)
from routes.air_network import air_network_bp
from routes.ogn_network import ogn_network_bp
from routes.hardware import (hardware_bp)
from routes.ds110 import ds110_bp
from routes.gps import gps_bp
from routes.readsb import readsb_bp
from routes.meshtastic import meshtastic_bp
from routes.notifications import notifications_bp
from services.ui.lcd import lcd
from services.readsb import start as start_readsb


app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

print(
    "MAX_CONTENT_LENGTH =",
    app.config["MAX_CONTENT_LENGTH"]
)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

from services.network_manager import start_hotspot
from services.ds110 import start as start_ds110
from routes.remoteid import remoteid_bp
from routes.logs import logs_bp
from routes.update import update_bp
from routes.dsc import dsc_bp
from services.dsc_heartbeat import (start_dsc_heartbeat)
from routes.air_local import air_local_bp
from routes.teams import teams_bp
from services.meshtastic_service import start as start_meshtastic

from config import SETTINGS

HELP_DIR = (
    Path(__file__).parent.parent
    / "frontend"
    / "help"
    / "site"
)


try:
    print("Starting local hotspot...")
    result = start_hotspot()
    print(result)
except Exception as e:
    print(f"Hotspot startup error: {e}")

# Start services based on settings
# RID
try:
    if SETTINGS.get(
        "traffic",
        {}
    ).get(
        "remoteid_enabled",
        True
    ):
        print("Starting DS110 service...")
        start_ds110()

except Exception as e:
    print(f"DS110 startup error: {e}")

# Meshtastic
try:

    print("Starting Meshtastic service...")

    start_meshtastic()

except Exception as e:
    print(f"Meshtastic startup error: {e}")

# ReadSB    
try:

    if SETTINGS.get(
        "traffic",
        {}
    ).get(
        "adsb_local_enabled",
        True
    ):

        print("Starting ADS-B receiver...")

        start_readsb()

except Exception as e:
    print(f"ADS-B startup error: {e}")




try:
    print("Starting DSC heartbeat...")
    start_dsc_heartbeat()
except Exception as e:
    print(
        f"DSC heartbeat startup error: {e}"
    )

try:
    print("Starting LCD service...")
    lcd.start()
except Exception as e:
    print(f"LCD startup error: {e}")


app.register_blueprint(
    update_bp,
    url_prefix="/api/update"
)

app.register_blueprint(status_bp)
app.register_blueprint(network_bp)
app.register_blueprint(network_manager_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(maps_bp)
app.register_blueprint(missions_bp)
app.register_blueprint(services_bp)
app.register_blueprint(air_network_bp)
app.register_blueprint(ogn_network_bp)
app.register_blueprint(hardware_bp)
app.register_blueprint(remoteid_bp)
app.register_blueprint(logs_bp)
app.register_blueprint(dsc_bp)
app.register_blueprint(ds110_bp)
app.register_blueprint(gps_bp, url_prefix="/api/gps")
app.register_blueprint(air_local_bp)
app.register_blueprint(readsb_bp)
app.register_blueprint(meshtastic_bp)
app.register_blueprint(teams_bp)
app.register_blueprint(notifications_bp)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/help/")
def help_index():

    return send_from_directory(
        HELP_DIR,
        "index.html"
    )

@app.route("/help/<path:path>")
def help_files(path):

    full_path = HELP_DIR / path

    if full_path.is_dir():

        return send_from_directory(
            full_path,
            "index.html"
        )

    if full_path.exists():

        return send_from_directory(
            HELP_DIR,
            path
        )

    index_file = HELP_DIR / path / "index.html"

    if index_file.exists():

        return send_from_directory(
            HELP_DIR / path,
            "index.html"
        )

    return (
        "Not Found",
        404
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)