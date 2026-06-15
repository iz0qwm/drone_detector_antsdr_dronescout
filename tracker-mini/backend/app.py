from flask import Flask, send_from_directory
from routes.status import status_bp
from routes.network import network_bp
from routes.network_manager import network_manager_bp
from routes.settings import settings_bp
from routes.maps import maps_bp
from routes.missions import missions_bp
from routes.services import (
    services_bp
)
from routes.air_network import air_network_bp
from routes.ogn_network import ogn_network_bp
from routes.hardware import (
    hardware_bp
)

app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

from services.network_manager import start_hotspot
from services.ds110 import start as start_ds110
from routes.remoteid import remoteid_bp
from routes.logs import logs_bp
from routes.update import update_bp
from routes.dsc import dsc_bp
from services.dsc_heartbeat import (
    start_dsc_heartbeat
)

try:
    print("Starting local hotspot...")
    result = start_hotspot()
    print(result)
except Exception as e:
    print(f"Hotspot startup error: {e}")


try:
    print("Starting DS110 service...")
    start_ds110()
except Exception as e:
    print(f"DS110 startup error: {e}")

try:
    print("Starting DSC heartbeat...")
    start_dsc_heartbeat()
except Exception as e:
    print(
        f"DSC heartbeat startup error: {e}"
    )

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

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)