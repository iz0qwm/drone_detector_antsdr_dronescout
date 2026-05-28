from flask import Flask, send_from_directory
from routes.status import status_bp
from routes.network import network_bp
from routes.network_manager import network_manager_bp
from routes.settings import settings_bp
from routes.maps import maps_bp

app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

from services.network_manager import start_hotspot

try:
    print("Starting local hotspot...")
    result = start_hotspot()
    print(result)
except Exception as e:
    print(f"Hotspot startup error: {e}")



app.register_blueprint(status_bp)
app.register_blueprint(network_bp)
app.register_blueprint(network_manager_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(maps_bp)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)