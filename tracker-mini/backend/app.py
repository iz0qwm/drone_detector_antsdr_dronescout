from flask import Flask, send_from_directory
from routes.status import status_bp
from routes.network import network_bp
from routes.network_manager import network_manager_bp
from routes.mode import mode_bp

app = Flask(__name__, static_folder="../frontend", static_url_path="")

app.register_blueprint(status_bp)
app.register_blueprint(network_bp)
app.register_blueprint(network_manager_bp)
app.register_blueprint(mode_bp)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)