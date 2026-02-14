# bridge_web.py
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import socket
import requests

CORE_BASE = "http://127.0.0.1:8090/api"


app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/drones")
def get_drones():
    data = requests.get(f"{CORE_BASE}/drones").json()
    return jsonify(data)

@app.route("/logs")
def get_logs():
    data = requests.get(f"{CORE_BASE}/logs").json()
    return jsonify(data)

@app.route("/receiver")
def get_receiver():
    data = requests.get(f"{CORE_BASE}/receiver").json()
    return jsonify(data)

@app.route("/sources")
def get_sources():
    data = requests.get(f"{CORE_BASE}/sources").json()
    return jsonify(data)


@app.route("/health")
def get_health():
    return jsonify({
        "service": "bridge_web",
        "host": socket.gethostname(),
        "time": datetime.utcnow().isoformat() + "Z",
        "core": CORE_BASE
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
