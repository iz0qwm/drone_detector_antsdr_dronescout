# bridge_web.py
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import socket
import requests
import subprocess
import os


CORE_BASE = "http://127.0.0.1:8090/api"

SERVICES = [
    "bridge-core.service",
    "bridge-web.service",
    "dji-receiver.service",
    "remotetrack.service",
    "ogn-rf.service",
    "ogn-decode.service",
    "aprs-local.service",
    "mm-socat.service",
    "mm2",
    "dump1090-fa",
    "tailscaled",
    "lighttpd"
]

LOG_DIR = "/home/pi/bridge/logs"



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

@app.route("/coverage")
def get_coverage():
    data = requests.get(f"{CORE_BASE}/coverage").json()
    return jsonify(data)

    
@app.route("/health")
def get_health():
    return jsonify({
        "service": "bridge_web",
        "host": socket.gethostname(),
        "time": datetime.utcnow().isoformat() + "Z",
        "core": CORE_BASE
    })

@app.route("/services")
def get_services():
    result = {}

    for s in SERVICES:

        try:
            out = subprocess.run(
                ["systemctl", "is-active", s],
                capture_output=True,
                text=True
            )
            status = out.stdout.strip()
        except:
            status = "unknown"

        # controllo reale per ogn
        if s == "ogn-decode.service":
            if not process_running("ogn-decode"):
                status = "dead"

        if s == "ogn-rf.service":
            if not process_running("ogn-rf"):
                status = "dead"

        result[s] = status

    return jsonify(result)

@app.route("/service/restart/<name>")
def restart_service(name):
    if name not in SERVICES:
        return jsonify({"error": "unknown service"}), 400

    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", name],
            capture_output=True
        )
        return jsonify({"status":"ok","service":name})
    except Exception as e:
        return jsonify({"status":"error","msg":str(e)})



@app.route("/logfiles")
def list_logs():
    try:
        files = os.listdir(LOG_DIR)
        files = sorted(files)
        return jsonify(files)
    except Exception as e:
        return jsonify({"error":str(e)})


@app.route("/logfile/<name>")
def read_log(name):

    path = os.path.join(LOG_DIR, name)

    if not os.path.isfile(path):
        return jsonify({"error":"not found"}),404

    try:
        with open(path) as f:
            lines = f.readlines()

        return jsonify({
            "file": name,
            "lines": lines[-300:]
        })

    except Exception as e:
        return jsonify({"error":str(e)})



@app.route('/favicon.ico')
def favicon():
    return "",204

def process_running(name):
    try:
        out = subprocess.run(
            ["pgrep", "-f", name],
            capture_output=True,
            text=True
        )
        return out.stdout.strip() != ""
    except:
        return False



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
