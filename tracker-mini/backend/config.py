import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SETTINGS_FILE = BASE_DIR / "config" / "settings.json"

with open(SETTINGS_FILE, "r") as f:
    SETTINGS = json.load(f)