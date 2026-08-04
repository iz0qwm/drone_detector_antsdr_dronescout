"""
Mini Tracker test configuration.

Tests are development-only and not deployed by the updater.
"""
import sys
from pathlib import Path

# Add backend to path so services can be imported
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = WORKSPACE_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
