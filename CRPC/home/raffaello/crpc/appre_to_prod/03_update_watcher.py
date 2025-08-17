#!/usr/bin/env python3
import re, sys, pathlib
from re import sub, escape
def patch_watcher(path, new_model, new_data):
    p = pathlib.Path(path)
    txt = p.read_text()
    txt2 = sub(r'(--model\s+)(\S+)', r'\1' + escape(new_model), txt)
    txt3 = sub(r'(--data\s+)(\S+)',  r'\1' + escape(new_data), txt2)
    if txt3 != txt:
        backup = p.with_suffix(p.suffix + ".bak")
        backup.write_text(txt)
        p.write_text(txt3)
        print(f"Patched {p} (backup at {backup})")
    else:
        print("No changes applied. Patterns not found or already set.")
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: 03_update_watcher.py <yolo_watcher.py> <weights.pt> <data.yaml>")
        sys.exit(2)
    patch_watcher(sys.argv[1], sys.argv[2], sys.argv[3])
