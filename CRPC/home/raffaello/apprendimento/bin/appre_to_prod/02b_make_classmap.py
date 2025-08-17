#!/usr/bin/env python3
import sys, json, pathlib, yaml
def main(path_yaml, out_json):
    data = yaml.safe_load(open(path_yaml, 'r'))
    names = data.get('names')
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not isinstance(names, (list, tuple)):
        raise SystemExit("data.yaml does not contain a 'names' list/dict")
    by_id   = {str(i): n for i, n in enumerate(names)}
    by_name = {n: i for i, n in enumerate(names)}
    out = {'by_id': by_id, 'by_name': by_name}
    pathlib.Path(out_json).write_text(json.dumps(out, indent=2))
    print(f"Saved classmap to: {out_json}")
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: 02b_make_classmap.py <data.yaml> <out.json>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
