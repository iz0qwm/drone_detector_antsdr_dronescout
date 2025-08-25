#!/usr/bin/env python3
# 02b_make_classmap.py — zero-config by default
# Se lanciato senza argomenti:
#   data.yaml  = /home/raffaello/dataset/yolo_custom/data.yaml
#   out.json   = /home/raffaello/apprendimento/class_map.json
# Opzioni (facoltative):
#   --exclude "Noise,Background"     classi da escludere dai video_ids
#   --video-ids "0,1"                forza gli ID video (disabilita auto)
#   --no-video-auto                  niente video_ids automatici

import sys, json, pathlib, argparse
try:
    import yaml
except ImportError:
    print("PyYAML non installato. Installa con: pip install pyyaml", file=sys.stderr)
    raise

DEFAULT_DATA_YAML = "/home/raffaello/dataset/yolo_custom/data.yaml"
DEFAULT_OUT_JSON  = "/home/raffaello/apprendimento/class_map.json"
DEF_EXCLUDES = {"Noise", "Background", "Background_Noise"}

def load_names(path_yaml):
    with open(path_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    # YOLO può fornire una lista o un dict {id:name}
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not isinstance(names, (list, tuple)):
        raise SystemExit("data.yaml non contiene 'names' come lista/dict")
    return list(names)

def build_maps(names):
    by_id   = {str(i): n for i, n in enumerate(names)}
    by_name = {n: i for i, n in enumerate(names)}
    return by_id, by_name

def parse_video_ids_arg(arg, max_id):
    if not arg:
        return None
    vids = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            raise SystemExit(f"--video-ids contiene un id non numerico: {tok!r}")
        if v < 0 or v > max_id:
            raise SystemExit(f"--video-ids fuori range [0..{max_id}]: {v}")
        vids.append(str(v))
    return vids

def build_video_ids(names, excludes, force_ids=None, auto=True):
    if force_ids is not None:
        return force_ids
    if not auto:
        return []
    return [str(i) for i, name in enumerate(names) if name not in excludes]

def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]

    # Parser “morbido”: i due posizionali sono opzionali
    ap = argparse.ArgumentParser(
        description="Genera class_map.json da data.yaml",
        add_help=True
    )
    ap.add_argument("data_yaml", nargs="?", default=DEFAULT_DATA_YAML,
                    help=f"Percorso a data.yaml (default: {DEFAULT_DATA_YAML})")
    ap.add_argument("out_json", nargs="?", default=DEFAULT_OUT_JSON,
                    help=f"Percorso di output (default: {DEFAULT_OUT_JSON})")
    ap.add_argument("--exclude", default=",".join(sorted(DEF_EXCLUDES)),
                    help=f"Classi escluse da video_ids (default: {','.join(sorted(DEF_EXCLUDES))})")
    ap.add_argument("--video-ids", default=None,
                    help='Lista esplicita di ID video (es. "0,1"). Se presente disabilita auto.')
    ap.add_argument("--no-video-auto", action="store_true",
                    help="Non generare video_ids automaticamente (se non usi --video-ids).")
    args = ap.parse_args(argv)

    names = load_names(args.data_yaml)
    by_id, by_name = build_maps(names)

    excludes = {x.strip() for x in (args.exclude or "").split(",") if x.strip()}
    force_ids = parse_video_ids_arg(args.video_ids, max_id=len(names)-1)
    video_ids = build_video_ids(names, excludes, force_ids=force_ids, auto=(not args.no_video_auto))

    out = {"by_id": by_id, "by_name": by_name, "video_ids": video_ids}

    out_path = pathlib.Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✅ Saved classmap to: {out_path}")
    print(f"   classes: {len(names)} → {names}")
    print(f"   video_ids: {video_ids} (exclude={sorted(excludes)})")
    print(f"   data_yaml: {args.data_yaml}")

if __name__ == "__main__":
    main()
