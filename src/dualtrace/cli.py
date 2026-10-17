"""Offline content inference and serving; no test-ID or CSV answer lookup."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

from .paths import checkpoints


def read_rows(path, track):
    field = "path" if track == "image" else "text"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", field}.issubset(reader.fieldnames or []):
            raise ValueError(f"Input CSV requires id,{field}")
        rows = list(reader)
    ids = [r["id"] for r in rows]
    if not rows or any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Input must contain nonempty, unique IDs and at least one row")
    for r in rows:
        if r.get(field) is None:
            raise ValueError(f"Missing {field} in row {r['id']}")
        if track == "image":
            p = Path(r[field])
            r[field] = str(p if p.is_absolute() else path.resolve().parent / p)
            if not Path(r[field]).is_file():
                raise FileNotFoundError(r[field])
    return rows


def predict(args):
    rows = read_rows(args.input, args.track)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.track == "image":
        from .image.v696_runtime import V696Runtime
        from PIL import Image
        runtime = V696Runtime()
    else:
        from .text.runtime import TV388Runtime
        runtime = TV388Runtime()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".dualtrace-", suffix=".csv", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "score"])
            for i, row in enumerate(rows):
                if args.track == "image":
                    with Image.open(row["path"]) as image:
                        image.load()
                        score, _ = runtime.predict(image)
                else:
                    score, _ = runtime.predict(row["text"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise RuntimeError(f"Invalid score for {row['id']}")
                writer.writerow([row["id"], format(score, ".17g")])
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    print(json.dumps({"done": i + 1, "total": len(rows)}), flush=True)
        os.replace(temporary, args.output)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, help="Override DUALTRACE_CHECKPOINTS")
    sub = p.add_subparsers(dest="command", required=True)
    inf = sub.add_parser("predict", help="Score new content into an id,score CSV")
    inf.add_argument("--track", choices=["image", "text"], required=True)
    inf.add_argument("--input", type=Path, required=True)
    inf.add_argument("--output", type=Path, required=True)
    serve = sub.add_parser("serve", help="Serve the preserved HTTP protocol")
    serve.add_argument("--track", choices=["image", "text"], required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8080, type=int)
    sub.add_parser("paths", help="Print asset location without importing GPU libraries")
    args = p.parse_args()
    if args.checkpoints:
        os.environ["DUALTRACE_CHECKPOINTS"] = str(args.checkpoints.resolve())
    if args.command == "paths":
        print(checkpoints())
    elif args.command == "predict":
        predict(args)
    else:
        import uvicorn
        uvicorn.run(f"dualtrace.{args.track}.server:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
