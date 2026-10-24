
import json
from pathlib import Path

path = Path("crack-seg.ndjson")

if not path.exists():
    print("NDJSON file not found in project root.")
    raise SystemExit(1)

with path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 3:
            break

        try:
            record = json.loads(line)
            print(f"\nRecord {i + 1}:")
            print(json.dumps(record, indent=2)[:3000])
        except json.JSONDecodeError as e:
            print("Invalid JSON on line", i + 1, e)
