"""Export the FastAPI OpenAPI schema to JSON (D15).

Run:  venv/bin/python scripts/export_openapi.py
Then generate TS types in the frontend with:
  npx openapi-typescript ../backend/openapi.json -o src/types/api.generated.ts
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.server import app  # noqa: E402

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openapi.json")
with open(out, "w") as f:
    json.dump(app.openapi(), f, indent=2)
print(f"Wrote OpenAPI schema → {out}")
print(f"Endpoints: {len(app.openapi().get('paths', {}))}")
