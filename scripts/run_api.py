"""Start the PathSnap API on a configurable host and port."""

import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=os.getenv("PATHSNAP_HOST", "0.0.0.0"),
        port=int(os.getenv("PATHSNAP_PORT", "8000")),
        app_dir=str(ROOT),
        reload=False,
    )
