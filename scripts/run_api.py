"""Start the PathSnap API on a configurable host and port."""

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=os.getenv("PATHSNAP_HOST", "0.0.0.0"),
        port=int(os.getenv("PATHSNAP_PORT", "8000")),
        reload=False,
    )
