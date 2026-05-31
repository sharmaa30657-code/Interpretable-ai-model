import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from backend.app import app
except ImportError as exc:
    raise SystemExit(
        "This project uses backend/app.py for the web server. "
        "Install dependencies from backend/requirements.txt and run `python backend/app.py`. "
        f"Import error: {exc}"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
