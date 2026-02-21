import sys
import os
import traceback

# Force write to a log file we can read
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startup_log.txt")

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

# Clear old log
open(LOG, "w").close()

log(f"Python: {sys.executable}")
log(f"Version: {sys.version}")
log("")

# Test imports one by one
missing = []

try:
    import cv2
    log(f"[OK] cv2 {cv2.__version__}")
except ImportError as e:
    log(f"[FAIL] cv2: {e}")
    missing.append("opencv-python")

try:
    import flask
    log(f"[OK] flask {flask.__version__}")
except ImportError as e:
    log(f"[FAIL] flask: {e}")
    missing.append("flask")

try:
    import ultralytics
    log(f"[OK] ultralytics")
except ImportError as e:
    log(f"[FAIL] ultralytics: {e}")
    missing.append("ultralytics")

try:
    import numpy as np
    log(f"[OK] numpy {np.__version__}")
except ImportError as e:
    log(f"[FAIL] numpy: {e}")
    missing.append("numpy")

try:
    from PIL import Image
    log(f"[OK] Pillow")
except ImportError as e:
    log(f"[FAIL] Pillow: {e}")
    missing.append("Pillow")

log("")

if missing:
    log(f"MISSING packages: {', '.join(missing)}")
    log(f"Run: pip install {' '.join(missing)}")
else:
    log("All imports OK! Starting Flask server...")
    log("=" * 40)
    log("Open http://localhost:5000 in your browser")
    log("=" * 40)
    log("")

    # Now actually run the app
    try:
        # Import and run
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Temporarily redirect to log exceptions
        import app as flask_app
        flask_app.app.run(debug=False, host="0.0.0.0", port=5000)
    except Exception as e:
        log(f"\nFLASK CRASH:\n{traceback.format_exc()}")
