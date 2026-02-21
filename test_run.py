import sys, traceback, os

log_path = os.path.join(os.path.dirname(__file__), "diag_log.txt")

with open(log_path, "w") as f:
    f.write(f"Python: {sys.executable}\n")
    f.write(f"Version: {sys.version}\n\n")

    checks = [
        ("cv2",        "import cv2; f.write(f'cv2: {cv2.__version__}\\n')"),
        ("flask",      "import flask; f.write(f'flask: {flask.__version__}\\n')"),
        ("ultralytics","import ultralytics; f.write('ultralytics: OK\\n')"),
        ("numpy",      "import numpy as np; f.write(f'numpy: {np.__version__}\\n')"),
        ("PIL",        "from PIL import Image; f.write('PIL: OK\\n')"),
    ]

    all_ok = True
    for name, stmt in checks:
        try:
            exec(stmt, {"f": f})
        except Exception as e:
            f.write(f"MISSING {name}: {e}\n")
            all_ok = False

    if all_ok:
        f.write("\nALL IMPORTS OK — starting app...\n")
        f.flush()
        # Now actually start the flask app
        try:
            exec(open(os.path.join(os.path.dirname(__file__), "app.py")).read())
        except Exception as e:
            f.write(f"\nAPP CRASH:\n{traceback.format_exc()}\n")
    else:
        f.write("\nSome packages are missing — cannot start.\n")

print("Done — check diag_log.txt")
