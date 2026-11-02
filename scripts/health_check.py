"""
scripts/health_check.py

Prints a quick OK/FAIL summary of every subsystem. Can be run standalone:
    python scripts/health_check.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_health_check():
    import config

    results = {}

    # Python
    results["Python"] = "OK" if sys.version_info >= (3, 9) else "WARNING (upgrade recommended)"

    # OpenCV
    try:
        import cv2
        results["OpenCV"] = f"OK ({cv2.__version__})"
    except Exception as e:
        results["OpenCV"] = f"FAIL ({e})"

    # TensorFlow
    try:
        import tensorflow as tf
        results["TensorFlow"] = f"OK ({tf.__version__})"
    except Exception as e:
        results["TensorFlow"] = f"FAIL ({e})"

    # Model
    try:
        from utils.model_manager import get_model_status
        status = get_model_status()
        results["Model"] = "OK" if status["exists_locally"] else "NOT DOWNLOADED (run scripts/setup.py)"
    except Exception as e:
        results["Model"] = f"FAIL ({e})"

    # Database
    try:
        from utils import database
        database.init_db()
        results["Database"] = "OK" if os.path.exists(config.DATABASE_PATH) else "FAIL (not created)"
    except Exception as e:
        results["Database"] = f"FAIL ({e})"

    # Camera (cannot be verified headlessly/reliably -> flagged for manual check)
    results["Camera"] = "CHECK REQUIRED (run the app and click START CAMERA)"

    # Storage
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        test_path = os.path.join(config.DATA_DIR, ".write_test")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        results["Storage"] = "OK"
    except Exception as e:
        results["Storage"] = f"FAIL ({e})"

    print("=" * 50)
    print("EmotionVision AI Health Check")
    print("=" * 50)
    for k, v in results.items():
        print(f"{k}: {v}")
    print("=" * 50)

    if all("FAIL" not in v for v in results.values()):
        print("System ready.")
    else:
        print("Some checks failed — see above.")

    return results


if __name__ == "__main__":
    run_health_check()
