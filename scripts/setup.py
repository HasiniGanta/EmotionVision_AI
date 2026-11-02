"""
scripts/setup.py

One-time (idempotent) setup script:
1. Checks Python version.
2. Creates required directories.
3. Creates the SQLite database.
4. Downloads the pretrained emotion model automatically.
5. Verifies OpenCV / TensorFlow.
6. Runs a health check.

Run with:  python scripts/setup.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_python_version():
    print("Checking Python version...")
    major, minor = sys.version_info[0], sys.version_info[1]
    if (major, minor) < (3, 9):
        print(f"❌ Python {major}.{minor} detected. Python 3.10 or 3.11 is recommended.")
    else:
        print(f"✅ Python {major}.{minor} OK")


def create_directories():
    import config
    print("Creating required directories...")
    for d in [config.MODELS_DIR, config.DATA_DIR, config.SNAPSHOTS_DIR, config.LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories ready")


def init_database():
    print("Initializing SQLite database...")
    from utils import database
    database.init_db()
    print("✅ Database ready")


def verify_opencv():
    print("Verifying OpenCV...")
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} OK")
        return True
    except Exception as e:
        print(f"❌ OpenCV not available: {e}")
        return False


def verify_tensorflow():
    print("Verifying TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} OK")
        return True
    except Exception as e:
        print(f"❌ TensorFlow not available: {e}")
        return False


def download_model():
    print("Downloading pretrained emotion model (if not already present)...")
    from utils.model_manager import ensure_model_downloaded, ModelManagerError
    try:
        path = ensure_model_downloaded(progress_callback=print)
        print(f"✅ Model available at {path}")
        return True
    except ModelManagerError as e:
        print(f"❌ Model download failed: {e}")
        return False


def main():
    print("=" * 60)
    print("EmotionVision AI - Setup")
    print("=" * 60)

    check_python_version()
    create_directories()
    init_database()
    ok_cv = verify_opencv()
    ok_tf = verify_tensorflow()
    ok_model = download_model() if ok_tf else False

    print()
    print("=" * 60)
    if ok_cv and ok_tf and ok_model:
        print("✅ Setup complete. You can now run:  streamlit run app.py")
    else:
        print("⚠️ Setup completed with warnings. See messages above.")
        print("Fix any ❌ items, then re-run:  python scripts/setup.py")
    print("=" * 60)

    print()
    print("Running health check...")
    from scripts.health_check import run_health_check
    run_health_check()


if __name__ == "__main__":
    main()
