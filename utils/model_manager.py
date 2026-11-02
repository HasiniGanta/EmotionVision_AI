"""
utils/model_manager.py

Responsible for:
1. Checking whether the pretrained emotion model exists locally.
2. Downloading it automatically (with fallback mirrors) if missing.
3. Validating the downloaded file.
4. Loading it into memory exactly once (cached).
5. Reporting clear, human-readable errors on failure.

No dataset, no API key, no paid service is required.
"""

import os
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_CACHE = {"model": None}


class ModelManagerError(Exception):
    """Raised when the model cannot be obtained or loaded."""


def _download_with_progress(url: str, dest_path: str, timeout: int = 60) -> None:
    logger.info(f"Downloading model from {url}")
    tmp_path = dest_path + ".part"

    req = urllib.request.Request(url, headers={"User-Agent": "EmotionVisionAI/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response, open(tmp_path, "wb") as out_file:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 65536
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                logger.info(f"Download progress: {pct:.1f}%")

    os.replace(tmp_path, dest_path)


def _validate_model_file(path: str) -> bool:
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < config.MODEL_MIN_EXPECTED_BYTES:
        logger.warning(f"Model file too small ({size} bytes) - likely corrupt/incomplete.")
        return False
    return True


def ensure_model_downloaded(progress_callback=None) -> str:
    """
    Ensures the emotion model exists locally. Downloads it if missing.
    Returns the local file path. Raises ModelManagerError on failure.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = config.MODEL_PATH

    if _validate_model_file(path):
        logger.info("Local emotion model found and validated. Skipping download.")
        if progress_callback:
            progress_callback("Model already available locally.")
        return path

    if progress_callback:
        progress_callback("Emotion model not found. Downloading model...")
    logger.info("Emotion model not found locally. Attempting download...")

    last_error = None
    for source_url in config.MODEL_SOURCES:
        try:
            _download_with_progress(source_url, path)
            if _validate_model_file(path):
                logger.info("Model downloaded successfully.")
                if progress_callback:
                    progress_callback("Model downloaded successfully.")
                return path
            else:
                last_error = "Downloaded file failed validation (unexpected size)."
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_error = str(e)
            logger.warning(f"Download attempt failed from {source_url}: {e}")
            continue

    # All sources failed
    if os.path.exists(path):
        os.remove(path)
    msg = (
        "Could not download the emotion recognition model automatically. "
        f"Last error: {last_error}. "
        "Please check your internet connection, or manually place a valid model file at "
        f"'{path}', or update MODEL_SOURCES in config.py with a working URL."
    )
    logger.error(msg)
    raise ModelManagerError(msg)


def load_model(progress_callback=None):
    """
    Loads (and caches) the Keras emotion model. Downloads it first if needed.
    """
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"]

    path = ensure_model_downloaded(progress_callback=progress_callback)

    try:
        # Imported lazily so the rest of the app can start even if TF
        # is briefly unavailable (e.g., during setup diagnostics).
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        if progress_callback:
            progress_callback("Loading model into memory...")
        start = time.time()
        model = tf.keras.models.load_model(path, compile=False)
        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.2f}s from {path}")
        _MODEL_CACHE["model"] = model
        return model
    except Exception as e:
        msg = f"Failed to load the emotion model from '{path}': {e}"
        logger.error(msg)
        raise ModelManagerError(msg)


def get_model_status() -> dict:
    """Returns a dict describing current model availability, for UI display."""
    exists = _validate_model_file(config.MODEL_PATH)
    return {
        "exists_locally": exists,
        "path": config.MODEL_PATH,
        "size_bytes": os.path.getsize(config.MODEL_PATH) if exists else 0,
        "loaded_in_memory": _MODEL_CACHE["model"] is not None,
        "name": config.MODEL_NAME,
        "type": config.MODEL_TYPE,
        "framework": config.MODEL_FRAMEWORK,
        "input_size": config.MODEL_INPUT_SIZE,
        "source_url": config.MODEL_SOURCE_URL,
        "license": config.MODEL_LICENSE,
        "reported_val_accuracy": config.MODEL_REPORTED_VAL_ACCURACY,
        "emotions": config.EMOTION_LABELS,
    }
