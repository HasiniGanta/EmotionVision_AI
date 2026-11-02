# EmotionVision AI

**Real-Time Facial Emotion Intelligence** — a complete, self-contained AI computer-vision
application built with OpenCV, TensorFlow/Keras, and Streamlit.

EmotionVision AI detects human facial emotions in real time from your webcam, from
uploaded images, and from uploaded videos — using a genuine pretrained deep learning
model that is downloaded automatically the first time you run the app. No dataset,
no API key, no paid service, and no manual model download required.

---

## 1. Project Overview

EmotionVision AI is a modular Streamlit web application that:

- Detects faces in a live webcam feed, an uploaded image, or an uploaded video
- Classifies each detected face into one of 7 emotions using a real CNN
- Displays bounding boxes, emotion labels, and confidence scores
- Tracks real statistics (total detections, response time, emotion distribution,
  trends) in a local SQLite database — nothing is fabricated
- Provides history browsing, filtering, CSV/JSON export, and analytics
- Includes an optional advanced local model-training module

Supported emotions: **Happy 😊 · Sad 😢 · Angry 😠 · Surprise 😮 · Fear 😨 · Disgust 🤢 · Neutral 😐**

---

## 2. Features

- 🎥 Real-time webcam detection with START/STOP controls
- 🖼️ Image upload analysis (JPG/JPEG/PNG)
- 🎬 Video upload analysis with configurable processing FPS
- 👥 Multiple simultaneous face support
- 📊 Live dashboard: status, total detections, model accuracy, avg response time
- 🍩 Emotion distribution donut chart (from real data)
- 🕒 Recent detections feed
- 📈 Real-time emotion trend line chart
- 🗂️ Filterable detection history (date, session, emotion, confidence)
- ⬇️ CSV export and JSON session reports
- 🧩 Model Information page (source, license, benchmark accuracy — never fabricated)
- ⚙️ Settings page (camera index, FPS, confidence threshold, alerts, theme)
- 📘 "Learn" page explaining the pipeline and core ML concepts
- 🧪 Demo Mode — clearly labeled simulated data for UI preview without a webcam
- 📸 Snapshot saving
- 🔒 Local-only processing — no webcam frames leave your machine
- ✅ Automated test suite (23 tests, no physical webcam required)

---

## 3. Architecture

```
Webcam / Image / Video
        │
        ▼
  Face Detection (OpenCV Haar Cascade)
        │
        ▼
  Face Crop + Preprocessing (grayscale, 64x64, normalized)
        │
        ▼
  Emotion Model (mini-XCEPTION CNN, TensorFlow/Keras)
        │
        ▼
  Emotion Probabilities → Predicted Emotion + Confidence
        │
        ▼
  Draw Bounding Box + Label
        │
        ▼
  Store Detection (SQLite + CSV)
        │
        ▼
  Dashboard / Charts / Statistics (Streamlit + Plotly)
```

---

## 4. Technology Stack

| Layer            | Technology                     |
|-------------------|---------------------------------|
| Web application   | Streamlit                      |
| Computer vision   | OpenCV                         |
| Deep learning     | TensorFlow / Keras             |
| Data              | NumPy, Pandas                  |
| Visualization     | Plotly                         |
| Image handling    | Pillow                         |
| Storage           | SQLite + CSV (local, no cloud) |

---

## 5. Installation (Windows 10/11)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Requires Python 3.10 or 3.11. Check with `python --version`.

---

## 6. Setup

Run the automated setup script once:

```bat
python scripts\setup.py
```

This will:
1. Check your Python version
2. Create `models/`, `data/`, `data/snapshots/`, and `logs/` directories
3. Create the SQLite database (`data/emotionvision.db`)
4. Download the pretrained emotion model automatically into `models/`
5. Verify OpenCV and TensorFlow are working
6. Run a full health check and print the results

---

## 7. Model Download Process

The first time `scripts/setup.py` (or the app itself) runs, it checks whether
`models/emotion_model.hdf5` already exists locally. If not, it downloads it
automatically from the model's public GitHub source (see **Model Information**
below), validates the file size, and saves it to `models/`. Every run after that
reuses the local file — **no repeated downloads, and the app works fully offline
for inference once the model is cached.**

If a download URL ever goes stale, edit `MODEL_SOURCES` in `config.py` with a
working URL — no other code changes are required.

---

## 8. Dataset Information

**No dataset is required to run this application.** EmotionVision AI uses a
pretrained model out of the box. Training your own model from scratch is
entirely optional and lives in the separate `training/` module — see
`training/README.md`. That module honestly reports when it cannot obtain a
dataset automatically (e.g., FER2013 requires a free Kaggle account) rather
than fabricating one.

---

## 9. How to Run

```bat
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in your
browser. Use the sidebar to navigate between Dashboard, Live Detection, Image
Analysis, Video Analysis, History, Analytics, Model, Settings, and Learn.

---

## 10. How Webcam Detection Works

1. Go to **Live Detection**.
2. Click **START CAMERA**. The app opens your default webcam (index configurable
   in Settings).
3. Each processed frame is resized, passed through Haar-cascade face detection,
   then every detected face is cropped, preprocessed, and classified by the
   emotion model.
4. Bounding boxes, emotion labels, and confidence scores are drawn live.
5. Click **STOP CAMERA** to end the session. Click **TAKE SNAPSHOT** any time to
   save the current annotated frame to `data/snapshots/`.
6. Enable **Demo Mode** (toggle above the video) to preview the UI with clearly
   labeled simulated data when no webcam is available.

---

## 11. How Image Analysis Works

Go to **Image Analysis**, upload a JPG/JPEG/PNG file. The same detection +
classification pipeline runs once on the uploaded image, annotates it, and
lists every detected face with its emotion and confidence.

---

## 12. How Video Analysis Works

Go to **Video Analysis**, choose a **Processing FPS** (how many frames per
second are actually analyzed — not every frame, to keep CPU usage reasonable),
upload an MP4/MOV/AVI file, and click **Process Video**. Frames are sampled at
the chosen rate, annotated, and tallied into a per-emotion summary.

---

## 13. Database Structure

SQLite database at `data/emotionvision.db`, created automatically.

**`sessions`**

| Column                  | Type    |
|--------------------------|---------|
| id                       | TEXT PK |
| start_time               | TEXT    |
| end_time                 | TEXT    |
| total_detections         | INTEGER |
| average_confidence       | REAL    |
| average_inference_time   | REAL    |

**`detections`**

| Column              | Type    |
|----------------------|---------|
| id                   | INTEGER PK |
| session_id           | TEXT    |
| timestamp            | TEXT    |
| emotion              | TEXT    |
| confidence           | REAL    |
| inference_time_ms    | REAL    |
| face_count           | INTEGER |

A mirrored `data/detections.csv` is also maintained for easy export/inspection.

---

## 14. Model Information

- **Model:** FER2013 Mini-XCEPTION (compact CNN, depthwise-separable convolutions)
- **Framework:** TensorFlow / Keras
- **Input:** 64×64 grayscale face crop
- **Output:** 7-way softmax over `angry, disgust, fear, happy, sad, surprise, neutral`
- **Source:** [github.com/oarriaga/face_classification](https://github.com/oarriaga/face_classification)
- **License:** MIT License (see `LICENSE`)
- **Reported benchmark accuracy:** ~66% on the FER2013 validation set, as
  documented by the model's original author in the source repository. This is
  shown in the app as "Model Validation Accuracy" — it is the model author's
  own reported figure, not measured by this application and not a claim about
  performance on any specific person or camera.
- **Size:** ~850 KB — small enough to run comfortably on CPU

---

## 15. Privacy

Camera frames are processed **locally** by the application and are **not**
uploaded to any remote server. By default, only the timestamp, predicted
emotion, confidence score, and inference time are stored — not the images
themselves. Image saving only happens if you explicitly click **TAKE SNAPSHOT**.

---

## 16. Limitations

- Facial emotion recognition is an AI prediction and can be inaccurate or
  biased across different people, lighting conditions, expressions, camera
  quality, and demographics.
- This tool does **not** measure a person's actual internal emotional state —
  it estimates the most likely expression category from facial appearance.
- **Do not** use this application for medical or mental-health diagnosis,
  employment decisions, law enforcement, or any other high-stakes decision.
  It is intended for educational, research, demonstration, and low-risk use.
- The bundled model is intentionally lightweight for CPU speed, which trades
  off some accuracy versus larger, heavier research models.

---

## 17. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Activate your venv, then re-run `pip install -r requirements.txt` |
| TensorFlow install fails | Ensure you're on Python 3.10/3.11 (64-bit); upgrade `pip` first: `python -m pip install --upgrade pip` |
| OpenCV can't open the camera | Close other apps using the camera; try a different `Camera Index` in Settings |
| Camera permission denied | On Windows, check Settings → Privacy → Camera, and allow desktop apps |
| Port already in use | Run `streamlit run app.py --server.port 8502` |
| Model download failure | Check your internet connection; or manually place a valid `.hdf5` file at `models/emotion_model.hdf5`; or edit `MODEL_SOURCES` in `config.py` |
| Low FPS / high CPU usage | Lower **Processing FPS** in Settings; close other heavy applications |
| Memory problems | Lower `FRAME_RESIZE_WIDTH` in `config.py`; close other apps |
| No face detected | Improve lighting; face the camera more directly; move closer |
| Model loading error | Delete `models/emotion_model.hdf5` and re-run `python scripts/setup.py` to re-download |

---

## 18. Testing

```bat
pip install pytest
pytest tests/ -v
```

All tests use synthetic frames/mock data and do **not** require a physical
webcam. 23/23 tests pass in the reference build of this project, covering the
face detector, emotion classifier, database layer, metrics helpers, and the
full end-to-end inference pipeline.

---

## 19. Optional Model Training

An advanced, fully optional local training module lives in `training/`:

```bat
python training\prepare_dataset.py
python training\train_model.py
python training\evaluate_model.py
```

This is **not required** for normal use — the main app already works with the
bundled pretrained model. See `training/README.md` for details, including why
the FER2013 dataset can't be auto-downloaded (it requires a free Kaggle
account) and how to supply it yourself.

---

## 20. Future Improvements

- Optional ONNX / TFLite export for faster CPU inference
- Optional DNN-based face detector for higher accuracy in difficult lighting
- Multi-camera / RTSP stream support
- Configurable notification integrations (fully local, opt-in)
- Additional pretrained model options selectable from the UI

---

## Project Structure

```
emotionvision-ai/
├── app.py                  # Streamlit entry point (Dashboard)
├── config.py                # All configuration in one place
├── ui_theme.py               # Shared CSS/UI components
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── models/                  # Auto-downloaded pretrained model
├── data/                    # SQLite DB, CSV history, snapshots
├── core/                    # Face detection, classification, inference, webcam, sessions
├── utils/                   # Model manager, database, image utils, metrics, logger
├── pages/                   # Streamlit multipage app (Live Detection, Image/Video, etc.)
├── training/                # Optional advanced local training module
├── tests/                   # Automated tests (no webcam required)
└── scripts/                 # setup.py and health_check.py
```

---




