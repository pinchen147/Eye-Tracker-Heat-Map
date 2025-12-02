# Eye-Tracker-Heat-Map
Transparent, click-through gaze overlay for macOS using NativeSensors/EyeGestures v3. Renders a smoothed gaze cursor, fixation trail, and decaying heat map over the entire screen.

## Requirements
- macOS with a webcam (overlay uses PyObjC and Quartz).
- Python 3.10+.
- Packages in `requirements.txt` (includes `eyeGestures`, OpenCV, MediaPipe, NumPy, PyObjC, etc.).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the overlay (main app)
```bash
python gaze_overlay.py [--camera 0] [--no-heatmap]
```
- `--camera`: webcam index (default 0).
- `--no-heatmap`: hide the heat map, show only gaze/trail.
- While running: press `q` to force recalibration; `Ctrl+C` to quit.

## Alternative tracker (pygame window + optional API)
```bash
python gaze_tracker.py [--camera 0] [--trail 30] [--api-port 5000]
```
- Space to recalibrate, `C` to cycle cameras, `Ctrl+Q` to quit.
- `--api-port` enables a simple Flask HTTP API (requires Flask, already listed).

## Notes
- EyeGestures is fetched from PyPI: https://github.com/NativeSensors/EyeGestures
- If the EyeGestures install ever skips heavy deps (mediapipe, opencv, scikit-learn), re-run `pip install -r requirements.txt`.
