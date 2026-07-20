# Navi Gestures — User Guide

Train the Navi robot dog to respond to your hand gestures using AI!

---

## Quick Start

```bash
# Activate the virtual environment (do this once per terminal)
source .venv/bin/activate

# Check everything is installed
python main.py --version
```

---

## Commands Overview

```
main.py collect   — Record hand gesture data
main.py train     — Train the AI model
main.py control   — Live gesture recognition + Navi commands
```

---

## Step 1: Collect Gesture Data

Record hand poses so the AI can learn them.

```bash
# Collect 50 samples of a "SIT" gesture (e.g. fist ✊)
python main.py collect --label SIT --samples 50

# Collect 50 samples of a "WALK" gesture (e.g. open palm 🖐️)
python main.py collect --label WALK --samples 50
```

**While collecting:**
- A webcam window appears — press **SPACE** to start recording
- The progress bar fills up as samples are captured
- The MediaPipe skeleton (21 points) is overlaid on your hand
- Press **Q** to quit early

**Collect more gestures (optional):**
```bash
python main.py collect --label STOP --samples 50
python main.py collect --label LEFT --samples 50
python main.py collect --label RIGHT --samples 50
python main.py collect --label BACK --samples 50
python main.py collect --label DOWN --samples 50
```

**Tips for good data:**
- Move your hand around a bit while recording (different positions in frame)
- Vary the angle slightly so the model learns robust features
- Make sure your hand is clearly visible with good lighting
- Use distinctly different hand poses for each gesture (e.g. fist vs open palm)

**To start over:** `rm gestures.csv`

---

## Step 2: Train the AI Model

```bash
# Train with KNN (default — simpler, best for small datasets)
python main.py train

# Train with Random Forest (more complex)
python main.py train --model rf

# With custom settings
python main.py train --model rf --estimators 200
```

The model is saved as `gesture_model.pkl`.

---

## Step 3: Live Controller

### Dry-run (no robot needed — just test the gesture recognition)

```bash
export FF_SDK_DRY_RUN=1
python main.py control --target NV-demo
```

### Real robot (when you have the Navi connected)

```bash
# Replace NV-A100-XXXX with your robot's serial number
python main.py control --target NV-A100-XXXX --no-dry-run
```

**While running the controller:**
- Webcam window shows your hand with the MediaPipe skeleton
- Top of screen shows predicted gesture + confidence percentage
- Probability bar shows confidence for each trained class
- Green text = command sent to the dog
- Grey "No hand detected" = hand not visible
- Press **Q** to quit

**Troubleshooting:**
- Camera not found? The tool auto-detects. If it fails, specify: `--camera 1`
- Only one gesture detected? Make sure your hand poses are very different
- Low confidence? Collect more samples (100+ per gesture)

---

## Gesture-to-Command Mapping

When the model predicts a gesture, this is what happens:

| Gesture | Robot Action | Details |
|---------|-------------|---------|
| **SIT**  | `stand()` | Stand up (toggles sit/stand on the dog) |
| **WALK** | `cmd_vel(0.5, 0.0)` | Walk forward at 0.5 m/s |
| **STOP** | `stop()` | Zero velocity — safe halt |
| **LEFT** | `cmd_vel(0.0, 2.0)` | Turn left |
| **RIGHT**| `cmd_vel(0.0, -2.0)` | Turn right |
| **BACK** | `cmd_vel(-0.5, 0.0)` | Walk backward |
| **RUN** | `cmd_vel(1.0, 0.0)` | Walk forward at 1.0 m/s |

**Safety:** The `damping` (estop) command is intentionally **not mapped** to any gesture to prevent accidental drops. If you need an emergency stop, use a separate safety mechanism.

---

## Customising the Gesture Map

Edit the `DEFAULT_GESTURE_MAP` in `gesture_control/controller.py` (around line 44).

Each gesture entry has:

```python
"GESTURE_NAME": {
    "type": "motion" or "cmd_vel",   # Type of command
    "action": "stand"/"stop"/"damping",  # For "motion" type
    "params": {"linear": 0.0, "angular": 0.0},  # For "cmd_vel" type
    "label": "🪑 Display Name",        # Shown on screen
}
```

---

## Model Types

### KNN (k-Nearest Neighbors) — default
```
python main.py train
python main.py train --model knn
```
- Simple, fast, good with small datasets
- Shows 100% confidence when all neighbors agree (normal!)
- Recommended for beginners

### Random Forest
```
python main.py train --model rf
python main.py train --model rf --estimators 200
```
- More complex ensemble model
- Softer probability estimates
- Better with larger datasets (200+ samples per class)

---

## Project Files

| File | Purpose |
|------|---------|
| `main.py` | Unified CLI — `collect`, `train`, `control` |
| `gesture_control/` | Source code package |
| `gesture_control/data_collector.py` | Step 1: webcam → MediaPipe → CSV |
| `gesture_control/trainer.py` | Step 2: CSV → train → save model |
| `gesture_control/controller.py` | Step 3: webcam → predict → command Navi |
| `models/hand_landmarker.task` | MediaPipe model (finds 21 hand points) |
| `gestures.csv` | Your training data |
| `gesture_model.pkl` | Your trained AI model |
| `ff_sdk/` | Navi robot dog SDK |

---

## How It Works (for Students)

1. **MediaPipe** finds 21 landmarks on your hand (each with x, y, z = 63 numbers)
2. These 63 numbers are saved to a CSV file
3. A classifier (KNN or Random Forest) learns to tell gestures apart
4. In real-time, the model predicts which gesture you're making
5. The prediction sends a command to the Navi robot dog

---

## Lesson Plan Ideas

**Lesson 1 — Garbage In, Garbage Out**
Try triggering your gesture in a dark room or wearing a glove. The AI fails because it only knows what you trained it on.

**Lesson 2 — Feature Engineering**  
Look at the 21 hand landmarks. Which points move most between a fist and an open palm?

**Lesson 3 — The Feedback Loop**
How fast does the camera → model → robot pipeline run? Why does latency matter in robotics?

---

## Troubleshooting

**"No working camera found"**
```bash
# Manually specify camera index
python main.py collect --label SIT --camera 1
```

**"Need at least 2 gesture classes"**
You only trained one gesture. Collect a second one:
```bash
python main.py collect --label WALK --samples 50
python main.py train
```

**Segmentation fault / Python crash**
The camera double-open bug — already fixed in the latest code. Pull the latest version.

**Wrong gesture detected**
Your hand poses are too similar. Re-collect with very different poses:
```bash
rm gestures.csv
python main.py collect --label SIT --samples 50   # ✊ fist
python main.py collect --label WALK --samples 50  # 🖐️ open palm
python main.py train
```

---

## Third-Party Licenses

This repository uses the following open-source software:

- **MediaPipe** (https://github.com/google-ai-edge/mediapipe)
  - Copyright: Copyright 2019 - Present The MediaPipe Authors
  - License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

