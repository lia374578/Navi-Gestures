"""Real-time gesture controller — predicts hand gestures and commands Navi.

Usage (dry-run, no dog needed):
    export FF_SDK_DRY_RUN=1
    python -m gesture_control.controller --model gesture_model.pkl --target NV-demo

Usage (with actual Navi):
    python -m gesture_control.controller --model gesture_model.pkl --target NV-A100-XXXX

Maps recognised gestures to Navi commands via configurable action table.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pickle
import signal
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)

from gesture_control.utils import flatten_landmarks

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default gesture → robot command mapping
# Students can customise this or load from a JSON config.
# ---------------------------------------------------------------------------
DEFAULT_GESTURE_MAP: dict[str, dict[str, Any]] = {
    "SIT": {
        "type": "motion",
        "action": "stand",        # sit → stand up
        "label": "🪑 Sit",
    },
    "STOP": {
        "type": "motion",
        "action": "stop",         # zero velocity — safe halt
        "label": "🛑 Stop",
    },
    "WALK": {
        "type": "cmd_vel",
        "params": {"linear": 0.5, "angular": 0.0},
        "label": "🚶 Walk",
    },
    "RUN": {
        "type": "cmd_vel",
        "params": {"linear": 1.0, "angular": 0.0},
        "label": "🏃 Run",
    },
    "LEFT": {
        "type": "cmd_vel",
        "params": {"linear": 0.0, "angular": 2.0},
        "label": "⬅️ Left",
    },
    "RIGHT": {
        "type": "cmd_vel",
        "params": {"linear": 0.0, "angular": -2.0},
        "label": "➡️ Right",
    },
    "BACK": {
        "type": "cmd_vel",
        "params": {"linear": -0.5, "angular": 0.0},
        "label": "🔙 Back",
    },
}


class GestureController:
    """Real-time gesture recognition + robot command bridge."""

    def __init__(
        self,
        model_path: str = "gesture_model.pkl",
        gesture_map: dict[str, dict[str, Any]] | None = None,
        confidence_threshold: float = 0.8,
        camera_id: int = -1,
        cooldown_s: float = 1.0,
        target: str = "NV-demo",
        dry_run: bool = True,
        hand_model_path: str = "models/hand_landmarker.task",
    ) -> None:
        self.model_path = Path(model_path)
        self.gesture_map = gesture_map or DEFAULT_GESTURE_MAP
        self.confidence_threshold = confidence_threshold
        self.camera_id = camera_id
        self.cooldown_s = cooldown_s
        self.target = target
        self.dry_run = dry_run

        # MediaPipe HandLandmarker (tasks API)
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
        )
        self._landmarker = HandLandmarker.create_from_options(hand_options)

        # Model (loaded in setup)
        self._model: Any = None
        self._classes: list[str] = []

        # Navi session (async, set by run)
        self._session: Any = None
        self._last_command_time: float = 0.0
        self._running = False
        self._last_cmd_vel_time: float = 0.0

        # Handle Ctrl+C gracefully — important for real robot control
        signal.signal(signal.SIGINT, self._signal_handler)

    def load_model(self) -> None:
        """Load the trained pickle model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run the trainer first."
            )
        with open(self.model_path, "rb") as f:
            self._model = pickle.load(f)
        self._classes = list(self._model.classes_)
        log.info("Loaded model with %d classes: %s", len(self._classes), self._classes)

    def _signal_handler(self, sig: int, frame: object) -> None:
        """Handle Ctrl+C — set flag so the main loop can exit cleanly."""
        log.info("Ctrl+C received — shutting down...")
        self._running = False
        cv2.destroyAllWindows()

    async def _connect_navi(self) -> None:
        """Connect to the Navi robot dog via ff_sdk."""
        import ff_sdk
        from ff_sdk import Config

        cfg = Config(dry_run=self.dry_run)
        log.info("Connecting to Navi: target=%s  dry_run=%s", self.target, self.dry_run)
        session = await ff_sdk.connect(self.target, config=cfg)
        log.info("Connected!  capabilities=%s  state=%s",
                 session.capabilities(), session.session_state.value)
        self._session = session

    async def _execute_command(self, gesture_label: str, confidence: float) -> None:
        """Map a recognised gesture to a robot action and execute it."""
        now = time.monotonic()
        if now - self._last_command_time < self.cooldown_s:
            return  # debounce

        gesture = self.gesture_map.get(gesture_label)
        if gesture is None:
            log.warning("No command mapping for gesture '%s'", gesture_label)
            return

        self._last_command_time = now
        display_name = gesture.get("label", gesture_label)
        log.info("🏷️ %s (%.0f%%) → %s", display_name, confidence * 100, gesture.get("action", "?"))

        if self._session is None:
            return

        action_type = gesture.get("type", "motion")

        try:
            if action_type == "motion":
                action_name = gesture["action"]
                if action_name == "stop":
                    await self._session.motion.stop()
                elif action_name == "stand":
                    await self._session.motion.stand()
                elif action_name == "damping":
                    await self._session.motion.damping()
                else:
                    await self._session.motion.do_preset(action_name)

            elif action_type == "cmd_vel":
                params = gesture.get("params", {})
                await self._session.motion.cmd_vel(
                    linear=params.get("linear", 0.0),
                    angular=params.get("angular", 0.0),
                    lateral=params.get("lateral", 0.0),
                )
                self._last_cmd_vel_time = now

        except Exception as e:
            log.warning("Command failed: %s", e)

    def _open_camera(self) -> cv2.VideoCapture | None:
        """Open the first working camera. Tries self.camera_id first, then scans."""
        ids_to_try: list[int] = []

        if self.camera_id not in (-1, None):
            ids_to_try = [self.camera_id]
        else:
            # Scan real cameras (1-9) first; index 0 is often a ghost camera
            ids_to_try = list(range(1, 10)) + [0]

        for idx in ids_to_try:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            for _ in range(15):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0 and frame.mean() > 1.0:
                    log.info("Camera opened at index %d", idx)
                    self.camera_id = idx
                    return cap
            cap.release()

        log.error("No working camera found (tried indices %s)", ids_to_try)
        return None

    async def run(self) -> None:
        """Main real-time loop."""
        self.load_model()
        await self._connect_navi()
        self._running = True

        cap = self._open_camera()
        if cap is None:
            return

        log.info("Camera opened. Press 'q' to quit.")

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)

                result = self._landmarker.detect(mp_image)

                prediction_label = ""
                confidence = 0.0
                gesture_info = None

                if result.hand_landmarks:
                    hand_lms = result.hand_landmarks[0]

                    # Draw skeleton on the frame
                    drawing_utils.draw_landmarks(
                        frame, hand_lms, HandLandmarksConnections.HAND_CONNECTIONS,
                    )

                    # Predict
                    features = flatten_landmarks(hand_lms)
                    probs = self._model.predict_proba([features])[0]
                    max_idx = int(np.argmax(probs))
                    confidence = float(probs[max_idx])
                    prediction_label = self._classes[max_idx]

                    if confidence >= self.confidence_threshold:
                        gesture_info = self.gesture_map.get(prediction_label)
                        await self._execute_command(prediction_label, confidence)
                    else:
                        gesture_info = None

                # ── Overlay UI ──────────────────────────────────────
                if result.hand_landmarks:
                    probs_text = "  ".join(
                        f"{c}: {p:.0%}" for c, p in zip(self._classes, probs)
                    )
                    cv2.putText(frame, probs_text, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

                    if prediction_label and gesture_info:
                        display = gesture_info.get("label", prediction_label)
                        colour = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 255, 255)
                        cv2.putText(frame, f"{display}  ({confidence:.0%})", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
                    elif prediction_label:
                        cv2.putText(frame, f"🤷 {prediction_label}  ({confidence:.0%})", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)

                # Connection status
                status_colour = (0, 255, 0) if self._session is not None else (0, 0, 255)
                status_text = f"Navi: {'CONNECTED' if self._session else 'OFFLINE'}  |  Q=Quit"
                cv2.putText(frame, status_text, (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_colour, 1)

                # Educational hint
                cv2.putText(frame, "MediaPipe 21 landmarks → 63 features → RandomForest", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

                cv2.imshow("Navi Gesture Controller", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    break
                if not self._running:  # Ctrl+C was pressed
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
            await self._disconnect()

    async def _disconnect(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
                log.info("Navi session closed.")
            except Exception as e:
                log.warning("Error closing session: %s", e)
            self._session = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time gesture control for Navi robot dog.",
    )
    parser.add_argument("--model", default="gesture_model.pkl", help="Trained model path")
    parser.add_argument("--target", default="NV-demo", help="Navi target (NV-<sn>)")
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera device id (-1 = auto-detect)")
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence threshold (0-1)")
    parser.add_argument("--cooldown", type=float, default=1.0, help="Min seconds between commands")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task",
                        help="MediaPipe HandLandmarker model path")
    parser.add_argument("--no-dry-run", action="store_true", help="Disable dry-run (real hardware)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    dry_run = not args.no_dry_run
    if not dry_run:
        log.warning("⚠️  REAL HARDWARE MODE — robot will move!")

    controller = GestureController(
        model_path=args.model,
        camera_id=args.camera,
        confidence_threshold=args.confidence,
        cooldown_s=args.cooldown,
        target=args.target,
        dry_run=dry_run,
        hand_model_path=args.hand_model,
    )

    try:
        asyncio.run(controller.run())
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except FileNotFoundError as e:
        log.error(e)
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
