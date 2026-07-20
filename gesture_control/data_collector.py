"""Data Collector — capture hand landmarks to CSV via MediaPipe.

Usage:
    python -m gesture_control.data_collector --label SIT --samples 100
    python -m gesture_control.data_collector --label WALK --samples 100

Press SPACE to start capturing, 'q' to quit early.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)

from gesture_control.utils import NUM_LANDMARKS, flatten_landmarks

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drawing colors
# ---------------------------------------------------------------------------
_COLOUR_ACTIVE = (0, 255, 0)    # green — capturing
_COLOUR_READY = (255, 255, 0)   # cyan — waiting for space
_COLOUR_DONE = (0, 255, 255)    # yellow — done


class DataCollector:
    """Capture hand-landmark samples for a single gesture label."""

    def __init__(
        self,
        label: str,
        num_samples: int = 100,
        output: str = "gestures.csv",
        camera_id: int | None = None,
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        model_path: str = "models/hand_landmarker.task",
    ) -> None:
        self.label = label.upper().strip()
        self.num_samples = num_samples
        self.output = Path(output)
        self.camera_id = camera_id  # None or -1 = auto-detect later in run()
        self.max_hands = max_hands

        # Load HandLandmarker (MediaPipe 0.10 tasks API)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        self._samples: list[list[float]] = []
        self._capturing = False  # starts paused

    def _open_camera(self) -> cv2.VideoCapture | None:
        """Open the first working camera. Tries self.camera_id first, then scans."""
        ids_to_try: list[int] = []

        if self.camera_id is not None and self.camera_id not in (-1, None):
            ids_to_try = [self.camera_id]
        else:
            # Scan real cameras (1-9) first; index 0 is often a ghost camera
            ids_to_try = list(range(1, 10)) + [0]

        for idx in ids_to_try:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            # Confirm it produces real frames (not just opens)
            for _ in range(15):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0 and frame.mean() > 1.0:
                    log.info("Camera opened at index %d", idx)
                    self.camera_id = idx
                    return cap
            # Ghost camera — try next
            cap.release()

        log.error("No working camera found (tried indices %s)", ids_to_try)
        return None

    def run(self) -> int:
        """Open webcam and collect samples. Returns number collected."""
        cap = self._open_camera()
        if cap is None:
            return 0

        # Write CSV header on first run
        csv_exists = self.output.exists()
        if not csv_exists:
            header = ",".join(
                [f"lm{i}_{j}" for i in range(NUM_LANDMARKS) for j in ("x", "y", "z")]
            )
            header += ",label\n"
            with open(self.output, "w") as f:
                f.write(header)

        log.info(
            "Ready to collect '%s'. Press SPACE to start. "
            "Aim for %d samples.",
            self.label,
            self.num_samples,
        )

        try:
            while len(self._samples) < self.num_samples:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Lost camera feed — stopping.")
                    break

                frame = cv2.flip(frame, 1)  # mirror
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)

                # Detect hand landmarks
                result = self._landmarker.detect(mp_image)
                handedness_text = ""

                if result.hand_landmarks:
                    for idx, hand_lms in enumerate(result.hand_landmarks):
                        drawing_utils.draw_landmarks(
                            frame, hand_lms, HandLandmarksConnections.HAND_CONNECTIONS,
                        )
                        if idx < len(result.handedness):
                            handedness_text = result.handedness[idx][0].category_name

                    # Capture first detected hand
                    if self._capturing:
                        self._samples.append(
                            flatten_landmarks(result.hand_landmarks[0])
                        )

                # ── Overlay UI ──────────────────────────────────────────
                captured = len(self._samples)
                colour = _COLOUR_ACTIVE if self._capturing else _COLOUR_READY
                status = f"{'⚫ REC' if self._capturing else '○ PAUSED'}  {captured}/{self.num_samples}"
                cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, colour, 2)

                if result.hand_landmarks:
                    cv2.putText(frame, f"Hand: {handedness_text}  ({len(result.hand_landmarks[0])} landmarks)",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                else:
                    cv2.putText(frame, "No hand detected — show your hand to the camera",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

                # Progress bar
                if self.num_samples > 0:
                    frac = min(captured / self.num_samples, 1.0)
                    bar_w = int(frac * 300)
                    cv2.rectangle(frame, (10, 90), (10 + 300, 105), (60, 60, 60), -1)
                    cv2.rectangle(frame, (10, 90), (10 + bar_w, 105), colour, -1)

                # Instructions
                if not self._capturing:
                    cv2.putText(frame, "SPACE = Start recording  |  Q = Quit", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

                cv2.imshow("Gesture Data Collection", frame)

                # ── Key handling ────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    self._capturing = not self._capturing
                    log.info("Capture %s", "STARTED" if self._capturing else "PAUSED")
                elif key == ord("q"):
                    log.info("User quit early after %d samples.", captured)
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            try:
                self._landmarker.close()
            except Exception:
                pass  # macOS cleanup race — safe to ignore
            self._landmarker = None  # prevents __del__ conflict during shutdown

        # Append collected samples to CSV
        if self._samples:
            import pandas as pd

            rows = []
            for feat in self._samples:
                rows.append(feat + [self.label])
            df = pd.DataFrame(rows)
            df.to_csv(self.output, mode="a", header=False, index=False)
            log.info("Appended %d samples to %s", len(self._samples), self.output)

        return len(self._samples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect hand-gesture landmark data to CSV.",
    )
    parser.add_argument("--label", required=True, help="Gesture name (e.g. SIT, WALK)")
    parser.add_argument("--samples", type=int, default=100, help="Samples per gesture")
    parser.add_argument("--output", default="gestures.csv", help="CSV output path")
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera device id (-1 = auto-detect)")
    parser.add_argument("--model", default="models/hand_landmarker.task",
                        help="MediaPipe HandLandmarker model path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    collector = DataCollector(
        label=args.label,
        num_samples=args.samples,
        output=args.output,
        camera_id=args.camera,
        model_path=args.model,
    )
    count = collector.run()
    print(f"\n✓ Collected {count} samples for '{args.label}' → {args.output}")


if __name__ == "__main__":
    main()
