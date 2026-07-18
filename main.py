#!/usr/bin/env python3
"""Navi Gestures — unified CLI for gesture-controlled robot dog.

Subcommands:
    collect   Capture hand-landmark samples for a gesture label
    train     Train a classifier from collected data
    control   Real-time gesture recognition + Navi command bridge

Examples:
    # Collect samples for two gestures
    python main.py collect --label SIT --samples 150
    python main.py collect --label WALK --samples 150

    # Train the model
    python main.py train --input gestures.csv

    # Run the controller (dry-run mode)
    export FF_SDK_DRY_RUN=1
    python main.py control --model gesture_model.pkl --target NV-demo
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Navi Gestures — train Navi the robot dog with hand gestures",
    )
    parser.add_argument("--version", action="store_true", help="Show version")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- collect ---
    collect_parser = subparsers.add_parser(
        "collect", help="Capture hand-gesture landmarks to CSV",
    )
    collect_parser.add_argument("--label", required=True, help="Gesture name (e.g. SIT)")
    collect_parser.add_argument("--samples", type=int, default=100, help="Samples per gesture")
    collect_parser.add_argument("--output", default="gestures.csv", help="CSV output path")
    collect_parser.add_argument("--camera", type=int, default=-1, help="Camera device id (-1 = auto-detect)")
    collect_parser.add_argument("--hand-model", default="models/hand_landmarker.task",
                                help="MediaPipe HandLandmarker model path")
    collect_parser.add_argument("--verbose", "-v", action="store_true")

    # --- train ---
    train_parser = subparsers.add_parser(
        "train", help="Train a classifier from CSV data",
    )
    train_parser.add_argument("--input", default="gestures.csv", help="CSV input path")
    train_parser.add_argument("--output", default="gesture_model.pkl", help="Model output")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    train_parser.add_argument("--estimators", type=int, default=100, help="Number of trees (RF only)")
    train_parser.add_argument("--model", choices=["knn", "rf"], default="knn",
                              help="Classifier: knn (simpler) or rf (Random Forest)")
    train_parser.add_argument("--verbose", "-v", action="store_true")

    # --- control ---
    ctrl_parser = subparsers.add_parser(
        "control", help="Real-time gesture recognition + Navi commands",
    )
    ctrl_parser.add_argument("--model", default="gesture_model.pkl", help="Trained model path")
    ctrl_parser.add_argument("--target", default="NV-demo", help="Navi target (NV-<sn>)")
    ctrl_parser.add_argument("--camera", type=int, default=-1, help="Camera device id (-1 = auto-detect)")
    ctrl_parser.add_argument("--confidence", type=float, default=0.8, help="Confidence threshold")
    ctrl_parser.add_argument("--cooldown", type=float, default=1.0, help="Command cooldown (s)")
    ctrl_parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="MediaPipe model path")
    ctrl_parser.add_argument("--no-dry-run", action="store_true", help="Disable dry-run for real hardware")
    ctrl_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.version:
        try:
            from gesture_control import __version__
            print(f"navi-gestures v{__version__}")
        except ImportError:
            print("navi-gestures (development)")
        return

    if args.command == "collect":
        from gesture_control.data_collector import DataCollector
        import logging
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )
        collector = DataCollector(
            label=args.label,
            num_samples=args.samples,
            output=args.output,
            camera_id=args.camera,
            model_path=args.hand_model,
        )
        count = collector.run()
        print(f"\n✓ Collected {count} samples for '{args.label}' → {args.output}")

    elif args.command == "train":
        from gesture_control.trainer import GestureTrainer
        import logging
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )
        trainer = GestureTrainer(
            input_csv=args.input,
            output_model=args.output,
            test_size=args.test_size,
            n_estimators=args.estimators,
            model_type=args.model,
        )
        accuracy = trainer.train()
        info = trainer.summary()
        print(f"\n✓ Training complete — accuracy: {accuracy:.1%}")
        print(f"  Classes: {info['classes']}")
        print(f"  Model:   {info['model_path']}")

    elif args.command == "control":
        from gesture_control.controller import GestureController
        import asyncio
        import logging

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

        dry_run = not args.no_dry_run
        if not dry_run:
            log = logging.getLogger("main")
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
        asyncio.run(controller.run())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
