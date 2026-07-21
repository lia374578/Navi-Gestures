#!/usr/bin/env python3
"""Navi Gestures — Gradio GUI for gesture-controlled robot dog.

Usage:
    python app.py

Opens a browser-based UI with three tabs:
    📸 Collect — capture hand-landmark samples (opens OpenCV window)
    🧠 Train   — train a classifier from collected data
    🎮 Control — real-time gesture recognition (opens OpenCV window)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.expanduser("~"), ".cache", "matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import io
import logging
import pickle
import subprocess
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
CSV_PATH = PROJECT_ROOT / "gestures.csv"
MODEL_PATH = PROJECT_ROOT / "gesture_model.pkl"
GESTURES = ["SIT", "WALK", "STOP", "LEFT", "RIGHT", "BACK", "RUN"]
PYTHON = sys.executable
_MAX_UNDO = 50

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def get_status() -> str:
    lines: list[str] = []
    if CSV_PATH.exists():
        try:
            import pandas as pd
            df = pd.read_csv(CSV_PATH)
            labels = df.iloc[:, -1].value_counts()
            total = len(df)
            lines.append(f"📊 **gestures.csv**: {total} sample{'s' if total != 1 else ''}")
            for label, count in labels.items():
                lines.append(f"   · {label}: {count}")
        except Exception:
            lines.append("📊 **gestures.csv**: exists but couldn't read it")
    else:
        lines.append("📊 **gestures.csv**: not found — collect some data first!")
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            classes = list(model.classes_)
            lines.append(f"🧠 **gesture_model.pkl**: trained on {classes}")
        except Exception:
            lines.append("🧠 **gesture_model.pkl**: exists but couldn't load it")
    else:
        lines.append("🧠 **gesture_model.pkl**: not found — train a model first!")
    return "\n".join(lines) if lines else "No data yet."


# ---------------------------------------------------------------------------
# Undo / Redo helpers
# ---------------------------------------------------------------------------


def _csv_snapshot() -> str | None:
    return CSV_PATH.read_text() if CSV_PATH.exists() else None


def _restore_snapshot(snapshot: str | None) -> None:
    if snapshot is None:
        if CSV_PATH.exists():
            CSV_PATH.unlink()
    else:
        CSV_PATH.write_text(snapshot)


def _trim(stack: list) -> list:
    return stack[-_MAX_UNDO:] if len(stack) > _MAX_UNDO else stack


def _bu(u): return gr.update(interactive=len(u) > 0)
def _br(r): return gr.update(interactive=len(r) > 0)


# ---------------------------------------------------------------------------
# Collect — runs main.py collect as a subprocess
# ---------------------------------------------------------------------------


def run_collect(label: str, samples: int, camera: int):
    """Run the data collector in a subprocess, streaming log output."""
    cmd = [
        PYTHON, str(PROJECT_ROOT / "main.py"),
        "collect", "--label", label, "--samples", str(samples),
    ]
    if camera >= 0:
        cmd += ["--camera", str(camera)]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, bufsize=1,
    )

    collected: list[str] = []
    try:
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            collected.append(line)
            yield "\n".join(collected[-30:])
    finally:
        process.stdout.close()
        process.wait()

    yield "\n".join(collected[-20:]) + "\n\n" + get_status()


def run_collect_managed(label: str, samples: int, camera: int, undo_stack: list, redo_stack: list):
    """Wrap run_collect — snapshot before, push undo after."""
    snapshot = _csv_snapshot()
    final = ""
    for text in run_collect(label, samples, camera):
        final = text
        yield final, undo_stack, redo_stack, _bu(undo_stack), _br(redo_stack)
    undo_stack = _trim(list(undo_stack) + [snapshot])
    redo_stack = []
    yield final, undo_stack, redo_stack, _bu(undo_stack), _br(redo_stack)


# ---------------------------------------------------------------------------
# Train — imports and runs GestureTrainer in-process
# ---------------------------------------------------------------------------


def run_train(model_type: str, estimators: int, test_size: float):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        from gesture_control.trainer import GestureTrainer
        trainer = GestureTrainer(
            input_csv=str(CSV_PATH), output_model=str(MODEL_PATH),
            test_size=test_size, n_estimators=estimators, model_type=model_type,
        )
        yield "⏳ Loading data…"
        accuracy = trainer.train()
        info = trainer.summary()
        yield (
            f"✅ **Training complete!**\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| **Accuracy** | {accuracy:.1%} |\n"
            f"| **Model type** | {model_type.upper()} |\n"
            f"| **Classes** | {info['classes']} |\n"
            f"| **Total samples** | {info['total_samples']} |\n"
            f"| **Model saved to** | `{info['model_path']}` |\n"
            f"{get_status()}"
        )
    except FileNotFoundError as e:
        yield f"❌ **Error:** {e}\n\nCollect some data first!"
    except ValueError as e:
        yield f"❌ **Error:** {e}\n\nCollect at least **two** different gestures!"
    except Exception as e:
        yield f"❌ **Unexpected error:** {e}\n\n```\n{buf.getvalue()}\n```"
    finally:
        logging.getLogger().removeHandler(handler)


# ---------------------------------------------------------------------------
# Undo / Redo actions
# ---------------------------------------------------------------------------


def do_undo(undo_stack: list, redo_stack: list):
    if not undo_stack:
        return "ℹ️ Nothing to undo.", undo_stack, redo_stack, _bu(undo_stack), _br(redo_stack)
    current = _csv_snapshot()
    redo_stack = _trim(list(redo_stack) + [current])
    snapshot = undo_stack[-1]
    undo_stack = undo_stack[:-1]
    _restore_snapshot(snapshot)
    return (f"↩ Undone.\n\n{get_status()}", undo_stack, redo_stack,
            _bu(undo_stack), _br(redo_stack))


def do_redo(undo_stack: list, redo_stack: list):
    if not redo_stack:
        return "ℹ️ Nothing to redo.", undo_stack, redo_stack, _bu(undo_stack), _br(redo_stack)
    current = _csv_snapshot()
    undo_stack = _trim(list(undo_stack) + [current])
    snapshot = redo_stack[-1]
    redo_stack = redo_stack[:-1]
    _restore_snapshot(snapshot)
    return (f"↪ Redone.\n\n{get_status()}", undo_stack, redo_stack,
            _bu(undo_stack), _br(redo_stack))


def do_clear(undo_stack: list, redo_stack: list):
    snapshot = _csv_snapshot()
    if snapshot is None:
        return "ℹ️ No `gestures.csv` to clear.", undo_stack, redo_stack, _bu(undo_stack), _br(redo_stack)
    CSV_PATH.unlink()
    undo_stack = _trim(list(undo_stack) + [snapshot])
    redo_stack = []
    return (f"✅ Cleared all data.\n\n{get_status()}", undo_stack, redo_stack,
            _bu(undo_stack), _br(redo_stack))


# ---------------------------------------------------------------------------
# Control — runs main.py control as a subprocess
# ---------------------------------------------------------------------------


def run_control(target: str, confidence: float, cooldown: float, dry_run: bool):
    cmd = [
        PYTHON, str(PROJECT_ROOT / "main.py"),
        "control", "--target", target,
        "--confidence", str(confidence),
        "--cooldown", str(cooldown),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if dry_run:
        env["FF_SDK_DRY_RUN"] = "1"

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, bufsize=1,
    )

    collected: list[str] = []
    try:
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            collected.append(line)
            yield "\n".join(collected[-30:])
    finally:
        process.stdout.close()
        process.wait()

    yield "\n".join(collected[-20:])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Navi Gestures") as app:
        gr.Markdown(
            "# 🐶 Navi Gestures\n"
            "Train a robot dog to respond to your hand gestures using AI!"
        )

        undo_stack = gr.State([])
        redo_stack = gr.State([])

        with gr.Tabs():
            # ── Tab 1: Collect ─────────────────────────────────────────
            with gr.TabItem("📸 Collect Gesture Data"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=280):
                        label_dd = gr.Dropdown(
                            choices=GESTURES, label="Gesture label",
                            value="SIT", allow_custom_value=True,
                        )
                        samples_sl = gr.Slider(
                            minimum=10, maximum=500, value=100, step=10,
                            label="Number of samples",
                        )
                        camera_in = gr.Number(
                            value=-1, label="Camera index", precision=0,
                            info="-1 = auto-detect",
                        )
                        collect_btn = gr.Button(
                            "▶ Start Collecting", variant="primary", size="lg",
                        )
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown(
                            "**Got it:**\n\n"
                            "1. Pick a gesture label (e.g. SIT, WALK)\n"
                            "2. Set how many samples to collect\n"
                            "3. Click **Start**\n"
                            "4. An OpenCV window opens — press **SPACE** to record\n"
                            "5. Press **Q** to quit early\n\n"
                            "Samples are appended to `gestures.csv`."
                        )

                with gr.Column(elem_classes=["output-wrap"]):
                    collect_output = gr.Textbox(
                        label="Output", lines=12, max_lines=30,
                        elem_classes=["output-textbox"],
                    )

                with gr.Row():
                    undo_btn = gr.Button("↩ Undo", size="lg", interactive=False)
                    redo_btn = gr.Button("↪ Redo", size="lg", interactive=False)
                    clear_btn = gr.Button("🗑 Clear all data", size="lg", variant="stop")

                collect_btn.click(
                    fn=run_collect_managed,
                    inputs=[label_dd, samples_sl, camera_in, undo_stack, redo_stack],
                    outputs=[collect_output, undo_stack, redo_stack, undo_btn, redo_btn],
                )

                _uo = [collect_output, undo_stack, redo_stack]
                undo_btn.click(fn=do_undo, inputs=[undo_stack, redo_stack], outputs=[*_uo, undo_btn, redo_btn])
                redo_btn.click(fn=do_redo, inputs=[undo_stack, redo_stack], outputs=[*_uo, undo_btn, redo_btn])
                clear_btn.click(fn=do_clear, inputs=[undo_stack, redo_stack], outputs=[*_uo, undo_btn, redo_btn])

            # ── Tab 2: Train ───────────────────────────────────────────
            with gr.TabItem("🧠 Train Model"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=280):
                        model_radio = gr.Radio(
                            choices=[
                                ("KNN (simple, good for small data)", "knn"),
                                ("Random Forest (complex, larger data)", "rf"),
                            ],
                            value="knn", label="Model type",
                        )
                        estimators_sl = gr.Slider(
                            minimum=10, maximum=500, value=100, step=10,
                            label="Number of trees (RF only)", visible=False,
                        )
                        test_size_sl = gr.Slider(
                            minimum=0.1, maximum=0.5, value=0.2, step=0.05,
                            label="Test split ratio",
                        )
                        train_btn = gr.Button("▶ Start Training", variant="primary", size="lg")
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown(
                            "**Got it:**\n\n"
                            "1. Choose a model type\n"
                            "2. Click **Start Training**\n"
                            "3. The model is saved to `gesture_model.pkl`\n\n"
                            "You need at least **2 different gestures** with 10+ samples."
                        )

                with gr.Column(elem_classes=["output-wrap"]):
                    train_output = gr.Textbox(
                        label="Output", lines=12, max_lines=25,
                        elem_classes=["output-textbox"],
                    )

                def _toggle_estimators(v: str):
                    return gr.update(visible=(v == "rf"))
                model_radio.change(fn=_toggle_estimators, inputs=model_radio, outputs=estimators_sl)
                train_btn.click(fn=run_train, inputs=[model_radio, estimators_sl, test_size_sl], outputs=train_output)

            # ── Tab 3: Control ─────────────────────────────────────────
            with gr.TabItem("🎮 Live Control"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=280):
                        target_tb = gr.Textbox(
                            value="NV-demo", label="Navi target",
                            info="NV-demo for dry-run, or robot serial for real hardware",
                        )
                        confidence_sl = gr.Slider(
                            minimum=0.5, maximum=1.0, value=0.8, step=0.05,
                            label="Confidence threshold",
                        )
                        cooldown_sl = gr.Slider(
                            minimum=0.0, maximum=3.0, value=1.0, step=0.1,
                            label="Command cooldown (seconds)",
                        )
                        dry_run_cb = gr.Checkbox(
                            value=True, label="Dry run (no real robot)",
                        )
                        control_btn = gr.Button("▶ Start Control", variant="primary", size="lg")
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown(
                            "**Got it:**\n\n"
                            "1. Keep **Dry run** checked unless you have the robot\n"
                            "2. Click **Start Control**\n"
                            "3. OpenCV window opens with live skeleton tracking\n"
                            "4. Make a gesture — prediction shows on screen\n"
                            "5. Press **Q** to quit"
                        )

                with gr.Column(elem_classes=["output-wrap"]):
                    control_output = gr.Textbox(
                        label="Output", lines=12, max_lines=30,
                        elem_classes=["output-textbox"],
                    )

                control_btn.click(
                    fn=run_control,
                    inputs=[target_tb, confidence_sl, cooldown_sl, dry_run_cb],
                    outputs=control_output,
                )

        # ── Footer ────────────────────────────────────────────────────
        gr.Markdown("---")
        status_md = gr.Markdown(get_status())
        gr.Button("🔄 Refresh status").click(fn=lambda: get_status(), outputs=status_md)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Navi Gestures GUI")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--open", action="store_true", default=True, help="Open browser on launch")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        show_error=True,
        inbrowser=args.open,
        theme=gr.themes.Soft(),
        css="""
            footer { display: none !important; }
            .gradio-container { max-width: 1000px !important; }
            .output-textbox textarea { font-size: 16px !important; }
            .output-wrap { min-height: 300px !important; }
        """,
    )
