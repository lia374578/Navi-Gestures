"""Gesture Control — Educational tool for training Navi the robot dog.

Teaches AI/ML concepts by capturing hand landmarks via MediaPipe,
training a classifier with scikit-learn, and sending commands to a
Navi robot dog over ROS/rosbridge.

Modules:
    data_collector — Record hand-gesture landmarks to CSV.
    trainer        — Train a Random Forest classifier from CSV data.
    controller     — Real-time gesture recognition + Navi command bridge.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("navi-gestures")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
