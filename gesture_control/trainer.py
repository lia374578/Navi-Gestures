"""Brain Trainer — train a Random Forest classifier from gesture CSV data.

Usage:
    python -m gesture_control.trainer --input gestures.csv --output gesture_model.pkl

The model file can then be loaded by the controller for real-time inference.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


class GestureTrainer:
    """Train a Random Forest model from landmark CSV data."""

    def __init__(
        self,
        input_csv: str = "gestures.csv",
        output_model: str = "gesture_model.pkl",
        test_size: float = 0.2,
        n_estimators: int = 100,
        model_type: str = "knn",
        random_state: int = 42,
    ) -> None:
        self.input_csv = Path(input_csv)
        self.output_model = Path(output_model)
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.model_type = model_type.lower()
        self.random_state = random_state

    def train(self) -> float:
        """Train the model and save to disk. Returns accuracy on test set."""
        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.input_csv}. "
                "Run data_collector first."
            )

        log.info("Loading data from %s", self.input_csv)
        df = pd.read_csv(self.input_csv)

        # Last column is the label; all others are features
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Sanity
        n_classes = len(set(y))
        if n_classes < 2:
            raise ValueError(
                f"Need at least 2 gesture classes, found {n_classes}. "
                "Collect data for multiple gestures first."
            )

        log.info("Loaded %d samples, %d features, %d classes: %s",
                  len(X), X.shape[1], n_classes, sorted(set(y)))

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y,
        )

        # Train
        if self.model_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=min(7, len(set(y))))
            log.info("Using KNN (k=%d) — simpler, better for small datasets",
                     min(7, len(set(y))))
        else:
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight="balanced",
            )
            log.info("Using RandomForest (%d trees)", self.n_estimators)
        model.fit(X_train, y_train)

        # Evaluate
        accuracy = model.score(X_test, y_test)
        log.info("Test accuracy: %.2f%%", accuracy * 100)

        # Save
        with open(self.output_model, "wb") as f:
            pickle.dump(model, f)

        log.info("Model saved to %s", self.output_model)
        return accuracy

    def summary(self) -> dict:
        """Return training metadata (call after .train())."""
        import pandas as pd
        df = pd.read_csv(self.input_csv)
        y = df.iloc[:, -1].values
        from collections import Counter
        return {
            "total_samples": len(df),
            "classes": dict(Counter(y)),
            "model_path": str(self.output_model),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture classifier.")
    parser.add_argument("--input", default="gestures.csv", help="CSV input")
    parser.add_argument("--output", default="gesture_model.pkl", help="Model output")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--estimators", type=int, default=100, help="Random Forest trees (only used with --model rf)")
    parser.add_argument("--model", choices=["knn", "rf"], default="knn",
                        help="Classifier type: knn (simpler) or rf (Random Forest)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
