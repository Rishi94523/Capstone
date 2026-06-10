"""
Retrain the model from human-verified labels — the feedback loop.

Pulls verified labels out of the database (golden-dataset consensus entries
first; optionally single human verifications with --min-verifications 1),
fine-tunes the current NumPy model on them mixed with replay data (so the
model doesn't forget), evaluates, and exports a new model version with fresh
checksums. The server picks up the new version on restart, and clients are
automatically protected against stale mixing because tasks pin the model
checksum at assignment time.

Usage (from repo root; server DB must exist):
    python scripts/retrain_from_golden.py [--min-verifications 1] [--epochs 3]
"""

import argparse
import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_mnist_numpy import (  # noqa: E402
    LAYER_SPECS,
    download_mnist,
    evaluate,
    export,
    forward,
    load_idx_images,
    load_idx_labels,
    train,
)

MODEL_DIR = REPO_ROOT / "models" / "mnist-tiny"
DB_PATH = REPO_ROOT / "server" / "pouw_captcha.db"


def load_current_model():
    with open(MODEL_DIR / "manifest.json") as f:
        manifest = json.load(f)
    weights = np.load(MODEL_DIR / manifest["weights_file"])
    params = []
    for i in range(len(LAYER_SPECS)):
        params.append([weights[f"W{i}"].copy(), weights[f"b{i}"].copy()])
    return params, manifest


def bump_version(version: str) -> str:
    parts = version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def blob_to_vector(blob: bytes) -> np.ndarray:
    from PIL import Image

    image = Image.open(io.BytesIO(blob)).convert("L").resize((28, 28))
    return (np.asarray(image, dtype=np.float32) / 255.0).flatten()


def collect_verified_labels(db_path: Path, labels: list, min_verifications: int):
    """
    Verified (sample, label) pairs from the DB.

    Golden-dataset entries are consensus-backed; with --min-verifications 1
    individual human confirmations are accepted too (useful early on, before
    enough volume exists for 3-way consensus).
    """
    db = sqlite3.connect(db_path)
    rows = db.execute(
        """
        SELECT s.data_blob, g.verified_label
        FROM golden_dataset g JOIN samples s ON g.sample_id = s.id
        WHERE s.data_blob IS NOT NULL
        """
    ).fetchall()
    source = "golden_dataset"

    if not rows and min_verifications <= 1:
        rows = db.execute(
            """
            SELECT s.data_blob, v.verified_label
            FROM verifications v JOIN samples s ON v.sample_id = s.id
            WHERE v.verified_label IS NOT NULL AND s.data_blob IS NOT NULL
            """
        ).fetchall()
        source = "individual verifications"
    db.close()

    x_list, y_list = [], []
    for blob, label in rows:
        if label not in labels:
            continue
        x_list.append(blob_to_vector(blob))
        y_list.append(labels.index(label))

    if not x_list:
        return None, None, source
    return np.stack(x_list), np.asarray(y_list, dtype=np.int64), source


def main():
    parser = argparse.ArgumentParser(description="Retrain from verified labels")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--min-verifications",
        type=int,
        default=3,
        help="1 accepts single human confirmations; 3 requires golden consensus",
    )
    parser.add_argument(
        "--replay-size",
        type=int,
        default=10000,
        help="Original training samples mixed in to prevent forgetting",
    )
    args = parser.parse_args()

    params, manifest = load_current_model()
    labels = manifest["labels"]

    x_verified, y_verified, source = collect_verified_labels(
        Path(args.db), labels, args.min_verifications
    )
    if x_verified is None:
        print("No verified labels available yet — nothing to retrain on.")
        print("(Solve more CAPTCHAs and answer the human-verification prompts.)")
        return

    print(f"Collected {len(x_verified)} verified samples from {source}")

    # Replay buffer from the original training set
    paths = download_mnist(REPO_ROOT / "data" / "mnist")
    x_train = load_idx_images(paths["train_images"])
    y_train = load_idx_labels(paths["train_labels"])
    x_test = load_idx_images(paths["test_images"])
    y_test = load_idx_labels(paths["test_labels"])

    rng = np.random.default_rng(5)
    replay_idx = rng.choice(len(x_train), size=args.replay_size, replace=False)

    # Verified samples upweighted by repetition so they matter against replay
    repeat = max(1, args.replay_size // (len(x_verified) * 10))
    x_mix = np.concatenate([x_train[replay_idx], np.tile(x_verified, (repeat, 1))])
    y_mix = np.concatenate([y_train[replay_idx], np.tile(y_verified, repeat)])

    before_acc = evaluate(params, x_test, y_test)
    print(f"Accuracy before fine-tune: {before_acc:.4f}")

    params = train(
        params, x_mix, y_mix, x_test[:2000], y_test[:2000],
        epochs=args.epochs, batch_size=128, lr=args.lr,
    )

    after_acc = evaluate(params, x_test, y_test)
    print(f"Accuracy after fine-tune:  {after_acc:.4f}")

    new_version = bump_version(manifest["version"])
    new_manifest = export(params, MODEL_DIR, after_acc, new_version)
    print(f"Exported {new_manifest['name']} v{new_version}")
    print(f"New checksum: {new_manifest['checksum']}")
    print("Restart the server to serve the updated model.")


if __name__ == "__main__":
    main()
