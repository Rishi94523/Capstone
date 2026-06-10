"""
Seed the sample pool with REAL MNIST test images.

These images are the "unlabeled" data the distributed CAPTCHA pipeline
labels. A fraction keeps its true label as a hidden honeypot (known_label)
for quality control; the rest are treated as unlabeled, with the true label
stashed separately so labeling accuracy can be measured offline.

Run from the server/ directory (uses the same DB as the server):
    python ../scripts/seed_data.py [--count 200] [--honeypot-rate 0.1]
"""

import argparse
import gzip
import hashlib
import io
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "server"))

from app.models.base import Base  # noqa: E402
from app.models.sample import Sample  # noqa: E402
from app.config import get_settings  # noqa: E402

settings = get_settings()

MNIST_DIR = REPO_ROOT / "data" / "mnist"


def load_mnist_test():
    """Load MNIST test images and labels (downloaded by train_mnist_numpy.py)."""
    images_path = MNIST_DIR / "t10k-images-idx3-ubyte.gz"
    labels_path = MNIST_DIR / "t10k-labels-idx1-ubyte.gz"
    if not images_path.exists():
        raise FileNotFoundError(
            f"{images_path} not found — run scripts/train_mnist_numpy.py first"
        )
    with gzip.open(images_path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
    with gzip.open(labels_path, "rb") as f:
        struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels


def seed_samples(count: int, honeypot_rate: float):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker
    from PIL import Image

    sync_url = settings.database_url.replace("+aiosqlite", "").replace("+asyncpg", "")
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    images, labels = load_mnist_test()
    rng = np.random.default_rng(2026)
    indices = rng.choice(len(images), size=count, replace=False)

    existing_hashes = set(
        h for (h,) in session.execute(select(Sample.data_hash)).all()
    )

    created = 0
    honeypots = 0
    for idx in indices:
        image = Image.fromarray(images[idx], mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        data_hash = hashlib.sha256(image_bytes).hexdigest()
        if data_hash in existing_hashes:
            continue

        true_label = str(int(labels[idx]))
        is_honeypot = rng.random() < honeypot_rate

        metadata = {
            "width": 28,
            "height": 28,
            "channels": 1,
            "source": "mnist-test",
            "source_index": int(idx),
            # true label kept for offline accuracy measurement; the pipeline
            # never reads it except for honeypots
            "true_label": true_label,
        }
        if is_honeypot:
            metadata["known_label"] = true_label
            honeypots += 1

        session.add(
            Sample(
                data_type="image",
                model_type="mnist",
                data_hash=data_hash,
                data_blob=image_bytes,
                metadata_=metadata,
            )
        )
        existing_hashes.add(data_hash)
        created += 1

    session.commit()
    session.close()
    print(f"Seeded {created} real MNIST samples ({honeypots} honeypots)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed real MNIST samples")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--honeypot-rate", type=float, default=0.1)
    args = parser.parse_args()
    seed_samples(args.count, args.honeypot_rate)
