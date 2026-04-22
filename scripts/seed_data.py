"""
Script to create dummy ML samples for testing without real models.
"""

import hashlib
import random
import sys
import os
import io

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.models.base import Base
from app.models.sample import Sample
from app.config import get_settings

settings = get_settings()


def create_dummy_samples(count: int = 100):
    """Create dummy samples for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from PIL import Image, ImageDraw
    
    # Use sync engine for script
    sync_url = settings.database_url.replace("+asyncpg", "")
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # CIFAR-10 labels
    cifar_labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    # IMDB labels
    imdb_labels = ["positive", "negative"]
    
    print(f"Creating {count} dummy samples...")
    
    for i in range(count):
        # Alternate between image and text samples
        if i % 2 == 0:
            # Image sample (MNIST-style digit)
            digit = str(random.randint(0, 9))
            image = Image.new("L", (28, 28), color=0)
            draw = ImageDraw.Draw(image)

            # Draw simple, deterministic-looking digit strokes without requiring fonts.
            if digit == "0":
                draw.ellipse((6, 4, 21, 23), outline=255, width=3)
            elif digit == "1":
                draw.line((14, 5, 14, 23), fill=255, width=3)
            elif digit == "2":
                draw.arc((6, 4, 21, 14), start=0, end=180, fill=255, width=3)
                draw.line((21, 14, 7, 23), fill=255, width=3)
                draw.line((7, 23, 21, 23), fill=255, width=3)
            elif digit == "3":
                draw.arc((6, 4, 21, 14), start=270, end=90, fill=255, width=3)
                draw.arc((6, 13, 21, 23), start=270, end=90, fill=255, width=3)
            elif digit == "4":
                draw.line((19, 5, 19, 23), fill=255, width=3)
                draw.line((8, 14, 21, 14), fill=255, width=3)
                draw.line((8, 14, 16, 5), fill=255, width=3)
            elif digit == "5":
                draw.line((8, 5, 20, 5), fill=255, width=3)
                draw.line((8, 5, 8, 14), fill=255, width=3)
                draw.line((8, 14, 19, 14), fill=255, width=3)
                draw.arc((7, 13, 20, 24), start=270, end=110, fill=255, width=3)
            elif digit == "6":
                draw.arc((7, 4, 21, 23), start=40, end=320, fill=255, width=3)
                draw.line((9, 15, 19, 15), fill=255, width=3)
            elif digit == "7":
                draw.line((7, 6, 21, 6), fill=255, width=3)
                draw.line((21, 6, 11, 23), fill=255, width=3)
            elif digit == "8":
                draw.ellipse((8, 4, 20, 13), outline=255, width=3)
                draw.ellipse((8, 14, 20, 24), outline=255, width=3)
            else:
                draw.arc((7, 4, 21, 17), start=180, end=500, fill=255, width=3)
                draw.line((20, 13, 13, 23), fill=255, width=3)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data = buf.getvalue()
            data_hash = hashlib.sha256(data).hexdigest()
            known_label = digit if random.random() < 0.3 else None
            
            sample = Sample(
                data_type="image",
                model_type="mnist",
                data_hash=data_hash,
                data_blob=data,
                metadata_={
                    "width": 28,
                    "height": 28,
                    "channels": 1,
                    "known_label": known_label,
                    "is_dummy": True,
                },
            )
        else:
            # Text sample (IMDB)
            text = f"This is dummy review text {i} for testing purposes."
            data = text.encode('utf-8')
            data_hash = hashlib.sha256(data).hexdigest()
            known_label = random.choice(imdb_labels) if random.random() < 0.1 else None
            
            sample = Sample(
                data_type="text",
                model_type="imdb",
                data_hash=data_hash,
                data_blob=data,
                metadata_={
                    "text": text,
                    "known_label": known_label,
                    "is_dummy": True,
                },
            )
        
        session.add(sample)
        
        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{count} samples...")
    
    session.commit()
    print(f"✓ Successfully created {count} dummy samples!")
    session.close()


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    create_dummy_samples(count)
