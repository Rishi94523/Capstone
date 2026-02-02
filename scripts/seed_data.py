"""
Script to create dummy ML samples for testing without real models.
"""

import hashlib
import random
import sys
import os

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
            # Image sample (CIFAR-10)
            data = os.urandom(32 * 32 * 3)  # 32x32 RGB
            data_hash = hashlib.sha256(data).hexdigest()
            known_label = random.choice(cifar_labels) if random.random() < 0.1 else None
            
            sample = Sample(
                data_type="image",
                model_type="cifar10",
                data_hash=data_hash,
                data_blob=data,
                metadata_={
                    "width": 32,
                    "height": 32,
                    "channels": 3,
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
