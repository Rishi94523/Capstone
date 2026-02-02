"""
Database models package.

SQLAlchemy models for PoUW CAPTCHA system.
"""

from app.models.base import Base, init_db, close_db, get_db
from app.models.sample import Sample
from app.models.session import Session
from app.models.task import Task
from app.models.prediction import Prediction
from app.models.verification import Verification
from app.models.golden_dataset import GoldenDataset
from app.models.reputation import ReputationScore
from app.models.domain_config import DomainConfig

__all__ = [
    "Base",
    "init_db",
    "close_db",
    "get_db",
    "Sample",
    "Session",
    "Task",
    "Prediction",
    "Verification",
    "GoldenDataset",
    "ReputationScore",
    "DomainConfig",
]
