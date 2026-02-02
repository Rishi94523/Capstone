"""
API routes package.
"""

from app.api.captcha import router as captcha_router
from app.api.verification import router as verification_router
from app.api.federated import router as federated_router

__all__ = ["captcha_router", "verification_router", "federated_router"]
