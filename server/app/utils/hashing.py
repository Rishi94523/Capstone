"""
Hashing utilities.
"""

import hashlib
import secrets
from typing import Union


def sha256_hash(data: Union[str, bytes]) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: String or bytes to hash

    Returns:
        Hex-encoded hash string
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def generate_random_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.

    Args:
        length: Number of bytes (token will be 2x this length in hex)

    Returns:
        Hex-encoded random token
    """
    return secrets.token_hex(length)


def generate_url_safe_token(length: int = 32) -> str:
    """
    Generate a URL-safe random token.

    Args:
        length: Number of bytes

    Returns:
        URL-safe base64 token
    """
    return secrets.token_urlsafe(length)


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.

    Args:
        a: First string
        b: Second string

    Returns:
        True if strings are equal
    """
    return secrets.compare_digest(a, b)
