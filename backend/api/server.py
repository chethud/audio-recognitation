"""
Uvicorn entry: `uvicorn backend.api.server:app --reload` from project root.
"""
from backend.main import app

__all__ = ["app"]
