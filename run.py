#!/usr/bin/env python3
"""
Start the ALM-Lite FastAPI server.

Usage (from project root):
  python run.py
  python run.py --port 8001
"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="ALM-Lite FastAPI server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Dev auto-reload (slower restarts)")
    args = p.parse_args()

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
