"""Tiny .env loader — no external dependency.

Reads KEY=VALUE lines from a local .env into os.environ (without overwriting
anything already set in the real environment). Never prints values.
"""
import os


def load_env(path=None):
    path = path or os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
