"""Shared slowapi Limiter instance.

Defined in its own module so routers and `app.main` can both import
it without creating a circular dependency. The same instance must be
registered on the FastAPI app (`app.state.limiter = limiter`) and used
to decorate routes; otherwise each limiter tracks its own in-memory
counters and the configured limits do not bind globally.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
