"""Structured logging + global error handling for the API (D14).

JSON logs in production-like envs, a correlation-id per request, and a global
exception handler that never leaks stack traces to the client.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "correlation_id"):
            payload["correlationId"] = record.correlation_id
        for k in ("path", "method", "status", "duration_ms"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> logging.Logger:
    log = logging.getLogger("arena")
    if log.handlers:
        return log
    handler = logging.StreamHandler(sys.stdout)
    # JSON in non-dev; plain text locally for readability.
    if os.getenv("ENV", "development") != "development":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s · %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


logger = configure_logging()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to each request + log method/path/status/timing."""

    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("x-correlation-id") or uuid.uuid4().hex[:12]
        request.state.correlation_id = cid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            dur = round((time.perf_counter() - start) * 1000, 1)
            logger.error("Unhandled exception", extra={
                "correlation_id": cid, "path": request.url.path,
                "method": request.method, "duration_ms": dur,
            }, exc_info=True)
            raise
        dur = round((time.perf_counter() - start) * 1000, 1)
        response.headers["x-correlation-id"] = cid
        # Skip the chatty polling endpoints at INFO; log them at DEBUG only.
        if request.url.path not in ("/api/state", "/api/frames"):
            logger.info("request", extra={
                "correlation_id": cid, "path": request.url.path,
                "method": request.method, "status": response.status_code,
                "duration_ms": dur,
            })
        return response


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a clean error envelope; full details stay in the server log."""
    cid = getattr(request.state, "correlation_id", None)
    logger.error("Request failed", extra={
        "correlation_id": cid, "path": request.url.path, "method": request.method,
    }, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": {
            "code": "INTERNAL_ERROR",
            "message": "Lỗi hệ thống, vui lòng thử lại sau",
            "correlationId": cid,
        }},
    )
