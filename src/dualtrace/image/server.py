from __future__ import annotations

import asyncio
import base64
import binascii
from io import BytesIO
import logging
import threading

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, StrictStr

from .v696_runtime import V696Runtime, audit_assets


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v696")
app = FastAPI(title="AIGC Image Detector V696", docs_url=None, redoc_url=None, openapi_url=None)
runtime: V696Runtime | None = None
startup_error: str | None = None
inference_lock = threading.Lock()
MAX_ENCODED_BYTES = 72 * 1024 * 1024


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    req_id: StrictStr
    data: StrictStr


def error(req_id: str, status: int, code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"req_id": req_id or None, "code": code, "message": message, "score": None})


@app.on_event("startup")
async def load_runtime() -> None:
    global runtime, startup_error
    try:
        log.info("asset audit: %s", audit_assets())
        runtime = await asyncio.to_thread(V696Runtime)
        log.info("V696 ready; model load %.1fs", runtime.load_seconds)
    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        log.exception("V696 startup failed")


@app.get("/health/startup")
def startup() -> JSONResponse:
    if runtime is not None:
        return JSONResponse(status_code=200, content={"status": "started", "version": "V696"})
    return JSONResponse(status_code=503, content={"status": "starting" if startup_error is None else "failed", "detail": startup_error})


@app.get("/health/liveness")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/readiness")
def readiness() -> JSONResponse:
    if runtime is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(status_code=200, content={"status": "ready"})


@app.post("/predict")
async def predict(payload: PredictRequest) -> JSONResponse:
    req_id = payload.req_id
    if runtime is None:
        return error(req_id, 503, 50301, "model is not ready")
    if not req_id or not payload.data or len(payload.data) > MAX_ENCODED_BYTES:
        return error(req_id, 400, 40001, "missing or invalid request field")
    try:
        raw = base64.b64decode(payload.data, validate=True)
        probe = Image.open(BytesIO(raw))
        if probe.format not in {"JPEG", "PNG", "BMP", "WEBP"}:
            raise UnidentifiedImageError("unsupported format")
        probe.verify()
        image = Image.open(BytesIO(raw))
        image.load()
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return error(req_id, 400, 40002, "invalid base64 or image")

    def run() -> tuple[float, dict[str, float]]:
        with inference_lock:
            assert runtime is not None
            return runtime.predict(image)

    try:
        score, diagnostics = await asyncio.to_thread(run)
        log.info("req_id=%s score=%.8f elapsed=%.3fs", req_id, score, diagnostics["elapsed_seconds"])
        return JSONResponse(status_code=200, content={"req_id": req_id, "code": 0, "score": score})
    except Exception:
        log.exception("inference failed req_id=%s", req_id)
        return error(req_id, 500, 50001, "internal inference error")


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    del exc
    req_id = ""
    try:
        body = await request.json()
        if isinstance(body, dict) and isinstance(body.get("req_id"), str):
            req_id = body["req_id"]
    except Exception:
        pass
    return error(req_id, 400, 40001, "invalid request body")
