from __future__ import annotations

import asyncio
import logging
import threading

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictStr

from .runtime import TV388Runtime, audit_assets


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tv388")
app = FastAPI(title="AIGC Text Detector TV388", docs_url=None, redoc_url=None, openapi_url=None)
runtime: TV388Runtime | None = None
startup_error: str | None = None
inference_lock = threading.Lock()
MAX_TEXT_CHARS = 2_000_000


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
        runtime = await asyncio.to_thread(TV388Runtime)
        log.info("TV388 ready; model load %.1fs", runtime.load_seconds)
    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        log.exception("TV388 startup failed")


@app.get("/health/startup")
def startup() -> JSONResponse:
    if runtime is not None:
        return JSONResponse(status_code=200, content={"status": "started", "version": "TV388"})
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
    if not req_id or len(payload.data) > MAX_TEXT_CHARS:
        return error(req_id, 400, 40001, "missing or invalid request field")

    def run() -> tuple[float, dict[str, float]]:
        with inference_lock:
            assert runtime is not None
            return runtime.predict(payload.data)

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
