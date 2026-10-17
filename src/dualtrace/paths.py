"""Portable model paths. No dependency on the retired server checkout."""
import os
from pathlib import Path


def project_root():
    return Path(os.environ.get("DUALTRACE_ROOT", Path(__file__).resolve().parents[2]))


def checkpoints():
    return Path(os.environ.get("DUALTRACE_CHECKPOINTS", project_root() / ".local/checkpoints"))
