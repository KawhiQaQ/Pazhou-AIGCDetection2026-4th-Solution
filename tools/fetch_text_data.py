#!/usr/bin/env python3
"""Download the pinned official NLPCC training payload and verify its LFS hash.

Data remains under .local and is not licensed by this code repository. Review
the upstream dataset terms before reuse or redistribution. No test files fetched.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://media.githubusercontent.com/media/NLP2CT/NLPCC-2026-Task6-Detection/961a485fa72f9821d6315836d46c9d94818b18b1/data/train_data.json'
SHA256 = 'd07059d8af0c7ba86629a8ceba3f09669d79f13356f09d0853861b53a8d5c01f'
SIZE = 212700850


def checksum(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'.local/legacy/external_data/NLPCC2026-Task6/train_data.json')
    parser.add_argument('--timeout',type=float,default=60,help='Network read timeout in seconds')
    parser.add_argument('--max-seconds',type=float,default=600,help='Total download time limit')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        if output.stat().st_size==SIZE and checksum(output)==SHA256:
            print(f'Already verified: {output}'); return
        raise RuntimeError(f'Refusing to overwrite an existing unverified file: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        request=urllib.request.Request(URL,headers={'User-Agent':'DualTrace-reproducibility/1.0'})
        started=time.monotonic(); total=0; previous=0; digest=hashlib.sha256()
        with urllib.request.urlopen(request,timeout=args.timeout) as response:
            with tempfile.NamedTemporaryFile(dir=output.parent,prefix='.nlpcc-download-',suffix='.part',delete=False) as stream:
                temporary=Path(stream.name)
                while True:
                    block=response.read(1024*1024)
                    if not block: break
                    if time.monotonic()-started>args.max_seconds:
                        raise TimeoutError('Total download time limit exceeded')
                    total+=len(block)
                    if total>SIZE: raise RuntimeError('Download exceeds expected length')
                    stream.write(block); digest.update(block)
                    if total-previous>=32*1024*1024:
                        print(f'Downloaded {total}/{SIZE} bytes',flush=True); previous=total
        if total!=SIZE or digest.hexdigest()!=SHA256:
            raise RuntimeError(f'Download validation failed: bytes={total},sha256={digest.hexdigest()}')
        if output.exists(): raise RuntimeError('Output appeared during download; refusing overwrite')
        temporary.rename(output); temporary=None
        print(f'Verified {total} bytes, SHA256={SHA256}, output={output}',flush=True)
    finally:
        # Only remove the exact disposable temporary file created above.
        if temporary is not None and temporary.exists(): temporary.unlink()


if __name__=='__main__': main()
