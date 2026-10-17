#!/usr/bin/env python3
"""Export an allowlisted public source tree, excluding all local model/data assets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ['README.md', 'README.en.md', 'LICENSE', 'NOTICE.md', 'pyproject.toml', 'requirements-cpu.txt', 'requirements-gpu.txt', '.gitignore', '.dockerignore']
DIRECTORIES = ['src/dualtrace', 'training', 'research', 'docs', 'examples', 'tools', 'tests']
EXTENSIONS = {'.py', '.json', '.yaml', '.yml', '.md', '.txt', '.svg', '.csv', '.toml', '.sh'}
SENSITIVE = [
    re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    re.compile(r'(?i)(?:password|passwd|api_key|access_token)\s*=\s*[\"\x27][^\"\x27\s]{8,}[\"\x27]'),
    re.compile(r'sshpass\s+-p\s+\S+'),
]


def public_files():
    paths = [ROOT / name for name in FILES if (ROOT / name).is_file()]
    for name in DIRECTORIES:
        paths += [p for p in (ROOT / name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in EXTENSIONS]
    return sorted(set(paths))


def audit(paths):
    problems = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            problems.append(rel + ': symlink or external file')
            continue
        limit = 20 * 1024 * 1024 if rel.startswith('research/image/manifests/') and path.suffix == '.csv' else 5 * 1024 * 1024
        if path.stat().st_size > limit:
            problems.append(rel + ': exceeds source-file size limit')
        text = path.read_text(encoding='utf-8')
        if any(pattern.search(text) for pattern in SENSITIVE):
            problems.append(rel + ': possible credential; review locally')
        if re.search('/' + r'Users/[A-Za-z0-9_.-]+/', text):
            problems.append(rel + ': personal absolute path')
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    paths = public_files()
    problems = audit(paths)
    if problems:
        print(json.dumps({'ready': False, 'problems': problems}, indent=2))
        raise SystemExit(2)
    if a.check or a.output is None:
        print(json.dumps({'ready': True, 'files': len(paths), 'bytes': sum(p.stat().st_size for p in paths), 'private_assets_excluded': True}))
        return
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError('Choose a new, empty export path')
    out.mkdir(parents=True)
    manifest = []
    for path in paths:
        relative = path.relative_to(ROOT)
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest.append({'path': relative.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (out / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'export': str(out), 'files': len(paths), 'private_assets_excluded': True}))


if __name__ == '__main__':
    main()
