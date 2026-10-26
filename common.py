"""Paths, constants and small shared helpers."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
MANIFESTS = DATA / "manifests"
RUNS = REPO / "runs"
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
TABLES = REPO / "tables"
CONFIGS = REPO / "configs"
FROZEN = CONFIGS / "frozen"

SEQ_LEN = 2048
SCALES = ("50M", "100M", "400M", "1.2B")
FAMILIES = ("gdlm", "wang", "transformer")

# Tokenizer groups. Llama-2 ids fit uint16; Llama-3.2 ids (128,256) need uint32.
TOKENIZERS = {
    "llama2": {"hf_id": "hf-internal-testing/llama-tokenizer", "vocab": 32000, "bos": 1, "eos": 2,
               "dtype": "uint16", "data_dir": DATA / "llama2"},
    "llama3": {"hf_id": "meta-llama/Llama-3.2-1B", "vocab": 128256, "bos": 128000, "eos": 128001,
               "dtype": "uint32", "data_dir": DATA / "llama3"},
}


def write_json(path, obj, indent=1):
    """Atomic JSON write (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=indent, default=str))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def source_files():
    return sorted(list(REPO.glob("*.py")) + list((REPO / "slurm").glob("*")) + list((REPO / "tests").glob("*.py")))


def git_commit():
    """HEAD commit if the repo is under git, else None (the repo is not a git repository as of 2026-10-20)."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10)
        if out.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def code_sha256():
    """Hash over all source files (provenance that works with or without git)."""
    h = hashlib.sha256()
    for p in source_files():
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()
