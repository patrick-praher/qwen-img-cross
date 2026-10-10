#!/usr/bin/env python3
"""
qwenimg - local text-to-image generation with Qwen-Image-2.1.

Backend: stable-diffusion.cpp (leejet) - pure C/C++ ggml inference, with
GPU acceleration via Metal (macOS), CUDA/Vulkan/ROCm (Windows) and
Vulkan/ROCm (Linux). Prebuilt backend binaries and GGUF model weights are
downloaded on first use; no Python packages are required (stdlib only).

Usage:
    qwenimg.py "a photo of a red fox in snow"     # one image
    qwenimg.py -f prompts.txt                     # one image per line
    qwenimg.py                                    # interactive prompt loop
    qwenimg.py setup                              # pre-download everything
    qwenimg.py doctor                             # show environment/paths
"""

from __future__ import annotations

import argparse
import hashlib
import json
import locale
import os
import platform
import random
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

TOOL = "qwenimg"
VERSION = "1.0.0"

if sys.version_info < (3, 8):      # Path.unlink(missing_ok=) and friends
    raise SystemExit(f"{TOOL} needs Python 3.8 or newer, found "
                     f"{platform.python_version()}")

SDCPP_REPO = "leejet/stable-diffusion.cpp"
SDCPP_RELEASES = f"https://api.github.com/repos/{SDCPP_REPO}/releases"
# Known-good release, used when the GitHub API is unreachable or rate-limited.
SDCPP_PINNED_TAG = "master-951-f89d9b1"
SDCPP_PINNED_ASSETS = {
    ("Windows", "cuda"):   "sd-master-f89d9b1-bin-win-cuda12-x64.zip",
    ("Windows", "rocm"):   "sd-master-f89d9b1-bin-win-rocm-7.14.0-x64.zip",
    ("Windows", "vulkan"): "sd-master-f89d9b1-bin-win-vulkan-x64.zip",
    ("Windows", "cpu"):    "sd-master-f89d9b1-bin-win-cpu-x64.zip",
    ("Linux", "rocm"):     "sd-master-f89d9b1-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip",
    ("Linux", "vulkan"):   "sd-master-f89d9b1-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
    ("Linux", "cpu"):      "sd-master-f89d9b1-bin-Linux-Ubuntu-24.04-x86_64.zip",
    ("Darwin", "metal"):   "sd-master-f89d9b1-bin-Darwin-macOS-26.6.2-arm64.zip",
}
# The Windows CUDA build is shipped without the CUDA runtime DLLs; they live in
# a companion archive so the binary also works without a CUDA toolkit install.
CUDART_PATTERN = r"^cudart-.*win.*\.zip$"
HF = "https://huggingface.co"

# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------
# Quantisation ladder, smallest/lowest-quality first. Used to pick the nearest
# available quant when a repo does not publish the exact one that was asked for.
QUANT_LADDER = [
    "q2_k", "q3_k", "q4_0", "q4_k", "q4_k_m", "q5_0", "q5_k",
    "q5_k_m", "q6_k", "q8_0", "bf16", "f16",
]

# sd.cpp names these Q4_K/Q5_K where the ComfyUI GGUF repacks say Q4_K_M/Q5_K_M.
# They are the same quantisation, so prefer the alias over a ladder neighbour.
QUANT_ALIASES = {
    "q4_k_m": ("q4_k",), "q4_k": ("q4_k_m",),
    "q5_k_m": ("q5_k",), "q5_k": ("q5_k_m",),
}

VARIANTS = {
    # Community "uncensored"/decensored build plus the abliterated ("heretic")
    # Qwen3-VL text encoder, which is the part that actually refuses prompts.
    "uncensored": {
        "label": "Qwen-Image-2.1 Uncensored (community)",
        "diffusion": {
            "repo": "abenzerps/Qwen-Image-2.1-Uncensored-GGUF",
            "files": {
                "q4_0":   "qwen-image-2.1-UC-Q4_0.gguf",
                "q4_k_m": "qwen-image-2.1-UC-Q4_K_M.gguf",
                "q5_k_m": "qwen-image-2.1-UC-Q5_K_M.gguf",
                "q6_k":   "qwen-image-2.1-UC-Q6_K.gguf",
                "q8_0":   "qwen-image-2.1-UC-Q8_0.gguf",
                "bf16":   "qwen-image-2.1-UC-BF16.gguf",
            },
        },
        "vae": {
            "repo": "abenzerps/Qwen-Image-2.1-Uncensored-GGUF",
            "files": {"bf16": "vae/qwen_image_2.1_vae_bf16.safetensors"},
        },
        "llm": {
            "repo": "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF",
            "files": {
                "q4_k_m": "qwen3vl_8b_heretic-Q4_K_M.gguf",
                "q6_k":   "qwen3vl_8b_heretic-Q6_K.gguf",
                "q8_0":   "qwen3vl_8b_heretic-Q8_0.gguf",
            },
        },
        "llm_vision": {
            "repo": "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF",
            "files": {"f16": "mmproj-qwen3vl_8b_heretic-f16.gguf"},
        },
    },
    # Official upstream weights (quantised by the sd.cpp author).
    "base": {
        "label": "Qwen-Image-2.1 (official Qwen weights)",
        "diffusion": {
            "repo": "leejet/Qwen-Image-2.1-GGUF",
            "files": {
                "q2_k": "qwen_image_2.1-Q2_K.gguf",
                "q3_k": "qwen_image_2.1-Q3_K.gguf",
                "q4_0": "qwen_image_2.1-Q4_0.gguf",
                "q4_k": "qwen_image_2.1-Q4_K.gguf",
                "q5_0": "qwen_image_2.1-Q5_0.gguf",
                "q6_k": "qwen_image_2.1-Q6_K.gguf",
                "q8_0": "qwen_image_2.1-Q8_0.gguf",
            },
        },
        "vae": {
            "repo": "Comfy-Org/Qwen-Image-2.1",
            "files": {"bf16": "vae/qwen_image_2.1_vae_bf16.safetensors"},
        },
        "llm": {
            "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
            "files": {
                "q4_k_m": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                "q8_0":   "Qwen3VL-8B-Instruct-Q8_0.gguf",
                "f16":    "Qwen3VL-8B-Instruct-F16.gguf",
            },
        },
        "llm_vision": {
            "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
            "files": {"f16": "mmproj-Qwen3VL-8B-Instruct-F16.gguf"},
        },
    },
}

# Official turbo/distillation LoRA. It trades a little fidelity for roughly a
# 5x speedup: far fewer steps, and cfg 1.0 skips the unconditional pass, so each
# step costs one forward pass instead of two.
TURBO_LORA = {
    "repo": "Comfy-Org/Qwen-Image-2.1",
    "path": "loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors",
    # Local basename (no dots) is what the <lora:NAME:1> prompt tag refers to.
    "name": "qwen-image-2-1-turbo-lora",
}
TURBO_DEFAULTS = {"steps": 8, "cfg_scale": 1.0}

# Qwen-Image-2.1 wants side lengths divisible by 32.
ASPECTS = {
    "1:1":  (1344, 1344),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3":  (1440, 1088),
    "3:4":  (1088, 1440),
    "3:2":  (1536, 1024),
    "2:3":  (1024, 1536),
}

DEFAULTS = {
    "variant": "uncensored",
    "quant": "auto",
    "llm_quant": "q4_k_m",
    "size": "1:1",
    "steps": 20,
    "cfg_scale": 6.0,
    "sampling_method": "euler",
    "negative": "",
    "seed": -1,
    "batch": 1,
    "offload": "auto",
    "flash_attn": True,
    "threads": 0,
    "outdir": "outputs",
}

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
IS_TTY = sys.stderr.isatty()


def log(msg: str, *, nl: bool = True) -> None:
    sys.stderr.write(msg + ("\n" if nl else ""))
    sys.stderr.flush()


def die(msg: str, code: int = 1):
    log(f"error: {msg}")
    raise SystemExit(code)


def human(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return "?"


def slugify(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (s[:limit].rstrip("-") or "image")


def decode_text(data: bytes) -> str:
    """Decode text as Windows tools write it: UTF-8 with or without a BOM
    (Notepad, Set-Content), or UTF-16 (Windows PowerShell's `>`/Out-File)."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:       # legacy ANSI code page, e.g. cmd's echo
        return data.decode(locale.getpreferredencoding(False), errors="replace")


def read_text(path: Path) -> str:
    return decode_text(path.read_bytes())


def round32(v: int) -> int:
    return max(32, int(round(v / 32.0)) * 32)


def app_home() -> Path:
    """Cache root for backend binaries and model weights."""
    env = os.environ.get("QWENIMG_HOME")
    if env:
        return Path(env).expanduser()
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / TOOL
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / TOOL
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / TOOL


# --------------------------------------------------------------------------
# platform / GPU detection
# --------------------------------------------------------------------------
def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


GPU_FLAVORS = ("metal", "cuda", "rocm", "vulkan", "cpu")


def detect_gpu() -> str:
    """Best guess at the fastest usable backend: metal|cuda|rocm|vulkan|cpu."""
    forced = (os.environ.get("QWENIMG_BACKEND") or "").strip().lower()
    if forced in GPU_FLAVORS:
        return forced
    if forced:
        log(f"warning: ignoring QWENIMG_BACKEND={forced!r}; "
            f"expected one of: {', '.join(GPU_FLAVORS)}")

    system = platform.system()
    if system == "Darwin":
        return "metal"

    if _has("nvidia-smi"):
        return "cuda"
    if system == "Linux" and Path("/dev/nvidiactl").exists():
        return "cuda"
    if _has("rocminfo") or Path("/sys/class/kfd").exists():
        return "rocm"

    if system == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=25,
            ).stdout.lower()
            if "nvidia" in out or "geforce" in out or "quadro" in out or "rtx" in out:
                return "cuda"
            if "radeon" in out or "amd" in out:
                return "rocm"
            if "intel" in out or "arc" in out:
                return "vulkan"
        except Exception:
            pass
        return "vulkan"

    if _has("vulkaninfo") or any(
        Path(p).exists() for p in ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
                                   "/usr/lib64/libvulkan.so.1",
                                   "/usr/lib/libvulkan.so.1")
    ):
        return "vulkan"
    return "cpu"


# Release-asset patterns per (system, flavor). Order within a chain is the
# preference order: first match in the release wins.
ASSET_PATTERNS = {
    ("Windows", "cuda"):   r"bin-win-cuda\d*-x64\.zip$",
    ("Windows", "rocm"):   r"bin-win-rocm[^/]*-x64\.zip$",
    ("Windows", "vulkan"): r"bin-win-vulkan-x64\.zip$",
    ("Windows", "cpu"):    r"bin-win-cpu-x64\.zip$",
    ("Linux", "rocm"):     r"bin-Linux-[^/]*x86_64-rocm[^/]*\.zip$",
    ("Linux", "vulkan"):   r"bin-Linux-[^/]*x86_64-vulkan\.zip$",
    ("Linux", "cpu"):      r"bin-Linux-[^/]*x86_64\.zip$",
    ("Darwin", "metal"):   r"bin-Darwin-[^/]*arm64\.zip$",
}

# Linux has no prebuilt CUDA archive; Vulkan runs fine on NVIDIA hardware.
FALLBACK_CHAIN = {
    ("Windows", "cuda"):   ["cuda", "vulkan", "cpu"],
    ("Windows", "rocm"):   ["rocm", "vulkan", "cpu"],
    ("Windows", "vulkan"): ["vulkan", "cpu"],
    ("Windows", "cpu"):    ["cpu"],
    ("Linux", "cuda"):     ["vulkan", "cpu"],
    ("Linux", "rocm"):     ["rocm", "vulkan", "cpu"],
    ("Linux", "vulkan"):   ["vulkan", "cpu"],
    ("Linux", "cpu"):      ["cpu"],
    ("Darwin", "metal"):   ["metal"],
    ("Darwin", "cpu"):     ["metal"],
}


def total_ram_bytes() -> int | None:
    try:
        system = platform.system()
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=10).stdout
            return int(out.strip())
        if system == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        if system == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=25).stdout
            return int(out.strip())
    except Exception:
        pass
    return None


def vram_bytes() -> int | None:
    """Dedicated VRAM, or unified memory on Apple silicon."""
    if platform.system() == "Darwin":
        return total_ram_bytes()
    if _has("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=25).stdout
            vals = [int(v.strip()) for v in out.splitlines() if v.strip()]
            if vals:
                return max(vals) * 1024 * 1024
        except Exception:
            pass
    return None


def auto_quant() -> str:
    """Pick a diffusion-model quant that fits the available accelerator memory."""
    mem = vram_bytes() or total_ram_bytes()
    if mem is None:
        return "q4_k_m"
    gib = mem / (1024 ** 3)
    if gib >= 40:
        return "q8_0"
    if gib >= 28:
        return "q6_k"
    if gib >= 18:
        return "q5_k_m"
    if gib >= 11:
        return "q4_k_m"
    return "q4_0"


def resolve_quant(available: dict[str, str], want: str) -> str:
    """Exact quant if the repo has it, else the nearest one on the ladder."""
    want = want.lower()
    if want in available:
        return want
    for alias in QUANT_ALIASES.get(want, ()):
        if alias in available:
            return alias
    if want not in QUANT_LADDER:
        die(f"unknown quant {want!r}; choose from: {', '.join(QUANT_LADDER)}")
    target = QUANT_LADDER.index(want)
    have = [q for q in available if q in QUANT_LADDER]
    if not have:
        die("model registry has no usable quantisations")
    # Nearest on the ladder; on a tie prefer the larger (higher quality) one.
    return min(have, key=lambda q: (abs(QUANT_LADDER.index(q) - target),
                                    -QUANT_LADDER.index(q)))


# --------------------------------------------------------------------------
# downloading
# --------------------------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keeps Hugging Face's x-linked-* headers, which the CDN hop drops."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _headers(url: str, extra: dict | None = None) -> dict:
    h = {"User-Agent": f"{TOOL}/{VERSION}", "Accept": "*/*"}
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN"))
    if token and "huggingface.co" in url:
        h["Authorization"] = f"Bearer {token}"
    if extra:
        h.update(extra)
    return h


def http_json(url: str, payload: dict | None = None, timeout: int = 60):
    data = json.dumps(payload).encode() if payload is not None else None
    extra = {"Content-Type": "application/json"} if data else None
    req = urllib.request.Request(url, data=data, headers=_headers(url, extra),
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def hf_url(repo: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/main/{path}"


def hf_paths_info(repo: str, paths: list[str]) -> dict[str, dict]:
    """{path: {"size": int, "sha256": str|None}} for LFS files in one call."""
    try:
        rows = http_json(f"{HF}/api/models/{repo}/paths-info/main", {"paths": paths})
    except Exception as exc:
        log(f"  note: could not query metadata for {repo} ({exc}); "
            "falling back to size-only checks")
        return {}
    out = {}
    for row in rows or []:
        if row.get("type") != "file":
            continue
        lfs = row.get("lfs") or {}
        out[row["path"]] = {
            "size": lfs.get("size") or row.get("size"),
            "sha256": lfs.get("oid"),
        }
    return out


def _as_size(value) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def remote_size(url: str) -> int | None:
    """Content length, following redirects only when we have to.

    The first hop is unredirected because Hugging Face reports the true LFS
    size in x-linked-size, which the CDN hop drops. A plain redirect (GitHub
    release assets) carries a content-length of 0 for the redirect body, so
    that value must never be mistaken for the file size.
    """
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers=_headers(url), method="HEAD")
    hs = None
    try:
        with opener.open(req, timeout=60) as resp:
            hs = resp.headers
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308):
            size = _as_size(exc.headers.get("x-linked-size"))
            if size:
                return size
        elif exc.code < 400:
            hs = exc.headers
    except Exception:
        return None

    if hs is not None:
        size = _as_size(hs.get("x-linked-size")) or _as_size(hs.get("content-length"))
        if size:
            return size

    # Redirect without a usable length: follow it and ask the final host.
    try:
        req = urllib.request.Request(url, headers=_headers(url), method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return _as_size(resp.headers.get("content-length"))
    except Exception:
        return None


def sha256_file(path: Path, label: str = "") -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    seen = 0
    last = 0.0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
            seen += len(block)
            now = time.time()
            if IS_TTY and now - last > 0.2:
                last = now
                log(f"\r  verifying {label} {100.0 * seen / max(1, total):5.1f}%", nl=False)
    if IS_TTY:
        log("\r" + " " * 60 + "\r", nl=False)
    return h.hexdigest()


class _Progress:
    def __init__(self, label: str, total: int | None, done: int = 0):
        self.label, self.total = label, total
        self.done = self.start_done = done
        self.start = time.time()
        self.lock = threading.Lock()
        self._last = 0.0

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n
            now = time.time()
            if now - self._last < 0.2:
                return
            self._last = now
            self._render(now)

    def _render(self, now: float) -> None:
        elapsed = max(1e-6, now - self.start)
        rate = (self.done - self.start_done) / elapsed
        pct = f"{100.0 * self.done / self.total:5.1f}%" if self.total else "  ?  "
        eta = ""
        if self.total and rate > 0:
            secs = int((self.total - self.done) / rate)
            eta = f" eta {secs // 60:d}m{secs % 60:02d}s"
        line = (f"\r  {self.label}  {pct}  {human(self.done)}"
                f"/{human(self.total)}  {human(rate)}/s{eta}   ")
        if IS_TTY:
            log(line[:118], nl=False)

    def finish(self) -> None:
        elapsed = max(1e-6, time.time() - self.start)
        rate = (self.done - self.start_done) / elapsed
        if IS_TTY:
            log("\r" + " " * 118 + "\r", nl=False)
        log(f"  {self.label}  {human(self.done)} in {int(elapsed)}s "
            f"({human(rate)}/s)")


CHUNK = 32 * 1024 * 1024       # resume granularity
CONNECTIONS = int(os.environ.get("QWENIMG_CONNECTIONS", "4"))


class RangeUnsupported(Exception):
    """The server will not serve byte ranges, so chunking is impossible."""


def range_supported(url: str) -> bool:
    """Hugging Face serves ranges; GitHub's release CDN answers 501."""
    req = urllib.request.Request(url, headers=_headers(url, {"Range": "bytes=0-0"}))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return getattr(resp, "status", resp.getcode()) == 206
    except Exception:
        return False


def _download_stream(url: str, part: Path, prog: _Progress) -> None:
    """Single-connection download, used when the server refuses ranges."""
    req = urllib.request.Request(url, headers=_headers(url))
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as fh:
        for block in iter(lambda: resp.read(1024 * 1024), b""):
            fh.write(block)
            prog.add(len(block))


def _fetch_range(url: str, fh, lock: threading.Lock, start: int, end: int,
                 prog: _Progress) -> None:
    """Download bytes [start, end] into fh at the right offset."""
    headers = _headers(url, {"Range": f"bytes={start}-{end}"})
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        if getattr(resp, "status", resp.getcode()) != 206:
            # A 200 would be the whole file, not the slice we asked for;
            # writing it at this offset would corrupt the output.
            raise RangeUnsupported("server ignored the Range header")
        pos = start
        while True:
            block = resp.read(1024 * 1024)
            if not block:
                break
            with lock:
                fh.seek(pos)
                fh.write(block)
            pos += len(block)
            prog.add(len(block))


def download(url: str, dest: Path, *, size: int | None = None,
             sha256: str | None = None, label: str | None = None,
             verify: bool = False) -> Path:
    """Resumable, parallel download with size (and optional sha256) checks."""
    label = label or dest.name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if size is None:
        size = remote_size(url)
    if size is not None and size <= 0:
        size = None

    if dest.exists():
        if size is None or dest.stat().st_size == size:
            if verify and sha256:
                if sha256_file(dest, label) == sha256:
                    log(f"  {label}  present (verified)")
                    return dest
                log(f"  {label}  checksum mismatch, re-downloading")
                dest.unlink()
            else:
                log(f"  {label}  present")
                return dest
        else:
            log(f"  {label}  wrong size ({human(dest.stat().st_size)} != "
                f"{human(size)}), re-downloading")
            dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")
    state_path = dest.with_suffix(dest.suffix + ".state")

    # Resume state: which fixed-size chunks already landed on disk.
    chunks_done: set[int] = set()
    if part.exists() and state_path.exists():
        try:
            st = json.loads(state_path.read_text())
            if st.get("url") == url and st.get("size") == size:
                chunks_done = set(st.get("done", []))
            else:
                part.unlink(missing_ok=True)
        except Exception:
            part.unlink(missing_ok=True)
    elif part.exists():
        part.unlink(missing_ok=True)

    # Unknown length, a single connection, or a server that refuses ranges:
    # fall back to one sequential stream (no resume).
    if size is None or CONNECTIONS < 2 or not range_supported(url):
        part.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        prog = _Progress(label, size)
        _download_stream(url, part, prog)
        prog.finish()
        return _finalise(part, state_path, dest, label, size, sha256, verify)

    n_chunks = max(1, (size + CHUNK - 1) // CHUNK)
    chunks_done = {i for i in chunks_done if i < n_chunks}

    with open(part, "r+b" if part.exists() else "wb") as fh:
        fh.truncate(size)

    todo = [i for i in range(n_chunks) if i not in chunks_done]
    prog = _Progress(label, size, done=len(chunks_done) * CHUNK)
    if not todo:
        prog.finish()
    else:
        lock = threading.Lock()
        work = list(todo)
        work_lock = threading.Lock()
        errors: list[BaseException] = []

        def worker():
            with open(part, "r+b") as fh:
                while True:
                    with work_lock:
                        if not work or errors:
                            return
                        idx = work.pop(0)
                    start = idx * CHUNK
                    end = min(size - 1, start + CHUNK - 1)
                    for attempt in range(5):
                        try:
                            _fetch_range(url, fh, lock, start, end, prog)
                            break
                        except Exception as exc:
                            if attempt == 4:
                                with work_lock:
                                    errors.append(exc)
                                return
                            time.sleep(1.5 * (attempt + 1))
                    with work_lock:
                        chunks_done.add(idx)
                        state_path.write_text(json.dumps(
                            {"url": url, "size": size,
                             "done": sorted(chunks_done)}))

        threads = [threading.Thread(target=worker, daemon=True)
                   for _ in range(max(1, min(CONNECTIONS, len(todo))))]
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            log("\n  interrupted; progress is saved, re-run to resume")
            raise
        if errors:
            if any(isinstance(e, RangeUnsupported) for e in errors):
                log(f"  {label}  ranges unsupported, retrying sequentially")
                part.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
                prog = _Progress(label, size)
                _download_stream(url, part, prog)
                prog.finish()
                return _finalise(part, state_path, dest, label, size,
                                 sha256, verify)
            raise errors[0]
        prog.finish()

    return _finalise(part, state_path, dest, label, size, sha256, verify)


def _finalise(part: Path, state_path: Path, dest: Path, label: str,
              size: int | None, sha256: str | None, verify: bool) -> Path:
    """Check what landed on disk, then move it into place."""
    actual = part.stat().st_size
    if size is not None and actual != size:
        die(f"{label}: downloaded {human(actual)}, expected {human(size)}")
    if verify and sha256:
        got = sha256_file(part, label)
        if got != sha256:
            part.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
            die(f"{label}: sha256 mismatch (got {got[:12]}..., "
                f"expected {sha256[:12]}...)")
        log(f"  {label}  sha256 ok")
    part.replace(dest)
    state_path.unlink(missing_ok=True)
    return dest


# --------------------------------------------------------------------------
# backend (stable-diffusion.cpp) installation
# --------------------------------------------------------------------------
def _asset_url(tag: str, name: str) -> str:
    return f"https://github.com/{SDCPP_REPO}/releases/download/{tag}/{name}"


def pick_release_asset(flavor_chain: list[str], system: str) -> dict:
    """Newest release asset matching this platform.

    -> {"tag", "name", "url", "flavor", "extras": [(name, url), ...]}
    """
    releases = None
    try:
        releases = http_json(f"{SDCPP_RELEASES}?per_page=10", timeout=45)
    except Exception as exc:
        log(f"note: GitHub API unavailable ({exc}); using pinned release "
            f"{SDCPP_PINNED_TAG}")
    if not releases:
        releases = [{
            "tag_name": SDCPP_PINNED_TAG,
            "assets": [],
            "_pinned": True,
        }]

    for release in releases:
        tag = release["tag_name"]
        assets = release.get("assets") or []
        for flavor in flavor_chain:
            pattern = ASSET_PATTERNS.get((system, flavor))
            if not pattern:
                continue
            name = url = None
            if assets:
                for asset in assets:
                    if re.search(pattern, asset["name"]):
                        name, url = asset["name"], asset["browser_download_url"]
                        break
            elif release.get("_pinned"):
                name = SDCPP_PINNED_ASSETS.get((system, flavor))
                url = _asset_url(tag, name) if name else None
            if not name or not url:
                continue

            extras = []
            if system == "Windows" and flavor == "cuda":
                cudart = next((a["name"] for a in assets
                               if re.search(CUDART_PATTERN, a["name"])), None)
                if cudart is None and release.get("_pinned"):
                    cudart = "cudart-sd-bin-win-cu12-x64.zip"
                if cudart:
                    extras.append((cudart, _asset_url(tag, cudart)))
            return {"tag": tag, "name": name, "url": url,
                    "flavor": flavor, "extras": extras}
    die("no prebuilt stable-diffusion.cpp binary matches this platform "
        f"({system}/{platform.machine()}). Build it from source: "
        f"https://github.com/{SDCPP_REPO} and point {TOOL} at it with "
        "QWENIMG_SD_CLI=/path/to/sd-cli")


def _find_binary(root: Path, name: str) -> Path | None:
    exe = name + (".exe" if platform.system() == "Windows" else "")
    for path in sorted(root.rglob(exe)):
        if path.is_file():
            return path
    return None


def ensure_backend(*, binary: str = "sd-cli", force: bool = False,
                   verify: bool = False) -> Path:
    override = os.environ.get("QWENIMG_SD_CLI")
    if override and binary == "sd-cli":
        path = Path(override).expanduser()
        if not path.exists():
            die(f"QWENIMG_SD_CLI points at a missing file: {path}")
        return path

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux" and machine not in ("x86_64", "amd64"):
        die(f"no prebuilt Linux binary for {machine}; build stable-diffusion.cpp "
            "from source and set QWENIMG_SD_CLI")

    gpu = detect_gpu()
    chain = FALLBACK_CHAIN.get((system, gpu)) or ["cpu"]
    rel = pick_release_asset(chain, system)

    install_dir = app_home() / "backend" / f"{rel['tag']}_{rel['flavor']}"
    found = None if force else _find_binary(install_dir, binary)
    if found:
        return found

    log(f"[backend] stable-diffusion.cpp {rel['tag']} ({rel['flavor']})")
    archives = [rel["name"]]
    download(rel["url"], app_home() / "downloads" / rel["name"],
             label=rel["name"], verify=verify)
    for extra_name, extra_url in rel["extras"]:
        log(f"  companion archive {extra_name} (CUDA runtime DLLs)")
        download(extra_url, app_home() / "downloads" / extra_name,
                 label=extra_name, verify=verify)
        archives.append(extra_name)

    if install_dir.exists() and force:
        shutil.rmtree(install_dir, ignore_errors=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    log(f"  extracting -> {install_dir}")
    for name in archives:
        with zipfile.ZipFile(app_home() / "downloads" / name) as zf:
            zf.extractall(install_dir)

    for path in install_dir.rglob("*"):
        if path.is_file() and (path.suffix in ("", ".exe", ".dylib", ".so")
                               or ".so." in path.name):
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IRUSR)

    if system == "Darwin" and _has("xattr"):
        # Downloaded binaries are quarantined by Gatekeeper; clear it so the
        # user is not prompted per file.
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(install_dir)],
                       capture_output=True)

    found = _find_binary(install_dir, binary)
    if not found:
        die(f"{binary} not found inside {rel['name']}")
    log(f"  ready: {found}")
    return found


# --------------------------------------------------------------------------
# model installation
# --------------------------------------------------------------------------
def model_plan(variant: str, quant: str, llm_quant: str,
               need_vision: bool, turbo: bool = False) -> dict[str, dict]:
    if variant not in VARIANTS:
        die(f"unknown variant {variant!r}; choose from: {', '.join(VARIANTS)}")
    spec = VARIANTS[variant]
    if quant == "auto":
        quant = auto_quant()

    wanted = {
        "diffusion": resolve_quant(spec["diffusion"]["files"], quant),
        "vae": "bf16",
        "llm": resolve_quant(spec["llm"]["files"], llm_quant),
    }
    parts = ["diffusion", "vae", "llm"]
    if need_vision:
        parts.append("llm_vision")
        wanted["llm_vision"] = next(iter(spec["llm_vision"]["files"]))

    plan = {}
    for part in parts:
        entry = spec[part]
        key = wanted[part]
        rel = entry["files"][key]
        plan[part] = {
            "repo": entry["repo"],
            "path": rel,
            "quant": key,
            "url": hf_url(entry["repo"], rel),
            "dest": app_home() / "models" / variant / Path(rel).name,
        }
    if turbo:
        plan["lora"] = {
            "repo": TURBO_LORA["repo"],
            "path": TURBO_LORA["path"],
            "quant": "bf16",
            "url": hf_url(TURBO_LORA["repo"], TURBO_LORA["path"]),
            "dest": (app_home() / "models" / "loras"
                     / f"{TURBO_LORA['name']}.safetensors"),
        }
    return plan


def ensure_models(variant: str, quant: str, llm_quant: str, *,
                  need_vision: bool = False, turbo: bool = False,
                  verify: bool = False, quiet: bool = False) -> dict[str, Path]:
    plan = model_plan(variant, quant, llm_quant, need_vision, turbo)

    by_repo: dict[str, list[str]] = {}
    for item in plan.values():
        by_repo.setdefault(item["repo"], []).append(item["path"])
    meta: dict[tuple[str, str], dict] = {}
    missing = [i for i in plan.values()
               if not i["dest"].exists() or verify]
    if missing:
        for repo, paths in by_repo.items():
            for path, info in hf_paths_info(repo, paths).items():
                meta[(repo, path)] = info

    if not quiet:
        label = VARIANTS[variant]["label"]
        total = sum((meta.get((i["repo"], i["path"])) or {}).get("size") or 0
                    for i in missing)
        log(f"[models] {label}")
        if total:
            log(f"  {len(missing)} file(s) to fetch, about {human(total)}")

    out = {}
    for part, item in plan.items():
        info = meta.get((item["repo"], item["path"])) or {}
        out[part] = download(
            item["url"], item["dest"],
            size=info.get("size"), sha256=info.get("sha256"),
            label=f"{part}:{item['dest'].name}", verify=verify,
        )
    return out


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def parse_size(spec: str) -> tuple[int, int]:
    spec = str(spec).strip().lower()
    if spec in ASPECTS:
        return ASPECTS[spec]
    m = re.fullmatch(r"(\d+)\s*[x*]\s*(\d+)", spec)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    elif spec.isdigit():
        w = h = int(spec)
    else:
        die(f"bad size {spec!r}; use WxH or one of: {', '.join(ASPECTS)}")
    rw, rh = round32(w), round32(h)
    if (rw, rh) != (w, h):
        log(f"note: size {w}x{h} rounded to {rw}x{rh} "
            "(Qwen-Image-2.1 needs multiples of 32)")
    return rw, rh


def want_offload(opts, models: dict[str, Path]) -> bool:
    mode = str(opts["offload"]).lower()
    if mode in ("on", "true", "yes", "1"):
        return True
    if mode in ("off", "false", "no", "0"):
        return False
    if platform.system() == "Darwin":
        return False      # unified memory; --auto-fit places weights already
    vram = vram_bytes()
    if vram is None:
        return True
    try:
        weights = models["diffusion"].stat().st_size
    except OSError:
        weights = 5 * 1024 ** 3      # not downloaded yet (--dry-run)
    return vram < weights + 3 * 1024 ** 3


def build_argv(sd_cli: Path, models: dict[str, Path], prompt: str, seed: int,
               out: Path, opts: dict) -> list[str]:
    w, h = parse_size(opts["size"])
    if models.get("lora"):
        # sd.cpp takes LoRAs through the prompt, A1111-style.
        prompt = f"{prompt}<lora:{models['lora'].stem}:1>"
    argv = [
        str(sd_cli),
        "-M", "img_gen",
        "--diffusion-model", str(models["diffusion"]),
        "--vae", str(models["vae"]),
        "--llm", str(models["llm"]),
        "-p", prompt,
        "-W", str(w), "-H", str(h),
        "--steps", str(int(opts["steps"])),
        "--cfg-scale", str(float(opts["cfg_scale"])),
        "--sampling-method", str(opts["sampling_method"]),
        "-s", str(seed),
        "-o", str(out),
    ]
    if models.get("llm_vision"):
        argv += ["--llm_vision", str(models["llm_vision"])]
    if models.get("lora"):
        argv += ["--lora-model-dir", str(models["lora"].parent)]
    for ref in opts.get("ref") or []:
        argv += ["-r", str(ref)]
    if opts.get("negative"):
        argv += ["-n", str(opts["negative"])]
    if int(opts.get("batch", 1)) > 1:
        argv += ["-b", str(int(opts["batch"]))]
    if opts.get("flash_attn", True):
        argv.append("--fa")
    if want_offload(opts, models):
        argv.append("--offload-to-cpu")
    if opts.get("mmap"):
        argv.append("--mmap")
    if int(opts.get("threads", 0)) > 0:
        argv += ["-t", str(int(opts["threads"]))]
    if opts.get("backend"):
        argv += ["--backend", str(opts["backend"])]
    if opts.get("verbose"):
        argv.append("-v")
    argv += list(opts.get("extra") or [])
    return argv


def generate(sd_cli: Path, models: dict[str, Path], prompt: str,
             opts: dict) -> list[Path]:
    outdir = Path(opts["outdir"]).expanduser()
    seed = int(opts["seed"])
    if seed < 0:
        seed = random.randrange(1, 2 ** 31 - 1)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    batch = int(opts.get("batch", 1))
    stem = f"{stamp}_s{seed}_{slugify(prompt)}"
    out = outdir / (f"{stem}_%02d.png" if batch > 1 else f"{stem}.png")

    argv = build_argv(sd_cli, models, prompt, seed, out, opts)
    if opts.get("dry_run"):
        print(" ".join(_quote(a) for a in argv))
        return []
    outdir.mkdir(parents=True, exist_ok=True)

    w, h = parse_size(opts["size"])
    log(f"[generate] {w}x{h}  steps={opts['steps']}  cfg={opts['cfg_scale']}  "
        f"seed={seed}" + (f"  batch={batch}" if batch > 1 else ""))
    _turbo_hint(opts)
    log(f"  prompt: {prompt[:160]}{'...' if len(prompt) > 160 else ''}")

    started = time.time()
    rc = subprocess.run(argv).returncode
    if rc != 0:
        log(f"error: sd-cli exited with status {rc}")
        return []
    took = time.time() - started

    produced = sorted(p for p in outdir.glob(f"{stem}*.png"))
    for path in produced:
        log(f"  -> {path}  ({human(path.stat().st_size)}, {took:.0f}s)")
    _record(outdir, prompt, seed, opts, produced, took)
    return produced


_HINTED = False


def _turbo_hint(opts: dict) -> None:
    """Mention --turbo once when the current settings will be slow."""
    global _HINTED
    if (_HINTED or opts.get("turbo") or float(opts["cfg_scale"]) <= 1.0
            or int(opts["steps"]) < 16):
        return
    _HINTED = True
    log("  (cfg > 1 costs two forward passes per step; --turbo runs "
        f"{TURBO_DEFAULTS['steps']} steps at cfg {TURBO_DEFAULTS['cfg_scale']} "
        "for roughly a 5x speedup)")


def _quote(arg: str) -> str:
    return f'"{arg}"' if re.search(r"[\s\"']", arg) else arg


def _record(outdir: Path, prompt: str, seed: int, opts: dict,
            files: list[Path], took: float) -> None:
    """Append a reproducibility log line next to the images."""
    w, h = parse_size(opts["size"])
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt": prompt,
        "negative": opts.get("negative") or "",
        "seed": seed,
        "width": w, "height": h,
        "steps": int(opts["steps"]),
        "cfg_scale": float(opts["cfg_scale"]),
        "sampling_method": opts["sampling_method"],
        "variant": opts["variant"],
        "quant": opts.get("resolved_quant"),
        "turbo": bool(opts.get("turbo")),
        "seconds": round(took, 1),
        "files": [f.name for f in files],
    }
    try:
        with open(outdir / "index.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_prompts(path: str, *, whole: bool = False) -> list[str]:
    if path == "-":
        text = decode_text(sys.stdin.buffer.read())
    else:
        p = Path(path).expanduser()
        if not p.exists():
            die(f"prompt file not found: {p}")
        text = read_text(p)
    if whole:
        # The whole file is one prompt, newlines included.
        if not text.strip():
            die(f"{path}: empty prompt")
        return [text.strip()]
    prompts = [ln.strip() for ln in text.splitlines()]
    prompts = [p for p in prompts if p and not p.startswith("#")]
    if not prompts and text.strip():
        prompts = [text.strip()]
    return prompts


def open_file(path: Path) -> None:
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        elif system == "Windows":
            os.startfile(str(path))          # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        log(f"could not open {path}: {exc}")


# --------------------------------------------------------------------------
# interactive loop
# --------------------------------------------------------------------------
REPL_HELP = """
commands:
  <text>              generate an image from <text>
  /size 16:9|WxH      set output size (presets: """ + ", ".join(ASPECTS) + """)
  /steps N            sampling steps
  /cfg F              cfg scale
  /seed N             fixed seed, or -1 for random each time
  /batch N            images per prompt
  /neg <text>         negative prompt ("/neg" alone clears it)
  /sampler NAME       euler, euler_a, dpm++2m, ...
  /set KEY VALUE      any of the above by name
  /show               show current settings
  /open               open the most recent image
  /help, /quit
"""

_REPL_KEYS = {
    "size": str, "steps": int, "cfg": float, "cfg_scale": float,
    "seed": int, "batch": int, "neg": str, "negative": str,
    "sampler": str, "sampling_method": str, "outdir": str,
}


def repl(sd_cli: Path, models: dict[str, Path], opts: dict) -> int:
    try:
        import readline  # noqa: F401  (gives history + line editing where available)
    except Exception:
        pass

    log(f"{TOOL} {VERSION} - {VARIANTS[opts['variant']]['label']}")
    log("type a prompt and press enter, /help for commands, /quit to exit")
    last: list[Path] = []

    while True:
        try:
            line = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            log("")
            return 0
        if not line:
            continue

        if line.startswith("/"):
            parts = line[1:].split(None, 1)
            cmd = parts[0].lower() if parts else ""
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                return 0
            if cmd in ("help", "h", "?"):
                log(REPL_HELP)
                continue
            if cmd == "show":
                for key in ("variant", "size", "steps", "cfg_scale",
                            "sampling_method", "seed", "batch", "negative",
                            "outdir"):
                    log(f"  {key:16} {opts.get(key)!r}")
                continue
            if cmd == "open":
                if last:
                    open_file(last[-1])
                else:
                    log("  nothing generated yet")
                continue
            if cmd == "set":
                bits = arg.split(None, 1)
                if len(bits) != 2:
                    log("  usage: /set KEY VALUE")
                    continue
                cmd, arg = bits[0].lower(), bits[1].strip()
            if cmd in ("neg", "negative"):
                opts["negative"] = arg
                log(f"  negative = {arg!r}")
                continue
            if cmd not in _REPL_KEYS:
                log(f"  unknown command /{cmd} (try /help)")
                continue
            key = {"cfg": "cfg_scale", "sampler": "sampling_method",
                   "neg": "negative"}.get(cmd, cmd)
            try:
                value = _REPL_KEYS[cmd](arg)
            except ValueError:
                log(f"  bad value for {key}: {arg!r}")
                continue
            if key == "size":
                vw, vh = parse_size(value)
                value = f"{vw}x{vh}"   # normalised, so we warn only once
            opts[key] = value
            log(f"  {key} = {opts[key]!r}")
            continue

        try:
            produced = generate(sd_cli, models, line, opts)
            if produced:
                last = produced
        except KeyboardInterrupt:
            log("\n  cancelled")
    return 0


# --------------------------------------------------------------------------
# config + CLI
# --------------------------------------------------------------------------
def load_config(explicit: str | None) -> dict:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        candidates += [Path.cwd() / f"{TOOL}.json", app_home() / "config.json"]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(read_text(path))
            except json.JSONDecodeError as exc:
                die(f"{path}: invalid JSON ({exc})")
            if not isinstance(data, dict):
                die(f"{path}: expected a JSON object")
            log(f"[config] {path}")
            return data
        if explicit:
            die(f"config file not found: {path}")
    return {}


COMMANDS = ("gen", "setup", "doctor", "models", "serve", "clean")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=TOOL,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Local text-to-image generation with Qwen-Image-2.1 "
                    "(stable-diffusion.cpp backend).",
        epilog=f"""commands:
  gen            generate images (default; omit to use the interactive loop)
  setup          download the backend and model weights, then exit
  doctor         show platform, GPU backend, paths and detected devices
  models         show which model files are selected and whether they exist
  serve          run the stable-diffusion.cpp web UI with these models
  clean          delete cached downloads (add --all to drop model weights too)

examples:
  {TOOL} "a red fox in deep snow, cinematic lighting"
  {TOOL} -f prompts.txt --size 16:9 --steps 24
  {TOOL} --quant q8_0 --variant base "studio portrait of a tabby cat"
  echo "a neon city street in the rain" | {TOOL}
  {TOOL}                      # interactive: waits for prompts

environment:
  QWENIMG_HOME         cache root for binaries and weights (default: OS cache dir)
  QWENIMG_BACKEND      force backend flavor: metal|cuda|rocm|vulkan|cpu
  QWENIMG_SD_CLI       use an existing sd-cli binary instead of downloading one
  QWENIMG_CONNECTIONS  parallel download connections (default 4)
  HF_TOKEN             Hugging Face token, for gated or rate-limited repos
""")
    p.add_argument("words", nargs="*", help="prompt text")
    p.add_argument("-p", "--prompt", help="prompt text (alternative to positional)")
    p.add_argument("-f", "--prompt-file",
                   help="file with one prompt per line ('-' reads stdin); "
                        "'#' lines are ignored")
    p.add_argument("--whole-file", action="store_true",
                   help="treat the whole prompt file (or stdin) as one "
                        "multi-line prompt instead of one per line")
    p.add_argument("-n", "--negative", help="negative prompt")
    p.add_argument("--negative-file", help="file containing the negative prompt")

    g = p.add_argument_group("generation")
    g.add_argument("-S", "--size",
                   help=f"WxH or preset ({', '.join(ASPECTS)}); "
                        f"default {DEFAULTS['size']}")
    g.add_argument("--steps", type=int, help=f"sampling steps (default {DEFAULTS['steps']})")
    g.add_argument("--cfg", type=float, dest="cfg_scale",
                   help=f"cfg scale (default {DEFAULTS['cfg_scale']})")
    g.add_argument("--sampler", dest="sampling_method",
                   help=f"sampling method (default {DEFAULTS['sampling_method']})")
    g.add_argument("-s", "--seed", type=int, help="seed; -1 picks a random one")
    g.add_argument("-b", "--batch", type=int, help="images per prompt")
    g.add_argument("-o", "--outdir", help=f"output directory (default {DEFAULTS['outdir']})")
    g.add_argument("-r", "--ref", action="append", metavar="IMAGE",
                   help="reference image for image editing (repeatable); "
                        "also downloads the vision projector")
    g.add_argument("--open", action="store_true", dest="open_result",
                   help="open the last produced image when done")

    m = p.add_argument_group("model selection")
    m.add_argument("--variant", choices=sorted(VARIANTS),
                   help=f"weights to use (default {DEFAULTS['variant']})")
    m.add_argument("--quant", help="diffusion quant: auto, " + ", ".join(QUANT_LADDER))
    m.add_argument("--llm-quant", dest="llm_quant",
                   help=f"text-encoder quant (default {DEFAULTS['llm_quant']})")
    m.add_argument("--turbo", action="store_true",
                   help="use the official turbo LoRA: ~5x faster "
                        f"({TURBO_DEFAULTS['steps']} steps at cfg "
                        f"{TURBO_DEFAULTS['cfg_scale']}), slightly lower fidelity")
    m.add_argument("--with-vision", action="store_true",
                   help="also fetch the vision projector (setup only)")

    r = p.add_argument_group("runtime")
    r.add_argument("--backend", help="sd.cpp device assignment, e.g. cpu or "
                                     "diffusion=cuda0,vae=cpu")
    r.add_argument("--threads", type=int, help="CPU threads (0 = auto)")
    r.add_argument("--offload", choices=["auto", "on", "off"],
                   help="keep weights in RAM and stream to VRAM "
                        f"(default {DEFAULTS['offload']})")
    r.add_argument("--no-fa", action="store_true", help="disable flash attention")
    r.add_argument("--mmap", action="store_true",
                   help="memory-map weights (faster reloads, uses page cache)")
    r.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="everything after --extra is passed to sd-cli verbatim")

    o = p.add_argument_group("other")
    o.add_argument("--config", help=f"config JSON (default ./{TOOL}.json)")
    o.add_argument("--verify", action="store_true",
                   help="verify downloads against their sha256")
    o.add_argument("--force-backend", action="store_true",
                   help="re-download and re-extract the backend")
    o.add_argument("--port", type=int, default=1234, help="serve: listen port")
    o.add_argument("--listen", default="127.0.0.1", help="serve: listen address")
    o.add_argument("--all", action="store_true", help="clean: also delete weights")
    o.add_argument("--dry-run", action="store_true",
                   help="print the sd-cli command instead of running it")
    o.add_argument("-v", "--verbose", action="store_true", help="verbose backend logs")
    o.add_argument("--version", action="version", version=f"{TOOL} {VERSION}")
    return p


def resolve_opts(args) -> dict:
    opts = dict(DEFAULTS)
    config = load_config(args.config)
    opts.update(config)
    chosen = set(config)       # keys the user set, so --turbo will not override

    for key in ("size", "steps", "cfg_scale", "sampling_method", "seed",
                "batch", "outdir", "variant", "quant", "llm_quant",
                "backend", "threads", "offload"):
        value = getattr(args, key, None)
        if value is not None:
            opts[key] = value
            chosen.add(key)

    if args.turbo or opts.get("turbo"):
        opts["turbo"] = True
        for key, value in TURBO_DEFAULTS.items():
            if key not in chosen:
                opts[key] = value
    if args.negative is not None:
        opts["negative"] = args.negative
    if args.negative_file:
        opts["negative"] = read_text(Path(args.negative_file).expanduser()).strip()
    if args.no_fa:
        opts["flash_attn"] = False
    if args.mmap:
        opts["mmap"] = True
    opts["ref"] = args.ref or []
    opts["extra"] = [a for a in (args.extra or []) if a != "--"]
    opts["verbose"] = args.verbose
    opts["dry_run"] = args.dry_run

    w, h = parse_size(opts["size"])
    opts["size"] = f"{w}x{h}"          # normalised, so we warn only once
    if opts["variant"] not in VARIANTS:
        die(f"unknown variant {opts['variant']!r}; "
            f"choose from: {', '.join(sorted(VARIANTS))}")
    return opts


def prepare(opts: dict, args, *, need_vision: bool,
            binary: str = "sd-cli") -> tuple[Path, dict[str, Path]]:
    turbo = bool(opts.get("turbo"))
    plan = model_plan(opts["variant"], opts["quant"], opts["llm_quant"],
                      need_vision, turbo)
    opts["resolved_quant"] = plan["diffusion"]["quant"]
    if opts.get("dry_run"):
        # Show the command without fetching ~10 GB of weights first.
        exe = Path(os.environ.get("QWENIMG_SD_CLI") or binary) \
            if binary == "sd-cli" else Path(binary)
        return exe, {part: item["dest"] for part, item in plan.items()}
    exe = ensure_backend(binary=binary, force=args.force_backend,
                         verify=args.verify)
    models = ensure_models(opts["variant"], opts["quant"], opts["llm_quant"],
                           need_vision=need_vision, turbo=turbo,
                           verify=args.verify)
    return exe, models


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_setup(opts: dict, args) -> int:
    sd_cli, models = prepare(opts, args, need_vision=args.with_vision)
    log("")
    log("ready.")
    log(f"  backend  {sd_cli}")
    for part, path in models.items():
        log(f"  {part:11}{path.name}  ({human(path.stat().st_size)})")
    log("")
    log(f'generate with:  {TOOL} "your prompt here"')
    return 0


def cmd_models(opts: dict, args) -> int:
    plan = model_plan(opts["variant"], opts["quant"], opts["llm_quant"],
                      bool(args.ref) or args.with_vision, bool(opts.get("turbo")))
    log(f"variant: {opts['variant']} - {VARIANTS[opts['variant']]['label']}")
    requested = opts["quant"]
    log(f"quant:   {requested}"
        + (f" -> {plan['diffusion']['quant']}" if requested == "auto"
           or requested != plan["diffusion"]["quant"] else ""))
    log("")
    total = 0
    for part, item in plan.items():
        dest = item["dest"]
        if dest.exists():
            size = dest.stat().st_size
            total += size
            state = f"ok   {human(size):>10}"
        else:
            state = "missing          "
        log(f"  {part:11} {state}  {item['repo']}/{item['path']}")
        log(f"              {dest}")
    log("")
    log(f"on disk: {human(total)}   cache root: {app_home()}")
    return 0


def cmd_doctor(opts: dict, args) -> int:
    system, machine = platform.system(), platform.machine()
    gpu = detect_gpu()
    chain = FALLBACK_CHAIN.get((system, gpu)) or ["cpu"]
    log(f"{TOOL} {VERSION}")
    log(f"  python        {platform.python_version()} ({sys.executable})")
    log(f"  platform      {system} {platform.release()} / {machine}")
    log(f"  detected gpu  {gpu}   backend preference: {' -> '.join(chain)}")
    log(f"  total ram     {human(total_ram_bytes())}")
    log(f"  accel memory  {human(vram_bytes())}")
    log(f"  auto quant    {auto_quant()}")
    log(f"  cache root    {app_home()}")
    log(f"  outputs       {Path(opts['outdir']).expanduser().resolve()}")
    try:
        rel = pick_release_asset(chain, system)
        log(f"  backend asset {rel['name']}  ({rel['tag']}, {rel['flavor']})")
    except SystemExit:
        log("  backend asset none for this platform (build from source)")
    log("")
    try:
        sd_cli = ensure_backend(force=args.force_backend, verify=args.verify)
    except SystemExit:
        return 1
    log(f"  sd-cli        {sd_cli}")
    log("")
    log("ggml devices:")
    proc = subprocess.run([str(sd_cli), "--list-devices"],
                          capture_output=True, text=True)
    for line in (proc.stdout or "").splitlines():
        if line.strip():
            log("  " + line.replace("\t", "   "))
    if proc.returncode != 0:
        log((proc.stderr or "").strip()[-800:])
        return 1
    return 0


def cmd_serve(opts: dict, args) -> int:
    sd_server, models = prepare(
        opts, args, binary="sd-server",
        need_vision=bool(args.ref) or args.with_vision)
    argv = [
        str(sd_server),
        "--diffusion-model", str(models["diffusion"]),
        "--vae", str(models["vae"]),
        "--llm", str(models["llm"]),
        "-l", args.listen, "--listen-port", str(args.port),
    ]
    if models.get("llm_vision"):
        argv += ["--llm_vision", str(models["llm_vision"])]
    if opts.get("flash_attn", True):
        argv.append("--fa")
    if want_offload(opts, models):
        argv.append("--offload-to-cpu")
    if opts.get("verbose"):
        argv.append("-v")
    argv += list(opts.get("extra") or [])
    if opts.get("dry_run"):
        print(" ".join(_quote(a) for a in argv))
        return 0
    log(f"[serve] http://{args.listen}:{args.port}  (ctrl-c to stop)")
    log("  the model stays loaded between prompts, unlike one-shot generation")
    try:
        return subprocess.run(argv).returncode
    except KeyboardInterrupt:
        return 0


def cmd_clean(opts: dict, args) -> int:
    targets = [app_home() / "downloads", app_home() / "backend"]
    if args.all:
        targets.append(app_home() / "models")
    total = 0
    for path in targets:
        if path.exists():
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
    if total == 0:
        log("nothing to clean")
        return 0
    log(f"about to delete {human(total)}:")
    for path in targets:
        if path.exists():
            log(f"  {path}")
    if not args.all:
        log("  (model weights are kept; use --all to delete those too)")
    try:
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            log("cancelled")
            return 1
    except (EOFError, KeyboardInterrupt):
        log("\ncancelled")
        return 1
    for path in targets:
        shutil.rmtree(path, ignore_errors=True)
    log("done")
    return 0


def cmd_gen(opts: dict, args) -> int:
    prompts: list[str] = []
    if args.prompt_file:
        prompts += read_prompts(args.prompt_file, whole=args.whole_file)
    if args.prompt:
        prompts.append(args.prompt)
    if args.words:
        prompts.append(" ".join(args.words))
    if not prompts and not sys.stdin.isatty():
        prompts += read_prompts("-", whole=args.whole_file)

    interactive = not prompts
    if interactive and not sys.stdin.isatty():
        die("no prompt given and stdin is empty")

    sd_cli, models = prepare(opts, args, need_vision=bool(opts["ref"]))

    if interactive:
        return repl(sd_cli, models, opts)

    last: list[Path] = []
    failures = 0
    for i, prompt in enumerate(prompts, 1):
        if len(prompts) > 1:
            log(f"\n[{i}/{len(prompts)}]")
        produced = generate(sd_cli, models, prompt, opts)
        if produced:
            last = produced
        elif not opts.get("dry_run"):
            failures += 1
    if last and args.open_result:
        open_file(last[-1])
    if failures:
        log(f"\n{failures} of {len(prompts)} prompt(s) failed")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "gen"
    if argv and argv[0] in COMMANDS:
        command = argv.pop(0)
    args = build_parser().parse_args(argv)

    if command != "gen" and args.words:
        die(f"'{command}' takes no prompt text (got {args.words[0]!r})")

    opts = resolve_opts(args)
    handlers = {
        "gen": cmd_gen, "setup": cmd_setup, "doctor": cmd_doctor,
        "models": cmd_models, "serve": cmd_serve, "clean": cmd_clean,
    }
    return handlers[command](opts, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\ninterrupted")
        raise SystemExit(130)
