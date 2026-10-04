"""Helpers for the Step 0.1 environment check.

Runs on: M4 and molab GPU (pure Python, no torch, no CUDA).

Used by notebooks/step0_setup.py to parse the evaluation output of the
official R2-Gaussian and 4DRGS code and to compare it with published numbers.
Nothing here touches the method; it is tooling only.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

_FLOAT = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def scrub(text: str, secret: str | None) -> str:
    """Remove a secret (and its base64 form) from text before it is printed.

    Ref: OURS (tooling). Reason: the GitHub token must never reach notebook
    output or results/step0.1.json.
    """
    if not secret:
        return text
    encoded = base64.b64encode(f"x-access-token:{secret}".encode()).decode()
    return text.replace(secret, "***").replace(encoded, "***")


def git_auth_env(token: str | None) -> dict:
    """Environment variables that give git a token without storing it.

    The token is passed as an HTTP header through GIT_CONFIG_* variables, so
    it is not written to .git/config and is not on the command line.

    Ref: OURS (tooling). Mechanism: git-config documentation, ENVIRONMENT
    (GIT_CONFIG_COUNT / GIT_CONFIG_KEY_n / GIT_CONFIG_VALUE_n). Section
    numbering UNVERIFIED.
    """
    if not token:
        return {}
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {encoded}",
        "GIT_TERMINAL_PROMPT": "0",
    }


def parse_r2g_eval_yml(text: str) -> dict:
    """Read scalar metrics from an R2-Gaussian eval3d.yml / eval2d_*.yml file.

    Only top-level `key: number` lines are read (psnr_3d, ssim_3d, psnr_2d,
    ssim_2d, ...). Per-projection lists are ignored.

    Ref: R2-Gaussian code, train.py (eval_dict written with yaml.dump) at
    commit f2579bf. File format taken as is.
    """
    out = {}
    for m in re.finditer(rf"^(\w+):\s*{_FLOAT}\s*$", text, flags=re.M):
        out[m.group(1)] = float(m.group(2))
    return out


def parse_r2g_stdout(text: str) -> dict:
    """Read metrics from the last R2-Gaussian evaluation line in stdout.

    Line format: "[ITER 30000] Evaluating: psnr3d 35.1, ssim3d 0.93,
    psnr2d 50.2, ssim2d 0.99". Returns {} if no such line exists.

    Ref: R2-Gaussian code, train.py (evaluation print) at commit f2579bf.
    """
    pat = (
        rf"\[ITER (\d+)\] Evaluating: psnr3d {_FLOAT}, ssim3d {_FLOAT}, "
        rf"psnr2d {_FLOAT}, ssim2d {_FLOAT}"
    )
    matches = re.findall(pat, text)
    if not matches:
        return {}
    it, p3, s3, p2, s2 = max(matches, key=lambda m: int(m[0]))
    return {
        "iteration": int(it),
        "psnr_3d": float(p3),
        "ssim_3d": float(s3),
        "psnr_2d": float(p2),
        "ssim_2d": float(s2),
    }


def parse_4drgs_log(text: str) -> dict:
    """Read PSNR / SSIM from 4DRGS test.py output or its logs.txt.

    Lines: "PSNR train: x dB", "SSIM train: x", "PSNR eval: x dB",
    "SSIM eval: x", and "GauN: n" (logs.txt only).

    Ref: 4DRGS code, test.py (log_message) at commit 38006ed.
    """
    out = {}
    for key, pat in [
        ("psnr_train", rf"PSNR train:\s*{_FLOAT}"),
        ("ssim_train", rf"SSIM train:\s*{_FLOAT}"),
        ("psnr_eval", rf"PSNR eval:\s*{_FLOAT}"),
        ("ssim_eval", rf"SSIM eval:\s*{_FLOAT}"),
        ("n_gaussians", r"GauN:\s*(\d+)"),
    ]:
        found = re.findall(pat, text)
        if found:
            out[key] = float(found[-1])
    return out


def compare_psnr(ours: float | None, expected: float, tol_db: float = 1.0) -> dict:
    """Compare a reproduced PSNR with the published one.

    Pass when |ours - expected| <= tol_db.

    Ref: OURS. Reason: Step 0.1 verification rule "pass if within about 1 dB
    PSNR" set by the project owner (not from a paper).
    """
    if ours is None:
        return {"ours": None, "expected": expected, "diff_db": None, "pass": False}
    diff = ours - expected
    return {
        "ours": ours,
        "expected": expected,
        "diff_db": round(diff, 3),
        "pass": abs(diff) <= tol_db,
    }


def cube_center_ray_integral(side_mm: float, mu: float = 1.0) -> float:
    """Line integral of the central ray through a uniform cube at the isocenter.

    The central ray is parallel to a cube edge, so the path length is the
    side length and the integral is mu * side_mm.

    Ref: R2-Gaussian, Sec 3.1 Eq 1 (Beer-Lambert line integral in log space),
    evaluated in closed form for a uniform cube. Used only as a sanity value
    for the TIGRE forward projector.
    """
    return mu * side_mm


def prepare_pip_cuda_home(cuda_home) -> list:
    """Make a pip-installed CUDA prefix usable by setup.py builds.

    The wheels ship only versioned libraries (libcudart.so.13) and a `lib`
    folder. Linkers ask for `-lcudart` (needs libcudart.so) and build scripts
    such as TIGRE's expect `lib64`. This adds the missing symlinks and makes
    the compiler binaries executable. Returns the list of links created.

    Ref: OURS (tooling). No file of R2-Gaussian, TIGRE or 4DRGS is touched.
    """
    cuda_home = Path(cuda_home)
    created = []
    lib = cuda_home / "lib"
    if lib.is_dir():
        for f in sorted(lib.iterdir()):
            m = re.match(r"^(lib[\w+-]+\.so)\.\d+", f.name)
            if m and not (lib / m.group(1)).exists():
                os.symlink(f.name, lib / m.group(1))
                created.append(str(lib / m.group(1)))
        lib64 = cuda_home / "lib64"
        if not lib64.exists():
            os.symlink("lib", lib64)
            created.append(str(lib64))
    bin_dir = cuda_home / "bin"
    if bin_dir.is_dir():
        for f in bin_dir.iterdir():
            if f.is_file():
                try:
                    f.chmod(f.stat().st_mode | 0o111)
                except OSError:  # read-only system file behind a symlink; already executable
                    pass
    return created


def nvcc_release(version_text: str | None) -> str | None:
    """CUDA release ("13.0") from the output of `nvcc --version`, or None.

    Ref: OURS (tooling). nvcc prints a line like
    "Cuda compilation tools, release 13.0, V13.0.88".
    """
    m = re.search(r"release (\d+\.\d+)", version_text or "")
    return m.group(1) if m else None


def same_cuda(a: str | None, b: str | None) -> bool:
    """True when two CUDA versions agree on major.minor ("13.0" vs "13.0.88").

    Ref: OURS (tooling). Used so every extension is built with the CUDA
    version torch was built with.
    """
    if not a or not b:
        return False
    return a.split(".")[:2] == b.split(".")[:2]


def pip_cuda_specs(torch_cuda: str) -> list:
    """pip requirement strings for an nvcc toolkit matching torch's CUDA.

    CUDA 13: the `cuda-toolkit` metapackage with extras (all into
    nvidia/cu13). CUDA 12: the per-component `-cu12` wheels (each into its
    own nvidia/<component> folder).

    Ref: OURS (tooling). Package names from PyPI: cuda-toolkit 13.0.2 extras;
    nvidia-cuda-nvcc-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-cccl-cu12.
    File lists checked Oct 3, 2026: nvidia-cuda-nvcc-cu12 12.8.93 has
    bin/ptxas, include/crt, nvvm/ but NO nvcc binary, so CUDA 12.8 cannot be
    built from pip alone.
    """
    major_minor = ".".join(torch_cuda.split(".")[:2])
    if int(major_minor.split(".")[0]) >= 13:
        return [f"cuda-toolkit[nvcc,nvvm,crt,cccl,cudart]=={major_minor}.*"]
    return [
        f"nvidia-cuda-nvcc-cu12=={major_minor}.*",
        f"nvidia-cuda-runtime-cu12=={major_minor}.*",
        f"nvidia-cuda-cccl-cu12=={major_minor}.*",
    ]


def find_pip_cuda12_prefixes(site_dirs) -> list:
    """All per-component CUDA 12 wheel folders, <site>/nvidia/<name>.

    Folders named like a CUDA 13 unified prefix (cu13, ...) are skipped so a
    leftover CUDA 13 install is never mixed in.

    Ref: OURS (tooling).
    """
    prefixes = []
    for site in site_dirs:
        base = Path(site) / "nvidia"
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if p.is_dir() and not re.fullmatch(r"cu\d+", p.name) and p not in prefixes:
                prefixes.append(p)
    return prefixes


def merge_cuda_prefixes(prefixes, dest) -> Path:
    """Build one CUDA_HOME from several pip prefixes with a symlink tree.

    For each prefix, every entry of bin/, include/, lib/ and nvvm/ is linked
    into the same subfolder of `dest` (first prefix wins on a name clash).
    Nothing inside site-packages is changed.

    Ref: OURS (tooling). Needed because the CUDA 12 wheels unpack into
    separate folders while setup.py builds expect a single CUDA_HOME.
    """
    dest = Path(dest)
    for prefix in prefixes:
        for sub in ("bin", "include", "lib", "nvvm"):
            src = Path(prefix) / sub
            if not src.is_dir():
                continue
            (dest / sub).mkdir(parents=True, exist_ok=True)
            for entry in sorted(src.iterdir()):
                link = dest / sub / entry.name
                if not link.exists() and not link.is_symlink():
                    os.symlink(entry, link)
    return dest


def find_include_dir(roots, header: str = "cub/cub.cuh") -> Path | None:
    """Directory D such that D/<header> exists, searched under each root.

    CUDA 13 moved the CCCL headers (cub, thrust, libcu++) from include/ to
    include/cccl/, so `#include <cub/cub.cuh>` no longer resolves with only
    CUDA_HOME/include on the path. Returns the shallowest match in the first
    root that has one, or None.

    Ref: OURS (tooling). CUDA 13 layout: NVIDIA CUDA 13.0 release notes, CCCL
    header location change (section UNVERIFIED).
    """
    parts = Path(header).parts
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        # Known spots first (these also work through symlinked folders,
        # which rglob does not descend into).
        for cand in (root / "include", root / "include" / "cccl", root):
            if (cand / header).exists():
                return cand
        hits = [p for p in root.rglob(parts[-1]) if p.parts[-len(parts):] == parts]
        if hits:
            best = min(hits, key=lambda p: len(p.parts))
            return Path(*best.parts[: -len(parts)])
    return None


def first_compiler_error(log: str | None) -> str | None:
    """First compiler error line of a failed build log, or None.

    Order: a line that names a source file and says "error" (gcc / nvcc
    style "file.cu:18: fatal error: ..."), then "nvcc fatal", then the first
    line with "error:" or "Error" that is not pip's own summary.

    Ref: OURS (tooling). Used to report why an Option A build failed.
    """
    lines = [l.strip() for l in (log or "").splitlines() if l.strip()]
    pip_noise = re.compile(r"subprocess-exited-with-error|Failed building|failed-wheel-build|^ERROR: ", re.I)
    patterns = [
        re.compile(r"\.(cu|cuh|cpp|cc|c|h|hpp)(\(\d+\)|:\d+)[^ ]*.*\berror\b", re.I),
        re.compile(r"nvcc fatal", re.I),
        re.compile(r"\berror\b\s*[:#]|\w+Error\b"),
    ]
    for pat in patterns:
        for line in lines:
            if pat.search(line) and not pip_noise.search(line):
                return line
    return None


def find_pip_cuda_prefixes(site_dirs, major: int) -> list:
    """All unified CUDA pip prefixes <site>/nvidia/cu<major>, in site order.

    With CUDA 13 the image keeps torch's CUDA libraries (cudart headers
    included) in the system site-packages, while the matching nvcc is
    installed into the venv. Both are nvidia/cu13 folders; together they form
    one toolkit. Duplicates (same real path) are dropped.

    Ref: OURS (tooling). Seen on molab, Oct 3: nvcc 13.0 in
    /tmp/uv-venv/.../nvidia/cu13, cuda_runtime.h only in
    /usr/local/lib/python3.13/site-packages/nvidia/cu13.
    """
    found, seen = [], set()
    for site in site_dirs:
        p = Path(site) / "nvidia" / f"cu{major}"
        if p.is_dir() and os.path.realpath(p) not in seen:
            seen.add(os.path.realpath(p))
            found.append(p)
    return found


def merge_r2g_eval_files(files) -> dict:
    """Merge R2-Gaussian eval*.yml files without letting the train and test
    2D files overwrite each other: 2D keys get the split as suffix
    (psnr_2d_test, psnr_2d_train); psnr_2d / ssim_2d are the TEST values,
    which is what R2-Gaussian prints (train.py evaluates render_train, then
    render_test, and prints the last one).

    Ref: R2-Gaussian code, train.py training_report (validation_configs
    order) at commit f2579bf. Added Oct 4 after the Step 0.1 stdout / yml
    mismatch (49.55 vs 45.26 dB) turned out to be test vs train views.
    """
    out = {}
    for name, text in files:
        vals = parse_r2g_eval_yml(text)
        split = None
        if "render_test" in name:
            split = "test"
        elif "render_train" in name:
            split = "train"
        for k, v in vals.items():
            out[f"{k}_{split}" if split and k.endswith("_2d") else k] = v
    for k in ("psnr_2d", "ssim_2d"):
        if f"{k}_test" in out:
            out[k] = out[f"{k}_test"]
    return out
