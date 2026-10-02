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


def find_pip_cuda_home(site_dirs) -> Path | None:
    """Find a CUDA toolkit root installed by NVIDIA's pip wheels.

    The CUDA 13 wheels (nvidia-cuda-nvcc, nvidia-cuda-runtime, ...) all unpack
    into one prefix, <site-packages>/nvidia/cu13/{bin,include,lib}, so that
    folder can serve as CUDA_HOME. Returns the first prefix that holds
    bin/nvcc, or None.

    Ref: OURS (tooling). Layout read from the file lists of the wheels
    nvidia_cuda_nvcc-13.0.88 and nvidia_cuda_runtime-13.0.96 on PyPI
    (Oct 2, 2026).
    """
    for site in site_dirs:
        base = Path(site) / "nvidia"
        if not base.is_dir():
            continue
        for prefix in sorted(base.glob("cu*"), reverse=True):
            if (prefix / "bin" / "nvcc").exists():
                return prefix
    return None


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
                f.chmod(f.stat().st_mode | 0o111)
    return created
