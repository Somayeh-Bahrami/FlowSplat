# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 (needs CUDA).
#
# Step 0.1: GPU environment and code base.
#   a) GitHub token from the GH_TOKEN environment variable if set, else from a
#      password field (never hardcoded, never printed)
#   b) use the FlowSplat repo this notebook already sits in (bootstrap case);
#      only if it is not inside the repo, clone or pull it
#   c) print torch / CUDA / GPU / compute capability; pick ONE CUDA_HOME whose
#      nvcc matches torch's CUDA version; install ninja
#      Option A: image torch (CUDA 13) + pinned CUDA 13 pip toolkit + CCCL
#      include flag. Option B (automatic if an A build fails): torch cu128 +
#      CUDA 12.8 pip toolkit; saved in WORK/cuda_option.json, script restarts
#      itself (in marimo: restart the kernel). Delete that file or set
#      FLOWSPLAT_CUDA_OPTION=A to retry A.
#   d) clone the OFFICIAL repos, each pinned to a commit hash
#   e) install and build them for this GPU (CUDA extensions included)
#   f) run R2-Gaussian on its own example case, print PSNR / SSIM vs published
#   g) one small TIGRE forward projection, checked against a closed-form value
#   g2) run 4DRGS on its own example case (plan bullet: "if public, run one case")
#   h) save everything to results/step0.1.json
#
# Rule: no source file of R2-Gaussian, TIGRE or 4DRGS is edited. Every
# deviation from their READMEs (environment versions, build flags, CLI flags)
# is appended to `deviations` and saved. `git diff` of each repo is checked at
# the end and stored as `source_untouched`.
#
# Two ways to run:
#   - opened in marimo: press the "Run Step 0.1 setup" button
#   - from a bootstrap notebook or a terminal, inside the cloned repo:
#       python notebooks/step0_setup.py
#     (script mode starts without the button; FLOWSPLAT_RUN_4DRGS=0 skips 4DRGS;
#     FLOWSPLAT_BUILD_ONLY=1 stops after the builds; FLOWSPLAT_ONLY_4DRGS=1 skips
#     the R2-Gaussian example and saves results/step0.1_4drgs.json)

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    import zipfile
    from pathlib import Path

    import marimo as mo

    return Path, json, mo, os, shutil, subprocess, sys, time, zipfile


@app.cell
def _(Path, os):
    # ---- Pins and published numbers. Looked up on Oct 1, 2026. ----
    FLOWSPLAT_URL = "https://github.com/Somayeh-Bahrami/FlowSplat"

    # Official repos only (see REFERENCES.md, Sources).
    R2G_URL = "https://github.com/Ruyi-Zha/r2_gaussian"
    R2G_COMMIT = "f2579bfddd9aac009cb797c8503bef8119bbd022"  # main, 2025-10-10
    TIGRE_URL = "https://github.com/CERN/TIGRE"
    TIGRE_COMMIT = "6b0951a8a88aa88a7db5b2a95181a33a95aa8b28"  # master, 2026-09-29 (v3.1.3 + 1)
    DRGS_URL = "https://github.com/ShanghaiTech-IMPACT/4DRGS"
    DRGS_COMMIT = "38006edbb105b363ae2ed6b075e1766ec70a91dc"  # 2026-08-24
    TCNN_URL = "https://github.com/NVlabs/tiny-cuda-nn"  # dependency of 4DRGS
    TCNN_COMMIT = "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38"
    # LEAP CT toolbox: 4DRGS imports it unconditionally (ct/leap_ct.py), README
    # "Setup" installs it from source. Default branch head, 2024-12-14.
    LEAP_URL = "https://github.com/LLNL/LEAP"
    LEAP_COMMIT = "0c8846f42b2e59340d5559fc1271d590a292f9a0"

    # R2-Gaussian example data: README Sec 2 "Dataset" link, synthetic_dataset,
    # file cone_ntrain_50_angle_360.zip. Example case from README Sec 3.2.
    R2G_DATA_ID = "1kv8GyTSMO_oRok5Qsgr5LVPFCrviko9B"
    R2G_CASE = "0_chest_cone"

    # Expected numbers.
    # PSNR: R2-Gaussian repo, assets/results.md, table "Synthetic dataset",
    #       row "Chest", column "R2-Gaussian (50-view)" = 35.09 dB (PSNR3D).
    # SSIM: no per-case SSIM is published. Only the 15-case average exists:
    #       R2-Gaussian paper (arXiv:2405.20693), Table 1, synthetic dataset,
    #       50-view, "Ours (iter=30k)": PSNR 37.98, SSIM 0.952.
    R2G_EXPECTED = {
        "psnr_3d_case": 35.09,
        "psnr_source": "R2-Gaussian repo assets/results.md, Synthetic dataset table, Chest, R2-Gaussian (50-view)",
        "ssim_3d_case": None,
        "ssim_3d_dataset_avg": 0.952,
        "psnr_3d_dataset_avg": 37.98,
        "avg_source": "R2-Gaussian paper Table 1, synthetic, 50-view, Ours (iter=30k). Average of 15 cases, not this case.",
    }
    PSNR_TOL_DB = 1.0

    # 4DRGS example data: README "Data-Preparation" link (folder case2).
    DRGS_DATA_URL = "https://drive.google.com/drive/folders/1J5eK-O0129R8u2LUHnvDLogw_NTmNwje"
    # Expected: 4DRGS README, table under "Sort-free X-ray rasterization",
    # case2, 30 views, "10k / 5k", Sort-free: Eval PSNR 36.349, Eval SSIM 0.9021.
    # (2D rendering metrics, measured by the authors on an RTX 4060 Ti.)
    DRGS_EXPECTED = {
        "psnr_eval": 36.349,
        "ssim_eval": 0.9021,
        "source": "4DRGS README, Sort-free X-ray rasterization table, case2, 30 views, 10k / 5k, Sort-free",
    }

    WORK = Path(os.environ.get("FLOWSPLAT_WORK", str(Path.home() / "flowsplat_work")))
    return (
        DRGS_COMMIT,
        DRGS_DATA_URL,
        DRGS_EXPECTED,
        DRGS_URL,
        FLOWSPLAT_URL,
        PSNR_TOL_DB,
        R2G_CASE,
        R2G_COMMIT,
        R2G_DATA_ID,
        R2G_EXPECTED,
        R2G_URL,
        LEAP_COMMIT,
        LEAP_URL,
        TCNN_COMMIT,
        TCNN_URL,
        TIGRE_COMMIT,
        TIGRE_URL,
        WORK,
    )


@app.cell
def _(mo, os):
    # a) Token: GH_TOKEN from the environment wins; else it is typed here at run
    # time. It is never written to disk or printed.
    token_box = mo.ui.text(
        kind="password",
        label="GitHub token (not needed when GH_TOKEN is set or the repo is already cloned)",
    )
    run_4drgs = mo.ui.checkbox(
        value=os.environ.get("FLOWSPLAT_RUN_4DRGS", "1") != "0",
        label="Also run 4DRGS on its example case (adds a tiny-cuda-nn build)",
    )
    run_btn = mo.ui.run_button(label="Run Step 0.1 setup")
    # Script mode (python notebooks/step0_setup.py) has no button to press.
    AUTORUN = mo.app_meta().mode == "script" or os.environ.get("FLOWSPLAT_AUTORUN") == "1"
    mo.vstack([mo.md("## Step 0.1: GPU environment and code base"), token_box, run_4drgs, run_btn])
    return AUTORUN, run_4drgs, run_btn, token_box


@app.cell
def _(os, shutil, subprocess, sys):
    def sh(cmd, cwd=None, env=None, secret=None, check=True):
        """Run a command, stream its output with secrets removed, return the text.

        Ref: OURS (tooling).
        """
        full_env = dict(os.environ)
        full_env.update(env or {})
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines = []
        for line in proc.stdout:
            if secret:
                line = line.replace(secret, "***")
            lines.append(line)
            print(line, end="")
        proc.wait()
        out = "".join(lines)
        if check and proc.returncode != 0:
            err = RuntimeError(f"Command failed ({proc.returncode}): {' '.join(str(c) for c in cmd[:4])} ...")
            err.output = out  # full log, so the caller can report the first compiler error
            raise err
        return out

    def pip_install(args, cwd=None, env=None, check=True, shadow=False):
        """pip install into the notebook's interpreter (uv if pip is missing).

        shadow=True installs a fresh copy into this interpreter's own
        site-packages even when the package is already visible from the
        system site-packages (pip --ignore-installed, uv --reinstall). Used
        for the Option B torch, because the image's torch sits outside the
        venv and can be neither uninstalled nor upgraded from inside it.

        Ref: OURS (tooling).
        """
        has_pip = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0
        if has_pip:
            cmd = [sys.executable, "-m", "pip", "install", *(["--ignore-installed"] if shadow else []), *args]
        elif shutil.which("uv"):
            cmd = ["uv", "pip", "install", "--python", sys.executable, *(["--reinstall"] if shadow else []), *args]
        else:
            raise RuntimeError("Neither pip nor uv is available in this environment.")
        return sh(cmd, cwd=cwd, env=env, check=check)

    def clone_pinned(url, commit, dest, env=None, secret=None):
        """Clone `url` into `dest` and check out exactly `commit` (with submodules).

        Ref: OURS (tooling).
        """
        if not (dest / ".git").exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            sh(["git", "clone", url, dest], env=env, secret=secret)
        else:
            sh(["git", "fetch", "--all", "--tags"], cwd=dest, env=env, secret=secret)
        sh(["git", "checkout", "--detach", commit], cwd=dest)
        sh(["git", "submodule", "update", "--init", "--recursive"], cwd=dest)
        head = sh(["git", "rev-parse", "HEAD"], cwd=dest).strip()
        if head != commit:
            raise RuntimeError(f"{dest.name}: HEAD {head} is not the pinned commit {commit}")
        return head

    def source_untouched(repo):
        """True when no tracked file of the repo (or its submodules) was edited.

        Ref: OURS (tooling). Enforces the rule "do not modify R2-Gaussian or
        TIGRE source".
        """
        a = subprocess.run(["git", "diff", "--quiet"], cwd=str(repo)).returncode == 0
        b = subprocess.run(
            ["git", "submodule", "foreach", "--recursive", "--quiet", "git diff --quiet"], cwd=str(repo)
        ).returncode == 0
        return a and b

    return clone_pinned, pip_install, sh, source_untouched


@app.cell
def _(AUTORUN, FLOWSPLAT_URL, Path, WORK, mo, os, run_btn, sh, sys, token_box):
    # b) Locate the FlowSplat repo. If this notebook already runs inside a
    # clone (molab bootstrap case), use it as is. Otherwise clone or pull.
    mo.stop(not (run_btn.value or AUTORUN), mo.md("Press **Run Step 0.1 setup** to start."))
    import base64 as _b64

    def _repo_root():
        """First parent of the notebook dir or cwd that holds src/flowsplat. Ref: OURS (tooling)."""
        _starts = [Path.cwd()]
        try:
            _starts.insert(0, Path(mo.notebook_dir()))
        except Exception:
            pass
        for _start in _starts:
            for _p in [_start.resolve(), *_start.resolve().parents]:
                if (_p / "src" / "flowsplat" / "setup_check.py").exists():
                    return _p
        return None

    WORK.mkdir(parents=True, exist_ok=True)
    FLOWSPLAT_DIR = _repo_root()
    if FLOWSPLAT_DIR is not None:
        print("Already inside the repo, clone skipped:", FLOWSPLAT_DIR)
    else:
        _token = (os.environ.get("GH_TOKEN") or token_box.value or "").strip()
        _env = {"GIT_TERMINAL_PROMPT": "0"}
        if _token:
            # Token goes to git as an HTTP header through the environment, so it
            # is not stored in .git/config and not on the command line.
            # Same mechanism as flowsplat.setup_check.git_auth_env (not
            # importable yet, because the repo is not cloned at this point).
            _enc = _b64.b64encode(f"x-access-token:{_token}".encode()).decode()
            _env.update(
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {_enc}",
                }
            )
        FLOWSPLAT_DIR = WORK / "FlowSplat"
        if (FLOWSPLAT_DIR / ".git").exists():
            sh(["git", "pull", "--ff-only"], cwd=FLOWSPLAT_DIR, env=_env, secret=_token or None)
        else:
            sh(["git", "clone", FLOWSPLAT_URL, FLOWSPLAT_DIR], env=_env, secret=_token or None)
    flowsplat_commit = sh(["git", "rev-parse", "HEAD"], cwd=FLOWSPLAT_DIR, check=False).strip()

    sys.path.insert(0, str(FLOWSPLAT_DIR / "src"))
    try:
        from flowsplat import setup_check
    except ImportError as _e:
        raise RuntimeError(
            "src/flowsplat/setup_check.py is not in the repo. Pull the latest main first."
        ) from _e
    print("FlowSplat at", flowsplat_commit)
    return FLOWSPLAT_DIR, flowsplat_commit, setup_check


@app.cell
def _(Path, WORK, flowsplat_commit, json, mo, os, pip_install, setup_check, shutil, subprocess, sys):
    # c) torch version, CUDA version, GPU name, compute capability, and ONE
    # CUDA_HOME whose nvcc has the same CUDA version as torch.
    #
    # Option A (default): keep the image's torch; nvcc toolkit pinned to
    #   torch's CUDA; CCCL include dir added as a build flag.
    # Option B (automatic, only if an Option A build fails): torch +
    #   torchvision cu128 wheels and the CUDA 12.8 pip toolkit. Blackwell
    #   (sm_120) is supported from CUDA 12.8. The choice is kept in
    #   WORK/cuda_option.json so it survives the restart that the torch
    #   switch needs. FLOWSPLAT_CUDA_OPTION=A|B overrides it.
    import site as _site
    import sysconfig as _sysconfig

    deviations = []
    CUDA_OPTION_FILE = WORK / "cuda_option.json"
    _saved = json.loads(CUDA_OPTION_FILE.read_text()) if CUDA_OPTION_FILE.exists() else {}
    CUDA_OPTION = os.environ.get("FLOWSPLAT_CUDA_OPTION") or _saved.get("option", "A")
    _B_TORCH_CUDA = "12.8"
    _B_INDEX = "https://download.pytorch.org/whl/cu128"

    _venv_sites = list(dict.fromkeys([_sysconfig.get_paths()["purelib"], _sysconfig.get_paths()["platlib"]]))
    if CUDA_OPTION == "B":
        # The image's torch (cu130) is in the system site-packages, outside
        # this venv: pip can neither uninstall nor replace it. So a cu128 copy
        # is installed INTO the venv and put first on the path (also for the
        # build subprocesses, via PYTHONPATH) so that it shadows the system one.
        for _p in reversed(_venv_sites):
            if _p in sys.path:
                sys.path.remove(_p)
            sys.path.insert(0, _p)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [*_venv_sites, *[x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x]]
        )
        _venv_torch = [d for site in _venv_sites for d in Path(site).glob("torch-*.dist-info")]
        _have_cu128 = any("+cu128" in d.name for d in _venv_torch)
        if not _have_cu128:
            if os.environ.get("FLOWSPLAT_B_RESTARTED") == "1":
                raise RuntimeError(f"Option B: torch cu128 still not in the venv {_venv_sites} after install and restart.")
            # All deps go into the venv too (cu12 NVIDIA libraries are not in
            # the image), so torch and its CUDA libraries come from one index.
            pip_install(["torch", "torchvision", "--index-url", _B_INDEX], shadow=True)
            # A fresh process, so no stale import state from the system torch.
            print("Option B: torch cu128 installed into the venv. Restarting the Python process.", flush=True)
            if mo.app_meta().mode == "script":
                os.environ["FLOWSPLAT_B_RESTARTED"] = "1"
                sys.stdout.flush()
                os.execv(sys.executable, [sys.executable, *sys.argv])
            raise RuntimeError("Option B: torch cu128 installed into the venv. Restart the marimo kernel and press Run again.")
    try:
        import torch
    except ImportError:
        # Blackwell (sm_120) needs a recent wheel built against CUDA 12.8.
        pip_install(["torch", "torchvision", "--index-url", _B_INDEX])
        import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to torch. Apply the Step 0.1 fallback (another cloud GPU).")
    # Which torch copy was loaded, and from where.
    print("sys.executable:", sys.executable)
    print("sys.path (in order):")
    for _i, _p in enumerate(sys.path):
        print(f"  [{_i}] {_p}")
    print("torch.__file__:", torch.__file__)
    print("torch:", torch.__version__, "| CUDA", torch.version.cuda, "| arch list", torch.cuda.get_arch_list())
    _cap = torch.cuda.get_device_capability(0)
    _arch = f"sm_{_cap[0]}{_cap[1]}"
    if CUDA_OPTION == "B":
        if torch.version.cuda != _B_TORCH_CUDA:
            raise RuntimeError(
                f"Option B expected torch with CUDA {_B_TORCH_CUDA}, got {torch.version.cuda} from {torch.__file__}. "
                "See sys.path above: the venv copy must come before the system site-packages."
            )
        if _arch not in torch.cuda.get_arch_list():
            raise RuntimeError(f"Option B torch {torch.__version__} has no {_arch} in its arch list.")
    if _saved.get("first_error"):
        print("Option A failed with:", _saved["first_error"])
    if CUDA_OPTION == "B":
        deviations.append(
            {
                "what": f"Option B: torch {torch.__version__} + torchvision from {_B_INDEX}, CUDA {_B_TORCH_CUDA} pip toolkit",
                "instead_of": "the image's torch (CUDA 13) with a CUDA 13 toolkit (Option A)",
                "reason": "Option A build failed: "
                + str(_saved.get("first_error") or _saved.get("reason", "forced by FLOWSPLAT_CUDA_OPTION"))[:300]
                + f". Installed into the venv with --ignore-installed so it shadows the image's torch in the system "
                f"site-packages; torch loaded from {torch.__file__}. Blackwell sm_120 is supported in CUDA 12.8. "
                "No source change.",
            }
        )

    _torch_cuda = torch.version.cuda

    def _run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        except OSError:
            return None

    def _nvcc_ok(path):
        """True when `path` is an nvcc whose release equals torch's CUDA. Ref: OURS (tooling)."""
        return bool(path) and os.path.exists(path) and setup_check.same_cuda(
            setup_check.nvcc_release(_run([path, "--version"])), _torch_cuda
        )

    # 1) A system toolkit is used only if its version matches torch exactly.
    _nvcc = None
    _rejected = []
    for _cand in [
        shutil.which("nvcc"),
        *[os.path.join(h, "bin", "nvcc") for h in [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"] if h],
    ]:
        if _cand and os.path.exists(_cand):
            if _nvcc_ok(_cand) and "site-packages" not in _cand:
                _nvcc = _cand
                break
            _rejected.append(f"{_cand} ({setup_check.nvcc_release(_run([_cand, '--version']))})")
    _toolkit_source = "system"

    # 2) Otherwise the toolkit comes from NVIDIA's pip wheels, pinned to torch's
    # CUDA version, into this interpreter's site-packages.
    if _nvcc is None:
        _specs = setup_check.pip_cuda_specs(_torch_cuda)
        pip_install(_specs)
        _sites = [_sysconfig.get_paths()["purelib"], _sysconfig.get_paths()["platlib"], *_site.getsitepackages()]
        _major = int(_torch_cuda.split(".")[0])
        if _major >= 13:
            # The matching nvcc lands in the venv's nvidia/cu13, but cudart
            # (cuda_runtime.h, libcudart) stays in the system's nvidia/cu13,
            # where torch's copy already satisfies the pin. Merge them into
            # one CUDA_HOME by symlinks, venv first so its nvcc 13.0 wins.
            _home = setup_check.merge_cuda_prefixes(
                setup_check.find_pip_cuda_prefixes([*_venv_sites, *_sites], _major), WORK / f"cuda-{_torch_cuda}"
            )
        else:
            _home = setup_check.merge_cuda_prefixes(
                # venv only: the system site-packages hold the image's CUDA 13 libraries.
                setup_check.find_pip_cuda12_prefixes(_venv_sites), WORK / f"cuda-{_torch_cuda}"
            )
            if not (_home / "bin" / "nvcc").exists():
                raise RuntimeError(
                    "Option B: no nvcc for CUDA 12.8 on PyPI. The wheel nvidia-cuda-nvcc-cu12 12.8.93 ships only "
                    "ptxas, crt headers and libnvvm, not the nvcc driver (file list checked Oct 3, 2026). "
                    "A CUDA 12.8 nvcc would need another source (conda or NVIDIA runfile), which is a decision "
                    "for the user. Plan fallback: \"If the CUDA rasterizer will not build: another cloud GPU. If "
                    "nothing works by Oct 4: plain PyTorch Gaussian splatter (slower, no custom CUDA). Do not "
                    "change the project.\""
                )
        if (_home / "bin" / "nvcc").exists():
            _links = setup_check.prepare_pip_cuda_home(_home)
            _nvcc = str(_home / "bin" / "nvcc")
            _toolkit_source = f"pip ({', '.join(_specs)})"
            deviations.append(
                {
                    "what": f"CUDA compiler installed from pip: {', '.join(_specs)}; CUDA_HOME = {_home} "
                    "(symlink tree: venv nvcc prefix first, then the system prefix with cudart)",
                    "instead_of": "a system CUDA toolkit, as the READMEs assume",
                    "reason": "The molab image has no matching nvcc. Pinned to the CUDA version of the installed torch "
                    f"({_torch_cuda}); rejected: {_rejected or 'none'}. "
                    f"Symlinks added for the linker: {len(_links)}. No source change.",
                }
            )
    if _nvcc is not None and not _nvcc_ok(_nvcc):
        raise RuntimeError(
            f"nvcc at {_nvcc} reports CUDA {setup_check.nvcc_release(_run([_nvcc, '--version']))}, "
            f"torch was built with {_torch_cuda}. One CUDA version must be used for every build."
        )

    # 3) ninja, so torch's cpp_extension builds in parallel instead of falling
    # back to the slow distutils path.
    pip_install(["ninja"])
    _scripts = _sysconfig.get_paths()["scripts"]
    os.environ["PATH"] = _scripts + os.pathsep + os.environ.get("PATH", "")

    env_info = {
        "flowsplat_commit": flowsplat_commit,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{_cap[0]}.{_cap[1]}",
        "torch_arch_list": torch.cuda.get_arch_list(),
        "torch_supports_gpu_arch": _arch in torch.cuda.get_arch_list(),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "cuda_option": CUDA_OPTION,
        "option_a_first_error": _saved.get("first_error"),
        "torch_file": torch.__file__,
        "sys_path": list(sys.path),
        "nvcc_path": _nvcc,
        "nvcc_rejected": _rejected,
        "cuda_toolkit_source": _toolkit_source,
        "ninja": _run(["ninja", "--version"]),
        "nvcc_version": _run([_nvcc, "--version"]).splitlines()[-2:] if _nvcc else None,
        "gcc": (_run(["gcc", "--version"]) or "").splitlines()[:1],
    }
    for _k, _v in env_info.items():
        print(f"{_k}: {_v}")
    if _nvcc is None:
        raise RuntimeError(
            "nvcc not found, and the pip install of the CUDA compiler did not provide one. "
            "The CUDA extensions cannot be built. Apply the Step 0.1 fallback (another cloud GPU)."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(_nvcc))
    # ONE environment for every CUDA build below (R2-Gaussian, TIGRE,
    # tiny-cuda-nn, 4DRGS). Build scripts look for nvcc on PATH (tiny-cuda-nn)
    # and for CUDA_HOME (torch, TIGRE).
    CUDA_ENV = {"CUDA_HOME": CUDA_HOME, "CUDA_PATH": CUDA_HOME}
    # CUDA 13 keeps cub / thrust under include/cccl. If that is not already
    # CUDA_HOME/include, add it to the include path for gcc (CPATH) and nvcc.
    _cccl = setup_check.find_include_dir([CUDA_HOME, os.path.join(_sysconfig.get_paths()["platlib"], "nvidia")])
    if _cccl is None:
        print("WARNING: cub/cub.cuh not found under", CUDA_HOME)
    elif os.path.realpath(_cccl) != os.path.realpath(os.path.join(CUDA_HOME, "include")):
        CUDA_ENV["CPATH"] = str(_cccl) + (os.pathsep + os.environ["CPATH"] if os.environ.get("CPATH") else "")
        CUDA_ENV["NVCC_APPEND_FLAGS"] = f"-I{_cccl}"
        deviations.append(
            {
                "what": f"CCCL include dir added as a build flag: CPATH and NVCC_APPEND_FLAGS='-I{_cccl}'",
                "instead_of": "plain build (cub/cub.cuh found under CUDA_HOME/include, CUDA 12 layout)",
                "reason": "CUDA 13 moved cub/thrust to include/cccl; simple-knn failed with "
                "'cub/cub.cuh: No such file or directory'. Build flag only; source untouched.",
            }
        )
    env_info["cccl_include"] = str(_cccl) if _cccl else None
    os.environ.update(CUDA_ENV)
    os.environ["PATH"] = os.path.join(CUDA_HOME, "bin") + os.pathsep + os.environ.get("PATH", "")
    deviations.append(
        {
            "what": f"Environment: python {env_info['python']}, torch {env_info['torch']} (CUDA {env_info['torch_cuda']})",
            "instead_of": "R2-Gaussian README: python 3.9, torch 2.1.2+cu118 (or 1.12.1 + cudatoolkit 11.6); requirements.txt version pins",
            "reason": f"GPU compute capability {env_info['compute_capability']} is not supported by those old builds. "
            "requirements.txt is not installed with its pins; the same packages are installed unpinned. No source change.",
        }
    )
    return CUDA_ENV, CUDA_OPTION, CUDA_OPTION_FILE, deviations, env_info, torch


@app.cell
def _(
    DRGS_COMMIT,
    DRGS_URL,
    R2G_COMMIT,
    R2G_URL,
    LEAP_COMMIT,
    LEAP_URL,
    TCNN_COMMIT,
    TCNN_URL,
    TIGRE_COMMIT,
    TIGRE_URL,
    WORK,
    clone_pinned,
    deviations,
    env_info,
    run_4drgs,
):
    # d) Clone the official repos, each at its pinned commit.
    _ = env_info
    EXT = WORK / "external"
    R2G_DIR = EXT / "r2_gaussian"
    TIGRE_DIR = EXT / "TIGRE"
    DRGS_DIR = EXT / "4DRGS"
    TCNN_DIR = EXT / "tiny-cuda-nn"
    LEAP_DIR = EXT / "LEAP"
    pins = {
        "r2_gaussian": {"url": R2G_URL, "commit": clone_pinned(R2G_URL, R2G_COMMIT, R2G_DIR)},
        "tigre": {"url": TIGRE_URL, "commit": clone_pinned(TIGRE_URL, TIGRE_COMMIT, TIGRE_DIR)},
    }
    if run_4drgs.value:
        pins["4drgs"] = {"url": DRGS_URL, "commit": clone_pinned(DRGS_URL, DRGS_COMMIT, DRGS_DIR)}
        pins["tiny_cuda_nn"] = {"url": TCNN_URL, "commit": clone_pinned(TCNN_URL, TCNN_COMMIT, TCNN_DIR)}
        pins["leap"] = {"url": LEAP_URL, "commit": clone_pinned(LEAP_URL, LEAP_COMMIT, LEAP_DIR)}
    deviations.append(
        {
            "what": f"TIGRE at commit {TIGRE_COMMIT[:7]} (master, v3.1.3 line)",
            "instead_of": "R2-Gaussian README: TIGRE v2.3 release zip",
            "reason": "Project rule: pin the official repo to a commit. v3.1.x lists GPU architectures from nvcc "
            "at build time, which is needed for a Blackwell GPU. No source change.",
        }
    )
    print(pins)
    return DRGS_DIR, LEAP_DIR, R2G_DIR, TCNN_DIR, TIGRE_DIR, pins


@app.cell
def _(
    CUDA_ENV,
    CUDA_OPTION,
    CUDA_OPTION_FILE,
    R2G_DIR,
    TIGRE_DIR,
    deviations,
    json,
    mo,
    os,
    pins,
    pip_install,
    setup_check,
    subprocess,
    sys,
    time,
):
    # e) Install and build R2-Gaussian and TIGRE for this GPU.
    _ = pins
    _t0 = time.time()
    _env = dict(CUDA_ENV)  # the one CUDA environment from cell c

    # Same packages as r2_gaussian/requirements.txt and environment.yml, unpinned.
    # Installed one by one so that a package without a wheel for this Python
    # is reported by name instead of stopping everything.
    _failed = []
    for _pkg in [
        "numpy", "Cython", "setuptools", "wheel", "gdown", "pyyaml", "tqdm", "matplotlib",
        "open3d", "opencv-python-headless", "plyfile", "scipy", "SimpleITK", "scikit-image",
        "tensorboard", "tensorboardX", "pydicom", "pyvista", "tifffile", "torchvision",
    ]:
        try:
            pip_install([_pkg])
        except RuntimeError:
            _failed.append(_pkg)
    if _failed:
        print("WARNING: could not install:", _failed)

    # CUDA extensions import torch in setup.py, so pip's isolated build env
    # (which has no torch) cannot build them. Build flag only, no source change.
    deviations.append(
        {
            "what": "All CUDA extensions (simple-knn, xray rasterizer/voxelizer, TIGRE, tiny-cuda-nn, "
            "4DRGS submodules) installed with `pip install --no-build-isolation`",
            "instead_of": "plain `pip install` from the READMEs",
            "reason": "Isolated build env has no torch: ModuleNotFoundError at the simple-knn build "
            "(molab, Oct 2). Builds against the installed torch. Build flag only; source untouched.",
        }
    )

    def _importable(module):
        """True when `module` imports in a fresh interpreter, run outside the repo dirs.

        Ref: OURS (tooling). Catches a build that succeeds but is not on the
        import path, as seen on molab Oct 3: "No module named 'simple_knn'"
        in train.py after a successful `pip install -e`.
        """
        _r = subprocess.run(
            [sys.executable, "-c", f"import torch, {module}"], cwd="/", capture_output=True, text=True
        )
        if _r.returncode != 0:
            print(f"import {module} failed:", (_r.stderr.strip().splitlines() or ["?"])[-1])
        return _r.returncode == 0

    def _build(path, label, module):
        """Build one CUDA extension, then check that `module` imports.

        Install modes, in order, each kept only if the import works:
        `pip install -e` (README), `-e` with setuptools editable_mode=compat,
        regular (non-editable) install. Each is tried plain and, if the
        compile fails, once more with a logged nvcc flag.

        Ref: R2-Gaussian README Sec 1 (pip path: `pip install -e
        r2_gaussian/submodules/simple-knn`). Fallback modes: OURS (tooling).
        Why -e can fail: simple-knn's setup.py builds `simple_knn._C` without
        listing a Python package, so the setuptools >= 64 editable finder may
        not map `simple_knn` (setup.py contents UNVERIFIED: gitlab.inria.fr
        blocked automated reading).
        """
        _modes = [
            ("editable", ["-e", str(path)]),
            ("editable_compat", ["-e", str(path), "--config-settings", "editable_mode=compat"]),
            ("regular", [str(path)]),
        ]
        for _i, (_mode, _args) in enumerate(_modes):
            _how = _compile(_args, label)
            if _importable(module):
                if _i > 0:
                    deviations.append(
                        {
                            "what": f"{label}: installed as '{_mode}' ({' '.join(_args[1:]) or 'pip install <path>'})",
                            "instead_of": "`pip install -e` from the R2-Gaussian README",
                            "reason": f"The -e build compiled but `import {module}` failed (module not on the "
                            "import path). Install mode only; source untouched.",
                        }
                    )
                return f"{_mode}+{_how}"
        # ImportError, not RuntimeError: this is not a compiler failure, so it
        # must not trigger the Option A -> B switch below.
        raise ImportError(f"{label}: built, but `import {module}` fails in every install mode (see messages above).")

    def _compile(args, label):
        """pip-install one extension; plain first, one retry with a logged nvcc flag. Ref: OURS (tooling)."""
        try:
            pip_install(["--no-build-isolation", *args], env=_env)
            return "plain"
        except RuntimeError:
            # Build-flag-only retry. NVCC_APPEND_FLAGS is read by nvcc itself, so
            # no file of the repo is edited. Needed when a newer GCC no longer
            # pulls in <cfloat> / <cstdint> implicitly.
            _flags = "-include cfloat -include cstdint"
            _append = (_env.get("NVCC_APPEND_FLAGS", "") + " " + _flags).strip()
            pip_install(["--no-build-isolation", *args], env={**_env, "NVCC_APPEND_FLAGS": _append})
            deviations.append(
                {
                    "what": f"{label}: built with NVCC_APPEND_FLAGS='{_flags}'",
                    "instead_of": "plain `pip install -e` from the README",
                    "reason": "Plain build failed with this compiler. Flag only; source untouched.",
                }
            )
            return "nvcc_append_flags"

    try:
        build_info = {
            "cuda_option": CUDA_OPTION,
            # Module names as imported by R2-Gaussian at commit f2579bf:
            # gaussian_model.py `from simple_knn._C import distCUDA2`,
            # render_query.py `from xray_gaussian_rasterization_voxelization import (...)`.
            "simple_knn": _build(
                R2G_DIR / "r2_gaussian" / "submodules" / "simple-knn", "simple-knn", "simple_knn._C"
            ),
            "xray_rasterizer_voxelizer": _build(
                R2G_DIR / "r2_gaussian" / "submodules" / "xray-gaussian-rasterization-voxelization",
                "xray-gaussian-rasterization-voxelization",
                "xray_gaussian_rasterization_voxelization",
            ),
        }
        # TIGRE Python package (pyproject.toml is at the repo root at this commit).
        pip_install(["--no-build-isolation", str(TIGRE_DIR)], env=_env)
        build_info["tigre"] = "plain"
    except RuntimeError as _e:
        if CUDA_OPTION == "B":
            print("\n=== Option B failed. First compiler error line: ===")
            print(setup_check.first_compiler_error(getattr(_e, "output", "")) or "(none found; see the log above)")
            raise RuntimeError(
                "Option B (torch cu128 + CUDA 12.8) build failed too. Plan fallback: \"If the CUDA rasterizer "
                "will not build: another cloud GPU. If nothing works by Oct 4: plain PyTorch Gaussian splatter "
                "(slower, no custom CUDA). Do not change the project.\""
            ) from _e
        # Option A failed: switch to Option B. torch cannot be swapped inside a
        # running process, so the choice is saved and the run restarts.
        _first = setup_check.first_compiler_error(getattr(_e, "output", ""))
        print("\n=== Option A failed. First compiler error line: ===")
        print(_first or "(no compiler error line found; see the build log above)")
        CUDA_OPTION_FILE.write_text(
            json.dumps(
                {
                    "option": "B",
                    "first_error": _first,
                    "reason": f"Option A build failed ({_e}); see the build log above",
                    "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
        )
        print(f"\n=== Option A build failed. Saved Option B to {CUDA_OPTION_FILE}. ===")
        if mo.app_meta().mode == "script":
            print("Restarting the script with Option B (torch cu128 + CUDA 12.8 toolkit).", flush=True)
            # Drop the CUDA 13 build variables of Option A so they do not leak
            # into the CUDA 12.8 builds of the restarted process.
            for _k in CUDA_ENV:
                os.environ.pop(_k, None)
            sys.stdout.flush()
            os.execv(sys.executable, [sys.executable, *sys.argv])
        raise RuntimeError(
            "Option A build failed; Option B is saved. Restart the marimo kernel and press Run again."
        ) from _e
    build_info["pip_failed"] = _failed
    build_info["build_minutes"] = round((time.time() - _t0) / 60, 1)
    print(build_info)
    return (build_info,)


@app.cell
def _(
    PSNR_TOL_DB,
    R2G_CASE,
    R2G_DATA_ID,
    R2G_DIR,
    R2G_EXPECTED,
    WORK,
    build_info,
    mo,
    os,
    setup_check,
    sh,
    sys,
    time,
    zipfile,
):
    # f) R2-Gaussian on its own example case (README Sec 3.2: `python train.py -s XXX/0_chest_cone`).
    _ = build_info
    # FLOWSPLAT_BUILD_ONLY=1: stop after the builds (used by
    # notebooks/step0_imagecas_recon.py in a fresh molab session). Nothing
    # below runs, so results/step0.1.json is not overwritten.
    mo.stop(os.environ.get("FLOWSPLAT_BUILD_ONLY") == "1", mo.md("Build only: stopped after the builds."))
    # FLOWSPLAT_ONLY_4DRGS=1: skip this example (already reproduced, see
    # results/step0.1.json) and run only the TIGRE check and 4DRGS.
    if os.environ.get("FLOWSPLAT_ONLY_4DRGS") == "1":
        r2g_result = {"skipped": "FLOWSPLAT_ONLY_4DRGS=1; see results/step0.1.json"}
    else:
        _data = WORK / "data_r2g"
        _data.mkdir(parents=True, exist_ok=True)
        _zip = _data / "cone_ntrain_50_angle_360.zip"
        _found = list(_data.rglob(f"{R2G_CASE}/meta_data.json"))
        if not _found:
            if not _zip.exists():
                sh([sys.executable, "-m", "gdown", R2G_DATA_ID, "-O", _zip])
            with zipfile.ZipFile(_zip) as _z:
                _members = [m for m in _z.namelist() if f"{R2G_CASE}/" in m]
                _z.extractall(_data, members=_members)
            _found = list(_data.rglob(f"{R2G_CASE}/meta_data.json"))
        if not _found:
            raise RuntimeError(f"{R2G_CASE} not found inside the R2-Gaussian dataset zip.")
        _case = _found[0].parent
        _out = WORK / "output_r2g" / R2G_CASE

        _t0 = time.time()
        # Default settings (30k iterations), exactly the README command plus -m.
        _log = sh([sys.executable, "train.py", "-s", _case, "-m", _out], cwd=R2G_DIR)
        _minutes = round((time.time() - _t0) / 60, 1)

        _metrics = {}
        _eval_dir = _out / "eval" / "iter_030000"
        if _eval_dir.exists():
            _metrics.update(setup_check.merge_r2g_eval_files(
                [(_f.name, _f.read_text()) for _f in sorted(_eval_dir.glob("eval*.yml"))]))
        if "psnr_3d" not in _metrics:
            _metrics = setup_check.parse_r2g_stdout(_log)

        _cmp = setup_check.compare_psnr(_metrics.get("psnr_3d"), R2G_EXPECTED["psnr_3d_case"], PSNR_TOL_DB)
        r2g_result = {
            "case": R2G_CASE,
            "views": 50,
            "iterations": 30000,
            "train_minutes": _minutes,
            "psnr_3d": _metrics.get("psnr_3d"),
            "ssim_3d": _metrics.get("ssim_3d"),
            "psnr_2d": _metrics.get("psnr_2d"),
            "ssim_2d": _metrics.get("ssim_2d"),
            "expected": R2G_EXPECTED,
            "psnr_diff_db": _cmp["diff_db"],
            "pass_within_1db": _cmp["pass"],
        }
        print("\n=== R2-Gaussian,", R2G_CASE, "===")
        print(f"PSNR3D ours {r2g_result['psnr_3d']} dB | expected {R2G_EXPECTED['psnr_3d_case']} dB ({R2G_EXPECTED['psnr_source']})")
        print(f"SSIM3D ours {r2g_result['ssim_3d']} | no per-case value published; 15-case average {R2G_EXPECTED['ssim_3d_dataset_avg']} ({R2G_EXPECTED['avg_source']})")
        print(f"diff {_cmp['diff_db']} dB | PASS (within {PSNR_TOL_DB} dB): {_cmp['pass']} | time {_minutes} min")
    return (r2g_result,)


@app.cell
def _(r2g_result, setup_check, time, torch):
    # g) One small TIGRE forward projection on this GPU.
    # A uniform cube (mu = 1) at the isocenter: the central ray travels the
    # full side length, so the central detector value must equal sVoxel.
    # Ref: R2-Gaussian Sec 3.1 Eq 1 (line integral); TIGRE default geometry
    # (demo file number UNVERIFIED).
    _ = r2g_result, torch
    import importlib.metadata as _md

    import numpy as _np
    import tigre as _tigre

    _geo = _tigre.geometry(mode="cone", nVoxel=_np.array([64, 64, 64]), default=True)
    _vol = _np.ones((64, 64, 64), dtype=_np.float32)
    _angles = _np.linspace(0, 2 * _np.pi, 4, endpoint=False).astype(_np.float32)
    _t0 = time.time()
    _proj = _tigre.Ax(_vol, _geo, _angles)
    _ms = round((time.time() - _t0) * 1000, 1)
    _nv, _nu = _proj.shape[1], _proj.shape[2]
    _center = float(_proj[0, _nv // 2 - 1 : _nv // 2 + 1, _nu // 2 - 1 : _nu // 2 + 1].mean())
    _expected = setup_check.cube_center_ray_integral(float(_geo.sVoxel[1]))
    _rel = abs(_center - _expected) / _expected
    tigre_result = {
        "version": _md.version("tigre"),  # tigre has no __version__
        "proj_shape": list(_proj.shape),
        "finite": bool(_np.isfinite(_proj).all()),
        "center_value": round(_center, 3),
        "expected_center_value": _expected,
        "rel_error": round(_rel, 5),
        "forward_ms": _ms,
        "pass": bool(_np.isfinite(_proj).all() and _rel < 0.02),
    }
    print("=== TIGRE forward projection ===")
    print(tigre_result)
    return (tigre_result,)


@app.cell
def _(
    CUDA_ENV,
    DRGS_DATA_URL,
    DRGS_DIR,
    DRGS_EXPECTED,
    LEAP_DIR,
    PSNR_TOL_DB,
    Path,
    TCNN_DIR,
    WORK,
    deviations,
    os,
    pip_install,
    run_4drgs,
    setup_check,
    sh,
    sys,
    tigre_result,
    time,
):
    # g2) 4DRGS on its own example case (README "Training", fast version).
    # A failure here is recorded and does not block saving the other results.
    _ = tigre_result
    drgs_result = {"ran": False}
    if run_4drgs.value:
        try:
            _env = dict(CUDA_ENV)  # same CUDA environment as the R2-Gaussian builds
            pip_install(["pyhocon", "PyMCubes", "pymeshlab"])
            def _install(path, label):
                """Plain build; on failure one retry with the same nvcc flag that
                fixed the R2-Gaussian rasterizer (GCC 14 no longer pulls in
                <cstdint>: 'std has no member uintptr_t', molab Oct 3). Flag only."""
                try:
                    pip_install(["--no-build-isolation", str(path)], env=_env)
                except RuntimeError:
                    _fl = "-include cfloat -include cstdint"
                    pip_install(["--no-build-isolation", str(path)],
                                env={**_env, "NVCC_APPEND_FLAGS": (_env.get("NVCC_APPEND_FLAGS", "") + " " + _fl).strip()})
                    deviations.append({"what": f"{label}: built with NVCC_APPEND_FLAGS='{_fl}'",
                                       "instead_of": "plain pip install from the 4DRGS README",
                                       "reason": "Plain build failed with GCC 14 (missing <cstdint>). Flag only; source untouched."})

            _install(TCNN_DIR / "bindings" / "torch", "tiny-cuda-nn")
            for _sub in [
                "diff-Xray-gaussian-rasterization-voxelization-sortfree",
                "diff-Xray-gaussian-rasterization-voxelization-legacy",
                "simple-knn",
            ]:
                _install(DRGS_DIR / "submodules" / _sub, f"4DRGS {_sub}")

            # LEAP, as in the 4DRGS README "Setup" (linux): sh ./etc/build.sh,
            # cp ./build/lib/libleapct.so ., python manual_install.py. CMake is
            # taken from pip; the build uses the same CUDA_HOME as all others.
            import importlib.util as _ilu
            import shutil as _sh
            import subprocess as _sp
            import sysconfig as _sc

            def _leap_ok():
                return _sp.run([sys.executable, "-c", "import leapctype"], cwd="/", capture_output=True).returncode == 0

            if not _leap_ok():
                if _ilu.find_spec("cmake") is None:
                    pip_install(["cmake"])
                # The molab image has no `make`, so CMake's default generator
                # (Unix Makefiles) fails ("unable to find a build program",
                # molab Oct 4). Ninja is installed; CMAKE_GENERATOR is read by
                # CMake >= 3.15, so build.sh runs unchanged. Compilers explicit.
                _benv = {**_env, "CUDACXX": str(Path(_env["CUDA_HOME"]) / "bin" / "nvcc"),
                         "CUDA_TOOLKIT_ROOT_DIR": _env["CUDA_HOME"],
                         "CMAKE_GENERATOR": "Ninja", "CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++",
                         "PATH": _sc.get_paths()["scripts"] + os.pathsep + os.environ.get("PATH", "")}
                sh(["bash", "./etc/build.sh"], cwd=LEAP_DIR, env=_benv)
                deviations.append({"what": "LEAP built with CMAKE_GENERATOR=Ninja, CC=gcc, CXX=g++, CUDAHOSTCXX=g++",
                                   "instead_of": "plain `sh ./etc/build.sh` (Unix Makefiles) from the 4DRGS README",
                                   "reason": "No `make` in the molab image. Environment only; LEAP source untouched."})
                _sh.copy(LEAP_DIR / "build" / "lib" / "libleapct.so", LEAP_DIR / "libleapct.so")
                # manual_install.py copies into site.getusersitepackages(), which
                # does not exist on molab (FileNotFoundError, Oct 4): create it.
                sh([sys.executable, "-c", "import os, site; os.makedirs(site.getusersitepackages(), exist_ok=True)"])
                sh([sys.executable, "manual_install.py"], cwd=LEAP_DIR)
                if not _leap_ok():
                    # manual_install.py copies to the user site-packages, which a
                    # venv does not read. Same files into the venv instead.
                    _dst = Path(_sc.get_paths()["purelib"])
                    for _f in ["libleapct.so", "src/leaptorch.py", "src/leapctype.py",
                               "src/leap_filter_sequence.py", "src/leap_preprocessing_algorithms.py"]:
                        _sh.copy(LEAP_DIR / _f, _dst)
                    deviations.append({"what": f"LEAP files copied to {_dst}",
                                       "instead_of": "manual_install.py target site.getusersitepackages()",
                                       "reason": "The uv venv does not read the user site-packages. Same file list; no source change."})
                if not _leap_ok():
                    raise RuntimeError("leapctype still not importable after the LEAP build")

            _case = WORK / "data_4drgs" / "case2"
            if not (_case / "transforms.json").exists():
                _case.parent.mkdir(parents=True, exist_ok=True)
                sh([sys.executable, "-m", "gdown", "--folder", DRGS_DATA_URL, "-O", _case])
            _out = WORK / "output_4drgs" / "case2_30v_10k"
            _common = [f"-m={_out}", f"-s={_case}", "--Nviews=30", "--iteration=10000", "--fdk_toolbox=tigre"]
            _t0 = time.time()
            sh([sys.executable, "train.py", *_common, "--ADC_until_iter=5000"], cwd=DRGS_DIR)
            _minutes = round((time.time() - _t0) / 60, 1)
            _log = sh([sys.executable, "test.py", *_common, "--render_2d"], cwd=DRGS_DIR)
            _m = setup_check.parse_4drgs_log(_log)
            _cmp = setup_check.compare_psnr(_m.get("psnr_eval"), DRGS_EXPECTED["psnr_eval"], PSNR_TOL_DB)
            drgs_result = {
                "ran": True,
                "case": "case2",
                "views": 30,
                "iterations": 10000,
                "train_minutes": _minutes,
                **_m,
                "expected": DRGS_EXPECTED,
                "psnr_diff_db": _cmp["diff_db"],
                "pass_within_1db": _cmp["pass"],
            }
            deviations.append(
                {
                    "what": "4DRGS run with --fdk_toolbox=tigre and test.py with --render_2d only",
                    "instead_of": "4DRGS default fdk_toolbox='leap'; README test command also has --render_fixview --VQR",
                    "reason": "TIGRE is already built here, so LEAP is not built. CLI flags only; source untouched. "
                    "Whether the README numbers used LEAP or TIGRE is UNVERIFIED.",
                }
            )
        except Exception as _e:  # recorded, not hidden
            drgs_result = {"ran": False, "error": f"{type(_e).__name__}: {_e}"}
    print("=== 4DRGS ===")
    print(drgs_result)
    return (drgs_result,)


@app.cell
def _(
    DRGS_DIR,
    FLOWSPLAT_DIR,
    R2G_DIR,
    TIGRE_DIR,
    build_info,
    deviations,
    drgs_result,
    env_info,
    json,
    mo,
    os,
    pins,
    r2g_result,
    run_4drgs,
    source_untouched,
    tigre_result,
    time,
):
    # h) Save everything to results/step0.1.json and show it for copy / paste.
    _untouched = {"r2_gaussian": source_untouched(R2G_DIR), "tigre": source_untouched(TIGRE_DIR)}
    if run_4drgs.value and (DRGS_DIR / ".git").exists():
        _untouched["4drgs"] = source_untouched(DRGS_DIR)
    results = {
        "step": "0.1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pins": pins,
        "env": env_info,
        "build": build_info,
        "r2_gaussian": r2g_result,
        "tigre": tigre_result,
        "4drgs": drgs_result,
        "deviations": deviations,
        "source_untouched": _untouched,
    }
    _only = os.environ.get("FLOWSPLAT_ONLY_4DRGS") == "1"
    _path = FLOWSPLAT_DIR / "results" / ("step0.1_4drgs.json" if _only else "step0.1.json")
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(json.dumps(results, indent=2, default=str))
    print("Saved", _path)
    print(json.dumps(results, indent=2, default=str))  # visible in script mode too
    mo.vstack(
        [
            mo.md(f"**Saved** `{_path}`. Paste the block below back to Claude to fill `reports/step0.1.md`."),
            mo.ui.code_editor(value=json.dumps(results, indent=2, default=str), language="json", disabled=True),
        ]
    )
    return (results,)


if __name__ == "__main__":
    app.run()
