"""Runs on: M4 (CPU only). Tests for src/flowsplat/setup_check.py (Step 0.1)."""

import ast
from pathlib import Path

from flowsplat import setup_check as sc

ROOT = Path(__file__).resolve().parents[1]


def test_scrub_removes_token_and_base64_form():
    token = "ghp_exampleTOKEN123"
    env = sc.git_auth_env(token)
    text = f"fatal: {token} rejected; header {env['GIT_CONFIG_VALUE_0']}"
    cleaned = sc.scrub(text, token)
    assert token not in cleaned
    assert env["GIT_CONFIG_VALUE_0"].split()[-1] not in cleaned
    assert sc.scrub("nothing here", None) == "nothing here"


def test_git_auth_env_empty_without_token():
    assert sc.git_auth_env("") == {}
    assert sc.git_auth_env(None) == {}


def test_parse_r2g_eval_yml_reads_scalars_only():
    text = (
        "psnr_3d: 35.0912\n"
        "ssim_3d: 0.9312\n"
        "ssim_3d_x: 0.93\n"
        "psnr_2d_projs:\n- 50.1\n- 49.9\n"
    )
    out = sc.parse_r2g_eval_yml(text)
    assert out["psnr_3d"] == 35.0912
    assert out["ssim_3d"] == 0.9312
    assert "psnr_2d_projs" not in out


def test_parse_r2g_stdout_takes_highest_iteration():
    text = (
        "[ITER 10000] Evaluating: psnr3d 34.100, ssim3d 0.920, psnr2d 49.000, ssim2d 0.990\n"
        "noise\n"
        "[ITER 30000] Evaluating: psnr3d 35.200, ssim3d 0.931, psnr2d 50.500, ssim2d 0.995\n"
        "[ITER 1] Evaluating: psnr3d 20.000, ssim3d 0.500, psnr2d 30.000, ssim2d 0.800\n"
    )
    out = sc.parse_r2g_stdout(text)
    assert out == {
        "iteration": 30000,
        "psnr_3d": 35.2,
        "ssim_3d": 0.931,
        "psnr_2d": 50.5,
        "ssim_2d": 0.995,
    }
    assert sc.parse_r2g_stdout("no metrics") == {}


def test_parse_4drgs_log():
    text = "GauN: 41234\nPSNR train: 40.1 dB\nSSIM train: 0.95\nPSNR eval: 36.349 dB\nSSIM eval: 0.9021"
    out = sc.parse_4drgs_log(text)
    assert out["psnr_eval"] == 36.349
    assert out["ssim_eval"] == 0.9021
    assert out["n_gaussians"] == 41234


def test_compare_psnr_tolerance():
    assert sc.compare_psnr(35.6, 35.09)["pass"] is True
    assert sc.compare_psnr(33.9, 35.09)["pass"] is False
    assert sc.compare_psnr(None, 35.09)["pass"] is False
    assert sc.compare_psnr(34.59, 35.09)["diff_db"] == -0.5


def test_cube_center_ray_integral():
    assert sc.cube_center_ray_integral(256.0) == 256.0
    assert sc.cube_center_ray_integral(100.0, mu=0.02) == 2.0


def test_notebook_parses_and_has_no_hardcoded_token():
    src = (ROOT / "notebooks" / "step0_setup.py").read_text()
    ast.parse(src)
    assert 'kind="password"' in src
    assert 'os.environ.get("GH_TOKEN")' in src
    for prefix in ("ghp_", "github_pat_", "gho_"):
        assert prefix not in src


def test_notebook_pins_are_full_commit_hashes():
    src = (ROOT / "notebooks" / "step0_setup.py").read_text()
    tree = ast.parse(src)
    pins = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            name = getattr(node.targets[0], "id", "")
            if name.endswith("_COMMIT"):
                pins[name] = node.value.value
    assert {"R2G_COMMIT", "TIGRE_COMMIT", "DRGS_COMMIT", "TCNN_COMMIT"} <= set(pins)
    for value in pins.values():
        assert len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def test_find_and_prepare_pip_cuda_home(tmp_path):
    assert sc.find_pip_cuda_home([tmp_path]) is None
    home = tmp_path / "nvidia" / "cu13"
    (home / "bin").mkdir(parents=True)
    (home / "lib").mkdir()
    (home / "bin" / "nvcc").write_text("#!/bin/sh\n")
    (home / "lib" / "libcudart.so.13").write_text("")
    (home / "lib" / "libcudart_static.a").write_text("")
    assert sc.find_pip_cuda_home([tmp_path / "missing", tmp_path]) == home

    created = sc.prepare_pip_cuda_home(home)
    assert (home / "lib" / "libcudart.so").is_symlink()
    assert (home / "lib64" / "libcudart.so").exists()
    assert len(created) == 2
    assert (home / "bin" / "nvcc").stat().st_mode & 0o111
    assert sc.prepare_pip_cuda_home(home) == []  # second call changes nothing


def test_nvcc_release_and_same_cuda():
    text = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.3, V13.3.12\n"
    assert sc.nvcc_release(text) == "13.3"
    assert sc.nvcc_release(None) is None
    assert not sc.same_cuda("13.3", "13.0")
    assert sc.same_cuda("13.0", "13.0.88")
    assert sc.same_cuda("12.8", "12.8")
    assert not sc.same_cuda(None, "13.0")


def test_pip_cuda_specs_pins_torch_version():
    assert sc.pip_cuda_specs("13.0") == ["cuda-toolkit[nvcc,nvvm,crt,cccl,cudart]==13.0.*"]
    specs = sc.pip_cuda_specs("12.8")
    assert all(s.endswith("==12.8.*") and "-cu12" in s for s in specs)
    assert any(s.startswith("nvidia-cuda-cccl-cu12") for s in specs)


def test_find_include_dir_cuda13_cccl_layout(tmp_path):
    home = tmp_path / "nvidia" / "cu13"
    (home / "include" / "cccl" / "cub").mkdir(parents=True)
    (home / "include" / "cccl" / "cub" / "cub.cuh").write_text("")
    (home / "include" / "cccl" / "thrust").mkdir()
    assert sc.find_include_dir([tmp_path / "missing", home]) == home / "include" / "cccl"
    assert sc.find_include_dir([home], "nothere/x.h") is None


def test_merge_cuda12_prefixes(tmp_path):
    site = tmp_path / "site"
    for name, sub, f in [
        ("cuda_nvcc", "bin", "nvcc"),
        ("cuda_nvcc", "nvvm", "libdevice"),
        ("cuda_runtime", "include", "cuda_runtime.h"),
        ("cuda_runtime", "lib", "libcudart.so.12"),
        ("cuda_cccl", "include", "cub"),
        ("cu13", "bin", "nvcc"),
    ]:
        (site / "nvidia" / name / sub).mkdir(parents=True, exist_ok=True)
        (site / "nvidia" / name / sub / f).write_text("")
    prefixes = sc.find_pip_cuda12_prefixes([site])
    assert [p.name for p in prefixes] == ["cuda_cccl", "cuda_nvcc", "cuda_runtime"]
    home = sc.merge_cuda_prefixes(prefixes, tmp_path / "cuda-12.8")
    for rel in ["bin/nvcc", "nvvm/libdevice", "include/cuda_runtime.h", "include/cub", "lib/libcudart.so.12"]:
        assert (home / rel).exists(), rel
    links = sc.prepare_pip_cuda_home(home)
    assert (home / "lib" / "libcudart.so").exists() and (home / "lib64").exists()
    assert len(links) == 2
    sc.merge_cuda_prefixes(prefixes, home)  # idempotent


def test_find_include_dir_through_symlinked_cccl(tmp_path):
    real = tmp_path / "site" / "nvidia" / "cuda_cccl" / "include" / "cub"
    real.mkdir(parents=True)
    (real / "cub.cuh").write_text("")
    home = sc.merge_cuda_prefixes([real.parent.parent], tmp_path / "cuda-12.8")
    assert sc.find_include_dir([home]) == home / "include"


def test_notebook_cell_wiring():
    """Every marimo cell input is returned by exactly one cell, and every
    cross-cell name a cell reads is one of its inputs (marimo is not installed
    on the M4, so this stands in for its dependency check)."""
    tree = ast.parse((ROOT / "notebooks" / "step0_setup.py").read_text())
    cells = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and any("app.cell" in ast.unparse(d) for d in n.decorator_list)
    ]
    returned = {}
    for c in cells:
        ret = c.body[-1]
        if isinstance(ret, ast.Return) and ret.value is not None:
            elts = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
            for e in elts:
                assert e.id not in returned, f"{e.id} returned by two cells"
                returned[e.id] = c.lineno
    for c in cells:
        params = {a.arg for a in c.args.args}
        assert params <= set(returned), f"cell at line {c.lineno}: undefined inputs {params - set(returned)}"
        stored = {n.id for n in ast.walk(c) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        stored |= {
            (a.asname or a.name).split(".")[0]
            for n in ast.walk(c) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names
        }
        stored |= {n.name for n in ast.walk(c) if isinstance(n, ast.FunctionDef) and n is not c}
        loaded = {n.id for n in ast.walk(c) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = {x for x in loaded & set(returned) if x not in params and x not in stored}
        assert not missing, f"cell at line {c.lineno} reads {missing} without taking them as inputs"


def test_first_compiler_error():
    log = (
        "Building wheel for simple_knn\n"
        "  error: subprocess-exited-with-error\n"
        "/x/simple_knn.cu:18:10: fatal error: cub/cub.cuh: No such file or directory\n"
        "/x/other.cu(42): error: identifier \"foo\" is undefined\n"
        "ERROR: Failed building wheel for simple_knn\n"
    )
    assert sc.first_compiler_error(log) == "/x/simple_knn.cu:18:10: fatal error: cub/cub.cuh: No such file or directory"
    assert sc.first_compiler_error("ModuleNotFoundError: No module named 'torch'\n").startswith("ModuleNotFoundError")
    assert sc.first_compiler_error("nvcc fatal   : Unsupported gpu architecture 'compute_120'").startswith("nvcc fatal")
    assert sc.first_compiler_error("all good") is None
    assert sc.first_compiler_error(None) is None
