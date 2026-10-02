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
