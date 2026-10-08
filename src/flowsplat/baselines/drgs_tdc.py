"""Baseline 1, molab part: time-density curves from a trained 4DRGS model.

Runs on: molab GPU, inside the 4DRGS environment (cwd = the 4DRGS repo,
which must be importable). 4DRGS source is not modified: the model is
loaded exactly as 4DRGS test.py does (Scene with load_iteration), and the
time-dependent opacity of every Gaussian is taken from its field
(gaussians._field(gaussians, t, recon_args)["final_opacity"], as in test.py
voxel_query_reconstruction). The density at a point x and time t is
sum_i o_i(t) exp(-1/2 (x - p_i)^T Sigma_i^-1 (x - p_i)) (4DRGS Sec 3.1,
Eq 1 and Eq 2), evaluated at our centerline points for the 133 frame
times t_j = j / 133 (4DRGS timestamp = uid / N_views).

Usage: python -m flowsplat.baselines.drgs_tdc --model M --data D --points P.npy
       --out OUT.npz [--iteration 10000 --nviews 60]
Points are in the 4DRGS world frame (= TIGRE world frame, mm; see
model.fourdrgs_volume_to_lumen_frame). Ref: 4DRGS code test.py, OURS.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iteration", type=int, default=10000)
    ap.add_argument("--nviews", type=int, default=60)
    ap.add_argument("--n_frames", type=int, default=133)
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from argparse import ArgumentParser
    from pyhocon import ConfigFactory

    from arguments import ModelParams, PipelineParams, get_combined_args
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from utils.general_utils import build_rotation

    # same argument handling as 4DRGS test.py
    sys.argv = ["test.py", f"-m={a.model}", f"-s={a.data}", f"--Nviews={a.nviews}", f"--iteration={a.iteration}"]
    parser = ArgumentParser()
    model = ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--Nviews", type=int, default=30)
    parser.add_argument("--field_conf", type=str, default="./arguments/nerf.conf")
    args = get_combined_args(parser)
    dataset = model.extract(args)
    field_conf = ConfigFactory.parse_file(args.field_conf)
    with torch.no_grad():
        scale_bound = np.asarray([dataset.scale_min, dataset.scale_max]) if dataset.use_scale_bound else None
        g = GaussianModel(field_conf)
        scene = Scene(dataset, g, scale_bound, load_iteration=args.iteration, shuffle=False)
        xyz = g.get_xyz
        rot = build_rotation(g.get_rotation)
        inv_s2 = 1.0 / g.get_scaling ** 2
        prec = rot @ torch.diag_embed(inv_s2) @ rot.transpose(1, 2)  # Sigma^-1
        pts = torch.tensor(np.load(a.points), dtype=torch.float32, device=xyz.device)
        w = []
        for k in range(0, len(pts), 64):  # spatial weights (points x Gaussians)
            d = pts[k:k + 64, None, :] - xyz[None]
            w.append(torch.exp(-0.5 * torch.einsum("pni,nij,pnj->pn", d, prec, d)))
        w = torch.cat(w)
        curves = []
        for j in range(a.n_frames):
            o = g._field(g, j / a.n_frames, scene.recon_args)["final_opacity"].reshape(-1)
            curves.append(w @ o)
        curves = torch.stack(curves, dim=1).cpu().numpy()
    np.savez(a.out, curves=curves.astype(np.float32), frame=np.arange(a.n_frames))
    print("tdc", a.out, curves.shape, float(curves.max()))


if __name__ == "__main__":
    main()
