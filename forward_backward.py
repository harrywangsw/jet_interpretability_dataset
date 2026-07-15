#!/usr/bin/env python3

from math import perm
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

REPO_ROOT = "/home/hiboy/jet_interpretability_dataset/GenerativeModelsOnPhaseSpace-main"
sys.path.insert(0, REPO_ROOT)

from omnilearn_lightning.model import PETLightning
from omnilearn_lightning.qspace import qs_to_ps, get_b_x_from_qs, sample_qspace, ps_to_qs
from omnilearn_lightning.diffusion import forward_process, reverse_step


# ----------------------------
# Small utilities
# ----------------------------

def get_model_attr(model, name, default=None):
    if hasattr(model, name):
        return getattr(model, name)
    if hasattr(model, "hparams") and hasattr(model.hparams, name):
        return getattr(model.hparams, name)
    return default

def fluff_in_q_space(Nmult, pspace_sample, seed=-1):
    nevents = pspace_sample.shape[0]
    nparticles = pspace_sample.shape[1]
    device = pspace_sample.device
    dtype = pspace_sample.dtype

    gen = torch.Generator(device=device)
    if seed < 0:
        gen.seed()
    else:
        gen.manual_seed(seed)

    boost_seed = int(torch.randint(0, 2**31, (1,), generator=gen).item())
    bs, xs = get_b_x_from_qs(
        sample_qspace(
            nevents=Nmult,
            nparticles=nparticles,
            seed=boost_seed,
            device=device,
            dtype=dtype,
        )
    )
    qs_list = []
    for i in range(Nmult):
        qs_list.append(ps_to_qs(pspace_sample, bs[None, i, :], xs[None, i]))
    qs = torch.stack(qs_list, dim=0).reshape(-1, *pspace_sample.shape[1:])
    return qs

def make_full_gammas_from_model(model, device, dtype=torch.float32):
    """
    Returns a full gamma schedule of length T.

    Handles both cases:
        model.gammas has length T
        model.gammas has length T - t_gaus, so we prepend gamma_gaus
    """
    if not hasattr(model, "gammas"):
        raise AttributeError("q-space model should have model.gammas, but it does not.")

    gammas = model.gammas.detach().to(device=device, dtype=dtype)

    T = int(get_model_attr(model, "t_steps", gammas.numel()))
    t_gaus = int(get_model_attr(model, "t_gaus", 0) or 0)
    gamma_gaus = float(get_model_attr(model, "gamma_gaus", 1e-4))

    if gammas.numel() == T:
        return gammas

    if gammas.numel() == T - t_gaus:
        gaus = torch.full(
            (t_gaus,),
            gamma_gaus,
            device=device,
            dtype=dtype,
        )
        return torch.cat([gaus, gammas], dim=0)

    raise ValueError(
        f"Cannot interpret gamma schedule: len(gammas)={gammas.numel()}, "
        f"T={T}, t_gaus={t_gaus}"
    )


def finite_check(name, x):
    if not torch.isfinite(x).all():
        print(f"[NONFINITE] {name}")
        print("shape:", tuple(x.shape))
        print("nan:", torch.isnan(x).any().item())
        print("inf:", torch.isinf(x).any().item())
        print("min:", x.nan_to_num().min().item())
        print("max:", x.nan_to_num().max().item())
        raise RuntimeError(f"{name} is non-finite")


# ----------------------------
# Forward-backward functions
# ----------------------------

@torch.no_grad()
def qspace_forward_to_qk_using_diffusion_py(model, Q0, k, gammas_full):
    """
    Q0 -> Q_k using diffusion.forward_process.

    k means number of forward steps.
        k=0 returns Q0 unchanged.
        k=1 applies gamma[0].
        k=10 applies gamma[0:10].
    """
    if k == 0:
        return Q0.clone()

    t_gaus = int(get_model_attr(model, "t_gaus", 0) or 0)

    # forward_process applies all steps in the passed gamma array.
    # Passing gammas_full[:k] gives exactly k forward steps.
    return forward_process(
        Q0,
        gammas_full[:k],
        t_gaus=t_gaus,
    )


@torch.no_grad()
def qspace_reverse_from_qk_using_diffusion_py(
    model,
    Qk,
    k,
    gammas_full,
):
    """
    Q_k -> Qhat_0 using diffusion.reverse_step.

    k means number of reverse steps.
        k=0 returns Qk unchanged.
        k=1 applies reverse step s=0.
        k=10 applies reverse steps s=9,...,0.
    """
    if k == 0:
        return Qk.clone()

    Q = Qk.clone()
    device = Q.device
    B = Q.shape[0]

    T = int(get_model_attr(model, "t_steps", len(gammas_full)))
    t_gaus = int(get_model_attr(model, "t_gaus", 0) or 0)
    loss_type = get_model_attr(model, "loss_type", "ism")
    predict_eps = loss_type == "dsm_eps"

    score_net = model.model

    for s in range(k - 1, -1, -1):
        gamma = gammas_full[s]

        # Current object is approximately Q_{s+1}, so use (s+1)/T.
        t_normalized = torch.full(
            (B,),
            (s + 1) / T,
            device=device,
            dtype=torch.float32,
        )

        noscore = s < t_gaus

        Q = reverse_step(
            Q,
            t_normalized,
            gamma,
            score_net,
            noscore=noscore,
            predict_eps=predict_eps,
        )

    return Q


@torch.no_grad()
def save_xhats_many_trajectories_qspace_using_diffusion_py(
    model,
    datapoints,
    k_values,
    n_traj,
    out_path,
    batch_size=1024,
    same_forward_noise=True,
    max_events=None,
    seed=-1,
    exact_k0=True,
):
    """
    Saves p-space reconstructions with shape:

        (n_t, n_traj, n_events, n_particles, 3)

    Workflow:
        P0 -> Q0 -> Qk -> Qhat0 -> Phat0

    Uses:
        diffusion.forward_process for forward noising
        diffusion.reverse_step for reverse denoising
    """

    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    diffusion_mode = get_model_attr(model, "diffusion_mode", None)
    loss_type = get_model_attr(model, "loss_type", None)
    t_gaus = int(get_model_attr(model, "t_gaus", 0) or 0)

    gammas_full = make_full_gammas_from_model(
        model,
        device=device,
        dtype=torch.float32,
    )

    T = len(gammas_full)

    print("diffusion_mode:", diffusion_mode)
    print("loss_type:", loss_type)
    print("t_gaus:", t_gaus)
    print("len(gammas_full):", T)
    print("gamma min/max:", gammas_full.min().item(), gammas_full.max().item())

    k_values = [int(k) for k in k_values]

    for k in k_values:
        if not (0 <= k <= T):
            raise ValueError(f"k={k} invalid; must satisfy 0 <= k <= T={T}")

    if max_events is not None:
        datapoints = datapoints[:max_events]

    # Keep full dataset on CPU. Only move batches to GPU.
    datapoints = datapoints.detach().cpu().float()

    n_t = len(k_values)
    n_events = datapoints.shape[0]
    event_shape = tuple(datapoints.shape[1:])

    # Keep xhats on CPU to avoid huge GPU memory use.
    xhats = torch.empty(
        (n_t, n_traj, n_events, *event_shape),
        dtype=torch.float32,
        device="cpu",
    )
    
    P_test = datapoints[:16]
    Q_test = fluff_in_q_space(1, P_test)
    P_roundtrip = qs_to_ps(Q_test).detach().cpu()
    rt_err = P_roundtrip - P_test
    print("roundtrip max abs error:", rt_err.abs().max().item())
    print("roundtrip mean abs error:", rt_err.abs().mean().item())

    for start in tqdm(range(0, n_events, batch_size), desc="batches"):
        end = min(start + batch_size, n_events)

        P0_batch_cpu = datapoints[start:end].float()
        B = P0_batch_cpu.shape[0]

        Q0_batch = fluff_in_q_space(1, P0_batch_cpu).to(device)
        finite_check("Q0_batch", Q0_batch)

        for i_t, k in enumerate(k_values):

            # Guarantee exact P0 for k=0 if desired.
            # This avoids any P -> Q -> P roundtrip mismatch.
            if k == 0 and exact_k0:
                for j in range(n_traj):
                    xhats[i_t, j, start:end] = P0_batch_cpu
                continue

            if same_forward_noise:
                Qk_shared = qspace_forward_to_qk_using_diffusion_py(
                    model,
                    Q0_batch,
                    k,
                    gammas_full,
                )
                finite_check(f"Qk_shared k={k}", Qk_shared)

            for j in range(n_traj):
                if same_forward_noise:
                    Qk = Qk_shared
                else:
                    Qk = qspace_forward_to_qk_using_diffusion_py(
                        model,
                        Q0_batch,
                        k,
                        gammas_full,
                    )
                    finite_check(f"Qk k={k} traj={j}", Qk)

                Qhat0 = qspace_reverse_from_qk_using_diffusion_py(
                    model,
                    Qk,
                    k,
                    gammas_full,
                )
                finite_check(f"Qhat0 k={k} traj={j}", Qhat0)

                Phat0 = qs_to_ps(Qhat0).detach().cpu()
                finite_check(f"Phat0 k={k} traj={j}", Phat0)

                xhats[i_t, j, start:end] = Phat0

                del Qhat0, Phat0

            if same_forward_noise:
                del Qk_shared

        del Q0_batch, P0_batch_cpu

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "xhats": xhats,
            "k_values": torch.tensor(k_values, dtype=torch.long),
            "n_traj": n_traj,
            "same_forward_noise": same_forward_noise,
            "mode": "qspace",
            "loss_type": loss_type,
            "exact_k0": exact_k0,
        },
        out_path,
    )

    print(f"\nsaved {out_path}")
    print(f"xhats shape: {tuple(xhats.shape)}")

    return xhats



# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datapoints_path = (
        "/home/hiboy/jet_interpretability_dataset/pythia_events/intermediate_by_nbranch/"
        "ee2qqbar_S=2000GeV_a=0p1365_N=2000000_NBranch=8_Nparticles=10_p3_normalized.pt"
    )

    ckpt_path = (
        "/home/hiboy/jet_interpretability_dataset/GenerativeModelsOnPhaseSpace-main/TrainedModels/"
        "_qspace_small_ism_N10_T5000_ee2qqbar_S=2000GeV_a=0p1365_N=2000000_NBranch=8_Nparticles=10_p3_normalized_harryw/"
        "v1_qspace_small_ism_N10_T5000_ee2qqbar_S=2000GeV_a=0p1365_N=2000000_NBranch=8_Nparticles=10_p3_normalized_harryw/"
        "checkpoints/last.ckpt"
    )

    datapoints = torch.load(
        datapoints_path,
        map_location="cpu",
        weights_only=True,
    ).float()

    print("datapoints:", datapoints.shape)

    model = PETLightning.load_from_checkpoint(
        ckpt_path,
        map_location=device,
    ).to(device).eval()

    N_events_total = datapoints.shape[0]

    k_max = 20
    num_timesteps = 20
    k_values = np.linspace(1, k_max, num_timesteps, dtype=int).tolist()

    out_path = (
        f"/home/hiboy/jet_interpretability_dataset/forward_backward_outputs/"
        f"N={N_events_total}_xhats_many_traj_qspace_diffusionpy_"
        f"k~0-{k_max}_timesteps={num_timesteps}.pt"
    )

    start_time = time.time()

    xhats = save_xhats_many_trajectories_qspace_using_diffusion_py(
        model=model,
        datapoints=datapoints,
        k_values=k_values,
        n_traj=20,
        out_path=out_path,
        batch_size=1000,
        same_forward_noise=True,
        max_events=1000,
        seed=123,
        exact_k0=True,
    )

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")