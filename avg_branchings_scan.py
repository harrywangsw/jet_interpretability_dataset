#!/usr/bin/env python3
import argparse
import re
import matplotlib
matplotlib.use("Agg")  # safe for Slurm / headless WSL runs
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import pandas as pd


PARTON_IDS = torch.tensor([1, 2, 3, 4, 5, 6, 21])


def parse_scan_labels(path: Path):
    """
    Expected filename:
        ee2qqbar_S=91p1876GeV_a=0p118.pt
        ee2qqbar_S=10000GeV_a=0p088.pt
    """
    m = re.search(r"S=([0-9p.]+)GeV_a=([0-9p.]+)", path.stem)
    if not m:
        raise ValueError(f"Could not parse eCM/alphaS from filename: {path.name}")

    eCM_GeV = float(m.group(1).replace("p", "."))
    alphaS = float(m.group(2).replace("p", "."))
    return eCM_GeV, alphaS


def count_final_qcd_partons_from_dict(data):
    """
    Your parser produces a dict like:
        data["pid"]    shape: [n_events, max_particles]
        data["status"] shape: [n_events, max_particles]
        data["mask"]   shape: [n_events, max_particles]

    We count final-state q/g partons for each event.
    """

    pid = data["pid"]
    status = data["status"]
    mask = data["mask"]

    # Make sure tensors are on CPU and have simple types.
    pid = torch.as_tensor(pid).cpu()
    status = torch.as_tensor(status).cpu()
    mask = torch.as_tensor(mask).cpu().bool()

    abs_pid = pid.abs()

    is_parton = torch.zeros_like(mask, dtype=torch.bool)
    for qid in PARTON_IDS:
        is_parton |= (abs_pid == qid)

    is_final = status > 0

    final_qcd_mask = mask & is_final & is_parton

    # Count final q/g partons event-by-event.
    final_counts = final_qcd_mask.sum(dim=1)

    # For clean e+e- -> q qbar with no ISR and no hadronization:
    # branchings = final leaves - initial q/qbar pair.
    branching_counts = torch.clamp(final_counts - 2, min=0)

    return final_counts, branching_counts


def analyze_file(path: Path):
    data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict-of-tensors .pt file, got {type(data)}")

    required = ["pid", "status", "mask"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path.name} is missing keys {missing}. Available keys: {list(data.keys())}")

    final_counts, branching_counts = count_final_qcd_partons_from_dict(data)

    eCM_GeV, alphaS = parse_scan_labels(path)

    return {
        "file": path.name,
        "eCM_GeV": eCM_GeV,
        "alphaS": alphaS,
        "n_events": int(final_counts.numel()),
        "avg_final_qcd_partons": float(final_counts.float().mean()),
        "avg_qcd_branchings": float(branching_counts.float().mean()),
        "min_final_qcd_partons": int(final_counts.min()),
        "max_final_qcd_partons": int(final_counts.max()),
        "min_qcd_branchings": int(branching_counts.min()),
        "max_qcd_branchings": int(branching_counts.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt-dir",
        type=Path,
        default=Path("/home/hiboy/jet_interpretability_dataset/pythia_events"),
        help="Directory containing scan .pt files.",
    )
    parser.add_argument(
        "--pattern",
        default="ee2qqbar_S=*GeV_a=*.pt",
        help="Glob pattern for scan .pt files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    args = parser.parse_args()

    paths = sorted(args.pt_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No .pt files matched {args.pt_dir / args.pattern}")

    rows = []
    for path in paths:
        print(f"Reading {path}")
        rows.append(analyze_file(path))

    df = pd.DataFrame(rows)

    # Sort rows by increasing energy, but decreasing alphaS.
    df = df.sort_values(["eCM_GeV", "alphaS"], ascending=[True, False]).reset_index(drop=True)

    print("\n=== Average final QCD partons and branchings ===")
    print(df.to_string(index=False))

    # Put larger alphaS values on the left.
    alpha_cols = sorted(df["alphaS"].unique(), reverse=True)
    eCM_rows = sorted(df["eCM_GeV"].unique())

    print("\n=== Pivot: average QCD branchings ===")
    pivot = df.pivot(index="eCM_GeV", columns="alphaS", values="avg_qcd_branchings")
    pivot = pivot.reindex(index=eCM_rows, columns=alpha_cols)
    print(pivot.to_string())

    print("\n=== Pivot: average final QCD partons ===")
    pivot_final = df.pivot(index="eCM_GeV", columns="alphaS", values="avg_final_qcd_partons")
    pivot_final = pivot_final.reindex(index=eCM_rows, columns=alpha_cols)
    print(pivot_final.to_string())

    if args.out is None:
        args.out = args.pt_dir / "avg_branchings_from_final_particles_scan.csv"

    df.to_csv(args.out, index=False)
    print(f"\nSaved CSV to {args.out}")
    
    plot_dir = args.pt_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: average branchings vs eCM, one curve per alphaS.
    plt.figure(figsize=(8, 5))

    for alphaS in alpha_cols:
        y = pivot[alphaS]
        plt.plot(pivot.index, y, marker="o", label=fr"$\alpha_s={alphaS:g}$")

    plt.xscale("log")
    plt.xlabel(r"$e_{\rm CM}$ [GeV]")
    plt.ylabel("Average QCD branchings per event")
    plt.title(r"Average branchings vs $e_{\rm CM}$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    out1 = plot_dir / "avg_branchings_vs_eCM.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    print(f"\nSaved plot: {out1}")


    # Plot 2: average branchings vs alphaS, one curve per eCM.
    plt.figure(figsize=(8, 5))

    for eCM_GeV in eCM_rows:
        sub = df[df["eCM_GeV"] == eCM_GeV].sort_values("alphaS", ascending=False)
        plt.plot(
            sub["alphaS"],
            sub["avg_qcd_branchings"],
            marker="o",
            label=fr"$e_{{\rm CM}}={eCM_GeV:g}$ GeV",
        )

    plt.xlabel(r"$\alpha_s$")
    plt.ylabel("Average QCD branchings per event")
    plt.title(r"Average branchings vs $\alpha_s$")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # larger alphaS on the left
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    out2 = plot_dir / "avg_branchings_vs_alphaS.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    print(f"Saved plot: {out2}")


if __name__ == "__main__":
    main()