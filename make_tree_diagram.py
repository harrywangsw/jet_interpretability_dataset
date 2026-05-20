from __future__ import annotations
#!/usr/bin/env python3
"""
Load a parsed PYTHIA .pt file and draw an event-record process tree/DAG.

This expects the .pt format produced by parse_pythia_event_log.py, containing at
least:

    pid, status, mothers, p4, mask

and optionally:

    no, names, daughters, colors, mass

The output is always a Graphviz DOT file. If the Graphviz command-line program
`dot` is installed, the script can also render PNG/PDF/SVG.

Examples
--------
    python plot_pythia_process_tree.py events.pt --event 0 -o event0.dot

Render to PNG:

    python plot_pythia_process_tree.py events.pt --event 0 -o event0.png --format png

Show only hard-process-ish particles and descendants:

    python plot_pythia_process_tree.py events.pt --event 0 --hard-descendants -o hard_event0.png --format png

Final-state particles can make the graph huge, so for showered events you may
want --max-nodes 150 or --hard-descendants.
"""
"""
make mymain01 && ./mymain01 > mymain01.log && /home/hiboy/miniforge3/bin/python /home/hiboy/jet_interpretability_dataset/python/parse_pythia_log.py /home/hiboy/jet_interpretability_dataset/pythia8317/examples/mymain01.log -o /home/hiboy/jet_interpretability_dataset/pythia_events/mymain01.pt --randomize-branching-daughter-nos && /home/hiboy/miniforge3/bin/python /home/hiboy/jet_interpretability_dataset/python/make_tree_diagram.py /home/hiboy/jet_interpretability_dataset/pythia_events/mymain01.pt --events-output-dir /home/hiboy/jet_interpretability_dataset/pythia_events/tree_diagrams --hard-descendants --all-events --num-events 100000 --format dot --final-state-output /home/hiboy/jet_interpretability_dataset/pythia_events/final_state_particles_by_event.txt --label-colors
"""



import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import torch


# Small PDG-name fallback if the saved file does not contain names.
PDG_NAMES = {
    1: "d", -1: "dbar",
    2: "u", -2: "ubar",
    3: "s", -3: "sbar",
    4: "c", -4: "cbar",
    5: "b", -5: "bbar",
    6: "t", -6: "tbar",
    11: "e-", -11: "e+",
    12: "nu_e", -12: "nu_ebar",
    13: "mu-", -13: "mu+",
    14: "nu_mu", -14: "nu_mubar",
    15: "tau-", -15: "tau+",
    16: "nu_tau", -16: "nu_taubar",
    21: "g",
    22: "gamma",
    23: "Z0",
    24: "W+", -24: "W-",
    25: "h0",
    2212: "p+", -2212: "pbar-",
}


def dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def get_event_field(data: dict, key: str, event_index: int):
    x = data[key]
    if torch.is_tensor(x):
        return x[event_index]
    return x[event_index]


def tensor_item(x):
    if torch.is_tensor(x):
        return x.item()
    return x


def particle_name(pid: int, names: Optional[list[str]], j: int) -> str:
    if names is not None and j < len(names) and names[j]:
        return str(names[j])
    return PDG_NAMES.get(pid, str(pid))


def p4_kinematics(px: float, py: float, pz: float, energy: float) -> tuple[float, float, float]:
    """Return pT, eta, phi. eta is inf at exactly zero transverse momentum."""
    pt = math.hypot(px, py)
    phi = math.atan2(py, px)
    p = math.sqrt(px * px + py * py + pz * pz)
    if pt == 0.0:
        eta = math.copysign(float("inf"), pz) if pz != 0 else 0.0
    else:
        # eta = asinh(pz / pT), numerically stable.
        eta = math.asinh(pz / pt)
    return pt, eta, phi


def fmt_float(x: float, ndigits: int = 3) -> str:
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    if math.isnan(x):
        return "nan"
    return f"{x:.{ndigits}g}"


def collect_event_particles(data: dict, event_index: int) -> list[dict]:
    mask = get_event_field(data, "mask", event_index).bool()
    pid = get_event_field(data, "pid" if "pid" in data else "id", event_index).long()
    status = get_event_field(data, "status", event_index).long()
    mothers = get_event_field(data, "mothers", event_index).long()
    p4 = get_event_field(data, "p4", event_index).detach().cpu()

    no_field = get_event_field(data, "no", event_index).long() if "no" in data else None
    mass = get_event_field(data, "mass", event_index) if "mass" in data else None
    scale = get_event_field(data, "scale", event_index) if "scale" in data else None
    colors = get_event_field(data, "colors", event_index).long() if "colors" in data else None
    names = get_event_field(data, "names", event_index) if "names" in data else None

    particles: list[dict] = []
    for j in range(mask.numel()):
        if not bool(mask[j].item()):
            continue

        no = int(no_field[j].item()) if no_field is not None else j + 1
        this_pid = int(pid[j].item())
        p4_vec = p4[j].to(dtype=torch.float64).clone()
        px, py, pz, energy = [float(v) for v in p4_vec.tolist()]
        pt, eta, phi = p4_kinematics(px, py, pz, energy)

        particle = {
            "array_index": j,
            "no": no,
            "pid": this_pid,
            "name": particle_name(this_pid, names, j),
            "status": int(status[j].item()),
            "mother1": int(mothers[j, 0].item()),
            "mother2": int(mothers[j, 1].item()),
            "px": px,
            "py": py,
            "pz": pz,
            "energy": energy,
            # Store the four-momentum as a torch tensor in the convention
            # (px, py, pz, E). The scalar fields above are kept only for
            # labels/backwards compatibility with older helper code.
            "p4": p4_vec,
            "pt": pt,
            "eta": eta,
            "phi": phi,
        }
        if mass is not None:
            particle["mass"] = float(tensor_item(mass[j]))
        if scale is not None:
            particle["scale"] = float(tensor_item(scale[j]))
        else:
            particle["scale"] = float("nan")

        if colors is not None:
            particle["color1"] = int(colors[j, 0].item())
            particle["color2"] = int(colors[j, 1].item())
        particles.append(particle)
        # print(particle["pid"])
    return particles


def build_edges(particles: list[dict]) -> list[tuple[int, int]]:
    """
    Build mother -> child edges using PYTHIA mother columns.

    PYTHIA mother entries are event-record line numbers, usually 1-based. A row
    with mother1=mother2=m gives one edge m -> child. A row with two distinct
    mothers gives two edges.
    """
    existing = {p["no"] for p in particles}
    edges: set[tuple[int, int]] = set()

    for p in particles:
        child = p["no"]
        m1 = p["mother1"]
        m2 = p["mother2"]
        for mother in sorted({m1, m2}):
            if mother > 0 and mother in existing and mother != child:
                edges.add((mother, child))

    return sorted(edges)



def p4_tensor(p: dict) -> torch.Tensor:
    """Return the four-momentum tensor in the convention (px, py, pz, E).

    New parser output stores this as p["p4"]. The fallback keeps the plotting
    script compatible with older particle dictionaries that only had scalar
    px/py/pz/energy fields.
    """
    if "p4" in p and torch.is_tensor(p["p4"]):
        return p["p4"].to(dtype=torch.float64)
    return torch.tensor(
        [p["px"], p["py"], p["pz"], p["energy"]],
        dtype=torch.float64,
    )


def add_p4(*vecs: torch.Tensor) -> torch.Tensor:
    """Add four-momentum tensors in the convention (px, py, pz, E)."""
    if not vecs:
        return torch.zeros(4, dtype=torch.float64)
    return torch.stack([v.to(dtype=torch.float64) for v in vecs], dim=0).sum(dim=0)


def sub_p4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Subtract four-momentum tensors in the convention (px, py, pz, E)."""
    return a.to(dtype=torch.float64) - b.to(dtype=torch.float64)


def p4_norm(v: torch.Tensor) -> float:
    """Euclidean norm used only as a numerical residual for momentum matching."""
    return float(torch.linalg.vector_norm(v.to(dtype=torch.float64)).item())


def inv_mass2(v: torch.Tensor) -> float:
    """Return E^2 - |p|^2 for a four-momentum tensor (px, py, pz, E)."""
    v = v.to(dtype=torch.float64)
    return float((v[3] * v[3] - torch.dot(v[:3], v[:3])).item())


def children_by_mother(edges: list[tuple[int, int]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for mother, child in edges:
        children.setdefault(mother, []).append(child)
    return children



def boost_to_rest_frame(p4: torch.Tensor, total_p4: torch.Tensor) -> torch.Tensor:
    """Boost p4 into the rest frame of total_p4.

    Four-vectors use the convention (px, py, pz, E). The returned vector uses
    the same convention. If total_p4 is already at rest, this returns p4.
    """
    p4 = p4.to(dtype=torch.float64)
    total_p4 = total_p4.to(dtype=torch.float64)
    total_e = total_p4[3]
    if abs(float(total_e)) < 1e-15:
        return p4.clone()

    beta = total_p4[:3] / total_e
    beta2 = torch.dot(beta, beta)
    beta2_float = float(beta2.item())
    if beta2_float < 1e-30:
        return p4.clone()
    if beta2_float >= 1.0:
        # This should not happen for a physical timelike dipole system. Return
        # the original vector rather than crashing, so the caller can skip it.
        return p4.clone()

    gamma = 1.0 / math.sqrt(1.0 - beta2_float)
    p_vec = p4[:3]
    e = p4[3]
    beta_dot_p = torch.dot(beta, p_vec)

    e_prime = gamma * (e - beta_dot_p)
    p_prime = p_vec + (((gamma - 1.0) * beta_dot_p / beta2) - gamma * e) * beta
    return torch.cat([p_prime, e_prime.reshape(1)])


def choose_z_daughter(parent: dict, child1: dict, child2: dict) -> tuple[dict, dict]:
    """Choose which daughter defines z.

    For q -> q g, use the daughter with the same PDG id as the parent as b, so
    z is the energy fraction retained by the quark/antiquark. For ambiguous
    gluon splittings, fall back to event-record order.
    """
    parent_pid = parent.get("pid")
    candidates = [child1, child2]

    if parent_pid != 21:
        same_flavour = [c for c in candidates if c.get("pid") == parent_pid]
        if same_flavour:
            b = same_flavour[0]
            c = child2 if b is child1 else child1
            return b, c

    return child1, child2


def reconstructed_z_for_branch(by_no: dict[int, dict], a: int, bc: list[int], rp: int) -> tuple[float, float, int, int]:
    """Compute a reconstructed shower z for a -> b,c with recoiler copy rp.

    The z returned here is not PYTHIA's hidden internal random number. It is a
    kinematic reconstruction using the energy fraction of daughter b in the
    rest frame of the post-branching dipole (b+c+r'):

        z = E_b^* / (E_b^* + E_c^*).

    The second return value is the same ratio in the lab frame, useful for
    debugging. The last two return values are the selected b and c line numbers.
    """
    if len(bc) < 2:
        raise ValueError("Need at least two daughter candidates to reconstruct z.")

    child1 = by_no[bc[0]]
    child2 = by_no[bc[1]]
    b_part, c_part = choose_z_daughter(by_no[a], child1, child2)
    b = int(b_part["no"])
    c = int(c_part["no"])

    pb = p4_tensor(by_no[b])
    pc = p4_tensor(by_no[c])
    prp = p4_tensor(by_no[rp])
    dipole_total = add_p4(pb, pc, prp)

    pb_star = boost_to_rest_frame(pb, dipole_total)
    pc_star = boost_to_rest_frame(pc, dipole_total)

    denom_star = float((pb_star[3] + pc_star[3]).item())
    denom_lab = float((pb[3] + pc[3]).item())
    if denom_star <= 0 or denom_lab <= 0:
        return float("nan"), float("nan"), b, c

    z_star = float((pb_star[3] / (pb_star[3] + pc_star[3])).item())
    z_lab = float((pb[3] / (pb[3] + pc[3])).item())
    return z_star, z_lab, b, c


def infer_fsr_branchings(
    particles: list[dict],
    edges: list[tuple[int, int]],
    selected: Optional[set[int]] = None,
    *,
    scale_tol: float = 2e-3,
    residual_tol: float = 2e-2,
) -> list[dict]:
    """Infer local FSR branchings (a,r)->(b,c,r') from a PYTHIA event record.

    PYTHIA does not store an explicit recoiler column or the sampled z in the
    standard event record. This routine reconstructs likely FSR branchings from
    the event history pattern

        radiator a -> daughters b,c
        recoiler r -> copied recoiler r'

    by matching daughter/recoiler-copy scales and requiring approximate
    four-momentum conservation.
    """
    by_no = {p["no"]: p for p in particles}
    children = children_by_mother(edges)
    if selected is None:
        selected = set(by_no)

    branch_candidates: list[tuple[int, list[int]]] = []
    for a, ch in children.items():
        ch_sel = [c for c in ch if c in selected]
        if a in selected and len(ch_sel) >= 2:
            branch_candidates.append((a, ch_sel))

    branchings: list[dict] = []
    seen: set[tuple[int, tuple[int, ...], int, int]] = set()

    for r, rp in edges:
        if r not in selected or rp not in selected:
            continue
        r_part = by_no[r]
        rp_part = by_no[rp]

        # Recoiler copies usually have mother1=mother2=r and the same species.
        if not (rp_part.get("mother1") == r and rp_part.get("mother2") == r):
            continue
        if rp_part.get("pid") != r_part.get("pid"):
            continue
        if not math.isfinite(rp_part.get("scale", float("nan"))) or rp_part["scale"] <= 0:
            continue

        best: Optional[dict] = None
        for a, ch in branch_candidates:
            if a == r:
                continue

            same_scale_children = [
                c for c in ch
                if math.isfinite(by_no[c].get("scale", float("nan")))
                and abs(by_no[c]["scale"] - rp_part["scale"]) <= scale_tol
            ]
            if len(same_scale_children) < 2:
                continue

            bc_raw = sorted(same_scale_children)[:2]
            lhs = add_p4(p4_tensor(by_no[a]), p4_tensor(r_part))
            rhs = add_p4(*(p4_tensor(by_no[c]) for c in bc_raw), p4_tensor(rp_part))
            residual = p4_norm(sub_p4(lhs, rhs))

            pbpc = add_p4(*(p4_tensor(by_no[c]) for c in bc_raw))
            ma = float(by_no[a].get("mass", 0.0))
            q2 = inv_mass2(pbpc) - ma * ma
            q = math.sqrt(max(q2, 0.0))
            z, z_lab, b, c = reconstructed_z_for_branch(by_no, a, bc_raw, rp)

            record = {
                "radiator": int(a),
                "recoiler_before": int(r),
                "recoiler_after": int(rp),
                "daughter_b": int(b),
                "daughter_c": int(c),
                "daughters_raw": [int(x) for x in bc_raw],
                "radiator_pid": int(by_no[a].get("pid", 0)),
                "daughter_b_pid": int(by_no[b].get("pid", 0)),
                "daughter_c_pid": int(by_no[c].get("pid", 0)),
                "recoiler_pid": int(by_no[r].get("pid", 0)),
                "scale": float(rp_part["scale"]),
                "Q": float(q),
                "Q2": float(q2),
                "z": float(z),
                "z_lab": float(z_lab),
                "residual": float(residual),
            }

            if best is None or residual < best["residual"]:
                best = record

        if best is not None and best["residual"] <= residual_tol:
            key = (
                best["radiator"],
                tuple(best["daughters_raw"]),
                best["recoiler_before"],
                best["recoiler_after"],
            )
            if key not in seen:
                seen.add(key)
                branchings.append(best)
            # print(f"Branching kinematics for {record['radiator']}→", record['daughter_b'], record['daughter_c'])
            # print("check momentum conservation: ", residual)
            # marsq = inv_mass2(add_p4(p4_tensor(by_no[record['radiator']]), p4_tensor(by_no[record['recoiler_before']])))
            # # print("m_arsq: ", marsq)
            # print("ma'^2: ", inv_mass2(add_p4(p4_tensor(by_no[record['daughter_b']]), p4_tensor(by_no[record['daughter_c']]))))
            # print("ma^2: ", inv_mass2(p4_tensor(by_no[record['radiator']])))
            # # print("scale: ", by_no[record['daughter_c']]["scale"])
            # # Q2 = inv_mass2(add_p4(p4_tensor(by_no[record['daughter_b']]), p4_tensor(by_no[record['daughter_c']]))) - inv_mass2(p4_tensor(by_no[record['radiator']]))
            # print("Q2 from virtuality: ", best["Q2"])
            # print("z(1-z)Q^2", best["z"] * (1 - best["z"]) * best["Q2"])
            # print("p_evol^2: ", best["scale"]**2)
            # Q2 = record["scale"]**2/(record["z"] * (1 - record["z"]))
            # print("Q2 from p_evol and z: ", Q2)
            # print("RHS of eq 103 in PYTHIA manuel: ", add_p4(p4_tensor(by_no[record['radiator']]), best["Q2"]/marsq*p4_tensor(by_no[record['recoiler_before']])))
            # print("LHS of eq 103 in PYTHIA manuel: ", add_p4(p4_tensor(by_no[record['daughter_b']]), p4_tensor(by_no[record['daughter_c']])))

    return branchings


def infer_recoil_q_edge_labels(
    particles: list[dict],
    edges: list[tuple[int, int]],
    selected: set[int],
    *,
    scale_tol: float = 2e-3,
    residual_tol: float = 2e-2,
) -> dict[tuple[int, int], str]:
    """Infer FSR recoiler-copy edges r -> r' and label them with Q and z."""
    labels: dict[tuple[int, int], str] = {}
    for br in infer_fsr_branchings(
        particles,
        edges,
        selected,
        scale_tol=scale_tol,
        residual_tol=residual_tol,
    ):
        r = br["recoiler_before"]
        rp = br["recoiler_after"]
        a = br["radiator"]
        b = br["daughter_b"]
        c = br["daughter_c"]
        labels[(r, rp)] = f"Q={br['Q']:.3g}, z={br['z']:.3g}, recoil for {a}→{b},{c}"
    return labels


def strip_particle_for_saving(p: dict) -> dict:
    """Make a lightweight, torch.save-friendly particle record."""
    keep = {
        "array_index", "no", "pid", "name", "status", "mother1", "mother2",
        "px", "py", "pz", "energy", "p4", "pt", "eta", "phi", "mass",
        "scale", "color1", "color2",
    }
    return {k: v for k, v in p.items() if k in keep}




def safe_file_component(s: str) -> str:
    """Return a filesystem-safe component for output filenames."""
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "unknown"


def pid_label(pid: int) -> str:
    """Short label used in branching names."""
    return PDG_NAMES.get(pid, str(pid)).replace("bar", "bar")


def branching_class_from_record(br: dict) -> str:
    """Return a coarse physics branching class.

    The returned labels are intended for grouping z distributions by splitting
    kernel, not by exact quark flavour. Charge-conjugate quark splittings are
    combined.
    """
    a = int(br.get("radiator_pid", 0))
    b = int(br.get("daughter_b_pid", 0))
    c = int(br.get("daughter_c_pid", 0))

    if abs(a) in {1, 2, 3, 4, 5, 6}:
        # q -> q g, with q and qbar both grouped together.
        if b == a and c == 21:
            return "q_to_qg"
        if c == a and b == 21:
            return "q_to_qg"
        return f"q_to_{safe_file_component(pid_label(b))}_{safe_file_component(pid_label(c))}"

    if a == 21:
        if b == 21 and c == 21:
            return "g_to_gg"
        if b == -c and abs(b) in {1, 2, 3, 4, 5, 6}:
            return "g_to_qqbar"
        return f"g_to_{safe_file_component(pid_label(b))}_{safe_file_component(pid_label(c))}"

    return f"{safe_file_component(pid_label(a))}_to_{safe_file_component(pid_label(b))}_{safe_file_component(pid_label(c))}"


def branching_exact_from_record(br: dict) -> str:
    """Return an exact signed-PDG branching label, e.g. -1_to_-1_21."""
    return (
        f"{int(br.get('radiator_pid', 0))}"
        f"_to_{int(br.get('daughter_b_pid', 0))}"
        f"_{int(br.get('daughter_c_pid', 0))}"
    )


def branching_key_from_record(br: dict, split_by: str) -> str:
    if split_by == "class":
        return branching_class_from_record(br)
    if split_by == "exact":
        return branching_exact_from_record(br)
    raise ValueError(f"Unknown split_by={split_by!r}. Expected 'class' or 'exact'.")


def save_z_values_by_branching(
    branchings_all: list[dict],
    *,
    output_dir: Path,
    split_by: str = "class",
    bins: int = 50,
) -> dict[str, dict]:
    """Save and plot z values separately for each branching type.

    For each group this writes three files:
      - <branching>.pt: torch-save dictionary with tensor z_values and records
      - <branching>.csv: lightweight table of event, line numbers, Q, z, etc.
      - <branching>_z_distribution.png: histogram of z for that branching type

    It also writes z_by_branching_overlay.png, an optional quick-look overlay of
    all non-empty per-branching histograms.

    Returns metadata about the files written, suitable for inclusion in the
    combined analysis output.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = {}
    for br in branchings_all:
        key = branching_key_from_record(br, split_by)
        groups.setdefault(key, []).append(br)

    written: dict[str, dict] = {}
    overlay_data: dict[str, list[float]] = {}

    for key, records in sorted(groups.items()):
        finite_records = [
            br for br in records
            if math.isfinite(float(br.get("z", float("nan")))) and 0.0 <= float(br["z"]) <= 1.0
        ]
        z_list = [float(br["z"]) for br in finite_records]
        z_tensor = torch.tensor(z_list, dtype=torch.float64)
        key_safe = safe_file_component(key)
        pt_path = output_dir / f"{key_safe}.pt"
        csv_path = output_dir / f"{key_safe}.csv"
        plot_path = output_dir / f"{key_safe}_z_distribution.png"

        torch.save(
            {
                "branching_key": key,
                "split_by": split_by,
                "z_values": z_tensor,
                "branchings": finite_records,
                "all_branchings_for_key": records,
                "metadata": {
                    "num_branchings_for_key": len(records),
                    "num_finite_z_values": len(finite_records),
                    "z_definition": "E_b/(E_b+E_c) in rest frame of post-branching dipole b+c+r'",
                    "plot_path": str(plot_path),
                },
            },
            pt_path,
        )

        csv_fields = [
            "event",
            "radiator",
            "recoiler_before",
            "recoiler_after",
            "daughter_b",
            "daughter_c",
            "radiator_pid",
            "daughter_b_pid",
            "daughter_c_pid",
            "recoiler_pid",
            "scale",
            "Q",
            "Q2",
            "z",
            "z_lab",
            "residual",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for br in finite_records:
                writer.writerow({field: br.get(field, "") for field in csv_fields})

        if z_list:
            overlay_data[key] = z_list
            plt.figure(figsize=(6.0, 4.0))
            plt.hist(z_list, bins=bins, range=(0.0, 1.0), density=True, histtype="step", linewidth=1.5)
            plt.xlabel(r"reconstructed $z$")
            plt.ylabel("density")
            plt.title(f"{key}: reconstructed z ({len(z_list)} branchings)")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
            plot_path_str: Optional[str] = str(plot_path)
        else:
            plot_path_str = None

        written[key] = {
            "pt_path": str(pt_path),
            "csv_path": str(csv_path),
            "plot_path": plot_path_str,
            "num_branchings": len(records),
            "num_finite_z_values": len(finite_records),
        }

    if overlay_data:
        overlay_path = output_dir / "z_by_branching_overlay.png"
        plt.figure(figsize=(6.5, 4.5))
        for key, values in sorted(overlay_data.items()):
            plt.hist(values, bins=bins, range=(0.0, 1.0), density=True, histtype="step", linewidth=1.5, label=f"{key} ({len(values)})")
        plt.xlabel(r"reconstructed $z$")
        plt.ylabel("density")
        plt.title("Reconstructed z by branching type")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=200)
        plt.close()
        written["__overlay__"] = {
            "plot_path": str(overlay_path),
            "num_branching_types": len(overlay_data),
        }

    return written

def analyze_z_distribution(
    data: dict,
    *,
    max_events: Optional[int] = None,
    z_output: Path = Path("z_distribution.png"),
    analysis_output: Optional[Path] = None,
    by_branching_dir: Optional[Path] = None,
    split_by: str = "class",
    bins: int = 50,
    scale_tol: float = 2e-3,
    residual_tol: float = 2e-2,
) -> None:
    """Compute reconstructed z over many events, plot it, and save analysis data.

    The saved analysis dictionary contains:
      - z_values: tensor of reconstructed z values
      - branchings: list of inferred FSR branching records
      - final_state_particles_by_event: list whose event entries are lists of
        final-state particle dictionaries. This is stored for later use but not
        otherwise analyzed here.

    In addition, z values are split by branching type and written to separate
    files under by_branching_dir. By default, split_by="class" groups charge
    conjugates/flavours into q_to_qg, g_to_gg, and g_to_qqbar style labels.
    Use split_by="exact" to separate by signed PDG IDs.
    """
    import matplotlib.pyplot as plt

    num_events_total = int(data["mask"].shape[0])
    num_events = num_events_total if max_events is None else min(max_events, num_events_total)

    z_values: list[float] = []
    branchings_all: list[dict] = []
    final_state_particles_by_event: list[list[dict]] = []

    for event_index in range(num_events):
        particles = collect_event_particles(data, event_index)
        edges = build_edges(particles)
        selected = {p["no"] for p in particles}

        final_state_particles_by_event.append([
            strip_particle_for_saving(p) for p in particles if p["status"] > 0
        ])

        branchings = infer_fsr_branchings(
            particles,
            edges,
            selected,
            scale_tol=scale_tol,
            residual_tol=residual_tol,
        )
        for br in branchings:
            br = dict(br)
            br["event"] = event_index
            br["branching_class"] = branching_class_from_record(br)
            br["branching_exact"] = branching_exact_from_record(br)
            branchings_all.append(br)
            if math.isfinite(br["z"]) and 0.0 <= br["z"] <= 1.0:
                z_values.append(float(br["z"]))

    if not z_values:
        raise RuntimeError(
            "No reconstructed z values found. Try increasing --z-residual-tol or "
            "check that your parsed file contains shower histories and scale values."
        )

    z_tensor = torch.tensor(z_values, dtype=torch.float64)
    z_output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.0, 4.0))
    plt.hist(z_values, bins=bins, range=(0.0, 1.0), density=True, histtype="step", linewidth=1.5)
    plt.xlabel(r"reconstructed $z$")
    plt.ylabel("density")
    plt.title(f"FSR reconstructed z distribution ({len(z_values)} branchings, {num_events} events)")
    plt.tight_layout()
    plt.savefig(z_output, dpi=200)
    plt.close()

    if analysis_output is None:
        analysis_output = z_output.with_suffix(".pt")
    analysis_output.parent.mkdir(parents=True, exist_ok=True)

    if by_branching_dir is None:
        by_branching_dir = analysis_output.with_suffix("").parent / f"{analysis_output.with_suffix('').name}_by_branching"
    written_by_branching = save_z_values_by_branching(
        branchings_all,
        output_dir=by_branching_dir,
        split_by=split_by,
        bins=bins,
    )

    torch.save(
        {
            "z_values": z_tensor,
            "branchings": branchings_all,
            "final_state_particles_by_event": final_state_particles_by_event,
            "z_files_by_branching": written_by_branching,
            "metadata": {
                "num_events_processed": num_events,
                "num_events_total": num_events_total,
                "num_branchings": len(branchings_all),
                "num_finite_z_values": len(z_values),
                "z_definition": "E_b/(E_b+E_c) in rest frame of post-branching dipole b+c+r'",
                "scale_tol": scale_tol,
                "residual_tol": residual_tol,
                "split_by": split_by,
                "by_branching_dir": str(by_branching_dir),
            },
        },
        analysis_output,
    )

    print(f"Processed events: {num_events} / {num_events_total}")
    print(f"Inferred FSR branchings: {len(branchings_all)}")
    print(f"Finite z values in [0,1]: {len(z_values)}")
    print(f"Mean z: {float(z_tensor.mean()):.6g}")
    print(f"Std z: {float(z_tensor.std(unbiased=False)):.6g}")
    print(f"Wrote z distribution plot: {z_output}")
    print(f"Saved z values, branch records, and final_state_particles_by_event: {analysis_output}")
    print(f"Saved per-branching z files under: {by_branching_dir}")
    for key, info in sorted(written_by_branching.items()):
        if key == "__overlay__":
            print(f"  overlay plot -> {info['plot_path']}")
        else:
            plot_msg = f", plot {info['plot_path']}" if info.get('plot_path') else ""
            print(
                f"  {key}: {info['num_finite_z_values']} z values "
                f"-> {info['pt_path']} and {info['csv_path']}{plot_msg}"
            )


def _hist_bins_for_positive_values(values: list[float], bins: int, logx: bool):
    """Return histogram bins for positive pT/evolution-scale-like values."""
    import numpy as np

    finite = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0]
    if not finite:
        return bins

    vmin = min(finite)
    vmax = max(finite)
    if vmax <= vmin:
        # Make a tiny range so matplotlib does not warn about identical limits.
        return np.linspace(max(0.0, 0.5 * vmin), 1.5 * vmax, bins + 1)

    if logx:
        return np.logspace(math.log10(vmin), math.log10(vmax), bins + 1)

    return np.linspace(0.0, 1.05 * vmax, bins + 1)


def save_scale_values_by_branching(
    branchings_all: list[dict],
    *,
    output_dir: Path,
    split_by: str = "class",
    bins: int = 50,
    logx: bool = False,
) -> dict[str, dict]:
    """Save and plot p_evol/scale values separately for each branching type.

    For each group this writes three files:
      - <branching>.pt: torch-save dictionary with tensor scale_values and records
      - <branching>.csv: lightweight table of event, line numbers, scale, Q, z, etc.
      - <branching>_p_evol_distribution.png: histogram of p_evol for that branching type

    It also writes p_evol_by_branching_overlay.png, a quick-look overlay of all
    non-empty per-branching histograms.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = {}
    for br in branchings_all:
        key = branching_key_from_record(br, split_by)
        groups.setdefault(key, []).append(br)

    written: dict[str, dict] = {}
    overlay_data: dict[str, list[float]] = {}

    for key, records in sorted(groups.items()):
        finite_records = [
            br for br in records
            if math.isfinite(float(br.get("scale", float("nan")))) and float(br["scale"]) > 0.0
        ]
        scale_list = [float(br["scale"]) for br in finite_records]
        scale_tensor = torch.tensor(scale_list, dtype=torch.float64)
        key_safe = safe_file_component(key)
        pt_path = output_dir / f"{key_safe}_p_evol.pt"
        csv_path = output_dir / f"{key_safe}_p_evol.csv"
        plot_path = output_dir / f"{key_safe}_p_evol_distribution.png"

        torch.save(
            {
                "branching_key": key,
                "split_by": split_by,
                "scale_values": scale_tensor,
                "p_evol_values": scale_tensor,
                "branchings": finite_records,
                "all_branchings_for_key": records,
                "metadata": {
                    "num_branchings_for_key": len(records),
                    "num_finite_scale_values": len(finite_records),
                    "scale_definition": "PYTHIA Particle::scale() for the inferred branching, interpreted here as p_evol in GeV",
                    "plot_path": str(plot_path),
                    "logx": logx,
                },
            },
            pt_path,
        )

        csv_fields = [
            "event",
            "radiator",
            "recoiler_before",
            "recoiler_after",
            "daughter_b",
            "daughter_c",
            "radiator_pid",
            "daughter_b_pid",
            "daughter_c_pid",
            "recoiler_pid",
            "scale",
            "Q",
            "Q2",
            "z",
            "z_lab",
            "residual",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for br in finite_records:
                writer.writerow({field: br.get(field, "") for field in csv_fields})

        if scale_list:
            overlay_data[key] = scale_list
            plt.figure(figsize=(6.0, 4.0))
            hist_bins = _hist_bins_for_positive_values(scale_list, bins, logx)
            plt.hist(scale_list, bins=hist_bins, density=True, histtype="step", linewidth=1.5)
            if logx:
                plt.xscale("log")
            plt.xlabel(r"$p_{\perp\mathrm{evol}}$ / PYTHIA scale [GeV]")
            plt.ylabel("density")
            plt.title(f"{key}: p_evol distribution ({len(scale_list)} branchings)")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
            plot_path_str: Optional[str] = str(plot_path)
        else:
            plot_path_str = None

        written[key] = {
            "pt_path": str(pt_path),
            "csv_path": str(csv_path),
            "plot_path": plot_path_str,
            "num_branchings": len(records),
            "num_finite_scale_values": len(finite_records),
        }

    if overlay_data:
        overlay_path = output_dir / "p_evol_by_branching_overlay.png"
        all_values = [v for values in overlay_data.values() for v in values]
        hist_bins = _hist_bins_for_positive_values(all_values, bins, logx)
        plt.figure(figsize=(6.5, 4.5))
        for key, values in sorted(overlay_data.items()):
            plt.hist(values, bins=hist_bins, density=True, histtype="step", linewidth=1.5, label=f"{key} ({len(values)})")
        if logx:
            plt.xscale("log")
        plt.xlabel(r"$p_{\perp\mathrm{evol}}$ / PYTHIA scale [GeV]")
        plt.ylabel("density")
        plt.title("p_evol by branching type")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=200)
        plt.close()
        written["__overlay__"] = {
            "plot_path": str(overlay_path),
            "num_branching_types": len(overlay_data),
        }


    return written


def q_to_qg_z_integral(z_min: float, z_max: float) -> float:
    """Integral of Eq. (70), P_{q->qg}(z)=C_F(1+z^2)/(1-z).

    The endpoint z=1 is singular, so z_max must be strictly smaller than 1.
    """
    cf = 4.0 / 3.0
    if not (0.0 <= z_min < z_max < 1.0):
        raise ValueError("Need 0 <= z_min < z_max < 1 for q->qg z integration.")

    def antiderivative(z: float) -> float:
        # Integral of (1+z^2)/(1-z) dz = -z - z^2/2 - 2 log(1-z).
        return -z - 0.5 * z * z - 2.0 * math.log(1.0 - z)

    return cf * (antiderivative(z_max) - antiderivative(z_min))


def sample_q_to_qg_p_evol_from_eqs(
    *,
    n_samples: int,
    p_min: float,
    p_max: float,
    alpha_s: float,
    z_min: float,
    z_max: float,
    seed: int = 12345,
) -> tuple[list[float], dict]:
    """Sample p_evol for q->qg using a simplified Eq. (70),(75),(76),(99).

    We use the z-integrated q->qg rate

        dP_a(t) = dt/t * alpha_s/(2 pi) * int dz P_q->qg(z),

    with t = p_evol^2, constant alpha_s, fixed z limits, and Sudakov

        Pi(t_max,t) = exp[- A log(t_max/t)] = (t/t_max)^A,

    where A = alpha_s/(2 pi) * int dz P(z).  Samples are drawn from the
    first-accepted-emission distribution Eq. (99), conditional on an emission in
    [p_min, p_max]. This is not a full PYTHIA shower: it intentionally keeps
    only the analytic ingredients in Eqs. (70),(75),(76),(99).
    """
    if n_samples <= 0:
        return [], {}
    if p_min <= 0 or p_max <= p_min:
        raise ValueError("Need 0 < p_min < p_max for theory p_evol sampling.")
    if alpha_s <= 0:
        raise ValueError("Need alpha_s > 0.")

    iz = q_to_qg_z_integral(z_min, z_max)
    a_rate = alpha_s / (2.0 * math.pi) * iz
    if a_rate <= 0:
        raise ValueError("Integrated rate is not positive; check z limits and alpha_s.")

    t_min = p_min * p_min
    t_max = p_max * p_max
    lower = (t_min / t_max) ** a_rate

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    u = torch.rand(int(n_samples), generator=gen, dtype=torch.float64)
    t = t_max * (lower + u * (1.0 - lower)).pow(1.0 / a_rate)
    p = torch.sqrt(t)
    p_list = [float(x) for x in p.tolist()]
    metadata = {
        "n_samples": int(n_samples),
        "p_min": float(p_min),
        "p_max": float(p_max),
        "alpha_s": float(alpha_s),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "integral_P_q_to_qg_dz": float(iz),
        "A_rate": float(a_rate),
        "emission_probability_above_p_min": float(1.0 - lower),
        "sampling_description": (
            "Eq.70 q->qg kernel, Eq.75 z-integrated branching rate, "
            "Eq.76 Sudakov, Eq.99 first-emission density; constant alpha_s, fixed z limits, conditional on an emission."
        ),
    }
    return p_list, metadata


def plot_q_to_qg_p_evol_pythia_vs_theory(
    branchings_all: list[dict],
    *,
    output_path: Path,
    bins: int = 50,
    logx: bool = False,
    theory_samples: int = 200_000,
    theory_alpha_s: float = 0.118,
    theory_z_min: float = 0.0,
    theory_z_max: float = 0.99,
    theory_p_min: Optional[float] = None,
    theory_p_max: Optional[float] = None,
    theory_seed: int = 12345,
    save_theory_pt: Optional[Path] = None,
) -> Optional[dict]:
    """Overlay PYTHIA q->qg p_evol with simplified analytic sampling."""
    import matplotlib.pyplot as plt

    q_to_qg_records = [
        br for br in branchings_all
        if branching_class_from_record(br) == "q_to_qg"
        and math.isfinite(float(br.get("scale", float("nan"))))
        and float(br["scale"]) > 0.0
    ]
    pythia_values = [float(br["scale"]) for br in q_to_qg_records]
    if not pythia_values:
        print("No q_to_qg branchings found, so no q->qg theory overlay was made.")
        return None

    p_min = float(theory_p_min) if theory_p_min is not None else min(pythia_values)
    p_max = float(theory_p_max) if theory_p_max is not None else max(pythia_values)
    print("p_max", p_max)
    if p_max <= p_min:
        # Avoid failure in tiny one-emission samples.
        p_min = max(1e-6, 0.5 * p_min)
        p_max = 1.5 * p_max

    theory_values, theory_meta = sample_q_to_qg_p_evol_from_eqs(
        n_samples=theory_samples,
        p_min=p_min,
        p_max=p_max,
        alpha_s=theory_alpha_s,
        z_min=theory_z_min,
        z_max=theory_z_max,
        seed=theory_seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_values = pythia_values + theory_values
    hist_bins = _hist_bins_for_positive_values(all_values, bins, logx)

    plt.figure(figsize=(6.8, 4.8))
    plt.hist(
        pythia_values,
        bins=hist_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label=f"PYTHIA q->qg ({len(pythia_values)})",
    )
    plt.hist(
        theory_values,
        bins=hist_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        label=f"Eq.70/75/76/99 sample ({len(theory_values)})",
    )
    if logx:
        plt.xscale("log")
    plt.xlabel(r"$p_{\perp\mathrm{evol}}$ [GeV]")
    plt.ylabel("density")
    plt.title(r"q→qg $p_{\perp\mathrm{evol}}$: PYTHIA vs simplified Sudakov sample")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    payload = {
        "pythia_p_evol_q_to_qg": torch.tensor(pythia_values, dtype=torch.float64),
        "theory_p_evol_q_to_qg": torch.tensor(theory_values, dtype=torch.float64),
        "pythia_branchings_q_to_qg": q_to_qg_records,
        "metadata": {
            **theory_meta,
            "plot_path": str(output_path),
            "note": (
                "The theory curve is a stripped-down Sudakov sample with constant alpha_s and fixed z limits. "
                "It is useful for comparing shapes, but it is not expected to exactly reproduce PYTHIA's full shower."
            ),
        },
    }
    if save_theory_pt is None:
        save_theory_pt = output_path.with_suffix(".pt")
    save_theory_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_theory_pt)

    print(f"q->qg PYTHIA branchings for overlay: {len(pythia_values)}")
    print(f"q->qg theory samples: {len(theory_values)}")
    print(f"Theory A = alpha_s/(2pi) * integral P dz = {theory_meta['A_rate']:.6g}")
    print(f"Wrote q->qg PYTHIA-vs-theory plot: {output_path}")
    print(f"Saved q->qg overlay values: {save_theory_pt}")
    return payload


def analyze_scale_distribution(
    data: dict,
    *,
    max_events: Optional[int] = None,
    scale_output: Path = Path("p_evol_distribution.png"),
    analysis_output: Optional[Path] = None,
    by_branching_dir: Optional[Path] = None,
    split_by: str = "class",
    bins: int = 50,
    scale_tol: float = 2e-3,
    residual_tol: float = 2e-2,
    logx: bool = False,
    overlay_q_to_qg_theory: bool = False,
    theory_output: Optional[Path] = None,
    theory_samples: int = 200_000,
    theory_alpha_s: float = 0.118,
    theory_z_min: float = 0.0,
    theory_z_max: float = 0.99,
    theory_p_min: Optional[float] = None,
    theory_p_max: Optional[float] = None,
    theory_seed: int = 12345,
) -> None:
    """Compute p_evol/scale over many inferred FSR branchings and plot it.

    The saved analysis dictionary contains:
      - scale_values / p_evol_values: tensor of PYTHIA scale values for branchings
      - branchings: list of inferred FSR branching records
      - final_state_particles_by_event: list whose event entries are lists of
        final-state particle dictionaries. This is stored for later use but not
        otherwise analyzed here.

    The per-branching output files and plots are written under by_branching_dir.
    """
    import matplotlib.pyplot as plt

    num_events_total = int(data["mask"].shape[0])
    num_events = num_events_total if max_events is None else min(max_events, num_events_total)

    scale_values: list[float] = []
    branchings_all: list[dict] = []
    final_state_particles_by_event: list[list[dict]] = []

    for event_index in range(num_events):
        particles = collect_event_particles(data, event_index)
        edges = build_edges(particles)
        selected = {p["no"] for p in particles}

        final_state_particles_by_event.append([
            strip_particle_for_saving(p) for p in particles if p["status"] > 0
        ])

        branchings = infer_fsr_branchings(
            particles,
            edges,
            selected,
            scale_tol=scale_tol,
            residual_tol=residual_tol,
        )
        for br in branchings:
            br = dict(br)
            br["event"] = event_index
            br["branching_class"] = branching_class_from_record(br)
            br["branching_exact"] = branching_exact_from_record(br)
            branchings_all.append(br)
            if math.isfinite(float(br.get("scale", float("nan")))) and float(br["scale"]) > 0.0:
                scale_values.append(float(br["scale"]))

    if not scale_values:
        raise RuntimeError(
            "No finite p_evol/scale values found. Try increasing --scale-residual-tol or "
            "check that your parsed file contains shower histories and scale values."
        )

    scale_tensor = torch.tensor(scale_values, dtype=torch.float64)
    scale_output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.0, 4.0))
    hist_bins = _hist_bins_for_positive_values(scale_values, bins, logx)
    plt.hist(scale_values, bins=hist_bins, density=True, histtype="step", linewidth=1.5)
    if logx:
        plt.xscale("log")
    plt.xlabel(r"$p_{\perp\mathrm{evol}}$ / PYTHIA scale [GeV]")
    plt.ylabel("density")
    plt.title(f"FSR p_evol distribution ({len(scale_values)} branchings, {num_events} events)")
    plt.tight_layout()
    plt.savefig(scale_output, dpi=200)
    plt.close()

    if analysis_output is None:
        analysis_output = scale_output.with_suffix(".pt")
    analysis_output.parent.mkdir(parents=True, exist_ok=True)

    if by_branching_dir is None:
        by_branching_dir = analysis_output.with_suffix("").parent / f"{analysis_output.with_suffix('').name}_p_evol_by_branching"
    written_by_branching = save_scale_values_by_branching(
        branchings_all,
        output_dir=by_branching_dir,
        split_by=split_by,
        bins=bins,
        logx=logx,
    )

    q_to_qg_theory_overlay_metadata = None
    if overlay_q_to_qg_theory:
        if theory_output is None:
            theory_output = by_branching_dir / "q_to_qg_p_evol_pythia_vs_eq70_75_76_99.png"
        q_to_qg_payload = plot_q_to_qg_p_evol_pythia_vs_theory(
            branchings_all,
            output_path=theory_output,
            bins=bins,
            logx=logx,
            theory_samples=theory_samples,
            theory_alpha_s=theory_alpha_s,
            theory_z_min=theory_z_min,
            theory_z_max=theory_z_max,
            theory_p_min=theory_p_min,
            theory_p_max=theory_p_max,
            theory_seed=theory_seed,
            save_theory_pt=theory_output.with_suffix(".pt"),
        )
        if q_to_qg_payload is not None:
            q_to_qg_theory_overlay_metadata = q_to_qg_payload["metadata"]

    torch.save(
        {
            "scale_values": scale_tensor,
            "p_evol_values": scale_tensor,
            "branchings": branchings_all,
            "final_state_particles_by_event": final_state_particles_by_event,
            "scale_files_by_branching": written_by_branching,
            "q_to_qg_theory_overlay": q_to_qg_theory_overlay_metadata,
            "metadata": {
                "num_events_processed": num_events,
                "num_events_total": num_events_total,
                "num_branchings": len(branchings_all),
                "num_finite_scale_values": len(scale_values),
                "scale_definition": "PYTHIA Particle::scale() for the inferred branching, interpreted here as p_evol in GeV",
                "scale_tol": scale_tol,
                "residual_tol": residual_tol,
                "split_by": split_by,
                "by_branching_dir": str(by_branching_dir),
                "logx": logx,
            },
        },
        analysis_output,
    )

    print(f"Processed events: {num_events} / {num_events_total}")
    print(f"Inferred FSR branchings: {len(branchings_all)}")
    print(f"Finite p_evol/scale values: {len(scale_values)}")
    print(f"Mean p_evol: {float(scale_tensor.mean()):.6g} GeV")
    print(f"Std p_evol: {float(scale_tensor.std(unbiased=False)):.6g} GeV")
    print(f"Wrote p_evol distribution plot: {scale_output}")
    print(f"Saved p_evol values, branch records, and final_state_particles_by_event: {analysis_output}")
    print(f"Saved per-branching p_evol files under: {by_branching_dir}")
    for key, info in sorted(written_by_branching.items()):
        if key == "__overlay__":
            print(f"  overlay plot -> {info['plot_path']}")
        else:
            plot_msg = f", plot {info['plot_path']}" if info.get("plot_path") else ""
            print(
                f"  {key}: {info['num_finite_scale_values']} p_evol values "
                f"-> {info['pt_path']} and {info['csv_path']}{plot_msg}"
            )

def descendants_of(nodes: set[int], edges: list[tuple[int, int]]) -> set[int]:
    children: dict[int, list[int]] = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)

    seen = set(nodes)
    frontier = list(nodes)
    while frontier:
        current = frontier.pop()
        for child in children.get(current, []):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return seen


def select_nodes(
    particles: list[dict],
    edges: list[tuple[int, int]],
    *,
    final_only: bool = False,
    hard_only: bool = False,
    hard_descendants: bool = False,
    status_abs_min: Optional[int] = None,
    status_abs_max: Optional[int] = None,
    max_nodes: Optional[int] = None,
) -> set[int]:
    selected = {p["no"] for p in particles}

    if final_only:
        selected &= {p["no"] for p in particles if p["status"] > 0}

    if hard_only:
        selected &= {p["no"] for p in particles if 20 <= abs(p["status"]) <= 29}

    if hard_descendants:
        hard = {p["no"] for p in particles if 20 <= abs(p["status"]) <= 29}
        selected &= descendants_of(hard, edges)

    if status_abs_min is not None:
        selected &= {p["no"] for p in particles if abs(p["status"]) >= status_abs_min}

    if status_abs_max is not None:
        selected &= {p["no"] for p in particles if abs(p["status"]) <= status_abs_max}

    if max_nodes is not None and len(selected) > max_nodes:
        # Keep earliest event-record entries. This is crude, but prevents enormous
        # event displays from becoming unreadable.
        selected = set(sorted(selected)[:max_nodes])

    return selected


def node_shape(status: int) -> str:
    if status > 0:
        return "box"      # final state
    if 20 <= abs(status) <= 29:
        return "doublecircle"  # hard process, approximately
    if abs(status) in {11, 12, 21, 22, 23, 31, 41, 42, 43, 44, 51, 52, 53, 59}:
        return "ellipse"  # common intermediate categories
    return "oval"


def node_label(p: dict, *, label_kinematics: bool = True, label_colors: bool = False, label_momentum: bool = False) -> str:
    lines = [
        f"{p['no']}: {p['name']}",
        f"status={p['status']}, scale={fmt_float(p['scale'])}",
    ]
    if label_kinematics:
        lines.append(f"E={p['energy']:.3f}")
        lines.append(f"px={p['px']:.3f}")
        lines.append(f"(py,pz,mass)={p['py']:.3f},{p['pz']:.3f},{p['mass']:.3f})")
    if label_colors and "color1" in p:
        lines.append(f"col=({p['color1']},{p['color2']})")
    if label_momentum:
        lines.append(f"px={p['energy']:.3f}")
        lines.append(f"{p['px']:.3f}")
        lines.append(f"{p['py']:.3f},{p['pz']:.3f},{p['mass']:.3f}) off-shell:{(p['energy']**2-p['px']**2-p['py']**2-p['pz']**2):.3f}")
    return "\n".join(lines)



def topological_depths_for_selected(
    selected: set[int],
    edges: list[tuple[int, int]],
) -> dict[int, int]:
    """Assign a simple left-to-right generation/depth to selected nodes.

    This is used only for Graphviz layout hints. The event record is a DAG, so
    the longest selected mother->child path is a reasonable column/rank index.
    """
    selected_edges = [(a, b) for a, b in edges if a in selected and b in selected]
    parents: dict[int, list[int]] = {n: [] for n in selected}
    children: dict[int, list[int]] = {n: [] for n in selected}
    indegree: dict[int, int] = {n: 0 for n in selected}

    for a, b in selected_edges:
        children.setdefault(a, []).append(b)
        parents.setdefault(b, []).append(a)
        indegree[b] = indegree.get(b, 0) + 1
        indegree.setdefault(a, 0)

    # Process lower event-record numbers first for determinism. The vertical
    # ordering itself is handled later by descending no. within each depth.
    queue = sorted([n for n in selected if indegree.get(n, 0) == 0])
    depth: dict[int, int] = {n: 0 for n in selected}
    seen_count = 0

    while queue:
        n = queue.pop(0)
        seen_count += 1
        for child in sorted(children.get(n, [])):
            depth[child] = max(depth.get(child, 0), depth.get(n, 0) + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort()

    # If a weird cycle or disconnected filtered fragment defeats the DAG pass,
    # keep the remaining nodes instead of crashing.
    if seen_count < len(selected):
        for n in selected:
            if n not in depth:
                depth[n] = 0

    return depth

def make_dot(
    particles: list[dict],
    edges: list[tuple[int, int]],
    selected: set[int],
    *,
    event_index: int,
    label_kinematics: bool = True,
    label_colors: bool = False,
    label_momentum: bool = False,
    label_recoil_q: bool = True,
    order_by_number: bool = True,
    rankdir: str = "LR",
) -> str:
    by_no = {p["no"]: p for p in particles}
    selected_edges = [(a, b) for a, b in edges if a in selected and b in selected]
    recoil_q_labels = (
        infer_recoil_q_edge_labels(particles, edges, selected)
        if label_recoil_q else {}
    )

    lines: list[str] = []
    lines.append("digraph PythiaEvent {")
    lines.append(f'  label="PYTHIA event {event_index}";')
    lines.append("  labelloc=t;")
    lines.append(f"  rankdir={rankdir};")
    lines.append("  graph [fontsize=18, overlap=false, splines=true];")
    lines.append("  node [fontname=Helvetica, fontsize=10, margin=0.06];")
    lines.append("  edge [arrowsize=0.7];")
    lines.append("")

    # Emit nodes. Descending order helps but is not enough by itself.
    for no in sorted(selected, reverse=order_by_number):
        p = by_no[no]
        label = dot_escape(
            node_label(
                p,
                label_kinematics=label_kinematics,
                label_colors=label_colors,
                label_momentum=label_momentum,
            )
        )
        shape = node_shape(p["status"])
        lines.append(f'  n{no} [label="{label}", shape={shape}];')

    lines.append("")

    # Group selected edges by parent.
    children_by_parent: dict[int, list[int]] = {}
    for a, b in selected_edges:
        children_by_parent.setdefault(a, []).append(b)

    # Emit edges parent-by-parent. For every actual branching, explicitly mark
    # the parent with ordering=out before writing its outgoing edges.
    for a in sorted(children_by_parent):
        children = children_by_parent[a]

        if order_by_number:
            children = sorted(children, reverse=True)  # larger no. first/above
        else:
            children = sorted(children)

        if len(children) >= 2:
            lines.append(f"  n{a} [ordering=out];")

        for b in children:
            edge_label = recoil_q_labels.get((a, b))
            if edge_label is None:
                lines.append(f"  n{a} -> n{b};")
            else:
                lines.append(
                    f'  n{a} -> n{b} '
                    f'[label="{dot_escape(edge_label)}", fontsize=9];'
                )

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_with_graphviz(dot_path: Path, output_path: Path, fmt: str) -> None:
    dot_exe = shutil.which("dot")
    if dot_exe is None:
        raise RuntimeError(
            "Graphviz executable 'dot' was not found. The .dot file was written, "
            "but rendering requires Graphviz. Install it with, for example:\n"
            "  sudo apt-get install graphviz\n"
            "or render manually later with:\n"
            f"  dot -T{fmt} {dot_path} -o {output_path}"
        )

    subprocess.run(
        [dot_exe, f"-T{fmt}", str(dot_path), "-o", str(output_path)],
        check=True,
    )


def graphviz_y_positions_from_dot(dot_text: str) -> dict[int, float]:
    """Return Graphviz-rendered y-positions for nodes named n<event-record-no>.

    This uses `dot -Tplain`, so the ordering matches the actual Graphviz layout
    used for the tree diagram. In Graphviz's coordinate system, larger y means
    visually higher in the rendered graph.
    """
    dot_exe = shutil.which("dot")
    if dot_exe is None:
        raise RuntimeError("Graphviz executable 'dot' was not found; cannot infer visual top-to-bottom order.")

    proc = subprocess.run(
        [dot_exe, "-Tplain"],
        input=dot_text,
        text=True,
        capture_output=True,
        check=True,
    )

    y_by_no: dict[int, float] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        # Plain-format node line:
        # node <name> <x> <y> <width> <height> <label> ...
        if len(parts) >= 4 and parts[0] == "node" and parts[1].startswith("n"):
            try:
                no = int(parts[1][1:])
                y = float(parts[3])
            except ValueError:
                continue
            y_by_no[no] = y
    return y_by_no


def final_hard_descendant_particles_in_tree_order(
    particles: list[dict],
    edges: list[tuple[int, int]],
    selected: set[int],
    y_by_no: dict[int, float],
) -> list[dict]:
    """Return final hard-descendant particles ordered top-to-bottom in the rendered tree.

    The hard-process roots are the outgoing hard particles with |status| == 23.
    We then keep final/current descendants with status > 0. Only particles in
    `selected` are included, so the order matches the actually drawn graph.
    """
    hard_outgoing = {p["no"] for p in particles if abs(p["status"]) == 23}
    hard_descendants = descendants_of(hard_outgoing, edges)

    final_hard_particles = [
        p for p in particles
        if p["status"] > 0
        and p["no"] in selected
        and p["no"] in hard_descendants
    ]

    # Top-most first. If a node somehow has no Graphviz position, put it last;
    # tie-break by event-record number for determinism.
    return sorted(
        final_hard_particles,
        key=lambda p: (-y_by_no.get(p["no"], float("-inf")), p["no"]),
    )


def save_final_state_hard_descendant_pid_rows_txt(
    rows: list[tuple[int, list[int]]],
    output_path: Path,
) -> None:
    """Save one line per event: event_index pid1 pid2 ..."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for event_index, pids in rows:
            pid_str = " ".join(str(pid) for pid in pids)
            f.write(f"{event_index} {pid_str}\n")
    print(f"Saved tree-ordered final-state hard-descendant PIDs: {output_path}")



def infer_graph_output_format(output_path: Path, explicit_fmt: Optional[str]) -> str:
    """Infer Graphviz output format from --format or the output suffix."""
    if explicit_fmt is not None:
        return explicit_fmt
    suffix = output_path.suffix.lower().lstrip(".")
    return suffix if suffix in {"dot", "png", "pdf", "svg"} else "dot"


def render_single_event_graph(
    data: dict,
    *,
    event_index: int,
    output_path: Path,
    fmt: str,
    final_only: bool = False,
    hard_only: bool = False,
    hard_descendants: bool = False,
    status_abs_min: Optional[int] = None,
    status_abs_max: Optional[int] = None,
    max_nodes: Optional[int] = None,
    label_kinematics: bool = True,
    label_colors: bool = False,
    label_momentum: bool = False,
    label_recoil_q: bool = True,
    order_by_number: bool = True,
    rankdir: str = "LR",
) -> tuple[int, int, int]:
    """Render one event and return (num_selected_nodes, num_particles, num_edges)."""
    particles = collect_event_particles(data, event_index)
    edges = build_edges(particles)
    selected = select_nodes(
        particles,
        edges,
        final_only=final_only,
        hard_only=hard_only,
        hard_descendants=hard_descendants,
        status_abs_min=status_abs_min,
        status_abs_max=status_abs_max,
        max_nodes=max_nodes,
    )

    if not selected:
        raise RuntimeError(
            f"No particles selected for event {event_index}. "
            "Try removing filters such as --hard-only or --final-only."
        )

    dot_text = make_dot(
        particles,
        edges,
        selected,
        event_index=event_index,
        label_kinematics=label_kinematics,
        label_colors=label_colors,
        label_momentum=label_momentum,
        label_recoil_q=label_recoil_q,
        order_by_number=order_by_number,
        rankdir=rankdir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "dot":
        dot_path = output_path.with_suffix(".dot") if output_path.suffix.lower() != ".dot" else output_path
        dot_path.write_text(dot_text, encoding="utf-8")
    else:
        render_path = output_path.with_suffix(f".{fmt}")
        dot_path = render_path.with_suffix(".dot")
        dot_path.write_text(dot_text, encoding="utf-8")
        render_with_graphviz(dot_path, render_path, fmt)

    selected_edge_count = sum(1 for a, b in edges if a in selected and b in selected)
    return len(selected), len(particles), selected_edge_count


def collect_final_state_particles_by_event(
    data: dict,
    event_indices: Iterable[int],
) -> list[list[dict]]:
    """Return a list of final-state particle records for each requested event."""
    final_state_particles_by_event: list[list[dict]] = []
    for event_index in event_indices:
        particles = collect_event_particles(data, event_index)
        final_state_particles_by_event.append([
            strip_particle_for_saving(p) for p in particles if p["status"] > 0
        ])
    return final_state_particles_by_event


def collect_final_state_hard_descendant_pids_by_event(
    data: dict,
    event_indices: list[int],
) -> list[list[int]]:
    """
    For each event, save PIDs of final-state particles descended from the
    outgoing hard-process particles.

    Uses PYTHIA convention:
      |status| = 23  : outgoing hard-process particles
      status > 0     : final/current particles
    """
    pid_lists_by_event: list[list[int]] = []

    for event_index in event_indices:
        particles = collect_event_particles(data, event_index)
        edges = build_edges(particles)

        hard_outgoing = {
            p["no"]
            for p in particles
            if abs(p["status"]) == 23
        }

        hard_descendants = descendants_of(hard_outgoing, edges)

        final_pids = [
            int(p["pid"])
            for p in particles
            if p["status"] > 0 and p["no"] in hard_descendants
        ]

        pid_lists_by_event.append(final_pids)

    return pid_lists_by_event


def save_final_state_hard_descendant_pids_txt(
    data: dict,
    event_indices: list[int],
    output_path: Path,
) -> None:
    """
    Save one line per event:

        event_index pid1 pid2 pid3 ...

    containing only final-state PIDs descended from hard-process outgoing
    particles.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pid_lists_by_event = collect_final_state_hard_descendant_pids_by_event(
        data,
        event_indices,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for event_index, pids in zip(event_indices, pid_lists_by_event):
            pid_str = " ".join(str(pid) for pid in pids)
            f.write(f"{event_index} {pid_str}\n")

    print(f"Saved final-state hard-descendant PIDs: {output_path}")


def render_many_event_graphs(
    data: dict,
    *,
    event_indices: list[int],
    output_dir: Path,
    fmt: str,
    output_prefix: str = "event",
    final_state_output: Optional[Path] = None,
    final_only: bool = False,
    hard_only: bool = False,
    hard_descendants: bool = False,
    status_abs_min: Optional[int] = None,
    status_abs_max: Optional[int] = None,
    max_nodes: Optional[int] = None,
    label_kinematics: bool = True,
    label_colors: bool = False,
    label_momentum: bool = False,
    label_recoil_q: bool = True,
    order_by_number: bool = True,
    rankdir: str = "LR",
) -> None:
    """Render one tree diagram per event and save tree-ordered final-state PIDs.

    The txt output contains one row per event:

        event_index pid1 pid2 pid3 ...

    where the PIDs are final-state descendants of the outgoing hard-process
    particles, ordered from top to bottom according to the actual Graphviz
    layout used for the rendered tree.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    width = max(4, len(str(max(event_indices))) if event_indices else 4)
    pid_rows: list[tuple[int, list[int]]] = []

    for event_index in event_indices:
        particles = collect_event_particles(data, event_index)
        edges = build_edges(particles)
        selected = select_nodes(
            particles,
            edges,
            final_only=final_only,
            hard_only=hard_only,
            hard_descendants=hard_descendants,
            status_abs_min=status_abs_min,
            status_abs_max=status_abs_max,
            max_nodes=max_nodes,
        )

        if not selected:
            raise RuntimeError(
                f"No particles selected for event {event_index}. "
                "Try removing filters such as --hard-only or --final-only."
            )

        dot_text = make_dot(
            particles,
            edges,
            selected,
            event_index=event_index,
            label_kinematics=label_kinematics,
            label_colors=label_colors,
            label_momentum=label_momentum,
            label_recoil_q=label_recoil_q,
            order_by_number=order_by_number,
            rankdir=rankdir,
        )

        out_path = output_dir / f"{output_prefix}_{event_index:0{width}d}.{fmt}"
        if fmt == "dot":
            dot_path = out_path.with_suffix(".dot") if out_path.suffix.lower() != ".dot" else out_path
            dot_path.write_text(dot_text, encoding="utf-8")
            rendered_path = dot_path
        else:
            rendered_path = out_path.with_suffix(f".{fmt}")
            dot_path = rendered_path.with_suffix(".dot")
            dot_path.write_text(dot_text, encoding="utf-8")
            render_with_graphviz(dot_path, rendered_path, fmt)

        # Use the same DOT text to ask Graphviz where nodes were placed, then
        # order final-state hard descendants by visual top-to-bottom position.
        try:
            y_by_no = graphviz_y_positions_from_dot(dot_text)
            final_hard_particles = final_hard_descendant_particles_in_tree_order(
                particles,
                edges,
                selected,
                y_by_no,
            )
        except Exception as exc:
            print(
                f"Warning: could not infer Graphviz top-to-bottom order for event {event_index}: {exc}. "
                "Falling back to descending event-record number."
            )
            hard_outgoing = {p["no"] for p in particles if abs(p["status"]) == 23}
            hard_desc = descendants_of(hard_outgoing, edges)
            final_hard_particles = sorted(
                [
                    p for p in particles
                    if p["status"] > 0 and p["no"] in selected and p["no"] in hard_desc
                ],
                key=lambda p: -p["no"],
            )

        pid_rows.append((event_index, [int(p["pid"]) for p in final_hard_particles]))

        selected_edge_count = sum(1 for a, b in edges if a in selected and b in selected)
        print(
            f"Rendered event {event_index}: {rendered_path} "
            f"({len(selected)}/{len(particles)} nodes, {selected_edge_count} selected edges; "
            f"{len(final_hard_particles)} final hard-descendant PIDs)"
        )

    if final_state_output is None:
        final_state_output = output_dir / "final_hard_descendant_pids_tree_order.txt"
    save_final_state_hard_descendant_pid_rows_txt(pid_rows, final_state_output)


# render_with_graphviz("/home/hiboy/jet_interpretability/pythia_events/qq2qq.dot", "/home/hiboy/jet_interpretability/pythia_events/qq2qq.png", "png")
# exit()


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Draw a process tree/DAG from a parsed PYTHIA events.pt file."
    )
    parser.add_argument("ptfile", type=Path, help="Input .pt file from parse_pythia_event_log.py")
    parser.add_argument("--event", type=int, default=0, help="Event index to plot, starting from 0")
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="Render a separate tree diagram for many events instead of only --event.",
    )
    parser.add_argument(
        "--event-start",
        type=int,
        default=0,
        help="First event index to render when using --all-events.",
    )
    parser.add_argument(
        "--events-output-dir",
        type=Path,
        default=None,
        help="Directory for per-event diagrams when using --all-events. Default: <output-stem>_events.",
    )
    parser.add_argument(
        "--event-output-prefix",
        type=str,
        default="event",
        help="Filename prefix for per-event diagrams written by --all-events.",
    )
    parser.add_argument(
        "--final-state-output",
        type=Path,
        default=None,
        help="Output .txt path for final hard-descendant PIDs in visual tree order when using --all-events. Default: <events-output-dir>/final_hard_descendant_pids_tree_order.txt.",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("event0.dot"), help="Output .dot/.png/.pdf/.svg path")
    parser.add_argument(
        "--format",
        choices=["dot", "png", "pdf", "svg"],
        default=None,
        help="Output format. Default: inferred from output suffix, or dot.",
    )
    parser.add_argument("--final-only", action="store_true", help="Plot only particles with status > 0")
    parser.add_argument("--hard-only", action="store_true", help="Plot only particles with 20 <= |status| <= 29")
    parser.add_argument(
        "--hard-descendants",
        action="store_true",
        help="Plot hard-process particles, approximately |status|=20..29, and all their descendants",
    )
    parser.add_argument("--status-abs-min", type=int, default=None, help="Keep only particles with |status| >= this")
    parser.add_argument("--status-abs-max", type=int, default=None, help="Keep only particles with |status| <= this")
    parser.add_argument("--max-nodes", type=int, default=None, help="Maximum number of nodes to draw")
    parser.add_argument("--no-kinematics", action="store_true", help="Do not include pT/eta/E in labels")
    parser.add_argument("--label-colors", action="store_true", help="Include PYTHIA color tags in labels")
    parser.add_argument("--label-momentum", action="store_true")
    parser.add_argument(
        "--no-recoil-q-labels",
        action="store_true",
        help="Do not label inferred recoiler-copy edges with the branching Q value.",
    )
    parser.add_argument(
        "--analyze-z",
        action="store_true",
        help="Compute reconstructed shower z over many events, plot its distribution, and save analysis data.",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=None,
        help="Number of events to process for --analyze-z, --analyze-scale, or --all-events. Default: all available events for that mode.",
    )
    parser.add_argument(
        "--z-output",
        type=Path,
        default=Path("z_distribution.png"),
        help="Output image path for the reconstructed z histogram used by --analyze-z.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=None,
        help="Output .pt path for z values, branch records, and final_state_particles_by_event. Default: z-output with .pt suffix.",
    )
    parser.add_argument(
        "--z-by-branching-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-branching z files written by --analyze-z. "
            "Default: <analysis-output-stem>_by_branching."
        ),
    )
    parser.add_argument(
        "--z-split-by",
        choices=["class", "exact"],
        default="class",
        help=(
            "How to split z files: 'class' groups q->qg/g->gg/g->qqbar; "
            "'exact' separates by signed PDG IDs."
        ),
    )
    parser.add_argument("--z-bins", type=int, default=50, help="Number of bins in the z histogram.")
    parser.add_argument("--z-scale-tol", type=float, default=2e-3, help="Scale matching tolerance for z reconstruction.")
    parser.add_argument("--z-residual-tol", type=float, default=2e-2, help="Four-momentum residual tolerance for z reconstruction.")
    parser.add_argument(
        "--analyze-scale",
        action="store_true",
        help="Compute PYTHIA shower scale/p_evol over many events, plot its distribution, and save analysis data.",
    )
    parser.add_argument(
        "--scale-output",
        type=Path,
        default=Path("p_evol_distribution.png"),
        help="Output image path for the p_evol/PYTHIA-scale histogram used by --analyze-scale.",
    )
    parser.add_argument(
        "--scale-analysis-output",
        type=Path,
        default=None,
        help="Output .pt path for p_evol values, branch records, and final_state_particles_by_event. Default: scale-output with .pt suffix.",
    )
    parser.add_argument(
        "--scale-by-branching-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-branching p_evol files written by --analyze-scale. "
            "Default: <scale-analysis-output-stem>_p_evol_by_branching."
        ),
    )
    parser.add_argument(
        "--scale-split-by",
        choices=["class", "exact"],
        default="class",
        help=(
            "How to split p_evol files: 'class' groups q->qg/g->gg/g->qqbar; "
            "'exact' separates by signed PDG IDs."
        ),
    )
    parser.add_argument("--scale-bins", type=int, default=50, help="Number of bins in the p_evol histogram.")
    parser.add_argument("--scale-scale-tol", type=float, default=2e-3, help="Scale matching tolerance for p_evol reconstruction.")
    parser.add_argument("--scale-residual-tol", type=float, default=2e-2, help="Four-momentum residual tolerance for p_evol reconstruction.")
    parser.add_argument(
        "--scale-logx",
        action="store_true",
        help="Use a logarithmic x-axis for p_evol histograms.",
    )
    parser.add_argument(
        "--overlay-q-to-qg-theory",
        action="store_true",
        help=(
            "When used with --analyze-scale, also sample p_evol for q->qg from "
            "Eqs. 70,75,76,99 and overlay it with PYTHIA's reconstructed q->qg scale distribution."
        ),
    )
    parser.add_argument(
        "--theory-output",
        type=Path,
        default=None,
        help=(
            "Output path for the q->qg PYTHIA-vs-theory p_evol overlay. "
            "Default: <scale-by-branching-dir>/q_to_qg_p_evol_pythia_vs_eq70_75_76_99.png."
        ),
    )
    parser.add_argument("--theory-samples", type=int, default=200000, help="Number of simplified Sudakov q->qg p_evol samples.")
    parser.add_argument("--theory-alpha-s", type=float, default=0.118, help="Constant alpha_s used in the simplified Eq.75 rate.")
    parser.add_argument("--theory-z-min", type=float, default=0.0, help="Lower z limit in the Eq.75 z integral.")
    parser.add_argument("--theory-z-max", type=float, default=0.7, help="Upper z limit in the Eq.75 z integral; must be less than 1.")
    parser.add_argument("--theory-p-min", type=float, default=None, help="Lower p_evol cutoff for theory sampling. Default: min PYTHIA q->qg scale.")
    parser.add_argument("--theory-p-max", type=float, default=None, help="Upper p_evol starting scale for theory sampling. Default: max PYTHIA q->qg scale.")
    parser.add_argument("--theory-seed", type=int, default=12345, help="Random seed for simplified q->qg p_evol sampling.")
    parser.add_argument(
        "--no-order-by-number",
        action="store_true",
        help="Disable Graphviz layout hints that place larger event-record numbers above smaller ones.",
    )
    parser.add_argument(
        "--rankdir",
        choices=["LR", "TB", "RL", "BT"],
        default="LR",
        help="Graph direction: LR left-to-right, TB top-to-bottom, etc.",
    )
    args = parser.parse_args(argv)

    data = torch.load(args.ptfile, map_location="cpu")
    if "mask" not in data or not torch.is_tensor(data["mask"]):
        raise ValueError("Input .pt file does not look like parser output: missing tensor key 'mask'.")

    num_events = data["mask"].shape[0]

    if args.all_events:
        if args.event_start < 0 or args.event_start >= num_events:
            raise IndexError(f"--event-start must be between 0 and {num_events - 1}, got {args.event_start}")
        end_event = num_events if args.num_events is None else min(num_events, args.event_start + args.num_events)
        event_indices = list(range(args.event_start, end_event))
        if not event_indices:
            raise RuntimeError("No events selected for --all-events.")

        fmt = infer_graph_output_format(args.output, args.format)
        if args.events_output_dir is not None:
            events_output_dir = args.events_output_dir
        else:
            events_output_dir = args.output.parent / f"{args.output.stem}_events"

        render_many_event_graphs(
            data,
            event_indices=event_indices,
            output_dir=events_output_dir,
            fmt=fmt,
            output_prefix=args.event_output_prefix,
            final_state_output=args.final_state_output,
            final_only=args.final_only,
            hard_only=args.hard_only,
            hard_descendants=args.hard_descendants,
            status_abs_min=args.status_abs_min,
            status_abs_max=args.status_abs_max,
            max_nodes=args.max_nodes,
            label_kinematics=not args.no_kinematics,
            label_colors=args.label_colors,
            label_momentum=args.label_momentum,
            label_recoil_q=not args.no_recoil_q_labels,
            order_by_number=not args.no_order_by_number,
            rankdir=args.rankdir,
        )
        return

    if args.analyze_scale:
        analyze_scale_distribution(
            data,
            max_events=args.num_events,
            scale_output=args.scale_output,
            analysis_output=args.scale_analysis_output,
            by_branching_dir=args.scale_by_branching_dir,
            split_by=args.scale_split_by,
            bins=args.scale_bins,
            scale_tol=args.scale_scale_tol,
            residual_tol=args.scale_residual_tol,
            logx=args.scale_logx,
            overlay_q_to_qg_theory=args.overlay_q_to_qg_theory,
            theory_output=args.theory_output,
            theory_samples=args.theory_samples,
            theory_alpha_s=args.theory_alpha_s,
            theory_z_min=args.theory_z_min,
            theory_z_max=args.theory_z_max,
            theory_p_min=args.theory_p_min,
            theory_p_max=args.theory_p_max,
            theory_seed=args.theory_seed,
        )
        return

    if args.analyze_z:
        analyze_z_distribution(
            data,
            max_events=args.num_events,
            z_output=args.z_output,
            analysis_output=args.analysis_output,
            by_branching_dir=args.z_by_branching_dir,
            split_by=args.z_split_by,
            bins=args.z_bins,
            scale_tol=args.z_scale_tol,
            residual_tol=args.z_residual_tol,
        )
        return

    if args.event < 0 or args.event >= num_events:
        raise IndexError(f"--event must be between 0 and {num_events - 1}, got {args.event}")

    particles = collect_event_particles(data, args.event)
    edges = build_edges(particles)
    selected = select_nodes(
        particles,
        edges,
        final_only=args.final_only,
        hard_only=args.hard_only,
        hard_descendants=args.hard_descendants,
        status_abs_min=args.status_abs_min,
        status_abs_max=args.status_abs_max,
        max_nodes=args.max_nodes,
    )

    if not selected:
        raise RuntimeError("No particles selected. Try removing filters such as --hard-only or --final-only.")

    dot_text = make_dot(
        particles,
        edges,
        selected,
        event_index=args.event,
        label_kinematics=not args.no_kinematics,
        label_colors=args.label_colors,
        label_momentum=args.label_momentum,
        label_recoil_q=not args.no_recoil_q_labels,
        order_by_number=not args.no_order_by_number,
        rankdir=args.rankdir,
    )

    fmt = args.format
    if fmt is None:
        suffix = args.output.suffix.lower().lstrip(".")
        fmt = suffix if suffix in {"dot", "png", "pdf", "svg"} else "dot"

    if fmt == "dot":
        dot_path = args.output
        dot_path.write_text(dot_text, encoding="utf-8")
        print(f"Wrote DOT file: {dot_path}")
        print(f"Selected nodes: {len(selected)} / {len(particles)}")
        print(f"Selected edges: {sum(1 for a, b in edges if a in selected and b in selected)} / {len(edges)}")
        return

    # For rendered output, also write a sibling .dot file for debugging/editing.
    dot_path = args.output.with_suffix(".dot")
    dot_path.write_text(dot_text, encoding="utf-8")
    render_with_graphviz(dot_path, args.output, fmt)
    print(f"Wrote DOT file: {dot_path}")
    print(f"Rendered graph: {args.output}")
    print(f"Selected nodes: {len(selected)} / {len(particles)}")
    print(f"Selected edges: {sum(1 for a, b in edges if a in selected and b in selected)} / {len(edges)}")


if __name__ == "__main__":
    main()
