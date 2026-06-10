#!/usr/bin/env python3
"""
Draw PYTHIA event-record tree/DAG diagrams from a parsed .pt file.

Keeps only tree-diagram generation, local-FSR recoiler edge labels, and the
print statements useful for debugging event-record numbering/status issues.

Expected input format: the .pt file produced by parse_pythia_log.py, with at
least these keys:

    pid or id, status, mothers, p4, mask

and optionally:

    no, names, daughters, colors, mass, scale
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import torch


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
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


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
    return PDG_NAMES.get(int(pid), str(int(pid)))


def p4_kinematics(px: float, py: float, pz: float) -> tuple[float, float, float]:
    """Return pT, eta, phi for a three-momentum."""
    pt = math.hypot(px, py)
    phi = math.atan2(py, px)
    if pt == 0.0:
        eta = math.copysign(float("inf"), pz) if pz != 0 else 0.0
    else:
        eta = math.asinh(pz / pt)
    return pt, eta, phi


def fmt_float(x: float, ndigits: int = 3) -> str:
    x = float(x)
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    if math.isnan(x):
        return "nan"
    return f"{x:.{ndigits}g}"


def collect_event_particles(data: dict, event_index: int) -> list[dict]:
    """Convert one padded event from parser tensors into particle dictionaries."""
    mask = get_event_field(data, "mask", event_index).bool()
    pid_key = "pid" if "pid" in data else "id"
    pid = get_event_field(data, pid_key, event_index).long()
    status = get_event_field(data, "status", event_index).long()
    mothers = get_event_field(data, "mothers", event_index).long()
    p4 = get_event_field(data, "p4", event_index).detach().cpu()

    no_field = get_event_field(data, "no", event_index).long() if "no" in data else None
    daughters = get_event_field(data, "daughters", event_index).long() if "daughters" in data else None
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
        pt, eta, phi = p4_kinematics(px, py, pz)

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
            "p4": p4_vec,
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": float(tensor_item(mass[j])) if mass is not None else float("nan"),
            "scale": float(tensor_item(scale[j])) if scale is not None else float("nan"),
        }

        if daughters is not None:
            particle["daughter1"] = int(daughters[j, 0].item())
            particle["daughter2"] = int(daughters[j, 1].item())
            particle["daughters"] = (particle["daughter1"], particle["daughter2"])
        else:
            particle["daughter1"] = 0
            particle["daughter2"] = 0
            particle["daughters"] = (0, 0)

        if colors is not None:
            particle["color1"] = int(colors[j, 0].item())
            particle["color2"] = int(colors[j, 1].item())

        particles.append(particle)

    return particles


def build_edges(particles: list[dict]) -> list[tuple[int, int]]:
    """Build mother -> child edges using PYTHIA mother columns."""
    existing = {int(p["no"]) for p in particles}
    edges: set[tuple[int, int]] = set()

    for p in particles:
        child = int(p["no"])
        for mother in sorted({int(p.get("mother1", 0)), int(p.get("mother2", 0))}):
            if mother > 0 and mother in existing and mother != child:
                edges.add((mother, child))

    return sorted(edges)


# ---------------------------------------------------------------------------
# Four-vector helpers for recoiler-label reconstruction.
# Convention: (px, py, pz, E).
# ---------------------------------------------------------------------------


def p4_tensor(p: dict) -> torch.Tensor:
    if "p4" in p and torch.is_tensor(p["p4"]):
        return p["p4"].to(dtype=torch.float64)
    return torch.tensor([p["px"], p["py"], p["pz"], p["energy"]], dtype=torch.float64)


def add_p4(*vecs: torch.Tensor) -> torch.Tensor:
    if not vecs:
        return torch.zeros(4, dtype=torch.float64)
    return torch.stack([v.to(dtype=torch.float64) for v in vecs], dim=0).sum(dim=0)


def sub_p4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.to(dtype=torch.float64) - b.to(dtype=torch.float64)


def p4_norm(v: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(v.to(dtype=torch.float64)).item())


def inv_mass2(v: torch.Tensor) -> float:
    v = v.to(dtype=torch.float64)
    return float((v[3] * v[3] - torch.dot(v[:3], v[:3])).item())


def children_by_mother(edges: list[tuple[int, int]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for mother, child in edges:
        children.setdefault(int(mother), []).append(int(child))
    return children


def boost_to_rest_frame(p4: torch.Tensor, total_p4: torch.Tensor) -> torch.Tensor:
    """Boost p4 into the rest frame of total_p4. Convention: (px, py, pz, E)."""
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
        return p4.clone()

    gamma = 1.0 / math.sqrt(1.0 - beta2_float)
    p_vec = p4[:3]
    e = p4[3]
    beta_dot_p = torch.dot(beta, p_vec)

    e_prime = gamma * (e - beta_dot_p)
    p_prime = p_vec + (((gamma - 1.0) * beta_dot_p / beta2) - gamma * e) * beta
    return torch.cat([p_prime, e_prime.reshape(1)])


def choose_z_daughter(parent: dict, child1: dict, child2: dict) -> tuple[dict, dict]:
    """Pick daughter b for z. For q->qg, b is the daughter with parent's PID."""
    parent_pid = parent.get("pid")
    if parent_pid != 21:
        for candidate in (child1, child2):
            if candidate.get("pid") == parent_pid:
                other = child2 if candidate is child1 else child1
                return candidate, other
    return child1, child2


def reconstructed_z_for_branch(
    by_no: dict[int, dict],
    a: int,
    bc: list[int],
    rp: int,
) -> tuple[float, float, int, int]:
    """Reconstruct z = E_b*/(E_b*+E_c*) in the post-branching dipole rest frame."""
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
    debug: bool = False,
) -> list[dict]:
    """Infer local FSR branchings (a,r)->(b,c,r') for recoiler labels."""
    by_no = {int(p["no"]): p for p in particles}
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
        r_part = by_no[int(r)]
        rp_part = by_no[int(rp)]

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
            z, z_lab, b, c = reconstructed_z_for_branch(by_no, a, bc_raw, int(rp))

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
    debug: bool = False,
) -> dict[tuple[int, int], str]:
    """Infer FSR recoiler-copy edges r -> r' and label them with Q and z."""
    labels: dict[tuple[int, int], str] = {}
    for br in infer_fsr_branchings(
        particles,
        edges,
        selected,
        scale_tol=scale_tol,
        residual_tol=residual_tol,
        debug=debug,
    ):
        r = br["recoiler_before"]
        rp = br["recoiler_after"]
        a = br["radiator"]
        b = br["daughter_b"]
        c = br["daughter_c"]
        labels[(r, rp)] = f"Q={br['Q']:.3g}, z={br['z']:.3g}, recoil for {a}->{b},{c}"
    return labels


# ---------------------------------------------------------------------------
# Tree-diagram selection, DOT generation, rendering, and final PID export.
# ---------------------------------------------------------------------------


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
    selected = {int(p["no"]) for p in particles}

    if final_only:
        selected &= {int(p["no"]) for p in particles if int(p["status"]) > 0}

    if hard_only:
        selected &= {int(p["no"]) for p in particles if 20 <= abs(int(p["status"])) <= 29}

    if hard_descendants:
        hard = {int(p["no"]) for p in particles if 20 <= abs(int(p["status"])) <= 29}
        selected &= descendants_of(hard, edges)

    if status_abs_min is not None:
        selected &= {int(p["no"]) for p in particles if abs(int(p["status"])) >= status_abs_min}

    if status_abs_max is not None:
        selected &= {int(p["no"]) for p in particles if abs(int(p["status"])) <= status_abs_max}

    if max_nodes is not None and len(selected) > max_nodes:
        selected = set(sorted(selected)[:max_nodes])

    return selected


def node_shape(status: int) -> str:
    status = int(status)
    if status > 0:
        return "box"
    if 20 <= abs(status) <= 29:
        return "doublecircle"
    if abs(status) in {11, 12, 21, 22, 23, 31, 41, 42, 43, 44, 51, 52, 53, 59}:
        return "ellipse"
    return "oval"


def node_label(
    p: dict,
    *,
    label_kinematics: bool = True,
    label_colors: bool = False,
    label_momentum: bool = False,
) -> str:
    lines = [
        f"{p['no']}: {p['name']}",
        f"pid={p['pid']}, status={p['status']}, scale={fmt_float(p.get('scale', float('nan')))}",
    ]
    if label_kinematics:
        lines.append(f"E={float(p['energy']):.3f}")
        lines.append(f"px={float(p['px']):.3f}")
        lines.append(
            f"(py,pz,mass)={float(p['py']):.3f},{float(p['pz']):.3f},{float(p.get('mass', float('nan'))):.3f})"
        )
    if label_colors and "color1" in p:
        lines.append(f"col=({p['color1']},{p['color2']})")
    if label_momentum:
        offshell = p["energy"] ** 2 - p["px"] ** 2 - p["py"] ** 2 - p["pz"] ** 2
        lines.append(f"off-shell={offshell:.3f}")
    return "\n".join(lines)


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
    debug_recoil: bool = False,
    order_by_number: bool = True,
    rankdir: str = "LR",
) -> str:
    by_no = {int(p["no"]): p for p in particles}
    selected_edges = [(a, b) for a, b in edges if a in selected and b in selected]
    recoil_q_labels = (
        infer_recoil_q_edge_labels(particles, edges, selected, debug=debug_recoil)
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
        lines.append(f'  n{no} [label="{label}", shape={node_shape(p["status"])}];')

    lines.append("")

    children_by_parent: dict[int, list[int]] = {}
    for a, b in selected_edges:
        children_by_parent.setdefault(a, []).append(b)

    # Emit edges parent-by-parent. With rankdir=LR, ordering=out makes the
    # first outgoing edge tend to appear above later outgoing edges.
    for a in sorted(children_by_parent):
        children = sorted(children_by_parent[a], reverse=order_by_number)
        if len(children) >= 2:
            lines.append(f"  n{a} [ordering=out];")

        for b in children:
            edge_label = recoil_q_labels.get((a, b))
            if edge_label is None:
                lines.append(f"  n{a} -> n{b};")
            else:
                lines.append(f'  n{a} -> n{b} [label="{dot_escape(edge_label)}", fontsize=9];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_with_graphviz(dot_path: Path, output_path: Path, fmt: str) -> None:
    dot_exe = shutil.which("dot")
    if dot_exe is None:
        raise RuntimeError(
            "Graphviz executable 'dot' was not found. The .dot file was written, "
            "but rendering requires Graphviz. Install it with:\n"
            "  sudo apt-get install graphviz\n"
            "or render manually with:\n"
            f"  dot -T{fmt} {dot_path} -o {output_path}"
        )
    subprocess.run([dot_exe, f"-T{fmt}", str(dot_path), "-o", str(output_path)], check=True)


def graphviz_y_positions_from_dot(dot_text: str) -> dict[int, float]:
    """Return rendered y-positions for DOT nodes named n<event-record-no>."""
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
        # plain format: node <name> <x> <y> <width> <height> <label> ...
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
    """Final/current hard descendants ordered top-to-bottom in rendered tree."""
    hard_outgoing = {int(p["no"]) for p in particles if abs(int(p["status"])) == 23}
    hard_descendants = descendants_of(hard_outgoing, edges) if hard_outgoing else set(selected)

    final_hard_particles = [
        p for p in particles
        if int(p["status"]) > 0
        and int(p["no"]) in selected
        and int(p["no"]) in hard_descendants
    ]

    return sorted(
        final_hard_particles,
        key=lambda p: (-y_by_no.get(int(p["no"]), float("-inf")), int(p["no"])),
    )


def save_final_state_hard_descendant_pid_rows_txt(rows: list[tuple[int, list[int]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for event_index, pids in rows:
            pid_str = " ".join(str(pid) for pid in pids)
            f.write(f"{event_index} {pid_str}\n")
    print(f"Saved tree-ordered final-state hard-descendant PIDs: {output_path}")


def infer_graph_output_format(output_path: Path, explicit_fmt: Optional[str]) -> str:
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
    debug_recoil: bool = False,
    debug_particles: bool = True,
    order_by_number: bool = True,
    rankdir: str = "LR",
) -> tuple[str, list[dict], list[tuple[int, int]], set[int], int, Path]:
    """Render one event and return the DOT text plus data used for PID output."""
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
        
    # print(
    #         "event_index, no, pid, status, daughters:",
    #         event_index,
    #         [(p["no"], p["pid"], p["status"], p.get("daughters")) for p in particles],
    #     )

    dot_text = make_dot(
        particles,
        edges,
        selected,
        event_index=event_index,
        label_kinematics=label_kinematics,
        label_colors=label_colors,
        label_momentum=label_momentum,
        label_recoil_q=label_recoil_q,
        debug_recoil=debug_recoil,
        order_by_number=order_by_number,
        rankdir=rankdir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "dot":
        written_path = output_path.with_suffix(".dot") if output_path.suffix.lower() != ".dot" else output_path
        written_path.write_text(dot_text, encoding="utf-8")
    else:
        render_path = output_path.with_suffix(f".{fmt}")
        dot_path = render_path.with_suffix(".dot")
        dot_path.write_text(dot_text, encoding="utf-8")
        render_with_graphviz(dot_path, render_path, fmt)
        written_path = render_path

    selected_edge_count = sum(1 for a, b in edges if a in selected and b in selected)
    return dot_text, particles, edges, selected, selected_edge_count, written_path


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
    debug_recoil: bool = False,
    debug_particles: bool = True,
    order_by_number: bool = True,
    rankdir: str = "LR",
) -> None:
    """Render one tree diagram per event and optionally save tree-ordered final PIDs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(max(event_indices))) if event_indices else 4)
    pid_rows: list[tuple[int, list[int]]] = []

    for event_index in event_indices:
        out_path = output_dir / f"{output_prefix}_{event_index:0{width}d}.{fmt}"
        dot_text, particles, edges, selected, selected_edge_count, written_path = render_single_event_graph(
            data,
            event_index=event_index,
            output_path=out_path,
            fmt=fmt,
            final_only=final_only,
            hard_only=hard_only,
            hard_descendants=hard_descendants,
            status_abs_min=status_abs_min,
            status_abs_max=status_abs_max,
            max_nodes=max_nodes,
            label_kinematics=label_kinematics,
            label_colors=label_colors,
            label_momentum=label_momentum,
            label_recoil_q=label_recoil_q,
            debug_recoil=debug_recoil,
            debug_particles=debug_particles,
            order_by_number=order_by_number,
            rankdir=rankdir,
        )

        final_hard_particles: list[dict] = []
        if final_state_output is not None:
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
                hard_outgoing = {int(p["no"]) for p in particles if abs(int(p["status"])) == 23}
                hard_desc = descendants_of(hard_outgoing, edges) if hard_outgoing else set(selected)
                final_hard_particles = sorted(
                    [
                        p for p in particles
                        if int(p["status"]) > 0 and int(p["no"]) in selected and int(p["no"]) in hard_desc
                    ],
                    key=lambda p: -int(p["no"]),
                )
            pid_rows.append((event_index, [int(p["pid"]) for p in final_hard_particles]))

        print(
            f"Rendered event {event_index}: {written_path} "
            f"({len(selected)}/{len(particles)} nodes, {selected_edge_count} selected edges; "
            f"{len(final_hard_particles)} final hard-descendant PIDs)"
        )

    if final_state_output is not None:
        save_final_state_hard_descendant_pid_rows_txt(pid_rows, final_state_output)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Draw a process tree/DAG from a parsed PYTHIA .pt file, with optional local-FSR recoiler labels."
    )
    parser.add_argument("ptfile", type=Path, help="Input .pt file from parse_pythia_log.py")
    parser.add_argument("--event", type=int, default=0, help="Event index to plot, starting from 0")
    parser.add_argument("--all-events", action="store_true", help="Render one diagram per selected event")
    parser.add_argument("--event-start", type=int, default=0, help="First event index for --all-events")
    parser.add_argument("--num-events", type=int, default=None, help="Number of events for --all-events. Default: all from --event-start")
    parser.add_argument("--events-output-dir", type=Path, default=None, help="Directory for per-event diagrams in --all-events mode")
    parser.add_argument("--event-output-prefix", type=str, default="event", help="Filename prefix for --all-events diagrams")
    parser.add_argument("--final-state-output", type=Path, default=None, help="Optional .txt output of final hard-descendant PIDs in visual top-to-bottom order")
    parser.add_argument("-o", "--output", type=Path, default=Path("event0.dot"), help="Single-event output path")
    parser.add_argument("--format", choices=["dot", "png", "pdf", "svg"], default=None, help="Output format. Default: inferred from output suffix, or dot")

    parser.add_argument("--final-only", action="store_true", help="Plot only particles with status > 0")
    parser.add_argument("--hard-only", action="store_true", help="Plot only particles with 20 <= |status| <= 29")
    parser.add_argument("--hard-descendants", action="store_true", help="Plot hard-process particles and all descendants")
    parser.add_argument("--status-abs-min", type=int, default=None, help="Keep only particles with |status| >= this")
    parser.add_argument("--status-abs-max", type=int, default=None, help="Keep only particles with |status| <= this")
    parser.add_argument("--max-nodes", type=int, default=None, help="Maximum number of nodes to draw")

    parser.add_argument("--no-kinematics", action="store_true", help="Do not include kinematic summaries in node labels")
    parser.add_argument("--label-colors", action="store_true", help="Include PYTHIA color tags in node labels")
    parser.add_argument("--label-momentum", action="store_true", help="Include extra momentum/off-shell info in node labels")
    parser.add_argument("--no-recoil-q-labels", action="store_true", help="Disable local-FSR recoiler Q,z edge labels")
    parser.add_argument("--debug-recoil-labels", action="store_true", help="Print debug kinematics while inferring recoiler labels")
    parser.add_argument("--no-debug-particles", action="store_true", help="Suppress debug print of (event,no,pid,status,daughters)")
    parser.add_argument("--no-order-by-number", action="store_true", help="Disable Graphviz ordering hints that put larger event-record numbers above smaller ones")
    parser.add_argument("--rankdir", choices=["LR", "TB", "RL", "BT"], default="LR", help="Graph direction")

    args = parser.parse_args(argv)

    data = torch.load(args.ptfile, map_location="cpu")
    if "mask" not in data or not torch.is_tensor(data["mask"]):
        raise ValueError("Input .pt file does not look like parser output: missing tensor key 'mask'.")

    num_events_total = int(data["mask"].shape[0])
    fmt = infer_graph_output_format(args.output, args.format)

    if args.all_events:
        if args.event_start < 0 or args.event_start >= num_events_total:
            raise IndexError(f"--event-start must be in [0,{num_events_total}); got {args.event_start}")
        end_event = num_events_total if args.num_events is None else min(num_events_total, args.event_start + args.num_events)
        event_indices = list(range(args.event_start, end_event))
        if not event_indices:
            raise RuntimeError("No events selected for --all-events.")

        events_output_dir = args.events_output_dir
        if events_output_dir is None:
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
            debug_recoil=args.debug_recoil_labels,
            debug_particles=not args.no_debug_particles,
            order_by_number=not args.no_order_by_number,
            rankdir=args.rankdir,
        )
        return

    if args.event < 0 or args.event >= num_events_total:
        raise IndexError(f"--event must be in [0,{num_events_total}); got {args.event}")

    dot_text, particles, edges, selected, selected_edge_count, written_path = render_single_event_graph(
        data,
        event_index=args.event,
        output_path=args.output,
        fmt=fmt,
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
        debug_recoil=args.debug_recoil_labels,
        debug_particles=not args.no_debug_particles,
        order_by_number=not args.no_order_by_number,
        rankdir=args.rankdir,
    )

    if fmt == "dot":
        print(f"Wrote DOT file: {written_path}")
    else:
        print(f"Wrote DOT file: {args.output.with_suffix('.dot')}")
        print(f"Rendered graph: {written_path}")
    print(f"Selected nodes: {len(selected)} / {len(particles)}")
    print(f"Selected edges: {selected_edge_count} / {len(edges)}")


if __name__ == "__main__":
    main()
