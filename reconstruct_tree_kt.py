#!/usr/bin/env python3
"""
Greedy reverse-shower reconstruction from parsed PYTHIA .pt files.

This script ignores PYTHIA's stored shower history and reconstructs a binary
branching tree from final-state particles only.  It repeatedly clusters the
globally closest pair in the e+e- kt/Durham distance, creates a parent with

    p_a = p_b + p_c,

and stores the remaining active particles as the spectators s_i for that
reverse step.

The reconstruction is heuristic.  It is useful for toy RHM-style experiments or
for building an ordered binary tree, not for recovering PYTHIA's exact hidden
branching history.

Examples
--------
# Reconstruct the first 100 events, stopping when two hard ancestors remain.
 /home/hiboy/miniforge3/bin/python   /home/hiboy/jet_interpretability_dataset/python/reconstruct_tree_kt.py     /home/hiboy/jet_interpretability_dataset/pythia_events/mymain01.pt     --num-events 1000     --hard-descendants     --stop-nodes 2     --output /home/hiboy/jet_interpretability_dataset/pythia_events/reconstructed_showers.pt     --json-output /home/hiboy/jet_interpretability_dataset/pythia_events/reconstructed_showers.json     --dot-dir /home/hiboy/jet_interpretability_dataset/pythia_events/reconstructed_dots     --format dot

# Build a single-root tree instead.
python reconstruct_shower_greedy.py events.pt --stop-nodes 1
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import random
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


def pdg_name(pid: int) -> str:
    return PDG_NAMES.get(int(pid), str(int(pid)))


def is_quark(pid: int) -> bool:
    return 1 <= abs(int(pid)) <= 6


def is_parton(pid: int) -> bool:
    """Return True for quarks, antiquarks, and gluons."""
    pid = int(pid)
    return pid == 21 or is_quark(pid)


def tensor_item(x):
    if torch.is_tensor(x):
        return x.item()
    return x


def get_event_field(data: dict, key: str, event_index: int):
    x = data[key]
    if torch.is_tensor(x):
        return x[event_index]
    return x[event_index]


def p4_tensor_from_particle(p: dict) -> torch.Tensor:
    if "p4" in p and torch.is_tensor(p["p4"]):
        return p["p4"].to(dtype=torch.float64)
    return torch.tensor([p["px"], p["py"], p["pz"], p["energy"]], dtype=torch.float64)


def add_p4(*vecs: torch.Tensor) -> torch.Tensor:
    if not vecs:
        return torch.zeros(4, dtype=torch.float64)
    return torch.stack([v.to(dtype=torch.float64) for v in vecs], dim=0).sum(dim=0)


def inv_mass2(p4: torch.Tensor) -> float:
    p4 = p4.to(dtype=torch.float64)
    return float((p4[3] * p4[3] - torch.dot(p4[:3], p4[:3])).item())


def pt_eta_phi(px: float, py: float, pz: float) -> tuple[float, float, float]:
    pt = math.hypot(px, py)
    phi = math.atan2(py, px)
    if pt == 0.0:
        eta = math.copysign(float("inf"), pz) if pz != 0 else 0.0
    else:
        eta = math.asinh(pz / pt)
    return pt, eta, phi


def collect_event_particles(data: dict, event_index: int) -> list[dict]:
    mask = get_event_field(data, "mask", event_index).bool()
    pid_key = "pid" if "pid" in data else "id"
    pid = get_event_field(data, pid_key, event_index).long()
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
        pt, eta, phi = pt_eta_phi(px, py, pz)
        name = str(names[j]) if names is not None and j < len(names) and names[j] else pdg_name(this_pid)
        p = {
            "array_index": j,
            "no": no,
            "pid": this_pid,
            "name": name,
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
        }
        if mass is not None:
            p["mass"] = float(tensor_item(mass[j]))
        else:
            p["mass"] = math.sqrt(max(inv_mass2(p4_vec), 0.0))
        if scale is not None:
            p["scale"] = float(tensor_item(scale[j]))
        else:
            p["scale"] = float("nan")
        if colors is not None:
            p["color1"] = int(colors[j, 0].item())
            p["color2"] = int(colors[j, 1].item())
        particles.append(p)
    return particles


def build_edges(particles: list[dict]) -> list[tuple[int, int]]:
    existing = {p["no"] for p in particles}
    edges: set[tuple[int, int]] = set()
    for p in particles:
        child = p["no"]
        for mother in sorted({p.get("mother1", 0), p.get("mother2", 0)}):
            if mother > 0 and mother in existing and mother != child:
                edges.add((mother, child))
    return sorted(edges)


def descendants_of(nodes: set[int], edges: list[tuple[int, int]]) -> set[int]:
    children: dict[int, list[int]] = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
    seen = set(nodes)
    frontier = list(nodes)
    while frontier:
        n = frontier.pop()
        for child in children.get(n, []):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return seen




def make_event_record_dot_for_order(
    particles: list[dict],
    edges: list[tuple[int, int]],
    selected: set[int],
    *,
    event_index: int,
    rankdir: str = "LR",
) -> str:
    """Build a lightweight event-record DOT graph for extracting Graphviz y positions.

    This is not the reconstructed shower tree. It is the original PYTHIA event
    record graph, rendered with the same sibling ordering convention used in the
    tree-diagram script: outgoing edges are emitted with larger event-record
    numbers first and parents with multiple children get ordering=out.
    """
    by_no = {int(p["no"]): p for p in particles}
    selected_edges = [(a, b) for a, b in edges if a in selected and b in selected]

    lines: list[str] = []
    lines.append("digraph PythiaEventOrder {")
    lines.append(f'  label="PYTHIA event {event_index}";')
    lines.append("  labelloc=t;")
    lines.append(f"  rankdir={rankdir};")
    lines.append("  graph [fontsize=18, overlap=false, splines=true];")
    lines.append("  node [fontname=Helvetica, fontsize=10, margin=0.06];")
    lines.append("  edge [arrowsize=0.7];")
    lines.append("")

    for no in sorted(selected, reverse=True):
        p = by_no[no]
        label = f'{no}: {p.get("name", pdg_name(int(p.get("pid", 0))))}\\nstatus={int(p.get("status", 0))}'
        shape = "box" if int(p.get("status", 0)) > 0 else "ellipse"
        lines.append(f'  n{no} [label="{dot_escape(label)}", shape={shape}];')

    children_by_parent: dict[int, list[int]] = {}
    for a, b in selected_edges:
        children_by_parent.setdefault(a, []).append(b)

    for a in sorted(children_by_parent):
        children = sorted(children_by_parent[a], reverse=True)
        if len(children) >= 2:
            lines.append(f"  n{a} [ordering=out];")
        for b in children:
            lines.append(f"  n{a} -> n{b};")

    lines.append("}")
    return "\n".join(lines) + "\n"

def selected_nodes_for_tree_order(
    particles: list[dict],
    edges: list[tuple[int, int]],
    *,
    hard_descendants: bool,
) -> set[int]:
    """Nodes to draw when extracting the event-record top-to-bottom order."""
    all_nodes = {int(p["no"]) for p in particles}
    if not hard_descendants:
        return all_nodes

    hard_outgoing = {int(p["no"]) for p in particles if abs(int(p.get("status", 0))) == 23}
    if not hard_outgoing:
        return all_nodes
    return descendants_of(hard_outgoing, edges)

def tree_relative_order_for_final_particles(
    particles: list[dict],
    edges: list[tuple[int, int]],
    selected: set[int],
    final_particles: list[dict],
    *,
    hard_descendants: bool,
) -> list[dict]:
    """Order final particles by their position in the event-record tree.

    This avoids asking Graphviz for rendered y-coordinates.  Instead, it walks
    the selected PYTHIA event-record graph top-to-bottom using the same sibling
    convention used in the DOT output: larger event-record numbers are visited
    first, so they appear above lower-numbered siblings when Graphviz respects
    ``ordering=out``.

    For ``--hard-descendants``, the traversal starts from outgoing hard-process
    particles with ``|status| == 23``.  Otherwise it starts from selected nodes
    with no selected parent.  Any final particles not reached by this traversal
    are appended deterministically at the end.
    """
    selected = {int(n) for n in selected}
    final_by_no = {int(p["no"]): p for p in final_particles}
    final_nos = set(final_by_no)

    children_by_parent: dict[int, list[int]] = {}
    parents_by_child: dict[int, list[int]] = {n: [] for n in selected}
    for a, b in edges:
        a = int(a)
        b = int(b)
        if a in selected and b in selected:
            children_by_parent.setdefault(a, []).append(b)
            parents_by_child.setdefault(b, []).append(a)

    for a in list(children_by_parent):
        # Same convention as make_dot(...): larger event-record number first.
        children_by_parent[a] = sorted(set(children_by_parent[a]), reverse=True)

    if hard_descendants:
        roots = [
            int(p["no"])
            for p in particles
            if int(p["no"]) in selected and abs(int(p.get("status", 0))) == 23
        ]
    else:
        roots = [
            n for n in selected
            if not any(parent in selected for parent in parents_by_child.get(n, []))
        ]

    if not roots:
        roots = list(selected)

    # Larger event-record roots first, matching the node/edge emission order in
    # the tree-diagram script.
    roots = sorted(set(roots), reverse=True)

    ordered_final_nos: list[int] = []
    visited: set[int] = set()

    def dfs(n: int) -> None:
        if n in visited or n not in selected:
            return
        visited.add(n)
        if n in final_nos:
            ordered_final_nos.append(n)
        for child in children_by_parent.get(n, []):
            dfs(child)

    for root in roots:
        dfs(root)

    # If a selected final particle was not reachable from the chosen roots
    # because the selected graph is disconnected, keep it rather than dropping it.
    missing = sorted(final_nos - set(ordered_final_nos), reverse=True)
    ordered_final_nos.extend(missing)

    rank = {no: i for i, no in enumerate(ordered_final_nos)}
    return sorted(
        final_particles,
        key=lambda p: (rank.get(int(p["no"]), len(rank)), int(p["no"])),
    )

def order_final_particles(
    particles: list[dict],
    final_particles: list[dict],
    *,
    order: str,
    event_index: int,
    hard_descendants: bool,
) -> list[dict]:
    """Order final particles before converting them to clustering leaves.

    ``order='tree'`` now means tree-structure order, not Graphviz-rendered
    y-coordinate order.  It traverses the PYTHIA event-record tree with the same
    sibling convention used when drawing the DOT graph: larger child line number
    first, lower child line number second.
    """
    # print("Event", event_index)
    # print([(p["no"], p["pid"], p["status"], p.get("daughters")) for p in final_particles])
    if order != "tree":
        return final_particles

    edges = build_edges(particles)
    selected = selected_nodes_for_tree_order(
        particles,
        edges,
        hard_descendants=hard_descendants,
    )
    return tree_relative_order_for_final_particles(
        particles,
        edges,
        selected,
        final_particles,
        hard_descendants=hard_descendants,
    )

def p4_to_list(p4: torch.Tensor) -> list[float]:
    return [float(x) for x in p4.to(dtype=torch.float64).tolist()]


def node_from_particle(p: dict) -> dict:
    p4 = p4_tensor_from_particle(p)
    px, py, pz, energy = [float(v) for v in p4.tolist()]
    pt, eta, phi = pt_eta_phi(px, py, pz)
    out = {
        "node_id": int(p["no"]),
        "pid": int(p["pid"]),
        "name": p.get("name", pdg_name(int(p["pid"]))),
        "p4": p4,
        "p4_list": p4_to_list(p4),
        "mass2": inv_mass2(p4),
        "mass": float(p.get("mass", math.sqrt(max(inv_mass2(p4), 0.0)))),
        "energy": energy,
        "px": px,
        "py": py,
        "pz": pz,
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "scale": float(p.get("scale", float("nan"))),
        "is_leaf": True,
        "leaf_no": int(p["no"]),
        "leaves": [int(p["no"])],
        "status": int(p.get("status", 0)),
    }
    if "color1" in p:
        out["color1"] = int(p["color1"])
        out["color2"] = int(p["color2"])
    return out


def infer_parent_pid(pid1: int, pid2: int) -> tuple[int, str, bool]:
    """Return (parent_pid, branching_class, is_qcd_like)."""
    pid1 = int(pid1)
    pid2 = int(pid2)
    if pid1 == 21 and pid2 == 21:
        return 21, "g_to_gg", True
    if pid1 == 21 and is_quark(pid2):
        return pid2, "q_to_qg", True
    if pid2 == 21 and is_quark(pid1):
        return pid1, "q_to_qg", True
    if is_quark(pid1) and pid2 == -pid1:
        return 21, "g_to_qqbar", True
    return 0, "unknown", False


def opening_angle_cos(p4a: torch.Tensor, p4b: torch.Tensor) -> float:
    va = p4a[:3].to(dtype=torch.float64)
    vb = p4b[:3].to(dtype=torch.float64)
    na = float(torch.linalg.vector_norm(va).item())
    nb = float(torch.linalg.vector_norm(vb).item())
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    cos = float((torch.dot(va, vb) / (na * nb)).item())
    return max(-1.0, min(1.0, cos))



def boost_to_rest_frame(p4: torch.Tensor, total_p4: torch.Tensor) -> torch.Tensor:
    """Boost p4 into the rest frame of total_p4. Convention: (px, py, pz, E)."""
    p4 = p4.to(dtype=torch.float64)
    total_p4 = total_p4.to(dtype=torch.float64)
    total_e = total_p4[3]
    if abs(float(total_e.item())) < 1e-15:
        return p4.clone()

    beta = total_p4[:3] / total_e
    beta2 = float(torch.dot(beta, beta).item())
    if beta2 < 1e-30:
        return p4.clone()
    if beta2 >= 1.0:
        return p4.clone()

    gamma = 1.0 / math.sqrt(1.0 - beta2)
    p_vec = p4[:3]
    e = p4[3]
    beta_dot_p = torch.dot(beta, p_vec)

    e_prime = gamma * (e - beta_dot_p)
    p_prime = p_vec + (((gamma - 1.0) * beta_dot_p / beta2) - gamma * e) * beta
    return torch.cat([p_prime, e_prime.reshape(1)])


def spatial_pt2_relative_to_axis(p4_star: torch.Tensor, axis_star: torch.Tensor) -> float:
    """Return |p x nhat|^2 for p relative to axis, using spatial parts only."""
    p_vec = p4_star[:3].to(dtype=torch.float64)
    axis = axis_star[:3].to(dtype=torch.float64)
    axis_norm = float(torch.linalg.vector_norm(axis).item())
    if axis_norm <= 1e-30:
        return float("nan")
    nhat = axis / axis_norm
    p_parallel = torch.dot(p_vec, nhat) * nhat
    p_perp = p_vec - p_parallel
    return float(torch.dot(p_perp, p_perp).item())


def parent_mass2_guess(parent_pid: int, left: dict, right: dict) -> float:
    """Best available on-shell parent mass^2 guess from daughter identities."""
    parent_pid = int(parent_pid)
    if parent_pid == int(left.get("pid", 0)):
        return max(float(left.get("mass2", inv_mass2(left["p4"]))), 0.0)
    if parent_pid == int(right.get("pid", 0)):
        return max(float(right.get("mass2", inv_mass2(right["p4"]))), 0.0)
    if parent_pid == 21:
        return 0.0
    return 0.0


def choose_b_c_for_z(parent_pid: int, left: dict, right: dict) -> tuple[dict, dict]:
    """Pick b,c ordering for z. For q->qg, b is the same-flavour quark."""
    parent_pid = int(parent_pid)
    if parent_pid != 21:
        if int(left.get("pid", 0)) == parent_pid:
            return left, right
        if int(right.get("pid", 0)) == parent_pid:
            return right, left
    return left, right


def update_node_p4_state(
    n: dict,
    new_p4: torch.Tensor,
    *,
    note: str = "",
    scale: Optional[float] = None,
) -> dict:
    """Return an active-state copy of node n with updated four-momentum."""
    out = dict(n)
    new_p4 = new_p4.to(dtype=torch.float64).clone()
    px, py, pz, energy = [float(v) for v in new_p4.tolist()]
    pt, eta, phi = pt_eta_phi(px, py, pz)
    out.update({
        "p4": new_p4,
        "p4_list": p4_to_list(new_p4),
        "mass2": inv_mass2(new_p4),
        "mass": math.sqrt(max(inv_mass2(new_p4), 0.0)),
        "energy": energy,
        "px": px,
        "py": py,
        "pz": pz,
        "pt": pt,
        "eta": eta,
        "phi": phi,
    })
    if scale is not None:
        out["scale"] = float(scale)
    if note:
        out["recoil_update_note"] = note
    return out

def order_final_nodes(nodes: list[dict], order: str) -> list[dict]:
    if order == "no":
        return sorted(nodes, key=lambda n: int(n.get("leaf_no", n["node_id"])))
    if order == "no_desc":
        return sorted(nodes, key=lambda n: int(n.get("leaf_no", n["node_id"])), reverse=True)
    if order == "eta":
        def eta(n: dict) -> float:
            px, py, pz, _ = p4_to_list(n["p4"])
            _, e, _ = pt_eta_phi(px, py, pz)
            return e
        return sorted(nodes, key=eta)
    if order == "pt_desc":
        return sorted(nodes, key=lambda n: math.hypot(float(n["p4"][0]), float(n["p4"][1])), reverse=True)
    if order == "input":
        return list(nodes)
    raise ValueError(f"Unknown order={order!r}")


def select_final_particles(
    particles: list[dict],
    *,
    hard_descendants: bool,
    exclude_neutrinos: bool,
) -> list[dict]:
    final = [p for p in particles if int(p["status"]) > 0]
    if exclude_neutrinos:
        final = [p for p in final if abs(int(p["pid"])) not in {12, 14, 16}]

    if hard_descendants:
        edges = build_edges(particles)
        hard_outgoing = {p["no"] for p in particles if abs(int(p["status"])) == 23}
        if hard_outgoing:
            hard_desc = descendants_of(hard_outgoing, edges)
            final = [p for p in final if p["no"] in hard_desc]
    return final


def reconstruct_event(
    data: dict,
    event_index: int,
    *,
    stop_nodes: int = 2,
    order: str = "no",
    metric: str = "durham",
    require_qcd: bool = False,
    invalid_penalty: float = 1e12,
    hard_descendants: bool = False,
    exclude_neutrinos: bool = False,
) -> dict:
    particles = collect_event_particles(data, event_index)
    final_particles = select_final_particles(
        particles,
        hard_descendants=hard_descendants,
        exclude_neutrinos=exclude_neutrinos,
    )
    final_particles = order_final_particles(
        particles,
        final_particles,
        order=order,
        event_index=event_index,
        hard_descendants=hard_descendants,
    )
    leaves = [node_from_particle(p) for p in final_particles]
    active0 = list(leaves) if order == "tree" else order_final_nodes(leaves, order)
    active = [update_node_p4_state(n, n["p4"], note="initial active state") for n in active0]
    initial_leaf_nodes_ordered = [int(n["node_id"]) for n in active]
    initial_leaf_pids_ordered = [int(n["pid"]) for n in active]

    if stop_nodes < 1:
        raise ValueError("stop_nodes must be >= 1.")
    if len(active) <= stop_nodes:
        return {
            "event": int(event_index),
            "initial_num_leaves": len(active),
            "stop_nodes": int(stop_nodes),
            "order": order,
            "metric": metric,
            "nodes": [serializable_node(n) for n in active],
            "branchings": [],
            "initial_leaf_nodes": initial_leaf_nodes_ordered,
            "initial_leaf_pids": initial_leaf_pids_ordered,
            "roots": [int(n["node_id"]) for n in active],
            "root_pids": [int(n["pid"]) for n in active],
        }

    nodes_by_id: dict[int, dict] = {int(n["node_id"]): n for n in active}
    branchings: list[dict] = []
    step = 0

    while len(active) > stop_nodes:
        # FastJet-like e+e- k_t clustering: scan every unordered pair in the
        # current active list, not just adjacent particles in the PYTHIA tree
        # ordering.  The default metric is the Durham/e+e- k_t distance
        #     d_ij = 2 min(E_i^2, E_j^2) (1 - cos theta_ij),
        # which is the distance FastJet's ee_kt_algorithm reports as dij.
        candidates: list[tuple[float, int, int, dict]] = []
        for cand_i in range(len(active)):
            for cand_j in range(cand_i + 1, len(active)):
                left = active[cand_i]
                right = active[cand_j]
                parent_pid, branch_class, is_qcd_like = infer_parent_pid(left["pid"], right["pid"])
                if require_qcd and not is_qcd_like:
                    continue

                pb = left["p4"].to(dtype=torch.float64)
                pc = right["p4"].to(dtype=torch.float64)
                cos_theta = opening_angle_cos(pb, pc)
                one_minus_cos = max(0.0, 1.0 - cos_theta)
                eb = max(float(pb[3].item()), 0.0)
                ec = max(float(pc[3].item()), 0.0)
                e_sum = eb + ec
                m2 = inv_mass2(add_p4(pb, pc))
                durham_kt2 = 2.0 * min(eb * eb, ec * ec) * one_minus_cos
                angle_score = one_minus_cos
                soft_fraction = min(eb, ec) / e_sum if e_sum > 0.0 else float("inf")

                if metric == "durham":
                    score = durham_kt2
                elif metric == "mass":
                    score = max(m2, 0.0)
                elif metric == "angle":
                    score = angle_score
                elif metric == "soft":
                    score = soft_fraction
                else:
                    raise ValueError(f"Unknown metric={metric!r}")

                # FastJet ignores flavour/PID labels.  Therefore, by default,
                # non-QCD-like PID combinations are NOT penalized.  The only
                # PID-based restriction left is the explicit --require-qcd flag.
                effective_score = float(score)
                z_lab = eb / e_sum if e_sum > 0.0 else float("nan")

                record = {
                    "score": float(score),
                    "metric": metric,
                    "durham_kt2": float(durham_kt2),
                    "dij": float(durham_kt2),
                    "invariant_mass2": float(m2),
                    "cos_theta": float(cos_theta),
                    "one_minus_cos": float(one_minus_cos),
                    "soft_fraction": float(soft_fraction),
                    "z_lab_left": float(z_lab),
                    "left_node": int(left["node_id"]),
                    "right_node": int(right["node_id"]),
                    "left_pid": int(left["pid"]),
                    "right_pid": int(right["pid"]),
                    "parent_pid": int(parent_pid),
                    "branching_class": branch_class,
                    "is_qcd_like": bool(is_qcd_like),
                    "effective_score": float(effective_score),
                }
                candidates.append((float(effective_score), cand_i, cand_j, record))

        if not candidates:
            raise RuntimeError(
                "No valid pairs found. If you used --require-qcd, try removing it; "
                "FastJet's ee_kt_algorithm itself does not impose flavour/QCD labels."
            )

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, i, j, pair_record = candidates[0]

        left = active[i]
        right = active[j]
        parent_pid = int(pair_record["parent_pid"])
        parent_p4 = add_p4(left["p4"], right["p4"])

        parent_id = -(step + 1)
        spectators = [int(n["node_id"]) for k, n in enumerate(active) if k not in {i, j}]
        spectator_pids = [int(n["pid"]) for k, n in enumerate(active) if k not in {i, j}]
        spectator_p4 = [p4_to_list(n["p4"]) for k, n in enumerate(active) if k not in {i, j}]

        durham_scale = math.sqrt(max(float(pair_record.get("durham_kt2", float("nan"))), 0.0))
        left["scale"] = durham_scale
        right["scale"] = durham_scale
        if int(left["node_id"]) in nodes_by_id:
            nodes_by_id[int(left["node_id"])] = dict(nodes_by_id[int(left["node_id"])], scale=durham_scale)
        if int(right["node_id"]) in nodes_by_id:
            nodes_by_id[int(right["node_id"])] = dict(nodes_by_id[int(right["node_id"])], scale=durham_scale)

        # if step == 0:
            # print("Left node:", left["node_id"], pdg_name(left["pid"]), "p4=", p4_to_list(left["p4"]))
            # print("Right node:", right["node_id"], pdg_name(right["pid"]), "p4=", p4_to_list(right["p4"]))
            # print("Parent pid:", parent_pid, "name:", pdg_name(parent_pid), "p4=", p4_to_list(parent_p4))

        px, py, pz, energy = [float(v) for v in parent_p4.tolist()]
        pt, eta, phi = pt_eta_phi(px, py, pz)
        parent_node = {
            "node_id": parent_id,
            "pid": parent_pid,
            "name": pdg_name(parent_pid) if parent_pid != 0 else "cluster",
            "p4": parent_p4,
            "p4_list": p4_to_list(parent_p4),
            "mass2": inv_mass2(parent_p4),
            "mass": math.sqrt(max(inv_mass2(parent_p4), 0.0)),
            "energy": energy,
            "px": px,
            "py": py,
            "pz": pz,
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "scale": durham_scale,
            "dij": float(pair_record.get("durham_kt2", float("nan"))),
            "is_leaf": False,
            "children": [int(left["node_id"]), int(right["node_id"])],
            "leaves": list(left.get("leaves", [])) + list(right.get("leaves", [])),
            "step_created": step,
        }
        nodes_by_id[parent_id] = parent_node

        # FastJet-style active-list update: remove the two merged objects and
        # append the new cluster.  Since all pairs are considered at every
        # step, the active ordering only affects deterministic tie-breaking.
        new_active = [n for k, n in enumerate(active) if k not in {i, j}]
        new_active.append(parent_node)

        branch_record = {
            "step": int(step),
            "parent_node": int(parent_id),
            "parent_pid": int(parent_pid),
            "branching_class": pair_record["branching_class"],
            "is_qcd_like": bool(pair_record["is_qcd_like"]),
            "b_node": int(left["node_id"]),
            "c_node": int(right["node_id"]),
            "b_pid": int(left["pid"]),
            "c_pid": int(right["pid"]),
            "p_a": p4_to_list(parent_p4),
            "p_b": p4_to_list(left["p4"]),
            "p_c": p4_to_list(right["p4"]),
            "scale_inferred_from_pair": durham_scale,
            "spectator_nodes": spectators,
            "spectator_pids": spectator_pids,
            "spectator_p4": spectator_p4,
            "active_before": [int(n["node_id"]) for n in active],
            "active_after": [int(n["node_id"]) for n in new_active],
            **pair_record,
        }
        branchings.append(branch_record)

        active = new_active
        step += 1

    return {
        "event": int(event_index),
        "initial_num_leaves": len(leaves),
        "final_num_roots": len(active),
        "stop_nodes": int(stop_nodes),
        "order": order,
        "metric": metric,
        "require_qcd": bool(require_qcd),
        "hard_descendants": bool(hard_descendants),
        "initial_leaf_nodes": initial_leaf_nodes_ordered,
        "initial_leaf_pids": initial_leaf_pids_ordered,
        "roots": [int(n["node_id"]) for n in active],
        "root_pids": [int(n["pid"]) for n in active],
        "nodes": [serializable_node(n) for _, n in sorted(nodes_by_id.items(), key=lambda kv: kv[0])],
        "branchings": branchings,
    }

def serializable_node(n: dict) -> dict:
    out: dict = {}
    for key, val in n.items():
        if key == "p4":
            out["p4"] = p4_to_list(val)
        elif torch.is_tensor(val):
            out[key] = p4_to_list(val) if val.ndim == 1 else val.tolist()
        else:
            out[key] = val
    return out


def dot_escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def dot_node_id(node_id: int) -> str:
    return f"n{'m' + str(abs(node_id)) if node_id < 0 else str(node_id)}"

def reconstructed_node_label(n: dict) -> str:
    """Node label matching make_fastjet_ee_kt_trees_from_pt.py format, plus particle type.

    Convention: p4 = (px, py, pz, E).
    """
    pid = int(n.get("pid", 0))
    name = n.get("name", pdg_name(pid))

    px = float(n.get("px", n.get("p4", [float("nan"), 0.0, 0.0, 0.0])[0]))
    py = float(n.get("py", n.get("p4", [0.0, float("nan"), 0.0, 0.0])[1]))
    pz = float(n.get("pz", n.get("p4", [0.0, 0.0, float("nan"), 0.0])[2]))
    energy = float(n.get("energy", n.get("p4", [0.0, 0.0, 0.0, float("nan")])[3]))

    lines = [
        f"type={name}, pid={pid}",
        f"px={px:.3f}",
        f"py={py:.3f}",
        f"pz={pz:.3f}",
        f"E={energy:.3f}",
    ]

    if not n.get("is_leaf", False):
        dij = float(n.get("dij", n.get("durham_kt2", float("nan"))))
        lines.append(f"dij={dij:.3g}")

    return "\n".join(lines)

def reconstruction_to_dot(reco: dict, *, rankdir: str = "LR") -> str:
    nodes = {int(n["node_id"]): n for n in reco["nodes"]}
    branchings = sorted(reco["branchings"], key=lambda br: int(br.get("step", 0)))
    root_ids = [int(x) for x in reco.get("roots", [])]

    lines: list[str] = []
    lines.append("digraph ReconstructedShower {")
    lines.append(f'  label="Python global e+e- kt tree, event {reco["event"]}";')
    lines.append("  labelloc=t;")
    lines.append(f"  rankdir={rankdir};")
    lines.append("  graph [fontsize=18, overlap=false, splines=true];")
    lines.append("  node [fontname=Helvetica, fontsize=10, margin=0.06];")
    lines.append("  edge [arrowsize=0.7];")
    lines.append("")

    for node_id, n in sorted(nodes.items()):
        shape = "box" if n.get("is_leaf", False) else "ellipse"
        if node_id in root_ids:
            shape = "doublecircle"
        label = reconstructed_node_label(n)
        lines.append(f'  {dot_node_id(node_id)} [label="{dot_escape(label)}", shape={shape}];')

    lines.append("")
    for br in branchings:
        parent = int(br["parent_node"])
        b = int(br["b_node"])
        c = int(br["c_node"])
        lines.append(f'  {dot_node_id(parent)} -> {dot_node_id(b)};')
        lines.append(f'  {dot_node_id(parent)} -> {dot_node_id(c)};')

    lines.append("}")
    return "\n".join(lines) + "\n"

def render_dot(dot_text: str, output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path = output_path.with_suffix(".dot")
    dot_path.write_text(dot_text, encoding="utf-8")
    if fmt == "dot":
        return
    dot_exe = shutil.which("dot")
    if dot_exe is None:
        raise RuntimeError("Graphviz 'dot' executable not found; cannot render image/PDF/SVG.")
    render_path = output_path.with_suffix(f".{fmt}")
    subprocess.run([dot_exe, f"-T{fmt}", str(dot_path), "-o", str(render_path)], check=True)


def event_indices_from_args(data: dict, args: argparse.Namespace) -> list[int]:
    num_events_total = int(data["mask"].shape[0])
    if args.event is not None:
        if args.event < 0 or args.event >= num_events_total:
            raise IndexError(f"--event must be in [0,{num_events_total}); got {args.event}")
        return [int(args.event)]
    start = max(0, int(args.start_event))
    if start >= num_events_total:
        return []
    stop = num_events_total if args.num_events is None else min(num_events_total, start + int(args.num_events))
    return list(range(start, stop))



def pythia_initial_parton_pids(data: dict, event_index: int) -> list[int]:
    """Return PYTHIA hard-process outgoing parton PIDs for comparison.

    In the parsed event record, the hard outgoing particles are identified by
    |status| == 23.  We keep only QCD partons, i.e. quarks/antiquarks/gluons.
    The returned list is sorted so the comparison is independent of left/right
    ordering in the tree.
    """
    particles = collect_event_particles(data, event_index)
    pids = [
        int(p["pid"])
        for p in particles
        if abs(int(p.get("status", 0))) == 23 and is_parton(int(p.get("pid", 0)))
    ]
    return sorted(pids)


def compare_reco_roots_to_pythia_initial_partons(
    data: dict,
    reconstructions: list[dict],
) -> dict:
    """Compare reconstructed root PIDs to PYTHIA's hard outgoing parton PIDs."""
    matches: list[int] = []
    mismatches: list[dict] = []

    for reco in reconstructions:
        event_index = int(reco["event"])
        kt_roots = sorted(int(x) for x in reco.get("root_pids", []))
        pythia_initial = pythia_initial_parton_pids(data, event_index)

        if kt_roots == pythia_initial:
            matches.append(event_index)
        else:
            mismatches.append({
                "event": event_index,
                "kt_root_pids": kt_roots,
                "pythia_initial_parton_pids": pythia_initial,
            })

    total = len(reconstructions)
    n_match = len(matches)
    percent = 100.0 * n_match / total if total else float("nan")
    return {
        "total_events": total,
        "num_matches": n_match,
        "percentage": percent,
        "matching_event_indices": matches,
        "mismatches": mismatches,
        "mismatch_event_indices": [int(m["event"]) for m in mismatches],
    }


def save_txt_summary(reconstructions: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for reco in reconstructions:
            event = int(reco["event"])
            leaves = " ".join(str(x) for x in reco.get("initial_leaf_pids", []))
            roots = " ".join(str(x) for x in reco.get("root_pids", []))
            f.write(f"event {event}\n")
            f.write(f"  leaves: {leaves}\n")
            f.write(f"  roots:  {roots}\n")
            for br in reco.get("branchings", []):
                f.write(
                    "  step {step}: {parent_pid} -> {b_pid} {c_pid}, "
                    "nodes {parent_node}->{b_node},{c_node}, "
                    "class={branching_class}, kt2={durham_kt2:.6g}, "
                    "m2={invariant_mass2:.6g}, score={score:.6g}, "
                    "soft_fraction={soft_fraction:.6g}\n".format(**br)
                )
            f.write("\n")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Greedily reconstruct a reverse-shower tree from final-state particles in a parsed PYTHIA .pt file."
    )
    parser.add_argument("ptfile", type=Path, help="Input .pt file from parse_pythia_log.py")
    parser.add_argument("--event", type=int, default=None, help="Single event index to reconstruct")
    parser.add_argument("--start-event", type=int, default=0, help="First event index when processing many events")
    parser.add_argument("--num-events", type=int, default=None, help="Number of events to process. Default: all from start-event")
    parser.add_argument("--output", type=Path, default=Path("reconstructed_showers.pt"), help="Output .pt file")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional readable JSON output")
    parser.add_argument("--txt-output", type=Path, default=None, help="Optional compact text summary")
    parser.add_argument("--dot-dir", type=Path, default=None, help="Optional directory for reconstructed-tree DOT/images")
    parser.add_argument("--format", choices=["dot", "png", "pdf", "svg"], default="dot", help="Graphviz output format for --dot-dir")
    parser.add_argument("--rankdir", choices=["LR", "TB", "RL", "BT"], default="LR", help="Direction for reconstructed-tree diagrams. LR means left-to-right; TB means top-to-bottom.")
    parser.add_argument("--stop-nodes", type=int, default=2, help="Stop clustering when this many roots remain. Use 1 for a single-root tree.")
    parser.add_argument("--order", choices=["no", "no_desc", "input", "eta", "pt_desc", "tree"], default="no", help="Initial ordering of final particles; global kT clustering considers all pairs, so this only affects deterministic tie-breaking and output leaf order. 'tree' orders final particles by their top-to-bottom position in the PYTHIA event-record tree.")
    parser.add_argument("--metric", choices=["durham", "mass", "angle", "soft"], default="durham", help="Soft/collinear score for adjacent pair selection")
    parser.add_argument("--require-qcd", action="store_true", help="Only merge QCD-like inverse branchings: qg->q, gg->g, qqbar->g")
    parser.add_argument("--invalid-penalty", type=float, default=1e12, help="Penalty added to non-QCD-like merges when --require-qcd is not used")
    parser.add_argument("--hard-descendants", action="store_true", help="Use only final particles descended from |status|=23 hard-process outgoing particles")
    parser.add_argument("--exclude-neutrinos", action="store_true", help="Drop neutrinos from the final-state list")
    args = parser.parse_args(argv)

    data = torch.load(args.ptfile, map_location="cpu")
    if "mask" not in data:
        raise ValueError("Input file does not look like parser output: missing key 'mask'.")

    event_indices = event_indices_from_args(data, args)
    if not event_indices:
        raise RuntimeError("No events selected.")

    reconstructions: list[dict] = []
    for event_index in event_indices:
        reco = reconstruct_event(
            data,
            event_index,
            stop_nodes=args.stop_nodes,
            order=args.order,
            metric=args.metric,
            require_qcd=args.require_qcd,
            invalid_penalty=args.invalid_penalty,
            hard_descendants=args.hard_descendants,
            exclude_neutrinos=args.exclude_neutrinos,
        )
        reconstructions.append(reco)
        print(
            f"event {event_index}: leaves={reco['initial_num_leaves']}, "
            f"branchings={len(reco['branchings'])}, roots={reco.get('root_pids', [])}"
        )

        if args.dot_dir is not None:
            width = max(4, len(str(max(event_indices))))
            dot_text = reconstruction_to_dot(reco, rankdir=args.rankdir)
            out_path = args.dot_dir / f"reco_event_{event_index:0{width}d}.{args.format}"
            render_dot(dot_text, out_path, args.format)

    initial_parton_comparison = compare_reco_roots_to_pythia_initial_partons(
        data,
        reconstructions,
    )
    print("\nInitial-parton comparison")
    print("-------------------------")
    print(
        "kT roots match PYTHIA hard outgoing partons for "
        f"{initial_parton_comparison['num_matches']}/"
        f"{initial_parton_comparison['total_events']} events "
        f"({initial_parton_comparison['percentage']:.2f}%)."
    )
    print(
        "Mismatch event indices: "
        f"{initial_parton_comparison['mismatch_event_indices']}"
    )
    if initial_parton_comparison["mismatches"]:
        print("Mismatch details:")
        for mismatch in initial_parton_comparison["mismatches"]:
            print(
                f"  event {mismatch['event']}: "
                f"kT roots={mismatch['kt_root_pids']}, "
                f"PYTHIA hard partons={mismatch['pythia_initial_parton_pids']}"
            )

    payload = {
        "reconstructions": reconstructions,
        "metadata": {
            "input_ptfile": str(args.ptfile),
            "event_indices": event_indices,
            "stop_nodes": args.stop_nodes,
            "order": args.order,
            "metric": args.metric,
            "require_qcd": args.require_qcd,
            "hard_descendants": args.hard_descendants,
            "initial_parton_comparison": initial_parton_comparison,
            "note": (
                "Greedy global-pair reverse clustering. At each step, "
                "the script merges the globally closest pair with the smallest selected score; "
                "by default this is the Durham kT score. No p_evol monotonicity, "
                "backtracking, or recoil update is used."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved reconstruction payload: {args.output}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved readable JSON: {args.json_output}")

    if args.txt_output is not None:
        save_txt_summary(reconstructions, args.txt_output)
        print(f"Saved text summary: {args.txt_output}")


if __name__ == "__main__":
    main()
