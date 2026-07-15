#!/usr/bin/env python3
"""
Parse a PYTHIA 8 event listing/log into PyTorch tensors.

This is designed for the standard PYTHIA 8 event table, whose header looks like

    no        id   name            status     mothers   daughters     colours      p_x        p_y        p_z         e          m

and whose rows look approximately like

     1       2212  (p+)              -12      0      0      3      0      0    101      0.000      0.000   6500.000   6500.000      0.938

The parser is intentionally tolerant of particle names containing spaces, e.g.
"(tbar)", "g", "(u)", etc. It extracts numerical columns by reading the first
2 tokens, the last 12 tokens, and treating the middle as the particle name.

Output is a .pt file containing padded tensors with shape

    [num_events, max_particles]

for scalar integer fields, and

    [num_events, max_particles, 4]

for four-momenta.

Example
-------
    python parse_pythia_event_log.py pythia.log -o events.pt

Then in Python:

    import torch
    data = torch.load("events.pt")
    ids = data["id"]              # LongTensor [E, Nmax]
    p4 = data["p4"]               # FloatTensor [E, Nmax, 4], default columns E,px,py,pz
    mask = data["mask"]           # BoolTensor [E, Nmax], true for real particles
    final = data["status"] > 0    # PYTHIA convention: final-state particles usually status > 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import torch


EVENT_HEADER_RE = re.compile(
    r"\bno\b\s+\bid\b\s+\bname\b\s+\bstatus\b\s+\bmothers\b\s+\bdaughters\b\s+\bcolou?rs\b",
    re.IGNORECASE,
)

# A line beginning with an integer particle number, followed somewhere by
# numerical particle content. This avoids parsing cross-section/log lines.
ROW_START_RE = re.compile(r"^\s*\d+\s+")


@dataclass
class Particle:
    no: int
    pid: int
    name: str
    status: int
    mother1: int
    mother2: int
    daughter1: int
    daughter2: int
    color1: int
    color2: int
    px: float
    py: float
    pz: float
    energy: float
    mass: float
    scale: float = -1.0


@dataclass
class Event:
    particles: list[Particle]
    event_index: int


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_float(s: str) -> bool:
    try:
        float(s.replace("D", "E").replace("d", "e"))
        return True
    except ValueError:
        return False


def _to_float(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))


def parse_particle_row(line: str) -> Optional[Particle]:
    if not ROW_START_RE.match(line):
        return None

    stripped = line.strip()
    if not stripped or stripped.startswith(("-", "=", "#", "|")):
        return None

    tokens = stripped.split()
    if len(tokens) < 15:
        return None

    if not (_is_int(tokens[0]) and _is_int(tokens[1])):
        return None

    # New custom format:
    # no id name status m1 m2 d1 d2 c1 c2 px py pz e m scale pTbeam
    has_scale = False
    if len(tokens) >= 17:
        candidate_tail = tokens[-14:]
        if (
            all(_is_int(x) for x in candidate_tail[:7])
            and all(_is_float(x) for x in candidate_tail[7:])
        ):
            tail = candidate_tail
            name_tokens = tokens[2:-14]
            has_scale = True
        else:
            tail = tokens[-12:]
            name_tokens = tokens[2:-12]
    else:
        tail = tokens[-12:]
        name_tokens = tokens[2:-12]

    # Old/default PYTHIA format:
    # no id name status m1 m2 d1 d2 c1 c2 px py pz e m
    if not has_scale:
        if not all(_is_int(x) for x in tail[:7]):
            return None
        if not all(_is_float(x) for x in tail[7:]):
            return None

    name = " ".join(name_tokens) if name_tokens else ""

    status = int(tail[0])
    mother1 = int(tail[1])
    mother2 = int(tail[2])
    daughter1 = int(tail[3])
    daughter2 = int(tail[4])
    color1 = int(tail[5])
    color2 = int(tail[6])
    px = _to_float(tail[7])
    py = _to_float(tail[8])
    pz = _to_float(tail[9])
    energy = _to_float(tail[10])
    mass = _to_float(tail[11])

    scale = _to_float(tail[12]) if has_scale else -1.0

    return Particle(
        no=int(tokens[0]),
        pid=int(tokens[1]),
        name=name,
        status=status,
        mother1=mother1,
        mother2=mother2,
        daughter1=daughter1,
        daughter2=daughter2,
        color1=color1,
        color2=color2,
        px=px,
        py=py,
        pz=pz,
        energy=energy,
        mass=mass,
        scale=scale,
    )

def parse_pythia_log(path: str | Path) -> list[Event]:
    """
    Parse only custom sections titled:
        PYTHIA Event Listing (complete event with scale)

    Ignores default PYTHIA hard-process and complete-event listings.
    """
    path = Path(path)
    events: list[Event] = []
    current_particles: list[Particle] = []

    in_wanted_section = False
    in_table = False

    def flush_event() -> None:
        nonlocal current_particles
        if current_particles:
            events.append(Event(particles=current_particles, event_index=len(events)))
            current_particles = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            lower = line.lower()

            # Start only the custom complete-event-with-scale block.
            if "pythia event listing" in lower and "complete event with scale" in lower:
                flush_event()
                in_wanted_section = True
                in_table = False
                continue

            # End the custom block.
            if "end pythia event listing" in lower and "complete event with scale" in lower:
                flush_event()
                in_wanted_section = False
                in_table = False
                continue

            # Ignore everything outside the custom block.
            if not in_wanted_section:
                continue

            # Inside custom block, wait for the custom column header.
            # This avoids accidentally parsing the title/border line.
            if (
                " no " in f" {lower} "
                and " id " in f" {lower} "
                and " status " in f" {lower} "
                and " scale " in f" {lower} "
                and "ptbeam" in lower
            ):
                in_table = True
                continue

            if not in_table:
                continue

            particle = parse_particle_row(line)
            if particle is not None:
                # Require real scale rows. This prevents old-format rows from
                # sneaking in with scale = -1.
                if particle.scale >= 0:
                    current_particles.append(particle)
                continue

    flush_event()
    return events



def _apply_no_permutation_to_particle(p: Particle, perm: dict[int, int]) -> Particle:
    """Return a copy of p with event-record line references relabelled by perm."""
    def map_no(x: int) -> int:
        return perm.get(x, x) if x > 0 else x

    return Particle(
        no=map_no(p.no),
        pid=p.pid,
        name=p.name,
        status=p.status,
        mother1=map_no(p.mother1),
        mother2=map_no(p.mother2),
        daughter1=map_no(p.daughter1),
        daughter2=map_no(p.daughter2),
        color1=p.color1,
        color2=p.color2,
        px=p.px,
        py=p.py,
        pz=p.pz,
        energy=p.energy,
        mass=p.mass,
        scale=p.scale,
    )


def randomize_branching_daughter_nos(
    events: list[Event],
    *,
    seed: Optional[int] = None,
    include_hard_process: bool = True,
) -> list[Event]:
    """Randomly relabel the two daughters of each two-daughter branching.

    For every unique pair (daughter1, daughter2) with daughter1 != daughter2,
    this composes a random transposition of the two event-record line numbers
    with probability 1/2.  The relabeling is applied consistently to:

      * Particle.no
      * mother1/mother2
      * daughter1/daughter2

    The particle identity and four-momentum stay attached to the same Particle
    object.  Only its event-record label `no` is changed.  This is useful when
    you want the higher/lower event-record number to be an artificial random
    ordering rather than PYTHIA's deterministic append order.

    If include_hard_process=False, rows with |status| in 20..29 are not used as
    sources of daughter pairs. This is a rough filter only; the relabelling is
    still global once a daughter pair is selected.
    """
    rng = random.Random(seed)
    randomized_events: list[Event] = []

    for event in events:
        original_nos = {p.no for p in event.particles}
        # Start with identity permutation on all event-record line numbers.
        perm: dict[int, int] = {no: no for no in original_nos}

        # Use unique daughter pairs so a 2->2 hard process with two mothers that
        # both list the same daughters is not flipped twice.
        daughter_pairs: list[tuple[int, int]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for p in event.particles:
            if not include_hard_process and 20 <= abs(p.status) <= 29:
                continue
            d1, d2 = p.daughter1, p.daughter2
            if d1 <= 0 or d2 <= 0 or d1 == d2:
                continue
            if d1 not in original_nos or d2 not in original_nos:
                continue
            key = (min(d1, d2), max(d1, d2))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            daughter_pairs.append((d1, d2))

        # Compose random transpositions in the original line-number labels.
        # Updating every current image of d1/d2 keeps perm a valid permutation.
        for d1, d2 in daughter_pairs:
            if rng.random() < 0.5:
                for old_no, current_no in list(perm.items()):
                    if current_no == d1:
                        perm[old_no] = d2
                    elif current_no == d2:
                        perm[old_no] = d1

        new_particles = [
            _apply_no_permutation_to_particle(p, perm)
            for p in event.particles
        ]
        new_particles.sort(key=lambda q: q.no)
        randomized_events.append(Event(particles=new_particles, event_index=event.event_index))

    return randomized_events

def events_to_tensors(
    events: list[Event],
    *,
    keep_status: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    metadata_extra: Optional[dict] = None,
    p4_only: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor | list[list[str]] | dict]:
    """
    Convert parsed events to padded PyTorch tensors.

    keep_status options:
        None      keep all particles
        "final"  keep particles with status > 0
        "hard"   keep particles with |status| in a rough PYTHIA hard-process range 20..29

    By default, the returned dict contains both tensors and names. Particle names
    are kept as nested Python lists because strings are not PyTorch tensor data.

    If p4_only=True, return only the padded four-momentum tensor with shape

        [num_events, max_particles, 4]

    and no metadata/mask/particle labels. Padding rows are all zeros.
    """
    if keep_status not in {None, "final", "hard"}:
        raise ValueError("keep_status must be one of: None, 'final', 'hard'")

    filtered: list[list[Particle]] = []
    for event in events:
        particles = event.particles
        if keep_status == "final":
            particles = [p for p in particles if p.status > 0]
        elif keep_status == "hard":
            particles = [p for p in particles if 20 <= abs(p.status) <= 29]
        filtered.append(particles)

    num_events = len(filtered)
    max_particles = max((len(ps) for ps in filtered), default=0)

    def long_pad(fill: int = 0) -> torch.Tensor:
        return torch.full((num_events, max_particles), fill, dtype=torch.long)

    no = long_pad(-1)
    pid = long_pad(0)
    status = long_pad(0)
    mothers = torch.zeros((num_events, max_particles, 2), dtype=torch.long)
    daughters = torch.zeros((num_events, max_particles, 2), dtype=torch.long)
    colors = torch.zeros((num_events, max_particles, 2), dtype=torch.long)
    p4 = torch.zeros((num_events, max_particles, 4), dtype=dtype)
    mass = torch.zeros((num_events, max_particles), dtype=dtype)
    mask = torch.zeros((num_events, max_particles), dtype=torch.bool)
    names: list[list[str]] = []
    scale = torch.full((num_events, max_particles), -1.0, dtype=dtype)

    for i, particles in enumerate(filtered):
        event_names: list[str] = []
        for j, p in enumerate(particles):
            no[i, j] = p.no
            pid[i, j] = p.pid
            status[i, j] = p.status
            mothers[i, j] = torch.tensor([p.mother1, p.mother2], dtype=torch.long)
            daughters[i, j] = torch.tensor([p.daughter1, p.daughter2], dtype=torch.long)
            colors[i, j] = torch.tensor([p.color1, p.color2], dtype=torch.long)
            p4[i, j] = torch.tensor([p.energy, p.px, p.py, p.pz], dtype=dtype)
            mass[i, j] = p.mass
            mask[i, j] = True
            scale[i, j] = p.scale
            event_names.append(p.name)
        names.append(event_names)

    p4_columns = ["E", "px", "py", "pz"]

    if p4_only:
        return p4

    metadata = {
        "num_events": num_events,
        "max_particles": max_particles,
        "p4_columns": p4_columns,
        "p4_order": "E_px_py_pz",
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    return {
        "no": no,
        "id": pid,
        "pid": pid,
        "status": status,
        "mothers": mothers,
        "daughters": daughters,
        "colors": colors,
        "p4": p4,
        "mass": mass,
        "scale": scale,
        "mask": mask,
        "names": names,
        "metadata": metadata,
    }

def save_json_sidecar(events: list[Event], out_json: Path) -> None:
    """Save a human-readable parsed version for debugging."""
    serializable = [
        {
            "event_index": event.event_index,
            "particles": [asdict(p) for p in event.particles],
        }
        for event in events
    ]
    out_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def summarize(data: dict | torch.Tensor) -> str:
    if torch.is_tensor(data):
        lines = []
        lines.append(f"saved tensor shape: {tuple(data.shape)}")
        if data.ndim == 3 and data.shape[-1] == 4:
            lines.append("p4 columns: E, px, py, pz")
            nonzero = torch.any(data != 0, dim=-1)
            counts = nonzero.sum(dim=1) if data.shape[0] > 0 else torch.tensor([])
            if data.shape[0] > 0:
                lines.append(
                    f"nonzero four-vectors/event: min={counts.min().item()}, "
                    f"mean={counts.float().mean().item():.2f}, max={counts.max().item()}"
                )
        return "\n".join(lines)

    mask = data["mask"]
    pid = data["pid"]
    status = data["status"]
    p4 = data["p4"]

    num_events = data["metadata"]["num_events"]
    max_particles = data["metadata"]["max_particles"]
    p4_columns = data.get("metadata", {}).get("p4_columns", ["E", "px", "py", "pz"])
    counts = mask.sum(dim=1) if num_events > 0 else torch.tensor([])

    lines = []
    lines.append(f"events: {num_events}")
    lines.append(f"max particles/event: {max_particles}")
    if num_events > 0:
        lines.append(
            f"particles/event: min={counts.min().item()}, "
            f"mean={counts.float().mean().item():.2f}, max={counts.max().item()}"
        )
        final_mask = mask & (status > 0)
        lines.append(f"final-state particles total: {final_mask.sum().item()}")
        top_ids = pid[mask]
        if top_ids.numel() > 0:
            unique, counts_id = torch.unique(top_ids, return_counts=True)
            order = torch.argsort(counts_id, descending=True)[:10]
            pairs = [f"{int(unique[k])}:{int(counts_id[k])}" for k in order]
            lines.append("most common PDG ids: " + ", ".join(pairs))
        finite_p4 = p4[mask]
        if finite_p4.numel() > 0:
            if p4_columns == ["E", "px", "py", "pz"]:
                px_col, py_col = 1, 2
            else:
                px_col, py_col = 0, 1
            pt = torch.sqrt(finite_p4[:, px_col] ** 2 + finite_p4[:, py_col] ** 2)
            lines.append(f"p4 columns: {p4_columns}")
            lines.append(f"pT range: [{pt.min().item():.6g}, {pt.max().item():.6g}]")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Parse a PYTHIA event listing/log into padded PyTorch tensors."
    )
    parser.add_argument("logfile", type=Path, help="Input PYTHIA log/event-listing text file")
    parser.add_argument("-o", "--output", type=Path, default=Path("pythia_events.pt"), help="Output .pt file")
    parser.add_argument(
        "--keep-status",
        choices=["all", "final", "hard"],
        default="all",
        help="Which particles to keep: all, final(status>0), or rough hard-process range |status|=20..29",
    )
    parser.add_argument(
        "--float64",
        action="store_true",
        help="Store floating tensors as float64 instead of float32",
    )
    parser.add_argument(
        "--only-p4",
        action="store_true",
        help=(
            "Only save the padded four-momentum tensor instead of the full event-record dict. "
            "With this option, the saved tensor has shape [events, max_particles, 4] and "
            "columns [E, px, py, pz]. Padding rows are zeros."
        ),
    )
    parser.add_argument(
        "--randomize-branching-daughter-nos",
        action="store_true",
        help=(
            "After parsing, randomly swap the event-record line numbers of the two "
            "daughters of each two-daughter branching with probability 1/2. "
            "All mother/daughter references are relabelled consistently; momenta stay "
            "attached to the same physical particle record."
        ),
    )
    parser.add_argument(
        "--daughter-no-seed",
        type=int,
        default=None,
        help="Random seed for --randomize-branching-daughter-nos.",
    )
    parser.add_argument(
        "--randomize-exclude-hard-process",
        action="store_true",
        help=(
            "Do not use hard-process rows, roughly |status|=20..29, when finding "
            "two-daughter pairs to randomize. By default they are included, since "
            "hard outgoing partons with status -23 can be the parents of the first FSR branch."
        ),
    )
    parser.add_argument(
        "--json-sidecar",
        type=Path,
        default=None,
        help="Optional JSON file with unpadded parsed events for debugging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printed summary",
    )
    args = parser.parse_args(argv)

    events = parse_pythia_log(args.logfile)
    if not events:
        raise RuntimeError(
            "No events found. Check that the log contains a PYTHIA event listing "
            "with a header containing: no id name status mothers daughters colours."
        )

    metadata_extra: dict = {}
    if args.randomize_branching_daughter_nos:
        events = randomize_branching_daughter_nos(
            events,
            seed=args.daughter_no_seed,
            include_hard_process=not args.randomize_exclude_hard_process,
        )
        metadata_extra.update({
            "randomize_branching_daughter_nos": True,
            "daughter_no_seed": args.daughter_no_seed,
            "randomize_include_hard_process": not args.randomize_exclude_hard_process,
            "randomization_note": (
                "Only event-record line labels no/mother/daughter references were relabelled. "
                "Particle identities and four-momenta were not exchanged or resampled."
            ),
        })
    else:
        metadata_extra["randomize_branching_daughter_nos"] = False

    keep_status = None if args.keep_status == "all" else args.keep_status
    dtype = torch.float64 if args.float64 else torch.float32

    metadata_extra.update({
        "only_p4": bool(args.only_p4),
    })

    data = events_to_tensors(
        events,
        keep_status=keep_status,
        dtype=dtype,
        metadata_extra=metadata_extra,
        p4_only=args.only_p4,
    )
    
    # print(data["p4"].shape)
    # # print(data["p4"])
    # my_momentum_p3 = data['p4'][..., 1:4]
    # print(my_momentum_p3)
    # my_momentum_sum = torch.linalg.norm(my_momentum_p3.sum(dim=1), dim=-1)
    # print("my momentum sum:", my_momentum_sum.mean().item(), my_momentum_sum.max().item(), my_momentum_sum.min().item())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)

    if args.json_sidecar is not None:
        args.json_sidecar.parent.mkdir(parents=True, exist_ok=True)
        save_json_sidecar(events, args.json_sidecar)

    if not args.quiet:
        print(f"Saved: {args.output}")
        print(summarize(data))


if __name__ == "__main__":
    main()
