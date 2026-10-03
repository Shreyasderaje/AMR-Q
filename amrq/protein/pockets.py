"""Curated AMR resistance-protein targets.

The target definitions live in ``data/proteins/known_targets.json`` so users
can extend the panel without touching code (add a new entry with its PDB ID
and curated pocket residues and it shows up in the dashboard automatically).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TARGETS_FILE = Path(__file__).resolve().parents[2] / "data" / "proteins" / "known_targets.json"


@dataclass
class Target:
    key: str
    name: str
    short: str
    enzyme_class: str
    resistance_class: str
    pdb_id: str
    uniprot: str
    chelation_relevant: bool
    pocket_method: str
    pocket_chain: str
    pocket_residues: list[int]
    resistance_mechanism: str
    clinical_note: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "short": self.short,
            "enzyme_class": self.enzyme_class,
            "resistance_class": self.resistance_class,
            "pdb_id": self.pdb_id,
            "uniprot": self.uniprot,
            "chelation_relevant": self.chelation_relevant,
            "pocket_method": self.pocket_method,
            "pocket_chain": self.pocket_chain,
            "pocket_residues": self.pocket_residues,
            "resistance_mechanism": self.resistance_mechanism,
            "clinical_note": self.clinical_note,
            "notes": self.notes,
        }


def load_targets() -> dict[str, Target]:
    raw = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    targets = {}
    for key, item in raw.items():
        pocket = item.get("pocket", {})
        targets[key] = Target(
            key=item.get("key", key),
            name=item.get("name", key),
            short=item.get("short", key),
            enzyme_class=item.get("enzyme_class", ""),
            resistance_class=item.get("resistance_class", "unknown"),
            pdb_id=item.get("pdb_id", ""),
            uniprot=item.get("uniprot", ""),
            chelation_relevant=bool(item.get("chelation_relevant", False)),
            pocket_method=pocket.get("method", "cocrystal"),
            pocket_chain=pocket.get("chain", "A"),
            pocket_residues=[int(r) for r in pocket.get("residues", [])],
            resistance_mechanism=item.get("resistance_mechanism", ""),
            clinical_note=item.get("clinical_note", ""),
            notes=item.get("notes", ""),
            extra={k: v for k, v in item.items()
                   if k not in {"key", "name", "short", "enzyme_class", "resistance_class",
                                "pdb_id", "uniprot", "chelation_relevant", "pocket",
                                "resistance_mechanism", "clinical_note", "notes"}},
        )
    return targets


def get_target(key: str) -> Target:
    targets = load_targets()
    if key not in targets:
        raise KeyError(f"Unknown target {key!r}. Available: {sorted(targets)}")
    return targets[key]
