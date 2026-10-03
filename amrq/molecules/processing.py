"""Molecule preparation and descriptor computation with RDKit.

Every candidate compound is protonated, embedded in 3D (ETKDG), energy
minimized (MMFF/UFF), and characterized: physicochemical descriptors,
Gasteiger partial charges, metal-chelating atoms, and a 2D SVG for the
dashboard.
"""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors, rdPartialCharges
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

RDLogger.DisableLog("rdApp.*")  # keep the run log clean

# SMARTS patterns for atoms able to chelate a metal center (Zn2+ of NDM-1):
_CHELATION_SMARTS = [
    ("carboxylate", Chem.MolFromSmarts("[CX3](=[OX1])-[OX2H0,OX1H1-]")),
    ("carboxylic_acid", Chem.MolFromSmarts("[CX3](=[OX1])-[OX2H1]")),
    ("tetrazole", Chem.MolFromSmarts("[nX3]1[nX3][nX3][nX3]c1")),
    ("thiol", Chem.MolFromSmarts("[SX2H1,SX2H0-]")),
    ("hydroxamate", Chem.MolFromSmarts("[NX3][OX2H1]")),
    ("N_hydroxypyridinone", Chem.MolFromSmarts("[OX1]~[nX3+]?")),
    ("catechol", Chem.MolFromSmarts("[OX2H]c[c,O][OX2H]")),
    ("phosphonate", Chem.MolFromSmarts("[PX4](=[OX1])([OX2H1])")),
    ("pyridine_N", Chem.MolFromSmarts("[nX2]:[c]")),
    ("enol_beta_dicarbonyl", Chem.MolFromSmarts("[OX2H]~[CX3]~[CX3]~[OX1]")),
]


def parse_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string, keep the largest fragment, sanitize."""
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        frags = (mol,)
    if len(frags) > 1:  # salt/metabolite: keep the parent
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return mol


def prepare_molecule(smiles: str, n_confs: int = 3, seed: int = 42) -> Chem.Mol | None:
    """SMILES -> protonated, 3D-embedded, energy-minimized mol (lowest-energy conf)."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    except Exception:
        conf_ids = []
    if not conf_ids:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=800)
        else:
            results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=800)
        best = min(range(len(conf_ids)), key=lambda i: results[i][1])
    except Exception:
        best = conf_ids[0]
    conf = Chem.Conformer(mol.GetConformer(best))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def gasteiger_charges(mol: Chem.Mol) -> np.ndarray:
    work = Chem.Mol(mol)
    rdPartialCharges.ComputeGasteigerCharges(work, nIter=12, throwOnParamFailure=False)
    charges = np.array([a.GetDoubleProp("_GasteigerCharge") for a in work.GetAtoms()])
    # Gasteiger parameters do not cover every element (e.g. boron) and yield
    # NaN there; neutralize so downstream math stays finite.
    return np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0)


def chelating_atoms(mol: Chem.Mol) -> tuple[list[int], list[str]]:
    """Atom indices + pattern names for likely metal-chelating atoms."""
    hits, names = [], []
    seen = set()
    for name, patt in _CHELATION_SMARTS:
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            for idx in match:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() in ("N", "O", "S") and idx not in seen:
                    seen.add(idx)
                    hits.append(idx)
                    names.append(name)
    return sorted(hits), names


def descriptors(mol: Chem.Mol) -> dict:
    charges = gasteiger_charges(mol)
    chel_idx, chel_names = chelating_atoms(mol)
    heavy = mol.GetNumHeavyAtoms()
    return {
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
        "mw": round(Descriptors.MolWt(mol), 2),
        "crippen_logp": round(Crippen.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": CalcNumRotatableBonds(mol),
        "heavy_atoms": heavy,
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "formal_charge": Chem.GetFormalCharge(mol),
        "max_positive_charge": round(float(charges.max()), 3),
        "max_negative_charge": round(float(charges.min()), 3),
        "chelating_atoms": len(chel_idx),
        "chelating_types": sorted(set(chel_names)),
        "chelating_indices": chel_idx,
        "radius_gyration": round(float(rdMolDescriptors.CalcRadiusOfGyration(mol)), 3),
    }


def ligand_coords(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    conf = mol.GetConformer(conf_id)
    return np.array(conf.GetPositions(), dtype=float)


def heavy_atom_mask(mol: Chem.Mol) -> np.ndarray:
    return np.array([a.GetSymbol() != "H" for a in mol.GetAtoms()])


def polar_acceptor_mask(mol: Chem.Mol) -> np.ndarray:
    """Atoms of the ligand that can act as H-bond acceptors/donors (N, O, S)."""
    return np.array([a.GetSymbol() in ("N", "O", "S") for a in mol.GetAtoms()])


def mol_to_svg(mol: Chem.Mol, size: int = 260) -> str:
    """2D depiction as a standalone SVG string."""
    work = Chem.RemoveHs(Chem.Mol(mol))
    Chem.AllChem.Compute2DCoords(work)
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False
    opts.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, work)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;utf8," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
