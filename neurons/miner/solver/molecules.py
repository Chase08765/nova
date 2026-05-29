from __future__ import annotations

import math
import os
import sqlite3
from functools import lru_cache
from typing import List, Tuple

import bittensor as bt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys

from combinatorial_db.reactions import get_reaction_info, get_smiles_from_reaction
from utils.molecules import get_heavy_atom_count


class MoleculeUtils:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
        try:
            abs_db_path = os.path.abspath(db_path)
            with sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True) as conn:
                conn.execute("PRAGMA query_only = ON")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?",
                    (role_mask, role_mask),
                )
                return cursor.fetchall()
        except Exception as e:
            bt.logging.error(f"[MoleculeUtils] role query failed role={role_mask}: {e}")
            return []

    @staticmethod
    def num_rotatable_bonds(smiles: str) -> int:
        if not smiles:
            return 0
        try:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol is None:
                return 0
            return Descriptors.NumRotatableBonds(mol)
        except Exception:
            return 0

    @staticmethod
    @lru_cache(maxsize=None)
    def mol_from_smiles_cached(smiles: str):
        if not smiles:
            return None
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    @lru_cache(maxsize=None)
    def get_smiles_from_reaction_cached(name: str):
        try:
            return get_smiles_from_reaction(name)
        except Exception:
            return None

    @staticmethod
    @lru_cache(maxsize=None)
    def maccs_fp_from_smiles_cached(smiles: str):
        if not smiles:
            return None
        try:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol is None:
                return None
            return MACCSkeys.GenMACCSKeys(mol)
        except Exception:
            return None

    @staticmethod
    def compute_maccs_entropy(smiles_list: list[str]) -> float:
        n_bits = 167
        bit_counts = np.zeros(n_bits)
        valid_mols = 0

        for smiles in smiles_list:
            fp = MoleculeUtils.maccs_fp_from_smiles_cached(smiles)
            if fp is None:
                continue
            bit_counts += np.array(fp)
            valid_mols += 1

        if valid_mols == 0:
            raise ValueError("No valid molecules found.")

        probs = bit_counts / valid_mols
        entropy_per_bit = np.array(
            [
                -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
                for p in probs
            ]
        )
        return float(np.mean(entropy_per_bit))

    @staticmethod
    def _heavy_atoms_dict_from_bitcounts(bitcounts: pd.DataFrame | None) -> dict[int, int]:
        if bitcounts is None or bitcounts.empty or "heavy_atoms" not in bitcounts.columns:
            return {}
        return dict(zip(bitcounts["mol_id"], bitcounts["heavy_atoms"]))


class MoleculeManager:
    def __init__(self, config: dict, db_path: str):
        self.rxn_id = int(config.get("rxn_id", 2))
        self.db_path = db_path

        reaction_info = get_reaction_info(self.rxn_id, db_path)
        if reaction_info is None:
            raise RuntimeError(f"Reaction {self.rxn_id} not found in {db_path}")

        _, self.roleA, self.roleB, self.roleC = reaction_info
        self.is_three_component = self.roleC is not None and self.roleC != 0

        self.molecules_A = MoleculeUtils.get_molecules_by_role(self.roleA, db_path)
        self.molecules_B = MoleculeUtils.get_molecules_by_role(self.roleB, db_path)
        self.molecules_C = (
            MoleculeUtils.get_molecules_by_role(self.roleC, db_path) if self.is_three_component else []
        )

        self.moles_A_id = [mol[0] for mol in self.molecules_A]
        self.moles_B_id = [mol[0] for mol in self.molecules_B]
        self.moles_C_id = [mol[0] for mol in self.molecules_C] if self.is_three_component else None

        self.role_A_bitcounts = pd.DataFrame(
            self.molecules_A, columns=["mol_id", "smiles", "_"]
        )[["mol_id", "smiles"]]
        self.role_A_bitcounts["heavy_atoms"] = self.role_A_bitcounts["smiles"].apply(get_heavy_atom_count)

        self.role_B_bitcounts = pd.DataFrame(
            self.molecules_B, columns=["mol_id", "smiles", "_"]
        )[["mol_id", "smiles"]]
        self.role_B_bitcounts["heavy_atoms"] = self.role_B_bitcounts["smiles"].apply(get_heavy_atom_count)

        if self.is_three_component:
            self.role_C_bitcounts = pd.DataFrame(
                self.molecules_C, columns=["mol_id", "smiles", "_"]
            )[["mol_id", "smiles"]]
            self.role_C_bitcounts["heavy_atoms"] = self.role_C_bitcounts["smiles"].apply(get_heavy_atom_count)
        else:
            self.role_C_bitcounts = None

        self.dict_A = MoleculeUtils._heavy_atoms_dict_from_bitcounts(self.role_A_bitcounts)
        self.dict_B = MoleculeUtils._heavy_atoms_dict_from_bitcounts(self.role_B_bitcounts)
        self.dict_C = (
            MoleculeUtils._heavy_atoms_dict_from_bitcounts(self.role_C_bitcounts)
            if self.role_C_bitcounts is not None
            else {}
        )