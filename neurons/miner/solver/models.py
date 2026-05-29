from __future__ import annotations

import hashlib

import bittensor as bt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

try:
    from nova_ph2.PSICHIC.wrapper import PsichicWrapper
    from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening
except ImportError:
    PsichicWrapper = None
    virtual_screening = None


class ModelManager:
    def __init__(self, config: dict):
        self.target_models = []
        self.antitarget_models = []
        self.target_sequences = list(config.get("target_sequences", []))
        self.antitarget_sequences = list(config.get("antitarget_sequences", []))
        self.use_psichic = PsichicWrapper is not None

        if self.use_psichic:
            for seq in self.target_sequences:
                wrapper = PsichicWrapper()
                wrapper.initialize_model(seq)
                self.target_models.append(wrapper)

            for seq in self.antitarget_sequences:
                wrapper = PsichicWrapper()
                wrapper.initialize_model(seq)
                self.antitarget_models.append(wrapper)

            bt.logging.info(
                f"[Init] ModelManager using PSICHIC target={len(self.target_models)} anti={len(self.antitarget_models)}"
            )
        else:
            bt.logging.warning(
                "[Init] PSICHIC unavailable; using deterministic heuristic prior for local development"
            )

    def get_target_score_from_data(self, data: pd.Series):
        if self.use_psichic:
            try:
                target_scores = []
                smiles_list = data.tolist()
                for target_model in self.target_models:
                    scores = target_model.score_molecules(smiles_list)
                    for antitarget_model in self.antitarget_models:
                        antitarget_model.smiles_list = smiles_list
                        antitarget_model.smiles_dict = target_model.smiles_dict
                    scores.rename(columns={"predicted_binding_affinity": "target"}, inplace=True)
                    target_scores.append(scores["target"])
                if not target_scores:
                    return pd.Series(dtype=float)
                return pd.DataFrame(target_scores).mean(axis=0)
            except Exception as e:
                bt.logging.error(f"Target scoring error: {e}")
                return pd.Series(dtype=float)

        values = [self._heuristic_score(smiles) for smiles in data.tolist()]
        return pd.Series(values, index=data.index, dtype=float)

    def get_antitarget_score(self):
        if not self.use_psichic:
            return pd.Series(dtype=float)
        try:
            antitarget_scores = []
            for i, antitarget_model in enumerate(self.antitarget_models):
                antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
                antitarget_model.screen_df = virtual_screening(
                    antitarget_model.screen_df,
                    antitarget_model.model,
                    antitarget_model.screen_loader,
                    ".",
                    save_interpret=False,
                    ligand_dict=antitarget_model.smiles_dict,
                    device=antitarget_model.device,
                    save_cluster=False,
                )
                scores = antitarget_model.screen_df[["predicted_binding_affinity"]].copy()
                scores.rename(columns={"predicted_binding_affinity": f"anti_{i}"}, inplace=True)
                antitarget_scores.append(scores[f"anti_{i}"])

            if not antitarget_scores:
                return pd.Series(dtype=float)
            return pd.DataFrame(antitarget_scores).mean(axis=0)
        except Exception as e:
            bt.logging.error(f"Antitarget scoring error: {e}")
            return pd.Series(dtype=float)

    def _heuristic_score(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float("nan")

        heavy_atoms = mol.GetNumHeavyAtoms()
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)

        seq_bias = 0.0
        sequences = self.target_sequences or ["default-target"]
        for seq in sequences:
            digest = hashlib.sha256(seq.encode()).digest()
            seq_bias += int.from_bytes(digest[:2], "little") / 65535.0 - 0.5
        seq_bias /= max(1, len(sequences))

        return (
            0.22 * heavy_atoms
            + 0.35 * logp
            + 0.18 * rings
            - 0.06 * tpsa
            - 0.12 * rotatable_bonds
            + 0.5 * seq_bias
        )