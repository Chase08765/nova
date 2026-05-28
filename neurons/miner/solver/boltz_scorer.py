"""
BoltzScorer — pluggable expensive-oracle wrapper.

Two backends:
  * `BoltzGpuScorer`   – real Boltz-2 call, GPU only. Enabled when env
                         var BOLTZ_ENABLED=1.
  * `MockOracle`        – fallback for local development: uses PSICHIC as
                         a stand-in so the pipeline runs end-to-end on
                         a CPU box.

Both expose the same interface:
    score(smiles_list: list[str]) -> list[float]
    name: str
    seconds_per_call: float   # rough estimate for budget planning

The miner code never branches on backend; it just calls `score(...)`.
"""

from __future__ import annotations

import os
import json
import shutil
import glob
import hashlib
import traceback
from typing import List, Optional

import bittensor as bt


# ─── factory ─────────────────────────────────────────────────────────────────
def make_oracle(config: dict, psichic_model_manager) -> "BaseOracle":
    """Pick the right oracle based on env. Single switch for local-vs-GPU."""
    enabled = os.environ.get("BOLTZ_ENABLED", "0") == "1"
    if enabled:
        try:
            scorer = BoltzGpuScorer(config)
            bt.logging.info("[Oracle] Using BoltzGpuScorer (GPU)")
            return scorer
        except Exception as e:
            bt.logging.warning(f"[Oracle] BoltzGpuScorer init failed ({e}); "
                               "falling back to MockOracle (PSICHIC).")
    bt.logging.info("[Oracle] Using MockOracle (PSICHIC) — set BOLTZ_ENABLED=1 for real Boltz.")
    return MockOracle(psichic_model_manager)


# ─── base ────────────────────────────────────────────────────────────────────
class BaseOracle:
    name: str = "base"
    seconds_per_call: float = 1.0

    def score(self, smiles_list: List[str]) -> List[Optional[float]]:
        raise NotImplementedError


# ─── PSICHIC fallback ────────────────────────────────────────────────────────
class MockOracle(BaseOracle):
    """Uses PSICHIC target score as a cheap stand-in for Boltz when GPU/Boltz unavailable."""

    name = "psichic_mock"
    seconds_per_call = 0.05  # very rough; PSICHIC is ms/mol on GPU, ~50ms/mol on CPU

    def __init__(self, psichic_model_manager):
        self.mm = psichic_model_manager

    def score(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not smiles_list:
            return []
        try:
            import pandas as pd
            target = self.mm.get_target_score_from_data(pd.Series(smiles_list))
            return [float(v) if v is not None else None for v in target.tolist()]
        except Exception as e:
            bt.logging.warning(f"[MockOracle] PSICHIC scoring failed: {e}")
            return [None] * len(smiles_list)


# ─── real Boltz ──────────────────────────────────────────────────────────────
class BoltzGpuScorer(BaseOracle):
    """
    Single-user, miner-side Boltz-2 invoker.

    Reads target sequences from config["boltz_targets"]:
        [{"name": "MyProt", "sequence": "MKT...", "msa_path": "/abs/path.a3m"}, ...]
    If msa_path is missing, runs single-sequence mode (less accurate but still useful).

    Combines per-target scores with the same `combination_strategy` used by the
    validator (`average` or `heavy_atom_normalization`).
    """

    name = "boltz_gpu"
    seconds_per_call = 45.0  # A100 estimate per miner docs

    def __init__(self, config: dict):
        # lazy imports: do not require boltz/torch at module import time
        import torch  # noqa: F401
        from boltz.main import predict  # noqa: F401
        self._predict = predict

        self.targets = config.get("boltz_targets") or []
        if not self.targets:
            raise RuntimeError(
                "BoltzGpuScorer requires config['boltz_targets'] "
                "[{name, sequence, msa_path?}, ...]"
            )

        self.combination_strategy = config.get("combination_strategy", "average")
        self.boltz_metric: List[str] = config.get("boltz_metric") or ["affinity_pred_value"]
        self.boltz_mode = config.get("boltz_mode", "max")

        # Boltz runtime config (matches validator defaults; safe fallbacks)
        self.recycling_steps           = int(config.get("boltz_recycling_steps", 3))
        self.sampling_steps            = int(config.get("boltz_sampling_steps", 200))
        self.diffusion_samples         = int(config.get("boltz_diffusion_samples", 1))
        self.sampling_steps_affinity   = int(config.get("boltz_sampling_steps_affinity", 200))
        self.diffusion_samples_affinity = int(config.get("boltz_diffusion_samples_affinity", 5))
        self.output_format             = config.get("boltz_output_format", "pdb")
        self.affinity_mw_correction    = bool(config.get("boltz_affinity_mw_correction", False))
        self.override                  = bool(config.get("boltz_override", True))
        self.base_seed                 = int(config.get("boltz_seed", 68))

        # tmp dirs — kept separate from validator's tree
        base = os.path.dirname(os.path.abspath(__file__))
        self.tmp_dir    = os.path.join(base, "_boltz_tmp")
        self.input_dir  = os.path.join(self.tmp_dir, "inputs")
        self.output_dir = os.path.join(self.tmp_dir, "outputs")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        bt.logging.info(
            f"[BoltzGpuScorer] init: {len(self.targets)} target(s), "
            f"strategy={self.combination_strategy}, metric={self.boltz_metric}, mode={self.boltz_mode}"
        )

    # ---- public API ---------------------------------------------------------
    def score(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not smiles_list:
            return []
        bt.logging.info(f"[BoltzGpuScorer] scoring {len(smiles_list)} molecule(s) on GPU …")

        # 1. write yaml inputs (per molecule × per target)
        self._cleanup_inputs()
        smiles_to_idx = {}
        for smi in smiles_list:
            mol_idx = self._mol_idx(smi)
            smiles_to_idx[smi] = mol_idx
            for tgt in self.targets:
                yaml_str = self._yaml(tgt, smi)
                path = os.path.join(self.input_dir, f"{mol_idx}_{tgt['name']}.yaml")
                with open(path, "w") as f:
                    f.write(yaml_str)

        # 2. run Boltz
        try:
            self._predict(
                data=self.input_dir,
                out_dir=self.output_dir,
                recycling_steps=self.recycling_steps,
                sampling_steps=self.sampling_steps,
                diffusion_samples=self.diffusion_samples,
                sampling_steps_affinity=self.sampling_steps_affinity,
                diffusion_samples_affinity=self.diffusion_samples_affinity,
                output_format=self.output_format,
                seed=self.base_seed,
                affinity_mw_correction=self.affinity_mw_correction,
                override=self.override,
                num_workers=0,
            )
        except Exception as e:
            bt.logging.error(f"[BoltzGpuScorer] predict() failed: {e}")
            bt.logging.error(traceback.format_exc())
            return [None] * len(smiles_list)

        # 3. collect + combine scores
        out: List[Optional[float]] = []
        for smi in smiles_list:
            mol_idx = smiles_to_idx[smi]
            try:
                per_target = []
                for tgt in self.targets:
                    metrics = self._load_metrics(mol_idx, tgt["name"])
                    score = self._combine(metrics, smi)
                    if score is not None:
                        per_target.append(score)
                if not per_target:
                    out.append(None)
                else:
                    # validator averages per-target then sums; for a single mol the
                    # average is the right per-molecule scalar.
                    out.append(float(sum(per_target) / len(per_target)))
            except Exception as e:
                bt.logging.warning(f"[BoltzGpuScorer] score-collect failed for {smi}: {e}")
                out.append(None)

        good = sum(1 for v in out if v is not None)
        bt.logging.info(f"[BoltzGpuScorer] done: {good}/{len(out)} scored")
        return out

    # ---- helpers ------------------------------------------------------------
    def _mol_idx(self, smiles: str) -> int:
        h = hashlib.sha256(smiles.encode()).digest()
        return (int.from_bytes(h[:8], "little") ^ self.base_seed) % (2**31 - 1)

    def _yaml(self, target: dict, ligand_smiles: str) -> str:
        msa_line = f"\n      msa: {target['msa_path']}" if target.get("msa_path") else ""
        return (
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {target['sequence']}{msa_line}\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: {ligand_smiles}\n"
            "properties:\n"
            "  - affinity:\n"
            "      binder: B\n"
        )

    def _load_metrics(self, mol_idx: int, target_name: str) -> dict:
        results_path = os.path.join(
            self.output_dir, "boltz_results_inputs", "predictions",
            f"{mol_idx}_{target_name}"
        )
        out: dict = {}
        if not os.path.exists(results_path):
            return out
        for fn in os.listdir(results_path):
            if fn.startswith("affinity") or fn.startswith("confidence"):
                try:
                    with open(os.path.join(results_path, fn)) as f:
                        out.update(json.load(f))
                except (json.JSONDecodeError, IOError) as e:
                    bt.logging.warning(f"[BoltzGpuScorer] bad metric file {fn}: {e}")
        return out

    def _combine(self, metrics: dict, smiles: str) -> Optional[float]:
        if not metrics:
            return None
        try:
            if self.combination_strategy == "average":
                vals = [metrics[m] for m in self.boltz_metric if m in metrics]
                return float(sum(vals) / len(vals)) if vals else None
            if self.combination_strategy == "heavy_atom_normalization":
                if len(self.boltz_metric) < 2:
                    bt.logging.warning("[BoltzGpuScorer] heavy_atom_normalization needs 2 metrics")
                    return None
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles)
                    heavy = mol.GetNumHeavyAtoms() if mol is not None else 0
                except Exception:
                    heavy = 0
                if heavy == 0:
                    return None
                m1, m2 = self.boltz_metric[0], self.boltz_metric[1]
                if m1 not in metrics or m2 not in metrics:
                    return None
                return float((metrics[m1] - metrics[m2]) / heavy)
            bt.logging.warning(f"[BoltzGpuScorer] unknown combination_strategy={self.combination_strategy}")
            return None
        except Exception as e:
            bt.logging.warning(f"[BoltzGpuScorer] combine failed: {e}")
            return None

    def _cleanup_inputs(self) -> None:
        # reset between rounds so stale yamls don't get re-predicted
        for p in glob.glob(os.path.join(self.input_dir, "*.yaml")):
            try:
                os.remove(p)
            except OSError:
                pass
        res = os.path.join(self.output_dir, "boltz_results_inputs")
        if os.path.exists(res):
            try:
                shutil.rmtree(res)
            except OSError as e:
                bt.logging.warning(f"[BoltzGpuScorer] cleanup failed: {e}")
