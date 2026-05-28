"""
Active-learning miner — entry point.

Strategy (see project docs for the full rationale):

  Phase 1  PSICHIC prior          — score a wide random sample once.
  Phase 2  Reactant-aware frontier — reassemble top reactants → ~1.5k candidates.
  Phase 3  Active-learning loop    — each round, train RF on collected oracle
                                     labels, pick K by UCB, send to oracle.
  Phase 4  Final pick               — submit the molecule with the best measured
                                     oracle score (with optional anti-target
                                     tie-break + entropy diversification).

The "oracle" is Boltz when BOLTZ_ENABLED=1 (GPU), otherwise PSICHIC (mock) so
the pipeline can be developed and tested on a CPU box. See boltz_scorer.py.

Same I/O contract as solver_46:
    input  : ./input.json
    output : $OUTPUT_DIR/result.json (default /output/result.json)
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import bittensor as bt

# --- repo + sibling solver_46 on path -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOVA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../.."))
SOLVER_46_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "solver_46"))
for p in (NOVA_DIR, SOLVER_46_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import nova_ph2  # noqa: E402
from molecules import MoleculeManager, MoleculeUtils  # from solver_46
from models import ModelManager  # from solver_46 (PSICHIC wrapper)

from boltz_scorer import make_oracle
from selection import build_prior_pool, build_frontier, pick_round_batch
from surrogate import Surrogate


# ─── runtime config ─────────────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

TIME_LIMIT = int(os.environ.get("MINER_TIME_LIMIT", "900"))     # whole epoch budget
PHASE1_SAMPLE = int(os.environ.get("PHASE1_SAMPLE", "4000"))     # how many to PSICHIC-score
PHASE2_TOP_PER_SLOT = int(os.environ.get("PHASE2_TOP_PER_SLOT", "30"))
PHASE2_MAX_FRONTIER = int(os.environ.get("PHASE2_MAX_FRONTIER", "1500"))
TANIMOTO = float(os.environ.get("TANIMOTO_DIVERSITY", "0.7"))

# Per-round oracle batch & UCB schedule. Tuned for ~15 Boltz evals in 900s.
ROUND_SCHEDULE = [
    # (k, alpha, kappa)
    (4, 0.0, 1.5),   # round 1: pure prior + heavy exploration (no surrogate yet)
    (4, 0.4, 1.2),   # round 2
    (4, 0.7, 0.6),   # round 3
    (3, 0.9, 0.2),   # round 4: pure exploitation
]


# ─── globals (set by initialize_solution) ───────────────────────────────────
molecule_manager: Optional[MoleculeManager] = None
model_manager: Optional[ModelManager] = None


# ─── config + init ──────────────────────────────────────────────────────────
def get_config(input_file: Optional[str] = None) -> dict:
    if input_file is None:
        input_file = os.path.join(BASE_DIR, "input.json")
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


def initialize_solution(config: dict) -> None:
    global molecule_manager, model_manager
    bt.logging.info("[Init] loading MoleculeManager + PSICHIC ModelManager …")
    molecule_manager = MoleculeManager(config=config, db_path=DB_PATH)
    model_manager = ModelManager(config)
    bt.logging.info(
        f"[Init] rxn_id={molecule_manager.rxn_id} "
        f"|A|={len(molecule_manager.moles_A_id)} "
        f"|B|={len(molecule_manager.moles_B_id)} "
        f"|C|={len(molecule_manager.moles_C_id) if molecule_manager.is_three_component else 0}"
    )


# ─── validity gate (mirror validator checks) ────────────────────────────────
def _is_submittable(smiles: str, config: dict) -> bool:
    try:
        heavy = MoleculeUtils.get_smiles_from_reaction_cached  # warm cache
        from utils.molecules import get_heavy_atom_count
        ha = get_heavy_atom_count(smiles)
        if ha < int(config.get("min_heavy_atoms", 10)):
            return False
        rb = MoleculeUtils.num_rotatable_bonds(smiles)
        if rb < int(config.get("min_rotatable_bonds", 1)) or rb > int(config.get("max_rotatable_bonds", 10)):
            return False
        return True
    except Exception as e:
        bt.logging.debug(f"[Validity] check failed for {smiles}: {e}")
        return False


# ─── final pick + write ─────────────────────────────────────────────────────
def _save_result(names: List[str]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "result.json")
    with open(path, "w") as f:
        json.dump({"molecules": names}, f, ensure_ascii=False, indent=2)
    bt.logging.info(f"[Submit] wrote {len(names)} molecule(s) → {path}")


def _pick_final(
    evaluated: pd.DataFrame,
    config: dict,
) -> List[str]:
    """
    Pick `num_molecules` from the evaluated set: best oracle score first,
    with entropy diversification if num_molecules > 1.
    """
    n = int(config.get("num_molecules", 1))
    mode = config.get("boltz_mode", "max")
    ascending = (mode == "min")

    valid = evaluated[evaluated["oracle"].notna()].copy()
    valid = valid[valid["smiles"].apply(lambda s: _is_submittable(s, config))]
    if valid.empty:
        bt.logging.warning("[Final] no validated, oracle-scored candidate; falling back to prior")
        return []

    valid = valid.sort_values("oracle", ascending=ascending).reset_index(drop=True)
    if n == 1:
        chosen = [valid.iloc[0]["name"]]
        bt.logging.info(
            f"[Final] best: {chosen[0]}  oracle={valid.iloc[0]['oracle']:.4f}"
        )
        return chosen

    # multi-pick with MACCS-entropy aware diversity
    picked_names: List[str] = []
    picked_smiles: List[str] = []
    entropy_min = float(config.get("entropy_min_threshold", 0.0))
    for _, row in valid.iterrows():
        candidate_smiles = picked_smiles + [row["smiles"]]
        try:
            ent = MoleculeUtils.compute_maccs_entropy(candidate_smiles)
        except Exception:
            ent = 0.0
        # always allow first one; for subsequent, require entropy to not drop
        if not picked_smiles or ent >= entropy_min * (len(picked_smiles)) / max(1, len(candidate_smiles)):
            picked_names.append(row["name"])
            picked_smiles.append(row["smiles"])
        if len(picked_names) >= n:
            break

    # backfill if entropy filter starved us
    if len(picked_names) < n:
        for _, row in valid.iterrows():
            if row["name"] not in picked_names:
                picked_names.append(row["name"])
            if len(picked_names) >= n:
                break

    bt.logging.info(
        f"[Final] picked {len(picked_names)} mol(s): "
        f"{[(n, float(valid[valid['name']==n]['oracle'].iloc[0])) for n in picked_names]}"
    )
    return picked_names


# ─── main loop ──────────────────────────────────────────────────────────────
def find_solution(config: dict, time_start: float) -> None:
    assert molecule_manager is not None and model_manager is not None, "call initialize_solution first"

    deadline = time_start + TIME_LIMIT
    oracle = make_oracle(config, model_manager)

    # ---- Phase 1 ---------------------------------------------------------------
    prior = build_prior_pool(
        molecule_manager, model_manager,
        sample_size=PHASE1_SAMPLE, seed=int(config.get("seed", 68)),
    )
    if prior.empty:
        bt.logging.error("[Solve] Phase 1 produced empty pool; aborting")
        return

    # ---- Phase 2 ---------------------------------------------------------------
    frontier = build_frontier(
        molecule_manager, model_manager,
        prior_df=prior,
        top_reactants_per_slot=PHASE2_TOP_PER_SLOT,
        max_frontier=PHASE2_MAX_FRONTIER,
        tanimoto_threshold=TANIMOTO,
    )
    if frontier.empty:
        bt.logging.error("[Solve] Phase 2 produced empty frontier; aborting")
        return

    # ---- Phase 3 ---------------------------------------------------------------
    surrogate = Surrogate(min_train=4)
    evaluated: List[dict] = []
    evaluated_smiles: Set[str] = set()

    # safety reserve: ~10s for Phase 4 + buffer
    safety = 30.0

    for round_i, (k, alpha, kappa) in enumerate(ROUND_SCHEDULE, start=1):
        remaining = deadline - time.time() - safety
        est_round_cost = k * oracle.seconds_per_call
        if remaining < est_round_cost:
            bt.logging.info(
                f"[Phase3] skipping round {round_i}: remaining={remaining:.0f}s < "
                f"est_cost={est_round_cost:.0f}s"
            )
            break

        bt.logging.info(
            f"[Phase3] === Round {round_i}/{len(ROUND_SCHEDULE)} "
            f"k={k} alpha={alpha} kappa={kappa} remaining={remaining:.0f}s ==="
        )

        batch = pick_round_batch(
            frontier=frontier,
            evaluated_smiles=evaluated_smiles,
            surrogate=surrogate,
            k=k, alpha=alpha, kappa=kappa,
            tanimoto_threshold=TANIMOTO,
        )
        if batch.empty:
            bt.logging.warning(f"[Phase3] round {round_i}: empty pick — stopping")
            break

        t0 = time.time()
        scores = oracle.score(batch["smiles"].tolist())
        bt.logging.info(
            f"[Phase3] round {round_i}: oracle({oracle.name}) "
            f"{len(batch)} mol in {time.time()-t0:.1f}s"
        )

        surrogate.add(batch["smiles"].tolist(), scores)
        surrogate.fit()

        for (_, row), s in zip(batch.iterrows(), scores):
            evaluated.append({
                "name": row["name"],
                "smiles": row["smiles"],
                "prior": float(row["prior"]),
                "oracle": float(s) if s is not None else None,
            })
            evaluated_smiles.add(row["smiles"])

        # incremental save so we always have *something* submittable
        df_eval = pd.DataFrame(evaluated)
        partial = _pick_final(df_eval, config)
        if partial:
            _save_result(partial)

    # ---- Phase 4 (final) ------------------------------------------------------
    df_eval = pd.DataFrame(evaluated)
    final = _pick_final(df_eval, config)
    if final:
        _save_result(final)
    else:
        # last-ditch: pure prior top
        bt.logging.warning("[Final] no oracle labels — falling back to top prior")
        fallback = (
            frontier.sort_values("prior", ascending=False)
            .head(int(config.get("num_molecules", 1)))["name"].tolist()
        )
        _save_result(fallback)

    bt.logging.info(
        f"[Solve] done in {time.time()-time_start:.1f}s | "
        f"evaluated={len(evaluated)} | trained={surrogate.is_trained}"
    )


# ─── cli ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    cfg = get_config()
    bt.logging.info(f"[Boot] config loaded ({len(cfg)} keys)")
    initialize_solution(cfg)
    bt.logging.info(f"[Boot] init done in {time.time()-t0:.2f}s")
    find_solution(cfg, t0)
