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
from typing import List, Optional, Set

import numpy as np
import pandas as pd
import bittensor as bt

# --- repo + sibling solver_46 on path -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOVA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../.."))
SOLVER_46_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "solver_46"))
for p in (NOVA_DIR, SOLVER_46_DIR, BASE_DIR):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from molecules import MoleculeManager, MoleculeUtils
from models import ModelManager
from config.config_loader import load_config as load_repo_config
from utils.proteins import get_sequence_from_protein_code

from boltz_scorer import make_oracle
from selection import build_prior_pool, build_frontier, pick_round_batch
from surrogate import Surrogate


# ─── runtime config ─────────────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
DB_PATH = str(Path(NOVA_DIR) / "combinatorial_db" / "molecules.sqlite")
DEFAULT_CONFIG_PATH = str(Path(NOVA_DIR) / "config" / "config.yaml")

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
def _resolve_target_sequences(config: dict) -> list[str]:
    sequences: list[str] = []
    targets = list(config.get("small_molecule_target", []))
    clips = list(config.get("small_molecule_target_clip_interval", []))
    if len(clips) < len(targets):
        clips.extend([None] * (len(targets) - len(clips)))

    for target, clip in zip(targets, clips):
        sequence = get_sequence_from_protein_code(target, clip_interval=clip)
        if sequence:
            sequences.append(sequence)
        else:
            bt.logging.warning(f"[Config] failed to resolve sequence for target={target}")
    return sequences


def _build_boltz_targets(config: dict, sequences: list[str]) -> list[dict]:
    targets = list(config.get("small_molecule_target", []))
    clips = list(config.get("small_molecule_target_clip_interval", []))
    if len(clips) < len(targets):
        clips.extend([None] * (len(targets) - len(clips)))

    boltz_targets = []
    for idx, (target, sequence) in enumerate(zip(targets, sequences)):
        target_cfg = {
            "name": str(target),
            "sequence": sequence,
        }
        msa_map = config.get("boltz_msa_paths", {}) or {}
        msa_path = msa_map.get(target)
        if msa_path:
            target_cfg["msa_path"] = msa_path
        boltz_targets.append(target_cfg)
    return boltz_targets


def get_config(config_path: Optional[str] = None) -> dict:
    path = config_path or os.environ.get("NOVA_CONFIG_PATH") or DEFAULT_CONFIG_PATH
    config = load_repo_config(path)

    target_sequences = _resolve_target_sequences(config)
    config["target_sequences"] = target_sequences
    config.setdefault("antitarget_sequences", [])
    config.setdefault("entropy_min_threshold", float(config.get("min_entropy", 0.0)))
    config.setdefault("seed", int(os.environ.get("MINER_SEED", "68")))
    config.setdefault("rxn_id", 2)
    config.setdefault("boltz_targets", _build_boltz_targets(config, target_sequences))

    bt.logging.info(
        f"[Config] loaded {path} targets={config.get('small_molecule_target', [])} "
        f"resolved_sequences={len(target_sequences)}"
    )
    return config


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


def _candidate_dir() -> str:
    path = os.path.join(OUTPUT_DIR, "candidates")
    os.makedirs(path, exist_ok=True)
    return path


def _snapshot_names(df: pd.DataFrame) -> Set[str]:
    if df.empty or "name" not in df.columns:
        return set()
    return set(df["name"].dropna().astype(str).tolist())


def _save_candidates_snapshot(
    stage: str,
    df: pd.DataFrame,
    previous_names: Optional[Set[str]] = None,
) -> Set[str]:
    path = os.path.join(_candidate_dir(), f"{stage}.csv")
    snapshot = df.copy()
    if not snapshot.empty:
        snapshot = snapshot.reset_index(drop=True)
    snapshot.to_csv(path, index=False)

    current_names = _snapshot_names(snapshot)
    prev = previous_names or set()
    added = current_names - prev
    removed = prev - current_names
    changed = len(added) + len(removed)
    bt.logging.info(
        f"[Trace] {stage}: count={len(snapshot)} changed={changed} "
        f"added={len(added)} removed={len(removed)} → {path}"
    )
    return current_names


def _save_name_list_snapshot(
    stage: str,
    names: List[str],
    previous_names: Optional[Set[str]] = None,
) -> Set[str]:
    frame = pd.DataFrame({"name": names})
    return _save_candidates_snapshot(stage, frame, previous_names=previous_names)


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
    last_snapshot_names: Set[str] = set()

    # ---- Phase 1 ---------------------------------------------------------------
    prior = build_prior_pool(
        molecule_manager, model_manager,
        sample_size=PHASE1_SAMPLE, seed=int(config.get("seed", 68)),
    )
    if prior.empty:
        bt.logging.error("[Solve] Phase 1 produced empty pool; aborting")
        return
    last_snapshot_names = _save_candidates_snapshot("phase1_prior", prior, last_snapshot_names)

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
    last_snapshot_names = _save_candidates_snapshot("phase2_frontier", frontier, last_snapshot_names)

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
        last_snapshot_names = _save_candidates_snapshot(
            f"phase3_round_{round_i:02d}_batch",
            batch,
            last_snapshot_names,
        )

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
        last_snapshot_names = _save_candidates_snapshot(
            f"phase3_round_{round_i:02d}_evaluated",
            df_eval,
            last_snapshot_names,
        )
        partial = _pick_final(df_eval, config)
        if partial:
            last_snapshot_names = _save_name_list_snapshot(
                f"phase3_round_{round_i:02d}_partial_result",
                partial,
                last_snapshot_names,
            )
            _save_result(partial)

    # ---- Phase 4 (final) ------------------------------------------------------
    df_eval = pd.DataFrame(evaluated)
    last_snapshot_names = _save_candidates_snapshot("phase4_evaluated", df_eval, last_snapshot_names)
    final = _pick_final(df_eval, config)
    if final:
        last_snapshot_names = _save_name_list_snapshot("phase4_final_result", final, last_snapshot_names)
        _save_result(final)
    else:
        # last-ditch: pure prior top
        bt.logging.warning("[Final] no oracle labels — falling back to top prior")
        fallback = (
            frontier.sort_values("prior", ascending=False)
            .head(int(config.get("num_molecules", 1)))["name"].tolist()
        )
        last_snapshot_names = _save_name_list_snapshot("phase4_fallback_result", fallback, last_snapshot_names)
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
