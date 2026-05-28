"""
Candidate selection: the three pre-Boltz stages.

Phase 1 — PSICHIC prior
    Sample a wide pool of valid molecules from the combinatorial DB and score
    them once with the PSICHIC target model. Result: a prior ranking.

Phase 2 — Reactant-aware frontier
    Aggregate PSICHIC scores per reactant ID (slot A/B/C). Take the top-K
    reactants per slot, enumerate their cross-product, score, dedupe + diversify.
    This surfaces strong combinations that random sampling missed.

Phase 3 — Round batch pick
    Given the frontier, the already-evaluated set, and the trained surrogate,
    pick the next K to send to the expensive oracle. Diversity is enforced via
    Tanimoto-greedy on already-picked-this-round.
"""

from __future__ import annotations

import random
import time
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import bittensor as bt

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from surrogate import ucb_rank, Surrogate


def _fp_bv(smiles: str):
    """RDKit ExplicitBitVect for Tanimoto. None if SMILES invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _FP_GEN.GetFingerprint(mol)


# Lazy: solver_46 utilities sit in a sibling dir and are added to sys.path by miner.py.
# Importing inside functions to avoid import-time failure when this module is loaded
# in isolation (e.g. tooling/linting).
def _utils():
    from molecules import MoleculeUtils
    return MoleculeUtils


_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ─── Phase 1 ─────────────────────────────────────────────────────────────────
def build_prior_pool(
    molecule_manager,
    model_manager,
    sample_size: int = 4000,
    seed: int = 68,
) -> pd.DataFrame:
    """
    Sample `sample_size` random valid molecules from the DB, get SMILES + PSICHIC.
    Returns df with columns [name, smiles, prior].
    """
    t0 = time.time()
    bt.logging.info(f"[Phase1] sampling {sample_size} candidates from DB …")

    rng = random.Random(seed)
    rxn_id = molecule_manager.rxn_id
    A_ids = molecule_manager.moles_A_id
    B_ids = molecule_manager.moles_B_id
    C_ids = molecule_manager.moles_C_id if molecule_manager.is_three_component else None

    seen: Set[str] = set()
    rows: List[Tuple[str, str]] = []
    attempts = 0
    max_attempts = sample_size * 6
    MU = _utils()

    while len(rows) < sample_size and attempts < max_attempts:
        attempts += 1
        a = rng.choice(A_ids)
        b = rng.choice(B_ids)
        if C_ids:
            c = rng.choice(C_ids)
            name = f"rxn:{rxn_id}:{a}:{b}:{c}"
        else:
            name = f"rxn:{rxn_id}:{a}:{b}"
        if name in seen:
            continue
        seen.add(name)
        smi = MU.get_smiles_from_reaction_cached(name)
        if smi:
            rows.append((name, smi))

    df = pd.DataFrame(rows, columns=["name", "smiles"])
    if df.empty:
        bt.logging.warning("[Phase1] no valid candidates produced — empty prior pool")
        return df

    bt.logging.info(f"[Phase1] PSICHIC-scoring {len(df)} candidates …")
    target = model_manager.get_target_score_from_data(df["smiles"])
    df["prior"] = target.values if hasattr(target, "values") else list(target)
    df = df.dropna(subset=["prior"]).reset_index(drop=True)
    bt.logging.info(
        f"[Phase1] done in {time.time()-t0:.1f}s — {len(df)} scored "
        f"(prior min={df['prior'].min():.4f}, max={df['prior'].max():.4f})"
    )
    return df


# ─── Phase 2 ─────────────────────────────────────────────────────────────────
def build_frontier(
    molecule_manager,
    model_manager,
    prior_df: pd.DataFrame,
    top_reactants_per_slot: int = 30,
    max_frontier: int = 1500,
    tanimoto_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Reactant-aware reassembly: keep top combos of strong reactant pieces.
    Returns df with columns [name, smiles, prior].
    """
    if prior_df.empty:
        return prior_df

    t0 = time.time()
    rxn_id = molecule_manager.rxn_id
    three = molecule_manager.is_three_component
    bt.logging.info(
        f"[Phase2] reactant aggregation (top_per_slot={top_reactants_per_slot}, "
        f"three_component={three})"
    )

    # parse name → (A, B, C)
    def _parse(n: str):
        p = n.split(":")
        if len(p) < 4:
            return None
        try:
            a, b = int(p[2]), int(p[3])
            c = int(p[4]) if len(p) > 4 else None
            return a, b, c
        except ValueError:
            return None

    parts = prior_df["name"].apply(_parse)
    valid = parts.notna()
    df = prior_df[valid].copy()
    if df.empty:
        bt.logging.warning("[Phase2] no parseable names; skipping")
        return df
    df[["A", "B", "C"]] = pd.DataFrame(parts[valid].tolist(), index=df.index)

    top_A = df.groupby("A")["prior"].mean().nlargest(top_reactants_per_slot).index.tolist()
    top_B = df.groupby("B")["prior"].mean().nlargest(top_reactants_per_slot).index.tolist()
    top_C = (
        df.dropna(subset=["C"]).groupby("C")["prior"].mean().nlargest(top_reactants_per_slot).index.tolist()
        if three else [None]
    )
    bt.logging.info(
        f"[Phase2] top reactants: |A|={len(top_A)} |B|={len(top_B)} |C|={len(top_C)}"
    )

    MU = _utils()
    seen = set(prior_df["name"].tolist())
    new_rows: List[Tuple[str, str]] = []
    for a in top_A:
        for b in top_B:
            for c in top_C:
                if c is None:
                    name = f"rxn:{rxn_id}:{a}:{b}"
                else:
                    name = f"rxn:{rxn_id}:{a}:{b}:{int(c)}"
                if name in seen:
                    continue
                seen.add(name)
                smi = MU.get_smiles_from_reaction_cached(name)
                if smi:
                    new_rows.append((name, smi))
                if len(new_rows) >= max_frontier:
                    break
            if len(new_rows) >= max_frontier:
                break
        if len(new_rows) >= max_frontier:
            break

    if not new_rows:
        bt.logging.info("[Phase2] no new combinations — returning prior pool")
        return prior_df

    new_df = pd.DataFrame(new_rows, columns=["name", "smiles"])
    bt.logging.info(f"[Phase2] PSICHIC-scoring {len(new_df)} new combinations …")
    target = model_manager.get_target_score_from_data(new_df["smiles"])
    new_df["prior"] = target.values if hasattr(target, "values") else list(target)
    new_df = new_df.dropna(subset=["prior"])

    merged = pd.concat([prior_df[["name", "smiles", "prior"]], new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["name"]).reset_index(drop=True)

    # Diversity-trim to keep frontier focused
    merged = _diversify(merged, max_keep=max_frontier * 2, threshold=tanimoto_threshold)
    bt.logging.info(
        f"[Phase2] frontier ready in {time.time()-t0:.1f}s — {len(merged)} molecules"
    )
    return merged


# ─── Phase 3 ─────────────────────────────────────────────────────────────────
def pick_round_batch(
    frontier: pd.DataFrame,
    evaluated_smiles: Set[str],
    surrogate: Surrogate,
    k: int,
    alpha: float,
    kappa: float,
    tanimoto_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Rank frontier by UCB, then greedy-pick K with batch diversity.

    Skips molecules already evaluated. Returns rows ready to send to the oracle.
    """
    if frontier.empty or k <= 0:
        return frontier.head(0)

    pool = frontier[~frontier["smiles"].isin(evaluated_smiles)].reset_index(drop=True)
    if pool.empty:
        bt.logging.warning("[Phase3] frontier exhausted vs evaluated set")
        return pool

    mu, sigma = surrogate.predict(pool["smiles"].tolist())
    prior = pool["prior"].to_numpy(dtype=float)
    ucb = ucb_rank(mu, sigma, prior, alpha=alpha, kappa=kappa)

    pool = pool.copy()
    pool["_mu"] = mu
    pool["_sigma"] = sigma
    pool["_ucb"] = ucb
    pool = pool.sort_values("_ucb", ascending=False).reset_index(drop=True)

    picked_idx: List[int] = []
    picked_fps: list = []
    for i, row in pool.iterrows():
        if len(picked_idx) >= k:
            break
        fp = _fp_bv(row["smiles"])
        if fp is None:
            continue
        too_close = False
        for prev in picked_fps:
            if DataStructs.TanimotoSimilarity(fp, prev) > tanimoto_threshold:
                too_close = True
                break
        if too_close:
            continue
        picked_idx.append(i)
        picked_fps.append(fp)

    chosen = pool.loc[picked_idx].reset_index(drop=True)
    top_ucb = chosen["_ucb"].iloc[0] if not chosen.empty else float("nan")
    bt.logging.info(
        f"[Phase3] picked {len(chosen)}/{k} | alpha={alpha:.2f} kappa={kappa:.2f} | "
        f"top _ucb={top_ucb:.3f}"
    )
    return chosen


# ─── shared diversity helper ─────────────────────────────────────────────────
def _diversify(df: pd.DataFrame, max_keep: int, threshold: float) -> pd.DataFrame:
    """Greedy Tanimoto trim. Keeps the highest-prior molecules under the threshold."""
    if df.empty or len(df) <= max_keep:
        return df
    df = df.sort_values("prior", ascending=False).reset_index(drop=True)
    kept_idx: List[int] = []
    kept_fps: list = []
    for i, row in df.iterrows():
        fp = _fp_bv(row["smiles"])
        if fp is None:
            continue
        ok = True
        for p in kept_fps:
            if DataStructs.TanimotoSimilarity(fp, p) > threshold:
                ok = False
                break
        if ok:
            kept_idx.append(i)
            kept_fps.append(fp)
            if len(kept_idx) >= max_keep:
                break
    return df.loc[kept_idx].reset_index(drop=True)
