"""
Surrogate model: Random Forest on Morgan fingerprints, trained online from the
oracle labels collected during the active-learning loop.

Used to rank Phase-3 candidates each round by UCB:

    score = alpha * mu_hat + (1 - alpha) * prior + kappa * sigma_hat

`alpha` and `kappa` are scheduled by the caller (miner.py) so we go from
exploration (early rounds) → exploitation (final round).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import bittensor as bt

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor


_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_FP_CACHE: dict = {}


def morgan_fp(smiles: str) -> Optional[np.ndarray]:
    """2048-bit Morgan fingerprint, cached. Returns None if SMILES is invalid."""
    cached = _FP_CACHE.get(smiles)
    if cached is not None:
        return cached
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = _FP_GEN.GetFingerprint(mol)
        arr = np.zeros(2048, dtype=np.uint8)
        arr[fp.GetOnBits()] = 1
    except Exception:
        return None
    _FP_CACHE[smiles] = arr
    if len(_FP_CACHE) > 50_000:
        for k in list(_FP_CACHE.keys())[:12_500]:
            del _FP_CACHE[k]
    return arr


class Surrogate:
    """Thin RF wrapper that returns (mu, sigma) over a candidate batch."""

    def __init__(self, min_train: int = 4):
        self.model = RandomForestRegressor(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            max_samples=0.8,
        )
        self.smiles: List[str] = []
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.is_trained = False
        self.min_train = min_train

    # ---- training ---------------------------------------------------------
    def add(self, smiles_list: List[str], scores: List[Optional[float]]) -> int:
        added = 0
        for s, v in zip(smiles_list, scores):
            if v is None:
                continue
            fp = morgan_fp(s)
            if fp is None:
                continue
            self.smiles.append(s)
            self.X.append(fp)
            self.y.append(float(v))
            added += 1
        if added:
            bt.logging.info(f"[Surrogate] +{added} label(s) → {len(self.y)} total")
        return added

    def fit(self) -> bool:
        if len(self.y) < self.min_train:
            bt.logging.debug(f"[Surrogate] not enough labels ({len(self.y)}/{self.min_train})")
            self.is_trained = False
            return False
        try:
            self.model.fit(np.array(self.X), np.array(self.y))
            self.is_trained = True
            bt.logging.info(f"[Surrogate] trained on {len(self.y)} sample(s)")
            return True
        except Exception as e:
            bt.logging.warning(f"[Surrogate] fit failed: {e}")
            self.is_trained = False
            return False

    # ---- inference --------------------------------------------------------
    def predict(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mu, sigma). sigma is std across trees (uncertainty proxy)."""
        n = len(smiles_list)
        if n == 0:
            return np.array([]), np.array([])
        if not self.is_trained:
            return np.zeros(n), np.full(n, 1e6)  # huge sigma → pure exploration

        X = np.stack([
            morgan_fp(s) if morgan_fp(s) is not None else np.zeros(2048, dtype=np.uint8)
            for s in smiles_list
        ])
        try:
            tree_preds = np.array([t.predict(X) for t in self.model.estimators_])
            return tree_preds.mean(axis=0), tree_preds.std(axis=0)
        except Exception as e:
            bt.logging.warning(f"[Surrogate] predict failed: {e}")
            return np.zeros(n), np.full(n, 1e6)


def ucb_rank(
    mu: np.ndarray,
    sigma: np.ndarray,
    prior: np.ndarray,
    alpha: float,
    kappa: float,
) -> np.ndarray:
    """Combined score: surrogate vs prior blend + exploration bonus.

    alpha in [0,1] — weight given to the surrogate vs the PSICHIC prior.
    kappa >= 0    — exploration weight on sigma.
    """
    # Normalize each signal to comparable magnitudes (z-score on the batch).
    def _z(v: np.ndarray) -> np.ndarray:
        s = v.std()
        if s < 1e-9:
            return np.zeros_like(v, dtype=float)
        return (v - v.mean()) / s

    return alpha * _z(mu) + (1.0 - alpha) * _z(prior) + kappa * _z(sigma)
