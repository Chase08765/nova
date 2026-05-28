"""
DPEX_DJA – Dual-Population EXchange with Discrete Jaya Algorithm
=================================================================
Population A  – global exploration via DJA update rule (discrete Jaya)
Population B  – local refinement via tabu-enhanced neighbourhood search
Exchange      – periodically injects best-of-A into B, evicts worst-of-B

Reference pseudocode: DPEX_DJA_algorithm.md

Algorithm structure
-------------------
FOR each iteration t:
    Part A  : apply DJA update to every member of pop_A
                  ai' = ai + r1*(best_A - |ai|) - r2*(worst_A - |ai|)
              (discrete: probabilistic component-slot attraction/repulsion)
    Part B  : tabu-enhanced local search on pop_B elites
              generate k neighbours per elite via synthon similarity,
              block tabu moves unless aspiration holds
    Part C  : every T_ex iters, exchange m best-of-A into B
              (pop_B is trimmed to N_B after merge)
    Global  : accumulate all scored candidates into top_pool
"""
from __future__ import annotations

import random
import bittensor as bt
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from molecules import MoleculeManager, MoleculeUtils
from tools import SynthonLibrary

# Global ranker weights (set by miner before each DJA call) — zero-overhead injection
_rw_A = None  # pre-normalized np.array for moles_A_id
_rw_B = None
_rw_C = None

def set_ranker_weights(w_A, w_B, w_C=None):
    global _rw_A, _rw_B, _rw_C
    _rw_A, _rw_B, _rw_C = w_A, w_B, w_C

def _smart_choice(pool, role='A'):
    """Ranker-weighted random — same speed as random.choice when no weights."""
    w = {'A': _rw_A, 'B': _rw_B, 'C': _rw_C}.get(role)
    if w is not None and len(w) == len(pool):
        return pool[np.random.choice(len(pool), p=w)]
    return random.choice(pool)

# ── tunables ──────────────────────────────────────────────────────────────────
N_A_DEFAULT  = 500   # population A capacity  (moving-window of scored mols)
N_B_DEFAULT  = 100   # population B capacity  (elite pool for tabu search)
T_EX_DEFAULT = 3     # exchange every T_ex iterations
M_EX_DEFAULT = 20    # molecules exchanged per cycle
TABU_MAXLEN  = 20    # maximum tabu entries per component role
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DPEXDJAState:
    """Persistent DPEX_DJA state carried across iterations."""
    pop_A:    List[Dict]        = field(default_factory=list)
    pop_B:    List[Dict]        = field(default_factory=list)
    tabu:     Dict[str, deque]  = field(default_factory=lambda: {
        'A': deque(maxlen=TABU_MAXLEN),
        'B': deque(maxlen=TABU_MAXLEN),
        'C': deque(maxlen=TABU_MAXLEN),
    })
    N_A:      int = N_A_DEFAULT
    N_B:      int = N_B_DEFAULT
    T_ex:     int = T_EX_DEFAULT
    m_ex:     int = M_EX_DEFAULT
    iteration: int = 0
    global_seen: Set[str] = field(default_factory=set)  # 5️⃣ Duplicate prevention
    global_best_molecule: Optional[Dict] = None  # 🔟 Global best tracking
    global_best_score: float = float('-inf')  # 🔟 Global best tracking

def _parse(name: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None, None
    try:
        return (
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]) if len(parts) > 4 else None,
        )
    except (ValueError, IndexError):
        return None, None, None, None


def _build(rxn: int, A: int, B: int, C: Optional[int]) -> str:
    return f"rxn:{rxn}:{A}:{B}" if C is None else f"rxn:{rxn}:{A}:{B}:{C}"

def _dja_move(
    name:      str,
    best_name: str,
    worst_name: str,
    manager:   MoleculeManager,
    avoid:     Set[str],
    exploration_rate: float = 0.25,  # 9️⃣ Iteration-dependent exploration
) -> Optional[str]:

    rxn, A,  B,  C  = _parse(name)
    _,   bA, bB, bC = _parse(best_name)
    _,   wA, wB, wC = _parse(worst_name)

    if rxn is None or bA is None or wA is None:
        return None

    # 2️⃣ Better DJA discrete move rule
    def _step(cur: int, best: int, worst: int, pool: List[int], role: str = 'A') -> int:
        r = random.random()

        explore_prob = exploration_rate * 0.25
        exploit_prob = 0.6 + (0.25 - explore_prob)

        if r < exploit_prob:
            return best
        elif r < exploit_prob + explore_prob:
            return _smart_choice(pool, role)  # ranker-weighted exploration
        else:
            return cur

    nA = _step(A, bA, wA, manager.moles_A_id, 'A')
    nB = _step(B, bB, wB, manager.moles_B_id, 'B')

    nC: Optional[int] = None
    if manager.is_three_component and C is not None:
        nC = _step(
            C,
            bC if bC is not None else C,
            wC if wC is not None else C,
            manager.moles_C_id,
            'C',
        )

    new_name = _build(rxn, nA, nB, nC)
    
    # 3️⃣ Add mutation operator (10% probability)
    if random.random() < 0.15:
        components = ['A', 'B']
        if manager.is_three_component and nC is not None:
            components.append('C')
        
        mutate_component = random.choice(components)
        if mutate_component == 'A':
            nA = _smart_choice(manager.moles_A_id, 'A')
        elif mutate_component == 'B':
            nB = _smart_choice(manager.moles_B_id, 'B')
        elif mutate_component == 'C' and nC is not None:
            nC = _smart_choice(manager.moles_C_id, 'C')
        
        new_name = _build(rxn, nA, nB, nC)
    
    return None if (new_name == name or new_name in avoid) else new_name


def dja_generate(
    state:    DPEXDJAState,
    manager:  MoleculeManager,
    n_samples: int,
    avoid:    Set[str],
    surrogate=None,
) -> pd.DataFrame:

    if not state.pop_A:
        return pd.DataFrame(columns=["name"])

    # 9️⃣ Iteration-dependent exploration
    exploration_rate = max(0.15, 1.5 - state.iteration / 80)

    by_score  = sorted(state.pop_A, key=lambda x: x.get('score', float('-inf')), reverse=True)
    best_mol  = by_score[0]
    worst_mol = by_score[-1]

    new_names: Set[str] = set()

    for mol in state.pop_A:
        if len(new_names) >= n_samples:
            break
        n = _dja_move(mol['name'], best_mol['name'], worst_mol['name'], manager, avoid, exploration_rate)
        # 5️⃣ Duplicate prevention
        if n and n not in state.global_seen:
            new_names.add(n)

    attempts = 0
    while len(new_names) < n_samples and attempts < n_samples * 4:
        attempts += 1
        mol = random.choice(state.pop_A)
        n = _dja_move(mol['name'], best_mol['name'], worst_mol['name'], manager, avoid, exploration_rate)
        # 5️⃣ Duplicate prevention
        if n and n not in new_names and n not in state.global_seen:
            new_names.add(n)

    if not new_names:
        return pd.DataFrame(columns=["name"])
    result_df = pd.DataFrame({"name": list(new_names)})
    # Surrogate-guided pre-validation filtering: reduce validation/GPU load
    if surrogate is not None and getattr(surrogate, 'is_trained', False):
        result_df['smiles'] = result_df['name'].map(MoleculeUtils.get_smiles_from_reaction_cached)
        result_df = result_df[result_df['smiles'].notna()]
        if not result_df.empty:
            # Aggressive filtering: keep top 50% or up to 2x requested samples
            n_keep = max(int(n_samples * 0.5), min(len(result_df), n_samples * 2))
            result_df = surrogate.filter_candidates(result_df, n_keep=n_keep, smiles_col="smiles")
            return result_df[["name"]]
    return pd.DataFrame({"name": list(new_names)})

def _tabu_hit(tabu_set: Set[Tuple[int, int]], old_id: int, new_id: int) -> bool:
    return (old_id, new_id) in tabu_set

def tabu_generate(
    state:             DPEXDJAState,
    synthon_lib:       SynthonLibrary,
    manager:           MoleculeManager,
    avoid:             Set[str],
    k_per_elite:       int   = 15,
    k_elites:          int   = 10,
    global_best_score: float = float('-inf'),
    tabued_molecules:  Set[str] = set(),
) -> Tuple[pd.DataFrame, List[Tuple[str, int, int]]]:

    if not state.pop_B or synthon_lib is None:
        return pd.DataFrame(columns=["name"]), []

    # 6️⃣ Adaptive tabu list
    adaptive_tabu_len = min(200, 20 + state.iteration * 2)
    for role in ('A', 'B', 'C'):
        if role in state.tabu:
            # Update maxlen by creating new deque
            state.tabu[role] = deque(state.tabu[role], maxlen=adaptive_tabu_len)

    tabu_sets: Dict[str, Set] = {r: set(state.tabu[r]) for r in ('A', 'B', 'C')}

    new_names:    List[str]                   = []
    applied_moves: List[Tuple[str, int, int]] = []

    n_elites = min(k_elites, len(state.pop_B))

    # 7️⃣ Score-aware elite selection (probabilistic)
    scores = [max(0, mol.get('score', 0)) for mol in state.pop_B]
    total_score = sum(scores)
    
    if total_score > 0:
        # Weighted random sampling
        weights = [s / total_score for s in scores]
        elite_indices = random.choices(range(len(state.pop_B)), weights=weights, k=n_elites)
        elites = [state.pop_B[i] for i in elite_indices]
    else:
        # Fallback to uniform random if all scores are non-positive
        elites = random.choices(state.pop_B, k=n_elites) if state.pop_B else []

    for mol in elites:

        if mol["name"] in tabued_molecules:
            continue

        rxn, A, B, C = _parse(mol['name'])
        mol_score    = mol.get('score', float('-inf'))
        if rxn is None:
            continue

        # 🔟 Use global best for aspiration
        aspiration = (
            state.global_best_score > float('-inf')
            and mol_score >= state.global_best_score * 0.9
        )

        similar = synthon_lib.find_similar_to_molecule_name(
            mol['name'],
            vary_component='both' if not manager.is_three_component else 'all',
            top_k_per_component=k_per_elite,
            min_similarity=0.50,
        )

        for new_A in similar.get('A', [])[:k_per_elite]:
            nn = _build(rxn, new_A, B, C)
            if _tabu_hit(tabu_sets['A'], A, new_A) and not aspiration:
                continue
            # 5️⃣ Duplicate prevention
            if nn not in avoid and nn not in new_names and nn not in state.global_seen:
                new_names.append(nn)
                applied_moves.append(('A', A, new_A))

        for new_B in similar.get('B', [])[:k_per_elite]:
            nn = _build(rxn, A, new_B, C)
            if _tabu_hit(tabu_sets['B'], B, new_B) and not aspiration:
                continue
            # 5️⃣ Duplicate prevention
            if nn not in avoid and nn not in new_names and nn not in state.global_seen:
                new_names.append(nn)
                applied_moves.append(('B', B, new_B))

        if manager.is_three_component and C is not None:
            for new_C in similar.get('C', [])[:k_per_elite]:
                nn = _build(rxn, A, B, new_C)
                if _tabu_hit(tabu_sets['C'], C, new_C) and not aspiration:
                    continue
                # 5️⃣ Duplicate prevention
                if nn not in avoid and nn not in new_names and nn not in state.global_seen:
                    new_names.append(nn)
                    applied_moves.append(('C', C, new_C))

    return (
        pd.DataFrame({"name": new_names}) if new_names else pd.DataFrame(columns=["name"]),
        applied_moves,
    )


def update_tabu(state: DPEXDJAState, moves: List[Tuple[str, int, int]]) -> None:
    for role, old_id, new_id in moves:
        if role in state.tabu:
            state.tabu[role].append((old_id, new_id))

def dpex_exchange(state: DPEXDJAState) -> None:
    if not state.pop_A or state.m_ex <= 0:
        return

    # 4️⃣ Bidirectional exchange (A → B and B → A)
    best_of_A = sorted(
        state.pop_A,
        key=lambda x: x.get('score', float('-inf')),
        reverse=True,
    )[:state.m_ex]

    best_of_B = sorted(
        state.pop_B,
        key=lambda x: x.get('score', float('-inf')),
        reverse=True,
    )[:state.m_ex]

    # Merge best_of_A into pop_B
    seen_B:   Set[str]   = set()
    merged_B: List[Dict] = []
    for mol in list(state.pop_B) + best_of_A:
        if mol['name'] not in seen_B:
            seen_B.add(mol['name'])
            merged_B.append(mol)

    merged_B.sort(key=lambda x: x.get('score', float('-inf')), reverse=True)
    state.pop_B = merged_B[:state.N_B]

    # Merge best_of_B into pop_A
    combined_A = {mol['name']: mol for mol in state.pop_A}
    for mol in best_of_B:
        if mol['name'] not in combined_A or mol.get('score', float('-inf')) > combined_A[mol['name']].get('score', float('-inf')):
            combined_A[mol['name']] = mol
    
    state.pop_A = sorted(
        combined_A.values(),
        key=lambda x: x.get('score', float('-inf')),
        reverse=True,
    )[:state.N_A]

    bt.logging.info(
        f"[DPEX] Bidirectional Exchange: {state.m_ex} best A→B, {state.m_ex} best B→A  |  pop_A={len(state.pop_A)}, pop_B={len(state.pop_B)}"
    )

def update_populations(
    state:    DPEXDJAState,
    scored_A: pd.DataFrame,
    scored_B: pd.DataFrame,
) -> None:

    _required = ('name', 'smiles', 'score')

    def _to_records(df: pd.DataFrame) -> List[Dict]:
        if df.empty or not all(c in df.columns for c in _required):
            return []
        cols = [c for c in ('name', 'smiles', 'score', 'target', 'anti') if c in df.columns]
        return df[cols].dropna(subset=['score']).to_dict('records')

    # 1️⃣ Population memory - merge instead of replace
    if not scored_A.empty:
        new_A = _to_records(scored_A)
        combined_A = {mol['name']: mol for mol in state.pop_A}
        
        for mol in new_A:
            # Keep higher score if duplicate
            if mol['name'] not in combined_A or mol.get('score', float('-inf')) > combined_A[mol['name']].get('score', float('-inf')):
                combined_A[mol['name']] = mol
        
        state.pop_A = sorted(
            combined_A.values(),
            key=lambda x: x.get('score', float('-inf')),
            reverse=True,
        )[:state.N_A]

    new_B = _to_records(scored_B)
    if new_B:
        by_name = {mol['name']: mol for mol in state.pop_B}
        for mol in new_B:
            by_name[mol['name']] = mol
        state.pop_B = sorted(
            by_name.values(),
            key=lambda x: x.get('score', float('-inf')),
            reverse=True,
        )[:state.N_B]

    # 5️⃣ Update global seen set
    for mol in _to_records(scored_A) + _to_records(scored_B):
        state.global_seen.add(mol['name'])

    # 🔟 Update global best
    all_mols = state.pop_A + state.pop_B
    if all_mols:
        best_mol = max(all_mols, key=lambda x: x.get('score', float('-inf')))
        best_score = best_mol.get('score', float('-inf'))
        
        if best_score > state.global_best_score:
            state.global_best_score = best_score
            state.global_best_molecule = best_mol
            bt.logging.info(f"[DPEX] New global best: {best_mol['name']} with score {best_score:.4f}")