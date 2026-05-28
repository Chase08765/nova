import math
import time
import random
import pandas as pd
import bittensor as bt
import numpy as np
from itertools import chain
from typing import List, Tuple, Dict
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit import Chem, DataStructs
from collections import defaultdict

from molecules import MoleculeManager, MoleculeUtils, MoleculeUtils

MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

class IterationParams:
    def __init__(self, config: dict):
        self.seen_molecules = set()
        self.use_synthon_search = False
        self.use_exploit_mode = False
        self.base_samples = 500
        self.n_samples_start = self.base_samples * 4
        self.synthon_lib = None
        self.no_improvement_counter = 0
        self.score_improvement_rate = 0.0
        self.mutation_prob = 0.40
        self.elite_prob = 0.60
        self.use_exploit_mode = False
        self.exploited_reactants = set()
        
    def get_nsamples_from_time(self, remaining_time: float) -> int:
        if remaining_time > 750:
            n_samples = self.base_samples
        elif remaining_time > 450:
            n_samples = int(self.base_samples * 0.95)
        elif remaining_time > 300:
            n_samples = int(self.base_samples * 0.90)
        elif remaining_time > 150:
            n_samples = int(self.base_samples * 0.85)
        else:
            n_samples = int(self.base_samples * 0.80)
        return n_samples

class SynthonLibrary:
    def __init__(self, molecule_manager: MoleculeManager):
        self.molecule_manager = molecule_manager
        self.fps_A = SynthonLibrary._build_fingerprint_index(self.molecule_manager.molecules_A)
        self.fps_B = SynthonLibrary._build_fingerprint_index(self.molecule_manager.molecules_B)
        self.fps_C = SynthonLibrary._build_fingerprint_index(self.molecule_manager.molecules_C) if self.molecule_manager.is_three_component else {}

        bt.logging.info(f"[Solution] SynthonLibrary initialized: {len(self.fps_A)} A components, "
                       f"{len(self.fps_B)} B components" +
                       (f", {len(self.fps_C)} C components" if self.molecule_manager.is_three_component else ""))

    @staticmethod
    def _build_fingerprint_index(molecules: List[Tuple[int, str, int]]) -> Dict[int, object]:
        fps = {}
        for mol_id, smiles, _ in molecules:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol:
                fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                fps[mol_id] = fp
        return fps

    def find_similar_components(
        self,
        target_smiles: str,
        role: str = 'A',
        top_k: int = 80,
        min_similarity: float = 0.5
    ) -> List[Tuple[int, float]]:
        
        target_mol = MoleculeUtils.mol_from_smiles_cached(target_smiles)
        if not target_mol: return []
        target_fp = MORGAN_FP_GENERATOR.GetFingerprint(target_mol)

        if role == 'A': fps_dict = self.fps_A
        elif role == 'B': fps_dict = self.fps_B
        elif role == 'C' and self.molecule_manager.is_three_component: fps_dict = self.fps_C
        else: return []

        similarities = []
        for mol_id, fp in fps_dict.items():
            try:
                sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                if sim >= min_similarity:
                    similarities.append((mol_id, sim))
            except Exception:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_similar_to_molecule_name(
        self,
        molecule_name: str,
        vary_component: str = 'both',
        top_k_per_component: int = 10,
        min_similarity: float = 0.6
    ) -> Dict[str, List[int]]:
        parts = molecule_name.split(":")
        if len(parts) < 4:
            return {}

        try:
            if len(parts) == 4:
                _, rxn, A_id, B_id = parts
                A_id, B_id = int(A_id), int(B_id)
                C_id = None
            else:
                _, rxn, A_id, B_id, C_id = parts
                A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)
        except (ValueError, IndexError):
            return {}

        result = {}

        if vary_component in ['A', 'both', 'all']:
            A_smiles = self._get_component_smiles(A_id, 'A')
            if A_smiles:
                similar_As = self.find_similar_components(
                    A_smiles, 'A', top_k_per_component, min_similarity
                )
                result['A'] = [mol_id for mol_id, _ in similar_As if mol_id != A_id]

        if vary_component in ['B', 'both', 'all']:
            B_smiles = self._get_component_smiles(B_id, 'B')
            if B_smiles:
                similar_Bs = self.find_similar_components(
                    B_smiles, 'B', top_k_per_component, min_similarity
                )
                result['B'] = [mol_id for mol_id, _ in similar_Bs if mol_id != B_id]

        if self.molecule_manager.is_three_component and C_id and vary_component in ['C', 'all']:
            C_smiles = self._get_component_smiles(C_id, 'C')
            if C_smiles:
                similar_Cs = self.find_similar_components(
                    C_smiles, 'C', top_k_per_component, min_similarity
                )
                result['C'] = [mol_id for mol_id, _ in similar_Cs if mol_id != C_id]

        return result

    def _get_component_smiles(self, mol_id: int, role: str) -> str:
        if role == 'A': molecules = self.molecule_manager.molecules_A
        elif role == 'B': molecules = self.molecule_manager.molecules_B
        elif role == 'C': molecules = self.molecule_manager.molecules_C
        else: return None

        for mid, smiles, _ in molecules:
            if mid == mol_id:
                return smiles
        return None

    def generate_similar_molecules(
        self,
        base_molecule_names: List[str],
        n_per_base: int = 5,
        min_similarity: float = 0.6
    ) -> List[str]:
        new_molecules = []
        is_single_molecule = len(base_molecule_names) == 1
        
        if is_single_molecule:
            if n_per_base >= 80:
                effective_n_per_base = n_per_base
            else:
                effective_n_per_base = n_per_base * 3
        else:
            effective_n_per_base = n_per_base

        for base_name in base_molecule_names:
            parts = base_name.split(":")
            if len(parts) < 4:
                continue
            try:
                if len(parts) == 4:
                    _, rxn, A_id, B_id = parts
                    A_id, B_id = int(A_id), int(B_id)

                    similar_comps = self.find_similar_to_molecule_name(
                        base_name, 'both', effective_n_per_base, min_similarity
                    )

                    for new_A in similar_comps.get('A', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}")

                    for new_B in similar_comps.get('B', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}")

                else:
                    _, rxn, A_id, B_id, C_id = parts
                    A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)

                    similar_comps = self.find_similar_to_molecule_name(
                        base_name, 'all', effective_n_per_base, min_similarity
                    )

                    for new_A in similar_comps.get('A', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}:{C_id}")

                    for new_B in similar_comps.get('B', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}:{C_id}")

                    for new_C in similar_comps.get('C', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{B_id}:{new_C}")

            except (ValueError, IndexError) as e:
                bt.logging.warning(f"Could not parse molecule name {base_name}: {e}")
                continue

        return list(dict.fromkeys(new_molecules))

def generate_molecules_from_synthon_library(
    synthon_lib: SynthonLibrary,
    top_molecules: pd.DataFrame,
    n_samples: int,
    min_similarity: float = 0.6,
    n_per_base: int = 10
) -> pd.DataFrame:

    if top_molecules.empty:
        return pd.DataFrame(columns=["name"])

    if len(top_molecules) == 1:
        seed_names = top_molecules["name"].tolist()
        if n_per_base >= 80:
            effective_n_per_base = n_per_base
        else:
            effective_n_per_base = n_per_base * 4
    else:
        n_seeds = min(30, len(top_molecules))
        seed_names = top_molecules.head(n_seeds)["name"].tolist()
        effective_n_per_base = n_per_base

    new_names = synthon_lib.generate_similar_molecules(
        seed_names,
        n_per_base=effective_n_per_base,
        min_similarity=min_similarity
    )

    if not new_names: return pd.DataFrame(columns=["name"])

    if len(new_names) > n_samples * 3.0:
        new_names = random.sample(new_names, int(n_samples * 2.0))

    return pd.DataFrame({"name": new_names})

def generate_offspring_from_elites(
    manager,
    samples: int,
    mutation_prob: float,
    elites_A=None,
    elites_B=None,
    elites_C=None,
    avoid_names=None,
    seen_names=None,
):
    if samples <= 0:
        return set()

    rxn_id = manager.rxn_id
    is_three = manager.is_three_component
    max_retries = 10

    elites_A = elites_A or []
    elites_B = elites_B or []
    elites_C = elites_C or []
    avoid_names = avoid_names or set()
    seen_names = seen_names or set()

    moles_A = manager.moles_A_id
    moles_B = manager.moles_B_id
    moles_C = manager.moles_C_id if is_three else None

    offsprings = set()

    def pick(moles, elites):
        if not elites or random.random() < mutation_prob:
            return random.choice(moles)
        return random.choice(elites)

    for _ in range(samples):
        for __ in range(max_retries):
            A = pick(moles_A, elites_A)
            B = pick(moles_B, elites_B)

            if is_three:
                C = pick(moles_C, elites_C)
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"

            if (
                name not in offsprings
                and name not in seen_names
                and name not in avoid_names
            ):
                offsprings.add(name)
                break

    return offsprings 

def generate_molecules_from_pools(manager: MoleculeManager, n_samples: int, component_weights: dict = None) -> List[str]:
    rxn_id = manager.rxn_id
    if component_weights:
        weights_A = [component_weights.get('A', {}).get(aid, 1.0) for aid in manager.moles_A_id]
        weights_B = [component_weights.get('B', {}).get(bid, 1.0) for bid in manager.moles_B_id]
        weights_C = [component_weights.get('C', {}).get(cid, 1.0) for cid in manager.moles_C_id] if manager.is_three_component else None
        
        if weights_A:
            sum_w = sum(weights_A)
            weights_A = [w / sum_w if sum_w > 0 else 1.0/len(weights_A) for w in weights_A]
        if weights_B:
            sum_w = sum(weights_B)
            weights_B = [w / sum_w if sum_w > 0 else 1.0/len(weights_B) for w in weights_B]
        if weights_C:
            sum_w = sum(weights_C)
            weights_C = [w / sum_w if sum_w > 0 else 1.0/len(weights_C) for w in weights_C]
        
        picks_A = random.choices(manager.moles_A_id, weights=weights_A, k=n_samples) if weights_A else random.choices(manager.moles_A_id, k=n_samples)
        picks_B = random.choices(manager.moles_B_id, weights=weights_B, k=n_samples) if weights_B else random.choices(manager.moles_B_id, k=n_samples)
        if manager.is_three_component:
            picks_C = random.choices(manager.moles_C_id, weights=weights_C, k=n_samples) if weights_C else random.choices(manager.moles_C_id, k=n_samples)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    else:
        picks_A = random.choices(manager.moles_A_id, k=n_samples)
        picks_B = random.choices(manager.moles_B_id, k=n_samples)
        if manager.is_three_component:
            picks_C = random.choices(manager.moles_C_id, k=n_samples)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    
    names = set(dict.fromkeys(names))
    return names

def generate_valid_random_molecules(
    config: dict,
    manager: MoleculeManager,
    n_samples: int,
    mutation_prob: float,
    elite_prob: float,
    executor,
    n_workers: int,
    avoid_names: set[str] = set(),
    elite_names: List[str] | None = None,
    component_weights: dict | None = None,
    batch_size: int = 200,
):
    elites_A = set()
    elites_B = set()
    elites_C = set()
    if elite_names is not None:
        for elite in elite_names:
            A, B, C = MoleculeUtils.parse_components(elite)
            if A is not None: elites_A.add(A)
            if B is not None: elites_B.add(B)
            if C is not None and manager.is_three_component: elites_C.add(C)
    
    elites_A = list(elites_A)
    elites_B = list(elites_B)
    elites_C = list(elites_C)
    
    bt.logging.info(f"{len(elites_A)} elite A components, {len(elites_B)} elite B components" + (f", {len(elites_C)} elite C components" if manager.is_three_component else ""))
    
    n_valid = 0
    valid_molecules = []
    seen_names = set()
    
    while n_valid < n_samples:
        needed = n_samples - n_valid
        actual_batch_size = min(max(batch_size, 300), needed * 2)
        batch_names = set()
        if elite_names:
            n_elites = max(0, min(actual_batch_size, int(actual_batch_size * elite_prob)))
            elite_batch = generate_offspring_from_elites(manager, n_elites, mutation_prob, elites_A, elites_B, elites_C, avoid_names, seen_names)
            n_rand = actual_batch_size - len(elite_batch)
            rand_batch = generate_molecules_from_pools(manager, n_rand, component_weights) - seen_names - elite_batch - avoid_names
            batch_names = elite_batch | rand_batch
        else:
            batch_names = generate_molecules_from_pools(manager, actual_batch_size, component_weights) - seen_names - avoid_names
        
        if not batch_names:
            continue
        
        batch_df = pd.DataFrame({"name": list(batch_names)})
        batch_df = manager.validate_molecules(config, batch_df)
        if batch_df.empty: continue
        
        seen_names = seen_names | set(batch_df["name"])
        n_valid = len(seen_names)
        valid_molecules.append(batch_df[["name", "smiles"]])
    
    result_df = pd.concat(valid_molecules, ignore_index = True)
    return result_df.head(n_samples)

def compute_tanimoto_similarity_to_pool(
    candidate_smiles: pd.Series,
    pool_smiles: pd.Series,
) -> pd.Series:
    if candidate_smiles.empty or pool_smiles.empty:
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    pool_fps = [
        fp
        for smi in pool_smiles.dropna().unique()
        if (fp := MoleculeUtils.maccs_fp_from_smiles_cached(smi)) is not None
    ]

    if not pool_fps:
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    result = pd.Series(0.0, index=candidate_smiles.index, dtype=float)
    for idx, smi in candidate_smiles.items():
        fp_cand = MoleculeUtils.maccs_fp_from_smiles_cached(smi)
        if fp_cand is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_cand, pool_fps)
        result.at[idx] = max(sims)

    return result

seen_cache = {}
def sample_random_valid_molecules(
    manager: MoleculeManager,
    n_samples: int,
    config: dict,
    avoid_names: set[str] | None = None,
    focus_neighborhood_of: pd.DataFrame | None = None,
) -> pd.DataFrame:
    global seen_cache

    names = []
    for name in focus_neighborhood_of["name"]:
        try:
            parts = name.split(":")
            if len(parts) == 4:
                rxn_prefix, rxn_type, comp1_id, comp2_id = parts
                comp1_id = int(comp1_id)
                comp2_id = int(comp2_id)

                seen_count = seen_cache.get(name, 0) + 1
                seen_cache[name] = seen_count

                comp1_range = chain(range(max(1, comp1_id - seen_count * n_samples), comp1_id - (seen_count-1) * n_samples), range(max(1, comp1_id + (seen_count - 1) * n_samples), comp1_id + seen_count * n_samples + 1))
                for new_comp1 in comp1_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{new_comp1}:{comp2_id}"
                    if avoid_names and new_name in avoid_names:
                        continue
                    names.append(new_name)

                comp2_range = chain(range(max(1, comp2_id - seen_count * n_samples), comp2_id - (seen_count-1) * n_samples), range(max(1, comp2_id + (seen_count - 1) * n_samples), comp2_id + seen_count * n_samples + 1))
                for new_comp2 in comp2_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{new_comp2}"
                    if avoid_names and new_name in avoid_names:
                        continue
                    names.append(new_name)

            if len(parts) == 5:
                rxn_prefix, rxn_type, comp1_id, comp2_id, comp3_id = parts
                comp1_id = int(comp1_id)
                comp2_id = int(comp2_id)
                comp3_id = int(comp3_id)

                seen_count = seen_cache.get(name, 0) + 1
                seen_cache[name] = seen_count
                
                comp1_range = chain(range(max(1, comp1_id - seen_count * n_samples), comp1_id - (seen_count-1) * n_samples), range(max(1, comp1_id + (seen_count - 1) * n_samples), comp1_id + seen_count * n_samples + 1))
                for new_comp1 in comp1_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{new_comp1}:{comp2_id}:{comp3_id}"
                    if avoid_names and new_name in avoid_names:
                        continue
                    names.append(new_name)

                comp2_range = chain(range(max(1, comp2_id - seen_count * n_samples), comp2_id - (seen_count-1) * n_samples), range(max(1, comp2_id + (seen_count - 1) * n_samples), comp2_id + seen_count * n_samples + 1))
                for new_comp2 in comp2_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{new_comp2}:{comp3_id}"
                    if avoid_names and new_name in avoid_names:
                        continue
                    names.append(new_name)

                comp3_range = chain(range(max(1, comp3_id - seen_count * n_samples), comp3_id - (seen_count-1) * n_samples), range(max(1, comp3_id + (seen_count - 1) * n_samples), comp3_id + seen_count * n_samples + 1))
                for new_comp3 in comp3_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{comp2_id}:{new_comp3}"
                    if avoid_names and new_name in avoid_names:
                        continue
                    names.append(new_name)

        except (ValueError, IndexError) as e:
            bt.logging.warning(f"Could not parse name '{name}': {e}")
            continue

    if not names: return pd.DataFrame(columns=["name", "smiles"])

    df = pd.DataFrame({"name": names})

    df = df[df["name"].notna()]
    if df.empty: return pd.DataFrame(columns=["name", "smiles"])

    df = manager.validate_molecules(config, df)
    if df.empty:
        return pd.DataFrame(columns=["name", "smiles"])

    return df[["name", "smiles"]].copy()

def cpu_random_candidates_with_similarity(
    manager: MoleculeManager,
    n_samples: int,
    config: dict,
    top_pool_df: pd.DataFrame,
    avoid_names: set[str] | None = None,
    thresh: float = 0.8
) -> pd.DataFrame:
    try:
        random_df = sample_random_valid_molecules(
            manager=manager,
            n_samples=n_samples,
            config=config,
            avoid_names=avoid_names,
            focus_neighborhood_of=top_pool_df
        )
        if random_df.empty or top_pool_df.empty:
            bt.logging.info("[CPU Executor] No random valid molecules are found.")
            return pd.DataFrame(columns=["name", "smiles"])

        sims = compute_tanimoto_similarity_to_pool(
            candidate_smiles=random_df["smiles"],
            pool_smiles=top_pool_df["smiles"],
        )
        random_df = random_df.copy()
        random_df["tanimoto_similarity"] = sims.reindex(random_df.index).fillna(0.0)
        random_df = random_df.sort_values(by="tanimoto_similarity", ascending=False)
        random_df_filtered = random_df[random_df["tanimoto_similarity"] >= thresh]
        if random_df_filtered.empty:
            bt.logging.info("[CPU Executor] No random filtered valid molecules are found.")
            return pd.DataFrame(columns=["name", "smiles", "tanimoto_similarity"])

        random_df_filtered = random_df_filtered.reset_index(drop=True)
        return random_df_filtered[["name", "smiles"]]
    except Exception as e:
        bt.logging.warning(f"[Solution] cpu_random_candidates_with_similarity failed: {e}")
        return pd.DataFrame(columns=["name", "smiles"])
    
def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
    
    if top_pool.empty:
        return weights
    
    max_score = top_pool['score'].max() if not top_pool.empty else 1.0
    
    for idx, row in top_pool.iterrows():
        name = row['name']
        score = row['score']

        rank = idx + 1
        rank_weight = 2.5 * math.exp(-rank / 18.0)
        weighted_score = max(0, score) * rank_weight
        
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += weighted_score
                weights['B'][B_id] += weighted_score
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += weighted_score
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue
    
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                avg_weight = weights[role][comp_id] / counts[role][comp_id]
                weights[role][comp_id] = avg_weight + 0.15
    
    return weights

def entropy_phase_fix(
    config: dict,
    all_pool: pd.DataFrame,
    rate: float
) -> (pd.DataFrame, float) :
    ap_scores = all_pool["score"].values.astype(np.float64)
    ap_fps = np.array(all_pool["maccs"].tolist(), dtype=np.float64)
    n_pool = len(all_pool)
    def _calc_ent(fp_sum, n=100):
        bf = fp_sum / n
        eps = 1e-10
        h = -bf * np.log2(bf + eps) - (1 - bf) * np.log2(1 - bf + eps)
        return float(h.mean())
    
    def _batch_ent(base_sum, add_fps, n=100):
        ns = base_sum[np.newaxis, :] + add_fps
        bf = ns / n
        eps = 1e-10
        h = -bf * np.log2(bf + eps) - (1 - bf) * np.log2(1 - bf + eps)
        return h.mean(axis=1)
    def _normalize(x):
        rng = x.max() - x.min()
        if rng > 1e-10:
            return (x - x.min()) / rng
        return np.zeros_like(x)

    sel = [0]
    in_s = np.zeros(n_pool, dtype=bool)
    in_s[0] = True
    fps_sum = ap_fps[0].copy()
    
    for k in range(1, 100):
        ci = np.where(~in_s)[0]
        if len(ci) > 5000:
            ci = ci[:5000]
        ents = _batch_ent(fps_sum, ap_fps[ci], n=k+1)
        scores = ap_scores[ci]
        combo = _normalize(scores) + rate * _normalize(ents)
        chosen = ci[np.argmax(combo)]
        sel.append(chosen)
        in_s[chosen] = True
        fps_sum = fps_sum + ap_fps[chosen]
    sel = np.array(sel, dtype=np.int64)
    seed_ent = _calc_ent(fps_sum)
    
    if seed_ent > config["entropy_min_threshold"]:
        candidates = np.where(~in_s)[0].copy()
        cand_scores = ap_scores[candidates].copy()
        score_sum_local = ap_scores[sel].sum()
        fp_sum_local = fps_sum.copy()
        
        for _ in range(200):
            wo = np.argsort(ap_scores[sel])
            found = False
            for vr in range(100):
                sp = wo[vr]
                v = sel[sp]
                vs = ap_scores[v]
                base = fp_sum_local - ap_fps[v]
                mb = cand_scores > vs + 1e-6
                if not mb.any():
                    continue
                vi = np.where(mb)[0]
                es = _batch_ent(base, ap_fps[candidates[vi]])
                me = es >= config["entropy_min_threshold"]
                if me.any():
                    ev = np.where(me)[0]
                    bl = ev[np.argmax(cand_scores[vi[ev]])]
                    ci_idx = vi[bl]
                    c = candidates[ci_idx]
                    in_s[v] = False
                    in_s[c] = True
                    sel[sp] = c
                    fp_sum_local = base + ap_fps[c]
                    score_sum_local = score_sum_local - vs + cand_scores[ci_idx]
                    candidates[ci_idx] = v
                    cand_scores[ci_idx] = vs
                    found = True
                    break
            if not found:
                break
    else:
        return all_pool.iloc[sel.copy()].copy().head(config["num_molecules"]), seed_ent
    
    final_ent = _calc_ent(fp_sum_local)
    final_pool = all_pool.iloc[sel.copy()].copy().head(config["num_molecules"])
    return final_pool, final_ent