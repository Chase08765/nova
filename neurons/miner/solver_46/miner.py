import os
import time
import json
import random
import pandas as pd
import numpy as np
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import nova_ph2
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit import DataStructs

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__)))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
DB_PATH    = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")
TIME_LIMIT        = 900
LIMIT_PER_REACTANT = 600

# Create global Morgan fingerprint generator to avoid deprecation warnings
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Cache for fingerprints to avoid recomputation
_fp_cache = {}
_mol_cache = {}

from molecules import (
    MoleculeManager,
    MoleculeUtils,
)
from tools import (
    IterationParams,
    SynthonLibrary,
    generate_valid_random_molecules,
    cpu_random_candidates_with_similarity,
    build_component_weights,
    entropy_phase_fix
)
from models import ModelManager
from exploit import (
    get_top_n_unexploited,
    run_exploit,
)
from dpex_dja import (
    DPEXDJAState,
    dja_generate,
    tabu_generate,
    update_tabu,
    dpex_exchange,
    update_populations,
    set_ranker_weights,
)


class ComponentRanker:
    """Learns per-reactant marginal quality via EMA. Zero-overhead: just updates weights."""
    def __init__(self, decay=0.90):
        self.decay = decay
        self.q_A, self.q_B, self.q_C = {}, {}, {}
        self.family_ema = {}
        self.family_decay = 0.85
        self.family_penalties = {}

    def update(self, scored_df):
        if scored_df.empty:
            return
        
        # Cross-family anti-dominance: track family performance
        family_batch_scores = {}
        for _, row in scored_df.iterrows():
            score = row.get('score', 0.0)
            if pd.isna(score):
                continue
            parts = row['name'].split(':')
            if len(parts) >= 2:
                try:
                    rxn_id = int(parts[1])
                    if rxn_id not in family_batch_scores:
                        family_batch_scores[rxn_id] = []
                    family_batch_scores[rxn_id].append(score)
                except (ValueError, IndexError):
                    continue
        
        # Update family EMAs and compute anti-dominance penalties
        for rxn_id, scores in family_batch_scores.items():
            batch_avg = sum(scores) / len(scores)
            if rxn_id in self.family_ema:
                old_ema = self.family_ema[rxn_id]
                self.family_ema[rxn_id] = self.family_decay * old_ema + (1 - self.family_decay) * batch_avg
            else:
                self.family_ema[rxn_id] = batch_avg
        
        # Calculate penalties: downweight dominant families, boost weak ones
        if self.family_ema:
            avg_family = sum(self.family_ema.values()) / len(self.family_ema)
            for rxn_id, ema_score in self.family_ema.items():
                ratio = ema_score / (avg_family + 1e-9)
                if ratio > 1.2:
                    self.family_penalties[rxn_id] = max(0.3, 1.0 - 0.5 * (ratio - 1.2))
                elif ratio > 1.0:
                    self.family_penalties[rxn_id] = max(0.6, 1.0 - 0.5 * (ratio - 1.0))
                else:
                    self.family_penalties[rxn_id] = min(1.4, 1.0 + 0.4 * (1.0 - ratio))
        
        # Update component EMAs with family penalties applied
        for _, row in scored_df.iterrows():
            score = row.get('score', 0.0)
            if pd.isna(score):
                continue
            parts = row['name'].split(':')
            if len(parts) < 4:
                continue
            try:
                rxn_id = int(parts[1])
                A, B = int(parts[2]), int(parts[3])
                C = int(parts[4]) if len(parts) > 4 else None
            except (ValueError, IndexError):
                continue
            # Apply anti-dominance penalty to component scores
            penalty = self.family_penalties.get(rxn_id, 1.0)
            adjusted_score = score * penalty
            self._ema(self.q_A, A, adjusted_score)
            self._ema(self.q_B, B, adjusted_score)
            if C is not None:
                self._ema(self.q_C, C, adjusted_score)

    def _ema(self, store, key, score):
        if key in store:
            old, cnt = store[key]
            store[key] = (self.decay * old + (1 - self.decay) * score, cnt + 1)
        else:
            store[key] = (score, 1)

    def compute_weights(self, pool, store):
        """Return normalized numpy weight array for a component pool."""
        if not store:
            return None
        w = np.array([max(0.01, store[mid][0]) if mid in store else 0.05 for mid in pool])
        w /= w.sum()
        return w

    def push_to_dja(self, manager):
        """Push ranker weights into dpex_dja module globals — zero per-call overhead."""
        w_A = self.compute_weights(manager.moles_A_id, self.q_A)
        w_B = self.compute_weights(manager.moles_B_id, self.q_B)
        w_C = self.compute_weights(manager.moles_C_id, self.q_C) if manager.is_three_component else None
        set_ranker_weights(w_A, w_B, w_C)

    def blend_component_weights(self, component_weights, manager):
        """Blend ranker EMA into winner's component_weights for traditional GA."""
        if component_weights is None:
            return None
        blended = dict(component_weights)
        rxn_id = manager.rxn_id
        for role, pool, q in [('A', manager.moles_A_id, self.q_A), ('B', manager.moles_B_id, self.q_B)]:
            key = f"{rxn_id}_{role}"
            if key not in blended or not q:
                continue
            orig = blended[key]
            new_w = {}
            for mid in pool:
                o = orig.get(mid, 0.05)
                e = max(0.01, q[mid][0]) if mid in q else 0.05
                new_w[mid] = 0.6 * o + 0.4 * e
            total = sum(new_w.values())
            if total > 0:
                new_w = {k: v/total for k, v in new_w.items()}
            blended[key] = new_w
        return blended


def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}

model_manager    = None
molecule_manager = None

def initialize_solution(config: dict):
    global molecule_manager, model_manager
    molecule_manager = MoleculeManager(config=config, db_path=DB_PATH)
    model_manager    = ModelManager(config)

def get_mol(smiles: str):
    """Get RDKit Mol object from SMILES, cached."""
    if smiles in _mol_cache:
        return _mol_cache[smiles]
    
    mol = Chem.MolFromSmiles(smiles)
    _mol_cache[smiles] = mol  # store None if invalid
    return mol

def get_morgan_fingerprint(smiles: str, n_bits: int = 2048):
    """Get Morgan fingerprint for a SMILES string using MorganGenerator (cached), reusing Mol objects."""
    if smiles in _fp_cache:
        return _fp_cache[smiles]

    mol = get_mol(smiles)  # <- use cached Mol
    if mol is None:
        return None

    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
    fp_array = np.zeros(n_bits, dtype=np.uint8)
    fp_array[fp.GetOnBits()] = 1

    _fp_cache[smiles] = fp_array

    # optional: maintain cache size limit
    if len(_fp_cache) > 50000:
        keys_to_remove = list(_fp_cache.keys())[:12500]
        for key in keys_to_remove:
            del _fp_cache[key]

    return fp_array

class SurrogateModel:
    """Fast surrogate model for score prediction using Random Forest.
    Used to pre-filter candidates before expensive GPU scoring for higher throughput."""
    
    def __init__(self, max_training_samples: int = 4000):
        # Improved RF: more trees + depth for better accuracy, n_jobs=-1 for speed
        self.model = RandomForestRegressor(
            n_estimators=80, max_depth=14, min_samples_leaf=3, random_state=42,
            n_jobs=-1, max_samples=0.8
        )
        self.is_trained = False
        self.X_train = []
        self.y_train = []
        self.min_train_size = 80  # Lower to enable earlier surrogate use
        self.max_training_samples = max_training_samples
        self.last_train_iteration = 0
        self.train_interval = 2  # Train more frequently when improving
        self.enabled = True
    
    def add_training_data(self, smiles_list: list, scores: list):
        """Add training data: favor top-scorers but include some low-scorers for discrimination."""
        if not self.enabled:
            return
        
        if len(smiles_list) > 600:
            scores_array = np.array(scores)
            # Keep top 500 + sample 100 from bottom/mid for better score discrimination
            top_indices = np.argsort(scores_array)[-500:]
            mid_low = np.argsort(scores_array)[:min(200, len(scores_array)//2)]
            sample_low = list(np.random.choice(mid_low, min(100, len(mid_low)), replace=False)) if len(mid_low) > 0 else []
            keep_indices = sorted(set(list(top_indices) + sample_low))
            smiles_list = [smiles_list[i] for i in keep_indices]
            scores = [scores[i] for i in keep_indices]
        
        new_fps = []
        new_scores = []
        for smiles, score in zip(smiles_list, scores):
            fp = get_morgan_fingerprint(smiles)
            if fp is not None:
                new_fps.append(fp)
                new_scores.append(score)
        
        self.X_train.extend(new_fps)
        self.y_train.extend(new_scores)
        
        if len(self.X_train) > self.max_training_samples:
            scores_array = np.array(self.y_train)
            top_count = int(self.max_training_samples * 0.5)  # More top-scorers
            recent_count = int(self.max_training_samples * 0.5)
            top_indices = np.argsort(scores_array)[-top_count:]
            recent_indices = list(range(len(self.X_train) - recent_count, len(self.X_train)))
            keep_indices = sorted(set(list(top_indices) + recent_indices))
            self.X_train = [self.X_train[i] for i in keep_indices]
            self.y_train = [self.y_train[i] for i in keep_indices]
    
    def train(self, iteration: int = 0):
        """Train the model periodically."""
        if self.is_trained and (iteration - self.last_train_iteration) < self.train_interval:
            return
        
        if len(self.X_train) < self.min_train_size:
            self.is_trained = False
            return
        
        try:
            X = np.array(self.X_train)
            y = np.array(self.y_train)
            train_start = time.time()
            self.model.fit(X, y)
            train_time = time.time() - train_start
            self.is_trained = True
            self.last_train_iteration = iteration
            if train_time > 0.5:
                bt.logging.info(f"[SURROGATE] Trained in {train_time:.2f}s on {len(self.X_train)} samples")
        except Exception as e:
            bt.logging.warning(f"Surrogate model training failed: {e}")
            self.is_trained = False
    
    def predict(self, smiles_list: list) -> np.ndarray:
        """Predict scores for a list of SMILES."""
        if not self.is_trained:
            return np.array([0.0] * len(smiles_list))
        
        try:
            fps = []
            for smiles in smiles_list:
                fp = get_morgan_fingerprint(smiles)
                if fp is None:
                    fps.append(np.zeros(2048, dtype=np.uint8))
                else:
                    fps.append(fp)
            
            X = np.array(fps)
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            bt.logging.warning(f"Surrogate prediction failed: {e}")
            return np.array([0.0] * len(smiles_list))
    
    def predict_with_std(self, smiles_list: list):
        """Predict scores and return std across trees (uncertainty estimate)."""
        if not self.is_trained:
            return np.array([0.0] * len(smiles_list)), np.array([1e6] * len(smiles_list))
        
        try:
            fps = []
            for smiles in smiles_list:
                fp = get_morgan_fingerprint(smiles)
                if fp is None:
                    fps.append(np.zeros(2048, dtype=np.uint8))
                else:
                    fps.append(fp)
            
            X = np.array(fps)
            # Get predictions from all trees for uncertainty estimation
            predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            means = predictions.mean(axis=0)
            stds = predictions.std(axis=0)
            return means, stds
        except Exception as e:
            bt.logging.warning(f"Surrogate predict_with_std failed: {e}")
            return np.array([0.0] * len(smiles_list)), np.array([1e6] * len(smiles_list))

    def filter_candidates(self, data: pd.DataFrame, n_keep: int, smiles_col: str = "smiles") -> pd.DataFrame:
        """
        Pre-filter candidates before GPU scoring using confidence variance filtering.
        Filters out high-uncertainty predictions to reduce wasted GPU time.
        Includes safety fallback if too few high-confidence candidates exist.
        """
        if not self.is_trained or data.empty or len(data) <= n_keep:
            return data
        
        smiles_list = data[smiles_col].tolist()
        pred_scores, pred_stds = self.predict_with_std(smiles_list)
        data = data.copy()
        data["_surrogate_pred"] = pred_scores
        data["_surrogate_std"] = pred_stds
        
        # Confidence variance filtering: keep candidates with uncertainty below threshold
        # Use 60th percentile of std (keep bottom 40% most confident)
        std_threshold = np.percentile(pred_stds, 60)
        high_conf_mask = pred_stds <= std_threshold
        high_conf_count = high_conf_mask.sum()
        
        # Safety fallback: if high-confidence candidates < n_keep//2, use all candidates
        if high_conf_count >= n_keep // 2:
            filtered = data[high_conf_mask].sort_values("_surrogate_pred", ascending=False).head(n_keep)
            bt.logging.info(f"[SURROGATE] Confidence filtered: {len(data)} -> {high_conf_count} high-conf -> {len(filtered)} candidates")
        else:
            # Fallback to standard top-n selection
            filtered = data.sort_values("_surrogate_pred", ascending=False).head(n_keep)
            bt.logging.info(f"[SURROGATE] Fallback (low conf): {len(data)} -> {len(filtered)} candidates")
        
        filtered = filtered.drop(columns=["_surrogate_pred", "_surrogate_std"])
        # Hard cap safety valve: never return more than 250 candidates to GPU
        if len(filtered) > 250:
            filtered = filtered.head(250)
        return filtered.reset_index(drop=True)

_morgan_bv_cache = {}

def get_morgan_fp_bv(smiles: str):
    fp = _morgan_bv_cache.get(smiles)
    if fp is not None:
        return fp
    mol = get_mol(smiles)
    if mol is None:
        return None
    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
    _morgan_bv_cache[smiles] = fp
    if len(_morgan_bv_cache) > 50000:
        for k in list(_morgan_bv_cache.keys())[:12500]:
            del _morgan_bv_cache[k]
    return fp

def select_tanimoto_diverse(df: pd.DataFrame,
                            n: int,
                            threshold: float = 0.9,
                            smiles_col: str = "smiles") -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.head(0)

    kept_indices = []
    kept_fps = []
    for idx, row in df.iterrows():
        smi = row.get(smiles_col)
        if not isinstance(smi, str) or not smi:
            continue
        fp = get_morgan_fp_bv(smi)
        if fp is None:
            continue
        if kept_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, kept_fps)
            if max(sims) >= threshold:
                continue
        kept_indices.append(idx)
        kept_fps.append(fp)

        if len(kept_indices) >= n:
            break

    return df.loc[kept_indices]

def find_solution(config: dict, time_start: float):
    global molecule_manager, model_manager

    iteration = 0
    rxn_id = 2
    n_workers = os.cpu_count() or 1
    bt.logging.info(f"[Solution] CPU Workers: {n_workers}")

    # Initialize surrogate model
    surrogate = SurrogateModel(max_training_samples=4000)
    use_surrogate = True
    exploit_counter = 0
    is_entropy_fixed = False
    last_saved_average = -1e6

    # ComponentRanker — learns per-reactant quality, pushes weights into DJA
    ranker = ComponentRanker(decay=0.90)
    plateau_counter = 0  # anti-plateau: track consecutive zero-improvement iters

    params   = IterationParams(config=config)
    dpex     = DPEXDJAState()
    seed_df  = pd.DataFrame(columns=["name", "smiles"])
    top_pool = pd.DataFrame(columns=["name", "smiles", "inchi", "score", "target", "anti"])
    all_pool = pd.DataFrame(columns=["name", "smiles", "inchi", "score"])
    
    tabued_molecules = set()

    tanimoto_max_threshold = config.get("tanimoto_max_threshold", 0.9)

    with ProcessPoolExecutor(max_workers=n_workers) as cpu_executor:
        while time.time() - time_start < TIME_LIMIT:
            iteration          += 1
            sur_filter = False
            dpex.iteration      = iteration
            iteration_start     = time.time()
            remaining_time      = TIME_LIMIT - (iteration_start - time_start)

            bt.logging.info(f"[Solution] --- Iteration {iteration} [DPEX-DJA] ---")
            n_base_samples = params.get_nsamples_from_time(remaining_time)
            if iteration <= 10:
                n_samples = n_base_samples
            elif iteration % 4 == 0:
                n_samples = n_base_samples
            else:
                n_samples = n_base_samples * 7
                sur_filter = True
            
            component_weights = (
                build_component_weights(top_pool.head(config["num_molecules"]), molecule_manager.rxn_id)
                if not top_pool.empty else None
            )
            # Blend ranker intelligence into component weights (zero extra cost)
            if component_weights is not None and iteration > 2:
                component_weights = ranker.blend_component_weights(component_weights, molecule_manager)
            # Push ranker weights into DJA module for smart random picks
            if iteration > 2:
                ranker.push_to_dja(molecule_manager)

            elite_df    = (
                MoleculeUtils.select_diverse_elites(top_pool, min(150, len(top_pool)))
                if not top_pool.empty else pd.DataFrame()
            )
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None

            if params.no_improvement_counter >= 4 and params.use_exploit_mode is False:
                params.use_exploit_mode = True
                params.no_improvement_counter = 0
                bt.logging.info(f"[Solution] === EXPLOIT MODE  (no_improvement={params.no_improvement_counter}) ===")
            elif params.no_improvement_counter >= 4 :
                params.use_exploit_mode = False
                exploit_counter = 0
                params.no_improvement_counter = 0

            if iteration >= 2 and params.synthon_lib is None:
                try:
                    bt.logging.info("[Solution] Building synthon library ...")
                    t0 = time.time()
                    params.synthon_lib     = SynthonLibrary(molecule_manager=molecule_manager)
                    params.use_synthon_search = True
                    bt.logging.info(f"[Solution] Synthon library ready in {time.time()-t0:.2f}s")
                except Exception as e:
                    bt.logging.warning(f"[Solution] Synthon library failed: {e}")
                    params.synthon_lib        = None
                    params.use_synthon_search = False

            if not top_pool.empty:
                cols       = [c for c in ('name', 'smiles', 'score', 'target', 'anti') if c in top_pool.columns]
                top_records = top_pool[cols].head(dpex.N_B).to_dict('records')
                existing   = {m['name']: m for m in dpex.pop_B}
                for mol in top_records:
                    if mol['name'] not in existing:
                        existing[mol['name']] = mol
                dpex.pop_B = sorted(
                    existing.values(),
                    key=lambda x: x.get('score', float('-inf')),
                    reverse=True,
                )[:dpex.N_B]

            data               = pd.DataFrame(columns=["name", "smiles"])
            data_dja           = pd.DataFrame(columns=["name"])
            data_tabu          = pd.DataFrame(columns=["name"])
            data_early_exploit = pd.DataFrame(columns=["name", "smiles"])
            data_tabu_moves: list = []
            exploited_status = False
            exploit_summary  = None

            if not top_pool.empty and iteration > 3 and iteration <= 15:
                try:
                    all_top_mols_ee = top_pool.to_dict("records")
                    unexploited_ee = get_top_n_unexploited(all_top_mols_ee, params.exploited_reactants, n=2)
                    if unexploited_ee:
                        t0_ee = time.time()
                        early_results, _early_summary = run_exploit(
                            manager=molecule_manager,
                            config=config,
                            top_molecules=unexploited_ee,
                            top_n=1,
                            limit_per_reactant=150,
                            avoid_names=params.seen_molecules,
                            exploited_reactants=set(),
                        )
                        if early_results:
                            data_early_exploit = pd.DataFrame(early_results)
                            bt.logging.info(
                                f"[Solution] Early exploit: {len(data_early_exploit)} candidates "
                                f"in {time.time()-t0_ee:.1f}s"
                            )
                except Exception as e:
                    bt.logging.debug(f"[Solution] Early exploit skipped: {e}")

            if params.use_exploit_mode:
                bt.logging.info("[Solution] Exploit: structure-guided deep search ...")
                all_top_mols = top_pool.to_dict("records")
                try:
                    unexploited = get_top_n_unexploited(all_top_mols, params.exploited_reactants)
                    if unexploited:
                        t0 = time.time()
                        exploit_results, exploit_summary = run_exploit(
                            manager=molecule_manager,
                            config=config,
                            top_molecules=unexploited,
                            limit_per_reactant=LIMIT_PER_REACTANT,
                            avoid_names=params.seen_molecules,
                            exploited_reactants=params.exploited_reactants,
                        )
                        bt.logging.info(
                            f"[Solution] Exploit: {len(exploit_results)} candidates "
                            f"in {time.time()-t0:.1f}s"
                        )
                        if exploit_results:
                            data             = pd.DataFrame(exploit_results)
                            exploited_status = True
                        else:
                            raise Exception("Exploit returned no molecules.")
                    else:
                        raise Exception("No unexploited top molecules available.")
                except Exception as e:
                    bt.logging.warning(f"[Solution] Exploit skipped: {e}")
                exploit_counter += 1

            if not exploited_status:
                if iteration == 1 or not dpex.pop_A:
                    bt.logging.info(
                        f"[Solution] Init: generating {params.n_samples_start} random molecules"
                    )
                    data = generate_valid_random_molecules(
                        config=config,
                        manager=molecule_manager,
                        n_samples=params.n_samples_start,
                        mutation_prob=0,
                        elite_prob=0,
                        executor=cpu_executor,
                        n_workers=n_workers,
                        avoid_names=params.seen_molecules,
                        elite_names=None,
                        component_weights=component_weights,
                    )

                else:
                    n_dja  = int(n_samples * 0.75)
                    n_tabu = n_samples - n_dja

                    bt.logging.info(
                        f"[Solution] DJA: generating {n_dja} candidates "
                        f"(pop_A size = {len(dpex.pop_A)})"
                    )
                    raw_dja = dja_generate(
                        state=dpex,
                        manager=molecule_manager,
                        n_samples=n_dja,
                        avoid=params.seen_molecules,
        surrogate=surrogate,
                    )
                    if not raw_dja.empty:
                        data_dja = molecule_manager.validate_molecules(
                            config, raw_dja,
                            time_elapsed=iteration_start - time_start,
                        )
                        bt.logging.info(f"[Solution] DJA: {len(data_dja)} validated")

                    if params.synthon_lib is not None and dpex.pop_B:
                        global_best = (
                            top_pool['score'].max()
                            if not top_pool.empty else float('-inf')
                        )
                        bt.logging.info(
                            f"[Solution] Tabu: generating candidates from "
                            f"pop_B ({len(dpex.pop_B)} elites), n_tabu≈{n_tabu}"
                        )

                        if params.score_improvement_rate > 0.05:
                            n_per_elite = 15
                            n_elites = 30
                            bt.logging.info(f"[Solution] High improvement ({params.score_improvement_rate:.4f})")

                        elif params.score_improvement_rate > 0.02:
                            n_per_elite = 20
                            n_elites = 40
                            bt.logging.info(f"[Solution] Good improvement ({params.score_improvement_rate:.4f})")

                        elif params.score_improvement_rate > 0.005:
                            n_per_elite = 25
                            n_elites = 60
                            bt.logging.info(f"[Solution] Moderate improvement ({params.score_improvement_rate:.4f})")
                        
                        else:
                            n_per_elite = 50
                            n_elites = 100
                            bt.logging.info(f"[Solution] Low improvement ({params.score_improvement_rate:.4f})")


                        raw_tabu, data_tabu_moves = tabu_generate(
                            state=dpex,
                            synthon_lib=params.synthon_lib,
                            manager=molecule_manager,
                            avoid=params.seen_molecules,
                            k_per_elite=n_per_elite,
                            k_elites=n_elites,
                            global_best_score=global_best,
                            tabued_molecules=tabued_molecules,
                        )

                        if params.score_improvement_rate <= 0.005:
                            tabued_molecules = tabued_molecules | set([x['name'] for x in dpex.pop_B])

                        if not raw_tabu.empty:
                            data_tabu = molecule_manager.validate_molecules(
                                config, raw_tabu,
                                time_elapsed=iteration_start - time_start,
                            )
                            if not data_dja.empty:
                                data_tabu = data_tabu[
                                    ~data_tabu["name"].isin(data_dja["name"].tolist())
                                ]
                            bt.logging.info(f"[Solution] Tabu: {len(data_tabu)} validated")

                    parts = [df for df in [data_dja, data_tabu, data_early_exploit] if not df.empty]

                    if parts:
                        data = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["name"])
                        if not seed_df.empty:
                            data    = pd.concat([data, seed_df], ignore_index=True).drop_duplicates(subset=["name"])
                            seed_df = pd.DataFrame(columns=["name", "smiles"])

                    traditional_df = generate_valid_random_molecules(
                        config=config,
                        manager=molecule_manager,
                        n_samples= int(n_samples * 0.35),
                        mutation_prob=params.mutation_prob,
                        elite_prob=params.elite_prob,
                        executor=cpu_executor,
                        n_workers=n_workers,
                        avoid_names=params.seen_molecules,
                        elite_names=elite_names,
                        component_weights=component_weights,
                    )

                    data = pd.concat([data, traditional_df], ignore_index = True).drop_duplicates(subset = ["name"])
                    

            gen_time = time.time() - iteration_start
            bt.logging.info(
                f"[Solution] Iteration {iteration}: {len(data)} candidates "
                f"generated in {gen_time:.2f}s (pre-score)"
            )
            if data.empty:
                bt.logging.warning(f"[Solution] No valid molecules this iteration; skipping")
                continue
            if not seed_df.empty:
                data    = pd.concat([data, seed_df], ignore_index=True).drop_duplicates(subset=["name"])
                seed_df = pd.DataFrame(columns=["name", "smiles"])
                
            # Universal Surrogate Pre-Filter Gate: apply consistently to all candidate sources
            if use_surrogate and surrogate.is_trained and len(data) > n_base_samples:
                n_keep = max(int(len(data) * 0.35), n_base_samples // 2)
                data = surrogate.filter_candidates(data, n_keep=n_keep, smiles_col="smiles")
            
            # CPU-based pre-filter for initialization (iteration 1) to save GPU budget
            if iteration == 1 and len(data) > n_base_samples:
                bt.logging.info(f"[Solution] CPU pre-filter: reducing {len(data)} candidates for GPU scoring")
                # Compute drug-likeness using cheap descriptors (CPU only)
                smiles_list = data['smiles'].tolist()
                cpu_scores = []
                for smi in smiles_list:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        cpu_scores.append(-1000.0)
                        continue
                    try:
                        mw = Descriptors.MolWt(mol)
                        logp = Descriptors.MolLogP(mol)
                        hbd = Descriptors.NumHDonors(mol)
                        hba = Descriptors.NumHAcceptors(mol)
                        # Lipinski-like score: reward drug-like properties
                        score = 0.0
                        if 150 <= mw <= 500: score += 2.0
                        if -0.5 <= logp <= 5: score += 2.0
                        if hbd <= 5: score += 1.0
                        if hba <= 10: score += 1.0
                        # Penalize extremes to prefer middle range
                        score -= abs(mw - 350) / 200
                        score -= abs(logp - 2.5) / 5
                        cpu_scores.append(score)
                    except:
                        cpu_scores.append(-1000.0)
                
                data = data.copy()
                data['_cpu_score'] = cpu_scores
                data = data.sort_values('_cpu_score', ascending=False).head(int(n_base_samples * 1.2))
                data = data.drop(columns=['_cpu_score'])
                bt.logging.info(f"[Solution] CPU pre-filter done: {len(data)} candidates remaining")
            
            try:
                filtered  = data[~data["name"].isin(params.seen_molecules)]
                dup_ratio = (len(data) - len(filtered)) / max(1, len(data))

                if dup_ratio > 0.7:
                    params.mutation_prob = min(0.90, params.mutation_prob * 1.5)
                elif dup_ratio > 0.5:
                    params.mutation_prob = min(0.70, params.mutation_prob * 1.3)
                elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                    params.mutation_prob = max(0.10, params.mutation_prob * 0.95)

                data = filtered
            except Exception as e:
                bt.logging.warning(f"[Solution] Deduplication error: {e}")

            if data.empty:
                bt.logging.error(
                    f"[Solution] All molecules were duplicates; boosting diversity"
                )
                params.mutation_prob = min(0.95, params.mutation_prob * 2.0)
                params.elite_prob    = max(0.10, params.elite_prob * 0.5)
                continue

            data = data.reset_index(drop=True)

            cpu_futures = []
            if not top_pool.empty and iteration > 10 and iteration % 3 != 2 and params.use_exploit_mode is False:
                cpu_start = time.time()
                cpu_futures.append((
                    cpu_executor.submit(
                        cpu_random_candidates_with_similarity,
                        molecule_manager, 30, config,
                        top_pool.head(100)[["name", "smiles"]],
                        params.seen_molecules, 0.65,
                    ), "top100"
                ))
                cpu_end = time.time()
            elif not top_pool.empty and iteration > 3 and params.score_improvement_rate <= 0.01:
                cpu_start = time.time()
                cpu_futures.append((
                    cpu_executor.submit(
                        cpu_random_candidates_with_similarity,
                        molecule_manager, 40, config,
                        top_pool.head(5)[["name", "smiles"]],
                        params.seen_molecules, 0.80,
                    ), "tight-top5"
                ))
                cpu_futures.append((
                    cpu_executor.submit(
                        cpu_random_candidates_with_similarity,
                        molecule_manager, 30, config,
                        top_pool.head(20)[["name", "smiles"]],
                        params.seen_molecules, 0.65,
                    ), "medium-top20"
                ))
                cpu_end = time.time()              
                
            bt.logging.info(f"[Solution] Scoring {len(data)} molecules on GPU ...")
            gpu_start       = time.time()
            data["target"]  = model_manager.get_target_score_from_data(data["smiles"])
            data["anti"]    = model_manager.get_antitarget_score()
            data["score"]   = data["target"] - (config["antitarget_weight"] * data["anti"])
            bt.logging.info(f"[Solution] GPU scoring time: {time.time()-gpu_start:.2f}s")

            # Update ComponentRanker (learns per-reactant quality from ALL scored data)
            valid_scores = data[~data["score"].isna()]
            if not valid_scores.empty:
                ranker.update(valid_scores)

            # Update surrogate model
            if len(valid_scores) > 0 and surrogate.enabled:
                surrogate.add_training_data(
                    valid_scores["smiles"].tolist(),
                    valid_scores["score"].tolist()
                )
                if len(surrogate.X_train) >= surrogate.min_train_size:
                    train_start = time.time()
                    surrogate.train(iteration)
                    train_time = time.time() - train_start
                    if surrogate.is_trained and (iteration - surrogate.last_train_iteration) == 0:
                        bt.logging.info(f"Surrogate trained: {len(surrogate.X_train)} samples in {train_time:.2f}s")
                    elif train_time > 10.0:
                        bt.logging.warning(f"Surrogate training slow ({train_time:.2f}s) - disabling")
                        surrogate.enabled = False
                        use_surrogate = False

            if cpu_futures:
                for fut, strategy_name in cpu_futures:
                    try:
                        cpu_df = fut.result(timeout=0)
                        if not cpu_df.empty:
                            seed_df = (
                                pd.concat([seed_df, cpu_df], ignore_index=True)
                                if not seed_df.empty else cpu_df.copy()
                            )
                            bt.logging.info(
                                f"[Solution] CPU ({strategy_name}): {len(cpu_df)} candidates"
                            )
                    except TimeoutError:
                        pass
                    except Exception as e:
                        bt.logging.warning(f"[Solution] CPU ({strategy_name}) failed: {e}")
                if not seed_df.empty:
                    seed_df = seed_df.drop_duplicates(subset=["name"])

# Bootstrap Surrogate Early: Generate remaining batch for iter 2 using trained model
            if iteration == 1 and surrogate.is_trained:
                bt.logging.info("[Solution] Bootstrap: Generating remaining candidates for iter 2...")
                remaining = generate_valid_random_molecules(
                    config=config,
                    manager=molecule_manager,
                    n_samples=1600,
                    mutation_prob=0,
                    elite_prob=0,
                    executor=cpu_executor,
                    n_workers=n_workers,
                    avoid_names=params.seen_molecules,
                    elite_names=None,
                    component_weights=None,
                )
                if not remaining.empty:
                    n_keep = max(int(len(remaining) * 0.6), 200)
                    remaining_filtered = surrogate.filter_candidates(remaining, n_keep=n_keep, smiles_col="smiles")
                    bt.logging.info(f"[Solution] Bootstrap: Filtered {len(remaining)} -> {len(remaining_filtered)} candidates")
                    if not seed_df.empty:
                        seed_df = pd.concat([seed_df, remaining_filtered], ignore_index=True).drop_duplicates(subset=["name"])
                    else:
                        seed_df = remaining_filtered.copy()
            
                        dja_names  = set(data_dja["name"].tolist())  if not data_dja.empty  else set()
            tabu_names = set(data_tabu["name"].tolist()) if not data_tabu.empty else set()

            scored_for_A = data[data["name"].isin(dja_names)]  if dja_names  else data
            scored_for_B = data[data["name"].isin(tabu_names)] if tabu_names else pd.DataFrame(columns=data.columns)

            update_populations(dpex, scored_for_A, scored_for_B)

            if data_tabu_moves:
                update_tabu(dpex, data_tabu_moves)

            if iteration % dpex.T_ex == 0:
                dpex_exchange(dpex)

            bt.logging.debug(
                f"[DPEX] pop_A={len(dpex.pop_A)}  pop_B={len(dpex.pop_B)}"
            )

            data["inchi"]      = data["smiles"].map(MoleculeUtils.generate_inchikey)
            params.seen_molecules = params.seen_molecules | set(data["name"].tolist())

            prev_avg  = top_pool.head(config["num_molecules"])['score'].mean() if not top_pool.empty else None
            data["maccs"] = data["smiles"].map(MoleculeUtils.maccs_fp_from_smiles_cached)

            total_data = data[["name", "smiles", "inchi", "score", "target", "anti", "maccs"]]

            if not total_data.empty:
                if not all_pool.empty:
                    all_pool = pd.concat([all_pool, total_data], ignore_index=True)
                else:
                    all_pool = pd.concat([total_data], ignore_index=True)
                all_pool = all_pool.sort_values(by="score", ascending=False)
                all_pool = all_pool.drop_duplicates(subset=["inchi"], keep="first")                    
            else:
                bt.logging.warning(f"[Solution] Iteration {iteration}: No valid scored data")
            top_pool = select_tanimoto_diverse(
                all_pool.reset_index(drop=True),
                n=config["num_molecules"] + 50,
                threshold=tanimoto_max_threshold,
                smiles_col="smiles",
            ).reset_index(drop=True)
            
            current_avg = (
                top_pool.head(config["num_molecules"])['score'].mean()
                if not top_pool.empty else None
            )
            if current_avg is not None and prev_avg is not None:
                params.score_improvement_rate = (
                    (current_avg - prev_avg) / max(abs(prev_avg), 1e-6)
                )
            elif current_avg is not None:
                params.score_improvement_rate = 1.0

            if params.score_improvement_rate <= 0.0001:
                params.no_improvement_counter += 1
                plateau_counter += 1
            else:
                params.no_improvement_counter = 0
                plateau_counter = 0

            final_top_pool = top_pool
            remaining_time = 900 - (time.time() - time_start)
            if remaining_time <= 100:
                entropy = MoleculeUtils.compute_maccs_entropy(top_pool.iloc[:config["num_molecules"]]['smiles'].to_list())
                final_entropy = entropy
                
                if entropy <= config['entropy_min_threshold'] and (is_entropy_fixed is False or (params.score_improvement_rate > 0)):
                    filtered_all_pool = select_tanimoto_diverse(
                        all_pool,
                        n=5000,
                        threshold=tanimoto_max_threshold,
                        smiles_col="smiles",
                    ).reset_index(drop=True)
                    is_entropy_fixed = True
                    start_param = 3
                    while start_param < 40:
                        new_top_pool, new_entropy = entropy_phase_fix(config, filtered_all_pool, start_param)
                        bt.logging.info(f"[Entropy] New entropy: {new_entropy}, New average score: {new_top_pool['score'].mean()}, Param: {start_param}")
                        if final_entropy <= config['entropy_min_threshold'] or (new_entropy > config['entropy_min_threshold'] and new_top_pool['score'].mean() > final_avg_score):
                            final_top_pool = new_top_pool
                            final_entropy = new_entropy
                            final_avg_score = new_top_pool['score'].mean()
                        if new_entropy > config['entropy_min_threshold']:
                            break
                        start_param += max(int((config['entropy_min_threshold'] - new_entropy) / 0.01), 2)

                        
            if plateau_counter >= 5:
                params.mutation_prob = min(0.85, params.mutation_prob * 2.0)
                bt.logging.info(f"[Solution] ANTI-PLATEAU: boosting mutation to {params.mutation_prob:.2f}")
                plateau_counter = 0

            if (
                exploit_summary
                and 'exploited_reactant_ids' in exploit_summary
                and (params.score_improvement_rate <= 0.0001 or exploited_status is False)
            ):
                bt.logging.info(f"[Solution] Droping the picked reactant.... Exploited reactants: {len(params.exploited_reactants)}")
                params.exploited_reactants.update(exploit_summary['exploited_reactant_ids'])

            iter_time  = time.time() - iteration_start
            total_time = time.time() - time_start
            pool_avg   = final_top_pool.head(config["num_molecules"])['score'].mean()
            pool_max   = final_top_pool['score'].max()
            try:
                pool_entropy = MoleculeUtils.compute_maccs_entropy(
                    final_top_pool.head(config["num_molecules"])['smiles'].tolist()
                )
            except Exception:
                pool_entropy = 0.0

            if exploited_status:
                mode_str = "EXPLOIT"
            elif iteration == 1 or not dpex.pop_A:
                mode_str = "INIT"
            elif params.synthon_lib is not None:
                mode_str = "DJA+TABU"
            else:
                mode_str = "DJA"

            bt.logging.info(
                f"Iteration {iteration} | {iter_time:.1f}s | Total: {total_time:.0f}s | "
                f"Mode: {mode_str} | "
                f"popA={len(dpex.pop_A)} popB={len(dpex.pop_B)} | "
                f"Pool: avg={pool_avg:.4f} max={pool_max:.4f} ent={pool_entropy:.3f}"
            )
            print(
                f"Iteration {iteration} | {iter_time:.1f}s | Total: {total_time:.0f}s | "
                f"Mode: {mode_str} | "
                f"popA={len(dpex.pop_A)} popB={len(dpex.pop_B)} | "
                f"Pool: avg={pool_avg:.4f} max={pool_max:.4f} ent={pool_entropy:.3f}"
            )

            top_entries = {"molecules": final_top_pool.head(config["num_molecules"])["name"].tolist()}
            if pool_entropy > config['entropy_min_threshold'] and last_saved_average < pool_avg:
                with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                    json.dump(top_entries, f, ensure_ascii=False, indent=2)
                bt.logging.info("[Solution] Top entries saved.")
                last_saved_average = pool_avg

if __name__ == "__main__":
    time_start = time.time()
    config     = get_config()

    initialize_solution(config)
    bt.logging.info(f"[Solution] Init time: {time.time()-time_start:.2f}s")

    find_solution(config, time_start)
