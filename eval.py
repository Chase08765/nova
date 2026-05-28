"""
Standalone molecule-evaluation pipeline.

Loads a local input file (same format used by the validator's --local_input_file),
runs the same validation + entropy pass as the validator, and optionally runs
Boltz scoring to produce a ``BoltzResult``. Useful for offline debugging without
a subtensor / chain connection.

Example:
    python eval.py --local_input_file example_local_input --run_boltz
"""

import argparse
import os
import sys

import bittensor as bt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from neurons.validator.molecule_validity import validate_molecules_and_calculate_entropy
from utils.inference import BoltzResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate molecule submissions offline.")
    parser.add_argument(
        "--local_input_file",
        type=str,
        default="example_local_input",
        help="Path to the local input file (uid|mol_names|sequences per line).",
    )
    parser.add_argument(
        "--run_boltz",
        action="store_true",
        help="If set, run Boltz scoring and emit a BoltzResult.",
    )
    parser.add_argument(
        "--allowed_reaction",
        type=str,
        default=None,
        help="Optional reaction filter (mirrors the per-epoch challenge param).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(BASE_DIR, "config", "config.yaml"),
        help="Path to config YAML.",
    )
    return parser.parse_args()


def load_local_input(file_path: str) -> dict:
    """
    Offline-friendly version of utils.files.read_local_input_file: parses the
    same ``uid|mol_names|sequences`` format without requiring a subtensor.
    """
    if not os.path.exists(file_path):
        bt.logging.error(f"❌ Local input file not found: {file_path}")
        return {}

    uid_to_data: dict[int, dict] = {}
    with open(file_path, "r") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                uid_s, mol_names, protein_sequences = line.split("|")
                uid = int(uid_s)
                uid_to_data[uid] = {
                    "molecules": mol_names.split(","),
                    "sequences": protein_sequences.split(","),
                    "block_submitted": 0,
                    "push_time": "",
                }
            except Exception as e:
                bt.logging.warning(f"⚠️  Skipping malformed line {line_no}: {line!r} ({e})")
    return uid_to_data


def build_score_dict(uid_to_data: dict, config: dict) -> dict:
    """Mirror the validator's score_dict initialization."""
    small_molecule_target = config["small_molecule_target"]
    nanobody_target = config["nanobody_target"]
    return {
        uid: {
            "molecule_scores": [[] for _ in range(len(small_molecule_target))],
            "nanobody_scores": [[] for _ in range(len(nanobody_target))],
            "entropy": None,
            "block_submitted": uid_to_data[uid].get("block_submitted"),
            "push_time": uid_to_data[uid].get("push_time", ""),
        }
        for uid in uid_to_data
    }


def show_boltz_result(boltz: BoltzResult | None, config: dict) -> None:
    """Pretty-print the contents of a ``BoltzResult`` with emoji-friendly logs."""
    if boltz is None:
        bt.logging.info("🚫 Boltz scoring was skipped (run_boltz=False or no valid molecules).")
        return

    targets = config["small_molecule_target"]
    n_unique = len(boltz.unique_molecules)
    n_uids = len(boltz.per_molecule_components)
    bt.logging.info(f"📦 BoltzResult: {n_unique} unique molecules across {n_uids} UID(s), targets={targets}")

    for smiles, ids in boltz.unique_molecules.items():
        bt.logging.debug(f"🧬 Unique SMILES {smiles} -> {ids}")

    for uid, by_smiles in boltz.per_molecule_components.items():
        bt.logging.info(f"👤 UID={uid}: {len(by_smiles)} scored molecule(s)")
        for smiles, by_target in by_smiles.items():
            bt.logging.info(f"  🧪 SMILES={smiles}")
            for target, metrics in by_target.items():
                bt.logging.info(f"    🎯 target={target} metrics={metrics}")


def main() -> None:
    args = parse_args()
    bt.logging.set_info()

    bt.logging.info(f"🚀 eval_molecule starting (input={args.local_input_file}, run_boltz={args.run_boltz})")

    # 1) Load config (dict-shaped, same as validator).
    config = load_config(args.config)
    bt.logging.info(
        f"⚙️  Loaded config: small_molecule_target={config['small_molecule_target']}, "
        f"num_molecules={config['num_molecules']}, random_valid_reaction={config['random_valid_reaction']}"
    )

    # 2) Load local input.
    uid_to_data = load_local_input(args.local_input_file)
    if not uid_to_data:
        bt.logging.error("❌ No submissions to evaluate. Exiting.")
        return
    bt.logging.info(f"📥 Loaded {len(uid_to_data)} submission(s): uids={list(uid_to_data)}")

    # 3) Initialize scoring structure.
    score_dict = build_score_dict(uid_to_data, config)

    # 4) Validate molecules and compute entropy (validator helper, unmodified).
    bt.logging.info("🔬 Validating molecules and computing entropy...")
    valid_molecules_by_uid = validate_molecules_and_calculate_entropy(
        uid_to_data=uid_to_data,
        score_dict=score_dict,
        config=config,
        allowed_reaction=args.allowed_reaction,
    )

    if not valid_molecules_by_uid:
        bt.logging.warning("⚠️  No UIDs passed molecule validation.")
    else:
        bt.logging.info(f"✅ {len(valid_molecules_by_uid)} UID(s) passed validation:")
        for uid, payload in valid_molecules_by_uid.items():
            entropy = score_dict[uid].get("entropy")
            bt.logging.info(
                f"  ✨ UID={uid}: {len(payload['smiles'])} valid molecule(s), entropy={entropy}"
            )
            for name, smiles in zip(payload["names"], payload["smiles"]):
                bt.logging.debug(f"    🧪 {name} -> {smiles}")

    # 5) Optionally run Boltz inference and build the BoltzResult.
    run_boltz = args.run_boltz and bool(valid_molecules_by_uid)
    per_molecule_components: dict = {}
    unique_molecules: dict = {}

    if run_boltz:
        bt.logging.info("🧠 Running Boltz inference...")
        try:
            from external_tools.boltz.boltz_wrapper import BoltzWrapper

            wrapper = BoltzWrapper()
            wrapper.score_molecules(valid_molecules_by_uid, score_dict, config)
            per_molecule_components = getattr(wrapper, "per_molecule_components", {}) or {}
            unique_molecules = getattr(wrapper, "unique_molecules", {}) or {}
            bt.logging.info("🏁 Boltz inference complete.")
        except Exception as e:
            bt.logging.error(f"💥 Boltz inference failed: {e}")
            run_boltz = False
    elif args.run_boltz and not valid_molecules_by_uid:
        bt.logging.warning("⚠️  --run_boltz set, but no valid molecules to score; skipping Boltz.")

    boltz = BoltzResult(per_molecule_components, unique_molecules) if run_boltz else None

    # 6) Show the boltz result.
    show_boltz_result(boltz, config)

    bt.logging.info("🎉 eval_molecule done.")


if __name__ == "__main__":
    main()
