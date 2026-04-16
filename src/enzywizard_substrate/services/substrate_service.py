from __future__ import annotations

from pathlib import Path
from ..utils.logging_utils import Logger
from ..utils.IO_utils import write_json_from_dict_inline_leaf_lists, save_substrate_structures, check_filename_length
from ..algorithms.substrate_algorithms import get_substrate_dict_list_from_input, get_completed_smiles_list,get_substrate_feature_list,generate_substrate_report
from ..utils.substrate_utils import get_substrate_report_suffix_from_feature_list
from ..utils.common_utils import get_optimized_filename

def run_substrate_service(substrate_names: str, output_dir: str | Path, max_synonyms: int = 20, fp_radius: int = 2, n_bits: int = 512, num_confs: int = 5, prune_rms: float = 0.5) -> bool:
    # ---- logger ----
    logger = Logger(output_dir)
    logger.print(f"[INFO] Substrate processing started: {substrate_names}")

    # ---- check output ----
    if max_synonyms <= 0 or max_synonyms > 200 or fp_radius <= 0 or fp_radius > 5 or n_bits <= 0 or n_bits > 2048 or num_confs <= 0 or num_confs > 20 or prune_rms <= 0 or prune_rms > 5.0:
        logger.print(
            f"[ERROR] Invalid substrate generation parameters. Require: max_synonyms (1–200), fp_radius (1–5), n_bits (1–2048), num_confs (1–20), prune_rms (0–5]."
        )
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- get substrates ----
    substrate_dict_list = get_substrate_dict_list_from_input(substrate_names, logger)
    if substrate_dict_list is None:
        return False
    logger.print("[INFO] Substrate input resolved")

    # ---- run algorithm ----
    substrate_dict_list = get_completed_smiles_list(substrate_dict_list, logger, max_synonyms=max_synonyms)
    if substrate_dict_list is None:
        return False
    logger.print("[INFO] Substrate SMILES completed")

    substrate_feature_list = get_substrate_feature_list(substrate_dict_list, logger, fp_radius=fp_radius, n_bits=n_bits, num_confs=num_confs, prune_rms=prune_rms)
    if substrate_feature_list is None:
        return False
    logger.print("[INFO] Substrate calculation finished")

    # ---- write output ----
    if not save_substrate_structures(substrate_feature_list, output_dir, logger):
        return False
    logger.print(f"[INFO] Substrate structures saved: {output_dir}")

    report = generate_substrate_report(substrate_feature_list, logger)
    if report is None:
        return False

    # ---- write output ----
    suffix=get_substrate_report_suffix_from_feature_list(substrate_feature_list,logger)
    if suffix is None:
        return False
    json_report_path = output_dir / get_optimized_filename(f"substrate_report_{suffix}.json")
    if not check_filename_length(json_report_path.stem,logger):
        return False
    write_json_from_dict_inline_leaf_lists(report, json_report_path)
    logger.print(f"[INFO] Report JSON saved: {json_report_path}")

    logger.print("[INFO] Substrate processing finished")
    return True