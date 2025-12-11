"""
This script is used to sample a dataset for human grading comparison.
"""

import json
import pandas as pd
import random
import ast
import os
import argparse
from collections import Counter

def load_used_ids(used_ids_file):
    """Load previously used student IDs from tracking file."""
    if not used_ids_file or not os.path.exists(used_ids_file):
        return set()
    try:
        with open(used_ids_file, "r") as f:
            data = json.load(f)
            return set(data.get("used_ids", []))
    except Exception as e:
        print(f"Warning: Could not load used_ids file: {e}")
        return set()

def save_used_ids(used_ids_file, new_ids, existing_ids=None):
    """Save used student IDs to tracking file."""
    if not used_ids_file:
        return
    
    # Create directory if needed
    os.makedirs(os.path.dirname(used_ids_file) if os.path.dirname(used_ids_file) else ".", exist_ok=True)
    
    # Merge with existing IDs
    all_ids = (existing_ids or set()) | set(new_ids)
    
    data = {
        "used_ids": sorted(list(all_ids)),
        "total_count": len(all_ids)
    }
    
    with open(used_ids_file, "w") as f:
        json.dump(data, f, indent=2)

def main(
    json_path="data/cip5/upload/cip5_diagnostic_feedback.json",
    csv_path="data/cip5/processed/cip5_student_data.csv",
    output_dir="samples",
    diag_name="diagnostic1",
    total_samples=100,
    error_target=70,
    random_seed=42,
    used_ids_file=None,
    batch_number=None,
):
    """
    Prepare a sample dataset for human grading comparison.
    Supports batch sampling to avoid overlapping data across multiple runs.

    Args:
        json_path: Path to grading JSON file
        csv_path: Path to student data CSV
        output_dir: Directory for output files
        diag_name: Diagnostic exercise name (e.g., "diagnostic1")
        total_samples: Number of samples to generate
        error_target: Target number of error cases to include
        random_seed: Random seed for reproducibility
        used_ids_file: Optional path to JSON file tracking used student IDs across batches.
                      If None, no tracking is performed (original behavior).
        batch_number: Optional batch number. If provided, files are saved in 
                      {output_dir}/batch{batch_number}/ folder. If None, files are saved directly in output_dir.

    Outputs:
    If batch_number is provided, files are saved in {output_dir}/batch{batch_number}/:
    1) selected_{total_samples}_{diag_name}.csv  - combined sample (ordered)
    2) selected_{total_samples}_{diag_name}_random.csv  - randomized order
    3) selected_{total_samples}_{diag_name}_with_results.csv  - with results
    4) report_{diag_name}.txt  - error distribution summary
    
    If batch_number is None, files are saved directly in output_dir with the same naming.
    """

    random.seed(random_seed)
    
    # Load previously used IDs if tracking is enabled
    used_ids = load_used_ids(used_ids_file) if used_ids_file else set()
    if used_ids:
        print(f"Found {len(used_ids)} previously used student IDs. Excluding them from sampling.")

    # ---------- Load grading JSON ----------
    with open(json_path, "r") as f:
        data = json.load(f)
    df_json = pd.DataFrame(data)
    df_json = df_json[df_json["diag_exercise"] == diag_name].copy()

    # ---------- Helper: parse grading dict ----------
    def parse_grading(g):
        if isinstance(g, str):
            try:
                return ast.literal_eval(g)
            except Exception:
                return {}
        elif isinstance(g, dict):
            return g
        return {}

    df_json["grading_dict"] = df_json["grading"].apply(parse_grading)

    def has_error(gdict):
        return any(v != 0 for v in gdict.values()) if isinstance(gdict, dict) else False

    df_json["has_error"] = df_json["grading_dict"].apply(has_error)

    # ---------- Extract student_ids ----------
    error_ids = df_json.loc[df_json["has_error"], "student_id"].unique().tolist()
    # Exclude already used IDs
    available_error_ids = [sid for sid in error_ids if sid not in used_ids]
    
    if len(available_error_ids) < error_target:
        print(f"Warning: Only {len(available_error_ids)} error cases available (excluding used IDs). "
              f"Requested {error_target}, will use {len(available_error_ids)}.")
    
    n_error = min(error_target, len(available_error_ids))
    selected_error_ids = random.sample(available_error_ids, n_error)
    print(f"Found {len(error_ids)} total error cases ({len(available_error_ids)} available), "
          f"sampled {n_error} of them.")

    # ---------- Load CSV (student submissions) ----------
    df_csv = pd.read_csv(csv_path)
    df_csv_diag1 = df_csv[df_csv["diag_exercise"] == diag_name].copy()
    
    # Exclude already used IDs from the entire dataset
    df_csv_diag1 = df_csv_diag1[~df_csv_diag1["student_id"].isin(used_ids)]

    subset_error = df_csv_diag1[df_csv_diag1["student_id"].isin(selected_error_ids)]
    remaining = df_csv_diag1[~df_csv_diag1["student_id"].isin(selected_error_ids)]

    n_remaining_needed = max(0, total_samples - len(subset_error))
    
    if len(remaining) < n_remaining_needed:
        print(f"Warning: Only {len(remaining)} non-error cases available (excluding used IDs). "
              f"Requested {n_remaining_needed}, will use {len(remaining)}.")
    
    subset_random = remaining.sample(
        n=min(n_remaining_needed, len(remaining)), random_state=random_seed
    )

    final_sample = pd.concat([subset_error, subset_random], ignore_index=True)
    print(f"Sampled {len(subset_random)} random examples to reach {len(final_sample)} total.")

    # ---------- Merge grading info ----------
    df_merged = final_sample.merge(
        df_json[["student_id", "grading_dict"]],
        on="student_id",
        how="left"
    )

    # ---------- Create randomized order ----------
    df_randomized = df_merged.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_randomized = df_randomized.drop(columns=["grading_dict"], errors="ignore")

    # ---------- Error count report ----------
    def error_count(gdict):
        if not gdict or not isinstance(gdict, dict):
            return None
        return sum(1 for v in gdict.values() if v != 0)

    df_merged["error_count"] = df_merged["grading_dict"].apply(error_count)

    counts = Counter(df_merged["error_count"].fillna(-1))
    total = len(df_merged)
    one_error = counts.get(1, 0)
    multi_error = sum(v for k, v in counts.items() if k > 1)
    perfect = counts.get(0, 0)

    report = (
        f"📊 Report Summary for {diag_name}\n"
        f"----------------------------------\n"
        f"Total samples: {total}\n"
        f"Perfect (all 0s): {perfect}\n"
        f"One error: {one_error}\n"
        f"Two or more errors: {multi_error}\n"
    )

    print(report)

    # ---------- Save outputs ----------
    # Create batch-specific folder if batch_number is provided
    if batch_number is not None:
        batch_dir = os.path.join(output_dir, f"batch{batch_number}")
        os.makedirs(batch_dir, exist_ok=True)
        base = os.path.join(batch_dir, f"selected_{total_samples}_{diag_name}")
        report_filename = os.path.join(batch_dir, f"report_{diag_name}.txt")
    else:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, f"selected_{total_samples}_{diag_name}")
        report_filename = os.path.join(output_dir, f"report_{diag_name}.txt")
    
    df_merged.to_csv(f"{base}.csv", index=False)
    df_randomized.to_csv(f"{base}_random.csv", index=False)
    df_merged.to_csv(f"{base}_with_results.csv", index=False)
    with open(report_filename, "w") as f:
        f.write(report)

    print(f"✅ Files created:\n"
          f" - {base}.csv\n"
          f" - {base}_random.csv\n"
          f" - {base}_with_results.csv\n"
          f" - {report_filename}")

    # ---------- Update tracking file if enabled ----------
    if used_ids_file:
        new_ids = df_merged["student_id"].unique().tolist()
        save_used_ids(used_ids_file, new_ids, used_ids)
        print(f"✅ Updated tracking file: {used_ids_file} "
              f"(now tracking {len(used_ids) + len(new_ids)} total IDs)")

    return df_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample dataset for human grading comparison with batch support"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="data/cip5/upload/cip5_diagnostic_feedback.json",
        help="Path to grading JSON file"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/cip5/processed/cip5_student_data.csv",
        help="Path to student data CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="samples",
        help="Directory for output files"
    )
    parser.add_argument(
        "--diag-name",
        type=str,
        default="diagnostic1",
        help="Diagnostic exercise name (e.g., 'diagnostic1')"
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--error-target",
        type=int,
        default=70,
        help="Target number of error cases to include"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--used-ids-file",
        type=str,
        default=None,
        help="Path to JSON file tracking used student IDs across batches (optional)"
    )
    parser.add_argument(
        "--batch-number",
        type=int,
        default=None,
        help="Batch number for output file naming (optional)"
    )
    
    args = parser.parse_args()
    
    main(
        json_path=args.json_path,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        diag_name=args.diag_name,
        total_samples=args.total_samples,
        error_target=args.error_target,
        random_seed=args.random_seed,
        used_ids_file=args.used_ids_file,
        batch_number=args.batch_number,
    )
