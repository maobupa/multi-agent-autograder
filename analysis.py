"""
This script is an initial analysis of adversarial grading results
It looks at the overall distribution of changes and breakdown of them
"""
import pandas as pd
import json
import os

def parse_json(cell):
    """Safely parse JSON-like cell content."""
    try:
        return json.loads(cell)
    except Exception:
        return None

def is_full_score(initial):
    """Return True if all rubric options are 0."""
    if not isinstance(initial, dict):
        return False
    return all(
        (v.get("option") == 0) if isinstance(v, dict) else False
        for v in initial.values()
    )

def main(
    csv_path="results/diag1_batch1.csv",
    original_csv_path="data/graded/diag1_100_random_batch1.csv",
    output_changed_csv="results/analysis/diag1_batch1_changed.csv",
    output_summary_txt="results/analysis/diag1_batch1_summary.txt"
):
    # === 1. Load and parse results ===
    df = pd.read_csv(csv_path)
    df["change"] = df["change"].apply(parse_json)
    df["initial_grade"] = df["initial_grade"].apply(parse_json)
    
    # === 2. Load original data with student code ===
    original_df = pd.read_csv(original_csv_path)
    # Merge to get the code column
    df = df.merge(original_df[["student_id", "code"]], on="student_id", how="left")

    # === 3. Compute columns ===
    df["changed"] = df["change"].apply(lambda x: bool(x and x.get("items_changed")))
    df["delta"] = df["change"].apply(lambda x: x.get("overall_delta") if x else 0)

    # === 4. Summary stats ===
    total = len(df)
    changed = df["changed"].sum()
    unchanged = total - changed

    pos = df[df["delta"] > 0].shape[0]
    neg = df[df["delta"] < 0].shape[0]
    zero = df[df["delta"] == 0].shape[0]

    # Build summary text
    summary_lines = []
    summary_lines.append("=== Debate Result Summary ===")
    summary_lines.append(f"Total rows: {total}")
    summary_lines.append(f"Changed: {changed} ({changed/total:.2%})")
    if changed > 0:
        summary_lines.append(f"  Positive Δ: {pos} ({pos/changed:.2%} of changed)")
        summary_lines.append(f"  Negative Δ: {neg} ({neg/changed:.2%} of changed)")
        summary_lines.append(f"  Zero Δ:     {zero} ({zero/changed:.2%} of changed)")
    summary_lines.append(f"No Change: {unchanged} ({unchanged/total:.2%})")

    # === 5. Unchanged breakdown ===
    unchanged_df = df[~df["changed"]]
    unchanged_full = unchanged_df["initial_grade"].apply(is_full_score).sum()

    if unchanged > 0:
        summary_lines.append("\n=== Breakdown of Unchanged ===")
        summary_lines.append(f"Unchanged total: {unchanged}")
        summary_lines.append(f"  Full-score unchanged: {unchanged_full} ({unchanged_full/unchanged:.2%})")
        summary_lines.append(f"  Non–full-score unchanged: {unchanged - unchanged_full} ({(unchanged - unchanged_full)/unchanged:.2%})")

    # Print to console
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # === 6. Save summary to file ===
    os.makedirs(os.path.dirname(output_summary_txt), exist_ok=True)
    with open(output_summary_txt, "w") as f:
        f.write(summary_text)
    print(f"\n✅ Summary saved to {output_summary_txt}")

    # === 7. Extract and save changed rows ===
    changed_df = df[df["changed"]]
    if len(changed_df) > 0:
        # Reorder columns to put code first after student_id
        cols = ["student_id", "code"] + [col for col in changed_df.columns if col not in ["student_id", "code", ]]
        changed_df = changed_df[cols]
        
        changed_df.to_csv(output_changed_csv, index=False)
        print(f"✅ Changed rows ({len(changed_df)}) saved to {output_changed_csv}")
    else:
        print("⚠️  No changed rows to export.")

    return df  # return full dataframe for further analysis

# Run only if executed directly
if __name__ == "__main__":
    main()
