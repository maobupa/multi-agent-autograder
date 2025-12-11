import pandas as pd
import json
import time
from grader import AdversarialGradingSystem  # adjust import if needed
import os


def main():
    # === CONFIGURATION ===
    INPUT_CSV_PATH = "data/cip5/processed/cip5_student_data.csv"
    RUBRIC_PATH = "data/rubrics/raw/diagnostic1/rubric.json"
    OUTPUT_CSV_PATH = "results/diag1_100.csv"

    # create output directory if not exist
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    # === LOAD DATA ===
    print("Loading CSV file...")
    df = pd.read_csv(INPUT_CSV_PATH)

    # Filter by diagnostic1 and select first 100 rows
    df_filtered = df[df["diag_exercise"] == "diagnostic1"].head(100)
    print(f"Selected {len(df_filtered)} diagnostic1 rows for grading.")

    # === LOAD RUBRIC ===
    print("Loading rubric...")
    with open(RUBRIC_PATH, "r") as f:
        rubric = json.load(f)

    # === INITIALIZE GRADING SYSTEM ===
    print("Initializing adversarial grading system...")
    system = AdversarialGradingSystem()

    # Try to load existing results if the file exists
    if os.path.exists(OUTPUT_CSV_PATH):
        existing_df = pd.read_csv(OUTPUT_CSV_PATH)
        completed_ids = set(existing_df["student_id"])
        results = existing_df.to_dict(orient="records")
        print(f"Resuming from {len(results)} already graded submissions.")
    else:
        completed_ids = set()
        results = []

    for i, row in df_filtered.iterrows():
        if row.get("id", i) in completed_ids:
            continue # Skip if already graded
        code = row["code"]
        try:
            print(f"\n[{i}] Grading submission...")
            result = system.adversarial_grade(code, rubric)
            # Flatten/serialize nested fields
            flat_result = {
                "id": row.get("id", i),
                "initial_grade": json.dumps(result.get("initial_grade", {}), ensure_ascii=False),
                "final_grade": json.dumps(result.get("final_grade", {}), ensure_ascii=False),
                "change": json.dumps(result.get("change", {}), ensure_ascii=False),
                "debate_log": json.dumps(result.get("debate_log", []), ensure_ascii=False),
            }

            results.append(flat_result)

            # Save periodically
            if len(results) % 5 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False)
                print(f"Saved {len(results)} results so far...")
                time.sleep(1)

        except Exception as e:
            print(f"Error grading row {i}: {e}")
            continue

    # === FINAL SAVE ===
    pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Finished grading {len(results)} submissions. Saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
