"""
Lenient Grading Script

This script grades submissions using the adversarial grading system with 
lenient_messages=True, meaning it won't penalize minor differences in 
input/output/print message wording as long as the semantic meaning is correct.

Output columns:
- student_id
- initial_input, initial_logic, initial_syntax, initial_print (numeric scores)
- initial_explanation (overall feedback)
- final_input, final_logic, final_syntax, final_print (numeric scores)  
- final_explanation (overall feedback)
"""

import pandas as pd
import json
import time
from grader import AdversarialGradingSystem
import os


def extract_scores(grade_dict: dict) -> dict:
    """
    Extract numeric scores from grade dictionary.
    
    Maps rubric IDs to simplified column names:
    - input -> input
    - conditional_logic -> logic
    - printing -> print
    - syntax_errors -> syntax
    
    Args:
        grade_dict: Dictionary with rubric item IDs as keys
        
    Returns:
        Dictionary with simplified keys and numeric option values
    """
    mapping = {
        'input': 'input',
        'conditional_logic': 'logic',
        'printing': 'print',
        'syntax_errors': 'syntax'
    }
    
    scores = {}
    for rubric_id, simple_name in mapping.items():
        if rubric_id in grade_dict:
            scores[simple_name] = grade_dict[rubric_id].get('option', None)
        else:
            scores[simple_name] = None
    
    return scores


def main():
    # === CONFIGURATION ===
    INPUT_CSV_PATH = "data/graded/diag1_100_random_batch1.csv"
    RUBRIC_PATH = "data/rubrics/raw/diagnostic1/rubric.json"
    OUTPUT_CSV_PATH = "results/diag1_batch1_lenient.csv"

    # Create output directory if not exist
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    # === LOAD DATA ===
    print("Loading CSV file...")
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Selected {len(df)} rows for grading.")

    # === LOAD RUBRIC ===
    print("Loading rubric...")
    with open(RUBRIC_PATH, "r") as f:
        rubric = json.load(f)

    # === INITIALIZE GRADING SYSTEM WITH LENIENT MESSAGES ===
    print("Initializing adversarial grading system with lenient_messages=True...")
    system = AdversarialGradingSystem(lenient_messages=True)

    # Try to load existing results if the file exists (for resuming)
    if os.path.exists(OUTPUT_CSV_PATH):
        existing_df = pd.read_csv(OUTPUT_CSV_PATH)
        completed_ids = set(existing_df["student_id"])
        results = existing_df.to_dict(orient="records")
        print(f"Resuming from {len(results)} already graded submissions.")
    else:
        completed_ids = set()
        results = []

    for i, row in df.iterrows():
        student_id = row["student_id"]
        if student_id in completed_ids:
            continue  # Skip if already graded
        
        code = row["code"]
        try:
            print(f"\n[{i+1}/{len(df)}] Grading submission for student_id: {student_id}...")
            result = system.adversarial_grade(code, rubric)
            
            # Extract initial scores
            initial_grade = result.get("initial_grade", {})
            initial_scores = extract_scores(initial_grade)
            initial_feedback = result.get("initial_feedback", "")
            
            # Extract final scores
            final_grade = result.get("final_grade", {})
            final_scores = extract_scores(final_grade)
            final_feedback = result.get("final_feedback", "")
            
            # Build flat result row
            flat_result = {
                "student_id": student_id,
                "initial_input": initial_scores["input"],
                "initial_logic": initial_scores["logic"],
                "initial_syntax": initial_scores["syntax"],
                "initial_print": initial_scores["print"],
                "initial_explanation": initial_feedback,
                "final_input": final_scores["input"],
                "final_logic": final_scores["logic"],
                "final_syntax": final_scores["syntax"],
                "final_print": final_scores["print"],
                "final_explanation": final_feedback,
            }

            results.append(flat_result)

            # Save periodically every 5 submissions
            if len(results) % 5 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False)
                print(f"💾 Saved {len(results)} results so far...")
                time.sleep(1)

        except Exception as e:
            print(f"❌ Error grading student_id {student_id} (row {i}): {e}")
            continue

    # === FINAL SAVE ===
    pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Finished grading {len(results)} submissions.")
    print(f"📁 Results saved to: {OUTPUT_CSV_PATH}")
    
    # Print summary statistics
    results_df = pd.DataFrame(results)
    print("\n=== GRADING SUMMARY ===")
    for stage in ['initial', 'final']:
        print(f"\n{stage.upper()} GRADES:")
        for cat in ['input', 'logic', 'syntax', 'print']:
            col = f"{stage}_{cat}"
            if col in results_df.columns:
                mean_val = results_df[col].mean()
                print(f"  {cat}: mean={mean_val:.2f}")


if __name__ == "__main__":
    main()
