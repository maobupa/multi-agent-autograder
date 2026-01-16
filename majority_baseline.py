"""
Majority Baseline Grading

This script generates baseline grading results by:
1. Grading each student submission 3 times using the initial Grader
2. Using majority vote to determine the final grade for each category
3. Saving results as new columns in the compiled grading file
"""

import pandas as pd
import json
import time
from grader import Grader
from statistics import mode, StatisticsError
from tqdm import tqdm
import os

# Configuration
INPUT_CSV_PATH = "results/analysis/batch1_compiled_grading_clean.csv"
RUBRIC_PATH = "data/rubrics/raw/diagnostic1/rubric.json"
OUTPUT_CSV_PATH = "results/analysis/batch1_compiled_grading_maj.csv"

# Mapping from rubric item IDs to output column names
CATEGORY_MAPPING = {
    'input': 'base_input',
    'conditional_logic': 'base_logic',
    'printing': 'base_print',
    'syntax_errors': 'base_syntax'
}

# Number of times to grade each submission
NUM_GRADES = 3


def get_majority_vote(grades_list):
    """
    Get the majority vote from a list of grades.
    If no clear majority, return the first grade.
    """
    if not grades_list:
        return None
    try:
        return mode(grades_list)
    except StatisticsError:
        # No unique mode - return the most common or first
        return grades_list[0]


def extract_category_grade(grading_result, category_id):
    """
    Extract the grade (option number) for a specific category from grading result.
    Handles potential variations in category naming.
    """
    scores = grading_result.get('scores', {})
    
    # Try exact match first
    if category_id in scores:
        return scores[category_id].get('option', None)
    
    # Try case-insensitive match
    for key in scores:
        if key.lower() == category_id.lower():
            return scores[key].get('option', None)
        # Handle variations like 'conditional_logic' vs 'conditionalLogic'
        if key.lower().replace('_', '') == category_id.lower().replace('_', ''):
            return scores[key].get('option', None)
    
    return None


def grade_with_majority(grader, code, rubric, num_grades=3):
    """
    Grade a submission multiple times and use majority vote for each category.
    
    Args:
        grader: Grader instance
        code: Student code string
        rubric: Rubric dictionary
        num_grades: Number of times to grade (default 3)
    
    Returns:
        Dictionary mapping category to majority vote grade
    """
    # Collect grades from multiple runs
    all_grades = {cat: [] for cat in CATEGORY_MAPPING.keys()}
    
    for _ in range(num_grades):
        try:
            result = grader.grade(code, rubric)
            
            for category_id in CATEGORY_MAPPING.keys():
                grade = extract_category_grade(result, category_id)
                if grade is not None:
                    all_grades[category_id].append(grade)
        except Exception as e:
            print(f"  Warning: Grading attempt failed: {e}")
            continue
    
    # Calculate majority vote for each category
    majority_grades = {}
    for category_id, output_col in CATEGORY_MAPPING.items():
        grades_list = all_grades[category_id]
        if grades_list:
            majority_grades[output_col] = get_majority_vote(grades_list)
        else:
            majority_grades[output_col] = None
    
    return majority_grades


def main():
    print("=" * 70)
    print("MAJORITY BASELINE GRADING")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(df)} submissions")
    
    # Load rubric
    print(f"Loading rubric from: {RUBRIC_PATH}")
    with open(RUBRIC_PATH, "r") as f:
        rubric = json.load(f)
    
    # Initialize grader
    print("Initializing Grader...")
    grader = Grader()
    
    # Initialize new columns
    for col in CATEGORY_MAPPING.values():
        df[col] = None
    
    # Check for existing progress (if output file exists with some results)
    if os.path.exists(OUTPUT_CSV_PATH):
        existing_df = pd.read_csv(OUTPUT_CSV_PATH)
        if 'base_input' in existing_df.columns:
            # Copy existing results
            for col in CATEGORY_MAPPING.values():
                if col in existing_df.columns:
                    df[col] = existing_df[col]
            completed_mask = df['base_input'].notna()
            completed_count = completed_mask.sum()
            print(f"Found {completed_count} already graded submissions")
        else:
            completed_count = 0
    else:
        completed_count = 0
    
    # Grade each submission
    print(f"\nGrading {len(df)} submissions with {NUM_GRADES} grades each...")
    print("Each submission will use majority vote across grades.\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Grading"):
        # Skip if already graded
        if pd.notna(df.loc[idx, 'base_input']):
            continue
        
        student_id = row['student_id']
        code = row['code']
        
        try:
            # Grade with majority vote
            majority_grades = grade_with_majority(grader, code, rubric, NUM_GRADES)
            
            # Store results
            for col, grade in majority_grades.items():
                df.loc[idx, col] = grade
            
            # Save periodically
            if (idx + 1) % 10 == 0:
                df.to_csv(OUTPUT_CSV_PATH, index=False)
                
        except Exception as e:
            print(f"\nError grading student {student_id}: {e}")
            continue
    
    # Final save
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Saved results to: {OUTPUT_CSV_PATH}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for col in CATEGORY_MAPPING.values():
        valid_count = df[col].notna().sum()
        print(f"{col}: {valid_count}/{len(df)} graded")
    
    print(f"\nNew columns added: {list(CATEGORY_MAPPING.values())}")


if __name__ == "__main__":
    main()
