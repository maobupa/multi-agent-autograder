"""
Perturbation Test for Grading Consistency

This script:
1. Loads 100 programs from the compiled grading data
2. Copies original grading results (initial & final) from existing CSV
3. Generates surface-level perturbations (variable renaming, comments)
4. Runs adversarial grading on perturbed code only (saves LLM calls)
5. Saves all results to a new CSV

Output CSV columns:
- student_id, code, perturbed_code, changes_made
- original_initial_input/logic/syntax/print (copied from existing data)
- original_final_input/logic/syntax/print (copied from existing data)
- perturbed_initial_input/logic/syntax/print (from new grading)
- perturbed_final_input/logic/syntax/print (from new grading)
"""

import pandas as pd
import json
import os
import argparse
from tqdm import tqdm

from src.llm_agents import PerturbationAgent
from src.grader import AdversarialGradingSystem


def load_rubric(rubric_path: str = "data/rubrics/raw/diagnostic1/rubric.json") -> dict:
    """Load the rubric from JSON file."""
    with open(rubric_path, 'r') as f:
        return json.load(f)


def extract_scores_from_json(grade_json: str, prefix: str = "") -> dict:
    """
    Extract scores from JSON string grade result into flat dictionary.
    
    Args:
        grade_json: JSON string with rubric item scores
        prefix: Prefix for column names (e.g., 'perturbed_initial_')
        
    Returns:
        Dictionary with flattened scores
    """
    try:
        scores = json.loads(grade_json) if isinstance(grade_json, str) else grade_json
    except (json.JSONDecodeError, TypeError):
        scores = {}
    
    # Map rubric IDs to column names
    id_mapping = {
        'input': 'input',
        'conditional_logic': 'logic',
        'printing': 'print',
        'syntax_errors': 'syntax'
    }
    
    result = {}
    for rubric_id, col_name in id_mapping.items():
        score_data = scores.get(rubric_id, {})
        if isinstance(score_data, dict):
            result[f'{prefix}{col_name}'] = score_data.get('option', None)
        else:
            result[f'{prefix}{col_name}'] = None
    
    return result


def run_perturbation_test(
    input_csv: str = "results/analysis/batch1_compiled_grading_clean.csv",
    original_grades_csv: str = "results/diag1_batch1.csv",
    output_csv: str = "results/analysis/perturbation_test_results.csv",
    rubric_path: str = "data/rubrics/raw/diagnostic1/rubric.json",
    num_samples: int = None,
    perturbation_types: list = None
):
    """
    Run the perturbation test.
    
    Args:
        input_csv: Path to input CSV with original programs
        original_grades_csv: Path to CSV with original grading results (initial & final)
        output_csv: Path to save results
        rubric_path: Path to rubric JSON
        num_samples: Number of samples to process (None = all)
        perturbation_types: Types of perturbations to apply
    """
    print("="*60)
    print("PERTURBATION TEST FOR GRADING CONSISTENCY")
    print("="*60)
    
    # Load program data
    print(f"\nLoading programs from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} programs")
    
    # Load original grading results
    print(f"\nLoading original grades from {original_grades_csv}...")
    grades_df = pd.read_csv(original_grades_csv)
    print(f"  Loaded {len(grades_df)} grading records")
    
    if num_samples is not None:
        df = df.head(num_samples)
        print(f"  Using first {num_samples} samples")
    
    # Load rubric
    print(f"\nLoading rubric from {rubric_path}...")
    rubric = load_rubric(rubric_path)
    print("  Rubric loaded successfully")
    
    # Initialize agents
    print("\nInitializing LLM agents...")
    perturbation_agent = PerturbationAgent(temperature=0.7)
    grading_system = AdversarialGradingSystem()
    print("  Agents initialized")
    
    if perturbation_types is None:
        perturbation_types = ["rename_variables", "add_comments"]
    
    # Results storage
    results = []
    
    print(f"\nProcessing {len(df)} programs...")
    print("="*60)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        student_id = row['student_id']
        original_code = row['code']
        
        # Initialize result row
        result_row = {
            'student_id': student_id,
            'code': original_code,
        }
        
        # Step 1: Copy original grades from existing data
        original_grade_row = grades_df[grades_df['student_id'] == student_id]
        
        if len(original_grade_row) > 0:
            original_grade_row = original_grade_row.iloc[0]
            
            # Extract initial (single-LLM) scores
            initial_scores = extract_scores_from_json(
                original_grade_row.get('initial_grade', '{}'),
                'original_initial_'
            )
            result_row.update(initial_scores)
            
            # Extract final (multi-LLM) scores
            final_scores = extract_scores_from_json(
                original_grade_row.get('final_grade', '{}'),
                'original_final_'
            )
            result_row.update(final_scores)
        else:
            print(f"\n  Warning: No original grades found for {student_id}")
            for cat in ['input', 'logic', 'syntax', 'print']:
                result_row[f'original_initial_{cat}'] = None
                result_row[f'original_final_{cat}'] = None
        
        # Step 2: Generate perturbation
        try:
            perturbation_result = perturbation_agent.generate_perturbation(
                original_code, 
                perturbation_types
            )
            perturbed_code = perturbation_result['perturbed_code']
            changes_made = "; ".join(perturbation_result.get('changes_made', []))
        except Exception as e:
            print(f"\n  Warning: Perturbation failed for {student_id}: {e}")
            perturbed_code = original_code
            changes_made = f"Error: {str(e)}"
        
        result_row['perturbed_code'] = perturbed_code
        result_row['changes_made'] = changes_made
        
        # Step 3: Grade perturbed code with adversarial system (gets both initial & final)
        try:
            grading_result = grading_system.adversarial_grade(perturbed_code, rubric)
            
            # Extract initial (single-LLM) scores from perturbed grading
            perturbed_initial_scores = extract_scores_from_json(
                grading_result.get('initial_grade', {}),
                'perturbed_initial_'
            )
            result_row.update(perturbed_initial_scores)
            
            # Extract final (multi-LLM) scores from perturbed grading
            perturbed_final_scores = extract_scores_from_json(
                grading_result.get('final_grade', {}),
                'perturbed_final_'
            )
            result_row.update(perturbed_final_scores)
            
        except Exception as e:
            print(f"\n  Warning: Grading failed for perturbed {student_id}: {e}")
            for cat in ['input', 'logic', 'syntax', 'print']:
                result_row[f'perturbed_initial_{cat}'] = None
                result_row[f'perturbed_final_{cat}'] = None
        
        results.append(result_row)
        
        # Periodic save (every 10 samples)
        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_csv.replace('.csv', '_checkpoint.csv'), index=False)
    
    # Create final DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        'student_id', 'code', 'perturbed_code', 'changes_made',
        'original_initial_input', 'original_initial_logic', 
        'original_initial_syntax', 'original_initial_print',
        'original_final_input', 'original_final_logic', 
        'original_final_syntax', 'original_final_print',
        'perturbed_initial_input', 'perturbed_initial_logic', 
        'perturbed_initial_syntax', 'perturbed_initial_print',
        'perturbed_final_input', 'perturbed_final_logic', 
        'perturbed_final_syntax', 'perturbed_final_print'
    ]
    
    # Ensure all columns exist
    for col in column_order:
        if col not in results_df.columns:
            results_df[col] = None
    
    results_df = results_df[column_order]
    
    # Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print("PERTURBATION TEST COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_csv}")
    print(f"Total programs processed: {len(results_df)}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Run perturbation test for grading consistency'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='results/analysis/batch1_compiled_grading_clean.csv',
        help='Input CSV file with programs'
    )
    parser.add_argument(
        '--grades', '-g',
        type=str,
        default='results/diag1_batch1.csv',
        help='CSV file with original grading results'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/analysis/perturbation_test_results.csv',
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--rubric', '-r',
        type=str,
        default='data/rubrics/raw/diagnostic1/rubric.json',
        help='Path to rubric JSON file'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--perturbation-types', '-p',
        nargs='+',
        default=['rename_variables', 'add_comments'],
        help='Types of perturbations to apply'
    )
    
    args = parser.parse_args()
    
    run_perturbation_test(
        input_csv=args.input,
        original_grades_csv=args.grades,
        output_csv=args.output,
        rubric_path=args.rubric,
        num_samples=args.num_samples,
        perturbation_types=args.perturbation_types
    )


if __name__ == "__main__":
    main()
