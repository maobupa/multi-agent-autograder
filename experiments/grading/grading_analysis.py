"""
Comprehensive grading analysis for comparing LLM and human graders.

This script provides various statistical analyses including:
1. Initial vs Final LLM grading comparison
2. Pairwise agreement rates between graders
3. Inter-rater reliability metrics (Krippendorff's alpha)
4. Consensus distribution analysis (unanimous, strong, weak, split, disagreement)
5. Human baseline vs AI performance comparison (majority voting)

Usage:
    python grading_analysis.py --all
    python grading_analysis.py --initial-final-comparison
    python grading_analysis.py --pairwise-agreement
    python grading_analysis.py --inter-rater-reliability
    python grading_analysis.py --consensus-distribution
    python grading_analysis.py --human-vs-ai
"""

import pandas as pd
import numpy as np
import argparse
import os
from itertools import combinations
import json


# ============================================================================
# 1. INITIAL VS FINAL GRADING COMPARISON
# ============================================================================

def analyze_initial_final_changes(df, output_dir="results/analysis/batch1"):
    """
    Compare initial (single LLM) vs final (adversarial) grading.
    
    Returns:
        dict: Statistics about changes, strictness, and perfect scores
    """
    categories = ['input', 'logic', 'syntax', 'print']
    
    results = {
        'total_data_points': len(df) * 4,
        'total_submissions': len(df),
        'per_category': {},
        'overall': {}
    }
    
    total_changed = 0
    total_too_strict = 0
    total_too_lenient = 0
    total_unchanged = 0
    all_perfect_scores = 0
    
    print("="*80)
    print("INITIAL vs FINAL GRADING COMPARISON")
    print("="*80)
    print("\nNote: Lower score = better (0=perfect, 1=minor error, 2=major error)")
    print()
    
    for category in categories:
        initial_col = f'initial_{category}'
        final_col = f'final_{category}'
        
        # Drop rows where either value is NaN
        valid_mask = df[initial_col].notna() & df[final_col].notna()
        initial = df.loc[valid_mask, initial_col]
        final = df.loc[valid_mask, final_col]
        
        # Calculate changes
        changed_mask = initial != final
        changed_count = changed_mask.sum()
        unchanged_count = (~changed_mask).sum()
        total_count = len(initial)
        
        # Too strict: final > initial (worse score)
        too_strict_count = (final > initial).sum()
        
        # Too lenient: final < initial (better score)
        too_lenient_count = (final < initial).sum()
        
        # Store results (convert numpy types to Python types for JSON serialization)
        results['per_category'][category] = {
            'total': int(total_count),
            'changed': int(changed_count),
            'changed_pct': float(changed_count / total_count * 100) if total_count > 0 else 0,
            'unchanged': int(unchanged_count),
            'unchanged_pct': float(unchanged_count / total_count * 100) if total_count > 0 else 0,
            'too_strict': int(too_strict_count),
            'too_strict_pct': float(too_strict_count / changed_count * 100) if changed_count > 0 else 0,
            'too_lenient': int(too_lenient_count),
            'too_lenient_pct': float(too_lenient_count / changed_count * 100) if changed_count > 0 else 0,
        }
        
        total_changed += changed_count
        total_too_strict += too_strict_count
        total_too_lenient += too_lenient_count
        total_unchanged += unchanged_count
        
        # Print per-category results
        print(f"\n{category.upper()}:")
        print(f"  Changed: {changed_count}/{total_count} ({changed_count/total_count*100:.1f}%)")
        print(f"    Too strict (worse): {too_strict_count} ({too_strict_count/changed_count*100:.1f}% of changes)" if changed_count > 0 else "")
        print(f"    Too lenient (better): {too_lenient_count} ({too_lenient_count/changed_count*100:.1f}% of changes)" if changed_count > 0 else "")
        print(f"  Unchanged: {unchanged_count}/{total_count} ({unchanged_count/total_count*100:.1f}%)")
    
    # Check for perfect scores (all four categories = 0) in unchanged submissions
    all_cats_initial = [f'initial_{cat}' for cat in categories]
    all_cats_final = [f'final_{cat}' for cat in categories]
    
    # Only consider rows where all values are present
    complete_mask = df[all_cats_initial + all_cats_final].notna().all(axis=1)
    df_complete = df[complete_mask]
    
    # Check if any category changed
    unchanged_submissions = df_complete[
        (df_complete['initial_input'] == df_complete['final_input']) &
        (df_complete['initial_logic'] == df_complete['final_logic']) &
        (df_complete['initial_syntax'] == df_complete['final_syntax']) &
        (df_complete['initial_print'] == df_complete['final_print'])
    ]
    
    # Among unchanged, how many are perfect scores
    perfect_mask = (
        (unchanged_submissions['initial_input'] == 0) &
        (unchanged_submissions['initial_logic'] == 0) &
        (unchanged_submissions['initial_syntax'] == 0) &
        (unchanged_submissions['initial_print'] == 0)
    )
    all_perfect_scores = perfect_mask.sum()
    
    # Overall statistics (convert numpy types to Python types for JSON serialization)
    total_data_points = total_changed + total_unchanged
    results['overall'] = {
        'total_data_points': int(total_data_points),
        'changed': int(total_changed),
        'changed_pct': float(total_changed / total_data_points * 100) if total_data_points > 0 else 0,
        'unchanged': int(total_unchanged),
        'unchanged_pct': float(total_unchanged / total_data_points * 100) if total_data_points > 0 else 0,
        'too_strict': int(total_too_strict),
        'too_strict_pct': float(total_too_strict / total_changed * 100) if total_changed > 0 else 0,
        'too_lenient': int(total_too_lenient),
        'too_lenient_pct': float(total_too_lenient / total_changed * 100) if total_changed > 0 else 0,
        'unchanged_submissions': int(len(unchanged_submissions)),
        'unchanged_perfect_scores': int(all_perfect_scores),
        'unchanged_perfect_scores_pct': float(all_perfect_scores / len(unchanged_submissions) * 100) if len(unchanged_submissions) > 0 else 0,
        'unchanged_non_perfect': int(len(unchanged_submissions) - all_perfect_scores),
    }
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total data points: {total_data_points}")
    print(f"Changed: {total_changed} ({total_changed/total_data_points*100:.1f}%)")
    print(f"  Too strict: {total_too_strict} ({total_too_strict/total_changed*100:.1f}% of changes)")
    print(f"  Too lenient: {total_too_lenient} ({total_too_lenient/total_changed*100:.1f}% of changes)")
    print(f"Unchanged: {total_unchanged} ({total_unchanged/total_data_points*100:.1f}%)")
    print(f"\nSubmissions with NO changes across all categories: {len(unchanged_submissions)}")
    print(f"  Perfect scores (all 0s): {all_perfect_scores} ({all_perfect_scores/len(unchanged_submissions)*100:.1f}%)")
    print(f"  Non-perfect but unchanged: {len(unchanged_submissions) - all_perfect_scores} ({(len(unchanged_submissions)-all_perfect_scores)/len(unchanged_submissions)*100:.1f}%)")
    
    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "initial_final_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
    
    return results


# ============================================================================
# 2. PAIRWISE AGREEMENT ANALYSIS
# ============================================================================

def calculate_pairwise_agreement(df, output_dir="results/analysis/batch1"):
    """
    Calculate pairwise agreement rates between all graders for each category.
    
    Returns:
        dict: Agreement rates for all grader pairs
    """
    categories = ['input', 'logic', 'syntax', 'print']
    # Define all graders including the fifth human grader (shu)
    graders = {
        'briana': 'Human 1 (Briana)',
        'chen': 'Human 2 (Chen)',
        'grace': 'Human 3 (Grace)',
        'sanya': 'Human 4 (Sanya)',
        'shu': 'Human 5 (Shu)',
        'initial': 'Single-LLM',
        'final': 'Multi-LLM'
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("PAIRWISE AGREEMENT ANALYSIS")
    print("="*80)
    
    # Calculate agreement for each pair of graders
    grader_list = list(graders.keys())
    
    for grader1, grader2 in combinations(grader_list, 2):
        pair_name = f"{graders[grader1]} vs {graders[grader2]}"
        results[pair_name] = {}
        
        print(f"\n{pair_name}:")
        
        for category in categories:
            col1 = f'{grader1}_{category}'
            col2 = f'{grader2}_{category}'
            
            # Only compare where both have valid grades
            valid_mask = df[col1].notna() & df[col2].notna()
            
            if valid_mask.sum() == 0:
                results[pair_name][category] = None
                print(f"  {category.capitalize()}: No valid comparisons")
                continue
            
            grades1 = df.loc[valid_mask, col1]
            grades2 = df.loc[valid_mask, col2]
            
            agreement = (grades1 == grades2).sum()
            total = len(grades1)
            agreement_rate = (agreement / total) * 100
            
            results[pair_name][category] = {
                'agreement': int(agreement),
                'total': int(total),
                'rate': float(agreement_rate)
            }
            
            print(f"  {category.capitalize()}: {agreement_rate:.1f}% ({agreement}/{total})")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pairwise_agreement.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
    
    # Create summary table
    create_agreement_summary_table(results, graders, categories, output_dir)
    
    return results


def create_agreement_summary_table(results, graders, categories, output_dir):
    """Create a formatted table of pairwise agreements."""
    
    print("\n" + "="*80)
    print("PAIRWISE AGREEMENT SUMMARY TABLE")
    print("="*80)
    
    # Create a summary CSV
    rows = []
    for pair_name, cat_results in results.items():
        row = {'Grader Pair': pair_name}
        for category in categories:
            if cat_results[category] is not None:
                row[category.capitalize()] = f"{cat_results[category]['rate']:.1f}%"
            else:
                row[category.capitalize()] = "N/A"
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    output_file = os.path.join(output_dir, "pairwise_agreement_summary.csv")
    summary_df.to_csv(output_file, index=False)
    print(summary_df.to_string(index=False))
    print(f"\n✅ Summary table saved to {output_file}")


# ============================================================================
# 3. INTER-RATER RELIABILITY (KRIPPENDORFF'S ALPHA)
# ============================================================================

def calculate_krippendorff_alpha(df, output_dir="results/analysis/batch1"):
    """
    Calculate Krippendorff's alpha for inter-rater reliability.
    
    Requires: pip install krippendorff
    
    Returns:
        dict: Alpha values for different grader groups
    """
    try:
        import krippendorff
    except ImportError:
        print("⚠️  krippendorff package not installed. Install with: pip install krippendorff")
        return None
    
    categories = ['input', 'logic', 'syntax', 'print']
    
    # Define grader groups (5 human graders + 2 AI graders)
    human_graders = ['briana', 'chen', 'grace', 'sanya', 'shu']
    ai_graders = ['initial', 'final']
    all_graders = human_graders + ai_graders
    
    results = {
        'all_graders': {},
        'human_only': {},
        'ai_only': {},
        'by_category': {}
    }
    
    print("\n" + "="*80)
    print("INTER-RATER RELIABILITY (KRIPPENDORFF'S ALPHA)")
    print("="*80)
    print("\nInterpretation: α > 0.8 = reliable, 0.67-0.8 = tentative, < 0.67 = unreliable")
    
    for category in categories:
        print(f"\n{category.upper()}:")
        
        # Prepare data for all graders (rows=graders, cols=submissions)
        all_cols = [f'{grader}_{category}' for grader in all_graders]
        reliability_data_all = df[all_cols].T.values
        
        # All graders (7 = 5 humans + 2 AI)
        try:
            alpha_all = krippendorff.alpha(reliability_data_all, level_of_measurement='ordinal')
            results['all_graders'][category] = float(alpha_all) if alpha_all is not None else None
            print(f"  All graders (7): α = {alpha_all:.4f}")
        except Exception as e:
            print(f"  All graders: Error - {e}")
            results['all_graders'][category] = None
        
        # Human only (5 graders)
        human_cols = [f'{grader}_{category}' for grader in human_graders]
        reliability_data_human = df[human_cols].T.values
        try:
            alpha_human = krippendorff.alpha(reliability_data_human, level_of_measurement='ordinal')
            results['human_only'][category] = float(alpha_human) if alpha_human is not None else None
            print(f"  Human only (5): α = {alpha_human:.4f}")
        except Exception as e:
            print(f"  Human only: Error - {e}")
            results['human_only'][category] = None
        
        # AI only
        ai_cols = [f'{grader}_{category}' for grader in ai_graders]
        reliability_data_ai = df[ai_cols].T.values
        try:
            alpha_ai = krippendorff.alpha(reliability_data_ai, level_of_measurement='ordinal')
            results['ai_only'][category] = float(alpha_ai) if alpha_ai is not None else None
            print(f"  AI only (2): α = {alpha_ai:.4f}")
        except Exception as e:
            print(f"  AI only: Error - {e}")
            results['ai_only'][category] = None
        
        # Store by category for easy access (convert to float if not None)
        results['by_category'][category] = {
            'all': float(results['all_graders'][category]) if results['all_graders'][category] is not None else None,
            'human': float(results['human_only'][category]) if results['human_only'][category] is not None else None,
            'ai': float(results['ai_only'][category]) if results['ai_only'][category] is not None else None
        }
    
    # Overall summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    summary_rows = []
    for category in categories:
        row = {
            'Category': category.capitalize(),
            'All Graders': f"{results['all_graders'][category]:.4f}" if results['all_graders'][category] is not None else "N/A",
            'Human Only': f"{results['human_only'][category]:.4f}" if results['human_only'][category] is not None else "N/A",
            'AI Only': f"{results['ai_only'][category]:.4f}" if results['ai_only'][category] is not None else "N/A",
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "inter_rater_reliability.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
    
    csv_file = os.path.join(output_dir, "inter_rater_reliability_summary.csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"✅ Summary table saved to {csv_file}")
    
    return results


# ============================================================================
# 4. CONSENSUS DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_consensus_distribution(df, output_dir="results/analysis/batch1"):
    """
    Analyze consensus levels across all 6 graders for each category.
    
    Categorizes submissions by agreement level:
    - Unanimous: All 6 agree
    - Strong consensus: 5-6 agree
    - Weak consensus: 4 agree
    - Split: 3-3 or 3-2-1
    - High disagreement: All different
    
    Returns:
        dict: Consensus statistics for each category
    """
    from collections import Counter
    
    categories = ['input', 'logic', 'syntax', 'print']
    # All 7 graders: 5 human + 2 AI
    graders = ['briana', 'chen', 'grace', 'sanya', 'shu', 'initial', 'final']
    
    results = {}
    
    print("\n" + "="*80)
    print("CONSENSUS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    def categorize_agreement(grades_for_submission):
        """
        Categorize by agreement level across all 7 graders
        """
        # Filter out NaN values
        valid_grades = [g for g in grades_for_submission if pd.notna(g)]
        
        if len(valid_grades) == 0:
            return "no_data"
        
        grade_counts = Counter(valid_grades)
        unique_grades = len(grade_counts)
        max_count = max(grade_counts.values())
        
        if unique_grades == 1:
            return "unanimous"  # All agree
        elif max_count >= 6:
            return "strong_consensus"  # 6-7 agree
        elif max_count >= 5:
            return "weak_consensus"  # 5 agree
        elif max_count >= 4:
            return "split"  # 4 agree or split votes
        else:
            return "high_disagreement"  # Wide disagreement
    
    for category in categories:
        print(f"\n{category.upper()} Category:")
        
        # Get grades for all graders
        grade_cols = [f'{grader}_{category}' for grader in graders]
        
        # Categorize each submission
        consensus_types = []
        for idx, row in df.iterrows():
            grades = [row[col] for col in grade_cols]
            consensus_type = categorize_agreement(grades)
            consensus_types.append(consensus_type)
        
        # Count each consensus type
        consensus_counts = Counter(consensus_types)
        total_valid = sum(v for k, v in consensus_counts.items() if k != 'no_data')
        
        category_results = {}
        for consensus_type in ['unanimous', 'strong_consensus', 'weak_consensus', 'split', 'high_disagreement']:
            count = consensus_counts.get(consensus_type, 0)
            pct = (count / total_valid * 100) if total_valid > 0 else 0
            category_results[consensus_type] = {
                'count': int(count),
                'percentage': float(pct)
            }
            print(f"  {consensus_type.replace('_', ' ').title()}: {count} submissions ({pct:.1f}%)")
        
        results[category] = category_results
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "consensus_distribution.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
    
    # Create summary CSV
    rows = []
    for category, consensus_data in results.items():
        for consensus_type, data in consensus_data.items():
            rows.append({
                'Category': category.capitalize(),
                'Consensus Type': consensus_type.replace('_', ' ').title(),
                'Count': data['count'],
                'Percentage': f"{data['percentage']:.1f}%"
            })
    
    summary_df = pd.DataFrame(rows)
    csv_file = os.path.join(output_dir, "consensus_distribution_summary.csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"✅ Summary table saved to {csv_file}")
    
    return results


# ============================================================================
# 5. COMPARATIVE STUDY: HUMAN BASELINE VS AI PERFORMANCE
# ============================================================================

def analyze_human_vs_ai_performance(df, output_dir="results/analysis/batch1"):
    """
    Compare human baseline with AI performance using majority voting.
    
    For each grader, calculates:
    - Agreement with human majority
    - Agreement with all-graders majority
    
    Returns:
        dict: Agreement statistics for each grader
    """
    from collections import Counter
    
    categories = ['input', 'logic', 'syntax', 'print']
    # 5 human graders + 2 AI graders
    human_graders = ['briana', 'chen', 'grace', 'sanya', 'shu']
    ai_graders = ['initial', 'final']
    all_graders = human_graders + ai_graders
    
    grader_labels = {
        'briana': 'Human 1 (Briana)',
        'chen': 'Human 2 (Chen)',
        'grace': 'Human 3 (Grace)',
        'sanya': 'Human 4 (Sanya)',
        'shu': 'Human 5 (Shu)',
        'initial': 'Single-LLM',
        'final': 'Multi-LLM'
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("COMPARATIVE STUDY: HUMAN BASELINE VS AI PERFORMANCE")
    print("="*80)
    
    def majority_vote(grades):
        """Get most common grade, or None if tie"""
        valid_grades = [g for g in grades if pd.notna(g)]
        if len(valid_grades) == 0:
            return None
        counts = Counter(valid_grades)
        max_count = max(counts.values())
        # Check for tie
        if list(counts.values()).count(max_count) > 1:
            return None  # Tie, no clear majority
        return counts.most_common(1)[0][0]
    
    for category in categories:
        print(f"\n{category.upper()} Category:")
        
        category_results = {}
        
        # Calculate majority for each submission
        human_majorities = []
        all_majorities = []
        
        for idx, row in df.iterrows():
            # Human majority
            human_grades = [row[f'{grader}_{category}'] for grader in human_graders]
            human_maj = majority_vote(human_grades)
            human_majorities.append(human_maj)
            
            # All graders majority
            all_grades = [row[f'{grader}_{category}'] for grader in all_graders]
            all_maj = majority_vote(all_grades)
            all_majorities.append(all_maj)
        
        # For each grader, calculate agreement
        for grader in all_graders:
            grader_grades = df[f'{grader}_{category}']
            
            # Agreement with human majority
            agrees_with_human = []
            for i in range(len(df)):
                if pd.notna(grader_grades.iloc[i]) and human_majorities[i] is not None:
                    agrees_with_human.append(grader_grades.iloc[i] == human_majorities[i])
            
            human_agreement = sum(agrees_with_human) / len(agrees_with_human) * 100 if agrees_with_human else 0
            
            # Agreement with all-graders majority
            agrees_with_all = []
            for i in range(len(df)):
                if pd.notna(grader_grades.iloc[i]) and all_majorities[i] is not None:
                    agrees_with_all.append(grader_grades.iloc[i] == all_majorities[i])
            
            all_agreement = sum(agrees_with_all) / len(agrees_with_all) * 100 if agrees_with_all else 0
            
            category_results[grader] = {
                'human_majority_agreement': float(human_agreement),
                'human_majority_count': f"{sum(agrees_with_human)}/{len(agrees_with_human)}",
                'all_graders_majority_agreement': float(all_agreement),
                'all_graders_majority_count': f"{sum(agrees_with_all)}/{len(agrees_with_all)}"
            }
            
            print(f"  {grader_labels[grader]}:")
            print(f"    Agrees with human majority: {human_agreement:.1f}% ({sum(agrees_with_human)}/{len(agrees_with_human)})")
            print(f"    Agrees with all-graders majority: {all_agreement:.1f}% ({sum(agrees_with_all)}/{len(agrees_with_all)})")
        
        results[category] = category_results
        
        # Calculate average human agreement with human majority
        human_agreements = [category_results[h]['human_majority_agreement'] for h in human_graders]
        avg_human_agreement = sum(human_agreements) / len(human_agreements)
        
        print(f"\n  Average human agreement with human majority: {avg_human_agreement:.1f}%")
        results[category]['avg_human_agreement'] = float(avg_human_agreement)
    
    # Overall summary
    print("\n" + "="*80)
    print("SUMMARY: Agreement with Human Majority")
    print("="*80)
    
    summary_rows = []
    for category in categories:
        for grader in all_graders:
            summary_rows.append({
                'Category': category.capitalize(),
                'Grader': grader_labels[grader],
                'Human Majority Agreement': f"{results[category][grader]['human_majority_agreement']:.1f}%",
                'All Graders Majority Agreement': f"{results[category][grader]['all_graders_majority_agreement']:.1f}%"
            })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "human_vs_ai_performance.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
    
    csv_file = os.path.join(output_dir, "human_vs_ai_performance_summary.csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"✅ Summary table saved to {csv_file}")
    
    return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Grading analysis for LLM and human graders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python grading_analysis.py --all
  python grading_analysis.py --initial-final-comparison
  python grading_analysis.py --pairwise-agreement
  python grading_analysis.py --inter-rater-reliability
  python grading_analysis.py --consensus-distribution
  python grading_analysis.py --human-vs-ai
        """
    )
    
    parser.add_argument('--csv', type=str, 
                        default='results/analysis/batch1_compiled_grading_clean.csv',
                        help='Path to compiled grading CSV file')
    parser.add_argument('--output-dir', type=str,
                        default='results/analysis/batch1',
                        help='Directory for output files')
    parser.add_argument('--all', action='store_true',
                        help='Run all analyses')
    parser.add_argument('--initial-final-comparison', action='store_true',
                        help='Compare initial vs final LLM grading')
    parser.add_argument('--pairwise-agreement', action='store_true',
                        help='Calculate pairwise agreement rates')
    parser.add_argument('--inter-rater-reliability', action='store_true',
                        help='Calculate Krippendorff\'s alpha')
    parser.add_argument('--consensus-distribution', action='store_true',
                        help='Analyze consensus distribution across graders')
    parser.add_argument('--human-vs-ai', action='store_true',
                        help='Compare human baseline with AI performance')
    
    args = parser.parse_args()
    
    # If no specific analysis is selected, show help
    if not (args.all or args.initial_final_comparison or 
            args.pairwise_agreement or args.inter_rater_reliability or
            args.consensus_distribution or args.human_vs_ai):
        parser.print_help()
        return
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"✅ Loaded {len(df)} submissions\n")
    
    # Run selected analyses
    if args.all or args.initial_final_comparison:
        analyze_initial_final_changes(df, args.output_dir)
    
    if args.all or args.pairwise_agreement:
        calculate_pairwise_agreement(df, args.output_dir)
    
    if args.all or args.inter_rater_reliability:
        calculate_krippendorff_alpha(df, args.output_dir)
    
    if args.all or args.consensus_distribution:
        analyze_consensus_distribution(df, args.output_dir)
    
    if args.all or args.human_vs_ai:
        analyze_human_vs_ai_performance(df, args.output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

