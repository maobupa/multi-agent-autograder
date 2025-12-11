"""
This script compiles grading results from multiple graders (LLM and human) into a single CSV file.
Each grading contains 4 categories: input, conditional_logic, printing, syntax_errors.
We extract only the numeric 'option' value for each category.
"""

import pandas as pd
import json
import os


def normalize_quotes(text):
    """Convert curly quotes to straight quotes for JSON parsing."""
    if not isinstance(text, str):
        return text
    mapping = {
        chr(8220): '"',  # " (left double quotation mark)
        chr(8221): '"',  # " (right double quotation mark)
        chr(8216): "'",  # ' (left single quotation mark)
        chr(8217): "'"   # ' (right single quotation mark)
    }
    for a, b in mapping.items():
        text = text.replace(a, b)
    return text


def parse_json_grade(cell):
    """Safely parse JSON grade content with robust error handling."""
    if pd.isna(cell) or cell == "":
        return None
    try:
        # Normalize quotes before parsing
        normalized = normalize_quotes(cell)
        
        # Fix common JSON formatting issues
        import re
        # Fix missing quotes around values like: "explanation": correct""" -> "explanation": "correct"
        normalized = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_\s]*)("""|")', r': "\1"', normalized)
        # Fix triple quotes
        normalized = normalized.replace('"""', '"')
        # Fix doubled quotes
        normalized = re.sub(r'""([^"]*?)""', r'"\1"', normalized)
        
        return json.loads(normalized)
    except Exception as e:
        # Try additional fixes for malformed JSON
        try:
            import re
            # More aggressive cleanup
            cleaned = normalize_quotes(str(cell))
            # Remove any control characters
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
            return json.loads(cleaned)
        except:
            print(f"Warning: Could not parse JSON: {str(cell)[:100]}...")
            return None


def extract_grade_options(grade_dict):
    """
    Extract numeric option values from grade dictionary.
    Returns dict with keys: input, logic, syntax, print
    """
    if not isinstance(grade_dict, dict):
        return {"input": None, "logic": None, "syntax": None, "print": None}
    
    return {
        "input": grade_dict.get("input", {}).get("option") if isinstance(grade_dict.get("input"), dict) else None,
        "logic": grade_dict.get("conditional_logic", {}).get("option") if isinstance(grade_dict.get("conditional_logic"), dict) else None,
        "syntax": grade_dict.get("syntax_errors", {}).get("option") if isinstance(grade_dict.get("syntax_errors"), dict) else None,
        "print": grade_dict.get("printing", {}).get("option") if isinstance(grade_dict.get("printing"), dict) else None,
    }


def load_and_process_human_grader(file_path, grader_name):
    """
    Load human grader CSV and extract grading columns.
    Returns DataFrame with student_id, code, and 4 grading columns.
    """
    df = pd.read_csv(file_path)
    
    # Parse the grade column
    df["parsed_grade"] = df["grade (fill this out)"].apply(parse_json_grade)
    
    # Extract options for each category
    grade_options = df["parsed_grade"].apply(extract_grade_options)
    
    # Create result dataframe
    result = pd.DataFrame({
        "student_id": df["student_id"],
        "code": df["code"],
        f"{grader_name}_input": grade_options.apply(lambda x: x["input"]),
        f"{grader_name}_logic": grade_options.apply(lambda x: x["logic"]),
        f"{grader_name}_syntax": grade_options.apply(lambda x: x["syntax"]),
        f"{grader_name}_print": grade_options.apply(lambda x: x["print"]),
    })
    
    return result


def load_and_process_llm_grading(file_path):
    """
    Load LLM grading CSV and extract initial and final grading columns.
    Returns DataFrame with student_id and 8 grading columns (4 for initial, 4 for final).
    Note: code column is not in LLM grading file, will be added from human grading files.
    """
    df = pd.read_csv(file_path)
    
    # Parse initial and final grades
    df["parsed_initial"] = df["initial_grade"].apply(parse_json_grade)
    df["parsed_final"] = df["final_grade"].apply(parse_json_grade)
    
    # Extract options for each category
    initial_options = df["parsed_initial"].apply(extract_grade_options)
    final_options = df["parsed_final"].apply(extract_grade_options)
    
    # Create result dataframe (no code column)
    result = pd.DataFrame({
        "student_id": df["student_id"],
        "initial_input": initial_options.apply(lambda x: x["input"]),
        "initial_logic": initial_options.apply(lambda x: x["logic"]),
        "initial_syntax": initial_options.apply(lambda x: x["syntax"]),
        "initial_print": initial_options.apply(lambda x: x["print"]),
        "final_input": final_options.apply(lambda x: x["input"]),
        "final_logic": final_options.apply(lambda x: x["logic"]),
        "final_syntax": final_options.apply(lambda x: x["syntax"]),
        "final_print": final_options.apply(lambda x: x["print"]),
    })
    
    return result


def main():
    # === CONFIGURATION ===
    LLM_GRADING_PATH = "results/diag1_batch1.csv"
    HUMAN_GRADERS = {
        "briana": "data/graded/batch1/briana.csv",
        "chen": "data/graded/batch1/chen.csv",
        "grace": "data/graded/batch1/grace.csv",
        "sanya": "data/graded/batch1/sanya_clean.csv",
        "shu": "data/graded/batch1/shu.csv",
    }
    OUTPUT_PATH = "results/analysis/batch1_compiled_grading.csv"
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    print("Loading LLM grading results...")
    llm_df = load_and_process_llm_grading(LLM_GRADING_PATH)
    print(f"  Loaded {len(llm_df)} LLM grading records")
    
    # Start with LLM grading as base (no code column yet)
    combined_df = llm_df
    
    # Load and merge each human grader
    first_grader = True
    for grader_name, file_path in HUMAN_GRADERS.items():
        print(f"Loading {grader_name}'s grading...")
        human_df = load_and_process_human_grader(file_path, grader_name)
        print(f"  Loaded {len(human_df)} records from {grader_name}")
        
        if first_grader:
            # Keep code column from first human grader
            combined_df = combined_df.merge(
                human_df,
                on="student_id",
                how="outer"
            )
            first_grader = False
        else:
            # Drop code column from subsequent graders
            combined_df = combined_df.merge(
                human_df.drop(columns=["code"]),
                on="student_id",
                how="outer"
            )
    
    # Reorder columns: student_id, code, then all grading columns
    grading_cols = [col for col in combined_df.columns if col not in ["student_id", "code"]]
    combined_df = combined_df[["student_id", "code"] + grading_cols]
    
    # Save to CSV
    combined_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Compiled grading data saved to {OUTPUT_PATH}")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Total columns: {len(combined_df.columns)}")
    print(f"\nColumn names:")
    for col in combined_df.columns:
        print(f"  - {col}")
    
    return combined_df


if __name__ == "__main__":
    main()

