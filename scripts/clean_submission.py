"""
This script is used to clean the human grading submission data.
"""
import csv
import json
import re

def normalize_quotes(text):
    """Convert curly quotes to straight quotes."""
    mapping = {
        chr(8220): '"',  # " (left double quotation mark)
        chr(8221): '"',  # " (right double quotation mark)
        chr(8216): "'",  # ' (left single quotation mark)
        chr(8217): "'"   # ' (right single quotation mark)
    }
    for a, b in mapping.items():
        text = text.replace(a, b)
    return text

def remove_extra_blank_lines(text):
    """Remove multiple consecutive blank lines."""
    return re.sub(r"\n\s*\n+", "\n", text).strip()

def clean_json(text):
    """Normalize, fix common issues, and validate JSON."""
    if text is None:
        return ""

    cleaned = normalize_quotes(text)
    cleaned = remove_extra_blank_lines(cleaned)

    try:
        obj = json.loads(cleaned)
        return json.dumps(obj, indent=4)
    except Exception:
        print("⚠ Warning: JSON still invalid in this row.")
        return cleaned  # Return best-effort version

def is_empty_row(row):
    """Return True if ALL fields in the row are blank or whitespace."""
    return all(not (value and value.strip()) for value in row.values())

def clean_csv(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # Default delimiter is comma
        cleaned_rows = []

        for row in reader:
            if is_empty_row(row):
                # Skip empty row completely
                continue

            # Clean student_id
            row["student_id"] = row["student_id"].strip()

            # Clean code column
            row["code"] = remove_extra_blank_lines(row["code"])

            # Clean grade column
            grade_col = "grade (fill this out)"
            row[grade_col] = clean_json(row[grade_col])

            cleaned_rows.append(row)

    # Write cleaned result
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cleaned_rows[0].keys())  # Default delimiter is comma
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"✨ Cleaning complete. Saved as {output_path}.")

if __name__ == "__main__":
    clean_csv("data/graded/batch1/grace.csv", "data/graded/batch1/grace_clean.csv")
