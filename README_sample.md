# Sampling Guide

This guide explains how to use `sampling.py` to generate sample datasets for human grading comparison, with support for batch sampling to avoid overlapping data.

## Overview

The `sampling.py` script samples student submissions based on error distribution, ensuring a balanced dataset for grading. It supports:
- **Batch sampling**: Generate multiple batches without overlapping data
- **Error targeting**: Control the number of error cases in each batch
- **Organized output**: Files are organized in batch-specific folders

## Quick Start

### Basic Usage (Single Batch, No Tracking)

Generate a single 100-sample dataset with default parameters:

```bash
python sampling.py
```

This creates files in the `samples/` directory:
- `selected_100_diagnostic1.csv`
- `selected_100_diagnostic1_random.csv`
- `selected_100_diagnostic1_with_results.csv`
- `report_diagnostic1.txt`

### Batch Sampling (Recommended)

When generating multiple batches, use the tracking file to ensure no overlap:

**First Batch:**
```bash
python sampling.py \
  --used-ids-file samples/used_ids_diagnostic1.json \
  --batch-number 1
```

**Second Batch:**
```bash
python sampling.py \
  --used-ids-file samples/used_ids_diagnostic1.json \
  --batch-number 2
```

**Third Batch:**
```bash
python sampling.py \
  --used-ids-file samples/used_ids_diagnostic1.json \
  --batch-number 3
```

Each batch will be saved in its own folder:
- `samples/batch1/` - Contains all files for batch 1
- `samples/batch2/` - Contains all files for batch 2
- `samples/batch3/` - Contains all files for batch 3

## Output Files

Each batch generates 4 files:

1. **`selected_{N}_{diag_name}.csv`** - Combined sample in original order
2. **`selected_{N}_{diag_name}_random.csv`** - Randomized order for blind grading
3. **`selected_{N}_{diag_name}_with_results.csv`** - Includes grading results merged
4. **`report_{diag_name}.txt`** - Error distribution summary

## Customization Examples

### Different Diagnostic Exercise

```bash
python sampling.py \
  --diag-name diagnostic2 \
  --used-ids-file samples/used_ids_diagnostic2.json \
  --batch-number 1
```

### Custom Sample Size and Error Target

```bash
python sampling.py \
  --total-samples 150 \
  --error-target 100 \
  --used-ids-file samples/used_ids_diagnostic1.json \
  --batch-number 1
```

### Different Random Seed

```bash
python sampling.py \
  --random-seed 123 \
  --used-ids-file samples/used_ids_diagnostic1.json \
  --batch-number 1
```

### Custom Output Directory

```bash
python sampling.py \
  --output-dir my_custom_samples \
  --used-ids-file my_custom_samples/used_ids_diagnostic1.json \
  --batch-number 1
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--json-path` | string | `data/cip5/upload/cip5_diagnostic_feedback.json` | Path to grading JSON file |
| `--csv-path` | string | `data/cip5/processed/cip5_student_data.csv` | Path to student data CSV |
| `--output-dir` | string | `samples` | Directory for output files |
| `--diag-name` | string | `diagnostic1` | Diagnostic exercise name |
| `--total-samples` | int | `100` | Number of samples to generate |
| `--error-target` | int | `70` | Target number of error cases |
| `--random-seed` | int | `42` | Random seed for reproducibility |
| `--used-ids-file` | string | `None` | Tracking file for batch sampling (optional) |
| `--batch-number` | int | `None` | Batch number for folder organization (optional) |

## View All Options

To see all available options with descriptions:

```bash
python sampling.py --help
```

## File Organization

### Without Batch Number
```
samples/
├── selected_100_diagnostic1.csv
├── selected_100_diagnostic1_random.csv
├── selected_100_diagnostic1_with_results.csv
└── report_diagnostic1.txt
```

### With Batch Number
```
samples/
├── used_ids_diagnostic1.json  (tracking file)
├── batch1/
│   ├── selected_100_diagnostic1.csv
│   ├── selected_100_diagnostic1_random.csv
│   ├── selected_100_diagnostic1_with_results.csv
│   └── report_diagnostic1.txt
├── batch2/
│   ├── selected_100_diagnostic1.csv
│   ├── selected_100_diagnostic1_random.csv
│   ├── selected_100_diagnostic1_with_results.csv
│   └── report_diagnostic1.txt
└── batch3/
    └── ...
```

## Tips

1. **Always use the same tracking file** for the same diagnostic exercise to prevent overlap
2. **Batch numbers are optional** - use them for better organization
3. **The tracking file is created automatically** on the first run
4. **Each batch uses the same sampling method** but excludes previously used IDs
5. **Warning messages** will appear if there aren't enough available samples

## Notes

- The tracking file (`used_ids_diagnostic1.json`) stores all student IDs that have been sampled across all batches
- Each new batch automatically excludes IDs from previous batches
- If you run out of available samples, the script will use as many as available and print a warning
