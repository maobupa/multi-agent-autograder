# Adversarial Grading: An LLM-Based Framework for Robust Code Assessment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation and experiments for Consistency-Based Evaluation of LLM Grading Systems in Computer Science Educations. 

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Experiments](#experiments)
- [Citation](#citation)

## 🎯 Overview

Automated code grading faces challenges in consistency, especially when evaluating semantically equivalent but syntactically different programs. This project introduces:

1. **Multi-agent Grading System**: A two-stage grading framework where a Grader evaluates code and a Critic reviews the grading for fairness and accuracy.

2. **Cognitive Decision Graph (CDG)**: A novel representation that abstracts programs into decision trees, capturing semantic equivalence while ignoring surface-level differences.

3. **Comprehensive Evaluation**: Experiments demonstrating improved consistency compared to single-pass LLM grading and majority voting baselines.


## 📁 Repository Structure

```
adversarial_grader/
├── src/                           # Core library
│   ├── grader.py                  # Adversarial grading system
│   ├── llm_agents.py              # Reusable LLM agents
│   ├── openai_client.py           # OpenAI API client
│   ├── sampling.py                # Sampling utilities
│   └── utils.py                   # Helper functions
│
├── cdg/                           # Cognitive Decision Graph
│   ├── README.md                  # CDG documentation
│   ├── cdg.py                     # CDG implementation
│   ├── cdg_demo.ipynb             # Interactive demo
│   ├── cdg_clustering.ipynb       # Program clustering analysis
│   └── consistency_analysis.ipynb # Consistency evaluation
│
├── experiments/                   # Experimental scripts
│   ├── grading/                   # Grading experiments
│   │   ├── grading.py             # Main grading script
│   │   ├── lenient_grading.py     # Lenient message mode
│   │   ├── majority_baseline.py   # Baseline comparison
│   │   ├── prepare_analysis.py    # Data preparation
│   │   ├── grading_analysis.py    # Statistical analysis
│   │   └── grading_visualizations.ipynb
│   │
│   └── perturbation/              # Perturbation testing
│       ├── perturbation.py        # Perturbation generation & testing
│       └── perturbation_analysis.ipynb
│
├── scripts/                       # Entry point scripts
│   ├── run_diag1_100.py           # Run grading on dataset
│   └── test_grader.py             # Unit tests
│
├── data/                          # Data directory (Protected, Not uploaded here)
│   ├── rubrics/                   # Grading rubrics
│   └── graded/                    # Student submissions
│
└── results/                       # Output directory (Protected, Not uploaded here)
    └── analysis/                  # Analysis results
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/adversarial_grader.git
cd adversarial_grader
```

2. **Install the package**
```bash
pip install -e .
```

This installs the package in development mode, allowing you to import from `src` anywhere.

3. **Set up OpenAI API key**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

### Dependencies

The setup will automatically install:
- `openai` - OpenAI API client
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib`, `seaborn` - Visualizations
- `tqdm` - Progress bars
- `krippendorff` - Inter-rater reliability

## 🎮 Quick Start

### 1. Test the Grader

```bash
python scripts/test_grader.py
```

This runs the multi-agent grading system on a sample program and displays the grading results.

### 2. Grade a Dataset

```bash
python scripts/run_diag1_100.py
```

Grades 100 student submissions using the adversarial framework.

### 3. Explore CDG

Open and run `cdg/cdg_demo.ipynb` to see how CDG represents and clusters programs.

### 4. Analyze Results

```bash
python experiments/grading/grading_analysis.py \
    --input results/diag1_batch1.csv \
    --output results/analysis/
```

Generates comprehensive analysis of grading consistency and agreement.

## 🔧 Core Components

### Multi-agent Grading System

The main grading framework consists of three agents:

```python
from src.grader import AdversarialGradingSystem

# Initialize system
system = AdversarialGradingSystem(lenient_messages=False)

# Grade student code
result = system.adversarial_grade(
    code=student_code,
    rubric=rubric_dict
)

# Access results
print(result['final_grading'])  # Final grades after critique
print(result['critique'])        # Critic's feedback
print(result['changes'])         # Grade changes made
```

**Key Parameters:**
- `lenient_messages`: If True, accepts semantically similar print messages
- Returns both initial and final grades with detailed explanations

### Cognitive Decision Graph (CDG)

CDG abstracts programs into semantic decision trees:

```python
from cdg.cdg import build_cdg, programs_equivalent

# Build CDG for a program
cdg = build_cdg(code)

# Check semantic equivalence
is_equiv, reason = programs_equivalent(code1, code2)
```

**Features:**
- Abstracts away variable names, statement ordering
- Captures decision regions (e.g., `height < 1.6`)
- Semantic message classification (uses LLM)

See `cdg/README.md` for detailed documentation.

### LLM Agents

Reusable LLM-based components:

```python
from src.llm_agents import (
    MessageClassifierAgent,
    PerturbationAgent,
    CodeAnalysisAgent
)

# Classify output messages
classifier = MessageClassifierAgent()
category = classifier.classify("Above maximum height")  # Returns "above"

# Generate code perturbations
perturber = PerturbationAgent()
result = perturber.generate_perturbation(code)
```

## 📊 Experiments

### Grading Consistency Analysis

Compares adversarial grading against baselines:

```bash
# Run adversarial grading
python experiments/grading/grading.py

# Run majority vote baseline
python experiments/grading/majority_baseline.py

# Analyze results
python experiments/grading/grading_analysis.py --input results/
```

**Metrics Computed:**
- Inter-grader agreement (Cohen's Kappa, Krippendorff's Alpha)
- Grade change statistics
- Consensus distribution

**Visualization:**
- Agreement matrices
- Grade distribution plots
- Change heatmaps

See `experiments/grading/grading_visualizations.ipynb` for interactive analysis.

### Perturbation Testing

Tests grading robustness under code transformations:

```bash
python experiments/perturbation/perturbation.py \
    --input data/graded/diag1_100_random_batch1.csv \
    --output results/perturbation_results.csv
```

**Perturbation Types:**
- Variable renaming
- Comment addition/removal
- Whitespace changes
- Statement reordering (semantically equivalent)

**Analysis:**
- Grade consistency between original and perturbed
- Identification of sensitive grading criteria
- Robustness metrics

See `experiments/perturbation/perturbation_analysis.ipynb` for results.

### CDG Clustering

Groups semantically equivalent programs:

```bash
jupyter notebook cdg/cdg_clustering.ipynb
```

**Output:**
- Program clusters (semantically equivalent groups)
- Cluster size distribution
- Representative programs for each cluster


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 🙏 Acknowledgments

- Thanks to the students whose anonymized code submissions were used in this research
- Thanks to the graders who contributed
- Built using OpenAI's GPT models
- Inspired by research in automated program grading and semantic program analysis

---

**Note**: This repository accompanies our conference paper submission. 
