# Adversarial Grading System

An LLM-based automated grading system that uses adversarial debate to improve grading accuracy and consistency. This system implements a two-stage grading process where an initial grader evaluates student code, a critic reviews the grading for fairness and edge cases, and the grader responds and potentially revises their assessment.

## 🎯 Overview

This project explores the use of Large Language Models (LLMs) for automated code grading with a focus on:
- **Adversarial Grading**: A multi-agent approach where a critic challenges the initial grading to improve accuracy
- **Human-AI Comparison**: Comprehensive analysis comparing LLM grades with multiple human graders
- **Consistency Testing**: Perturbation tests to evaluate grading stability under surface-level code changes

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AdversarialGradingSystem                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Grader    │───▶│   Critic    │───▶│ Grader (Revise) │  │
│  │  (Initial)  │    │  (Review)   │    │    (Final)      │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│        │                  │                    │            │
│        ▼                  ▼                    ▼            │
│   Initial Grade      Critique           Final Grade         │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
adversarial_grader/
├── grader.py              # Core adversarial grading system (Grader, Critic, AdversarialGradingSystem)
├── grading.py             # Main script to run grading on submissions
├── openai_client.py       # Shared OpenAI client utility (singleton pattern)
├── utils.py               # Utility functions (rubric formatting)
├── llm_agents.py          # Reusable LLM agents (PerturbationAgent, SummarizationAgent, CodeAnalysisAgent)
├── perturbation.py        # Perturbation test for grading consistency
├── grading_analysis.py    # Statistical analysis of grading results
├── prepare_analysis.py    # Data compilation for analysis
├── sampling.py            # Sampling utilities for creating grading datasets
├── grading_visualizations.ipynb    # Visualization notebook for grading analysis
├── perturbation_analysis.ipynb     # Analysis of perturbation test results
├── data/                  # (not tracked - see Data Privacy section)
│   └── rubrics/           # Grading rubrics (JSON format)
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adversarial_grader.git
cd adversarial_grader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Usage

#### Basic Grading

```python
from grader import AdversarialGradingSystem
import json

# Load rubric
with open("data/rubrics/raw/diagnostic1/rubric.json", "r") as f:
    rubric = json.load(f)

# Initialize system
system = AdversarialGradingSystem()

# Grade a submission
code = '''
def main():
    height = float(input("Enter your height in meters: "))
    if height > 1.6 and height < 1.9:
        print("Correct height to be an astronaut")
    elif height <= 1.6:
        print("Below minimum astronaut height")
    else:
        print("Above maximum astronaut height")

if __name__ == "__main__":
    main()
'''

result = system.adversarial_grade(code, rubric)

print("Initial Grade:", result['initial_grade'])
print("Final Grade:", result['final_grade'])
print("Changes:", result['change'])
```

#### Running Batch Grading

```bash
python grading.py
```

#### Running Analysis

```bash
# Run all analyses
python grading_analysis.py --all

# Run specific analysis
python grading_analysis.py --inter-rater-reliability
python grading_analysis.py --pairwise-agreement
python grading_analysis.py --human-vs-ai
```

#### Running Perturbation Test

```bash
python perturbation.py --num-samples 100
```

## 📊 Analysis Features

### Inter-Rater Reliability
- **Weighted Cohen's Kappa** for pairwise agreement
- **Krippendorff's Alpha** for multi-rater reliability
- Human-Human, Human-LLM, and LLM-LLM comparisons

### Consensus Analysis
- Unanimous agreement detection
- Consensus distribution (strong, weak, split)
- Majority voting comparisons

### Perturbation Testing
- Surface-level code modifications (variable renaming, comments)
- Self-consistency Kappa measurement
- Exact agreement rate calculation

## 🔧 Components

### AdversarialGradingSystem (`grader.py`)
The core grading system with three stages:
1. **Grader**: Initial evaluation against rubric
2. **Critic**: Reviews grading for fairness, edge cases, and consistency
3. **Grader (Response)**: Considers critique and potentially revises grades

### PerturbationAgent (`llm_agents.py`)
Generates surface-level code perturbations for consistency testing:
- Variable renaming
- Comment addition/modification
- Whitespace changes
- Statement reordering (where semantically valid)

### Analysis Scripts
- `grading_analysis.py`: Comprehensive statistical analysis
- `prepare_analysis.py`: Data compilation from multiple graders
- `grading_visualizations.ipynb`: Interactive visualizations

## 📈 Rubric Format

Rubrics are defined in JSON format:

```json
{
    "description": "Problem description...",
    "items": [
        {
            "id": "input",
            "label": "Getting input from user",
            "options": [
                {"optionId": 0, "label": "Correct implementation"},
                {"optionId": 1, "label": "Minor error"},
                {"optionId": 2, "label": "Major error"}
            ]
        }
    ]
}
```

## ⚠️ Data Privacy

This project was developed for research purposes with IRB-protected student data. The following folders are excluded from version control:
- `data/` - All data including rubrics, student submissions, and grader annotations
- `samples/` - Sampled student submissions
- `results/` - Grading results with student identifiers

To use this system, you'll need to provide your own:
1. Student submissions (CSV with `student_id` and `code` columns)
2. Rubrics in `data/rubrics/raw/<diagnostic_name>/rubric.json` (JSON format as shown above)
3. Human grader annotations (optional, for comparison studies)

## 📝 License

This project is for research purposes. Please contact the authors for licensing information.


## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{adversarial_grader,
  author = {Huijun},
  title = {Multi-agent Grading System: LLM-based Automated Code Grading with Critic Review},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/maobupa/multi-agent-autograder}
}
```

## 📧 Contact

For questions or collaboration inquiries, please open an issue or contact huijunm@stanford.edu.

