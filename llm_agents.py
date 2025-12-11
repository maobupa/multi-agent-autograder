"""
Reusable LLM Agents for various tasks.

This module provides LLM-based agents that can be reused across different scripts:
- PerturbationAgent: Generates surface-level code perturbations
- Other agents can be added here as needed
"""

import json
from typing import Dict, Any, Optional
from openai_client import OpenAIClient


class PerturbationAgent:
    """
    LLM Agent that generates surface-level code perturbations.
    
    Surface-level changes include:
    - Renaming variables (e.g., x -> height_value)
    - Adding/modifying comments
    - Changing whitespace/formatting
    - Reordering independent statements
    
    These changes should NOT affect the logical behavior of the code.
    """
    
    def __init__(self, model: str = "gpt-4.1", temperature: float = 0.7):
        """
        Initialize the perturbation agent.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for generation (higher = more creative)
        """
        self.client = OpenAIClient().get_client()
        self.model = model
        self.temperature = temperature
    
    def generate_perturbation(
        self,
        code: str,
        perturbation_types: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate a surface-level perturbation of the given code.
        
        Args:
            code: Original Python code as a string
            perturbation_types: List of perturbation types to apply.
                              Options: ["rename_variables", "add_comments", 
                                       "change_whitespace", "reorder_statements"]
                              If None, randomly selects perturbation types.
        
        Returns:
            Dictionary with:
            - 'perturbed_code': The modified code
            - 'changes_made': List of changes applied
            - 'perturbation_type': Type of perturbation used
        """
        if perturbation_types is None:
            perturbation_types = ["rename_variables", "add_comments"]
        
        types_str = ", ".join(perturbation_types)
        
        prompt = f"""You are a code transformation expert. Your task is to make SURFACE-LEVEL changes to Python code without changing its logical behavior or output.

ORIGINAL CODE:
```python
{code}
```

INSTRUCTIONS:
1. Apply the following types of perturbations: {types_str}
2. The perturbed code MUST produce the EXACT SAME OUTPUT as the original for all inputs
3. Do NOT change:
   - The algorithm or logic
   - The conditional flow
   - The mathematical operations
   - Input/output behavior
4. You MAY change:
   - Variable names (e.g., 'x' -> 'user_height', 'h' -> 'height_value')
   - Add meaningful comments explaining the code
   - Change formatting/indentation style (but keep it valid Python)
   - Reorder independent statements if they don't affect execution

Respond with a JSON object in this exact format:
{{
    "perturbed_code": "<the modified code as a string>",
    "changes_made": ["<description of change 1>", "<description of change 2>", ...],
    "perturbation_type": "{types_str}"
}}

IMPORTANT: 
- The perturbed_code must be valid Python that executes correctly
- Escape special characters properly in the JSON string
- Use \\n for newlines in the code string"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise code transformation assistant. You make surface-level changes to code without altering its behavior. Always respond with valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    def batch_generate_perturbations(
        self,
        codes: list,
        perturbation_types: Optional[list] = None,
        verbose: bool = True
    ) -> list:
        """
        Generate perturbations for a batch of code samples.
        
        Args:
            codes: List of code strings
            perturbation_types: Types of perturbations to apply
            verbose: Whether to print progress
            
        Returns:
            List of perturbation results
        """
        results = []
        total = len(codes)
        
        for i, code in enumerate(codes):
            if verbose:
                print(f"Generating perturbation {i+1}/{total}...", end="\r")
            
            try:
                result = self.generate_perturbation(code, perturbation_types)
                results.append(result)
            except Exception as e:
                print(f"\nError on sample {i+1}: {e}")
                results.append({
                    'perturbed_code': code,  # Return original on error
                    'changes_made': [f"Error: {str(e)}"],
                    'perturbation_type': 'error'
                })
        
        if verbose:
            print(f"\nCompleted {total} perturbations.")
        
        return results


class SummarizationAgent:
    """
    LLM Agent that summarizes text or code.
    Can be extended for various summarization tasks.
    """
    
    def __init__(self, model: str = "gpt-4.1", temperature: float = 0.3):
        self.client = OpenAIClient().get_client()
        self.model = model
        self.temperature = temperature
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            
        Returns:
            Summary string
        """
        prompt = f"""Summarize the following text in {max_length} words or less:

{text}

Provide a concise summary that captures the main points."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a concise summarization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        
        return response.choices[0].message.content


class CodeAnalysisAgent:
    """
    LLM Agent for analyzing code characteristics.
    Useful for understanding code structure, complexity, etc.
    """
    
    def __init__(self, model: str = "gpt-4.1", temperature: float = 0.3):
        self.client = OpenAIClient().get_client()
        self.model = model
        self.temperature = temperature
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze code characteristics.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with analysis results
        """
        prompt = f"""Analyze the following Python code and provide structured information:

```python
{code}
```

Respond with a JSON object containing:
{{
    "complexity": "low/medium/high",
    "num_functions": <number>,
    "num_variables": <number>,
    "has_loops": true/false,
    "has_conditionals": true/false,
    "has_error_handling": true/false,
    "main_purpose": "<brief description>",
    "potential_issues": ["<issue 1>", "<issue 2>", ...]
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a code analysis expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)


# Example usage
if __name__ == "__main__":
    # Test the perturbation agent
    test_code = '''def main():
    x = float(input("Enter your height in meters: "))
    if x > 1.6 and x < 1.9:
        print("Correct height to be an astronaut")
    elif x <= 1.6:
        print("Below minimum astronaut height")
    else:
        print("Above maximum astronaut height")

if __name__ == "__main__":
    main()'''
    
    print("Testing PerturbationAgent...")
    agent = PerturbationAgent()
    result = agent.generate_perturbation(test_code)
    
    print("\nOriginal code:")
    print(test_code)
    print("\n" + "="*50)
    print("\nPerturbed code:")
    print(result['perturbed_code'])
    print("\nChanges made:")
    for change in result['changes_made']:
        print(f"  - {change}")

