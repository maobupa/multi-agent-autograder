"""
Reusable LLM Agents for various tasks.

This module provides LLM-based agents that can be reused across different scripts:
- PerturbationAgent: Generates surface-level code perturbations
- MessageClassifierAgent: Classifies print/output messages into semantic categories
- Other agents can be added here as needed
"""

import json
from typing import Dict, Any, Optional, Literal
from .openai_client import OpenAIClient


# Global cache for message classifications to avoid repeated LLM calls
_MESSAGE_CLASSIFICATION_CACHE: Dict[str, str] = {}

# Global cache for input prompt classifications
_INPUT_PROMPT_CLASSIFICATION_CACHE: Dict[str, str] = {}


class MessageClassifierAgent:
    """
    LLM Agent that classifies print/output messages into semantic categories.
    
    Categories:
    - "above": Messages indicating value is above/over/too high (e.g., "Above maximum height")
    - "below": Messages indicating value is below/under/too low (e.g., "Below minimum height")
    - "correct": Messages indicating value is correct/valid/acceptable (e.g., "Correct height")
    - "other": Messages that don't fit the above categories
    
    Uses caching to avoid repeated API calls for the same message.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the message classifier agent.
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini for speed/cost)
            temperature: Temperature for generation (0.0 for deterministic)
        """
        if self._initialized:
            return
        self.client = OpenAIClient().get_client()
        self.model = model
        self.temperature = temperature
        self._initialized = True
    
    def classify(self, message: str) -> Literal["above", "below", "correct", "other"]:
        """
        Classify a message into one of four semantic categories.
        
        Args:
            message: The print/output message to classify
            
        Returns:
            One of: "above", "below", "correct", "other"
        """
        # Normalize message for caching
        normalized = message.strip().lower()
        
        # Check cache first
        if normalized in _MESSAGE_CLASSIFICATION_CACHE:
            return _MESSAGE_CLASSIFICATION_CACHE[normalized]
        
        prompt = f"""Classify the following output message into exactly one category.

MESSAGE: "{message}"

Categories:
- "above": The message indicates something is ABOVE, OVER, TOO HIGH, EXCEEDS a limit, or is GREATER than expected
  Examples: "Above maximum height", "Too tall", "Exceeds limit", "Over the max"
  
- "below": The message indicates something is BELOW, UNDER, TOO LOW, or is LESS than expected
  Examples: "Below minimum height", "Too short", "Under the limit", "Less than min"
  
- "correct": The message indicates something is CORRECT, VALID, OK, ACCEPTABLE, GOOD, or PASSES a check
  Examples: "Correct height", "Valid input", "Height is acceptable", "Good to go"
  
- "other": The message doesn't fit any of the above categories

Respond with ONLY one word: above, below, correct, or other"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise message classifier. Respond with exactly one word: above, below, correct, or other."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Validate result
        if result not in ["above", "below", "correct", "other"]:
            # Try to extract from longer response
            for category in ["above", "below", "correct", "other"]:
                if category in result:
                    result = category
                    break
            else:
                result = "other"
        
        # Cache the result
        _MESSAGE_CLASSIFICATION_CACHE[normalized] = result
        
        return result
    
    def classify_batch(self, messages: list) -> Dict[str, str]:
        """
        Classify multiple messages efficiently.
        
        Args:
            messages: List of messages to classify
            
        Returns:
            Dictionary mapping message -> category
        """
        results = {}
        for msg in messages:
            results[msg] = self.classify(msg)
        return results
    
    @classmethod
    def clear_cache(cls):
        """Clear the classification cache."""
        global _MESSAGE_CLASSIFICATION_CACHE
        _MESSAGE_CLASSIFICATION_CACHE = {}
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(_MESSAGE_CLASSIFICATION_CACHE),
            "entries": dict(_MESSAGE_CLASSIFICATION_CACHE)
        }


class InputPromptClassifierAgent:
    """
    LLM Agent that classifies input prompts into semantic categories.
    
    Categories:
    - "height": Prompts asking for height input (e.g., "Enter your height:", "Input height in meters:")
    - "other": Prompts asking for something else
    
    Uses caching to avoid repeated API calls for the same prompt.
    Called once during CDG construction, not during comparison.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the input prompt classifier agent.
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini for speed/cost)
            temperature: Temperature for generation (0.0 for deterministic)
        """
        if self._initialized:
            return
        self.client = OpenAIClient().get_client()
        self.model = model
        self.temperature = temperature
        self._initialized = True
    
    def classify(self, prompt: str) -> Literal["height", "other"]:
        """
        Classify an input prompt into a semantic category.
        
        Args:
            prompt: The input prompt to classify (e.g., "Enter your height: ")
            
        Returns:
            One of: "height", "other"
        """
        # Normalize prompt for caching
        normalized = prompt.strip().lower()
        
        # Check cache first
        if normalized in _INPUT_PROMPT_CLASSIFICATION_CACHE:
            return _INPUT_PROMPT_CLASSIFICATION_CACHE[normalized]
        
        llm_prompt = f"""Classify the following input prompt into exactly one category.

INPUT PROMPT: "{prompt}"

Categories:
- "height": The prompt is asking the user to enter their HEIGHT (in any unit like meters, cm, feet, inches)
  Examples: "Enter your height:", "Enter your height in meters:", "Input height:", "Please provide your height", "What is your height?"
  
- "other": The prompt is asking for something OTHER than height (e.g., age, weight, name, etc.)
  Examples: "Enter your age:", "What is your name?", "Input weight:", "Enter a number:"

Respond with ONLY one word: height or other"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise prompt classifier. Respond with exactly one word: height or other."
                },
                {"role": "user", "content": llm_prompt}
            ],
            temperature=self.temperature,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Validate result
        if result not in ["height", "other"]:
            # Try to extract from longer response
            if "height" in result:
                result = "height"
            else:
                result = "other"
        
        # Cache the result
        _INPUT_PROMPT_CLASSIFICATION_CACHE[normalized] = result
        
        return result
    
    @classmethod
    def clear_cache(cls):
        """Clear the classification cache."""
        global _INPUT_PROMPT_CLASSIFICATION_CACHE
        _INPUT_PROMPT_CLASSIFICATION_CACHE = {}
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(_INPUT_PROMPT_CLASSIFICATION_CACHE),
            "entries": dict(_INPUT_PROMPT_CLASSIFICATION_CACHE)
        }


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

