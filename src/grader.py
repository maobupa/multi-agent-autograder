"""
Adversarial Grading System

This module implements a two-stage grading system where:
1. First grader grades code based on rubric
2. Critic reviews the grading for fairness, edge cases, etc.
3. First grader responds and potentially revises
"""

import json
from .openai_client import OpenAIClient
from .utils import format_rubric
from typing import Dict, List, Any, Tuple


class Grader:
    """
    Initial grader that evaluates student code against a rubric.
    
    Args:
        lenient_messages: If True, be lenient about input/output/print message wording.
            As long as the semantic meaning is similar, consider them correct.
            Default is False (strict mode - exact matching per rubric).
    """
    
    def __init__(self, lenient_messages: bool = False):
        self.client = OpenAIClient().get_client()
        self.lenient_messages = lenient_messages
        
        # Additional prompt for lenient message grading
        self._lenient_prompt = """
IMPORTANT - MESSAGE LENIENCY:
Be lenient about input prompts, output messages, and print statements. 
As long as the semantic meaning is similar, consider them correct. For example:
- "Enter height in meters:" vs "Please input your height:" → Both acceptable
- "Above maximum astronaut height" vs "Too tall for astronaut requirements" → Both acceptable  
- "You qualify!" vs "Congratulations, you meet the requirements!" → Both acceptable
Do NOT penalize for minor wording differences, capitalization, punctuation, or phrasing 
variations in messages. Focus on whether the MESSAGE INTENT is correct, not exact wording.
"""
    
    def grade(self, code: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grade student code against a rubric.
        
        Args:
            code: Student's code as a string
            rubric: Rubric dictionary with 'description' and 'items'
            
        Returns:
            Dictionary with 'scores' and 'feedback'. Example:
            {
                'scores': {
                    'input': {
                        'option': 1,
                        'explanation': 'Does not convert input to float.'
                    },
                    'conditional_logic': {
                        'option': 0,
                        'explanation': 'Correct conditional logic'
                    }
                },
                'feedback': 'Overall feedback string...'
            }
        """
        rubric_text = format_rubric(rubric)
        
        # Add lenient message instruction if enabled
        lenient_section = self._lenient_prompt if self.lenient_messages else ""
        
        prompt = f"""You are a grading assistant. Grade the following student code against the provided rubric.
{lenient_section}
RUBRIC:
{rubric_text}

STUDENT CODE:
```python
{code}
```

INSTRUCTIONS:
1. Grade each rubric item by selecting the appropriate option (0 for best, higher numbers for worse)
2. Provide a brief explanation for each grade
3. Give overall feedback summarizing strengths and weaknesses
4. Cite specific lines of code in your explanations using line numbers (e.g., "Line 5: ...")

Respond ONLY with a valid JSON object in this exact format:
{{
    "scores": {{
        "rubric_item_id": {{
            "option": <0, 1, 2, or 3>,
            "explanation": "<your explanation>"
        }}
    }},
    "feedback": "<overall feedback>"
}}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a precise and fair grading assistant. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    def respond_to_critique(
        self,
        code: str,
        rubric: Dict[str, Any],
        initial_grading: Dict[str, Any],
        critique: str
    ) -> Dict[str, Any]:
        """
        Respond to critique and potentially revise grading.
        
        Args:
            code: Student's code
            rubric: Rubric dictionary
            initial_grading: Original grading results
            critique: Critique from the critic
            
        Returns:
            Dictionary with revised scores, feedback, and response
        """
        rubric_text = format_rubric(rubric)
        scores_text = "\n".join([
            f"  {item_id}: Option {score['option']} - {score['explanation']}"
            for item_id, score in initial_grading.get('scores', {}).items()
        ])
        
        # Add lenient message instruction if enabled
        lenient_section = self._lenient_prompt if self.lenient_messages else ""
        
        prompt = f"""You previously graded this code. A critic has reviewed your grading. 
Consider their critique carefully and decide whether to revise your grades.
{lenient_section}
RUBRIC:
{rubric_text}

STUDENT CODE:
```python
{code}
```

YOUR INITIAL GRADING:
{scores_text}

Initial Feedback: {initial_grading.get('feedback', 'N/A')}

CRITIQUE FROM REVIEWER:
{critique}

REVIEW YOUR GRADING:
1. Do you agree with the critique? Why or why not?
2. Should you revise any grades? Be fair and consistent.
3. If you revise, explain what changed and why.

Respond with valid JSON in this exact format:
{{
    "scores": {{
        "rubric_item_id": {{
            "option": <0, 1, 2, or 3>,
            "explanation": "<your explanation>"
        }}
    }},
    "feedback": "<revised overall feedback>",
    "response": "<your response to the critique, explaining your reasoning>"
}}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a reflective grader who carefully considers criticism and adjusts fairly when warranted. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result


class Critic:
    """
    Critic that reviews the initial grading for fairness and accuracy.
    
    Args:
        lenient_messages: If True, critique should account for leniency about 
            input/output/print message wording. Default is False (strict mode).
    """
    
    def __init__(self, lenient_messages: bool = False):
        self.client = OpenAIClient().get_client()
        self.lenient_messages = lenient_messages
        
        # Additional prompt for lenient message grading
        self._lenient_prompt = """
IMPORTANT - MESSAGE LENIENCY POLICY:
When critiquing grades related to input prompts, output messages, and print statements,
remember that we are being LENIENT about message wording. As long as the semantic meaning 
is correct, minor wording differences should NOT be penalized. For example:
- Different phrasing with same intent → Acceptable
- Minor spelling/capitalization differences → Acceptable  
- Same message category (e.g., "too tall" messages) → Acceptable
Do NOT critique the grader for being lenient on message wording. Focus on LOGIC and 
FUNCTIONALITY rather than exact message text.
"""
    
    def critique(
        self,
        code: str,
        rubric: Dict[str, Any],
        initial_grading: Dict[str, Any],
        grader_explanation: str = None
    ) -> str:
        """
        Critique the initial grading.
        
        Args:
            code: Student's code as a string
            rubric: Rubric dictionary
            initial_grading: Initial grading results from Grader
            grader_explanation: Optional explanation from grader
            
        Returns:
            String critique focusing on edge cases, fairness, etc.
        """
        rubric_text = format_rubric(rubric)
        scores_text = "\n".join([
            f"  {item_id}: Option {score['option']} - {score['explanation']}"
            for item_id, score in initial_grading.get('scores', {}).items()
        ])
        
        # Add lenient message instruction if enabled
        lenient_section = self._lenient_prompt if self.lenient_messages else ""
        
        prompt = f"""You are a critical reviewer of grading work. Review the following grading decisions carefully.
{lenient_section}
RUBRIC:
{rubric_text}

STUDENT CODE:
```python
{code}
```

INITIAL GRADING:
{scores_text}

Initial Overall Feedback: {initial_grading.get('feedback', 'N/A')}

CRITIQUE THE GRADING BASED ON:
1. **Edge cases**: Does the code handle boundary conditions correctly?
2. **Ambiguity**: Are there ambiguities in the rubric that affect grading?
3. **Syntax/Style/Comments**: Are there additional issues not captured?
4. **Partial credit reasoning**: Is the partial credit fair and consistent?
5. **Fairness**: Is the grading too harsh or too lenient?
6. **Code citations**: Are specific lines cited appropriately?

Provide a detailed critique. Be specific - cite exact line numbers and rubric items.
Focus on where the grader may have been:
- Too harsh (missed partial credit opportunities)
- Too lenient (overlooked important errors)
- Inconsistent (different standards across items)
- Missing important considerations

Write in a constructive but thorough manner."""
        
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a thorough and objective critic of grading quality. Focus on fairness and accuracy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        return response.choices[0].message.content


class AdversarialGradingSystem:
    """
    Orchestrates the adversarial grading process:
    1. Initial grading
    2. Critique
    3. Response and potential revision
    
    Args:
        lenient_messages: If True, both grader and critic will be lenient about 
            input/output/print message wording. As long as the semantic meaning 
            is similar, consider them correct. Default is False (strict mode).
    """
    
    def __init__(self, lenient_messages: bool = False):
        self.lenient_messages = lenient_messages
        self.grader = Grader(lenient_messages=lenient_messages)
        self.critic = Critic(lenient_messages=lenient_messages)
    
    def adversarial_grade(
        self,
        code: str,
        rubric: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run full adversarial grading process.
        
        Args:
            code: Student's code as a string
            rubric: Rubric dictionary
            
        Returns:
            Dictionary with full adversarial grading results:
            - initial_grade: Dict of rubric_id -> {option, explanation}
            - final_grade: Dict of rubric_id -> {option, explanation}
            - initial_feedback: String
            - final_feedback: String
            - debate_log: List of {role, content} dicts
            - change: Dict with 'items_changed' list and 'overall_delta' number
        """
        # Step 1: Initial grading
        initial_result = self.grader.grade(code, rubric)
        initial_scores = initial_result.get('scores', {})
        
        # Step 2: Critique
        critique = self.critic.critique(
            code=code,
            rubric=rubric,
            initial_grading=initial_result
        )
        
        # Step 3: Grader responds and potentially revises
        final_result = self.grader.respond_to_critique(
            code=code,
            rubric=rubric,
            initial_grading=initial_result,
            critique=critique
        )
        final_scores = final_result.get('scores', {})
        
        # Calculate changes
        changes = self._calculate_changes(initial_scores, final_scores)
        
        return {
            'initial_grade': initial_scores,
            'final_grade': final_scores,
            'initial_feedback': initial_result.get('feedback', ''),
            'final_feedback': final_result.get('feedback', ''),
            'debate_log': [
                {'role': 'critic', 'content': critique},
                {'role': 'grader', 'content': final_result.get('response', '')}
            ],
            'change': changes
        }
    
    def _calculate_changes(
        self,
        initial_scores: Dict[str, Dict],
        final_scores: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Calculate what changed between initial and final grades."""
        changes = {
            'items_changed': [],
            'overall_delta': 0
        }
        
        all_items = set(initial_scores.keys()) | set(final_scores.keys())
        
        for item_id in all_items:
            initial = initial_scores.get(item_id, {}).get('option', None)
            final = final_scores.get(item_id, {}).get('option', None)
            
            if initial != final:
                changes['items_changed'].append({
                    'item_id': item_id,
                    'initial': initial,
                    'final': final,
                    'delta': (final or 0) - (initial or 0)
                })
                
                # Calculate overall delta (negative is better, positive is worse)
                changes['overall_delta'] += (final or 0) - (initial or 0)
        
        return changes
