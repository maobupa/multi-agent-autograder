"""
Simple test script for the adversarial grading system.
"""

import json
from src.grader import AdversarialGradingSystem


def test_simple_case():
    """Test the adversarial grading system with diagnostic1."""
    
    # Load the rubric
    with open('data/rubrics/raw/diagnostic1/rubric.json', 'r') as f:
        rubric = json.load(f)
    
    # Example code from diagnostic1 (the one that doesn't convert to float)
    code = """def main():
    # TODO write your solution here
    height = input("Enter your height in meters: ")
    if height < 1.6:
        print("Below minimum astronaut height")
    elif height > 1.9:
        print("Above maximum astronaut height")
    else:
        print("Correct height to be an astronaut")

if __name__ == "__main__":
    main()"""
    
    print("=" * 80)
    print("TESTING ADVERSARIAL GRADING SYSTEM")
    print("=" * 80)
    print("\nRunning adversarial grading process...")
    print("This will:")
    print("1. Grade the code initially")
    print("2. Have a critic review the grading")
    print("3. Have the grader respond and potentially revise")
    print("\n" + "-" * 80 + "\n")
    
    # Create the system and run adversarial grading
    system = AdversarialGradingSystem()
    result = system.adversarial_grade(code, rubric)
    
    # Display results
    print("RESULTS:")
    print("=" * 80)
    
    print("\n1. INITIAL GRADE:")
    for item_id, score_data in result['initial_grade'].items():
        print(f"   {item_id}: Option {score_data['option']} - {score_data['explanation']}")
    
    print("\n2. FINAL GRADE:")
    for item_id, score_data in result['final_grade'].items():
        print(f"   {item_id}: Option {score_data['option']} - {score_data['explanation']}")
    
    print("\n3. CHANGES:")
    if result['change']['items_changed']:
        for change in result['change']['items_changed']:
            print(f"   {change['item_id']}: {change['initial']} → {change['final']} (Δ: {change['delta']})")
        print(f"   Overall delta: {result['change']['overall_delta']}")
    else:
        print("   No changes made")
    
    print("\n4. DEBATE LOG:")
    print("\n   CRITIC:")
    print(f"   {result['debate_log'][0]['content']}\n")
    print("\n   GRADER RESPONSE:")
    print(f"   {result['debate_log'][1]['content']}\n")
    
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    test_simple_case()

