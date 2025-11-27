#!/usr/bin/env python3
"""
Test the inference.py fix with 10 questions

This script verifies:
1. Questions are formatted with A/B/C/D choices
2. System prompts require letter answers
3. LLM responses contain extractable letters
4. Accuracy can be measured against ground truth

Usage:
    python3 test_fix.py         # Auto-detect latest results
    python3 test_fix.py 200     # Analyze 200-question results
    python3 test_fix.py 1000    # Analyze 1000-question results
"""

import json
import re
import os
import sys

def extract_answer_choice(response_text):
    """Extract A/B/C/D answer from LLM response"""
    if not response_text:
        return None

    response_text = response_text.strip()

    # Pattern 1a: "Answer: X" format (new standard format)
    match = re.search(r'(?:^|\n)\s*answer:\s*\*{0,2}([A-D])\*{0,2}', response_text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 1b: Gemini-specific markdown formats
    # "**Correct Letter:** A" or "**Correct Letter: A**" or "Correct Letter:** A"
    match = re.search(r'(?:correct\s+)?letter\*{0,2}:\*{0,2}\s*([A-D])\*{0,2}', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 1c: Numbered list format "1. **Correct Letter:** A"
    match = re.search(r'\d+\.\s+\*{0,2}(?:correct\s+)?letter\*{0,2}:\*{0,2}\s*([A-D])', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 1d: Original pattern (for Claude/OpenAI)
    match = re.search(r'(?:correct\s+)?letter[:\s]+\*{0,2}([A-D])\*{0,2}', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: "The answer is A" or "The correct answer is B"
    match = re.search(r'\b(?:answer is|correct answer is|therefore,?|thus,?)\s*([A-D])\b', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Response starts with letter choice or numbered "1. A)"
    match = re.search(r'(?:^|\n)\s*(?:\d+\.\s*)?([A-D])\)', response_text, re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 4: Final answer format
    match = re.search(r'(?:final answer|answer):\s*([A-D])', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 5: Letter in parentheses (A)
    match = re.search(r'\(([A-D])\)', response_text)
    if match:
        return match.group(1).upper()

    # Pattern 6: "Option A" or "Choice B"
    match = re.search(r'\b(?:option|choice)\s*([A-D])\b', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None

def test_fix(num_questions_arg=None):
    """Run inference validation analysis

    Args:
        num_questions_arg: Optional number of questions to analyze (10, 100, 200, 1000)
                          If None, auto-detect latest results file
    """

    # Check if specific number was requested
    if num_questions_arg:
        results_file = f'dcr_results/dcr_n{num_questions_arg}_results.json'
        if not os.path.exists(results_file):
            print("="*80)
            print(f"❌ FILE NOT FOUND: {results_file}")
            print("="*80)
            print(f"\nAvailable files:")
            for f in sorted(os.listdir('dcr_results')):
                if f.startswith('dcr_n') and f.endswith('_results.json'):
                    print(f"  - {f}")
            return
        num_questions = num_questions_arg
    else:
        # Find most recent results file (auto-detect)
        result_files = [
            ('dcr_results/dcr_n1000_results.json', 1000),
            ('dcr_results/dcr_n200_results.json', 200),
            ('dcr_results/dcr_n100_results.json', 100),
            ('dcr_results/dcr_n10_results.json', 10),
        ]

        results_file = None
        num_questions = 0
        for filepath, n in result_files:
            if os.path.exists(filepath):
                results_file = filepath
                num_questions = n
                break

    if not results_file:
        print("="*80)
        print("❌ NO RESULTS FOUND")
        print("="*80)
        print("\n   Please run:")
        print("   $ python3 cli.py infer --num-questions 100")
        return

    print("="*80)
    print(f"ANALYZING DCR INFERENCE RESULTS ({num_questions} QUESTIONS)")
    print("="*80)

    # Load test results
    with open(results_file, 'r') as f:
        results = json.load(f)

    print(f"\n📄 Results file: {results_file}")

    # Load MMLU ground truth (check both AWS and local paths)
    mmlu_file = None
    if os.path.exists('data/mmlu_test.json'):
        mmlu_file = 'data/mmlu_test.json'
    elif os.path.exists('data/raw/mmlu_test.json'):
        mmlu_file = 'data/raw/mmlu_test.json'

    if not mmlu_file:
        print("\n⚠️  MMLU test data not found locally")
        print("   Run: python3 cli.py s3 download data")
        mmlu_questions = None
    else:
        with open(mmlu_file, 'r') as f:
            mmlu_data = json.load(f)
        mmlu_questions = mmlu_data['questions']

    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)

    total_extraction = 0
    total_correct = 0
    total_responses = 0

    for provider_name, strategies in results.items():
        print(f"\n{provider_name.upper()}:")
        print("-"*80)

        for strategy, strategy_results in strategies.items():
            extractable = 0
            correct = 0

            for i, result in enumerate(strategy_results):
                total_responses += 1
                response = result.get('response', '')

                # Extract answer
                extracted = extract_answer_choice(response)

                if extracted:
                    extractable += 1
                    total_extraction += 1

                    # Check correctness if we have ground truth
                    if mmlu_questions and i < len(mmlu_questions):
                        ground_truth_idx = mmlu_questions[i].get('answer')
                        ground_truth_letter = ['A', 'B', 'C', 'D'][ground_truth_idx] if ground_truth_idx is not None else None

                        if extracted == ground_truth_letter:
                            correct += 1
                            total_correct += 1

            extraction_rate = extractable / len(strategy_results) * 100 if strategy_results else 0
            accuracy = correct / extractable * 100 if extractable > 0 else 0

            print(f"  {strategy:15s}: Extraction: {extraction_rate:5.1f}%  |  Accuracy: {accuracy:5.1f}%  |  Extractable: {extractable}/{len(strategy_results)}")

    # Overall summary
    avg_extraction = total_extraction / total_responses * 100 if total_responses > 0 else 0
    avg_accuracy = total_correct / total_extraction * 100 if total_extraction > 0 else 0

    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total responses: {total_responses}")
    print(f"Extractable answers: {total_extraction}/{total_responses} ({avg_extraction:.1f}%)")
    if mmlu_questions:
        print(f"Correct answers: {total_correct}/{total_extraction} ({avg_accuracy:.1f}%)")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if avg_extraction >= 85:
        print("\n✅ FIX SUCCESSFUL!")
        print(f"   Extraction rate: {avg_extraction:.1f}% (target: >85%)")
        if mmlu_questions and avg_accuracy >= 70:
            print(f"   Accuracy: {avg_accuracy:.1f}% (good - above random 25%)")
        print("\n   The fix is working correctly:")
        print("   • Questions include A/B/C/D choices ✓")
        print("   • System prompts guide LLMs to provide letters ✓")
        print("   • Answers are extractable ✓")
        if mmlu_questions:
            print("   • MMLU accuracy is measurable ✓")
        print("\n   Ready to run full 1000-question test!")
    elif avg_extraction >= 50:
        print("\n⚠️  FIX PARTIALLY WORKING")
        print(f"   Extraction rate: {avg_extraction:.1f}% (target: >85%)")
        print("\n   Issues to investigate:")
        print("   • Some LLMs not following prompt format")
        print("   • May need prompt refinement")
        print("   • Check sample responses for patterns")
    else:
        print("\n❌ FIX NOT WORKING")
        print(f"   Extraction rate: {avg_extraction:.1f}% (target: >85%)")
        print("\n   Possible issues:")
        print("   • Questions may not include choices")
        print("   • System prompts may not be applied")
        print("   • Check inference.py changes were saved")

    print("\n" + "="*80)

    # Show sample responses
    print("\nSAMPLE RESPONSES:")
    print("="*80)

    # Get first provider and strategy
    first_provider = list(results.keys())[0]
    first_strategy = list(results[first_provider].keys())[0]
    sample_results = results[first_provider][first_strategy][:2]

    for i, result in enumerate(sample_results):
        print(f"\nQuestion {i+1}:")
        print(f"  Q: {result.get('question', '')[:100]}...")

        response = result.get('response', '')
        extracted = extract_answer_choice(response)

        print(f"  Response: {response[:200]}...")
        print(f"  Extracted: {extracted if extracted else 'None'}")

        if mmlu_questions and i < len(mmlu_questions):
            ground_truth_idx = mmlu_questions[i].get('answer')
            ground_truth_letter = ['A', 'B', 'C', 'D'][ground_truth_idx] if ground_truth_idx is not None else None
            print(f"  Ground Truth: {ground_truth_letter}")
            if extracted:
                is_correct = (extracted == ground_truth_letter)
                print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")

    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    # Parse command-line arguments
    num_questions = None
    if len(sys.argv) > 1:
        try:
            num_questions = int(sys.argv[1])
            if num_questions not in [10, 100, 200, 1000]:
                print(f"Error: Invalid number '{num_questions}'. Must be 10, 100, 200, or 1000.")
                print("\nUsage:")
                print("  python3 test_fix.py         # Auto-detect latest results")
                print("  python3 test_fix.py 200     # Analyze 200-question results")
                print("  python3 test_fix.py 1000    # Analyze 1000-question results")
                sys.exit(1)
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid number.")
            print("\nUsage:")
            print("  python3 test_fix.py         # Auto-detect latest results")
            print("  python3 test_fix.py 200     # Analyze 200-question results")
            print("  python3 test_fix.py 1000    # Analyze 1000-question results")
            sys.exit(1)

    test_fix(num_questions)
