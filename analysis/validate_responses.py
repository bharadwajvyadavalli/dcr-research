#!/usr/bin/env python3
"""
Comprehensive Response Validation

Checks all 9000 responses (3 providers × 3 strategies × 1000 questions) for:
1. API errors / error messages
2. Empty or null responses
3. Valid extractable answers
4. Correct answers vs MMLU ground truth
5. Response quality issues

Usage:
    python3 validate_responses.py
"""

import json
import re
from collections import defaultdict

def extract_answer_choice(response_text):
    """Extract A/B/C/D answer from LLM response"""
    if not response_text:
        return None

    response_text = response_text.strip()

    # Pattern 1a: "Answer: X" format
    match = re.search(r'(?:^|\n)\s*answer:\s*\*{0,2}([A-D])\*{0,2}', response_text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 1b: Gemini-specific markdown formats
    match = re.search(r'(?:correct\s+)?letter\*{0,2}:\*{0,2}\s*([A-D])\*{0,2}', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 1c: Numbered list format
    match = re.search(r'\d+\.\s+\*{0,2}(?:correct\s+)?letter\*{0,2}:\*{0,2}\s*([A-D])', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 1d: Original pattern
    match = re.search(r'(?:correct\s+)?letter[:\s]+\*{0,2}([A-D])\*{0,2}', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: "The answer is A"
    match = re.search(r'\b(?:answer is|correct answer is|therefore,?|thus,?)\s*([A-D])\b', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Response starts with letter choice
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


def detect_error_response(response_text):
    """Detect if response contains actual API error messages"""
    if not response_text:
        return "EMPTY_RESPONSE"

    response_lower = response_text.lower()

    # Only check first 200 chars for API errors (errors appear at start)
    # This avoids false positives from words like "error" in explanations
    response_start = response_lower[:200]

    # Actual API error patterns (more specific)
    actual_error_patterns = [
        'rate limit exceeded',
        'quota exceeded',
        'api error',
        'service error',
        'service unavailable',
        'internal server error',
        'request failed',
        'authentication failed',
        'timeout error',
        'connection error',
        'http error',
        'status code: 429',
        'status code: 500',
        'status code: 503',
        'please retry',
        'try again later'
    ]

    for pattern in actual_error_patterns:
        if pattern in response_start:
            return f"API_ERROR: {pattern}"

    # Check if response is suspiciously short (<10 chars) and doesn't start with "Answer:"
    if len(response_text.strip()) < 10 and not response_text.strip().lower().startswith('answer'):
        return "SUSPICIOUSLY_SHORT"

    return None


def validate_all_responses():
    """Comprehensive validation of all responses"""

    print("="*80)
    print("COMPREHENSIVE RESPONSE VALIDATION")
    print("="*80)

    # Load results
    with open('dcr_results/dcr_n1000_results.json', 'r') as f:
        results = json.load(f)

    # Load MMLU ground truth
    mmlu_file = None
    if os.path.exists('data/mmlu_test.json'):
        mmlu_file = 'data/mmlu_test.json'
    elif os.path.exists('data/raw/mmlu_test.json'):
        mmlu_file = 'data/raw/mmlu_test.json'

    if mmlu_file:
        with open(mmlu_file, 'r') as f:
            mmlu_data = json.load(f)
        mmlu_questions = mmlu_data['questions']
    else:
        mmlu_questions = None
        print("\n⚠️  MMLU ground truth not found - skipping correctness check\n")

    # Statistics
    stats = defaultdict(lambda: {
        'total': 0,
        'empty': 0,
        'errors': 0,
        'valid': 0,
        'extractable': 0,
        'non_extractable': 0,
        'correct': 0,
        'incorrect': 0,
        'error_samples': [],
        'non_extractable_samples': []
    })

    overall_stats = {
        'total': 0,
        'empty': 0,
        'errors': 0,
        'valid': 0,
        'extractable': 0,
        'non_extractable': 0,
        'correct': 0,
        'incorrect': 0
    }

    # Validate each response
    for provider_name, strategies in results.items():
        for strategy, strategy_results in strategies.items():
            key = f"{provider_name}/{strategy}"

            for i, result in enumerate(strategy_results):
                response = result.get('response', '')

                stats[key]['total'] += 1
                overall_stats['total'] += 1

                # Check for empty
                if not response or not response.strip():
                    stats[key]['empty'] += 1
                    overall_stats['empty'] += 1
                    stats[key]['error_samples'].append({
                        'question_idx': i,
                        'issue': 'EMPTY_RESPONSE',
                        'response': response[:100] if response else 'NULL'
                    })
                    continue

                # Check for errors
                error_type = detect_error_response(response)
                if error_type:
                    stats[key]['errors'] += 1
                    overall_stats['errors'] += 1
                    stats[key]['error_samples'].append({
                        'question_idx': i,
                        'issue': error_type,
                        'response': response[:200]
                    })
                    continue

                # Valid response (no errors)
                stats[key]['valid'] += 1
                overall_stats['valid'] += 1

                # Try to extract answer
                extracted = extract_answer_choice(response)

                if extracted:
                    stats[key]['extractable'] += 1
                    overall_stats['extractable'] += 1

                    # Check correctness if we have ground truth
                    if mmlu_questions and i < len(mmlu_questions):
                        ground_truth_idx = mmlu_questions[i].get('answer')
                        ground_truth_letter = ['A', 'B', 'C', 'D'][ground_truth_idx] if ground_truth_idx is not None else None

                        if extracted == ground_truth_letter:
                            stats[key]['correct'] += 1
                            overall_stats['correct'] += 1
                        else:
                            stats[key]['incorrect'] += 1
                            overall_stats['incorrect'] += 1
                else:
                    stats[key]['non_extractable'] += 1
                    overall_stats['non_extractable'] += 1

                    # Save sample for review (limit to 3)
                    if len(stats[key]['non_extractable_samples']) < 3:
                        stats[key]['non_extractable_samples'].append({
                            'question_idx': i,
                            'response': response[:300]
                        })

    # Print results
    print("\n" + "="*80)
    print("OVERALL SUMMARY (All 9000 responses)")
    print("="*80)
    print(f"Total responses:        {overall_stats['total']}")
    print(f"Empty responses:        {overall_stats['empty']} ({overall_stats['empty']/overall_stats['total']*100:.2f}%)")
    print(f"Error responses:        {overall_stats['errors']} ({overall_stats['errors']/overall_stats['total']*100:.2f}%)")
    print(f"Valid responses:        {overall_stats['valid']} ({overall_stats['valid']/overall_stats['total']*100:.2f}%)")
    print()
    print(f"Extractable answers:    {overall_stats['extractable']} ({overall_stats['extractable']/overall_stats['total']*100:.2f}%)")
    print(f"Non-extractable:        {overall_stats['non_extractable']} ({overall_stats['non_extractable']/overall_stats['total']*100:.2f}%)")

    if mmlu_questions:
        print()
        print(f"Correct answers:        {overall_stats['correct']} ({overall_stats['correct']/overall_stats['extractable']*100:.2f}% of extractable)")
        print(f"Incorrect answers:      {overall_stats['incorrect']} ({overall_stats['incorrect']/overall_stats['extractable']*100:.2f}% of extractable)")

    print("\n" + "="*80)
    print("BREAKDOWN BY PROVIDER AND STRATEGY")
    print("="*80)

    for key in sorted(stats.keys()):
        s = stats[key]
        print(f"\n{key.upper()}:")
        print(f"  Total:          {s['total']}")
        print(f"  Empty:          {s['empty']} ({s['empty']/s['total']*100:.2f}%)")
        print(f"  Errors:         {s['errors']} ({s['errors']/s['total']*100:.2f}%)")
        print(f"  Valid:          {s['valid']} ({s['valid']/s['total']*100:.2f}%)")
        print(f"  Extractable:    {s['extractable']} ({s['extractable']/s['total']*100:.2f}%)")
        print(f"  Non-extract:    {s['non_extractable']} ({s['non_extractable']/s['total']*100:.2f}%)")

        if mmlu_questions:
            if s['extractable'] > 0:
                print(f"  Correct:        {s['correct']} ({s['correct']/s['extractable']*100:.2f}% of extractable)")
                print(f"  Incorrect:      {s['incorrect']} ({s['incorrect']/s['extractable']*100:.2f}% of extractable)")

        # Show error samples
        if s['error_samples']:
            print(f"\n  ⚠️  Error samples ({len(s['error_samples'])}):")
            for sample in s['error_samples'][:3]:
                print(f"    Q{sample['question_idx']}: {sample['issue']}")
                print(f"      Response: {sample['response'][:150]}...")

        # Show non-extractable samples
        if s['non_extractable_samples']:
            print(f"\n  ⚠️  Non-extractable samples ({len(s['non_extractable_samples'])}):")
            for sample in s['non_extractable_samples']:
                print(f"    Q{sample['question_idx']}:")
                print(f"      Response: {sample['response'][:150]}...")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    valid_rate = overall_stats['valid'] / overall_stats['total'] * 100
    extraction_rate = overall_stats['extractable'] / overall_stats['total'] * 100

    if valid_rate >= 99 and extraction_rate >= 95:
        print("\n✅ DATASET QUALITY: EXCELLENT")
        print(f"   Valid responses: {valid_rate:.2f}% (target: >99%)")
        print(f"   Extraction rate: {extraction_rate:.2f}% (target: >95%)")
        print("\n   The dataset is publication-ready:")
        print("   • No significant API errors ✓")
        print("   • High extraction rate ✓")
        print("   • Valid responses across all providers ✓")
    elif valid_rate >= 95 and extraction_rate >= 85:
        print("\n⚠️  DATASET QUALITY: GOOD (some issues)")
        print(f"   Valid responses: {valid_rate:.2f}%")
        print(f"   Extraction rate: {extraction_rate:.2f}%")
        print("\n   Minor issues detected - review samples above")
    else:
        print("\n❌ DATASET QUALITY: ISSUES DETECTED")
        print(f"   Valid responses: {valid_rate:.2f}%")
        print(f"   Extraction rate: {extraction_rate:.2f}%")
        print("\n   Significant issues - review error samples above")

    print("\n" + "="*80)


if __name__ == '__main__':
    import os
    validate_all_responses()
