#!/usr/bin/env python3
"""
Merge individual provider result files into one combined file

Usage:
    python3 merge_results.py 1000
    python3 merge_results.py 200
"""

import json
import sys
import os
from pathlib import Path


def merge_results(num_questions):
    """Merge provider-specific results into combined file"""

    print(f"Merging {num_questions}-question results...")
    print("="*80)

    # Check which provider files exist
    provider_files = {
        'openai': f'dcr_results/openai_dcr_results.json',
        'gemini': f'dcr_results/gemini_dcr_results.json',
        'claude': f'dcr_results/claude_dcr_results.json'
    }

    combined = {}
    missing = []

    for provider, filepath in provider_files.items():
        if os.path.exists(filepath):
            print(f"✓ Found {provider}: {filepath}")
            with open(filepath, 'r') as f:
                combined[provider] = json.load(f)

            # Verify it has the right number of questions
            for strategy, results in combined[provider].items():
                if len(results) != num_questions:
                    print(f"  ⚠️  Warning: {provider}/{strategy} has {len(results)} questions (expected {num_questions})")
        else:
            print(f"✗ Missing {provider}: {filepath}")
            missing.append(provider)

    if not combined:
        print("\n❌ No provider files found!")
        return False

    # Save combined results
    output_file = f'dcr_results/dcr_n{num_questions}_results.json'
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2)

    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"\n✓ Combined {len(combined)} providers")
    print(f"✓ Output: {output_file}")

    # Summary
    print(f"\nProviders included:")
    for provider in combined:
        strategies = list(combined[provider].keys())
        print(f"  {provider:8s}: {len(strategies)} strategies ({', '.join(strategies)})")

    if missing:
        print(f"\nProviders missing: {', '.join(missing)}")

    print()
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 merge_results.py <num_questions>")
        print("\nExamples:")
        print("  python3 merge_results.py 1000")
        print("  python3 merge_results.py 200")
        sys.exit(1)

    try:
        num_questions = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)

    success = merge_results(num_questions)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
