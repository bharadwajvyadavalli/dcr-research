#!/usr/bin/env python3
"""
Analyze DCR Multi-Provider Results

Generates comparison statistics and paper assets:
- Success rates by strategy and provider
- Template distribution analysis
- Router accuracy comparison
- Provider performance rankings

Usage:
    python3 src/analysis/analyze_dcr_results.py
    python3 src/analysis/analyze_dcr_results.py --num-questions 1000
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: tiktoken not installed. Install with: pip install tiktoken")
    print("   Falling back to approximate word-based counting.")
    TIKTOKEN_AVAILABLE = False


def load_dcr_results(num_questions: Optional[int] = None) -> Dict[str, Any]:
    """Load DCR results from dcr_results directory

    Args:
        num_questions: If specified, load results for this exact number of questions.
                      If None, load the file with the most questions.
    """
    results_files = list(Path('dcr_results').glob('dcr_n*_results.json'))

    if not results_files:
        raise FileNotFoundError("No DCR results found. Run run_dcr_pipeline.py first.")

    # Helper to extract number from filename
    def extract_n(path):
        # Extract number from "dcr_n1000_results.json" -> 1000
        import re
        match = re.search(r'dcr_n(\d+)_results\.json', path.name)
        return int(match.group(1)) if match else 0

    if num_questions is not None:
        # Look for exact match
        target_file = Path(f'dcr_results/dcr_n{num_questions}_results.json')
        if target_file.exists():
            results_file = target_file
        else:
            raise FileNotFoundError(
                f"Results file for {num_questions} questions not found: {target_file}\n"
                f"Available files: {[f.name for f in results_files]}"
            )
    else:
        # Load most recent results (sort by number in filename, not lexicographically)
        results_file = sorted(results_files, key=extract_n)[-1]

    print(f"Loading results from: {results_file}")

    with open(results_file, 'r') as f:
        return json.load(f)


def analyze_success_rates(results: Dict[str, Any]) -> pd.DataFrame:
    """Calculate success rates by provider and strategy"""
    data = []

    for provider, strategies in results.items():
        for strategy, responses in strategies.items():
            total = len(responses)
            succeeded = sum(1 for r in responses if r['success'])
            failed = total - succeeded
            success_rate = (succeeded / total * 100) if total > 0 else 0

            data.append({
                'Provider': provider,
                'Strategy': strategy,
                'Total': total,
                'Succeeded': succeeded,
                'Failed': failed,
                'Success Rate (%)': round(success_rate, 2)
            })

    return pd.DataFrame(data)


def analyze_template_distribution(results: Dict[str, Any]) -> pd.DataFrame:
    """Analyze template distribution by strategy"""
    data = []

    for provider, strategies in results.items():
        for strategy, responses in strategies.items():
            # Count templates used
            template_counts = {}
            for r in responses:
                template = r.get('template', 'unknown')
                template_counts[template] = template_counts.get(template, 0) + 1

            for template, count in template_counts.items():
                data.append({
                    'Provider': provider,
                    'Strategy': strategy,
                    'Template': template,
                    'Count': count,
                    'Percentage': round(count / len(responses) * 100, 2)
                })

    return pd.DataFrame(data)


def calculate_router_accuracy(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate router accuracy: how well routers match ground truth"""
    accuracy_data = {}

    # Load ground truth from router predictions
    predictions_file = Path('dcr_results/router_predictions.json')
    if not predictions_file.exists():
        print("⚠️  Warning: router_predictions.json not found - skipping accuracy calculation")
        return {}

    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    ground_truth_templates = [p.get('ground_truth', p.get('baseline', 'standard')) for p in predictions]
    total = len(ground_truth_templates)

    for provider in results.keys():
        strategies = results[provider]
        accuracy_data[provider] = {}

        # Simple Neural accuracy
        if 'simple_neural' in strategies:
            neural_templates = [r.get('template') for r in strategies['simple_neural']]
            matches = sum(1 for gt, n in zip(ground_truth_templates, neural_templates) if gt == n)
            accuracy_data[provider]['simple_neural_accuracy'] = round(matches / total * 100, 2)
            accuracy_data[provider]['simple_neural_matches'] = matches
            accuracy_data[provider]['simple_neural_misses'] = total - matches

        # RoBERTa accuracy
        if 'roberta' in strategies:
            roberta_templates = [r.get('template') for r in strategies['roberta']]
            matches = sum(1 for gt, r in zip(ground_truth_templates, roberta_templates) if gt == r)
            accuracy_data[provider]['roberta_accuracy'] = round(matches / total * 100, 2)
            accuracy_data[provider]['roberta_matches'] = matches
            accuracy_data[provider]['roberta_misses'] = total - matches

        # Router agreement (how often do both routers agree)
        if 'simple_neural' in strategies and 'roberta' in strategies:
            neural_templates = [r.get('template') for r in strategies['simple_neural']]
            roberta_templates = [r.get('template') for r in strategies['roberta']]
            agreement = sum(1 for n, r in zip(neural_templates, roberta_templates) if n == r)
            accuracy_data[provider]['router_agreement'] = round(agreement / total * 100, 2)

    return accuracy_data


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (accurate) or fallback to word count"""
    if not text:
        return 0

    if TIKTOKEN_AVAILABLE:
        try:
            # Use gpt-4 encoding (works for all providers as approximation)
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(text))
        except Exception as e:
            # Fallback if tiktoken fails
            pass

    # Fallback: approximate as ~1.3 tokens per word (rough average for English)
    words = len(text.split())
    return int(words * 1.3)


def calculate_actual_token_savings(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate ACTUAL token usage from real API response text"""
    token_data = {}

    for provider in results.keys():
        strategies = results[provider]
        token_data[provider] = {}

        for strategy_name, responses in strategies.items():
            total_tokens = 0
            successful_responses = 0

            for r in responses:
                # Only count successful responses
                if r.get('success', False) and r.get('response'):
                    tokens = count_tokens(r['response'])
                    total_tokens += tokens
                    successful_responses += 1

            token_data[provider][f'{strategy_name}_total_tokens'] = total_tokens
            token_data[provider][f'{strategy_name}_avg_tokens'] = round(total_tokens / successful_responses) if successful_responses > 0 else 0
            token_data[provider][f'{strategy_name}_count'] = successful_responses

        # Calculate savings vs baseline
        if 'baseline' in strategies:
            baseline_tokens = token_data[provider]['baseline_total_tokens']

            if 'simple_neural' in strategies:
                neural_tokens = token_data[provider]['simple_neural_total_tokens']
                savings = baseline_tokens - neural_tokens
                savings_pct = round((savings / baseline_tokens * 100), 2) if baseline_tokens > 0 else 0
                token_data[provider]['simple_neural_savings'] = savings
                token_data[provider]['simple_neural_savings_pct'] = savings_pct

            if 'roberta' in strategies:
                roberta_tokens = token_data[provider]['roberta_total_tokens']
                savings = baseline_tokens - roberta_tokens
                savings_pct = round((savings / baseline_tokens * 100), 2) if baseline_tokens > 0 else 0
                token_data[provider]['roberta_savings'] = savings
                token_data[provider]['roberta_savings_pct'] = savings_pct

    return token_data


def compare_routers(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare router performance (legacy - kept for compatibility)"""
    comparison = {
        'baseline_vs_simple_neural': {},
        'baseline_vs_roberta': {},
        'simple_neural_vs_roberta': {}
    }

    for provider in results.keys():
        strategies = results[provider]

        # Compare strategies if all exist
        if 'baseline' in strategies and 'simple_neural' in strategies:
            baseline_success = sum(1 for r in strategies['baseline'] if r['success'])
            simple_success = sum(1 for r in strategies['simple_neural'] if r['success'])
            total = len(strategies['baseline'])

            comparison['baseline_vs_simple_neural'][provider] = {
                'baseline_success_rate': round(baseline_success / total * 100, 2),
                'simple_neural_success_rate': round(simple_success / total * 100, 2),
                'improvement': round((simple_success - baseline_success) / total * 100, 2)
            }

        if 'baseline' in strategies and 'roberta' in strategies:
            baseline_success = sum(1 for r in strategies['baseline'] if r['success'])
            roberta_success = sum(1 for r in strategies['roberta'] if r['success'])
            total = len(strategies['baseline'])

            comparison['baseline_vs_roberta'][provider] = {
                'baseline_success_rate': round(baseline_success / total * 100, 2),
                'roberta_success_rate': round(roberta_success / total * 100, 2),
                'improvement': round((roberta_success - baseline_success) / total * 100, 2)
            }

        if 'simple_neural' in strategies and 'roberta' in strategies:
            simple_success = sum(1 for r in strategies['simple_neural'] if r['success'])
            roberta_success = sum(1 for r in strategies['roberta'] if r['success'])
            total = len(strategies['simple_neural'])

            comparison['simple_neural_vs_roberta'][provider] = {
                'simple_neural_success_rate': round(simple_success / total * 100, 2),
                'roberta_success_rate': round(roberta_success / total * 100, 2),
                'difference': round((roberta_success - simple_success) / total * 100, 2)
            }

    return comparison


def generate_summary_report(results: Dict[str, Any], router_accuracy: Dict[str, Any] = None, token_savings: Dict[str, Any] = None, validation: Dict[str, Any] = None) -> str:
    """Generate human-readable summary report"""
    report = []
    report.append("="*70)
    report.append("DCR MULTI-PROVIDER ANALYSIS REPORT")
    report.append("="*70)
    report.append("")

    # Data Quality Validation (NEW - critical!)
    if validation:
        report.append("DATA QUALITY VALIDATION:")
        report.append("-" * 70)
        has_issues = False
        for provider, strategies in validation.items():
            for strategy, issues in strategies.items():
                if issues['low_token_responses'] > 0 or issues['failed_responses'] > 0:
                    has_issues = True
                    report.append(f"\n⚠️  {provider.upper()} - {strategy}:")
                    if issues['failed_responses'] > 0:
                        report.append(f"    {issues['failed_responses']} failed responses")
                    if issues['low_token_responses'] > 0:
                        report.append(f"    {issues['low_token_responses']} suspiciously low token responses (<20 tokens)")

                        # Count likely errors
                        error_count = sum(1 for r in issues['suspicious_responses'] if r.get('likely_error', False))
                        if error_count > 0:
                            report.append(f"    → {error_count} likely error messages (contain error keywords)")
                        else:
                            report.append(f"    → Likely valid short answers (no error keywords detected)")

                        # Show sample responses
                        report.append(f"    Sample responses:")
                        for i, sample in enumerate(issues['suspicious_responses'][:3]):
                            report.append(f"      Q{sample['question_idx']}: {sample['output_tokens']} tokens - \"{sample['response_preview'][:80]}...\"")

                    stats = issues['token_stats']
                    report.append(f"    Token distribution: min={stats['min']}, avg={stats['avg']}, max={stats['max']}")
                    report.append(f"    <20 tokens: {stats['<20_tokens']}, 20-100: {stats['20-100_tokens']}, 100-300: {stats['100-300_tokens']}, >300: {stats['>300_tokens']}")

        if not has_issues:
            report.append("✅ All responses passed quality checks")
        report.append("")

    # Overall statistics
    total_questions = len(list(results.values())[0][list(list(results.values())[0].keys())[0]])
    num_providers = len(results)
    num_strategies = len(list(results.values())[0])
    total_calls = total_questions * num_providers * num_strategies

    report.append(f"Dataset:")
    report.append(f"  Questions: {total_questions}")
    report.append(f"  Providers: {num_providers} ({', '.join(results.keys())})")
    report.append(f"  Strategies: {num_strategies}")
    report.append(f"  Total API calls: {total_calls}")
    report.append("")

    # Router Accuracy (CRITICAL FOR DCR!)
    if router_accuracy:
        report.append("Router Accuracy (vs. Baseline Ground Truth):")
        report.append("-" * 70)
        for provider, metrics in router_accuracy.items():
            report.append(f"\n{provider.upper()}:")
            if 'simple_neural_accuracy' in metrics:
                report.append(f"  Simple Neural: {metrics['simple_neural_accuracy']:.1f}% "
                            f"({metrics['simple_neural_matches']}/{total_questions} correct)")
            if 'roberta_accuracy' in metrics:
                report.append(f"  RoBERTa:       {metrics['roberta_accuracy']:.1f}% "
                            f"({metrics['roberta_matches']}/{total_questions} correct)")
            if 'router_agreement' in metrics:
                report.append(f"  Agreement:     {metrics['router_agreement']:.1f}% "
                            f"(routers agree with each other)")
        report.append("")

    # Token Savings (CRITICAL FOR DCR!)
    if token_savings:
        report.append("Actual Token Savings from Real API Responses (vs. Baseline):")
        report.append("-" * 70)
        for provider, metrics in token_savings.items():
            report.append(f"\n{provider.upper()}:")
            if 'baseline_total_tokens' in metrics:
                baseline_total = metrics['baseline_total_tokens']
                baseline_avg = metrics.get('baseline_avg_tokens', 0)
                report.append(f"  Baseline:      {baseline_total:,} tokens (avg: {baseline_avg} tokens/response)")
            if 'simple_neural_total_tokens' in metrics:
                tokens = metrics['simple_neural_total_tokens']
                avg = metrics.get('simple_neural_avg_tokens', 0)
                savings = metrics.get('simple_neural_savings', 0)
                savings_pct = metrics.get('simple_neural_savings_pct', 0)
                report.append(f"  Simple Neural: {tokens:,} tokens (avg: {avg} tokens/response)")
                report.append(f"                 Saves {savings:,} tokens ({savings_pct:.1f}%)")
            if 'roberta_total_tokens' in metrics:
                tokens = metrics['roberta_total_tokens']
                avg = metrics.get('roberta_avg_tokens', 0)
                savings = metrics.get('roberta_savings', 0)
                savings_pct = metrics.get('roberta_savings_pct', 0)
                report.append(f"  RoBERTa:       {tokens:,} tokens (avg: {avg} tokens/response)")
                report.append(f"                 Saves {savings:,} tokens ({savings_pct:.1f}%)")
        report.append("")

    # Success rates
    report.append("API Success Rates by Provider and Strategy:")
    report.append("-" * 70)

    success_df = analyze_success_rates(results)
    for provider in success_df['Provider'].unique():
        report.append(f"\n{provider.upper()}:")
        provider_data = success_df[success_df['Provider'] == provider]
        for _, row in provider_data.iterrows():
            report.append(f"  {row['Strategy']:15s}: {row['Succeeded']}/{row['Total']} ({row['Success Rate (%)']:.1f}%)")

    report.append("")
    report.append("="*70)

    return "\n".join(report)


def validate_response_quality(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate response quality across all providers and strategies.
    Detects:
    - Suspiciously low token responses (< 20 tokens, likely errors)
    - Token distribution anomalies
    - Error patterns
    """
    validation_report = {}

    for provider, strategies in results.items():
        validation_report[provider] = {}

        for strategy, responses in strategies.items():
            issues = {
                'total_responses': len(responses),
                'low_token_responses': 0,  # < 20 tokens
                'failed_responses': 0,
                'token_stats': {},
                'suspicious_responses': []
            }

            tokens_list = []

            for idx, r in enumerate(responses):
                # Check for failures
                if not r.get('success', False):
                    issues['failed_responses'] += 1
                    continue

                # Get output tokens
                output_tokens = r.get('output_tokens', 0)
                tokens_list.append(output_tokens)

                # Flag suspiciously low token responses
                if output_tokens < 20:
                    issues['low_token_responses'] += 1
                    response_text = r.get('response', '')

                    # Check for error indicators
                    error_keywords = ['error', 'cannot', 'sorry', 'unable', 'refuse', 'declined', 'failed']
                    has_error = any(keyword in response_text.lower() for keyword in error_keywords)

                    issues['suspicious_responses'].append({
                        'question_idx': idx,
                        'question': r.get('question', '')[:100],
                        'output_tokens': output_tokens,
                        'response_preview': response_text[:200],
                        'full_response': response_text,
                        'likely_error': has_error,
                        'template_used': r.get('template_used', 'unknown')
                    })

            # Calculate token statistics
            if tokens_list:
                issues['token_stats'] = {
                    'min': min(tokens_list),
                    'max': max(tokens_list),
                    'avg': round(sum(tokens_list) / len(tokens_list), 1),
                    'median': round(sorted(tokens_list)[len(tokens_list)//2], 1),
                    '<20_tokens': sum(1 for t in tokens_list if t < 20),
                    '20-100_tokens': sum(1 for t in tokens_list if 20 <= t < 100),
                    '100-300_tokens': sum(1 for t in tokens_list if 100 <= t < 300),
                    '>300_tokens': sum(1 for t in tokens_list if t >= 300)
                }

            validation_report[provider][strategy] = issues

    return validation_report


def save_analysis(results: Dict[str, Any]):
    """Save all analysis outputs"""
    os.makedirs('dcr_analysis', exist_ok=True)

    print("\nGenerating analysis...")

    # 0. Validate response quality FIRST
    validation = validate_response_quality(results)
    with open('dcr_analysis/validation_report.json', 'w') as f:
        json.dump(validation, f, indent=2)
    print("  ✓ validation_report.json")

    # Print validation warnings and save detailed suspicious responses
    suspicious_samples = []
    for provider, strategies in validation.items():
        for strategy, issues in strategies.items():
            if issues['low_token_responses'] > 0:
                print(f"  ⚠️  {provider}/{strategy}: {issues['low_token_responses']} suspicious low-token responses (<20 tokens)")

                # Count likely errors
                error_count = sum(1 for r in issues['suspicious_responses'] if r.get('likely_error', False))
                if error_count > 0:
                    print(f"      → {error_count} likely error messages (contain error keywords)")

                # Show samples (first 3)
                for i, sample in enumerate(issues['suspicious_responses'][:3]):
                    print(f"      Sample #{i+1}: Q{sample['question_idx']}, {sample['output_tokens']} tokens, template={sample['template_used']}")
                    print(f"        Response: {sample['response_preview'][:100]}...")
                    suspicious_samples.append({
                        'provider': provider,
                        'strategy': strategy,
                        **sample
                    })

            if issues['failed_responses'] > 0:
                print(f"  ⚠️  {provider}/{strategy}: {issues['failed_responses']} failed responses")

    # Save detailed suspicious responses for manual review
    if suspicious_samples:
        with open('dcr_analysis/suspicious_responses.json', 'w') as f:
            json.dump(suspicious_samples, f, indent=2)
        print("  ✓ suspicious_responses.json (detailed samples for manual review)")

    # 1. Success rates table
    success_df = analyze_success_rates(results)
    success_df.to_csv('dcr_analysis/success_rates.csv', index=False)
    print("  ✓ success_rates.csv")

    # 2. Template distribution
    template_df = analyze_template_distribution(results)
    template_df.to_csv('dcr_analysis/template_distribution.csv', index=False)
    print("  ✓ template_distribution.csv")

    # 3. Router accuracy (NEW - critical for DCR!)
    router_accuracy = calculate_router_accuracy(results)
    with open('dcr_analysis/router_accuracy.json', 'w') as f:
        json.dump(router_accuracy, f, indent=2)
    print("  ✓ router_accuracy.json")

    # 4. Actual token savings from real API responses (CRITICAL for DCR!)
    token_savings = calculate_actual_token_savings(results)
    with open('dcr_analysis/token_savings.json', 'w') as f:
        json.dump(token_savings, f, indent=2)
    print("  ✓ token_savings.json (ACTUAL response token counts)")

    # 5. Router comparison (legacy success rates)
    router_comparison = compare_routers(results)
    with open('dcr_analysis/router_comparison.json', 'w') as f:
        json.dump(router_comparison, f, indent=2)
    print("  ✓ router_comparison.json")

    # 6. Summary report (include validation)
    summary = generate_summary_report(results, router_accuracy, token_savings, validation)
    with open('dcr_analysis/summary_report.txt', 'w') as f:
        f.write(summary)
    print("  ✓ summary_report.txt")

    print(f"\n{summary}\n")

    print(f"All analysis saved to: dcr_analysis/")


def main():
    parser = argparse.ArgumentParser(description='Analyze DCR Multi-Provider Results')
    parser.add_argument('--num-questions', type=int, default=None,
                       help='Number of questions in the results file to analyze (default: auto-select largest)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("DCR MULTI-PROVIDER ANALYSIS")
    print("="*70 + "\n")

    # Load results
    results = load_dcr_results(num_questions=args.num_questions)

    # Generate analysis
    save_analysis(results)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  • Review dcr_analysis/summary_report.txt")
    print("  • Check dcr_analysis/*.csv for detailed tables")
    print("  • Use results for paper figures")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
