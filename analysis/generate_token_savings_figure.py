#!/usr/bin/env python3
"""
Generate token savings by provider figure for the DCR paper
Shows percentage token reduction across providers for MLP and RoBERTa routers
"""

import matplotlib.pyplot as plt
import numpy as np
import json

# Set publication-quality parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300


def create_token_savings_by_provider_chart():
    """Create token savings by provider bar chart"""

    # Load token savings data
    with open('dcr_analysis/token_savings.json') as f:
        data = json.load(f)

    providers = ['OPENAI', 'GEMINI', 'CLAUDE']

    # Extract savings percentages
    simple_neural_pct = [
        data['openai']['simple_neural_savings_pct'],
        data['gemini']['simple_neural_savings_pct'],
        data['claude']['simple_neural_savings_pct']
    ]

    roberta_pct = [
        data['openai']['roberta_savings_pct'],
        data['gemini']['roberta_savings_pct'],
        data['claude']['roberta_savings_pct']
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(providers))
    width = 0.35

    bars1 = ax.bar(x - width/2, simple_neural_pct, width, label='Simple Neural MLP',
                   color='#2E7D32', edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width/2, roberta_pct, width, label='RoBERTa Transformer',
                   color='#F9A825', edgecolor='black', linewidth=0.7)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Provider', fontweight='bold')
    ax.set_ylabel('Token Reduction (%)', fontweight='bold')
    ax.set_title('Token Savings by Provider\n(vs. Always-Verbose Baseline)',
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(providers)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 40)

    plt.tight_layout()
    plt.savefig('paper/figures/token_savings_by_provider.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('paper/figures/token_savings_by_provider.png', bbox_inches='tight', dpi=300)
    print("✓ Created token_savings_by_provider.pdf/png")
    plt.close()


if __name__ == '__main__':
    print("Generating token savings by provider figure...")
    print("=" * 60)
    create_token_savings_by_provider_chart()
    print("=" * 60)
    print("✅ Token savings figure generated successfully!")
    print("\nGenerated files:")
    print("  - paper/figures/token_savings_by_provider.pdf/png")
