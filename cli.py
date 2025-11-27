#!/usr/bin/env python3
"""
DCR Research CLI - Unified Command Line Interface

Usage:
    python3 cli.py data setup
    python3 cli.py train mlp
    python3 cli.py train transformer
    python3 cli.py infer --num-questions 1000
    python3 cli.py analyze
    python3 cli.py s3 upload models
    python3 cli.py s3 download results
"""

import os
import sys
import subprocess
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def s3_operation(operation, target, bucket_name=None):
    """Handle S3 upload/download operations"""
    from data_prep.s3 import S3Manager

    # Get bucket name from env if not provided
    if bucket_name is None:
        bucket_name = os.getenv('S3_BUCKET_NAME')
        if not bucket_name:
            print("❌ Error: S3_BUCKET_NAME not set in .env file")
            return False

    try:
        s3_manager = S3Manager(bucket_name=bucket_name)

        # Define local and S3 paths
        targets = {
            'data': ('data', 'dcr_research/data'),
            'models': ('models', 'dcr_research/models'),
            'results': ('dcr_results', 'dcr_research/dcr_results'),
            'analysis': ('dcr_analysis', 'dcr_research/dcr_analysis'),
        }

        if target not in targets:
            print(f"❌ Error: Unknown target '{target}'. Choose: {', '.join(targets.keys())}")
            return False

        local_dir, s3_prefix = targets[target]

        if operation == 'upload':
            if not os.path.exists(local_dir):
                print(f"❌ Error: Local directory '{local_dir}' not found")
                return False

            print(f"☁️  Uploading {local_dir}/ to s3://{bucket_name}/{s3_prefix}/")
            count = s3_manager.upload_directory(local_dir, s3_prefix)
            print(f"✅ Uploaded {count} files")
            return True

        elif operation == 'download':
            print(f"☁️  Downloading s3://{bucket_name}/{s3_prefix}/ to {local_dir}/")
            count = s3_manager.download_directory(s3_prefix, local_dir)
            print(f"✅ Downloaded {count} files")
            return True

    except Exception as e:
        print(f"❌ S3 operation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='DCR Research - Dynamic Compute Routing CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Data & Training
  python3 cli.py data setup                    # Setup MMLU data
  python3 cli.py train mlp                     # Train MLP router
  python3 cli.py train transformer             # Train RoBERTa router

  # Inference & Analysis
  python3 cli.py infer --num-questions 100     # Run inference (real-time)
  python3 cli.py batch submit --num-questions 1000  # Submit batch (50% savings)
  python3 cli.py batch status --tracker <file>      # Check batch status
  python3 cli.py batch download --tracker <file>    # Download batch results
  python3 cli.py evaluate                      # Evaluate router accuracy
  python3 cli.py merge --num-questions 1000    # Merge provider results
  python3 cli.py analyze                       # Analyze results

  # S3 Sync
  python3 cli.py s3 upload data                # Upload data to S3
  python3 cli.py s3 upload models              # Upload trained models to S3
  python3 cli.py s3 download models            # Download models from S3
  python3 cli.py s3 download results           # Download inference results from S3
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Data command
    data_parser = subparsers.add_parser('data', help='Data operations')
    data_parser.add_argument('action', choices=['setup'], help='Data action')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train routers')
    train_parser.add_argument('model', choices=['mlp', 'transformer'], help='Model to train')

    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--num-questions', type=int, default=100, help='Number of questions')
    infer_parser.add_argument('--strategy', choices=['baseline', 'simple_neural', 'roberta', 'all'],
                            default='all', help='Routing strategy')
    infer_parser.add_argument('--provider', choices=['openai', 'gemini', 'claude', 'all'],
                            default='all', help='LLM provider')

    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge provider results into combined file')
    merge_parser.add_argument('--num-questions', type=int, required=True, help='Number of questions')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate router accuracy')
    evaluate_parser.add_argument('--num-questions', type=int, help='Number of questions to evaluate')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('--num-questions', type=int, help='Number of questions to analyze')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch inference (50% cost savings)')
    batch_subparsers = batch_parser.add_subparsers(dest='batch_action', help='Batch action')

    batch_submit = batch_subparsers.add_parser('submit', help='Submit batch jobs')
    batch_submit.add_argument('--num-questions', type=int, required=True, help='Number of questions')

    batch_status = batch_subparsers.add_parser('status', help='Check batch status')
    batch_status.add_argument('--tracker', type=str, required=True, help='Tracker filename')

    batch_download = batch_subparsers.add_parser('download', help='Download batch results')
    batch_download.add_argument('--tracker', type=str, required=True, help='Tracker filename')

    # S3 command
    s3_parser = subparsers.add_parser('s3', help='S3 upload/download operations')
    s3_parser.add_argument('operation', choices=['upload', 'download'], help='S3 operation')
    s3_parser.add_argument('target', choices=['data', 'models', 'results', 'analysis'],
                          help='What to upload/download')
    s3_parser.add_argument('--bucket', type=str, help='S3 bucket name (default: from .env)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute commands
    if args.command == 'data':
        if args.action == 'setup':
            print("📚 Setting up MMLU data...")
            subprocess.run([sys.executable, 'data_prep/data_setup.py'])

    elif args.command == 'train':
        if args.model == 'mlp':
            print("🔧 Training MLP router...")
            subprocess.run([sys.executable, 'training/train.py'])
        elif args.model == 'transformer':
            print("🔧 Training transformer router...")
            subprocess.run([sys.executable, 'training/train_transformer.py'])

    elif args.command == 'infer':
        print(f"🚀 Running inference on {args.num_questions} questions...")
        cmd = [sys.executable, 'inference/inference.py', '--num-questions', str(args.num_questions)]
        if args.strategy != 'all':
            cmd.extend(['--strategy', args.strategy])
        if args.provider != 'all':
            cmd.extend(['--provider', args.provider])
        subprocess.run(cmd)

    elif args.command == 'merge':
        print(f"🔀 Merging provider results for {args.num_questions} questions...")
        subprocess.run([sys.executable, 'inference/merge_results.py', str(args.num_questions)])

    elif args.command == 'evaluate':
        print("📊 Evaluating router accuracy...")
        cmd = [sys.executable, 'inference/evaluate_routers.py']
        if args.num_questions:
            cmd.extend(['--num-questions', str(args.num_questions)])
        subprocess.run(cmd)

    elif args.command == 'analyze':
        print("📊 Analyzing results...")
        cmd = [sys.executable, 'analysis/analyze.py']
        if args.num_questions:
            cmd.extend(['--num-questions', str(args.num_questions)])
        subprocess.run(cmd)

    elif args.command == 'batch':
        if not args.batch_action:
            print("❌ Error: Please specify batch action (submit|status|download)")
            return

        if args.batch_action == 'submit':
            print(f"📤 Submitting batch jobs for {args.num_questions} questions...")
            subprocess.run([sys.executable, 'inference/batch_inference.py', 'submit',
                          '--num-questions', str(args.num_questions)])
        elif args.batch_action == 'status':
            print(f"📊 Checking batch status...")
            subprocess.run([sys.executable, 'inference/batch_inference.py', 'status',
                          '--tracker', args.tracker])
        elif args.batch_action == 'download':
            print(f"📥 Downloading batch results...")
            subprocess.run([sys.executable, 'inference/batch_inference.py', 'download',
                          '--tracker', args.tracker])

    elif args.command == 's3':
        s3_operation(args.operation, args.target, args.bucket)


if __name__ == '__main__':
    main()
