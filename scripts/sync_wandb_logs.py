#!/usr/bin/env python3
"""
Script to parse validation metrics from wandb output.log and upload to wandb.

Usage:
    python sync_wandb_logs.py <wandb_run_dir> [--step STEP]
    python sync_wandb_logs.py /tmp/ray/.../wandb/run-xxx --step 150

The script will:
1. Parse the output.log file to extract metrics
2. Resume the wandb run
3. Upload the metrics to wandb
"""

import argparse
import os
import re
import json
import sys


def parse_final_validation_metrics(output_log_path: str) -> dict:
    """Parse Final validation metrics from output.log file."""
    with open(output_log_path, 'r') as f:
        content = f.read()
    
    # Find the "Final validation metrics" section
    match = re.search(r'"Final validation metrics: ({.*)', content, re.DOTALL)
    if not match:
        print(f"Warning: Could not find 'Final validation metrics' in {output_log_path}")
        return {}
    
    # The metrics are spread across multiple lines with quotes and newlines
    # We need to reconstruct the dict string
    metrics_str = match.group(1)
    
    # Clean up the string: remove newlines, extra quotes, and spaces
    # The format is like: "'key': value, \n \n"
    metrics_str = re.sub(r'\n\s*\n', '', metrics_str)
    metrics_str = re.sub(r'"\s*\'', "'", metrics_str)
    metrics_str = re.sub(r'\'\s*"', "'", metrics_str)
    metrics_str = metrics_str.strip()
    
    # Remove trailing incomplete parts
    if metrics_str.endswith(','):
        metrics_str = metrics_str[:-1]
    if not metrics_str.endswith('}'):
        # Find the last complete key-value pair
        last_complete = metrics_str.rfind("'}")
        if last_complete == -1:
            last_complete = metrics_str.rfind("}")
        if last_complete > 0:
            metrics_str = metrics_str[:last_complete+1]
    
    # Try to parse as Python dict
    try:
        # Replace single quotes with double quotes for JSON parsing
        # But be careful with nested quotes
        metrics_dict = eval(metrics_str)
        return metrics_dict
    except Exception as e:
        print(f"Warning: Could not parse metrics dict: {e}")
        # Fallback: extract key-value pairs using regex
        return parse_metrics_regex(content)


def parse_metrics_regex(content: str) -> dict:
    """Fallback: parse metrics using regex patterns."""
    metrics = {}
    
    # Find all val-core metrics (these are the important ones)
    # Pattern: 'val-core/xxx': value or "val-core/xxx": value
    pattern = r"['\"]?(val-core/[^'\"]+)['\"]?\s*:\s*([0-9.-]+)"
    matches = re.findall(pattern, content)
    
    for key, value in matches:
        try:
            metrics[key] = float(value)
        except ValueError:
            pass
    
    # Also extract val-aux metrics if needed
    pattern = r"['\"]?(val-aux/[^'\"]+)['\"]?\s*:\s*([0-9.-]+)"
    matches = re.findall(pattern, content)
    
    for key, value in matches:
        try:
            metrics[key] = float(value)
        except ValueError:
            pass
    
    return metrics


def parse_step_metrics(output_log_path: str, target_step: int = None) -> dict:
    """Parse metrics for a specific step from output.log.
    
    If target_step is None, returns the last logged metrics (Final validation).
    """
    with open(output_log_path, 'r') as f:
        content = f.read()
    
    if target_step is None:
        # Return Final validation metrics
        return parse_final_validation_metrics(output_log_path)
    
    # For specific step, we need to find the metrics logged at that step
    # This is more complex as the log format varies
    # For now, if target_step is the last step, return final validation metrics
    return parse_final_validation_metrics(output_log_path)


def get_run_id_from_dir(wandb_run_dir: str) -> str:
    """Extract wandb run ID from directory name."""
    # Directory name format: run-YYYYMMDD_HHMMSS-RUNID
    dirname = os.path.basename(wandb_run_dir)
    match = re.search(r'run-\d+_\d+-(\w+)', dirname)
    if match:
        return match.group(1)
    return None


def upload_to_wandb(run_id: str, project: str, metrics: dict, step: int):
    """Upload metrics to wandb by resuming the run."""
    import wandb
    
    print(f"Resuming wandb run {run_id} in project {project}...")
    run = wandb.init(
        project=project,
        id=run_id,
        resume="must"
    )
    
    print(f"Logging {len(metrics)} metrics at step {step}...")
    wandb.log(metrics, step=step)
    
    print("Finishing wandb run...")
    wandb.finish(exit_code=0)
    
    print(f"Done! Metrics uploaded to {run.url}")


def main():
    parser = argparse.ArgumentParser(description='Sync wandb logs from local output.log to wandb server')
    parser.add_argument('wandb_run_dir', help='Path to wandb run directory (e.g., /tmp/ray/.../wandb/run-xxx)')
    parser.add_argument('--step', type=int, default=None, help='Step number to log (default: auto-detect from log)')
    parser.add_argument('--project', type=str, default=None, help='Wandb project name (default: auto-detect)')
    parser.add_argument('--dry-run', action='store_true', help='Parse and print metrics without uploading')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.wandb_run_dir):
        print(f"Error: {args.wandb_run_dir} is not a directory")
        sys.exit(1)
    
    output_log = os.path.join(args.wandb_run_dir, 'files', 'output.log')
    if not os.path.exists(output_log):
        print(f"Error: {output_log} not found")
        sys.exit(1)
    
    # Get run ID
    run_id = get_run_id_from_dir(args.wandb_run_dir)
    if not run_id:
        print(f"Error: Could not extract run ID from {args.wandb_run_dir}")
        sys.exit(1)
    
    print(f"Run ID: {run_id}")
    print(f"Output log: {output_log}")
    
    # Parse metrics
    print("Parsing metrics from output.log...")
    metrics = parse_step_metrics(output_log, args.step)
    
    if not metrics:
        print("Error: No metrics found in output.log")
        sys.exit(1)
    
    # Filter to only val-core metrics for cleaner output
    core_metrics = {k: v for k, v in metrics.items() if 'val-core' in k}
    
    print(f"\nFound {len(metrics)} total metrics, {len(core_metrics)} core metrics")
    print("\nCore metrics (val-core):")
    for k, v in sorted(core_metrics.items()):
        print(f"  {k}: {v}")
    
    if args.dry_run:
        print("\n[Dry run] Not uploading to wandb")
        return
    
    # Detect project from metadata if not specified
    project = args.project
    if not project:
        metadata_file = os.path.join(args.wandb_run_dir, 'files', 'wandb-metadata.json')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                # Project might be in config
                print("Trying to detect project from metadata...")
            except:
                pass
        
        if not project:
            # Default project name
            project = "SCARL"
            print(f"Using default project: {project}")
    
    # Detect step if not specified
    step = args.step
    if step is None:
        # Try to detect from log - look for "Training Progress: XXX/YYY"
        with open(output_log, 'r') as f:
            content = f.read()
        match = re.search(r'Training Progress:\s*\d+%.*?(\d+)/(\d+)', content)
        if match:
            step = int(match.group(2))  # Total steps
            print(f"Auto-detected step: {step}")
        else:
            step = 150  # Default
            print(f"Using default step: {step}")
    
    # Upload to wandb
    upload_to_wandb(run_id, project, metrics, step)


if __name__ == '__main__':
    main()