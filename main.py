"""
Main Entry Point (Dispatcher).
Splits execution into Generation (simulation) and Evaluation (measurement/plotting).

Usage:
    python main.py --stage generation --mode baseline
    python main.py --stage evaluation --mode baseline
"""

import argparse
import sys
import os

# Force matplotlib to use Agg backend by setting env var before importing it
# This is often more robust than matplotlib.use() if imports happen early
os.environ["MPLBACKEND"] = "Agg"

from dotenv import load_dotenv

# Set matplotlib backend to Agg to avoid Tkinter thread issues
import matplotlib
matplotlib.use('Agg')

# Import workflow modules
import workflow_generation
import workflow_eval
from config import API_KEY, API_PROVIDER

import matplotlib.pyplot as plt
print(f"DEBUG: Matplotlib backend set to: {plt.get_backend()}")

def main():
    parser = argparse.ArgumentParser(description="Semantic Opinion Dynamics")
    parser.add_argument("--stage", choices=["generation", "evaluation"], required=True, help="Workflow stage")
    parser.add_argument("--mode", choices=["baseline", "intervention", "comparison", "degroot"], required=True, help="Experiment mode")
    parser.add_argument("--api-key", type=str, default=None, help="API Key for generation")

    # Check if user just ran python main.py without args (print help)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    
    # Load env for API key if generating
    load_dotenv()

    if args.stage == "generation":
        # Resolve API Key
        api_key = args.api_key or API_KEY
        if not api_key:
            if API_PROVIDER == "anthropic": api_key = os.getenv("ANTHROPIC_API_KEY")
            elif API_PROVIDER == "deepseek": api_key = os.getenv("DEEPSEEK_API_KEY")
            elif API_PROVIDER == "openai": api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            print("Error: API Key missing. Please provide via --api-key or .env")
            return

        if args.mode == "baseline":
            workflow_generation.generate_baseline(api_key)
        elif args.mode == "intervention":
            workflow_generation.generate_intervention(api_key)
        elif args.mode == "comparison":
            workflow_generation.generate_topology(api_key)
        elif args.mode == "degroot":
            workflow_generation.generate_degroot(api_key)

    elif args.stage == "evaluation":
        if args.mode == "baseline":
            workflow_eval.eval_baseline()
        elif args.mode == "intervention":
            workflow_eval.eval_intervention()
        elif args.mode == "comparison":
            workflow_eval.eval_topology()
        elif args.mode == "degroot":
            workflow_eval.eval_degroot()

if __name__ == "__main__":
    main()