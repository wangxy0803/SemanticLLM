"""
Main Entry Point (Dispatcher) - IMPROVED PROMPTS VERSION
Splits execution into Generation (simulation) and Evaluation (measurement/plotting).

This version uses improved prompts for realistic, diverse, opinionated discourse.

Usage:
    python main.py --stage generation --mode baseline
    python main.py --stage evaluation --mode baseline
"""

import argparse
import sys
import os

# Force matplotlib to use Agg backend
os.environ["MPLBACKEND"] = "Agg"

from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')

# Import workflow modules  
import workflow_generation
import workflow_eval
import workflow_visualization
from config import API_KEY, API_PROVIDER

import matplotlib.pyplot as plt
print(f"DEBUG: Matplotlib backend set to: {plt.get_backend()}")

def print_banner():
    """Print fancy banner for improved version."""
    print("\n" + "="*80)
    print("🎭 SEMANTIC OPINION DYNAMICS - IMPROVED PROMPTS VERSION")
    print("="*80)
    print("✨ Features:")
    print("   • Opinionated personas with extreme traits")
    print("   • Emotional, informal language (like real people)")
    print("   • Personal experiences and cognitive biases")
    print("   • Varied linguistic styles (no template repetition)")
    print("   • Higher temperature (1.0) for maximum diversity")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Semantic Opinion Dynamics - Improved Prompts for Realistic Discourse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage generation --mode baseline
  python main.py --stage evaluation --mode baseline
  python main.py --stage visualization --mode baseline
  
Before running, make sure you've generated personas with improved prompts:
  python persona_generation.py
        """
    )
    parser.add_argument("--stage", choices=["generation", "evaluation", "visualization"], required=True, 
                       help="Workflow stage")
    parser.add_argument("--mode", choices=["baseline", "intervention", "comparison"], required=True,
                       help="Experiment mode")
    parser.add_argument("--api-key", type=str, default=None, 
                       help="API Key for generation")

    # Print help if no args
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    
    print_banner()
    
    load_dotenv()

    if args.stage == "generation":
        print("🚀 GENERATION STAGE - Running simulations with improved prompts")
        print("   This will use API calls and cost money\n")
        
        # Resolve API Key
        api_key = args.api_key or API_KEY
        if not api_key:
            if API_PROVIDER == "anthropic": api_key = os.getenv("ANTHROPIC_API_KEY")
            elif API_PROVIDER == "deepseek": api_key = os.getenv("DEEPSEEK_API_KEY")
            elif API_PROVIDER == "openai": api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            print("❌ Error: API Key missing. Please provide via --api-key or .env")
            print("\nCreate a .env file with:")
            print("  DEEPSEEK_API_KEY=your-key-here")
            print("  # or ANTHROPIC_API_KEY=your-key-here")
            print("  # or OPENAI_API_KEY=your-key-here")
            return

        if args.mode == "baseline":
            workflow_generation.generate_baseline(api_key)
        elif args.mode == "intervention":
            workflow_generation.generate_intervention(api_key)
        elif args.mode == "comparison":
            workflow_generation.generate_topology(api_key)

    elif args.stage == "evaluation":
        print("📊 EVALUATION STAGE - Analyzing results (FREE - no API calls)")
        print("   Loading saved histories and computing metrics\n")
        
        if args.mode == "baseline":
            workflow_eval.eval_baseline()
        elif args.mode == "intervention":
            workflow_eval.eval_intervention()
        elif args.mode == "comparison":
            workflow_eval.eval_topology()
    
    elif args.stage == "visualization":
        workflow_visualization.run_animated_network_evolution(args.mode)

    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()