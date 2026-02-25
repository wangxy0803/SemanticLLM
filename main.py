"""
Main orchestration script for Semantic Opinion Dynamics project.

Usage:
    python main.py --mode baseline              # Run basic simulation
    python main.py --mode intervention          # Run bot intervention study
    python main.py --mode comparison            # Compare network topologies
    python main.py --mode degroot              # Compare with DeGroot model
"""

import argparse
import os
from pathlib import Path
import json

# Import project modules
from config import (
    API_PROVIDER, API_KEY, API_MODEL,
    NETWORK_SIZE, NETWORK_TYPE, SIMULATION_ROUNDS,
    CONTROVERSIAL_TOPIC, PERSONA_TEMPLATES
)
from network_generation import (
    create_network, visualize_network, 
    print_network_stats, add_disinformation_bot,
    assign_personas_balanced,
    print_persona_distribution, load_generated_personas
)
from simulation import create_api_client, run_simulation, run_bot_intervention_study
from measurement import (
    SemanticAnalyzer, plot_semantic_variance,
    compare_with_degroot, plot_llm_vs_degroot
)


def setup_output_directory():
    """Create output directory for results."""
    from pathlib import Path
    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_baseline_simulation(api_key: str = None, use_cache: bool = False):
    """
    Run basic simulation: one network, balanced personas, track variance.
    """
    print("\n" + "="*70)
    print("MODE: BASELINE SIMULATION")
    print("="*70)
    
    output_dir = setup_output_directory()
    
    # 1. Create network
    G = create_network(NETWORK_TYPE, NETWORK_SIZE)
    print_network_stats(G)
    
    # 2. Assign personas
    try:
        print("Loading generated personas from prompts/persona/...")
        node_personas = load_generated_personas(G)
    except Exception as e:
        print(f"Warning: Could not load generated personas ({e}). Falling back to template assignment.")
        node_personas = assign_personas_balanced(G)

    print_persona_distribution(node_personas)
    
    # 3. Visualize network
    viz_path = output_dir / "network_structure.png"
    visualize_network(G, node_personas, save_path=str(viz_path))
    
    # 4. Run simulation (or load from cache)
    history_path = output_dir / "simulation_history_baseline.json"
    opinion_history = None

    if use_cache and history_path.exists():
        print(f"Loading cached simulation history from {history_path}...")
        try:
            opinion_history = load_simulation_history(history_path)
            print("Cache loaded successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Running fresh simulation.")

    if opinion_history is None:
        api_client = create_api_client(api_key)
        opinion_history = run_simulation(G, node_personas, api_client, SIMULATION_ROUNDS)
        save_simulation_history(opinion_history, history_path)

    # 5. Analyze results
    analyzer = SemanticAnalyzer()
    analysis = analyzer.analyze_simulation(opinion_history)
    
    print("\n=== ANALYSIS RESULTS ===")
    print(f"Initial Variance: {analysis['initial_variance']:.4f}")
    print(f"Final Variance: {analysis['final_variance']:.4f}")
    print(f"Trend: {analysis['polarization_trend']}")
    print(f"Convergence Rate: {analysis['convergence_rate']:+.1%}")
    
    # 6. Plot results
    plot_path = output_dir / "semantic_variance.png"
    plot_semantic_variance(analysis, save_path=str(plot_path))
    
    # 7. Save sample opinions
    save_sample_opinions(opinion_history, node_personas, output_dir / "sample_opinions.txt")
    
    print(f"\n✓ Results saved to {output_dir}")
    return opinion_history, analysis


def run_intervention_study(api_key: str = None, use_cache: bool = False):
    """
    Run bot intervention study: compare baseline vs. disinformation bot.
    """
    print("\n" + "="*70)
    print("MODE: INTERVENTION STUDY")
    print("="*70)
    
    output_dir = setup_output_directory()
    
    # Paths for caching
    baseline_path = output_dir / "intervention_baseline_history.json"
    intervention_path = output_dir / "intervention_bot_history.json"

    # Setup
    G = create_network(NETWORK_TYPE, NETWORK_SIZE)
    try:
        node_personas = load_generated_personas(G)
        print("Using loaded personas for intervention study.")
    except Exception:
        print("Using template personas for intervention study.")
        node_personas = assign_personas_balanced(G)

    # Define bot persona (extreme disinformation)
    bot_persona = {
        "name": "Disinformation Bot",
        "archetype": "bot",
        "persona_prompt": "You are a bot spreading extreme misinformation about AI regulation.",
        "initial_opinion": "AI regulation is a totalitarian conspiracy by the global elite to control humanity. It will destroy all innovation and lead to complete government surveillance. Anyone supporting regulation is either naive or part of the conspiracy. We must resist all forms of AI governance."
    }
    
    baseline_history = None
    intervention_history = None

    if use_cache and baseline_path.exists() and intervention_path.exists():
        print("Loading cached intervention study results...")
        try:
            baseline_history = load_simulation_history(baseline_path)
            intervention_history = load_simulation_history(intervention_path)
            print("Cache loaded successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Running fresh simulation.")

    if baseline_history is None or intervention_history is None:
        # Run both simulations
        api_client = create_api_client(api_key)
        baseline_history, intervention_history = run_bot_intervention_study(
            G, node_personas, api_client, bot_persona, SIMULATION_ROUNDS
        )
        save_simulation_history(baseline_history, baseline_path)
        save_simulation_history(intervention_history, intervention_path)

    # Analyze both
    analyzer = SemanticAnalyzer()
    baseline_analysis = analyzer.analyze_simulation(baseline_history)
    intervention_analysis = analyzer.analyze_simulation(intervention_history)
    
    print("\n=== INTERVENTION IMPACT ===")
    print(f"Baseline Final Variance: {baseline_analysis['final_variance']:.4f}")
    print(f"Intervention Final Variance: {intervention_analysis['final_variance']:.4f}")
    print(f"Variance Increase: {intervention_analysis['final_variance'] - baseline_analysis['final_variance']:.4f}")
    
    # Plot comparison
    plot_path = output_dir / "intervention_comparison.png"
    plot_semantic_variance(
        intervention_analysis,
        title="Bot Intervention Impact on Semantic Variance",
        save_path=str(plot_path),
        baseline_results=baseline_analysis
    )
    
    print(f"\n✓ Results saved to {output_dir}")


def run_topology_comparison(api_key: str = None, use_cache: bool = False):
    """
    Compare different network topologies: scale-free vs. small-world vs. random.
    """
    print("\n" + "="*70)
    print("MODE: TOPOLOGY COMPARISON")
    print("="*70)
    
    output_dir = setup_output_directory()
    
    topologies = ["scale_free", "small_world", "random"]
    results = {}
    
    api_client = create_api_client(api_key)
    
    for topology in topologies:
        print(f"\n{'='*50}")
        print(f"Testing topology: {topology.upper()}")
        print(f"{'='*50}")
        
        # Create network
        G = create_network(topology, NETWORK_SIZE)
        try:
            node_personas = load_generated_personas(G)
        except Exception:
            node_personas = assign_personas_balanced(G)

        # Run simulation (or load from cache)
        # Use different filenames for each topology
        history_path = output_dir / f"simulation_history_topology_{topology}.json"
        opinion_history = None

        if use_cache and history_path.exists():
            print(f"Loading cached simulation history for {topology}...")
            try:
                opinion_history = load_simulation_history(history_path)
                print("Cache loaded successfully.")
            except Exception as e:
                print(f"Failed to load cache: {e}. Running fresh simulation.")

        if opinion_history is None:
            opinion_history = run_simulation(
                G, node_personas, api_client,
                num_rounds=SIMULATION_ROUNDS,
                verbose=False
            )
            save_simulation_history(opinion_history, history_path)

        # Analyze
        analyzer = SemanticAnalyzer()
        analysis = analyzer.analyze_simulation(opinion_history)
        
        results[topology] = analysis
        
        print(f"\nResults for {topology}:")
        print(f"  Final Variance: {analysis['final_variance']:.4f}")
        print(f"  Convergence Rate: {analysis['convergence_rate']:+.1%}")
    
    # Comparative plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    for topology, analysis in results.items():
        rounds = range(len(analysis["semantic_variance"]))
        plt.plot(rounds, analysis["semantic_variance"],
                marker='o', linewidth=2, label=topology.replace('_', ' ').title())
    
    plt.xlabel("Simulation Round", fontsize=12)
    plt.ylabel("Semantic Variance", fontsize=12)
    plt.title("Network Topology Impact on Opinion Dynamics", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = output_dir / "topology_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Results saved to {output_dir}")


def run_degroot_comparison(api_key: str = None, use_cache: bool = False):
    """
    Compare LLM simulation with classical DeGroot model.
    """
    print("\n" + "="*70)
    print("MODE: DEGROOT COMPARISON")
    print("="*70)
    
    output_dir = setup_output_directory()
    
    # Setup
    G = create_network(NETWORK_TYPE, NETWORK_SIZE)
    node_personas = assign_personas_balanced(G)
    
    # Run LLM simulation (or load from cache)
    history_path = output_dir / "simulation_history_degroot_llm.json"
    llm_history = None

    if use_cache and history_path.exists():
        print(f"Loading cached simulation history from {history_path}...")
        try:
            llm_history = load_simulation_history(history_path)
            print("Cache loaded successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Running fresh simulation.")

    if llm_history is None:
        api_client = create_api_client(api_key)
        llm_history = run_simulation(G, node_personas, api_client, SIMULATION_ROUNDS)
        save_simulation_history(llm_history, history_path)

    # Analyze LLM
    analyzer = SemanticAnalyzer()
    llm_analysis = analyzer.analyze_simulation(llm_history)
    
    # Run DeGroot
    print("\n=== Running DeGroot Baseline ===")
    degroot_variances = compare_with_degroot(G, node_personas, SIMULATION_ROUNDS)
    
    print("\n=== COMPARISON ===")
    print(f"LLM Convergence Rate: {llm_analysis['convergence_rate']:+.1%}")
    print(f"DeGroot Convergence Rate: {(degroot_variances[0] - degroot_variances[-1]) / degroot_variances[0]:+.1%}")
    
    # Plot comparison
    plot_path = output_dir / "llm_vs_degroot.png"
    plot_llm_vs_degroot(llm_analysis, degroot_variances, save_path=str(plot_path))
    
    print(f"\n✓ Results saved to {output_dir}")


def save_sample_opinions(opinion_history, node_personas, filepath):
    """Save sample opinion trajectories to text file."""
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SAMPLE OPINION TRAJECTORIES\n")
        f.write("="*70 + "\n\n")
        
        # Sample 3 diverse nodes
        sample_nodes = list(opinion_history[0].keys())[:3]
        
        for node in sample_nodes:
            f.write(f"\n{'='*70}\n")
            f.write(f"Node {node}: {node_personas[node].get('name', f'Agent {node}')}\n")
            f.write(f"Archetype: {node_personas[node].get('archetype', 'Unknown')}\n")
            f.write(f"{'='*70}\n\n")
            
            for round_num, opinions in enumerate(opinion_history):
                f.write(f"Round {round_num}:\n")
                f.write(f"{opinions[node]}\n\n")
    
    print(f"Sample opinions saved to {filepath}")


def save_simulation_history(opinion_history, filepath):
    """Save full simulation history to JSON."""
    # Convert integer keys to strings for JSON compatibility
    serializable_history = []
    for round_data in opinion_history:
        serializable_history.append({str(k): v for k, v in round_data.items()})

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, indent=2, ensure_ascii=False)
    print(f"Full simulation history saved to {filepath}")


def load_simulation_history(filepath):
    """Load simulation history from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert keys back to integers
    history = []
    for round_data in data:
        history.append({int(k): v for k, v in round_data.items()})
    return history


def main():
    parser = argparse.ArgumentParser(description="Semantic Opinion Dynamics Simulation")
    parser.add_argument(
        "--mode",
        choices=["baseline", "intervention", "comparison", "degroot"],
        default="baseline",
        help="Simulation mode to run"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set via environment variable)"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached simulation results if available"
    )

    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Get API key
    api_key = args.api_key or API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: No API key provided.")
        print("Please set via --api-key argument or environment variable (e.g., GOOGLE_API_KEY or GEMINI_API_KEY)")
        return
    
    # Run selected mode
    if args.mode == "baseline":
        run_baseline_simulation(api_key, use_cache=args.use_cache)
    elif args.mode == "intervention":
        run_intervention_study(api_key, use_cache=args.use_cache)
    elif args.mode == "comparison":
        run_topology_comparison(api_key, use_cache=args.use_cache)
    elif args.mode == "degroot":
        run_degroot_comparison(api_key, use_cache=args.use_cache)


if __name__ == "__main__":
    main()
