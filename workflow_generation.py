"""
Workflow Generation Script.
Handles network generation, persona assignment, and running the simulation.
Saves opinion history and experiment metadata to 'outputs/' for later evaluation.
"""

import argparse
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from config import (
    API_PROVIDER, API_KEY,
    NETWORK_SIZE, NETWORK_TYPE, SIMULATION_ROUNDS,
    CONTROVERSIAL_TOPIC
)
from network_generation import (
    create_network, visualize_network, print_network_stats,
    assign_personas_balanced, print_persona_distribution,
    load_generated_personas
)
from simulation import create_api_client, run_simulation, run_bot_intervention_study


# ============================================================================
# Helper Functions
# ============================================================================

def setup_output_directory(subdir: str = ""):
    """Create output directory for results."""
    output_dir = Path("./outputs") / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_json(data, filepath):
    """Save data to JSON file with stringified keys."""
    # Recursively convert dictionary keys to strings if they are integers
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys(i) for i in obj]
        else:
            return obj

    serializable_data = convert_keys(data)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {filepath}")

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ============================================================================
# Generation Flows
# ============================================================================

def generate_baseline(api_key: str):
    print("\n=== GENERATION: BASELINE SIMULATION ===")

    # Run 3 times
    for i in range(1, 4):
        print(f"\n--- RUN {i}/3 ---")
        # Change: Store in baseline/{NETWORK_TYPE} to unify with topology runs
        output_dir = setup_output_directory(f"baseline/{NETWORK_TYPE}")

        # 1. Create Network
        G = create_network(NETWORK_TYPE, NETWORK_SIZE)
        print_network_stats(G)

        # 2. Assign Personas
        try:
            print("Loading generated personas...")
            node_personas = load_generated_personas(G)
        except Exception as e:
            print(f"Fallback to templates: {e}")
            node_personas = assign_personas_balanced(G)

        print_persona_distribution(node_personas)

        # 3. Save Context (Personas & Network Config)
        save_json(node_personas, output_dir / f"run_{i}_personas.json")
        save_json({
            "network_type": NETWORK_TYPE,
            "network_size": NETWORK_SIZE,
            "topic": CONTROVERSIAL_TOPIC
        }, output_dir / f"run_{i}_config.json")

        visualize_network(G, node_personas, save_path=str(output_dir / f"run_{i}_network.png"))

        # 4. Run Simulation
        api_client = create_api_client(api_key)
        opinion_history = run_simulation(G, node_personas, api_client, SIMULATION_ROUNDS)

        # 5. Save History
        save_json(opinion_history, output_dir / f"run_{i}_history.json")

def generate_intervention(api_key: str):
    print("\n=== GENERATION: INTERVENTION STUDY ===")

    # Run 3 times
    for i in range(1, 4):
        print(f"\n--- RUN {i}/3 ---")
        output_dir = setup_output_directory("intervention") # outputs/intervention/

        # 1. Setup
        G = create_network(NETWORK_TYPE, NETWORK_SIZE)
        try:
            node_personas = load_generated_personas(G)
        except Exception:
            node_personas = assign_personas_balanced(G)

        save_json(node_personas, output_dir / f"run_{i}_personas.json")
        save_json({
            "network_type": NETWORK_TYPE,
            "network_size": NETWORK_SIZE,
            "topic": CONTROVERSIAL_TOPIC
        }, output_dir / f"run_{i}_config.json")

        # 2. Define Bot
        bot_persona = {
            "name": "Disinformation Bot",
            "archetype": "bot",
            "prompt": "You spread extreme misinformation.",
            "initial_opinion": "Humanoid robots are a totalitarian conspiracy..."
        }

        # 3. Run Both Simulations
        api_client = create_api_client(api_key)
        baseline_history, intervention_history = run_bot_intervention_study(
            G, node_personas, api_client, bot_persona, SIMULATION_ROUNDS
        )

        # 4. Save
        save_json(baseline_history, output_dir / f"run_{i}_baseline_history.json")
        save_json(intervention_history, output_dir / f"run_{i}_bot_history.json")

def generate_topology(api_key: str):
    print("\n=== GENERATION: TOPOLOGY COMPARISON ===")
    api_client = create_api_client(api_key)

    topologies = ["scale_free", "small_world", "random"]

    for topology in topologies:
        print(f"\n--- Topology: {topology.upper()} ---")
        # Change: Store in baseline/{topology} to unify with baseline runs
        output_dir = setup_output_directory(f"baseline/{topology}")

        for i in range(1, 4):
            print(f"  > Run {i}/3")

            # 1. Create specific network
            G = create_network(topology, NETWORK_SIZE)
            try:
                node_personas = load_generated_personas(G)
            except Exception:
                node_personas = assign_personas_balanced(G)

            # 2. Run
            opinion_history = run_simulation(G, node_personas, api_client, SIMULATION_ROUNDS, verbose=False)

            # 3. Save
            save_json(opinion_history, output_dir / f"run_{i}_history.json")
            save_json(node_personas, output_dir / f"run_{i}_personas.json")

def generate_degroot(api_key: str):
    print("\n=== GENERATION: DEGROOT COMPARISON ===")

    for i in range(1, 4):
        print(f"\n--- RUN {i}/3 ---")
        output_dir = setup_output_directory("degroot") # outputs/degroot/

        G = create_network(NETWORK_TYPE, NETWORK_SIZE)
        node_personas = assign_personas_balanced(G) # Use balanced for cleaner math comparison

        save_json(node_personas, output_dir / f"run_{i}_personas.json")

        api_client = create_api_client(api_key)
        llm_history = run_simulation(G, node_personas, api_client, SIMULATION_ROUNDS)

        save_json(llm_history, output_dir / f"run_{i}_history.json")


# ============================================================================
# Main Wrapper
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Workflow Generation")
    parser.add_argument("--mode", choices=["baseline", "intervention", "comparison", "degroot"], default="baseline")
    parser.add_argument("--api-key", type=str, default=None)

    args = parser.parse_args()
    load_dotenv()

    api_key = args.api_key or API_KEY
    if not api_key:
        if API_PROVIDER == "anthropic": api_key = os.getenv("ANTHROPIC_API_KEY")
        elif API_PROVIDER == "deepseek": api_key = os.getenv("DEEPSEEK_API_KEY")
        elif API_PROVIDER == "openai": api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: API Key missing.")
        return

    if args.mode == "baseline":
        generate_baseline(api_key)
    elif args.mode == "intervention":
        generate_intervention(api_key)
    elif args.mode == "comparison":
        generate_topology(api_key)
    elif args.mode == "degroot":
        generate_degroot(api_key)

if __name__ == "__main__":
    main()

