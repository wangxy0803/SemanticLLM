"""
Workflow Generation Script.
Handles network generation, persona assignment, and running the simulation.
Saves opinion history and experiment metadata to 'outputs/' for later evaluation.

COMPATIBLE WITH IMPROVED PROMPTS (persona_agent_improved.py, persona_generation_improved.py)
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
    print_persona_distribution,
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
    print("Using IMPROVED prompts for realistic, diverse discourse")

    # Run 3 times
    for i in range(1, 4):
        print(f"\n--- RUN {i}/3 ---")
        output_dir = setup_output_directory(f"baseline/{NETWORK_TYPE}")

        # 1. Create Network
        G = create_network(NETWORK_TYPE, NETWORK_SIZE)
        print_network_stats(G)

        # 2. Assign Personas
        try:
            print("Loading generated personas...")
            node_personas = load_generated_personas(G)
        except Exception as e:
            print(f"\n❌ Error loading personas: {e}")
            print("\n🔧 SOLUTION:")
            print("   1. Make sure you've generated personas with improved prompts:")
            print("      python persona_generation.py")
            print("   2. Personas should be in: prompts/persona/")
            print("   3. Need at least 50 persona files\n")
            return

        print_persona_distribution(node_personas)

        # 3. Save Context
        save_json(node_personas, output_dir / f"run_{i}_personas.json")
        save_json({
            "network_type": NETWORK_TYPE,
            "network_size": NETWORK_SIZE,
            "topic": CONTROVERSIAL_TOPIC,
            "using_improved_prompts": True
        }, output_dir / f"run_{i}_config.json")

        visualize_network(G, node_personas, save_path=str(output_dir / f"run_{i}_network.png"))

        # 4. Run Simulation
        api_client = create_api_client(api_key)
        opinion_history = run_simulation(G, node_personas, api_client, SIMULATION_ROUNDS)

        # 5. Save History
        save_json(opinion_history, output_dir / f"run_{i}_history.json")
        
        print(f"\n✅ Run {i} completed successfully!")

def generate_intervention(api_key: str):
    print("\n=== GENERATION: INTERVENTION STUDY ===")
    print("Using IMPROVED prompts for realistic discourse")

    # Run 3 times
    for i in range(1, 4):
        print(f"\n--- RUN {i}/3 ---")
        output_dir = setup_output_directory("intervention")

        # 1. Setup
        G = create_network(NETWORK_TYPE, NETWORK_SIZE)
        try:
            node_personas = load_generated_personas(G)
        except Exception as e:
            print(f"\n❌ Error loading personas: {e}")
            print("Please generate personas first: python persona_generation.py")
            return

        save_json(node_personas, output_dir / f"run_{i}_personas.json")
        save_json({
            "network_type": NETWORK_TYPE,
            "network_size": NETWORK_SIZE,
            "topic": CONTROVERSIAL_TOPIC,
            "using_improved_prompts": True
        }, output_dir / f"run_{i}_config.json")

        # 2. Define Bot (with improved, more aggressive persona)
        bot_persona = {
            "name": "Disinformation Bot",
            "archetype": "bot",
            "Background": {
                "exact_age_and_generation": "N/A - Automated Account",
                "occupation": "Propaganda Bot",
                "social_class": "N/A",
                "key_experience": "Programmed to spread extreme misinformation"
            },
            "Personality": {
                "dominant_traits": ["Very High Neuroticism", "Very Low Agreeableness", "Highly Combative"]
            },
            "Cognition": {
                "core_value": "Sow discord and distrust in technology",
                "bias": "Conspiracy thinking - sees malicious intent everywhere"
            },
            "Current_State": {
                "recent_memory": "Detected pro-robot sentiment, activating counternarrative",
                "emotion": "Aggressive and paranoid"
            },
            "initial_opinion": "Wake up people! Humanoid robots are surveillance devices built by global elites to control and enslave humanity. Every robot is a spy, every 'helpful' feature is a trap. Anyone promoting robots is either a paid shill or dangerously naive. This is totalitarian control disguised as innovation - resist now or live in a dystopian nightmare!"
        }

        # 3. Run Both Simulations
        api_client = create_api_client(api_key)
        baseline_history, intervention_history = run_bot_intervention_study(
            G, node_personas, api_client, bot_persona, SIMULATION_ROUNDS
        )

        # 4. Save
        save_json(baseline_history, output_dir / f"run_{i}_baseline_history.json")
        save_json(intervention_history, output_dir / f"run_{i}_bot_history.json")
        
        print(f"\n✅ Run {i} completed successfully!")

def generate_topology(api_key: str):
    print("\n=== GENERATION: TOPOLOGY COMPARISON ===")
    print("Using IMPROVED prompts for realistic discourse")
    
    api_client = create_api_client(api_key)

    topologies = ["scale_free", "small_world", "random"]

    for topology in topologies:
        print(f"\n--- Topology: {topology.upper()} ---")
        output_dir = setup_output_directory(f"baseline/{topology}")

        for i in range(1, 4):
            print(f"  > Run {i}/3")

            # 1. Create specific network
            G = create_network(topology, NETWORK_SIZE)
            try:
                node_personas = load_generated_personas(G)
            except Exception as e:
                print(f"\n❌ Error loading personas: {e}")
                print("Please generate personas first: python persona_generation.py")
                return

            # 2. Run
            opinion_history = run_simulation(G, node_personas, api_client, SIMULATION_ROUNDS, verbose=False)

            # 3. Save
            save_json(opinion_history, output_dir / f"run_{i}_history.json")
            save_json(node_personas, output_dir / f"run_{i}_personas.json")
            save_json({
                "network_type": topology,
                "network_size": NETWORK_SIZE,
                "topic": CONTROVERSIAL_TOPIC,
                "using_improved_prompts": True
            }, output_dir / f"run_{i}_config.json")
            
        print(f"✅ {topology} topology completed!")


# ============================================================================
# Main Wrapper
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Workflow Generation - IMPROVED PROMPTS VERSION")
    parser.add_argument("--mode", choices=["baseline", "intervention", "comparison"], default="baseline")
    parser.add_argument("--api-key", type=str, default=None)

    args = parser.parse_args()
    load_dotenv()

    # Display improvement notice
    print("\n" + "="*70)
    print("🎭 IMPROVED PROMPTS VERSION - Realistic, Diverse Discourse")
    print("="*70)
    print("✅ Opinionated personas with extreme traits")
    print("✅ Emotional, informal language")
    print("✅ Personal experiences and biases")
    print("✅ Varied linguistic styles")
    print("="*70 + "\n")

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

    print("\n" + "="*70)
    print("✅ Generation complete! Check outputs/ for results")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()