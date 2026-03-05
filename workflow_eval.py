"""
Workflow Evaluation Script.
Handles loading simulation history, running semantic analysis, and generating plots.
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from config import (
    NETWORK_SIZE, NETWORK_TYPE, SIMULATION_ROUNDS,
    CONTROVERSIAL_TOPIC
)
from network_generation import create_network
from measurement import (
    SemanticAnalyzer, plot_semantic_variance,
    compare_with_degroot, plot_llm_vs_degroot,
    plot_topic_drift, plot_hostility_trend,
    plot_polarization_index
)

# ============================================================================
# Helper Functions
# ============================================================================

def setup_output_directory(subdir: str = ""):
    output_dir = Path("./outputs") / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_json(filepath):
    """Load JSON file with integer keys restored."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def restore_keys(obj):
        if isinstance(obj, dict):
            # Try to convert keys to int if they look like ints
            new_dict = {}
            for k, v in obj.items():
                try:
                    new_key = int(k)
                except ValueError:
                    new_key = k
                new_dict[new_key] = restore_keys(v)
            return new_dict
        elif isinstance(obj, list):
            return [restore_keys(i) for i in obj]
        else:
            return obj

    return restore_keys(data)

def aggregate_analyses(analyses_list):
    """
    Aggregate a list of analysis result dicts into a single average result.
    Computes mean for list-based metrics (time series)
    """
    if not analyses_list:
        return None

    count = len(analyses_list)
    avg_result = analyses_list[0].copy()

    # Metrics that are lists of numbers (time series)
    list_metrics = ["semantic_variance", "polarization_indices", "topic_drifts", "hostility_scores"]

    for metric in list_metrics:
        if metric in avg_result:
            # Sum up
            summed = np.array(avg_result[metric])
            for i in range(1, count):
                if metric in analyses_list[i]:
                    # Ensure length match, truncate to min length just in case
                    min_len = min(len(summed), len(analyses_list[i][metric]))
                    summed = summed[:min_len] + np.array(analyses_list[i][metric][:min_len])

            # Divide by count to get mean
            avg_result[metric] = (summed / count).tolist()

    # Scalar metrics
    scalar_metrics = ["initial_variance", "final_variance", "convergence_rate"]
    for metric in scalar_metrics:
        if metric in avg_result:
            val = avg_result[metric]
            for i in range(1, count):
                val += analyses_list[i].get(metric, 0)
            avg_result[metric] = val / count

    # Re-calculate trend based on average
    if avg_result["final_variance"] > avg_result["initial_variance"]:
        avg_result["polarization_trend"] = "increasing"
    else:
        avg_result["polarization_trend"] = "decreasing"

    return avg_result

# ============================================================================
# Evaluation Flows
# ============================================================================

def eval_baseline():
    print("\n=== EVALUATION: BASELINE SIMULATION ===")

    # Load 3 runs
    analyses = []
    # Change: Load from baseline/{NETWORK_TYPE} to match generation structure
    output_dir = setup_output_directory(f"baseline/{NETWORK_TYPE}")

    analyzer = SemanticAnalyzer()

    for i in range(1, 4):
        history_path = output_dir / f"run_{i}_history.json"
        if history_path.exists():
            print(f"Analyzing Run {i}...")
            history = load_json(history_path)
            res = analyzer.analyze_simulation(history, topic=CONTROVERSIAL_TOPIC)
            analyses.append(res)
        else:
            print(f"Warning: {history_path} missing.")

    if not analyses:
        print("No valid runs found.")
        return

    # Average
    avg_analysis = aggregate_analyses(analyses)

    print(f"Avg Initial Variance: {avg_analysis['initial_variance']:.4f}")
    print(f"Avg Final Variance: {avg_analysis['final_variance']:.4f}")
    print(f"Trend: {avg_analysis['polarization_trend']}")

    plot_semantic_variance(avg_analysis, title="Semantic Variance (Avg of 3 Runs)",
                          save_path=str(output_dir / "avg_semantic_variance.png"))
    plot_topic_drift(avg_analysis, title="Topic Drift (Avg of 3 Runs)",
                    save_path=str(output_dir / "avg_topic_drift.png"))
    plot_polarization_index(avg_analysis, title="Polarization Index (Avg of 3 Runs)",
                           save_path=str(output_dir / "avg_polarization_index.png"))


def eval_intervention():
    print("\n=== EVALUATION: INTERVENTION STUDY ===")
    output_dir = setup_output_directory("intervention")

    analyzer = SemanticAnalyzer()

    base_analyses = []
    int_analyses = []

    for i in range(1, 4):
        b_path = output_dir / f"run_{i}_baseline_history.json"
        i_path = output_dir / f"run_{i}_bot_history.json"

        if b_path.exists() and i_path.exists():
            print(f"Analyzing Run {i}...")
            b_hist = load_json(b_path)
            i_hist = load_json(i_path)

            base_analyses.append(analyzer.analyze_simulation(b_hist, topic=CONTROVERSIAL_TOPIC))
            int_analyses.append(analyzer.analyze_simulation(i_hist, topic=CONTROVERSIAL_TOPIC))

    if not base_analyses:
        print("No valid runs.")
        return

    avg_base = aggregate_analyses(base_analyses)
    avg_int = aggregate_analyses(int_analyses)

    print(f"Avg Variance Increase: {avg_int['final_variance'] - avg_base['final_variance']:.4f}")

    plot_semantic_variance(avg_int, title="Bot Intervention Impact (Avg Variance)",
                          save_path=str(output_dir / "avg_intervention_comparison.png"),
                          baseline_results=avg_base)

    plot_topic_drift(avg_int, title="Bot Intervention Impact (Avg Drift)",
                    save_path=str(output_dir / "avg_intervention_topic_drift.png"),
                    baseline_results=avg_base)

    plot_polarization_index(avg_int, title="Bot Intervention Impact (Avg Polarization)",
                           save_path=str(output_dir / "avg_intervention_polarization.png"),
                           baseline_results=avg_base)


def eval_topology():
    print("\n=== EVALUATION: TOPOLOGY COMPARISON ===")
    topologies = ["scale_free", "small_world", "random"]
    results = {}

    analyzer = SemanticAnalyzer()

    for topology in topologies:
        # Change: Load from baseline/{topology} to match unified structure
        output_dir = setup_output_directory(f"baseline/{topology}")
        topo_analyses = []

        print(f"\nProcessing {topology}...")
        for i in range(1, 4):
            path = output_dir / f"run_{i}_history.json"
            if path.exists():
                print(f"  Run {i}...")
                hist = load_json(path)
                topo_analyses.append(analyzer.analyze_simulation(hist, topic=CONTROVERSIAL_TOPIC))

        if topo_analyses:
            results[topology] = aggregate_analyses(topo_analyses)

    # Comparative Plots
    if not results:
        return

    # Change: Save comparison results to a dedicated folder under baseline
    comp_dir = setup_output_directory("baseline/comparison_results")

    # Variance Plot
    plt.figure(figsize=(12, 6))
    for topology, analysis in results.items():
        data = analysis["semantic_variance"][1:]
        rounds = range(1, len(data) + 1)
        plt.plot(rounds, data, marker='o', label=topology.title())

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Simulation Round")
    plt.ylabel("Semantic Variance")
    plt.title("Topology Impact on Variance (Avg of 3 Runs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(comp_dir / "topology_comparison_variance.png")
    plt.close()

    # Polarization Plot
    plt.figure(figsize=(12, 6))
    for topology, analysis in results.items():
        if "polarization_indices" in analysis:
            data = analysis["polarization_indices"][1:]
            rounds = range(1, len(data) + 1)
            plt.plot(rounds, data, marker='d', label=topology.title())

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Simulation Round")
    plt.ylabel("Polarization Index")
    plt.title("Topology Impact on Polarization (Avg of 3 Runs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(comp_dir / "topology_comparison_polarization.png")
    plt.close()

    # Topic Drift Plot
    plt.figure(figsize=(12, 6))
    for topology, analysis in results.items():
        if "topic_drifts" in analysis:
            data = analysis["topic_drifts"][1:]
            rounds = range(1, len(data) + 1)
            plt.plot(rounds, data, marker='^', label=topology.title())

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Simulation Round")
    plt.ylabel("Topic Drift (Semantic Distance)")
    plt.title("Topology Impact on Topic Drift (Avg of 3 Runs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(comp_dir / "topology_comparison_topic_drift.png")
    plt.close()


def eval_degroot():
    print("\n=== EVALUATION: DEGROOT COMPARISON ===")
    output_dir = setup_output_directory("degroot")

    analyzer = SemanticAnalyzer()
    llm_analyses = []
    degroot_variances_list = []

    for i in range(1, 4):
        llm_path = output_dir / f"run_{i}_history.json"
        personas_path = output_dir / f"run_{i}_personas.json"

        if llm_path.exists() and personas_path.exists():
            print(f"Run {i}...")
            llm_hist = load_json(llm_path)
            node_personas = load_json(personas_path)

            # LLM Analysis
            llm_analyses.append(analyzer.analyze_simulation(llm_hist))

            # DeGroot Analysis
            G = create_network(NETWORK_TYPE, NETWORK_SIZE)
            d_var = compare_with_degroot(G, node_personas, SIMULATION_ROUNDS)
            degroot_variances_list.append(d_var)

    if not llm_analyses:
        return

    avg_llm = aggregate_analyses(llm_analyses)

    # Average DeGroot
    avg_degroot = np.mean(degroot_variances_list, axis=0).tolist()

    plot_llm_vs_degroot(avg_llm, avg_degroot, save_path=str(output_dir / "avg_llm_vs_degroot.png"))


# ============================================================================
# Main Wrapper
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Workflow Evaluation")
    parser.add_argument("--mode", choices=["baseline", "intervention", "comparison", "degroot"], default="baseline")

    args = parser.parse_args()

    if args.mode == "baseline":
        eval_baseline()
    elif args.mode == "intervention":
        eval_intervention()
    elif args.mode == "comparison":
        eval_topology()
    elif args.mode == "degroot":
        eval_degroot()

if __name__ == "__main__":
    main()

