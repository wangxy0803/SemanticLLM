"""
Workflow Evaluation Script.
Handles loading simulation history, running semantic analysis, and generating plots.

COMPATIBLE WITH IMPROVED PROMPTS - Same evaluation methods work fine!
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
    plot_topic_drift, plot_hostility_trend,
    plot_polarization_index, plot_model_comparison
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
    """Aggregate a list of analysis result dicts into a single average result."""
    if not analyses_list:
        return None

    count = len(analyses_list)
    avg_result = analyses_list[0].copy()

    # Metrics that are lists of numbers (time series)
    list_metrics = ["semantic_variance", "polarization_indices", "topic_drifts", "hostility_scores"]

    for metric in list_metrics:
        if metric in avg_result:
            summed = np.array(avg_result[metric])
            for i in range(1, count):
                if metric in analyses_list[i]:
                    min_len = min(len(summed), len(analyses_list[i][metric]))
                    summed = summed[:min_len] + np.array(analyses_list[i][metric][:min_len])
            avg_result[metric] = (summed / count).tolist()

    # Scalar metrics
    scalar_metrics = ["initial_variance", "final_variance", "convergence_rate"]
    for metric in scalar_metrics:
        if metric in avg_result:
            val = avg_result[metric]
            for i in range(1, count):
                val += analyses_list[i].get(metric, 0)
            avg_result[metric] = val / count

    # Re-calculate trend
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
    print("Analyzing results from IMPROVED PROMPTS")

    analyses = []
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

    print(f"\n{'='*70}")
    print("RESULTS (Averaged across 3 runs)")
    print(f"{'='*70}")
    print(f"Avg Initial Variance: {avg_analysis['initial_variance']:.4f}")
    print(f"Avg Final Variance: {avg_analysis['final_variance']:.4f}")
    print(f"Trend: {avg_analysis['polarization_trend']}")
    print(f"Convergence Rate: {avg_analysis['convergence_rate']:+.1%}")
    print(f"{'='*70}\n")

    plot_semantic_variance(avg_analysis, title="Semantic Variance (Improved Prompts - Avg of 3 Runs)",
                          save_path=str(output_dir / "avg_semantic_variance.png"))
    plot_topic_drift(avg_analysis, title="Topic Drift (Improved Prompts - Avg of 3 Runs)",
                    save_path=str(output_dir / "avg_topic_drift.png"))
    plot_polarization_index(avg_analysis, title="Polarization Index (Improved Prompts - Avg of 3 Runs)",
                           save_path=str(output_dir / "avg_polarization_index.png"))

    print(f"✅ Plots saved to {output_dir}")


def eval_intervention():
    print("\n=== EVALUATION: INTERVENTION STUDY ===")
    print("Analyzing bot intervention with IMPROVED PROMPTS")
    
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

    print(f"\n{'='*70}")
    print("BOT INTERVENTION IMPACT (Averaged across 3 runs)")
    print(f"{'='*70}")
    print(f"Avg Baseline Final Variance: {avg_base['final_variance']:.4f}")
    print(f"Avg Intervention Final Variance: {avg_int['final_variance']:.4f}")
    print(f"Avg Variance Increase: {avg_int['final_variance'] - avg_base['final_variance']:.4f}")
    print(f"Impact: {100*(avg_int['final_variance'] - avg_base['final_variance'])/avg_base['final_variance']:+.1f}%")
    print(f"{'='*70}\n")

    plot_semantic_variance(avg_int, title="Bot Intervention Impact (Improved Prompts)",
                          save_path=str(output_dir / "avg_intervention_comparison.png"),
                          baseline_results=avg_base)

    plot_topic_drift(avg_int, title="Bot Intervention - Topic Drift (Improved Prompts)",
                    save_path=str(output_dir / "avg_intervention_topic_drift.png"),
                    baseline_results=avg_base)

    plot_polarization_index(avg_int, title="Bot Intervention - Polarization (Improved Prompts)",
                           save_path=str(output_dir / "avg_intervention_polarization.png"),
                           baseline_results=avg_base)

    print(f"✅ Plots saved to {output_dir}")


def eval_topology():
    print("\n=== EVALUATION: TOPOLOGY COMPARISON ===")
    print("Analyzing topology effects with IMPROVED PROMPTS")
    
    topologies = ["scale_free", "small_world", "random"]
    results = {}

    analyzer = SemanticAnalyzer()

    for topology in topologies:
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

    if not results:
        print("No results found.")
        return

    # Print summary
    print(f"\n{'='*70}")
    print("TOPOLOGY COMPARISON SUMMARY (Improved Prompts)")
    print(f"{'='*70}")
    for topology, analysis in results.items():
        print(f"{topology.replace('_', ' ').title():15s}: Final Variance = {analysis['final_variance']:.4f}")
    print(f"{'='*70}\n")

    # Comparative Plots
    comp_dir = setup_output_directory("baseline/comparison_results")

    # Variance Plot
    plt.figure(figsize=(12, 6))
    for topology, analysis in results.items():
        data = analysis["semantic_variance"][1:]
        rounds = range(1, len(data) + 1)
        plt.plot(rounds, data, marker='o', linewidth=2, label=topology.replace('_', ' ').title())

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Simulation Round", fontsize=12)
    plt.ylabel("Semantic Variance", fontsize=12)
    plt.title("Topology Impact on Variance (Improved Prompts - Avg of 3 Runs)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(comp_dir / "topology_comparison_variance.png", dpi=300)
    plt.close()

    # Polarization Plot
    plt.figure(figsize=(12, 6))
    for topology, analysis in results.items():
        if "polarization_indices" in analysis:
            data = analysis["polarization_indices"][1:]
            rounds = range(1, len(data) + 1)
            plt.plot(rounds, data, marker='d', linewidth=2, label=topology.replace('_', ' ').title())

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Simulation Round", fontsize=12)
    plt.ylabel("Polarization Index", fontsize=12)
    plt.title("Topology Impact on Polarization (Improved Prompts - Avg of 3 Runs)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(comp_dir / "topology_comparison_polarization.png", dpi=300)
    plt.close()

    # Topic Drift Plot
    plt.figure(figsize=(12, 6))
    for topology, analysis in results.items():
        if "topic_drifts" in analysis:
            data = analysis["topic_drifts"][1:]
            rounds = range(1, len(data) + 1)
            plt.plot(rounds, data, marker='^', linewidth=2, label=topology.replace('_', ' ').title())

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Simulation Round", fontsize=12)
    plt.ylabel("Topic Drift (Semantic Distance)", fontsize=12)
    plt.title("Topology Impact on Topic Drift (Improved Prompts - Avg of 3 Runs)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(comp_dir / "topology_comparison_topic_drift.png", dpi=300)
    plt.close()

    print(f"✅ Comparison plots saved to {comp_dir}")


def eval_model_comparison():
    print("\n=== EVALUATION: MODEL COMPARISON ===")
    
    output_dir = setup_output_directory("model_comparison")
    analyzer = SemanticAnalyzer()
    
    model_analyses = {}
    
    # Find all run configs
    run_configs = list(output_dir.glob("run_*_config.json"))
    
    if not run_configs:
        print("No model comparison runs found in outputs/model_comparison/")
        # fallback to checking history files if config missing? No, need model name.
        return
        
    # Sort simply by extracting number
    try:
        run_configs.sort(key=lambda p: int(p.stem.split('_')[1]))
    except:
        pass # sort by name if parsing fails
    
    for config_path in run_configs:
        try:
            # Extract ID from filename "run_X_config.json"
            parts = config_path.stem.split('_')
            if len(parts) >= 2 and parts[0] == 'run' and parts[2] == 'config':
                run_id = parts[1]
            else:
                continue

            history_path = output_dir / f"run_{run_id}_history.json"
            
            if not history_path.exists():
                print(f"Skipping Run {run_id}: History file missing.")
                continue
                
            config = load_json(config_path)
            model_name = config.get("model", f"Model {run_id}")
            
            print(f"Analyzing Run {run_id}: {model_name}...")
            history = load_json(history_path)
            
            analysis = analyzer.analyze_simulation(history, topic=CONTROVERSIAL_TOPIC)
            
            # Ensure unique keys for plot
            if model_name in model_analyses:
                model_name = f"{model_name} ({run_id})"
            
            model_analyses[model_name] = analysis

        except Exception as e:
            print(f"Error processing {config_path}: {e}")
            continue
        
    if not model_analyses:
        print("No valid analyses generated.")
        return
        
    print("\nGenerating comparison plots...")
    
    plot_model_comparison(
        model_analyses, 
        metric="semantic_variance",
        title="Semantic Variance (Model Comparison)",
        ylabel="Variance (Lower = Consensus)",
        save_path=str(output_dir / "comparison_variance.png")
    )
    
    plot_model_comparison(
        model_analyses, 
        metric="polarization_indices",
        title="Polarization Index (Model Comparison)",
        ylabel="Polarization (Silhouette Score)",
        save_path=str(output_dir / "comparison_polarization.png")
    )
    
    plot_model_comparison(
        model_analyses, 
        metric="topic_drifts",
        title="Topic Drift (Model Comparison)",
        ylabel="Drift Distance",
        save_path=str(output_dir / "comparison_drift.png")
    )
    
    print(f"✅ Comparison plots saved to {output_dir}")


# ============================================================================
# Main Wrapper
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Workflow Evaluation - IMPROVED PROMPTS VERSION")
    parser.add_argument("--stage", choices=["generation", "evaluation", "visualization"], default="evaluation")
    parser.add_argument("--mode", choices=["baseline", "intervention", "comparison", "model_comparison"], default="baseline")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🎭 EVALUATION: IMPROVED PROMPTS VERSION")
    print("="*70 + "\n")

    if args.stage == "evaluation":
        print("📊 EVALUATION STAGE - Analyzing results (FREE - no API calls)")
        
        if args.mode == "baseline":
            eval_baseline()
        elif args.mode == "intervention":
            eval_intervention()
        elif args.mode == "comparison":
            eval_topology()
        elif args.mode == "model_comparison":
            eval_model_comparison()
    
    elif args.stage == "visualization":
        print("📈 VISUALIZATION STAGE - Generate plots from analysis results")
        # Placeholder for future visualization code
        # Currently, visualization is integrated within evaluation stages
        print("No standalone visualization tasks defined.")

    print("\n" + "="*70)
    print("✅ Evaluation complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()