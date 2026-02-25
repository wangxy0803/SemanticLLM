"""
Measurement module - computes semantic variance using sentence embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from typing import List, Dict
import networkx as nx


class SemanticAnalyzer:
    """Analyzes semantic polarization using SBERT embeddings."""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model name. 
                       'all-mpnet-base-v2' is much more precise than MiniLM
                       and fits easily on an RTX 4060 (uses < 500MB VRAM).
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading sentence transformer: {model_name} on {device}")

        # This runs LOCALLY.
        # First run downloads weights from HF Hub to local cache.
        # Subsequent runs use local cache.
        # It does NOT send data to an external API.
        try:
            self.model = SentenceTransformer(model_name, device=device)
            print("Model loaded successfully (Local Inference)")
        except Exception as e:
            print(f"Failed to load model from HF Hub: {e}")
            print("Try pre-downloading the model or checking internet connection.")
            raise e

    def embed_opinions(self, opinions: Dict[int, str]) -> Dict[int, np.ndarray]:
        """
        Convert opinion texts to embeddings.
        
        Args:
            opinions: Dict mapping node_id -> opinion_text
            
        Returns:
            Dict mapping node_id -> embedding vector
        """
        texts = [opinions[node] for node in sorted(opinions.keys())]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        return {
            node: embeddings[i] 
            for i, node in enumerate(sorted(opinions.keys()))
        }
    
    def compute_semantic_variance(self, embeddings: Dict[int, np.ndarray]) -> float:
        """
        Compute semantic variance as mean pairwise cosine distance.
        
        Higher variance = more polarization/diversity
        Lower variance = more consensus/convergence
        """
        vectors = np.array([embeddings[node] for node in sorted(embeddings.keys())])
        distances = cosine_distances(vectors)
        
        # Mean of upper triangle (excluding diagonal)
        n = len(vectors)
        variance = distances[np.triu_indices(n, k=1)].mean()
        
        return variance
    
    def analyze_simulation(self, opinion_history: List[Dict[int, str]]) -> Dict:
        """
        Full analysis of simulation run.
        
        Returns dict with:
            - semantic_variance: List of variance over time
            - embeddings_history: List of embedding dicts
            - initial_variance, final_variance
            - convergence_rate
        """
        print("\nComputing semantic embeddings for all rounds...")
        
        variances = []
        embeddings_history = []
        
        for round_num, opinions in enumerate(opinion_history):
            embeddings = self.embed_opinions(opinions)
            variance = self.compute_semantic_variance(embeddings)
            
            variances.append(variance)
            embeddings_history.append(embeddings)
            
            print(f"  Round {round_num}: Semantic Variance = {variance:.4f}")
        
        # Compute metrics
        initial_variance = variances[0]
        final_variance = variances[-1]
        convergence_rate = (initial_variance - final_variance) / initial_variance if initial_variance > 0 else 0
        
        return {
            "semantic_variance": variances,
            "embeddings_history": embeddings_history,
            "initial_variance": initial_variance,
            "final_variance": final_variance,
            "convergence_rate": convergence_rate,
            "polarization_trend": "increasing" if final_variance > initial_variance else "decreasing"
        }
    
    def compute_cluster_polarization(self, embeddings: Dict[int, np.ndarray],
                                    G: nx.Graph) -> Dict:
        """
        Analyze polarization between network communities.
        """
        # Detect communities
        communities = list(nx.community.greedy_modularity_communities(G))
        
        if len(communities) < 2:
            return {"num_communities": len(communities), "between_variance": 0}
        
        # Compute within vs between community distances
        within_distances = []
        between_distances = []
        
        for i, comm1 in enumerate(communities):
            for node1 in comm1:
                for j, comm2 in enumerate(communities):
                    for node2 in comm2:
                        if node1 < node2:  # Avoid duplicates
                            dist = cosine_distances(
                                [embeddings[node1]], 
                                [embeddings[node2]]
                            )[0][0]
                            
                            if i == j:
                                within_distances.append(dist)
                            else:
                                between_distances.append(dist)
        
        return {
            "num_communities": len(communities),
            "within_variance": np.mean(within_distances) if within_distances else 0,
            "between_variance": np.mean(between_distances) if between_distances else 0,
            "polarization_ratio": np.mean(between_distances) / np.mean(within_distances) 
                                 if within_distances and between_distances else 1.0
        }


def plot_semantic_variance(analysis_results: Dict, 
                          title: str = "Semantic Variance Over Time",
                          save_path: str = None,
                          baseline_results: Dict = None):
    """
    Plot semantic variance trajectory.
    
    Args:
        analysis_results: Output from SemanticAnalyzer.analyze_simulation()
        title: Plot title
        save_path: Optional path to save figure
        baseline_results: Optional baseline for comparison
    """
    plt.figure(figsize=(10, 6))
    
    rounds = range(len(analysis_results["semantic_variance"]))
    
    # Main line
    plt.plot(rounds, analysis_results["semantic_variance"], 
             marker='o', linewidth=2, markersize=8, 
             label="LLM Simulation", color='#2E86AB')
    
    # Baseline comparison if provided
    if baseline_results:
        plt.plot(rounds, baseline_results["semantic_variance"],
                marker='s', linewidth=2, markersize=6,
                label="Baseline", color='#A23B72', linestyle='--')
    
    plt.xlabel("Simulation Round", fontsize=12)
    plt.ylabel("Semantic Variance (Mean Cosine Distance)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add trend annotation
    trend = analysis_results["polarization_trend"]
    convergence = analysis_results["convergence_rate"]
    
    textstr = f"Trend: {trend.capitalize()}\n"
    textstr += f"Convergence Rate: {convergence:+.1%}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Variance plot saved to {save_path}")
    
    plt.close()


def compare_with_degroot(G: nx.Graph, 
                        node_personas: Dict[int, Dict],
                        num_rounds: int = 8) -> List[float]:
    """
    Run classical DeGroot model for comparison.
    
    Maps personas to scalar opinions in [0,1]:
        strong_pro: 0.9
        moderate_pro: 0.65
        centrist: 0.5
        moderate_anti: 0.35
        strong_anti: 0.1
    
    Returns list of variances over time.
    """
    # Map archetypes to scalars
    archetype_to_scalar = {
        "strong_pro": 0.9,
        "moderate_pro": 0.65,
        "centrist": 0.5,
        "moderate_anti": 0.35,
        "strong_anti": 0.1,
        "contrarian": 0.5
    }
    
    # Initialize opinions
    opinions = {
        node: archetype_to_scalar[node_personas[node]["archetype"]]
        for node in G.nodes()
    }
    
    variances = [np.var(list(opinions.values()))]
    
    # Run DeGroot dynamics
    for _ in range(num_rounds):
        new_opinions = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                new_opinions[node] = np.mean([opinions[n] for n in neighbors])
            else:
                new_opinions[node] = opinions[node]
        
        opinions = new_opinions
        variances.append(np.var(list(opinions.values())))
    
    return variances


def plot_llm_vs_degroot(llm_analysis: Dict,
                       degroot_variances: List[float],
                       save_path: str = None):
    """
    Compare LLM simulation with DeGroot baseline.
    """
    plt.figure(figsize=(10, 6))
    
    rounds = range(len(llm_analysis["semantic_variance"]))
    
    plt.plot(rounds, llm_analysis["semantic_variance"],
            marker='o', linewidth=2, markersize=8,
            label="LLM Agents (Semantic)", color='#2E86AB')
    
    # Normalize DeGroot to similar scale for comparison
    degroot_normalized = np.array(degroot_variances) / degroot_variances[0]
    llm_normalized = np.array(llm_analysis["semantic_variance"]) / llm_analysis["semantic_variance"][0]
    
    plt.plot(rounds, degroot_normalized,
            marker='s', linewidth=2, markersize=6,
            label="DeGroot (Scalar, Normalized)", color='#A23B72', linestyle='--')
    
    plt.xlabel("Simulation Round", fontsize=12)
    plt.ylabel("Normalized Variance", fontsize=12)
    plt.title("LLM Semantic Dynamics vs. Classical DeGroot Model", 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()
