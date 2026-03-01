"""
Network generation module - creates different graph topologies.
Supports multiple network types and persona assignment strategies.
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import random
import os
import json
from pathlib import Path
from config import PERSONA_TEMPLATES

def create_network(network_type: str = "karate", n: int = 30, seed: int = 42) -> nx.Graph:
    """
    Create a network of specified type.
    
    Args:
        network_type: One of "karate", "scale_free", "small_world", "random"
        n: Number of nodes (ignored for karate)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX Graph
    """
    if network_type == "karate":
        G = nx.karate_club_graph()
        print(f"Created Karate Club graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
    elif network_type == "scale_free":
        # Barabási-Albert preferential attachment
        m = 2  # Each new node attaches to m existing nodes
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        print(f"Created Scale-Free graph: {n} nodes, {G.number_of_edges()} edges")
        
    elif network_type == "small_world":
        # Watts-Strogatz small-world
        k = 4  # Each node connected to k nearest neighbors
        p = 0.1  # Rewiring probability
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        print(f"Created Small-World graph: {n} nodes, {G.number_of_edges()} edges")
        
    elif network_type == "random":
        # Erdős-Rényi random graph
        p = 0.15  # Edge probability
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        print(f"Created Random graph: {n} nodes, {G.number_of_edges()} edges")
        
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # Ensure connected
    if not nx.is_connected(G):
        print("Warning: Graph is not connected. Using largest component.")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def visualize_network(G: nx.Graph, node_personas: dict = None, 
                     save_path: str = None, title: str = "Network Structure"):
    """
    Visualize the network with optional persona coloring.
    """
    plt.figure(figsize=(12, 8))
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    # Color nodes by archetype if personas provided
    if node_personas:
        archetype_colors = {
            "strong_pro": "#d62728",      # Red
            "moderate_pro": "#ff7f0e",    # Orange
            "centrist": "#2ca02c",        # Green
            "moderate_anti": "#1f77b4",   # Blue
            "strong_anti": "#9467bd",     # Purple
            "contrarian": "#8c564b",      # Brown
            "unknown": "#808080"          # Gray
        }
        
        # Color nodes by archetype if personas provided and have archetype field
        node_colors = []
        for node in G.nodes():
            persona = node_personas.get(node, {})
            archetype = persona.get("archetype", "unknown")
            color = archetype_colors.get(archetype, "#808080")  # Default gray
            node_colors.append(color)

        # Create legend only for present archetypes
        present_archetypes = set(node_personas[n].get("archetype", "unknown") for n in G.nodes())
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=10,
                      label=arch.replace('_', ' ').title())
            for arch, color in archetype_colors.items()
            if arch in present_archetypes
        ]
    else:
        node_colors = "#1f77b4"
        legend_elements = []
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network visualization saved to {save_path}")
    
    plt.close()


def add_disinformation_bot(G: nx.Graph, bot_persona: dict, 
                          connection_strategy: str = "high_degree") -> Tuple[nx.Graph, int]:
    """
    Add a disinformation bot node to the network.
    
    Args:
        G: Existing graph
        bot_persona: Persona dict for the bot
        connection_strategy: "high_degree" (connect to hubs) or "random"
        
    Returns:
        Modified graph and bot node ID
    """
    G_with_bot = G.copy()
    bot_id = max(G.nodes()) + 1
    
    G_with_bot.add_node(bot_id)
    
    if connection_strategy == "high_degree":
        # Connect to top 30% highest-degree nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
        n_connections = max(3, int(0.3 * len(top_nodes)))
        target_nodes = top_nodes[:n_connections]
        
    elif connection_strategy == "random":
        # Connect to 30% of nodes randomly
        n_connections = max(3, int(0.3 * G.number_of_nodes()))
        target_nodes = random.sample(list(G.nodes()), n_connections)
    
    else:
        raise ValueError(f"Unknown connection strategy: {connection_strategy}")
    
    # Add edges
    for target in target_nodes:
        G_with_bot.add_edge(bot_id, target)
    
    print(f"Added bot node {bot_id} connected to {len(target_nodes)} nodes")
    
    return G_with_bot, bot_id


def assign_personas_balanced(G: nx.Graph, seed: int = 42) -> Dict[int, Dict]:
    """
    Assign personas to nodes with balanced distribution.

    Returns dict mapping node_id -> {name, persona_prompt, initial_opinion}
    """
    random.seed(seed)

    n = G.number_of_nodes()

    # Define target distribution (percentages)
    distribution = {
        "strong_pro": 0.15,      # ~15%
        "moderate_pro": 0.25,    # ~25%
        "centrist": 0.25,        # ~25%
        "moderate_anti": 0.25,   # ~25%
        "strong_anti": 0.07,     # ~7%
        "contrarian": 0.03       # ~3%
    }

    # Build pool of personas
    persona_pool = []
    for archetype, percentage in distribution.items():
        count = max(1, int(n * percentage))  # At least 1 of each type
        personas = PERSONA_TEMPLATES[archetype]

        # Cycle through available personas in this category
        for i in range(count):
            persona = personas[i % len(personas)]
            persona_pool.append({
                "archetype": archetype,
                "name": persona["name"],
                "persona_prompt": persona["prompt"],
                "initial_opinion": persona["initial_opinion"]
            })

    # Trim or extend to exact node count
    if len(persona_pool) > n:
        persona_pool = persona_pool[:n]
    elif len(persona_pool) < n:
        # Fill remaining with random selection
        while len(persona_pool) < n:
            archetype = random.choice(list(PERSONA_TEMPLATES.keys()))
            persona = random.choice(PERSONA_TEMPLATES[archetype])
            persona_pool.append({
                "archetype": archetype,
                "name": persona["name"],
                "persona_prompt": persona["prompt"],
                "initial_opinion": persona["initial_opinion"]
            })

    # Shuffle and assign
    random.shuffle(persona_pool)

    node_personas = {}
    for i, node in enumerate(G.nodes()):
        node_personas[node] = persona_pool[i]

    return node_personas


def print_persona_distribution(node_personas: Dict[int, Dict]):
    """Print summary of persona distribution."""
    archetype_counts = {}
    for persona_data in node_personas.values():
        archetype = persona_data.get("archetype", "unknown")
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

    print("\n=== Persona Distribution ===")
    for archetype, count in sorted(archetype_counts.items()):
        print(f"{archetype:20s}: {count:2d} ({100*count/len(node_personas):5.1f}%)")
    print()


def load_generated_personas(G: nx.Graph, persona_dir: str = "prompts/persona") -> Dict[int, Dict]:
    """
    Load pre-generated persona JSON files and assign them to graph nodes.

    Args:
        G: Network graph
        persona_dir: Directory containing agent_*.json files

    Returns:
        dict mapping node_id -> persona_data string/dict
    """
    persona_path = Path(persona_dir)
    if not persona_path.exists():
        raise FileNotFoundError(f"Persona directory not found: {persona_path}")

    # Get all json files
    persona_files = list(persona_path.glob("agent_*.json"))

    if len(persona_files) < G.number_of_nodes():
        raise ValueError(f"Not enough persona files ({len(persona_files)}) for network size ({G.number_of_nodes()})")

    # Sort to ensure deterministic assignment if needed, or shuffle
    persona_files.sort()
    # random.shuffle(persona_files) # Optional: shuffle if you want random assignment

    node_personas = {}
    for i, node in enumerate(G.nodes()):
        with open(persona_files[i], 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Ensure name exists for visualization/logging
            if "name" not in data:
                data["name"] = f"Agent {data.get('agent_id', i)}"

            # Ensure initial_opinion exists (generate placeholder if missing)
            if "initial_opinion" not in data:
                # Use recent_memory or cognition as a fallback starter thought
                memory = data.get("Current_State", {}).get("recent_memory", "")
                core_val = data.get("Cognition", {}).get("core_value", "")
                data["initial_opinion"] = f"{core_val} {memory}".strip() or "I have no opinion yet."

            node_personas[node] = data

    return node_personas


def print_network_stats(G: nx.Graph):
    """Print basic network statistics."""
    print("\n=== Network Statistics ===")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Clustering coefficient: {nx.average_clustering(G):.3f}")
    print(f"Average shortest path: {nx.average_shortest_path_length(G):.2f}")
    print(f"Diameter: {nx.diameter(G)}")
    print()