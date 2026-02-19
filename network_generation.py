"""
Network generation module - creates different graph topologies.
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple

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
            "contrarian": "#8c564b"       # Brown
        }
        
        node_colors = [archetype_colors.get(node_personas[node]["archetype"], "#gray") 
                      for node in G.nodes()]
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=10, label=archetype.replace('_', ' ').title())
            for archetype, color in archetype_colors.items()
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
        import random
        n_connections = max(3, int(0.3 * G.number_of_nodes()))
        target_nodes = random.sample(list(G.nodes()), n_connections)
    
    else:
        raise ValueError(f"Unknown connection strategy: {connection_strategy}")
    
    # Add edges
    for target in target_nodes:
        G_with_bot.add_edge(bot_id, target)
    
    print(f"Added bot node {bot_id} connected to {len(target_nodes)} nodes")
    
    return G_with_bot, bot_id


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
