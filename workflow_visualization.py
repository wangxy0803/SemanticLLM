import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
import scipy.cluster.hierarchy as sch
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

def run_animated_network_evolution(mode, outputs_dir="outputs", run_id=1):
    
    search_pattern = os.path.join(outputs_dir, mode, "**", f"run_{run_id}_history.json")
    history_files = glob.glob(search_pattern, recursive=True)

    if not history_files:
        print(f"[!] No history files found for mode '{mode}'. Please ensure the generation stage has been completed.")
        return

    target_file = history_files[0]
    with open(target_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    num_rounds = len(history)
    agent_ids = list(history[0].keys())
    num_agents = len(agent_ids)
    print(f"[*] Loading {num_rounds} rounds of history data...")

    all_texts = []
    for round_idx in range(num_rounds):
        for agent_id in agent_ids:
            all_texts.append(history[round_idx].get(agent_id, ""))

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_texts)

    pca = PCA(n_components=1)
    opinion_scores = pca.fit_transform(embeddings).flatten()
    score_matrix = opinion_scores.reshape(num_rounds, num_agents)
    
    color_matrix = np.zeros_like(score_matrix)
    for r in range(num_rounds):
        current_scores = score_matrix[r]
        r_min, r_max = current_scores.min(), current_scores.max()
        if r_max > r_min:
            color_matrix[r] = (current_scores - r_min) / (r_max - r_min)
        else:
            color_matrix[r] = np.full_like(current_scores, 0.5)


    G = nx.karate_club_graph()
    
    mapping = {i: str(i) for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    valid_nodes = [node for node in agent_ids if node in G.nodes()]
    sub_G = G.subgraph(valid_nodes)
    
    pos = nx.spring_layout(sub_G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    nx.draw_networkx_edges(sub_G, pos, ax=ax, alpha=0.3, edge_color='gray')

    cmap = plt.cm.coolwarm
    nodes_collection = nx.draw_networkx_nodes(
        sub_G, pos, ax=ax, nodelist=valid_nodes,
        node_color=color_matrix[0], cmap=cmap, 
        node_size=400, edgecolors='black', vmin=0, vmax=1
    )
    
    nx.draw_networkx_labels(sub_G, pos, ax=ax, font_size=10, font_color='white')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Semantic Alignment (PCA 1D)')
    
    def update(frame):
        ax.set_title(f"Network Semantic Evolution - Mode: {mode}\nRound: {frame}/{num_rounds-1}", fontsize=14)
        current_colors = color_matrix[frame]
        nodes_collection.set_array(current_colors)
        return nodes_collection,

    print("[*] rendering animation frames...")
    ani = FuncAnimation(fig, update, frames=num_rounds, interval=600, blit=True)
    
    save_dir = os.path.dirname(target_file)
    save_path = os.path.join(save_dir, 'semantic_network_evolution.gif')
    
    ani.save(save_path, writer=PillowWriter(fps=1.5))
    plt.close()
    
    print(f"[+] Dynamic evolution diagram saved successfully to: {save_path}")

    print("[*] Generating static snapshots grid...")
    interval = 10
    frames_to_plot = list(range(0, num_rounds, interval))
    
    if frames_to_plot[-1] != num_rounds - 1:
        frames_to_plot.append(num_rounds - 1)

    n_plots = len(frames_to_plot)
    cols = n_plots
    rows = (n_plots - 1) // cols + 1

    fig_grid, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    if n_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.array(axes).flatten()

    for idx, frame in enumerate(frames_to_plot):
        ax_g = axes_flat[idx]
        current_colors = color_matrix[frame]
        
        nx.draw_networkx_edges(sub_G, pos, ax=ax_g, alpha=0.3, edge_color='gray')
        nx.draw_networkx_nodes(
            sub_G, pos, ax=ax_g, nodelist=valid_nodes,
            node_color=current_colors, cmap=cmap, 
            node_size=150, edgecolors='black', vmin=0, vmax=1
        )
        nx.draw_networkx_labels(sub_G, pos, ax=ax_g, font_size=7, font_color='white')
        
        ax_g.set_title(f"Round {frame}", fontsize=12)
        ax_g.axis('off')

    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    fig_grid.subplots_adjust(right=0.92)
    cbar_ax = fig_grid.add_axes([0.94, 0.15, 0.02, 0.7])
    fig_grid.colorbar(sm, cax=cbar_ax, label='Semantic Alignment (PCA 1D)')

    grid_save_path = os.path.join(save_dir, 'semantic_network_snapshots.png')
    fig_grid.savefig(grid_save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_grid)
    
    print(f"[+] Static snapshots grid saved successfully to: {grid_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline")
    args = parser.parse_args()
    run_animated_network_evolution(args.mode)