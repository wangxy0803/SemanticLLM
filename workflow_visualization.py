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


def run_averaged_heatmap(mode, outputs_dir="outputs"):
    print(f"[*] Starting to generate 3-run averaged distance heatmap - Mode: {mode}")
    
    search_pattern = os.path.join(outputs_dir, mode, "**", "run_*_history.json")
    history_files = glob.glob(search_pattern, recursive=True)
    
    if len(history_files) == 0:
        print(f"[!] No history files found for mode '{mode}'. Please ensure the generation stage has been completed for this mode.")
        return
        
    print(f"[*] Found {len(history_files)} run result files, preparing to calculate average matrix...")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_distance_matrices = []
    agent_ids = None

    for file_path in history_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
            
        if agent_ids is None:
            agent_ids = list(history[0].keys())
            
        final_round = history[-1]
        final_texts = [final_round.get(aid, "") for aid in agent_ids]
        
        embeddings = model.encode(final_texts)
        dist_matrix = cosine_distances(embeddings)
        all_distance_matrices.append(dist_matrix)

    avg_dist_matrix = np.mean(np.stack(all_distance_matrices), axis=0)

    linkage = sch.linkage(sch.distance.squareform(avg_dist_matrix), method='ward')
    order = sch.leaves_list(linkage)
    
    sorted_avg_matrix = avg_dist_matrix[order, :][:, order]
    sorted_agent_ids = [agent_ids[i] for i in order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_avg_matrix, 
                cmap="viridis_r", 
                vmin=0, vmax=0.45,
                xticklabels=sorted_agent_ids, 
                yticklabels=sorted_agent_ids,
                cbar_kws={'label': 'Average Cosine Distance'})
    
    plt.title(f'Averaged Semantic Distance Heatmap (3 Runs) - Mode: {mode}\n(Hierarchically Clustered)', fontsize=14)
    plt.xlabel('Agent ID', fontsize=12)
    plt.ylabel('Agent ID', fontsize=12)
    
    plt.xticks(np.arange(0, len(agent_ids), 5), sorted_agent_ids[::5], rotation=45)
    plt.yticks(np.arange(0, len(agent_ids), 5), sorted_agent_ids[::5], rotation=0)

    save_dir = os.path.dirname(history_files[0])
    save_path = os.path.join(save_dir, 'avg_semantic_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Average heatmap saved successfully to: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline")
    args = parser.parse_args()
    run_averaged_heatmap(args.mode)
    run_animated_network_evolution(args.mode)