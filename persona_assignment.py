"""
Persona assignment module - assigns personas to network nodes strategically.
"""

import random
import networkx as nx
from typing import Dict, List, Tuple
from config import PERSONA_TEMPLATES, CONTROVERSIAL_TOPIC

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


def assign_personas_polarized(G: nx.Graph, seed: int = 42) -> Dict[int, Dict]:
    """
    Assign personas to create polarized clusters (tests echo chambers).
    Places similar personas in connected communities.
    """
    random.seed(seed)
    
    # Detect communities
    communities = nx.community.greedy_modularity_communities(G)
    
    node_personas = {}
    
    # Assign pro-regulation to first community, anti to second, mixed to rest
    for comm_idx, community in enumerate(communities):
        if comm_idx == 0:
            # Pro-regulation cluster
            archetypes = ["strong_pro", "moderate_pro"]
        elif comm_idx == 1:
            # Anti-regulation cluster
            archetypes = ["moderate_anti", "strong_anti"]
        else:
            # Mixed
            archetypes = list(PERSONA_TEMPLATES.keys())
        
        for node in community:
            archetype = random.choice(archetypes)
            persona = random.choice(PERSONA_TEMPLATES[archetype])
            node_personas[node] = {
                "archetype": archetype,
                "name": persona["name"],
                "persona_prompt": persona["prompt"],
                "initial_opinion": persona["initial_opinion"]
            }
    
    return node_personas


def print_persona_distribution(node_personas: Dict[int, Dict]):
    """Print summary of persona distribution."""
    archetype_counts = {}
    for persona_data in node_personas.values():
        archetype = persona_data["archetype"]
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
    
    print("\n=== Persona Distribution ===")
    for archetype, count in sorted(archetype_counts.items()):
        print(f"{archetype:20s}: {count:2d} ({100*count/len(node_personas):5.1f}%)")
    print()


def test_persona_consistency(api_client, node_personas: Dict[int, Dict], 
                            test_node: int = 0, num_tests: int = 3):
    """
    Test if a persona generates consistent responses.
    Useful for debugging persona prompts.
    """
    from simulation import generate_opinion
    
    print(f"\n=== Testing Persona: {node_personas[test_node]['name']} ===")
    print(f"Initial opinion: {node_personas[test_node]['initial_opinion'][:100]}...")
    
    # Create fake neighbor opinions
    neighbor_opinions = [
        ("Neighbor A", "AI regulation is essential for public safety."),
        ("Neighbor B", "We should let the market decide, not government bureaucrats."),
        ("Neighbor C", "I'm uncertain - both sides have valid points.")
    ]
    
    print(f"\nGiven these neighbor opinions:")
    for name, opinion in neighbor_opinions:
        print(f"  - {name}: {opinion}")
    
    print(f"\nGenerating {num_tests} responses...")
    for i in range(num_tests):
        response = generate_opinion(
            api_client,
            node_personas[test_node],
            node_personas[test_node]['initial_opinion'],
            neighbor_opinions
        )
        print(f"\nResponse {i+1}: {response}")
