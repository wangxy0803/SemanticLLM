"""
Simulation engine - runs discrete-time opinion dynamics with LLM agents.
"""

import time
from typing import Dict, List, Tuple
import anthropic
import openai
import networkx as nx
from config import (
    SYSTEM_PROMPT_TEMPLATE, 
    CONVERSATION_TEMPLATE, 
    CONTROVERSIAL_TOPIC,
    API_PROVIDER,
    API_MODEL
)


def create_api_client(api_key: str = None):
    """Create appropriate API client based on provider."""
    if API_PROVIDER == "anthropic":
        return anthropic.Anthropic(api_key=api_key)
    elif API_PROVIDER == "openai":
        return openai.OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported API provider: {API_PROVIDER}")


def generate_opinion(api_client, persona_data: dict, 
                    current_opinion: str, 
                    neighbor_opinions: List[Tuple[str, str]],
                    max_retries: int = 3) -> str:
    """
    Generate updated opinion for a single agent.
    
    Args:
        api_client: API client (Anthropic or OpenAI)
        persona_data: Dict with persona_prompt
        current_opinion: Agent's current opinion text
        neighbor_opinions: List of (neighbor_name, opinion_text) tuples
        
    Returns:
        Updated opinion text
    """
    # Format neighbor opinions
    neighbor_text = "\n\n".join([
        f"- {name}: {opinion}" 
        for name, opinion in neighbor_opinions
    ])
    
    # Build system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        topic=CONTROVERSIAL_TOPIC,
        persona_prompt=persona_data["persona_prompt"]
    )
    
    # Build user message
    user_message = CONVERSATION_TEMPLATE.format(
        current_opinion=current_opinion,
        neighbor_opinions=neighbor_text,
        topic=CONTROVERSIAL_TOPIC
    )
    
    # Call API with retry logic
    for attempt in range(max_retries):
        try:
            if API_PROVIDER == "anthropic":
                response = api_client.messages.create(
                    model=API_MODEL,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message}
                    ]
                )
                return response.content[0].text.strip()
                
            elif API_PROVIDER == "openai":
                response = api_client.chat.completions.create(
                    model=API_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                # Fallback: return current opinion if all retries fail
                print(f"All retries failed. Keeping previous opinion.")
                return current_opinion


def run_simulation(G: nx.Graph, 
                  node_personas: Dict[int, Dict],
                  api_client,
                  num_rounds: int = 8,
                  verbose: bool = True) -> List[Dict[int, str]]:
    """
    Run the full simulation.
    
    Args:
        G: Network graph
        node_personas: Mapping of node_id -> persona data
        api_client: API client
        num_rounds: Number of simulation rounds
        verbose: Print progress
        
    Returns:
        List of opinion snapshots (one dict per round)
        Each dict maps node_id -> opinion_text
    """
    # Initialize with initial opinions
    current_opinions = {
        node: node_personas[node]["initial_opinion"]
        for node in G.nodes()
    }
    
    # Store history
    opinion_history = [current_opinions.copy()]
    
    print(f"\n{'='*60}")
    print(f"Starting simulation: {num_rounds} rounds, {G.number_of_nodes()} agents")
    print(f"{'='*60}\n")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        start_time = time.time()
        
        # Store next round's opinions
        next_opinions = {}
        
        # Update each node
        for node in G.nodes():
            if verbose and node % 10 == 0:
                print(f"  Processing node {node}...")
            
            # Get neighbor opinions
            neighbors = list(G.neighbors(node))
            neighbor_opinions = [
                (node_personas[neighbor]["name"], current_opinions[neighbor])
                for neighbor in neighbors
            ]
            
            # Generate new opinion
            new_opinion = generate_opinion(
                api_client,
                node_personas[node],
                current_opinions[node],
                neighbor_opinions
            )
            
            next_opinions[node] = new_opinion
        
        # Update for next round
        current_opinions = next_opinions
        opinion_history.append(current_opinions.copy())
        
        elapsed = time.time() - start_time
        print(f"  Round {round_num} completed in {elapsed:.1f}s")
        
        # Show sample opinions
        if verbose:
            sample_node = list(G.nodes())[0]
            print(f"\n  Sample opinion (Node {sample_node} - {node_personas[sample_node]['name']}):")
            print(f"  {current_opinions[sample_node][:150]}...")
    
    print(f"\n{'='*60}")
    print(f"Simulation complete!")
    print(f"{'='*60}\n")
    
    return opinion_history


def run_bot_intervention_study(G: nx.Graph,
                              node_personas: Dict[int, Dict],
                              api_client,
                              bot_persona: Dict,
                              num_rounds: int = 8) -> Tuple[List, List]:
    """
    Run simulation with and without disinformation bot.
    
    Returns:
        (baseline_history, intervention_history)
    """
    from network_generation import add_disinformation_bot
    
    print("\n" + "="*60)
    print("INTERVENTION STUDY: Baseline vs. Bot")
    print("="*60)
    
    # Baseline run
    print("\n[1/2] Running BASELINE simulation (no bot)...")
    baseline_history = run_simulation(G, node_personas, api_client, num_rounds, verbose=False)
    
    # Bot intervention run
    print("\n[2/2] Running INTERVENTION simulation (with bot)...")
    G_bot, bot_id = add_disinformation_bot(G, bot_persona, connection_strategy="high_degree")
    
    # Add bot to personas
    node_personas_bot = node_personas.copy()
    node_personas_bot[bot_id] = bot_persona
    
    # Bot never updates its opinion - override generate_opinion for bot
    intervention_history = run_simulation(
        G_bot, node_personas_bot, api_client, num_rounds, verbose=False
    )
    
    # Force bot opinion to stay constant (manual override)
    for round_opinions in intervention_history:
        round_opinions[bot_id] = bot_persona["initial_opinion"]
    
    return baseline_history, intervention_history
