"""
Simulation engine - runs discrete-time opinion dynamics with LLM agents.
"""

import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
import openai
from google import genai
import networkx as nx
from config import (
    CONTROVERSIAL_TOPIC,
    API_PROVIDER,
    API_MODEL
)
from persona_agent import GraphPersonaNode


def create_api_client(api_key: str = None):
    """Create appropriate API client based on provider."""
    if API_PROVIDER == "anthropic":
        return anthropic.Anthropic(api_key=api_key)
    elif API_PROVIDER == "openai":
        return openai.OpenAI(api_key=api_key)
    elif API_PROVIDER == "gemini" or API_PROVIDER == "google":
        # Initialize Google GenAI client from environment variables
        return genai.Client(api_key=api_key)
    else:
        raise ValueError(f"Unsupported API provider: {API_PROVIDER}")


def _process_single_agent(node_id: int,
                          agent: GraphPersonaNode,
                          api_client,
                          round_num: int,
                          topic: str,
                          neighbor_messages: Dict[str, str],
                          is_bot: bool,
                          current_opinion: str,
                          model_name: str = None) -> Tuple[int, str]:
    """
    Helper function to process a single agent's round in a thread.
    Returns (node_id, new_opinion).
    """
    # Check if bot (stubborn agent)
    if is_bot:
        return node_id, current_opinion

    # Agent processes the round
    try:
        # Note: process_round returns a full dict with internal_analysis, etc.
        result = agent.process_round(
            client=api_client,
            round_num=round_num,
            topic=topic,
            neighbor_messages=neighbor_messages,
            model_name=model_name
        )

        # Extract the public statement
        new_opinion = result.get("new_statement", "")
        return node_id, new_opinion

    except Exception as e:
        print(f"Error processing node {node_id}: {e}")
        # Fallback: keep previous opinion
        return node_id, current_opinion


def run_simulation(G: nx.Graph,
                   node_personas: Dict[int, Dict],
                   api_client,
                   num_rounds: int = 8,
                   verbose: bool = True) -> List[Dict[int, str]]:
    """
    Run the full simulation using GraphPersonaNode agents.

    Args:
        G: Network graph
        node_personas: Mapping of node_id -> persona data dict
        api_client: API client (Must be compatible with Agent class)
        num_rounds: Number of simulation rounds
        verbose: Print progress
        
    Returns:
        List of opinion snapshots (one dict per round)
        Each dict maps node_id -> opinion_text
    """
    # 1. Initialize Agent Nodes
    agents: Dict[int, GraphPersonaNode] = {}
    for node_id in G.nodes():
        # Instantiate GraphPersonaNode for each graph node
        agents[node_id] = GraphPersonaNode(node_id=str(node_id), persona_data=node_personas[node_id])

        # Initialize history with initial opinion if available in persona data
        initial_op = node_personas[node_id].get("initial_opinion", "")
        if initial_op:
            agents[node_id].my_statements_history.append(initial_op)

    # Store history of opinions (text only) for analysis
    # Round 0 (Initial state)
    current_opinions = {
        node_id: agents[node_id].my_statements_history[-1] if agents[node_id].my_statements_history else ""
        for node_id in G.nodes()
    }
    opinion_history = [current_opinions.copy()]

    print(f"\n{'='*60}")
    print(f"Starting simulation: {num_rounds} rounds, {len(agents)} agents")
    print(f"{'='*60}\n")

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        start_time = time.time()
        
        # Store next round's opinions
        next_opinions = {}
        
        # Prepare tasks for parallel execution
        tasks = []
        with ThreadPoolExecutor() as executor:
            for node_id in G.nodes():
                # Prepare neighbor messages (READ-ONLY from current_opinions)
                neighbors = list(G.neighbors(node_id))
                neighbor_messages = {}
                for neighbor_id in neighbors:
                    # Use string node_id for neighbor keys as per persona_agent expectation
                    neighbor_name = node_personas[neighbor_id]["name"]
                    neighbor_msg = current_opinions.get(neighbor_id, "")
                    neighbor_messages[f"{neighbor_name} (ID: {neighbor_id})"] = neighbor_msg

                is_bot = node_personas[node_id].get("is_bot", False)
                current_op = current_opinions.get(node_id, "")

                # Submit task
                tasks.append(executor.submit(
                    _process_single_agent,
                    node_id=node_id,
                    agent=agents[node_id],
                    api_client=api_client,
                    round_num=round_num,
                    topic=CONTROVERSIAL_TOPIC,
                    neighbor_messages=neighbor_messages,
                    is_bot=is_bot,
                    current_opinion=current_op,
                    model_name=API_MODEL
                ))

            # Collect results as they complete
            completed_count = 0
            total_tasks = len(tasks)
            for future in as_completed(tasks):
                node_id, new_opinion = future.result()
                next_opinions[node_id] = new_opinion

                completed_count += 1
                if verbose and completed_count % 10 == 0:
                    print(f"  Processed {completed_count}/{total_tasks} agents...")

        # Update for next round
        current_opinions = next_opinions
        opinion_history.append(current_opinions.copy())
        
        elapsed = time.time() - start_time
        print(f"  Round {round_num} completed in {elapsed:.1f}s")
        
        # Show sample opinions
        if verbose and len(G.nodes()) > 0:
            sample_node = list(G.nodes())[0]
            name = node_personas[sample_node]['name']
            print(f"\n  Sample opinion (Node {sample_node} - {name}):")
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

    # Mark bot as stubborn agent
    bot_persona_with_flag = bot_persona.copy()
    bot_persona_with_flag["is_bot"] = True
    node_personas_bot[bot_id] = bot_persona_with_flag

    # Bot never updates its opinion - override via is_bot flag
    intervention_history = run_simulation(
        G_bot, node_personas_bot, api_client, num_rounds, verbose=False
    )
    
    return baseline_history, intervention_history
