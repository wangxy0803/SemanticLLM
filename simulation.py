"""
Simulation engine - runs discrete-time opinion dynamics with LLM agents.
Supports parallel processing for faster execution.
"""

import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
import openai
import networkx as nx
from config import (
    CONTROVERSIAL_TOPIC,
    API_PROVIDER,
    API_MODEL,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL
)
from persona_agent import GraphPersonaNode


def create_api_client(api_key: str = None):
    """
    Create appropriate API client based on provider configuration.
    
    Args:
        api_key: API key for the chosen provider
        
    Returns:
        Configured API client
    """
    if API_PROVIDER == "anthropic":
        return anthropic.Anthropic(api_key=api_key)
    
    elif API_PROVIDER == "deepseek":
        # DeepSeek uses OpenAI-compatible API
        return openai.OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL
        )
    
    elif API_PROVIDER == "openai":
        return openai.OpenAI(api_key=api_key)
    
    else:
        raise ValueError(f"Unsupported API provider: {API_PROVIDER}. Use 'anthropic', 'deepseek', or 'openai'")


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
    Process a single agent's round in parallel execution.
    
    Args:
        node_id: Agent's node ID
        agent: GraphPersonaNode instance
        api_client: API client
        round_num: Current round number
        topic: Discussion topic
        neighbor_messages: Dict of neighbor opinions
        is_bot: Whether this is a stubborn bot
        current_opinion: Current opinion text
        model_name: Model to use (optional)
        
    Returns:
        Tuple of (node_id, new_opinion)
    """
    # Bots never update their opinions
    if is_bot:
        return node_id, current_opinion

    try:
        # Agent processes the round and generates new opinion
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
                   verbose: bool = True,
                   model_name: str = None) -> List[Dict[int, str]]:
    """
    Run the full multi-round simulation with parallel agent processing.
    
    Args:
        G: Network graph
        node_personas: Mapping of node_id -> persona data dict
        api_client: API client (Anthropic, DeepSeek, or OpenAI)
        num_rounds: Number of simulation rounds
        verbose: Print progress updates
        model_name: Specific model to use (if None, uses default from config)
        
    Returns:
        List of opinion snapshots (one dict per round)
        Each dict maps node_id -> opinion_text
    """
    # Initialize agent instances
    agents: Dict[int, GraphPersonaNode] = {}
    for node_id in G.nodes():
        agents[node_id] = GraphPersonaNode(
            node_id=str(node_id), 
            persona_data=node_personas[node_id]
        )

        # Set initial opinion
        initial_opinion = node_personas[node_id].get("initial_opinion", "")
        if initial_opinion:
            agents[node_id].my_statements_history.append(initial_opinion)

    # Initialize opinion history with round 0
    current_opinions = {
        node_id: agents[node_id].my_statements_history[-1] 
        if agents[node_id].my_statements_history else ""
        for node_id in G.nodes()
    }
    opinion_history = [current_opinions.copy()]

    # Determine model to use
    if model_name is None:
        model_name = API_MODEL

    print(f"\n{'='*60}")
    print(f"Starting simulation: {num_rounds} rounds, {len(agents)} agents")
    print(f"Using API: {API_PROVIDER}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Run each round
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        start_time = time.time()
        
        next_opinions = {}
        
        # Process agents in parallel
        with ThreadPoolExecutor() as executor:
            tasks = []
            
            for node_id in G.nodes():
                # Gather neighbor opinions
                neighbors = list(G.neighbors(node_id))
                neighbor_messages = {}
                for neighbor_id in neighbors:
                    neighbor_name = node_personas[neighbor_id]["name"]
                    neighbor_msg = current_opinions.get(neighbor_id, "")
                    neighbor_messages[f"{neighbor_name} (ID: {neighbor_id})"] = neighbor_msg

                is_bot = node_personas[node_id].get("is_bot", False)
                current_op = current_opinions.get(node_id, "")

                # Submit parallel task
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
                    model_name=model_name
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
        
        # Show sample opinion
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
    Run simulation with and without disinformation bot to measure impact.
    
    Args:
        G: Network graph
        node_personas: Persona mappings
        api_client: API client
        bot_persona: Persona for the bot
        num_rounds: Number of rounds
        
    Returns:
        Tuple of (baseline_history, intervention_history)
    """
    from network_generation import add_disinformation_bot
    
    print("\n" + "="*60)
    print("INTERVENTION STUDY: Baseline vs. Bot")
    print("="*60)
    
    # Baseline run (no bot)
    print("\n[1/2] Running BASELINE simulation (no bot)...")
    baseline_history = run_simulation(G, node_personas, api_client, num_rounds, verbose=False)
    
    # Intervention run (with bot)
    print("\n[2/2] Running INTERVENTION simulation (with bot)...")
    G_bot, bot_id = add_disinformation_bot(G, bot_persona, connection_strategy="high_degree")
    
    # Add bot to personas with stubborn flag
    node_personas_bot = node_personas.copy()
    bot_persona_with_flag = bot_persona.copy()
    bot_persona_with_flag["is_bot"] = True
    node_personas_bot[bot_id] = bot_persona_with_flag

    intervention_history = run_simulation(
        G_bot, node_personas_bot, api_client, num_rounds, verbose=False
    )
    
    return baseline_history, intervention_history