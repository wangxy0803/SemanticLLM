"""
Persona generation module - creates diverse agent personas using LLM.
Generates 50+ unique personas with psychological profiles.
"""

import json
import random
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import openai
from config import API_PROVIDER, DEEPSEEK_BASE_URL


def load_seeds(filepath="prompts/seeds.json"):
    """
    Load weighted seed pools for persona generation.
    
    Args:
        filepath: Path to seeds.json file
        
    Returns:
        Dict of seed categories with weights
    """
    if not os.path.exists(filepath):
        filepath = "seeds.json"  # Fallback
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def weighted_choice(seed_list):
    """
    Randomly select from weighted list.
    
    Args:
        seed_list: List of {"trait": "...", "weight": X} dicts
        
    Returns:
        Selected trait string
    """
    traits = [item["trait"] for item in seed_list]
    weights = [item["weight"] for item in seed_list]
    return random.choices(traits, weights=weights, k=1)[0]


def generate_persona_anthropic(client: anthropic.Anthropic, 
                               occupation: str,
                               social_class: str,
                               personality: str) -> dict:
    """Generate persona using Claude API."""
    
    prompt = f"""Create a complete psychological profile for a simulated person.

Base Seeds:
- Occupation: {occupation}
- Social Class: {social_class}
- Core Personality: {personality}

Generate a realistic profile with:
1. Background (age/generation, occupation, social class, key life experience)
2. Personality (1-3 dominant Big Five traits)
3. Cognition (core value, cognitive bias)
4. Current State (recent memory, current emotion)

Output as JSON matching this structure:
{{
  "Background": {{
    "exact_age_and_generation": "e.g., 28 years old, Millennial",
    "occupation": "...",
    "social_class": "...",
    "key_experience": "A defining life event"
  }},
  "Personality": {{
    "dominant_traits": ["Trait 1", "Trait 2"]
  }},
  "Cognition": {{
    "core_value": "...",
    "bias": "..."
  }},
  "Current_State": {{
    "recent_memory": "Specific recent event",
    "emotion": "Current emotion"
  }}
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        
        content = response.content[0].text.strip()
        # Remove markdown if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        return json.loads(content.strip())
        
    except Exception as e:
        print(f"Anthropic generation failed: {e}")
        return None


def generate_persona_openai(client: openai.OpenAI,
                            occupation: str,
                            social_class: str,
                            personality: str) -> dict:
    """Generate persona using OpenAI-compatible API (DeepSeek, OpenAI)."""
    
    system_prompt = "You are an expert in psychology and sociology. Generate realistic persona profiles in JSON format."
    
    user_prompt = f"""Create a complete psychological profile for a simulated person.

Base Seeds:
- Occupation: {occupation}
- Social Class: {social_class}
- Core Personality: {personality}

Generate JSON with these fields:
- Background: exact_age_and_generation, occupation, social_class, key_experience
- Personality: dominant_traits (array of 1-3 Big Five traits)
- Cognition: core_value, bias
- Current_State: recent_memory, emotion

Make it realistic and specific."""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # Works for both DeepSeek and OpenAI
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=1000
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"OpenAI-compatible generation failed: {e}")
        return None


def generate_persona(seeds: dict, api_provider: str, client) -> dict:
    """
    Generate a single persona using configured API.
    
    Args:
        seeds: Seed pool dict
        api_provider: "anthropic" or "deepseek"/"openai"
        client: API client
        
    Returns:
        Generated persona dict or None
    """
    # Randomly select seeds with weights
    occupation = weighted_choice(seeds["occupations"])
    social_class = weighted_choice(seeds["social_classes"])
    personality = weighted_choice(seeds["personality_seeds"])

    if api_provider == "anthropic":
        return generate_persona_anthropic(client, occupation, social_class, personality)
    else:
        return generate_persona_openai(client, occupation, social_class, personality)


def generate_and_save_persona(seeds: dict,
                              agent_idx: int,
                              output_dir: Path,
                              api_provider: str,
                              client) -> bool:
    """
    Generate and save a single persona to JSON file.
    
    Args:
        seeds: Seed pool
        agent_idx: Agent index number
        output_dir: Output directory
        api_provider: API provider name
        client: API client
        
    Returns:
        True if successful
    """
    agent_id = f"agent_{agent_idx:04d}"
    print(f"Generating persona for {agent_id}...")

    persona = generate_persona(seeds, api_provider, client)
    if persona:
        persona["agent_id"] = agent_id
        output_path = output_dir / f"{agent_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(persona, f, ensure_ascii=False, indent=2)
        return True
    return False


def main():
    """Main execution - generate 50 personas."""
    # Load environment variables
    load_dotenv()
    
    # Setup
    seeds = load_seeds("prompts/seeds.json")
    num_agents_to_generate = 50
    output_dir = Path("prompts/persona")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create API client
    if API_PROVIDER == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        client = anthropic.Anthropic(api_key=api_key)
    elif API_PROVIDER in ["deepseek", "openai"]:
        if API_PROVIDER == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment")
            client = openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            client = openai.OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported API_PROVIDER: {API_PROVIDER}")

    print(f"Starting generation of {num_agents_to_generate} personas using {API_PROVIDER}...")

    # Generate with retries
    to_generate = list(range(num_agents_to_generate))
    success_count = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        while to_generate:
            print(f"\nAttempting {len(to_generate)} remaining personas (Success: {success_count}/{num_agents_to_generate})...")

            futures = {
                executor.submit(
                    generate_and_save_persona,
                    seeds, i, output_dir, API_PROVIDER, client
                ): i for i in to_generate
            }

            failed = []
            for future in futures:
                idx = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        failed.append(idx)
                except Exception as e:
                    print(f"Index {idx} raised exception: {e}")
                    failed.append(idx)

            to_generate = failed

            if to_generate:
                print(f"Retrying {len(to_generate)} failed generations...")

    print(f"\nDone! Successfully generated {success_count} personas.")
    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main()