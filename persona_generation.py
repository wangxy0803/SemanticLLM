"""
Persona generation module - creates diverse agent personas using LLM.
IMPROVED: Generates more extreme, opinionated, realistic personas
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
    """Load weighted seed pools for persona generation."""
    if not os.path.exists(filepath):
        filepath = "seeds.json"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def weighted_choice(seed_list):
    """Randomly select from weighted list."""
    traits = [item["trait"] for item in seed_list]
    weights = [item["weight"] for item in seed_list]
    return random.choices(traits, weights=weights, k=1)[0]


def generate_persona_anthropic(client: anthropic.Anthropic, 
                               occupation: str,
                               social_class: str,
                               personality: str) -> dict:
    """Generate persona using Claude API with IMPROVED prompting for diversity."""
    
    prompt = f"""Create a psychologically REALISTIC and OPINIONATED person for a social network simulation.

CRITICAL: Generate someone who would ACTUALLY participate in online discussions - NOT a balanced diplomat!

## Base Seeds (Use these but make them SPECIFIC)
- Occupation: {occupation}
- Social Class: {social_class}
- Core Personality: {personality}

## Requirements for REALISM

1. **STRONG OPINIONS**: This person has DEFINITE views, not "I see both sides"
2. **PERSONAL HISTORY**: A specific life experience that SHAPES their worldview
3. **COGNITIVE BIAS**: A real bias that makes them discount certain arguments
4. **EMOTIONAL STATE**: Not just "neutral" - they're frustrated, excited, worried, hopeful, etc.
5. **UNIQUE VOICE**: They should sound DIFFERENT from other people

## Generate This Structure:

{{
  "Background": {{
    "exact_age_and_generation": "Specific age + generation (e.g., '34 years old, Millennial')",
    "occupation": "{occupation}",
    "social_class": "{social_class}",
    "key_experience": "ONE specific experience that shaped their views on technology/robots (make it PERSONAL and VIVID)"
  }},
  "Personality": {{
    "dominant_traits": ["Pick 2-3 SPECIFIC Big Five traits that create a DISTINCTIVE personality", "Be extreme - 'Very High Openness' or 'Very Low Agreeableness' not just 'Moderate'"]
  }},
  "Cognition": {{
    "core_value": "ONE deeply held value they will NOT compromise on",
    "bias": "A SPECIFIC cognitive bias (confirmation bias, availability heuristic, sunk cost fallacy, etc.) and what triggers it"
  }},
  "Current_State": {{
    "recent_memory": "Something SPECIFIC they recently saw/read/experienced (last week, not vague)",
    "emotion": "Their CURRENT emotional state (anxious, fired up, frustrated, optimistic, cynical, etc.)"
  }}
}}

**CRITICAL: Output must be valid JSON format matching the structure above.**

## Examples of GOOD vs BAD Personas

❌ BAD (Generic):
- key_experience: "Has seen both benefits and drawbacks of technology"
- emotion: "Thoughtful and balanced"
- core_value: "Wants what's best for society"

✅ GOOD (Specific):
- key_experience: "Sister lost her factory job to automation in 2019, family struggled for 2 years"
- emotion: "Angry and protective of working-class families"
- core_value: "Workers' dignity matters more than corporate efficiency"

❌ BAD:
- bias: "Tends to be logical in evaluations"
- recent_memory: "Read various articles about robots"

✅ GOOD:
- bias: "Availability heuristic - recent viral video of robot malfunction dominates their thinking"
- recent_memory: "Cousin sent them TikTok yesterday of robot 'attacking' a warehouse worker"

## Make This Person MEMORABLE
- Give them a STRONG stance (very pro, very anti, or very conflicted)
- Give them a REASON for that stance (the key_experience)
- Give them a WEAKNESS in reasoning (the bias)
- Give them CURRENT motivation (recent_memory + emotion)

BE CREATIVE. Make each person DIFFERENT. Avoid corporate speak or diplomatic language."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9  # Higher temperature for more diversity
        )
        
        content = response.content[0].text.strip()
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
    """Generate persona using OpenAI-compatible API with IMPROVED prompting."""
    
    system_prompt = """You are a psychology expert creating REALISTIC, OPINIONATED personas for social simulation. 

Create people who would ACTUALLY participate in heated online discussions - NOT balanced diplomats or corporate spokespeople. Each person should have:
- STRONG opinions (not "I see both sides")
- SPECIFIC life experiences that shaped their views
- REAL cognitive biases
- CURRENT emotional states (not neutral)
- UNIQUE voices

Make each persona MEMORABLE and DIFFERENT."""
    
    user_prompt = f"""Create a realistic person for online discussion simulation.

Base traits to incorporate:
- Occupation: {occupation}
- Social Class: {social_class}
- Core Personality: {personality}

Generate JSON with:

1. Background:
   - exact_age_and_generation: Specific age + generation
   - occupation: {occupation}
   - social_class: {social_class}
   - key_experience: ONE vivid, personal experience that shaped their tech views

2. Personality:
   - dominant_traits: 2-3 EXTREME Big Five traits (e.g., "Very High Conscientiousness", "Very Low Agreeableness")

3. Cognition:
   - core_value: ONE non-negotiable value
   - bias: SPECIFIC cognitive bias and what triggers it

4. Current_State:
   - recent_memory: Something specific from last week
   - emotion: Current mood (angry, excited, worried, cynical, etc.)

Make them OPINIONATED and SPECIFIC. Avoid generic balanced personas."""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.9,  # Higher temperature
            max_tokens=1500
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"OpenAI-compatible generation failed: {e}")
        return None


def generate_persona(seeds: dict, api_provider: str, client) -> dict:
    """Generate a single persona using configured API."""
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
    """Generate and save a single persona to JSON file."""
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
    load_dotenv()
    
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

    print(f"Starting IMPROVED generation of {num_agents_to_generate} DIVERSE personas using {API_PROVIDER}...")
    print("Focusing on creating OPINIONATED, REALISTIC people...")

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

    print(f"\nDone! Successfully generated {success_count} DIVERSE, OPINIONATED personas.")
    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main()