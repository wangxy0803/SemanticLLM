import json
import random
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. 加载环境变量并初始化客户端
load_dotenv()
client = genai.Client()


def load_seeds(filepath="prompts/seeds.json"):
    """加载带权重的种子池"""
    if not os.path.exists(filepath):
        # Fallback to local if not in prompts/
        filepath = "seeds.json"
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def weighted_choice(seed_list):
    """
    根据权重列表进行随机抽取
    :param seed_list: 包含 {"trait": "...", "weight": X} 的字典列表
    :return: 抽中的 trait 字符串
    """
    traits = [item["trait"] for item in seed_list]
    weights = [item["weight"] for item in seed_list]
    # random.choices 返回的是列表，所以需要 [0]
    return random.choices(traits, weights=weights, k=1)[0]


def generate_persona(seeds):
    """
    加权随机抽取种子，并调用 LLM 生成完整的人格画像
    """
    # 使用加权函数抽取基础属性
    occupation = weighted_choice(seeds["occupations"])
    social_class = weighted_choice(seeds["social_classes"])
    personality = weighted_choice(seeds["personality_seeds"])

    prompt = f"""
    Role: Synthetic Data & Sociology Expert
    Task: Flesh out a complete background and psychological profile for a simulated social media user, based on the provided "base seeds".

    Base Seeds: 
    - Occupation: {occupation}
    - Social Class: {social_class}
    - Core Personality Seed: {personality}

    Requirements:
    1. Elaborate on their `key_experience` (a core life event), `core_value`, and `bias` (cognitive bias or blind spot) based on the seeds. The logic must be highly coherent and realistic.
    2. Generate a highly relatable, mundane, and specific `recent_memory` (e.g., "Dropped my AirPods on the subway tracks this morning") and their current `emotion`.
    3. For the Big Five personality traits: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, select 1 dominant traits that strongly align with the '{personality}' seed.
    4. Strictly follow the provided JSON Schema.
    """

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "Background": {
                "type": "OBJECT",
                "properties": {
                    "exact_age_and_generation": {"type": "STRING", "description": "e.g., '24 years old, Gen Z'"},
                    "occupation": {"type": "STRING"},
                    "social_class": {"type": "STRING"},
                    "key_experience": {"type": "STRING", "description": "A life-defining event"}
                }
            },
            "Personality": {
                "type": "OBJECT",
                "properties": {
                    "dominant_traits": {
                        "type": "ARRAY",
                        "description": "1 to 3 standout Big Five traits",
                        "items": {"type": "STRING"}
                    }
                }
            },
            "Cognition": {
                "type": "OBJECT",
                "properties": {
                    "core_value": {"type": "STRING"},
                    "bias": {"type": "STRING"}
                }
            },
            "Current_State": {
                "type": "OBJECT",
                "properties": {
                    "recent_memory": {"type": "STRING"},
                    "emotion": {"type": "STRING"}
                }
            }
        }
    }

    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.8
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Generation failed: {e}")
        return None


def generate_and_save_persona(seeds, agent_idx, output_dir):
    """
    生成单个角色画像并保存为 JSON 文件
    """
    agent_id = f"agent_{agent_idx:04d}"
    print(f"Generating persona for {agent_id}...")

    persona = generate_persona(seeds)
    if persona:
        persona["agent_id"] = agent_id
        output_path = output_dir / f"{agent_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(persona, f, ensure_ascii=False, indent=2)
        return True
    return False


def main():
    seeds = load_seeds("prompts/seeds.json")
    num_agents_to_generate = 50
    output_dir = Path("prompts/persona")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting multi-threaded generation of {num_agents_to_generate} Agent personas...")

    # Indices that still need to be generated successfully
    to_generate = list(range(num_agents_to_generate))
    success_count = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        while to_generate:
            print(f"\nAttempting to generate {len(to_generate)} remaining personas (Current success: {success_count}/{num_agents_to_generate})...")

            # Map index to future
            futures = {executor.submit(generate_and_save_persona, seeds, i, output_dir): i for i in to_generate}

            # Update to_generate list for the next iteration (only keep failed ones)
            failed = []
            for future in futures:
                idx = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        failed.append(idx)
                except Exception as e:
                    print(f"Index {idx} raised an exception: {e}")
                    failed.append(idx)

            to_generate = failed

            if to_generate:
                print(f"Retrying {len(to_generate)} failed generations...")

    print(f"\nDone! Successfully generated {success_count} personas.")
    print(f"Results saved as individual files in {output_dir}")


if __name__ == "__main__":
    main()