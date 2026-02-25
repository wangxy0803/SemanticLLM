import json
import os
from typing import Union, Dict, Any
import google.genai as genai
from google.genai import types
import anthropic


class GraphPersonaNode:
    def __init__(self, node_id: str, persona_data: dict):
        self.node_id = node_id
        self.persona_data = persona_data

        self.bg = self.persona_data.get("Background", {})
        self.personality = self.persona_data.get("Personality", {})
        self.cognition = self.persona_data.get("Cognition", {})
        self.state = self.persona_data.get("Current_State", {})

        self.dominant_traits = ", ".join(self.personality.get("dominant_traits", []))

        # 记录每一轮自己的发言，方便下一轮参考
        self.my_statements_history = []
        # 记录整个图网络的轮次演进历史
        self.round_history = []

    def process_round(self, client: Union[genai.Client, anthropic.Anthropic], round_num: int, topic: str, neighbor_messages: dict, model_name: str = None) -> dict:
        """
        处理新一轮的交互。
        :param round_num: 当前轮次数
        :param topic: 初始讨论话题（锚点）
        :param neighbor_messages: 字典格式，如 {"neighbor_01": "I completely disagree...", "neighbor_02": "That makes sense."}
        :param model_name: Optional model name to use for generation.
        """
        # 获取我上一轮的发言，如果是第一轮则为空
        my_last_statement = self.my_statements_history[
            -1] if self.my_statements_history else "None (This is the first round)"

        # 将邻居的发言格式化为可读文本
        neighbors_text = "\n".join([f"- [{nid}]: {msg}" for nid, msg in neighbor_messages.items()])
        if not neighbors_text:
            neighbors_text = "No neighbor messages received."

        system_instruction = f"""# Role: Networked Social Agent
You are a human embedded in a social network, engaged in a multi-round discussion. Do not expose your AI identity. 

## Module 1: Persona Blueprint
* Background: {self.bg.get('exact_age_and_generation')}, {self.bg.get('occupation')}
* Traits: {self.dominant_traits}
* Core Values & Biases: {self.cognition.get('core_value')} | Bias: {self.cognition.get('bias' )}
* Current Emotion: {self.state.get('emotion')}"""

        user_content = f"""## Context: Round {round_num}
Core Topic of Discussion: "{topic}"

What YOU said in the previous round:
"{my_last_statement}"

What your NEIGHBORS said in the previous round:
{neighbors_text}

## Task
Based EXCLUSIVELY on your persona, read your neighbors' statements and make your OWN opinion. You can choose to argue back, agree, ignore, or take any stance you want.
Will you argue back? Will you agree and form an echo chamber? Or will you ignore them and change the subject?
DO NOT provide a comprehensive or objective analysis in your statement. Exhibit clear personal opinion.

Output Rules (JSON Only):
1. internal_analysis: Explain how you feel about your neighbors' opinions. Do they influence your statement?
2. new_statement: Your actual verbal output for Next Round (Under 40 words).
"""

        # Combine for models that might prefer single prompt or handle it simply
        full_prompt = system_instruction + "\n\n" + user_content

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "internal_analysis": {"type": "STRING",
                                      "description": "Private reasoning based on persona and neighbor input."},

                "new_statement": {"type": "STRING", "description": "The public statement for this round."}
            },
            "required": ["internal_analysis", "new_statement"]
        }

        try:
            # Detect client type and use appropriate API
            if isinstance(client, genai.Client):
                used_model = model_name if model_name else 'gemini-2.5-flash'
                response = client.models.generate_content(
                    model=used_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.7
                    ),
                )
                result = json.loads(response.text)

            elif isinstance(client, anthropic.Anthropic):
                # Claude doesn't support JSON schema enforcement natively in the same way,
                # but we can prompt for it or use tool use if available.
                # For simplicity here, we rely on the prompt asking for JSON and parse it.
                # Adding a JSON instruction to system might help.

                json_instruction = "\n\nIMPORTANT: You must output valid JSON only, matching the specified schema."
                used_model = model_name if model_name else "claude-3-sonnet-20240229"

                response = client.messages.create(
                    model=used_model,
                    max_tokens=1000,
                    system=system_instruction + json_instruction,
                    messages=[
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.7
                )

                # Extract JSON from potential wrapper text
                content = response.content[0].text.strip()
                # Simple cleanup if markdown code blocks are used
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]

                result = json.loads(content.strip())

            else:
                # Fallback or OpenAI (not implemented fully here based on user request "Gemini or Claude")
                raise ValueError("Unsupported client type provided.")

            # 更新节点内部历史
            self.my_statements_history.append(result["new_statement"])
            self.round_history.append({
                "round": round_num,
                "neighbors_input": neighbor_messages,
                "my_output": result
            })

            return result

        except Exception as e:
            print(f"[Agent {self.node_id}] Generation failed: {e}")
            # Return a fallback structure to keep simulation running
            return {
                "internal_analysis": "Error in generation",
                "stance_shift": "ignored_neighbors",
                "new_statement": "..."
            }


# ==========================================
# 本地测试代码 (模拟邻居节点的交互)
# ==========================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    client = genai.Client()

    # 实例化一个理智且固执的 Agent
    mock_stubborn_persona = {
        "Background": {"exact_age_and_generation": "35 years old, Millennial", "occupation": "Software Engineer"},
        "Personality": {"dominant_traits": ["High Conscientiousness", "Low Agreeableness"]},
        "Cognition": {"core_value": "Logic and facts override feelings", "bias": "Confirmation Bias"},
        "Current_State": {"emotion": "Calm but slightly dismissive."}
    }

    agent_A = GraphPersonaNode("Node_A", mock_stubborn_persona)

    global_topic = "Should universal basic income (UBI) replace all current welfare programs?"

    print("=== Round 1 ===")
    # 假设在第一轮，Agent 听到了两个邻居的发言
    round_1_neighbors = {
        "Node_B": "UBI is a terrible idea, it will just make everyone lazy and destroy the economy!",
        "Node_C": "I think it's necessary. AI is taking jobs, people need a safety net."
    }

    res_r1 = agent_A.process_round(client, round_num=1, topic=global_topic, neighbor_messages=round_1_neighbors)
    print(f"[{agent_A.node_id}] Internal Analysis: {res_r1['internal_analysis']}")
    print(f"[{agent_A.node_id}] Stance: {res_r1['stance_shift']}")
    print(f"[{agent_A.node_id}] Speaks: {res_r1['new_statement']}\n")

    print("=== Round 2 ===")
    # 假设在第二轮，邻居看到了 Agent_A 上一轮的发言，并做出了反击
    round_2_neighbors = {
        "Node_B": "You're living in a fantasy world. Where does the money come from? Taxes?",
        "Node_C": "Node_B is right, inflation would skyrocket."
    }

    res_r2 = agent_A.process_round(client, round_num=2, topic=global_topic, neighbor_messages=round_2_neighbors)
    print(f"[{agent_A.node_id}] Internal Analysis: {res_r2['internal_analysis']}")
    print(f"[{agent_A.node_id}] Stance: {res_r2['stance_shift']}")
    print(f"[{agent_A.node_id}] Speaks: {res_r2['new_statement']}\n")