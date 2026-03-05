"""
Agent class with memory and two-phase response generation.
Each agent maintains conversation history and generates opinions based on persona.
"""

import json
from typing import Union, Dict
import anthropic
import openai


class GraphPersonaNode:
    """
    Represents a single agent in the opinion dynamics network.
    Each agent has a persona, memory, and generates opinions through LLM.
    """
    
    def __init__(self, node_id: str, persona_data: dict):
        """
        Initialize agent with persona data.
        
        Args:
            node_id: Unique identifier for this node
            persona_data: Dict containing persona information
        """
        self.node_id = node_id
        self.persona_data = persona_data

        # Extract persona components
        self.bg = self.persona_data.get("Background", {})
        self.personality = self.persona_data.get("Personality", {})
        self.cognition = self.persona_data.get("Cognition", {})
        self.state = self.persona_data.get("Current_State", {})

        # Format personality traits
        self.dominant_traits = ", ".join(self.personality.get("dominant_traits", []))

        # Memory systems
        self.my_statements_history = []  # What I said each round
        self.round_history = []  # Full conversation log

    def process_round(self, 
                     client: Union[anthropic.Anthropic, openai.OpenAI],
                     round_num: int,
                     topic: str,
                     neighbor_messages: dict,
                     model_name: str = None) -> dict:
        """
        Process one round of interaction and generate new opinion.
        
        Args:
            client: API client (Anthropic or OpenAI-compatible)
            round_num: Current round number
            topic: Discussion topic
            neighbor_messages: Dict of {neighbor_name: opinion_text}
            model_name: Model to use (optional)
            
        Returns:
            Dict with "internal_analysis" and "new_statement"
        """
        # Get previous statement
        my_last_statement = (
            self.my_statements_history[-1] 
            if self.my_statements_history 
            else "None (This is the first round)"
        )

        # Format neighbor opinions
        neighbors_text = "\n".join([
            f"- [{nid}]: {msg}" 
            for nid, msg in neighbor_messages.items()
        ])
        if not neighbors_text:
            neighbors_text = "No neighbor messages received."

        # Build system instruction
        system_instruction = f"""# Role: Networked Social Agent
You are a human in a social network discussing opinions. Stay in character.

## Persona
* Background: {self.bg.get('exact_age_and_generation', 'N/A')}, {self.bg.get('occupation', 'N/A')}
* Personality: {self.dominant_traits}
* Values: {self.cognition.get('core_value', 'N/A')}
* Bias: {self.cognition.get('bias', 'N/A')}
* Current Emotion: {self.state.get('emotion', 'N/A')}"""

        # Build user message
        user_content = f"""## Round {round_num} - Topic: "{topic}"

Your previous statement:
"{my_last_statement}"

Your neighbors' statements:
{neighbors_text}

## Task
Based on your persona and neighbors' opinions, generate your response.
Maintain consistency with your previous statements unless you see a compelling reason to change.
Do not flip-flop your opinion randomly.
You can agree, disagree, argue, or ignore them. Show clear personal opinion.

Output as JSON with:
1. internal_analysis: Your private reasoning (how neighbors influence you)
2. new_statement: Your public statement (under 40 words)
"""

        # Generate response using appropriate API
        try:
            if isinstance(client, anthropic.Anthropic):
                # Claude API
                response = client.messages.create(
                    model=model_name or "claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system=system_instruction,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=1.0
                )

                # Extract JSON from response
                content = response.content[0].text.strip()
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                result = json.loads(content.strip())

            elif isinstance(client, openai.OpenAI):
                # OpenAI-compatible API (DeepSeek, OpenAI)
                response = client.chat.completions.create(
                    model=model_name or "deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=1.0,
                    max_tokens=1000
                )

                result = json.loads(response.choices[0].message.content)

            else:
                raise ValueError("Unsupported client type")

            # Update agent's memory
            self.my_statements_history.append(result["new_statement"])
            self.round_history.append({
                "round": round_num,
                "neighbors_input": neighbor_messages,
                "my_output": result
            })

            return result

        except Exception as e:
            print(f"[Agent {self.node_id}] Generation failed: {e}")
            # Return fallback to keep simulation running
            return {
                "internal_analysis": "Error in generation",
                "new_statement": "..."
            }