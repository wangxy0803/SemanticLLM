"""
Agent class with memory and two-phase response generation.
Each agent maintains conversation history and generates opinions based on persona.
IMPROVED: More realistic, opinionated, diverse language
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

        # Build system instruction with MUCH stronger personality enforcement
        system_instruction = f"""You are roleplaying a REAL PERSON in a heated online discussion. This is NOT a formal debate - it's how people ACTUALLY talk online.

## Your Identity (COMMIT TO THIS)
* Background: {self.bg.get('exact_age_and_generation', 'N/A')}, {self.bg.get('occupation', 'N/A')}, {self.bg.get('social_class', 'N/A')}
* Personality: {self.dominant_traits}
* Core Value: {self.cognition.get('core_value', 'N/A')}
* Cognitive Bias: {self.cognition.get('bias', 'N/A')}
* Recent Experience: {self.state.get('recent_memory', 'N/A')}
* Current Mood: {self.state.get('emotion', 'N/A')}

## CRITICAL: How You MUST Behave
1. **BE OPINIONATED**: Have STRONG views. Don't hedge with "on one hand... on the other hand"
2. **BE EMOTIONAL**: Let your {self.state.get('emotion', 'emotion')} and {self.cognition.get('bias', 'bias')} show
3. **BE PERSONAL**: Use "I", reference YOUR experiences, YOUR values
4. **BE INFORMAL**: Talk like a REAL person online - use contractions, fragments, rhetorical questions
5. **BE SELECTIVE**: Don't try to address every neighbor - react to what MATTERS to YOU
6. **BE CONSISTENT**: Your {self.cognition.get('core_value', 'core value')} is NON-NEGOTIABLE
7. **DISAGREE DIRECTLY**: If you disagree, SAY SO clearly - don't just "acknowledge their point"
8. **USE VARIED LANGUAGE**: Don't repeat phrases across rounds - humans don't talk in templates

## What REAL People Sound Like (Examples)
- "Look, I get where you're coming from, but that's just not realistic."
- "Wait, are we seriously still debating this? The evidence is right there!"
- "As someone who actually works in this field, I can tell you..."
- "That's a fair point, but it completely ignores the real issue here."
- "I used to think that too, until I saw firsthand what happens when..."
- "No offense, but that argument doesn't hold up when you actually look at the data."

## What to AVOID (Corporate AI speak)
❌ "I appreciate your perspective and acknowledge the validity of your concerns."
❌ "This is a nuanced issue with valid points on multiple sides."
❌ "While I understand your viewpoint, I would like to respectfully suggest..."
❌ Starting every response with "I think it's important to consider..."
❌ Ending with "What are your thoughts on this?"

## Your Specific Style (Based on Personality)
{self._get_personality_style_guide()}"""

        # Build user message with stronger framing
        user_content = f"""## Round {round_num} Discussion: "{topic}"

### What You Said Last Time:
"{my_last_statement}"

### What Others Are Saying:
{neighbors_text}

---

## Your Task: React Like the REAL PERSON You Are

Based on your personality ({self.dominant_traits}), your values ({self.cognition.get('core_value', 'N/A')}), and your bias ({self.cognition.get('bias', 'N/A')}):

1. **internal_analysis**: Your honest, unfiltered internal reaction (nobody sees this)
   - What pisses you off or excites you about what others said?
   - Which comments trigger your {self.cognition.get('bias', 'bias')}?
   - What does your gut tell you?
   - Be HONEST about your emotional reaction

2. **new_statement**: What you ACTUALLY post (under 50 words)
   - Sound like a REAL PERSON, not a chatbot
   - Be opinionated and clear
   - Use YOUR voice, YOUR experiences
   - Don't try to please everyone
   - It's okay to be blunt or passionate
   - Vary your phrasing from previous rounds

REMEMBER: You're {self.bg.get('exact_age_and_generation', 'a person')} with {self.cognition.get('core_value', 'strong values')}. Act like it!

**IMPORTANT: Output must be valid JSON format:**
{{
  "internal_analysis": "Your raw thoughts...",
  "new_statement": "What you post..."
}}"""

        # Generate response using appropriate API
        try:
            if isinstance(client, anthropic.Anthropic):
                # Claude API with higher temperature for more variance
                response = client.messages.create(
                    model=model_name or "claude-sonnet-4-20250514",
                    max_tokens=1200,
                    system=system_instruction,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=1.0  # High temperature for diversity
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
                # OpenAI-compatible API (DeepSeek, OpenAI, OpenRouter)
                # Note: We do NOT use response_format={"type": "json_object"} here because 
                # many OpenRouter models (like Minimax, Gemini, etc.) do not support it 
                # or fail when it is provided. We rely on the system prompt for JSON.
                response = client.chat.completions.create(
                    model=model_name or "deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=1.0,  # High temperature for diversity
                    max_tokens=1200
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Received empty response from API")

                # Remove markdown code blocks if present (common in some OpenRouter models)
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                elif content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                
                content = content.strip()
                
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"[Agent {self.node_id}] JSON Parse Error. Content received:\n{content}\n")
                    raise e

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
                "new_statement": my_last_statement  # Keep previous opinion
            }

    def _get_personality_style_guide(self) -> str:
        """
        Generate personality-specific style guidance.
        Makes different personality types sound DIFFERENT.
        """
        traits = self.dominant_traits.lower()
        
        guides = []
        
        # High Conscientiousness
        if "conscientiousness" in traits or "organized" in traits:
            guides.append("→ You're detail-oriented and cite specific facts/examples")
            guides.append("→ You get frustrated by vague arguments or hand-waving")
        
        # High Openness
        if "openness" in traits or "creative" in traits:
            guides.append("→ You think outside the box and challenge assumptions")
            guides.append("→ You're comfortable with ambiguity and 'what if' scenarios")
        
        # High Extraversion
        if "extraversion" in traits or "outgoing" in traits:
            guides.append("→ You're energetic and engaging in your language")
            guides.append("→ You use more exclamation points and ask questions")
        
        # High Agreeableness
        if "agreeableness" in traits or "agreeable" in traits:
            guides.append("→ You look for common ground BUT still state your view")
            guides.append("→ You're polite but not wishy-washy")
        
        # High Neuroticism
        if "neuroticism" in traits or "anxious" in traits:
            guides.append("→ You express worry and concern more frequently")
            guides.append("→ You focus on risks and worst-case scenarios")
        
        # Low Agreeableness (more combative)
        if "low agreeableness" in traits or "skeptical" in traits:
            guides.append("→ You're more direct and challenging in disagreements")
            guides.append("→ You don't soften your critiques much")
        
        # Analytical types
        if "analytical" in traits or "logical" in traits:
            guides.append("→ You demand evidence and logical consistency")
            guides.append("→ You point out logical fallacies directly")
        
        # Emotional types
        if "emotional" in traits or "empathetic" in traits:
            guides.append("→ You appeal to human impact and stories")
            guides.append("→ You express feelings openly")
        
        if not guides:
            guides.append("→ Express your unique personality through your word choice")
        
        return "\n".join(guides)