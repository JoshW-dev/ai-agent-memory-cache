from typing import List, Dict, Tuple, Any, Optional, Type
import re

from openai import OpenAI
import numpy as np

from .llm import ChatLLM
from .tools.base import Tool as BaseTool
from .agent import Agent

# Constants for embedding-based tool selection
TOOL_SIMILARITY_THRESHOLD = 0.3
OPENAI_EMBEDDING_MODEL_FOR_TOOLS = "text-embedding-3-small"

# Prompt template for generating tool input
TOOL_INPUT_GENERATION_PROMPT_TEMPLATE = """
You are an AI assistant. The user's request is: "{user_prompt}"
The most relevant tool to address this request has been identified as:
Tool Name: {tool_name}
Tool Description: {tool_description}

Based on the user's request and the tool's purpose, what is the precise input string that should be provided to this tool?
The input should be concise and directly usable by the tool.
Tool Input:"""

# Prompt template for generating a direct answer when no tool is suitable (or creation fails)
DIRECT_ANSWER_PROMPT_TEMPLATE = """
You are an AI assistant. The user's request is: "{user_prompt}"
No specific tool was found to be a direct match for this request, and creating a new one was not feasible or failed.
Please provide a helpful and direct answer to the user's request based on your general knowledge.
Answer:"""

# Prompt template for defining a new tool
NEW_TOOL_DEFINITION_PROMPT_TEMPLATE = """
You are an AI assistant tasked with defining a new tool to help with a user's request.
The user's request is: "{user_prompt}"

Based on this request, please define a new tool by providing its name and a concise description.
The tool should be specific enough to be useful for similar future requests.

Provide the definition in the following format, and nothing else:
Tool Name: [A short, descriptive, CamelCase name for the tool, e.g., QueryDatabase, SetPlayerAttribute]
Tool Description: [A brief explanation of what the tool does and what its input should generally be]
"""

# Default agent prompt (less critical now, for fallback)
DEFAULT_AGENT_PROMPT_TEMPLATE = """
Today is {today_date}.
You are {agent_name}, an AI assistant.
User's question: {input}
{agent_scratchpad}
Please provide a helpful answer. If you use information from a previous step, mention it.
Final Answer:"""

class CapturingAgent(Agent):
    """
    An agent that captures the history of tool calls (name, input, observation)
    during its run.
    """
    
    def __init__(self, llm: ChatLLM, tools: List[BaseTool], prompt_template: str = DEFAULT_AGENT_PROMPT_TEMPLATE, **kwargs: Any):
        super().__init__(llm=llm, tools=tools, prompt_template=prompt_template, **kwargs)
        self._openai_client = OpenAI()
        self._initialize_tool_primary_embeddings()

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Helper function to generate embedding using OpenAI."""
        try:
            response = self._openai_client.embeddings.create(
                input=text,
                model=OPENAI_EMBEDDING_MODEL_FOR_TOOLS
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding for text '{text}': {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two L2-normalized vectors."""
        return np.dot(np.array(vec1), np.array(vec2))

    def _initialize_tool_primary_embeddings(self, specific_tool: Optional[BaseTool] = None):
        """Generates and stores the primary embedding on tool instances."""
        tools_to_process = [specific_tool] if specific_tool else self.tools
        # print(f"Initializing primary embeddings for {'specific tool' if specific_tool else str(len(tools_to_process)) + ' tools'}...")
        for tool_instance in tools_to_process:
            if tool_instance is None: continue
            text_to_embed = f"Tool: {tool_instance.name}, Description: {tool_instance.description}"
            embedding = self._generate_embedding(text_to_embed)
            if embedding:
                tool_instance.primary_embedding = embedding
                # print(f"  Initialized primary embedding for tool: {tool_instance.name}")
            else:
                print(f"  Failed to generate primary embedding for tool: {tool_instance.name}")
        # print("Tool primary embeddings initialization complete.")

    def _find_best_tool_by_similarity(self, user_prompt: str, exclude_tool_names: Optional[List[str]] = None) -> Optional[Tuple[BaseTool, float]]:
        """Finds the best tool based on semantic similarity to the user prompt."""
        prompt_embedding = self._generate_embedding(user_prompt)
        if not prompt_embedding:
            print("Error: Could not generate embedding for user prompt.")
            return None

        best_tool_match: Optional[BaseTool] = None
        highest_overall_similarity: float = -1.0
        exclude_tool_names_set = set(exclude_tool_names) if exclude_tool_names else set()

        if not self.tools:
            # print("Warning: No tools available for similarity search.")
            return None # Added return here
            
        for tool_instance in self.tools:
            if tool_instance.name in exclude_tool_names_set:
                # print(f"Skipping tool '{tool_instance.name}' due to exclusion list.")
                continue

            tool_specific_max_similarity = -1.0
            all_tool_embeddings = tool_instance.get_all_embeddings()
            if not all_tool_embeddings:
                # print(f"Warning: Tool '{tool_instance.name}' has no embeddings to compare.")
                continue

            for tool_emb in all_tool_embeddings:
                similarity = self._cosine_similarity(prompt_embedding, tool_emb)
                if similarity > tool_specific_max_similarity:
                    tool_specific_max_similarity = similarity
            
            # print(f"  Tool '{tool_instance.name}', Max Similarity for this tool: {tool_specific_max_similarity:.4f}")
            if tool_specific_max_similarity > highest_overall_similarity:
                highest_overall_similarity = tool_specific_max_similarity
                best_tool_match = tool_instance
        
        if best_tool_match and highest_overall_similarity >= TOOL_SIMILARITY_THRESHOLD:
            # print(f"Best tool match (considering exclusions): '{best_tool_match.name}' with overall similarity: {highest_overall_similarity:.4f}")
            return best_tool_match, highest_overall_similarity
        else:
            if best_tool_match: 
                 print(f"No existing tool met threshold ({TOOL_SIMILARITY_THRESHOLD}) (exclusions: {exclude_tool_names}). Best was '{best_tool_match.name}' ({highest_overall_similarity:.4f}).")
            # else: 
                 # print(f"No existing tool met threshold ({TOOL_SIMILARITY_THRESHOLD}) (exclusions: {exclude_tool_names}). No tools to compare or embeddings failed.")
            return None

    def _create_and_register_new_tool(self, llm_defined_name: str, llm_defined_description: str) -> Optional[BaseTool]:
        """Dynamically creates a new tool class and instance, and registers it."""
        tool_name = llm_defined_name
        tool_description = llm_defined_description

        # Define a new tool class dynamically
        class NewDynamicTool(BaseTool):
            name: str = tool_name
            description: str = tool_description

            def __call__(self, action_input: str) -> str:
                return f"Observation: Placeholder for newly created dynamic tool '{self.name}' called with input: '{action_input}'."
        
        new_tool_instance = NewDynamicTool(name=tool_name, description=tool_description)
        self._initialize_tool_primary_embeddings(specific_tool=new_tool_instance) # Generate and add its embedding first
        self.tools.append(new_tool_instance) # Then add to agent's main tool list
        print(f"Successfully created and registered new dynamic tool: {new_tool_instance.name}")
        return new_tool_instance

    def record_tool_usage_feedback(self, user_prompt_text: str, tool_name: str, was_upvoted: bool):
        """Records feedback for a tool. If upvoted, adds prompt embedding to tool's additional embeddings."""
        if not was_upvoted:
            # For now, downvotes are implicitly handled by not reinforcing.
            # Future: could add explicit negative markers if needed.
            print(f"Downvote recorded for tool '{tool_name}' for prompt '{user_prompt_text}'. It will be excluded in immediate retry.")
            return

        target_tool: Optional[BaseTool] = None
        for t in self.tools:
            if t.name == tool_name:
                target_tool = t
                break
        
        if not target_tool:
            print(f"Error recording feedback: Tool '{tool_name}' not found.")
            return

        prompt_embedding = self._generate_embedding(user_prompt_text)
        if prompt_embedding:
            target_tool.add_representative_prompt_embedding(prompt_embedding)
            # print(f"Upvote feedback processed for tool '{tool_name}'. Prompt embedding added.")
        # else:
            # print(f"Error recording feedback: Could not generate embedding for prompt '{user_prompt_text}'.")

    def run(self, input_str: str, agent_scratchpad_content: str = "", exclude_tool_names: Optional[List[str]] = None) -> Tuple[str, List[Dict[str, str]]]:
        history: List[Dict[str, str]] = []
        final_answer: str = "Error: Agent did not produce a final answer."

        tool_match_result = self._find_best_tool_by_similarity(input_str, exclude_tool_names=exclude_tool_names)
        selected_tool: Optional[BaseTool] = None
        similarity_score: float = 0.0

        if tool_match_result:
            selected_tool, similarity_score = tool_match_result
        else: # No suitable existing tool found (considering exclusions)
            print(f"No suitable existing tool found for prompt: '{input_str}' (exclusions: {exclude_tool_names}). Attempting to define a new tool.")
            new_tool_def_prompt = NEW_TOOL_DEFINITION_PROMPT_TEMPLATE.format(user_prompt=input_str)
            llm_tool_definition_str = self.llm.generate(new_tool_def_prompt).strip()
            
            parsed_name, parsed_desc = None, None
            name_match = re.search(r"Tool Name:\s*(.*?)(?:\n|$)", llm_tool_definition_str, re.IGNORECASE)
            desc_match = re.search(r"Tool Description:\s*(.*?)(?:\n|$)", llm_tool_definition_str, re.IGNORECASE)

            if name_match and desc_match:
                parsed_name = name_match.group(1).strip()
                parsed_desc = desc_match.group(1).strip()
                if parsed_name and parsed_desc and parsed_name not in [t.name for t in self.tools]:
                    print(f"LLM defined new tool - Name: '{parsed_name}', Desc: '{parsed_desc}'")
                    selected_tool = self._create_and_register_new_tool(parsed_name, parsed_desc)
                    if selected_tool:
                        similarity_score = 1.0 
                        history.append({
                            "tool_name": "ToolDefinitionAgent", "tool_input": input_str,
                            "observation": f"Defined and registered new tool: {selected_tool.name} - {selected_tool.description}",
                            "similarity_score": "N/A (Tool dynamically created)",
                            "original_user_prompt_for_feedback": input_str
                        })
                    else: print("Failed to instantiate or register the new dynamic tool.")
                elif parsed_name in [t.name for t in self.tools]:
                    print(f"LLM tried to define a tool '{parsed_name}' which already exists. Skipping creation.")
                else: print(f"LLM failed to provide valid name/description. Response: {llm_tool_definition_str}")        
            else: print(f"LLM output for new tool definition did not match expected format. Response: {llm_tool_definition_str}")

        if selected_tool: # This can be an existing tool or a newly created one
            tool_input_prompt_formatted = TOOL_INPUT_GENERATION_PROMPT_TEMPLATE.format(
                user_prompt=input_str,
                tool_name=selected_tool.name,
                tool_description=selected_tool.description
            )
            tool_input_str = self.llm.generate(tool_input_prompt_formatted).strip()

            if "error" in tool_input_str.lower() and len(tool_input_str) > 100: # Heuristic
                final_answer_prompt = DIRECT_ANSWER_PROMPT_TEMPLATE.format(user_prompt=input_str)
                final_answer = self.llm.generate(final_answer_prompt).strip()
                history.append({
                    "tool_name": "DirectAnswer", "tool_input": input_str, "observation": final_answer,
                    "similarity_score": "N/A (Fallback from tool input gen error)",
                    "original_user_prompt_for_feedback": input_str
                })
            else:
                observation = selected_tool(tool_input_str)
                history.append({
                    "tool_name": selected_tool.name, "tool_input": tool_input_str, "observation": observation,
                    "similarity_score": f"{similarity_score:.4f}",
                    "original_user_prompt_for_feedback": input_str
                })
                final_answer = f"Executed {selected_tool.name}. See observation."
        else:
            # Fallback: No tool found or created, generate direct answer
            print(f"Failed to find or create a suitable tool for: '{input_str}' (exclusions: {exclude_tool_names}). Generating direct answer.")
            direct_answer_prompt_formatted = DIRECT_ANSWER_PROMPT_TEMPLATE.format(user_prompt=input_str)
            final_answer = self.llm.generate(direct_answer_prompt_formatted).strip()
            history.append({
                "tool_name": "DirectAnswer", "tool_input": input_str, "observation": final_answer,
                "similarity_score": "N/A (No tool selected/created)",
                "original_user_prompt_for_feedback": input_str
            })
        
        effective_answer = final_answer # Default to final_answer
        if history and "observation" in history[-1]:
            # This logic attempts to make the 'effective_answer' more descriptive
            last_step = history[-1]
            effective_answer_candidate = last_step['observation']
            tool_name_hist = last_step['tool_name']
            sim_score_hist = last_step['similarity_score']
            
            if tool_name_hist not in ["DirectAnswer", "ToolDefinitionAgent"]:
                 effective_answer = f"Based on tool {tool_name_hist} (Similarity: {sim_score_hist}): {effective_answer_candidate}"
            elif tool_name_hist == "DirectAnswer":
                 effective_answer = f"Direct Answer: {effective_answer_candidate}"
            # If it was ToolDefinitionAgent, the `final_answer` (which is likely an error or default) is probably not what we want.
            # The actual execution of the new tool would be in a subsequent history entry if this `run` call included that.
            # For now, if the very last thing was ToolDef, the `final_answer` might be the best we have from this specific `run` call.
            # The test script handles iterative calls, so a subsequent call would use the new tool.

        return effective_answer, history


if __name__ == '__main__':
    from dotenv import load_dotenv
    from .llm import ChatLLM
    from .custom_tools import SetPlayerAttributeTool, SpawnEntityTool, ChangeSkyboxTool, PlaySoundTool

    load_dotenv()

    try:
        print("\n--- Test Section 1: Dynamic Tool Creation & Positive Feedback ---")
        test_llm_s1 = ChatLLM()
        # Initialize with a fresh list of tools for this section
        tools_s1 = [
            SetPlayerAttributeTool(name="SetPlayerAttribute", description="Sets an attribute for the player character. Input should be 'attribute_name=value'."),
            SpawnEntityTool(name="SpawnEntity", description="Spawns an entity. Input: 'entity_type,x,y,z'."),
            ChangeSkyboxTool(name="ChangeSkybox", description="Changes the skybox. Input: theme/color."),
            PlaySoundTool(name="PlaySoundEffect", description="Plays a sound effect. Input: sound_name.")
        ]
        agent_s1 = CapturingAgent(llm=test_llm_s1, tools=tools_s1)
        
        prompts_and_feedback_s1 = [
            ("Set the player's health to 75", "SetPlayerAttribute", True),
            ("make the sky look like a stormy night", "ChangeSkybox", True),
            ("Query the player's current inventory status", "QueryPlayerInventoryStatus", True),
            ("What items does the player have?", "QueryPlayerInventoryStatus", True),
            ("Teleport player to coordinates 0,100,0", "TeleportPlayerToCoordinates", True),
            ("move the player character to 0,100,0", "TeleportPlayerToCoordinates", True),
            ("What is the main ingredient in bread?", "DirectAnswer", True), 
            ("Set player speed to 20", "SetPlayerAttribute", True),
            ("Set player speed to 20", "SetPlayerAttribute", True) 
        ]

        for prompt, tool_name_for_feedback_unused, is_upvoted_feedback in prompts_and_feedback_s1: # tool_name_for_feedback is not used from the loop here
            print(f"\nUser Question: {prompt}")
            # print(f"Current tools in agent: {[t.name for t in agent_s1.tools]}")
            agent_final_answer, agent_history = agent_s1.run(prompt)
            print(f"Agent's Effective Answer: {agent_final_answer}")
        print("History:")
            for i, step in enumerate(agent_history):
            print(f"  Step {i+1}:")
                print(f"    Tool: {step.get('tool_name', 'N/A')}")
                print(f"    Similarity: {step.get('similarity_score', 'N/A')}")
                print(f"    Input: {step.get('tool_input', 'N/A')}")
                print(f"    Observation: {step.get('observation', 'N/A')}")

            # Determine the actual tool chosen by the agent for feedback
            actual_chosen_tool_name_s1 = None
            if agent_history:
                # Find the last step that was an actual tool execution or DirectAnswer for feedback purposes
                for step in reversed(agent_history):
                    if step["tool_name"] != "ToolDefinitionAgent": # We give feedback on the tool USED or DirectAnswer
                        actual_chosen_tool_name_s1 = step["tool_name"]
                        break
            
            if actual_chosen_tool_name_s1 and actual_chosen_tool_name_s1 != "DirectAnswer":
                original_prompt_for_feedback = agent_history[-1].get("original_user_prompt_for_feedback", prompt)
                # print(f"Simulating feedback for tool '{actual_chosen_tool_name_s1}' with prompt '{original_prompt_for_feedback}'. Upvoted: {is_upvoted_feedback}")
                agent_s1.record_tool_usage_feedback(original_prompt_for_feedback, actual_chosen_tool_name_s1, is_upvoted_feedback)
            elif actual_chosen_tool_name_s1 == "DirectAnswer":
                print("Outcome was DirectAnswer, no specific tool feedback to agent's tool list for S1.")
            print("-----------------------------")

        print("\n--- Test Section 2: Negative Feedback & Retry Logic ---")
        test_llm_s2 = ChatLLM()
        tools_s2 = [
            SetPlayerAttributeTool(name="SetPlayerAttribute", description="Sets an attribute for the player character. Input should be 'attribute_name=value'."),
            SpawnEntityTool(name="SpawnEntity", description="Spawns an entity. Input: 'entity_type,x,y,z'."),
            ChangeSkyboxTool(name="ChangeSkybox", description="Changes the skybox. Input: theme/color."),
            PlaySoundTool(name="PlaySoundEffect", description="Plays a sound effect. Input: sound_name.")
        ]
        agent_s2 = CapturingAgent(llm=test_llm_s2, tools=tools_s2)

        # Scenario 1: First attempt creates a tool, it's downvoted. Second attempt should try existing or create another.
        prompt_s2_1 = "Make the character jump higher" 
        print(f"\nUser Question (S2P1 Attempt 1): {prompt_s2_1}")
        # print(f"Current tools before S2P1 Att1: {[t.name for t in agent_s2.tools]}")
        ans_s2_1_att1, hist_s2_1_att1 = agent_s2.run(prompt_s2_1)
        tool_s2_1_att1 = next((s["tool_name"] for s in reversed(hist_s2_1_att1) if s["tool_name"] not in ["DirectAnswer", "ToolDefinitionAgent"]), 
                                hist_s2_1_att1[-1]["tool_name"] if hist_s2_1_att1 and hist_s2_1_att1[-1]["tool_name"] == "ToolDefinitionAgent" else None)
        print(f"Agent Answer (S2P1 Att 1 - Tool: {tool_s2_1_att1}): {ans_s2_1_att1}")
        # print(f"History S2P1 Att1: {hist_s2_1_att1}")
        # print(f"Tools after S2P1 Att1: {[t.name for t in agent_s2.tools]}")
        
        excluded_tools_s2_1 = []
        if tool_s2_1_att1 and tool_s2_1_att1 != "DirectAnswer": 
            # If ToolDefinitionAgent was the last, the actual tool name is in the history observation or was parsed_name
            # For feedback, we need the name of the tool that was *defined* or *chosen*.
            # The `next` logic above should find the used tool OR the defined tool name if ToolDefinitionAgent was the last relevant step.
            # Let's refine how we get the tool name from history for feedback
            tool_name_for_feedback_s2_1 = None
            if hist_s2_1_att1:
                # If last step was tool definition, the name is in that step's observation or from the parsed name if available
                if hist_s2_1_att1[-1]["tool_name"] == "ToolDefinitionAgent":
                    obs = hist_s2_1_att1[-1]["observation"]
                    match = re.search(r"Defined and registered new tool: (\w+)", obs)
                    if match: tool_name_for_feedback_s2_1 = match.group(1)
                else: # Otherwise, it's the tool name from the last execution step
                    tool_name_for_feedback_s2_1 = next((s["tool_name"] for s in reversed(hist_s2_1_att1) if s["tool_name"] not in ["DirectAnswer", "ToolDefinitionAgent"]), None)

            if tool_name_for_feedback_s2_1 and tool_name_for_feedback_s2_1 != "DirectAnswer":
                print(f"Simulating DOWNVOTE for tool '{tool_name_for_feedback_s2_1}' for prompt '{prompt_s2_1}'")
                agent_s2.record_tool_usage_feedback(prompt_s2_1, tool_name_for_feedback_s2_1, False)
                excluded_tools_s2_1.append(tool_name_for_feedback_s2_1)
            
            print(f"\nUser Question (S2P1 Attempt 2 - after downvoting {tool_name_for_feedback_s2_1 or 'N/A'}): {prompt_s2_1}")
            # print(f"Current tools before S2P1 Att2: {[t.name for t in agent_s2.tools]}")
            ans_s2_1_att2, hist_s2_1_att2 = agent_s2.run(prompt_s2_1, exclude_tool_names=excluded_tools_s2_1)
            tool_s2_1_att2 = next((s["tool_name"] for s in reversed(hist_s2_1_att2) if s["tool_name"] not in ["DirectAnswer", "ToolDefinitionAgent"]), None)
            print(f"Agent Answer (S2P1 Att 2 - Tool: {tool_s2_1_att2}): {ans_s2_1_att2}")
            # print(f"History S2P1 Att2: {hist_s2_1_att2}")
            # print(f"Tools after S2P1 Att2: {[t.name for t in agent_s2.tools]}")

            if tool_s2_1_att2 and tool_s2_1_att2 != "DirectAnswer":
                print(f"Simulating UPVOTE for tool '{tool_s2_1_att2}' for prompt '{prompt_s2_1}'")
                agent_s2.record_tool_usage_feedback(prompt_s2_1, tool_s2_1_att2, True)
        elif tool_s2_1_att1 == "DirectAnswer":
             print(f"Agent gave DirectAnswer on S2P1 Att1. No downvote/retry for this specific sub-scenario.")
        print("-----------------------------")

        # Scenario 2: First choice is an existing tool (e.g. ChangeSkybox for "dim lights"), downvoted. 
        # Next choice should be dynamic creation or another existing tool, or direct.
        prompt_s2_2 = "Dim the lights in the scene"
        print(f"\nUser Question (S2P2 Attempt 1): {prompt_s2_2}")
        # print(f"Current tools before S2P2 Att1: {[t.name for t in agent_s2.tools]}")
        ans_s2_2_att1, hist_s2_2_att1 = agent_s2.run(prompt_s2_2) 
        tool_s2_2_att1 = next((s["tool_name"] for s in reversed(hist_s2_2_att1) if s["tool_name"] not in ["DirectAnswer", "ToolDefinitionAgent"]), None)
        print(f"Agent Answer (S2P2 Att 1 - Tool: {tool_s2_2_att1}): {ans_s2_2_att1}")
        # print(f"History S2P2 Att1: {hist_s2_2_att1}")
        # print(f"Tools after S2P2 Att1: {[t.name for t in agent_s2.tools]}")

        excluded_tools_s2_2 = []
        if tool_s2_2_att1 and tool_s2_2_att1 != "DirectAnswer":
            print(f"Simulating DOWNVOTE for tool '{tool_s2_2_att1}' for prompt '{prompt_s2_2}'")
            agent_s2.record_tool_usage_feedback(prompt_s2_2, tool_s2_2_att1, False)
            excluded_tools_s2_2.append(tool_s2_2_att1)

            print(f"\nUser Question (S2P2 Attempt 2 - after downvoting {tool_s2_2_att1}): {prompt_s2_2}")
            # print(f"Current tools before S2P2 Att2: {[t.name for t in agent_s2.tools]}")
            ans_s2_2_att2, hist_s2_2_att2 = agent_s2.run(prompt_s2_2, exclude_tool_names=excluded_tools_s2_2)
            tool_s2_2_att2 = next((s["tool_name"] for s in reversed(hist_s2_2_att2) if s["tool_name"] not in ["DirectAnswer", "ToolDefinitionAgent"]), None)
            print(f"Agent Answer (S2P2 Att 2 - Tool: {tool_s2_2_att2}): {ans_s2_2_att2}") # Expecting new tool or direct answer
            # print(f"History S2P2 Att2: {hist_s2_2_att2}")
            # print(f"Tools after S2P2 Att2: {[t.name for t in agent_s2.tools]}")
            if tool_s2_2_att2 and tool_s2_2_att2 != "DirectAnswer":
                 print(f"Simulating UPVOTE for tool '{tool_s2_2_att2}' for prompt '{prompt_s2_2}'")
                 agent_s2.record_tool_usage_feedback(prompt_s2_2, tool_s2_2_att2, True)
        elif tool_s2_2_att1 == "DirectAnswer":
             print(f"Agent gave DirectAnswer on S2P2 Att1. No downvote/retry for this specific sub-scenario.")
        print("-----------------------------")

        # Scenario 3 (Exclusion of an upvoted tool - still relevant)
        prompt_s2_3 = "Set player attribute agility to max"
        print(f"\nUser Question (Initial Setup for S2P3): {prompt_s2_3}")
        # Ensure SetPlayerAttribute is present and upvote for this specific prompt
        # It should be present from agent_s2 initialization. If not, this test is flawed.
        # Let's explicitly record feedback for it with the agent.
        agent_s2.record_tool_usage_feedback(prompt_s2_3, "SetPlayerAttribute", True) 
        print(f"Manually recorded UPVOTE for 'SetPlayerAttribute' for prompt '{prompt_s2_3}' (S2P3 setup)")

        print(f"\nUser Question (Test Exclusion S2P3): {prompt_s2_3}")
        ans_s2_3_excluded, hist_s2_3_excluded = agent_s2.run(prompt_s2_3, exclude_tool_names=["SetPlayerAttribute"])
        print(f"Agent Answer (S2P3 with SetPlayerAttribute excluded): {ans_s2_3_excluded}")
        tool_s2_3_excluded = next((s["tool_name"] for s in reversed(hist_s2_3_excluded) if s["tool_name"] not in ["DirectAnswer", "ToolDefinitionAgent"]), None)
        # print(f"History S2P3 Excluded: {hist_s2_3_excluded}")
        # print(f"Tools after S2P3 Excluded: {[t.name for t in agent_s2.tools]}")
        assert tool_s2_3_excluded != "SetPlayerAttribute", f"SetPlayerAttribute should have been excluded (S2P3), but {tool_s2_3_excluded} was chosen!"
        
        print(f"\nUser Question (Test No Exclusion S2P3 - after upvote): {prompt_s2_3}")
        ans_s2_3_not_excluded, hist_s2_3_not_excluded = agent_s2.run(prompt_s2_3)
        print(f"Agent Answer (S2P3 without exclusion): {ans_s2_3_not_excluded}")
        tool_s2_3_not_excluded = next((s["tool_name"] for s in reversed(hist_s2_3_not_excluded) if s["tool_name"] not in ["DirectAnswer", "ToolDefinitionAgent"]), None)
        # print(f"History S2P3 Not Excluded: {hist_s2_3_not_excluded}")
        # print(f"Tools after S2P3 Not Excluded: {[t.name for t in agent_s2.tools]}")
        assert tool_s2_3_not_excluded == "SetPlayerAttribute", f"SetPlayerAttribute should have been chosen (S2P3), but {tool_s2_3_not_excluded} was chosen!"
        if tool_s2_3_not_excluded == "SetPlayerAttribute":
            # Find the similarity score for the SetPlayerAttribute step in the history
            score_s2_3_not_excluded = "0.0"
            for step in hist_s2_3_not_excluded:
                if step["tool_name"] == "SetPlayerAttribute":
                    score_s2_3_not_excluded = step["similarity_score"]
                    break
            final_similarity_s2_3 = float(score_s2_3_not_excluded)
            assert final_similarity_s2_3 > 0.99, f"Similarity for upvoted S2P3 prompt should be ~1.0 for SetPlayerAttribute, got {final_similarity_s2_3}"
        print("-----------------------------")

    except Exception as e:
        print(f"An error occurred during the CapturingAgent test: {e}")
        import traceback
        traceback.print_exc() 