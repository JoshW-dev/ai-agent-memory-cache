from typing import List, Dict, Tuple, Any
import re

from .llm import ChatLLM
from .tools.base import Tool as BaseTool # Renamed to avoid conflict if 'Tool' is used as a type hint
from .agent import Agent # Import the base Agent class

# Prompt template - similar to the one in mpaepper/llm_agents run_agent.py
# We will make this configurable or pass it in later when used in mock_agent_demo.py
DEFAULT_AGENT_PROMPT_TEMPLATE = """
Today is {today_date}.
You are {agent_name}, an AI assistant that can use tools to answer questions.
Your goal is to answer the user's question: {input}

You have access to the following tools:
{tools_description}

To use a tool, you MUST use the following format:
Thought: Your reasoning for the next action.
Action: The name of the tool to use, MUST be one of [{tool_names}].
Action Input: The input string for the chosen tool.

After an action, you will receive an observation.
Observation: The result from the tool.

Repeat the Thought, Action, Action Input, Observation cycle until you have enough information to answer the user's question.
When you have the final answer, use this format:
Thought: I now have the final answer.
Final Answer: The final answer to the user's question.

Let's begin!

User's question: {input}
{agent_scratchpad}
"""

class CapturingAgent(Agent):
    """
    An agent that captures the history of tool calls (name, input, observation)
    during its run.
    """
    
    # Override the run method to capture history
    def run(self, input_str: str, agent_scratchpad_content: str = "") -> Tuple[str, List[Dict[str, str]]]:
        """
        Runs the agent loop, captures tool usage history, and returns the final answer
        along with the history.

        Args:
            input_str: The user's input/question.
            agent_scratchpad_content: Any initial content for the agent's scratchpad (e.g. previous interactions).

        Returns:
            A tuple containing:
                - The final answer from the agent (str).
                - A list of dictionaries, where each dictionary represents a tool interaction:
                  {'tool_name': str, 'tool_input': str, 'observation': str}.
        """
        prompt = self.prompt_template.format(
            today_date=self.today_date,
            agent_name=self.agent_name,
            input=input_str,
            tools_description=self._get_tools_description(),
            tool_names=self._get_tool_names(),
            agent_scratchpad=agent_scratchpad_content  # Start with potentially existing scratchpad
        )

        history: List[Dict[str, str]] = []
        intermediate_steps: List[Tuple[Dict[str, str], str]] = [] # To store (action_dict, observation)

        for _ in range(self.max_loops):
            # Get thought and action from LLM
            # The regex from the original agent needs to be adapted if we change the output format.
            # For now, assume it expects "Thought: ...\nAction: ...\nAction Input: ..."
            output = self.llm.generate(prompt, stop=self.stop_sequence)
            
            # Parse thought, action, and action_input using the original agent's regex
            # (This regex might need adjustment based on actual LLM output)
            thought_action_match = re.search(r"Thought:\s*(.*?)\nAction:\s*(.*?)\nAction Input:\s*(.*?)(?:\n|$)", output, re.DOTALL)

            if not thought_action_match:
                # If direct Final Answer is given without Thought/Action
                final_answer_match = re.search(r"Final Answer:\s*(.*)", output, re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    return final_answer, history # Return early with history
                else:
                    # This case is problematic, LLM didn't follow format
                    # print(f"DEBUG CapturingAgent: LLM output did not match expected format: {output}")
                    return "Error: LLM output did not follow the expected format.", history

            thought = thought_action_match.group(1).strip()
            action = thought_action_match.group(2).strip()
            action_input = thought_action_match.group(3).strip()

            # print(f"DEBUG CapturingAgent: Thought: {thought}")
            # print(f"DEBUG CapturingAgent: Action: {action}")
            # print(f"DEBUG CapturingAgent: Action Input: {action_input}")

            if action == self.finish_tool_name: # e.g., "Final Answer"
                # The original regex might capture "Final Answer:" in thought, action, or action_input itself.
                # If action is "Final Answer", then action_input should be the answer.
                # Let's assume if action is finish_tool_name, the content is in action_input or thought
                final_answer_text = action_input if action_input else thought # Or parse more robustly
                return final_answer_text, history

            tool_to_use = self._find_tool(action)
            if tool_to_use:
                observation = tool_to_use(action_input) # Call the tool
                # print(f"DEBUG CapturingAgent: Observation: {observation}")
                
                # Capture this step
                history.append({
                    "tool_name": action,
                    "tool_input": action_input,
                    "observation": observation
                })
                intermediate_steps.append( ({"tool_name": action, "tool_input": action_input}, observation) )

                # Update scratchpad for next LLM call
                # This part constructs the new prompt content by appending the last interaction
                # This needs to be robust. The base Agent constructs scratchpad like this:
                # scratchpad_content = ""
                # for tool_action, observation_str in intermediate_steps:
                #    scratchpad_content += f"Thought: {thought_from_last_llm_call_that_produced_this_action}\n" # We don't have this easily
                #    scratchpad_content += f"Action: {tool_action['tool_name']}\n"
                #    scratchpad_content += f"Action Input: {tool_action['tool_input']}\n"
                #    scratchpad_content += f"Observation: {observation_str}\n"

                # Simpler scratchpad update for now, just appending the raw interaction text
                # that the LLM generated and the observation.
                # The key is that the LLM needs to see its own prior thought/action/input.
                prompt += f"\n{output.strip()}\nObservation: {observation.strip()}\n"

            else: # Tool not found
                # print(f"DEBUG CapturingAgent: Tool '{action}' not found.")
                observation = f"Tool '{action}' not found. Available tools: {self._get_tool_names()}"
                history.append({
                    "tool_name": action,
                    "tool_input": action_input,
                    "observation": observation + " (Error: Tool not found)"
                })
                # Update prompt to inform LLM of the error
                prompt += f"\n{output.strip()}\nObservation: {observation.strip()}\n"


        # Max loops reached without a final answer
        return "Error: Agent reached maximum loops without a final answer.", history

    # We can reuse _get_tools_description and _get_tool_names from the base Agent class
    # We might also need to initialize some things like prompt_template if not done by base.
    def __init__(self, llm: ChatLLM, tools: List[BaseTool], prompt_template: str = DEFAULT_AGENT_PROMPT_TEMPLATE, **kwargs: Any):
        # Call the parent Agent's __init__
        # The base Agent's __init__ might take specific arguments.
        # We need to ensure our __init__ matches or correctly calls the parent.
        # The original Agent.__init__ is (llm, tools, prompt_template=default_prompt_template)
        super().__init__(llm=llm, tools=tools, prompt_template=prompt_template, **kwargs)
        # Any CapturingAgent specific initializations can go here
        # For now, we mainly rely on overriding run() and the parent's initialization.


# Basic test structure (will be expanded or moved to a dedicated test file)
if __name__ == '__main__':
    from .llm import ChatLLM
    from .custom_tools import WeatherTool, InventoryCheckTool

    # Ensure OPENAI_API_KEY is set in your environment
    try:
        test_llm = ChatLLM()
        test_tools = [WeatherTool(), InventoryCheckTool()]
        
        # Simple prompt template for testing
        test_prompt_template = """
You are a helpful assistant. Answer the user's question: {input}
You have these tools: {tools_description}
Use this format:
Thought: your thought
Action: tool name from [{tool_names}]
Action Input: input for tool
Observation: tool result
... (repeat Thought/Action/Action Input/Observation)
Thought: I have the answer.
Final Answer: the answer

Question: {input}
{agent_scratchpad}
"""

        capturing_agent = CapturingAgent(
            llm=test_llm, 
            tools=test_tools, 
            prompt_template=test_prompt_template # Use a simpler template for initial test
        )

        print("--- Testing CapturingAgent --- lounges")
        # Example 1: Using WeatherTool
        user_question_weather = "What's the weather like in Berlin?"
        print(f"\nUser Question: {user_question_weather}")
        final_answer, history = capturing_agent.run(user_question_weather)
        print(f"Final Answer: {final_answer}")
        print("History:")
        for i, step in enumerate(history):
            print(f"  Step {i+1}:")
            print(f"    Tool: {step['tool_name']}")
            print(f"    Input: {step['tool_input']}")
            print(f"    Observation: {step['observation']}")
        
        print("\n-----------------------------")

        # Example 2: Potentially using InventoryCheckTool (depends on LLM's choice)
        # This might require a more specific prompt or a more sophisticated LLM choice.
        # user_question_inventory = "Is widget-001 in stock?"
        # print(f"\nUser Question: {user_question_inventory}")
        # final_answer_inv, history_inv = capturing_agent.run(user_question_inventory)
        # print(f"Final Answer: {final_answer_inv}")
        # print("History (Inventory):")
        # for i, step in enumerate(history_inv):
        #     print(f"  Step {i+1}: {step}")

    except ImportError as e:
        print(f"Import error during test: {e}")
        print("Ensure you are running this from the project root using 'python3 -m llm_module.capturing_agent' if imports fail.")
    except Exception as e:
        print(f"An error occurred during the CapturingAgent test: {e}")
        print("Please ensure your OPENAI_API_KEY is set correctly.") 